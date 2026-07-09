"""Helpers for resolving and rendering glycan cartoons from compact IUPAC."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from html import escape
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    import pandas as pd


GLYLOOKUP_BASE_URL = "https://glylookup.glyomics.org"
GLYMAGE_BASE_URL = "https://glymage.glyomics.org"
GLYTOUCAN_BASE_URL = "https://glytoucan.org/Structures/Glycans"


def _post_form_json(url: str, data: dict[str, Any], timeout: int = 60) -> Any:
    """POST one URL-encoded form and parse the JSON response."""
    # Both GlyLookup and Glymage use the same simple form-post API shape.
    payload = urlencode(data).encode("utf-8")
    request = Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8")
    return json.loads(text)


def _submit_task(
    base_url: str,
    task: dict[str, Any],
    developer_email: str,
    timeout: int = 60,
) -> dict[str, Any]:
    """Submit one glyomics task and return the assigned task payload."""
    submission = _post_form_json(
        f"{base_url.rstrip('/')}/submit",
        {
            "tasks": json.dumps([task]),
            "developer_email": developer_email,
        },
        timeout=timeout,
    )
    if not submission:
        raise RuntimeError(f"No task ID returned from {base_url}.")
    return submission[0]


def _retrieve_task(
    base_url: str,
    task_id: str,
    timeout: int = 60,
) -> dict[str, Any]:
    """Retrieve one glyomics task result."""
    retrieval = _post_form_json(
        f"{base_url.rstrip('/')}/retrieve",
        {
            "task_ids": json.dumps([task_id]),
            "timeout": timeout,
        },
        timeout=timeout,
    )
    if not retrieval:
        raise RuntimeError(f"No retrieval payload returned from {base_url}.")
    return retrieval[0]


def _request_task(
    base_url: str,
    task: dict[str, Any],
    developer_email: str,
    timeout: int = 60,
    poll_attempts: int = 8,
    poll_sleep_seconds: float = 1.0,
) -> dict[str, Any]:
    """Submit one task and poll until a final payload is available."""
    submission = _submit_task(base_url, task, developer_email, timeout=timeout)
    task_id = submission["id"]
    # The glyomics services are asynchronous, so we poll until a result/error appears.
    for _ in range(poll_attempts):
        retrieval = _retrieve_task(base_url, task_id, timeout=timeout)
        if "result" in retrieval or "error" in retrieval:
            return retrieval
        time.sleep(poll_sleep_seconds)
    return _retrieve_task(base_url, task_id, timeout=timeout)


@lru_cache(maxsize=512)
def resolve_cartoon_metadata(
    sequence: str,
    developer_email: str | None,
    image_format: str = "svg",
    display: str = "compact",
    lookup_timeout: int = 60,
) -> dict[str, str]:
    """Resolve GlyLookup/Glymage metadata for one compact IUPAC sequence."""
    # Cache by sequence so repeated notebook runs do not re-request the same cartoon
    # metadata over and over within one Python session.
    metadata = {
        "sequence": sequence,
        "accession": "",
        "glytoucan_url": "",
        "image_url": "",
        "lookup_status": "not_attempted",
        "lookup_errors": "",
    }

    if developer_email is None or str(developer_email).strip() == "":
        metadata["lookup_status"] = "missing_email"
        metadata["lookup_errors"] = "No developer email was provided for GlyLookup/Glymage."
        return metadata

    try:
        # Try GlyLookup first because it can connect the sequence to a real accession,
        # which gives us both a canonical ID and a stable precomputed image URL.
        lookup_response = _request_task(
            GLYLOOKUP_BASE_URL,
            {"seq": sequence},
            developer_email=str(developer_email).strip(),
            timeout=lookup_timeout,
        )
        result_rows = lookup_response.get("result") or []
        error_rows = [str(value) for value in (lookup_response.get("error") or [])]

        if result_rows and result_rows[0].get("accession"):
            accession = str(result_rows[0]["accession"])
            metadata.update(
                {
                    "accession": accession,
                    "glytoucan_url": f"{GLYTOUCAN_BASE_URL}/{accession}",
                    "image_url": f"{GLYMAGE_BASE_URL}/image/snfg/{display}/{accession}.{image_format}",
                    "lookup_status": "accession_found",
                    "lookup_errors": "; ".join(error_rows),
                }
            )
            return metadata

        metadata["lookup_errors"] = "; ".join(error_rows)
        if not error_rows or not any("Unable to parse" in value for value in error_rows):
            # If GlyLookup does not find an accession but the sequence still looks usable,
            # ask Glymage to render an on-demand image anyway.
            glymage_submission = _submit_task(
                GLYMAGE_BASE_URL,
                {
                    "seq": sequence,
                    "image_format": image_format,
                    "display": display,
                },
                developer_email=str(developer_email).strip(),
                timeout=lookup_timeout,
            )
            metadata["image_url"] = f"{GLYMAGE_BASE_URL}/getimage?task_id={glymage_submission['id']}"
            metadata["lookup_status"] = "on_demand_image"
        else:
            metadata["lookup_status"] = "parse_error"

        return metadata
    except (URLError, TimeoutError, json.JSONDecodeError, RuntimeError, KeyError, ValueError) as error:
        metadata["lookup_status"] = "lookup_error"
        metadata["lookup_errors"] = str(error)
        return metadata


def build_cartoon_manifest(
    sequences: Sequence[str],
    developer_email: str | None,
    image_format: str = "svg",
    display: str = "compact",
    lookup_timeout: int = 60,
) -> "pd.DataFrame":
    """Build one metadata table for all unique sequences in the analysis."""
    import pandas as pd

    unique_sequences = []
    seen_sequences: set[str] = set()
    for sequence in sequences:
        text = str(sequence)
        if text not in seen_sequences:
            unique_sequences.append(text)
            seen_sequences.add(text)

    # Resolve each unique sequence once, then keep the results in a flat table that can
    # be saved, inspected in the notebook, or converted back into a lookup dictionary.
    manifest_rows = [
        resolve_cartoon_metadata(
            sequence=sequence,
            developer_email=developer_email,
            image_format=image_format,
            display=display,
            lookup_timeout=lookup_timeout,
        )
        for sequence in unique_sequences
    ]
    return pd.DataFrame(manifest_rows)


def cartoon_lookup_from_manifest(cartoon_manifest_df) -> dict[str, dict[str, str]]:
    """Convert a cartoon manifest dataframe into a sequence-keyed lookup."""
    return cartoon_manifest_df.set_index("sequence").to_dict(orient="index")


def format_glycan_sequence_block(sequence: str, cartoon_row: dict[str, str] | None) -> str:
    """Render one compact HTML block for a glycan sequence and its cartoon."""
    image_html = "<div class='cartoon-missing'>No cartoon resolved</div>"
    caption_html = ""
    if cartoon_row is not None:
        image_url = cartoon_row.get("image_url", "")
        accession = cartoon_row.get("accession", "")
        glytoucan_url = cartoon_row.get("glytoucan_url", "")
        lookup_status = cartoon_row.get("lookup_status", "")
        if image_url:
            image_html = f"<img src='{escape(image_url, quote=True)}' alt='Cartoon for {escape(sequence)}'>"
        if accession and glytoucan_url:
            # When an accession is available, link it directly so the HTML report can
            # double as a quick lookup sheet.
            caption_html = (
                f"<div class='cartoon-caption'><a href='{escape(glytoucan_url, quote=True)}' "
                f"target='_blank' rel='noopener'>{escape(accession)}</a></div>"
            )
        elif lookup_status:
            caption_html = f"<div class='cartoon-caption'>{escape(lookup_status)}</div>"

    return (
        "<div class='sequence-block'>"
        f"<div class='cartoon'>{image_html}{caption_html}</div>"
        f"<pre class='sequence-text'>{escape(sequence)}</pre>"
        "</div>"
    )


def render_glycan_cartoon_gallery_html(
    sequences: Sequence[str],
    cartoon_lookup: dict[str, dict[str, str]],
    output_path,
    title: str,
    subtitle: str | None = None,
) -> Path:
    """Render a simple reusable HTML gallery for a sequence list."""
    unique_sequences = []
    seen_sequences: set[str] = set()
    for sequence in sequences:
        text = str(sequence)
        if text not in seen_sequences:
            unique_sequences.append(text)
            seen_sequences.add(text)

    # Keep the gallery generic so future notebooks can reuse it without depending on
    # similarity-specific tables or HTML layouts.
    card_html = "".join(
        (
            "<div class='gallery-card'>"
            f"{format_glycan_sequence_block(sequence, cartoon_lookup.get(sequence))}"
            "</div>"
        )
        for sequence in unique_sequences
    )
    subtitle_html = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px auto;
      max-width: 1180px;
      color: #222;
      line-height: 1.4;
      padding: 0 18px 36px;
    }}
    .gallery-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-top: 20px;
    }}
    .gallery-card {{
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 16px;
      background: #fafafa;
    }}
    .sequence-block {{
      display: flex;
      align-items: flex-start;
      gap: 16px;
    }}
    .cartoon img {{
      max-width: 180px;
      max-height: 70px;
      border: 1px solid #eee;
      background: white;
      padding: 4px;
    }}
    .cartoon-caption {{
      margin-top: 6px;
      font-size: 12px;
      color: #666;
    }}
    .cartoon-missing {{
      color: #777;
      font-size: 13px;
      border: 1px dashed #bbb;
      padding: 8px 10px;
      background: #fafafa;
    }}
    .sequence-text {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font-size: 13px;
      font-family: Menlo, Monaco, Consolas, monospace;
    }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  {subtitle_html}
  <div class="gallery-grid">
    {card_html}
  </div>
</body>
</html>
"""

    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    return output_path
