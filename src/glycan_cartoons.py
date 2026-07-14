"""Helpers for resolving and rendering glycan cartoons from compact IUPAC."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from functools import lru_cache
from html import escape
import hashlib
import json
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING, Any
from urllib.error import URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    import pandas as pd


GLYLOOKUP_BASE_URL = "https://glylookup.glyomics.org"
GLYMAGE_BASE_URL = "https://glymage.glyomics.org"
GLYTOUCAN_BASE_URL = "https://glytoucan.org/Structures/Glycans"
GLYTOUCAN_ACCESSION_PATTERN = re.compile(r"^[A-Z][0-9]{5}[A-Z]{2}$")


def _post_form_json(url: str, data: dict[str, Any], timeout: int = 60) -> Any:
    """POST one URL-encoded form and parse the JSON response."""
    # Both GlyLookup and Glymage use the same simple form-post API shape.
    payload = urlencode(data).encode("utf-8")
    request = Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8")
    return json.loads(text)


def _download_binary(url: str, timeout: int = 60) -> bytes:
    """Download one image payload as raw bytes."""
    request = Request(url, method="GET")
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _image_file_to_data_uri(image_path: str | Path) -> str:
    """Convert one saved local image file into a standalone HTML data URI."""
    image_path = Path(image_path)
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
    }.get(image_path.suffix.lower(), "application/octet-stream")
    encoded_bytes = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded_bytes}"


def _normalize_manifest_text(value) -> str:
    """Return one manifest value as a safe string, treating NaN-like values as blank."""
    if value is None:
        return ""
    # Pandas uses NaN floats for missing string values in CSV-backed dataframes.
    if isinstance(value, float) and value != value:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _normalize_glytoucan_accession(accession) -> str:
    """Return the accession only when it looks like a real GlyTouCan ID."""
    normalized_accession = _normalize_manifest_text(accession).upper()
    if GLYTOUCAN_ACCESSION_PATTERN.fullmatch(normalized_accession):
        return normalized_accession
    return ""


def _cartoon_asset_filename(
    sequence: str,
    accession: str,
    image_format: str,
) -> str:
    """Create a stable filename for one saved cartoon asset."""
    if accession:
        stem = accession
    else:
        stem = f"sequence_{hashlib.sha1(sequence.encode('utf-8')).hexdigest()[:12]}"
    normalized_suffix = image_format.lower().lstrip(".") or "svg"
    return f"{stem}.{normalized_suffix}"


def _candidate_cartoon_asset_paths(
    sequence: str,
    accession: str,
    asset_dir,
    image_format: str = "svg",
) -> list[Path]:
    """Return plausible local asset paths for one sequence/accession pair.

    This gives the cache layer a way to reuse previously downloaded cartoon
    files even when a fresh notebook run does not have enough information to
    re-resolve metadata from GlyLookup or Glymage.
    """
    asset_dir = Path(asset_dir)
    normalized_suffixes = []
    for suffix in (image_format, "svg", "png", "jpg", "jpeg"):
        normalized_suffix = str(suffix).lower().lstrip(".")
        if normalized_suffix and normalized_suffix not in normalized_suffixes:
            normalized_suffixes.append(normalized_suffix)

    candidate_stems = []
    if accession:
        candidate_stems.append(str(accession).strip())
    candidate_stems.append(f"sequence_{hashlib.sha1(sequence.encode('utf-8')).hexdigest()[:12]}")

    candidate_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for stem in candidate_stems:
        if not stem:
            continue
        for suffix in normalized_suffixes:
            candidate_path = asset_dir / f"{stem}.{suffix}"
            if candidate_path not in seen_paths:
                candidate_paths.append(candidate_path)
                seen_paths.add(candidate_path)

    return candidate_paths


def _direct_accession_metadata(
    sequence: str,
    accession: str,
    image_format: str,
    display: str,
) -> dict[str, str]:
    """Build the stable metadata we can derive directly from one accession."""
    normalized_accession = _normalize_glytoucan_accession(accession)
    return {
        "sequence": sequence,
        "accession": normalized_accession,
        "glytoucan_url": f"{GLYTOUCAN_BASE_URL}/{normalized_accession}",
        "image_url": f"{GLYMAGE_BASE_URL}/image/snfg/{display}/{normalized_accession}.{image_format}",
        "lookup_status": "accession_supplied",
        "lookup_errors": "",
    }


def _cache_only_manifest_row(
    sequence: str,
    accession: str,
) -> dict[str, str]:
    """Return one manifest row that records a cache-only miss without web lookup.

    This keeps the downstream HTML/report code simple: the row still has the
    standard manifest columns, but it intentionally carries no remote image URL.
    That guarantees a cache-only notebook run never sneaks in a live website
    request through the final HTML.
    """
    normalized_accession = _normalize_glytoucan_accession(accession)
    glytoucan_url = f"{GLYTOUCAN_BASE_URL}/{normalized_accession}" if normalized_accession else ""
    return {
        "sequence": sequence,
        "accession": normalized_accession,
        "glytoucan_url": glytoucan_url,
        "image_url": "",
        "lookup_status": "cache_only_miss",
        "lookup_errors": "Sequence was not found in the supplied cartoon cache.",
        "local_image_path": "",
        "local_image_status": "cache_only_miss",
    }


def _is_task_backed_image_url(image_url: str) -> bool:
    """Return True when the URL depends on a temporary Glymage task ID."""
    normalized_url = _normalize_manifest_text(image_url)
    return normalized_url.startswith(f"{GLYMAGE_BASE_URL}/getimage?task_id=")


def _is_accession_backed_image_url(image_url: str) -> bool:
    """Return True when the URL is one of the stable Glymage accession image URLs."""
    normalized_url = _normalize_manifest_text(image_url)
    return normalized_url.startswith(f"{GLYMAGE_BASE_URL}/image/snfg/")


def _reuse_or_refresh_existing_row(
    sequence: str,
    existing_row: dict[str, Any] | None,
    supplied_accession: str,
    image_format: str,
    display: str,
) -> dict[str, str] | None:
    """Return the best existing manifest row, or None when a fresh resolution is better.

    Reuse is intentionally selective:
    - if the notebook now knows an accession, prefer that stronger metadata
    - if an older row already has an accession, rebuild direct stable URLs from it
    - keep any working local-image cache pointer
    - do not blindly trust stale task-backed `getimage?task_id=...` URLs
    """
    if existing_row is None:
        return None

    normalized_row = {key: _normalize_manifest_text(value) for key, value in existing_row.items()}
    existing_accession = _normalize_glytoucan_accession(normalized_row.get("accession", ""))
    existing_image_url = normalized_row.get("image_url", "")
    existing_local_path = normalized_row.get("local_image_path", "")
    preferred_accession = _normalize_glytoucan_accession(supplied_accession) or existing_accession

    if preferred_accession:
        refreshed_row = dict(normalized_row)
        refreshed_row.update(
            _direct_accession_metadata(
                sequence=sequence,
                accession=preferred_accession,
                image_format=image_format,
                display=display,
            )
        )
        if existing_local_path:
            refreshed_row["local_image_path"] = existing_local_path
            refreshed_row["local_image_status"] = normalized_row.get("local_image_status", "")
        else:
            refreshed_row["local_image_path"] = ""
            refreshed_row["local_image_status"] = ""
        return refreshed_row

    if existing_local_path:
        normalized_row["accession"] = existing_accession
        if not existing_accession:
            normalized_row["glytoucan_url"] = ""
        return normalized_row

    if (
        existing_image_url
        and not _is_task_backed_image_url(existing_image_url)
        and (existing_accession or not _is_accession_backed_image_url(existing_image_url))
    ):
        normalized_row["accession"] = existing_accession
        if not existing_accession:
            normalized_row["glytoucan_url"] = ""
        return normalized_row

    return None


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
    accession: str | None = None,
    image_format: str = "svg",
    display: str = "compact",
    lookup_timeout: int = 60,
) -> dict[str, str]:
    """Resolve GlyLookup/Glymage metadata for one compact IUPAC sequence."""
    # Cache by sequence so repeated notebook runs do not re-request the same cartoon
    # metadata over and over within one Python session.
    metadata = {
        "sequence": sequence,
        "accession": _normalize_glytoucan_accession(accession),
        "glytoucan_url": "",
        "image_url": "",
        "lookup_status": "not_attempted",
        "lookup_errors": "",
    }

    if metadata["accession"]:
        # When an accession is already known, we can skip the fragile lookup step
        # and build the stable image/link metadata directly.
        return _direct_accession_metadata(
            sequence=sequence,
            accession=metadata["accession"],
            image_format=image_format,
            display=display,
        )

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
    accession_by_sequence: dict[str, str] | None = None,
    image_format: str = "svg",
    display: str = "compact",
    lookup_timeout: int = 60,
    existing_manifest_df=None,
    allow_live_lookup: bool = True,
) -> "pd.DataFrame":
    """Build one metadata table for all unique sequences in the analysis."""
    import pandas as pd

    accession_lookup = {
        str(sequence): _normalize_glytoucan_accession(accession)
        for sequence, accession in (accession_by_sequence or {}).items()
    }
    unique_sequences = []
    seen_sequences: set[str] = set()
    for sequence in sequences:
        text = str(sequence)
        if text not in seen_sequences:
            unique_sequences.append(text)
            seen_sequences.add(text)

    existing_lookup: dict[str, dict[str, Any]] = {}
    if existing_manifest_df is not None:
        existing_df = pd.DataFrame(existing_manifest_df).copy()
        if "sequence" in existing_df.columns:
            existing_df["sequence"] = existing_df["sequence"].fillna("").map(str)
            # Borrowed cache manifests can contain the same sequence more than once
            # across runs. Keep the last copy so later-appended sources still win.
            existing_df = existing_df.drop_duplicates(subset=["sequence"], keep="last")
            existing_lookup = existing_df.set_index("sequence").to_dict(orient="index")

    manifest_rows = []
    for sequence in unique_sequences:
        supplied_accession = accession_lookup.get(sequence, "")
        reusable_row = _reuse_or_refresh_existing_row(
            sequence=sequence,
            existing_row=existing_lookup.get(sequence),
            supplied_accession=supplied_accession,
            image_format=image_format,
            display=display,
        )
        if reusable_row is not None:
            manifest_rows.append(reusable_row)
            continue

        if not allow_live_lookup:
            # In cache-only mode we stop here on purpose. The calling notebook
            # wants a fully offline HTML example, so a miss should stay a miss
            # rather than triggering a fresh GlyLookup/Glymage request.
            manifest_rows.append(
                _cache_only_manifest_row(
                    sequence=sequence,
                    accession=supplied_accession,
                )
            )
            continue

        # Resolve each unique sequence once, then keep the results in a flat table that can
        # be saved, inspected in the notebook, or converted back into a lookup dictionary.
        manifest_rows.append(
            resolve_cartoon_metadata(
                sequence=sequence,
                developer_email=developer_email,
                accession=supplied_accession,
                image_format=image_format,
                display=display,
                lookup_timeout=lookup_timeout,
            )
        )
    return pd.DataFrame(manifest_rows)


def cache_cartoon_images(
    cartoon_manifest_df,
    asset_dir,
    image_format: str = "svg",
    download_timeout: int = 60,
    allow_download: bool = True,
):
    """Download resolved cartoon images so HTML reports do not depend on expiring URLs.

    GlyTouCan-backed images usually live at stable accession URLs, but Glymage
    on-demand renders use temporary task IDs. Saving both kinds of images into the
    results folder makes the final HTML portable and much less fragile.
    """
    import pandas as pd

    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    cached_df = cartoon_manifest_df.copy()
    if "local_image_path" not in cached_df.columns:
        cached_df["local_image_path"] = ""
    if "local_image_status" not in cached_df.columns:
        cached_df["local_image_status"] = ""

    for row_index, row in cached_df.iterrows():
        existing_local_image_path = str(row.get("local_image_path", "") or "").strip()
        if existing_local_image_path and Path(existing_local_image_path).exists():
            cached_df.at[row_index, "local_image_path"] = existing_local_image_path
            cached_df.at[row_index, "local_image_status"] = "cached_existing"
            continue

        accession = str(row.get("accession", "") or "").strip()
        sequence = str(row.get("sequence", "") or "")
        for candidate_path in _candidate_cartoon_asset_paths(
            sequence=sequence,
            accession=accession,
            asset_dir=asset_dir,
            image_format=image_format,
        ):
            if candidate_path.exists() and candidate_path.stat().st_size > 0:
                cached_df.at[row_index, "local_image_path"] = str(candidate_path)
                cached_df.at[row_index, "local_image_status"] = "cached_existing"
                break
        if str(cached_df.at[row_index, "local_image_path"]).strip():
            continue

        image_url = str(row.get("image_url", "") or "").strip()
        if not image_url:
            if not str(cached_df.at[row_index, "local_image_status"]).strip():
                cached_df.at[row_index, "local_image_status"] = "no_remote_image"
            continue

        if not allow_download:
            # Clear the remote URL before HTML generation so cache-only runs do
            # not quietly fall back to the web in the browser.
            cached_df.at[row_index, "image_url"] = ""
            cached_df.at[row_index, "local_image_status"] = "cache_only_miss"
            continue

        remote_suffix = Path(urlparse(image_url).path).suffix.lstrip(".")
        asset_name = _cartoon_asset_filename(
            sequence=sequence,
            accession=accession,
            image_format=remote_suffix or image_format,
        )
        asset_path = asset_dir / asset_name

        if asset_path.exists() and asset_path.stat().st_size > 0:
            cached_df.at[row_index, "local_image_path"] = str(asset_path)
            cached_df.at[row_index, "local_image_status"] = "cached_existing"
            continue

        try:
            # Download immediately while the task-backed image is still valid.
            asset_bytes = _download_binary(image_url, timeout=download_timeout)
            asset_path.write_bytes(asset_bytes)
            cached_df.at[row_index, "local_image_path"] = str(asset_path)
            cached_df.at[row_index, "local_image_status"] = "downloaded"
        except (URLError, TimeoutError, OSError, ValueError) as error:
            existing_errors = str(row.get("lookup_errors", "") or "").strip()
            combined_errors = [value for value in (existing_errors, f"local_cache_error: {error}") if value]
            cached_df.at[row_index, "lookup_errors"] = "; ".join(combined_errors)
            cached_df.at[row_index, "local_image_status"] = "download_failed"

    return cached_df


def cartoon_lookup_from_manifest(cartoon_manifest_df) -> dict[str, dict[str, str]]:
    """Convert a cartoon manifest dataframe into a sequence-keyed lookup."""
    normalized_df = cartoon_manifest_df.copy()
    for column_name in normalized_df.columns:
        normalized_df[column_name] = normalized_df[column_name].map(_normalize_manifest_text)
    normalized_df = normalized_df.drop_duplicates(subset=["sequence"], keep="last")
    return normalized_df.set_index("sequence").to_dict(orient="index")


def summarize_cartoon_manifest(cartoon_manifest_df) -> dict[str, "pd.DataFrame"]:
    """Return notebook-friendly summary tables for one cartoon manifest."""
    import pandas as pd

    manifest_df = pd.DataFrame(cartoon_manifest_df).copy()
    if manifest_df.empty:
        empty_counts_df = pd.DataFrame(columns=["value", "count"])
        overview_df = pd.DataFrame(
            [
                {
                    "total_sequences": 0,
                    "real_accession_rows": 0,
                    "stable_accession_url_rows": 0,
                    "task_backed_url_rows": 0,
                    "local_image_rows": 0,
                    "download_failed_rows": 0,
                }
            ]
        )
        return {
            "overview_df": overview_df,
            "lookup_status_df": empty_counts_df,
            "local_image_status_df": empty_counts_df,
        }

    normalized_df = manifest_df.copy()
    for column_name in normalized_df.columns:
        normalized_df[column_name] = normalized_df[column_name].map(_normalize_manifest_text)

    accession_series = normalized_df.get("accession", pd.Series("", index=normalized_df.index)).map(
        _normalize_glytoucan_accession
    )
    image_url_series = normalized_df.get("image_url", pd.Series("", index=normalized_df.index)).map(
        _normalize_manifest_text
    )
    local_image_path_series = normalized_df.get("local_image_path", pd.Series("", index=normalized_df.index)).map(
        _normalize_manifest_text
    )
    local_image_status_series = normalized_df.get(
        "local_image_status", pd.Series("", index=normalized_df.index)
    ).map(_normalize_manifest_text)
    lookup_status_series = normalized_df.get("lookup_status", pd.Series("", index=normalized_df.index)).map(
        _normalize_manifest_text
    )

    overview_df = pd.DataFrame(
        [
            {
                "total_sequences": int(len(normalized_df)),
                "real_accession_rows": int(accession_series.ne("").sum()),
                "stable_accession_url_rows": int(image_url_series.map(_is_accession_backed_image_url).sum()),
                "task_backed_url_rows": int(image_url_series.map(_is_task_backed_image_url).sum()),
                "local_image_rows": int(local_image_path_series.ne("").sum()),
                "download_failed_rows": int(local_image_status_series.eq("download_failed").sum()),
            }
        ]
    )

    lookup_status_df = (
        lookup_status_series.replace("", "blank")
        .value_counts(dropna=False)
        .rename_axis("value")
        .reset_index(name="count")
    )
    local_image_status_df = (
        local_image_status_series.replace("", "blank")
        .value_counts(dropna=False)
        .rename_axis("value")
        .reset_index(name="count")
    )

    return {
        "overview_df": overview_df,
        "lookup_status_df": lookup_status_df,
        "local_image_status_df": local_image_status_df,
    }


def format_glycan_sequence_block(sequence: str, cartoon_row: dict[str, str] | None) -> str:
    """Render one compact HTML block for a glycan sequence and its cartoon."""
    image_html = "<div class='cartoon-missing'>No cartoon resolved</div>"
    caption_html = ""
    if cartoon_row is not None:
        image_url = _normalize_manifest_text(cartoon_row.get("image_url", ""))
        local_image_path = _normalize_manifest_text(cartoon_row.get("local_image_path", ""))
        accession = _normalize_manifest_text(cartoon_row.get("accession", ""))
        glytoucan_url = _normalize_manifest_text(cartoon_row.get("glytoucan_url", ""))
        lookup_status = _normalize_manifest_text(cartoon_row.get("lookup_status", ""))
        local_image_status = _normalize_manifest_text(cartoon_row.get("local_image_status", ""))
        if local_image_path and Path(local_image_path).exists():
            # Embed the saved file directly so a copied HTML report still renders.
            image_html = (
                f"<img src='{escape(_image_file_to_data_uri(local_image_path), quote=True)}' "
                f"alt='Cartoon for {escape(sequence)}'>"
            )
        elif image_url and not _is_task_backed_image_url(image_url):
            # Stable accession-backed URLs are okay to reference directly if we do not
            # have a local cached copy yet.
            image_html = f"<img src='{escape(image_url, quote=True)}' alt='Cartoon for {escape(sequence)}'>"
        if accession and glytoucan_url:
            # When an accession is available, link it directly so the HTML report can
            # double as a quick lookup sheet.
            caption_html = (
                f"<div class='cartoon-caption'><a href='{escape(glytoucan_url, quote=True)}' "
                f"target='_blank' rel='noopener'>{escape(accession)}</a></div>"
            )
        elif local_image_status:
            # If a task-backed image was not cached successfully, show a plain status
            # instead of leaving the browser to display a broken stale-image request.
            caption_html = f"<div class='cartoon-caption'>{escape(local_image_status)}</div>"
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
