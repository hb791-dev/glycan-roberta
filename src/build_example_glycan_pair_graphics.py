from __future__ import annotations

import base64
import html
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMILARITY_DIR = PROJECT_ROOT / "public_reports/08_similarity_scaleup/similarity_scaleup_demo"
OUTPUT_DIR = PROJECT_ROOT / "public_reports/example_glycan_pairs"
REMOTE_SVG_DIR = Path("/tmp")

RESIDUES = ("GlcNAc", "GalNAc", "NeuAc", "Fuc", "Gal", "Man")


@dataclass(frozen=True)
class Glycan:
    accession: str
    sequence: str
    svg_path: Path


@dataclass(frozen=True)
class PairConfig:
    slug: str
    title: str
    anchor: Glycan
    similar: Glycan
    highlight_boxes: dict[str, tuple[float, float, float, float]]


def extract_svg_from_similarity_html(anchor_accession: str, target_accession: str) -> str:
    report_path = SIMILARITY_DIR / f"{anchor_accession}_specific_vs_all.html"
    text = report_path.read_text(errors="ignore")
    pattern = (
        rf"<img src='([^']+)' alt='Cartoon for .*?'>"
        rf"<div class='cartoon-caption'><a href='https://glytoucan.org/Structures/Glycans/{target_accession}'"
    )
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not find SVG source for {target_accession} in {report_path}")
    src = match.group(1)
    if not src.startswith("data:image/svg+xml;base64,"):
        raise ValueError(f"{target_accession} is not embedded in {report_path}")
    return base64.b64decode(src.split(",", 1)[1]).decode("utf-8")


def write_local_svgs() -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_dir = OUTPUT_DIR / "svgs"
    svg_dir.mkdir(exist_ok=True)

    local_files = {
        "G60230HH": svg_dir / "G60230HH.svg",
        "G95849ZD": svg_dir / "G95849ZD.svg",
        "G27893KR": svg_dir / "G27893KR.svg",
        "G56734EJ": svg_dir / "G56734EJ.svg",
    }

    local_files["G60230HH"].write_text(extract_svg_from_similarity_html("G60230HH", "G60230HH"))
    local_files["G95849ZD"].write_text(extract_svg_from_similarity_html("G60230HH", "G95849ZD"))

    for accession in ("G27893KR", "G56734EJ"):
        src = REMOTE_SVG_DIR / f"{accession}.svg"
        if not src.exists():
            raise FileNotFoundError(f"Expected downloaded SVG at {src}")
        shutil.copyfile(src, local_files[accession])

    return local_files


def tokenize_iupac(sequence: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(sequence):
        if sequence[i] in "()":
            tokens.append(sequence[i])
            i += 1
            continue

        next_positions = [sequence.find(residue, i + 1) for residue in RESIDUES]
        next_positions = [pos for pos in next_positions if pos != -1]
        next_paren = [sequence.find("(", i + 1), sequence.find(")", i + 1)]
        next_paren = [pos for pos in next_paren if pos != -1]
        end_candidates = next_positions + next_paren
        end = min(end_candidates) if end_candidates else len(sequence)
        tokens.append(sequence[i:end])
        i = end
    return tokens


def align_tokens(a_tokens: list[str], b_tokens: list[str]) -> list[tuple[str, str]]:
    match_score = 3
    mismatch_score = -2
    gap_score = -2

    dp = [[0] * (len(b_tokens) + 1) for _ in range(len(a_tokens) + 1)]
    move = [[""] * (len(b_tokens) + 1) for _ in range(len(a_tokens) + 1)]

    for i in range(1, len(a_tokens) + 1):
        dp[i][0] = dp[i - 1][0] + gap_score
        move[i][0] = "up"
    for j in range(1, len(b_tokens) + 1):
        dp[0][j] = dp[0][j - 1] + gap_score
        move[0][j] = "left"

    for i in range(1, len(a_tokens) + 1):
        for j in range(1, len(b_tokens) + 1):
            diag_score = dp[i - 1][j - 1] + (
                match_score if a_tokens[i - 1] == b_tokens[j - 1] else mismatch_score
            )
            up_score = dp[i - 1][j] + gap_score
            left_score = dp[i][j - 1] + gap_score
            best = max(diag_score, up_score, left_score)
            dp[i][j] = best
            if best == diag_score:
                move[i][j] = "diag"
            elif best == up_score:
                move[i][j] = "up"
            else:
                move[i][j] = "left"

    aligned: list[tuple[str, str]] = []
    i = len(a_tokens)
    j = len(b_tokens)
    while i > 0 or j > 0:
        step = move[i][j]
        if step == "diag":
            aligned.append((a_tokens[i - 1], b_tokens[j - 1]))
            i -= 1
            j -= 1
        elif step == "up":
            aligned.append((a_tokens[i - 1], ""))
            i -= 1
        else:
            aligned.append(("", b_tokens[j - 1]))
            j -= 1

    aligned.reverse()
    return aligned


def chunk_aligned_pairs(aligned_pairs: list[tuple[str, str]], max_width_ch: int) -> list[list[tuple[str, str]]]:
    chunks: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_width = 0

    for pair in aligned_pairs:
        pair_width = max(len(pair[0]), len(pair[1]), 2) + 1
        if current and current_width + pair_width > max_width_ch:
            chunks.append(current)
            current = []
            current_width = 0
        current.append(pair)
        current_width += pair_width

    if current:
        chunks.append(current)

    return chunks


def render_alignment_row(label: str, row_index: int, aligned_pairs: list[tuple[str, str]]) -> str:
    pieces = []
    for left, right in aligned_pairs:
        token = left if row_index == 0 else right
        other = right if row_index == 0 else left
        diff = token != other
        display = html.escape(token) if token else "&nbsp;"
        width_ch = max(len(left), len(right), 2) + 1
        classes = ["token"]
        if diff:
            classes.append("diff")
        if not token:
            classes.append("gap")
        pieces.append(
            f"<span class=\"{' '.join(classes)}\" style=\"width:{width_ch}ch\">{display}</span>"
        )
    return (
        "<div class=\"sequence-row\">"
        f"<div class=\"row-label\">{html.escape(label)}</div>"
        f"<div class=\"token-line\">{''.join(pieces)}</div>"
        "</div>"
    )


def render_pair_html(config: PairConfig) -> str:
    aligned_pairs = align_tokens(
        tokenize_iupac(config.anchor.sequence),
        tokenize_iupac(config.similar.sequence),
    )
    aligned_chunks = chunk_aligned_pairs(aligned_pairs, max_width_ch=48)

    def glycan_card(kind: str, glycan: Glycan, label: str) -> str:
        x, y, w, h = config.highlight_boxes[kind]
        return f"""
        <div class="glycan-card">
          <div class="glycan-meta">
            <div class="meta-label">{label}</div>
            <div class="meta-accession">{glycan.accession}</div>
          </div>
          <div class="cartoon-stage">
            <img class="glycan-svg" src="{glycan.svg_path.name}" alt="{glycan.accession} cartoon">
            <div class="highlight-box" style="left:{x}%;top:{y}%;width:{w}%;height:{h}%;"></div>
          </div>
        </div>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(config.title)}</title>
  <style>
    :root {{
      --border: #d9d9d9;
      --text: #1f1f1f;
      --muted: #666666;
      --yellow: #fff176;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #ffffff;
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
    }}
    .page {{
      width: 1600px;
      min-height: 1000px;
      padding: 40px;
    }}
    h1 {{
      margin: 0 0 20px;
      font-size: 34px;
      font-weight: 700;
    }}
    .main {{
      display: grid;
      grid-template-columns: 560px 1fr;
      gap: 24px;
      align-items: start;
    }}
    .stack {{
      display: grid;
      gap: 18px;
    }}
    .glycan-card,
    .alignment-panel {{
      border: 1px solid var(--border);
      border-radius: 12px;
      background: #ffffff;
    }}
    .glycan-card {{
      padding: 14px;
    }}
    .glycan-meta {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .meta-label {{
      font-size: 14px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .meta-accession {{
      font-size: 22px;
      font-weight: 700;
    }}
    .cartoon-stage {{
      position: relative;
      height: 250px;
      border: 1px solid var(--border);
      border-radius: 10px;
      display: grid;
      place-items: center;
      overflow: hidden;
    }}
    .glycan-svg {{
      width: 500px;
      height: 210px;
      object-fit: contain;
      display: block;
    }}
    .highlight-box {{
      position: absolute;
      border: 5px solid rgba(255, 241, 118, 0.98);
      border-radius: 12px;
      pointer-events: none;
    }}
    .alignment-panel {{
      padding: 16px;
    }}
    .alignment-title {{
      font-size: 22px;
      font-weight: 700;
      margin-bottom: 12px;
    }}
    .sequence-box {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
      margin-bottom: 14px;
    }}
    .sequence-row {{
      display: grid;
      grid-template-columns: 140px 1fr;
      gap: 16px;
      align-items: start;
      margin-bottom: 12px;
    }}
    .sequence-row:last-child {{
      margin-bottom: 0;
    }}
    .row-label {{
      font-size: 14px;
      font-weight: 700;
      color: var(--muted);
      padding-top: 5px;
    }}
    .token-line {{
      font-family: Menlo, Consolas, "Courier New", monospace;
      font-size: 20px;
      line-height: 1.4;
      white-space: nowrap;
    }}
    .token {{
      display: inline-block;
      vertical-align: top;
      padding: 3px 0;
      border-radius: 6px;
      text-align: left;
    }}
    .token.diff {{
      background: var(--yellow);
      box-shadow: inset 0 0 0 1px rgba(160, 130, 18, 0.25);
    }}
    .token.gap {{
      color: transparent;
      min-height: 1.7em;
    }}
    .legend {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 16px;
      color: var(--muted);
      margin-top: 4px;
    }}
    .swatch {{
      width: 28px;
      height: 18px;
      border-radius: 5px;
      background: var(--yellow);
      border: 1px solid rgba(160, 130, 18, 0.25);
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>{html.escape(config.title)}</h1>
    <div class="main">
      <div class="stack">
        {glycan_card("anchor", config.anchor, "Anchor")}
        {glycan_card("similar", config.similar, "Similar")}
      </div>
      <div class="alignment-panel">
        <div class="alignment-title">Aligned IUPAC</div>
        {''.join(
            '<div class="sequence-box">'
            + render_alignment_row(config.anchor.accession, 0, chunk)
            + render_alignment_row(config.similar.accession, 1, chunk)
            + '</div>'
            for chunk in aligned_chunks
        )}
        <div class="legend"><span class="swatch"></span>Yellow marks the difference.</div>
      </div>
    </div>
  </div>
</body>
</html>
"""


def build_pair_configs(svg_paths: dict[str, Path]) -> list[PairConfig]:
    g60230 = Glycan(
        accession="G60230HH",
        sequence="Mana1-2Mana1-2Mana1-3(Mana1-2Mana1-3(Mana1-2Mana1-6)Mana1-6)Manb1-4GlcNAcb1-4GlcNAcb",
        svg_path=svg_paths["G60230HH"],
    )
    g95849 = Glycan(
        accession="G95849ZD",
        sequence="Mana1-2Mana1-3(Mana1-2Mana1-2Mana1-6)Mana1-3(Mana1-2Mana1-6)Manb1-4GlcNAcb1-4GlcNAcb",
        svg_path=svg_paths["G95849ZD"],
    )
    g27893 = Glycan(
        accession="G27893KR",
        sequence="Fuca1-2(GalNAca1-3)Galb1-3(Galb1-4GlcNAcb1-6)GalNAca",
        svg_path=svg_paths["G27893KR"],
    )
    g56734 = Glycan(
        accession="G56734EJ",
        sequence="Fuca1-2(GalNAca1-3)Galb1-3(Galb1-3GlcNAcb1-6)GalNAca",
        svg_path=svg_paths["G56734EJ"],
    )

    return [
        PairConfig(
            slug="pair_high_mannose",
            title="Example Pair 1",
            anchor=g60230,
            similar=g95849,
            highlight_boxes={
                "anchor": (8.0, 18.0, 21.0, 24.0),
                "similar": (8.0, 18.0, 21.0, 24.0),
            },
        ),
        PairConfig(
            slug="pair_o_glycan_linkage",
            title="Example Pair 2",
            anchor=g27893,
            similar=g56734,
            highlight_boxes={
                "anchor": (58.0, 24.0, 22.0, 35.0),
                "similar": (58.0, 24.0, 22.0, 35.0),
            },
        ),
    ]


def main() -> None:
    svg_paths = write_local_svgs()
    for config in build_pair_configs(svg_paths):
        pair_dir = OUTPUT_DIR / config.slug
        pair_dir.mkdir(parents=True, exist_ok=True)

        for glycan in (config.anchor, config.similar):
            shutil.copyfile(glycan.svg_path, pair_dir / glycan.svg_path.name)

        html_path = pair_dir / f"{config.slug}.html"
        html_path.write_text(render_pair_html(config))


if __name__ == "__main__":
    main()
