"""Convert the two flight-reading markdown docs to iPad-friendly PDF.

Produces:
  - docs/99-raw/flight-pdfs/00-flight-reading-list.pdf
  - docs/99-raw/flight-pdfs/00-model-py-section-map.pdf

Pipeline:
  markdown --(pandoc)--> typst --(typst compile)--> pdf

Uses the shared iPad-optimized template `tools/flight-reading-template.typ`
and the pandoc wrapper `tools/pandoc-typst-template.typ`.

Requirements:
  - pandoc >= 3.1 (with typst writer)  -- already present on most systems
  - typst >= 0.12  -- `brew install typst`

Usage:
    python tools/build_md_to_pdf.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = PROJECT_ROOT / "tools"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "99-raw" / "flight-pdfs"
PANDOC_TEMPLATE = TOOLS_DIR / "pandoc-typst-template.typ"
TYPST_TEMPLATE = TOOLS_DIR / "flight-reading-template.typ"


@dataclass(frozen=True)
class MdSource:
    """A single markdown → PDF conversion job."""

    md_path: Path
    out_pdf: Path
    title: str
    subtitle: str | None = None
    date: str | None = None


SOURCES: tuple[MdSource, ...] = (
    MdSource(
        md_path=PROJECT_ROOT / "docs" / "10-research" / "flight-reading-list-2026-05-06.md",
        out_pdf=OUTPUT_DIR / "00-flight-reading-list.pdf",
        title="Flight Reading List",
        subtitle="16 小时飞机论文精读清单 · 2026-05-06 出差",
        date="2026-05-06",
    ),
    MdSource(
        md_path=PROJECT_ROOT / "docs" / "00-context" / "model-py-section-map.md",
        out_pdf=OUTPUT_DIR / "00-model-py-section-map.pdf",
        title="model.py ↔ 论文 Section Map",
        subtitle="源码 vs 论文对照阅读地图",
        date="2026-05-04",
    ),
)


def _check_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"'{name}' not found in PATH. Install it before running this script "
            f"(e.g. `brew install {name}` for pandoc/typst)."
        )
    return path


# Tuned column-width presets for common markdown table shapes on A5 pages.
# Pandoc's typst writer defaults to even-width columns; we override for
# "wide prose column, narrow numeric columns" which matches our sources.
_TABLE_WIDTH_OVERRIDES: dict[int, str] = {
    4: "(18%, 52%, 15%, 15%)",  # e.g. time-window | content | hours | cumulative
    3: "(25%, 50%, 25%)",        # e.g. label | description | value
    2: "(35%, 65%)",             # e.g. term | definition
}

_TABLE_COLUMNS_RE = re.compile(
    r"columns:\s*\(\s*(\d+%)(\s*,\s*\1){1,}\s*\)",
)


def _rebalance_table_columns(typst_path: Path) -> None:
    """Rewrite equal-width `columns: (x%, x%, ...)` to our A5-tuned presets."""
    text = typst_path.read_text(encoding="utf-8")

    def _replace(match: re.Match[str]) -> str:
        # Count the `%` occurrences to infer column count (avoid overcounting
        # commas since Typst tuples may have trailing commas).
        n_cols = match.group(0).count("%")
        override = _TABLE_WIDTH_OVERRIDES.get(n_cols)
        if override is None:
            return match.group(0)
        return f"columns: {override}"

    rewritten = _TABLE_COLUMNS_RE.sub(_replace, text)
    if rewritten != text:
        typst_path.write_text(rewritten, encoding="utf-8")


def _convert_one(src: MdSource) -> None:
    if not src.md_path.exists():
        raise FileNotFoundError(f"Source markdown not found: {src.md_path}")

    src.out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Strip the leading `# H1` line from the source — the template renders
    # the paper title as a cover header already, and re-emitting it as an
    # H1 in the body creates visual duplication on page 1.
    processed_md = src.out_pdf.parent / f".processed-{src.md_path.name}"
    raw = src.md_path.read_text(encoding="utf-8").splitlines()
    if raw and raw[0].startswith("# "):
        raw = raw[1:]
        while raw and raw[0].strip() == "":
            raw = raw[1:]
    processed_md.write_text("\n".join(raw), encoding="utf-8")

    # Intermediate typst source lives next to the final PDF for easier debug.
    typst_out = src.out_pdf.with_suffix(".typ")

    pandoc_cmd = [
        _check_tool("pandoc"),
        "--standalone",
        "--from=gfm",
        "--to=typst",
        f"--template={PANDOC_TEMPLATE}",
        f"--metadata=title:{src.title}",
        "-o",
        str(typst_out),
        str(processed_md),
    ]
    if src.subtitle:
        pandoc_cmd.insert(-3, f"--metadata=subtitle:{src.subtitle}")
    if src.date:
        pandoc_cmd.insert(-3, f"--metadata=date:{src.date}")

    subprocess.run(pandoc_cmd, check=True)

    # Post-process: rebalance equal-width table columns on A5.
    # Pandoc emits `columns: (25%, 25%, 25%, 25%)` by default; that wastes
    # width on numeric columns and squeezes prose columns. For 4-col tables
    # that follow a "time | content | number | number" shape, allocate more
    # width to the second (content) column.
    _rebalance_table_columns(typst_out)

    # typst compile needs the imported template to resolve relative to the
    # .typ file. We already point pandoc-typst-template.typ to import from
    # `flight-reading-template.typ`, so copy that alongside the output.
    sibling_template = typst_out.parent / TYPST_TEMPLATE.name
    shutil.copyfile(TYPST_TEMPLATE, sibling_template)

    typst_cmd = [
        _check_tool("typst"),
        "compile",
        "--root",
        str(typst_out.parent),
        str(typst_out),
        str(src.out_pdf),
    ]
    subprocess.run(typst_cmd, check=True)

    # Clean up the processed markdown intermediary. Keep the .typ + sibling
    # template around so a human can re-tweak the layout without touching
    # the python pipeline.
    processed_md.unlink(missing_ok=True)

    size_kb = src.out_pdf.stat().st_size / 1024
    print(f"  Generated: {src.out_pdf.relative_to(PROJECT_ROOT)} ({size_kb:.1f} KB)")


def main() -> None:
    print("Building iPad-friendly PDFs from markdown sources ...")
    for src in SOURCES:
        print(f"- {src.md_path.relative_to(PROJECT_ROOT)}")
        _convert_one(src)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as err:
        print(f"ERROR: subcommand failed with exit code {err.returncode}", file=sys.stderr)
        sys.exit(err.returncode)
