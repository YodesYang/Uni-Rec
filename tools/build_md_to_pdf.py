"""Convert the two flight-reading markdown docs to printable PDF.

Produces:
  - docs/99-raw/flight-pdfs/00-flight-reading-list.pdf
  - docs/99-raw/flight-pdfs/00-model-py-section-map.pdf

Uses a simple line-based renderer rather than full markdown parsing. Supports:
  - Headers (# ## ### #### -> font size steps)
  - Bullet lists (`-` / `*` / `1.` lines)
  - Inline emphasis/code is rendered as plain text (we keep `**bold**` and
    `` `code` `` markers visible as literal punctuation; that is acceptable
    for an offline reading aid).

Usage:
    python tools/build_md_to_pdf.py
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "docs" / "99-raw" / "flight-pdfs"

SOURCES: list[tuple[Path, Path, str]] = [
    (
        PROJECT_ROOT / "docs" / "10-research" / "flight-reading-list-2026-05-06.md",
        OUTPUT_DIR / "00-flight-reading-list.pdf",
        "Flight Reading List · 2026-05-06 出差",
    ),
    (
        PROJECT_ROOT / "docs" / "00-context" / "model-py-section-map.md",
        OUTPUT_DIR / "00-model-py-section-map.pdf",
        "model.py ↔ 论文 Section Map",
    ),
]


_CJK_FONT_CANDIDATES = [
    ("STHeiti", "/System/Library/Fonts/STHeiti Medium.ttc"),
    ("STHeitiLight", "/System/Library/Fonts/STHeiti Light.ttc"),
    ("Songti", "/System/Library/Fonts/Supplemental/Songti.ttc"),
]

CJK_FONT_NAME: str | None = None
for name, path in _CJK_FONT_CANDIDATES:
    if Path(path).exists():
        try:
            pdfmetrics.registerFont(TTFont(name, path))
            CJK_FONT_NAME = name
            break
        except Exception:  # noqa: BLE001
            continue


def _strip_inline_markdown(text: str) -> str:
    """Remove markdown emphasis tokens so they don't clutter the printed page."""
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<font face='Courier'>\1</font>", text)
    # reportlab paragraphs interpret `<` `>` `&`. Be sure we don't accidentally
    # break markup by checking for unmatched angle brackets.
    return text


def _make_styles():
    base = getSampleStyleSheet()
    font = CJK_FONT_NAME or "Helvetica"

    styles = {
        "title": ParagraphStyle(
            "Title", parent=base["Heading1"], fontName=font,
            fontSize=18, leading=22, spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName=font,
            fontSize=15, leading=19, spaceBefore=10, spaceAfter=6,
            textColor=colors.HexColor("#0b5cad"),
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName=font,
            fontSize=12.5, leading=16, spaceBefore=8, spaceAfter=4,
            textColor=colors.HexColor("#1f6bbf"),
        ),
        "h3": ParagraphStyle(
            "H3", parent=base["Heading3"], fontName=font,
            fontSize=11, leading=14, spaceBefore=6, spaceAfter=3,
            textColor=colors.HexColor("#444444"),
        ),
        "h4": ParagraphStyle(
            "H4", parent=base["Heading4"], fontName=font,
            fontSize=10, leading=13, spaceBefore=5, spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName=font,
            fontSize=9, leading=12, spaceAfter=3,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["BodyText"], fontName=font,
            fontSize=9, leading=12, leftIndent=14, bulletIndent=4,
        ),
        "code": ParagraphStyle(
            "Code", parent=base["Code"], fontName="Courier",
            fontSize=8, leading=10,
        ),
        "meta": ParagraphStyle(
            "Meta", parent=base["BodyText"], fontName=font,
            fontSize=8, leading=10, textColor=colors.HexColor("#777777"),
        ),
    }
    return styles


def _render_markdown(md_text: str, styles: dict, title: str):
    """Convert markdown text to a list of flowables.

    Supported constructs (line-based):
      # ~ ####   heading levels
      - item    bullet list
      * item    bullet list
      1. item   ordered list (rendered same as bullets)
      ```...``` fenced code blocks
      | a | b |  table rows (rendered as preformatted text, still readable)
      paragraph breaks on blank line
    """
    story: list = [Paragraph(title, styles["title"])]
    story.append(Spacer(1, 4 * mm))

    lines = md_text.splitlines()
    i = 0
    in_code = False
    code_buf: list[str] = []

    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()

        if stripped.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_buf), styles["code"]))
                code_buf = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        if stripped == "":
            story.append(Spacer(1, 2 * mm))
            i += 1
            continue

        # Horizontal rule
        if stripped in ("---", "***"):
            story.append(Spacer(1, 3 * mm))
            i += 1
            continue

        # Headings
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            text = _strip_inline_markdown(m.group(2))
            style = {1: "h1", 2: "h2", 3: "h3", 4: "h4"}.get(level, "h4")
            story.append(Paragraph(text, styles[style]))
            i += 1
            continue

        # Bullet / ordered list item
        m = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        if m:
            indent = len(m.group(1)) // 2
            text = _strip_inline_markdown(m.group(2))
            bullet_style = ParagraphStyle(
                f"b{indent}", parent=styles["bullet"], leftIndent=14 + indent * 10,
            )
            story.append(Paragraph(f"• {text}", bullet_style))
            i += 1
            continue

        m = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if m:
            indent = len(m.group(1)) // 2
            num = m.group(2)
            text = _strip_inline_markdown(m.group(3))
            bullet_style = ParagraphStyle(
                f"n{indent}", parent=styles["bullet"], leftIndent=14 + indent * 10,
            )
            story.append(Paragraph(f"{num}. {text}", bullet_style))
            i += 1
            continue

        # Simple table row: render as monospace table for readability. Use a
        # CJK-capable font when available so Chinese cells don't render as
        # boxes. (reportlab's default Courier is ASCII-only.)
        if stripped.startswith("|") and stripped.endswith("|"):
            table_buf = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_buf.append(lines[i])
                i += 1
            table_font = CJK_FONT_NAME or "Courier"
            table_style = ParagraphStyle(
                "Table", parent=styles["code"], fontName=table_font,
                fontSize=8, leading=10,
            )
            story.append(Preformatted("\n".join(table_buf), table_style))
            continue

        # Default paragraph
        text = _strip_inline_markdown(stripped)
        # Handle unescaped angle brackets for reportlab: only convert when they
        # don't look like our inline <b> / <font> tags.
        text = re.sub(r"<(?!/?(?:b|font\b))", "&lt;", text)
        story.append(Paragraph(text, styles["body"]))
        i += 1

    if in_code and code_buf:
        story.append(Preformatted("\n".join(code_buf), styles["code"]))

    return story


def build_pdf(md_path: Path, pdf_path: Path, title: str) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    md_text = md_path.read_text(encoding="utf-8")
    styles = _make_styles()
    story = _render_markdown(md_text, styles, title)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=12 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=title,
    )
    doc.build(story)
    size_kb = pdf_path.stat().st_size / 1024
    print(f"Generated: {pdf_path.relative_to(PROJECT_ROOT)}  ({size_kb:.1f} KB)")


def main() -> None:
    for md_path, pdf_path, title in SOURCES:
        if not md_path.exists():
            print(f"SKIP (not found): {md_path}")
            continue
        build_pdf(md_path, pdf_path, title)


if __name__ == "__main__":
    main()
