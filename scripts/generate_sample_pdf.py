"""
One-time helper: convert the IT password-reset markdown source into a PDF
so the sample_docs/ directory has a real PDF to exercise the PDF loader.

    python scripts/generate_sample_pdf.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SRC = Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "it_password_reset_source.md"
OUT = Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "it_password_reset.pdf"


def _md_line_to_flowable(line: str, styles):
    """Very lightweight md → reportlab converter for headings, bullets, paragraphs."""
    stripped = line.rstrip()
    if not stripped:
        return Spacer(1, 8)

    # Headings
    if stripped.startswith("# "):
        return Paragraph(stripped[2:], styles["Title"])
    if stripped.startswith("## "):
        return Paragraph(stripped[3:], styles["Heading1"])
    if stripped.startswith("### "):
        return Paragraph(stripped[4:], styles["Heading2"])

    # Bullet
    if stripped.startswith("- "):
        return Paragraph(f"• {stripped[2:]}", styles["MdBullet"])
    if re.match(r"^\d+\.\s", stripped):
        return Paragraph(stripped, styles["MdBullet"])

    # Bold/italic — strip markdown markers, reportlab supports <b>
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", stripped)
    html = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", html)
    return Paragraph(html, styles["BodyText"])


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: source not found: {SRC}", file=sys.stderr)
        return 1

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="MdBullet", parent=styles["BodyText"], leftIndent=16, spaceAfter=4))
    styles["Title"].fontSize = 20
    styles["Heading1"].fontSize = 14
    styles["Heading2"].fontSize = 12

    doc = SimpleDocTemplate(str(OUT), pagesize=letter, title="IT SOP — Password Reset")
    flow = []
    for line in SRC.read_text(encoding="utf-8").splitlines():
        flow.append(_md_line_to_flowable(line, styles))
    doc.build(flow)

    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
