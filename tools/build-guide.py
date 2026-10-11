"""Render the repository's Markdown guide with ReportLab; no browser required."""
import html
import re
import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Preformatted, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', nargs='?', type=Path, default=ROOT / 'docs/QWEN_PLAYGROUND_GUIDE.md')
parser.add_argument('--title', default='Qwen3.5-0.8B Playground — Tools and Experiments')
parser.add_argument('--footer', default='AI Pilot | Qwen3.5-0.8B playground | 11 October 2026')
parser.add_argument('--cover', type=Path, help='Optional branded cover metadata JSON')
args = parser.parse_args()
SOURCE = args.source.resolve()
fontdir = Path('/usr/share/fonts/truetype/dejavu')
pdfmetrics.registerFont(TTFont('Guide', str(fontdir / 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('Guide-Bold', str(fontdir / 'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFont(TTFont('Guide-Mono', str(fontdir / 'DejaVuSansMono.ttf')))
pdfmetrics.registerFontFamily('Guide', normal='Guide', bold='Guide-Bold', italic='Guide', boldItalic='Guide-Bold')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle('Body', fontName='Guide', fontSize=9, leading=13, spaceAfter=7))
styles.add(ParagraphStyle('H1', fontName='Guide-Bold', fontSize=22, leading=27, spaceAfter=16, textColor=colors.HexColor('#133c55')))
styles.add(ParagraphStyle('H2', fontName='Guide-Bold', fontSize=12, leading=16, spaceBefore=9, spaceAfter=8, keepWithNext=True, textColor=colors.HexColor('#133c55')))
styles.add(ParagraphStyle('GuideCode', fontName='Guide-Mono', fontSize=8, leading=12, spaceAfter=9, backColor=colors.HexColor('#edf3f7'), borderPadding=7))
styles.add(ParagraphStyle('Caption', parent=styles['Body'], fontSize=8, textColor=colors.HexColor('#52606d')))

def inline(text):
    text = html.escape(text)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<link href="\2" color="#176b98">\1</link>', text)
    text = re.sub(r'\[([^\]]+)\]\((?!https?://)[^)]+\)', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    return re.sub(r'`([^`]+)`', r'<font name="Guide-Mono" size="8">\1</font>', text)

story, paragraph, code, table = [], [], None, []
cover = json.loads(args.cover.read_text()) if args.cover else None
if cover:
    story.extend([Spacer(1, 1), PageBreak()])
def flush():
    if paragraph:
        story.append(Paragraph(inline(' '.join(paragraph)), styles['Body']))
        paragraph.clear()

def flush_table():
    if not table:
        return
    rows = [[Paragraph(inline(cell), styles['Body']) for cell in row] for row in table]
    rendered = Table(rows, colWidths=[170 * mm / len(rows[0])] * len(rows[0]), repeatRows=1, hAlign='LEFT')
    rendered.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e4eef5')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LINEBELOW', (0, 0), (-1, -1), .4, colors.HexColor('#c4d0da')),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.extend([rendered, Spacer(1, 8)])
    table.clear()

for line in SOURCE.read_text().splitlines():
    if code is None and line.startswith('|'):
        flush()
        cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
        if not all(re.fullmatch(r':?-+:?', cell) for cell in cells):
            table.append(cells)
        continue
    flush_table()
    if line.startswith('```'):
        flush()
        if code is None:
            code = []
        else:
            story.append(Preformatted('\n'.join(code), styles['GuideCode']))
            code = None
    elif code is not None:
        code.append(line)
    elif line == '<!-- pagebreak -->':
        flush(); story.append(PageBreak())
    elif line.startswith('# '):
        flush(); story.append(Paragraph(inline(line[2:]), styles['H1']))
    elif line.startswith('## '):
        flush(); story.append(Paragraph(inline(line[3:]), styles['H2']))
    elif line.startswith('### '):
        flush(); story.append(Paragraph(inline(line[4:]), styles['H2']))
    elif line.startswith('- '):
        flush(); story.append(Paragraph(inline(line[2:]), styles['Body'], bulletText='•'))
    elif line.startswith('!['):
        flush()
        match = re.fullmatch(r'!\[(.*?)\]\((.*?)\)', line)
        img = Image(str(SOURCE.parent / match[2]))
        img._restrictSize(160 * mm, 100 * mm)
        story.extend([img, Paragraph(inline(match[1]), styles['Caption'])])
    elif not line.strip():
        flush()
    else:
        paragraph.append(line.strip())
flush()
flush_table()

def footer(canvas, document):
    canvas.setFont('Guide', 8)
    canvas.setFillColor(colors.HexColor('#52606d'))
    canvas.drawString(20 * mm, 12 * mm, args.footer)
    canvas.drawRightString(190 * mm, 12 * mm, str(document.page))

def first_page(canvas, document):
    if not cover:
        footer(canvas, document)
        return
    canvas.saveState()
    navy = colors.HexColor('#17324d')
    muted = colors.HexColor('#586675')
    line = colors.HexColor('#d9e1e8')
    x, width = 20 * mm, 170 * mm
    canvas.setFillColor(navy); canvas.setFont('Guide-Bold', 22)
    canvas.drawString(x, 262 * mm, cover.get('brand', 'AI Pilot'))
    canvas.setFillColor(muted); canvas.setFont('Guide', 9.5)
    canvas.drawString(x, 253 * mm, cover['subtitle'].upper())
    canvas.setStrokeColor(line); canvas.line(x, 248 * mm, x + width, 248 * mm)
    canvas.setStrokeColor(colors.HexColor('#176b87')); canvas.setLineWidth(4)
    canvas.line(x, 152 * mm, x, 216 * mm)
    canvas.setFillColor(colors.HexColor('#0f766e')); canvas.setFont('Guide-Bold', 9)
    canvas.drawString(x + 8 * mm, 211 * mm, cover['kicker'].upper())
    title_style = ParagraphStyle('CoverTitle', fontName='Guide-Bold', fontSize=30,
                                 leading=34, textColor=navy)
    title = Paragraph(html.escape(cover['title']).replace('\n', '<br/>'), title_style)
    _, height = title.wrap(155 * mm, 60 * mm)
    title.drawOn(canvas, x + 8 * mm, 202 * mm - height)
    lead = Paragraph(html.escape(cover['lead']), ParagraphStyle('CoverLead',
                     fontName='Guide', fontSize=12, leading=17, textColor=muted))
    _, lead_height = lead.wrap(150 * mm, 35 * mm)
    lead.drawOn(canvas, x + 8 * mm, 196 * mm - height - lead_height)
    top, row_height = 105 * mm, 20 * mm
    canvas.setStrokeColor(navy); canvas.setLineWidth(1.2)
    canvas.line(x, top, x + width, top)
    for i, (label, value) in enumerate(cover['metadata']):
        left = x + (i % 2) * width / 2
        y = top - (i // 2) * row_height
        canvas.setFillColor(muted); canvas.setFont('Guide-Bold', 7.8)
        canvas.drawString(left + 2 * mm, y - 6 * mm, label.upper())
        canvas.setFillColor(navy); canvas.setFont('Guide', 9)
        canvas.drawString(left + 2 * mm, y - 13 * mm, value)
        canvas.setStrokeColor(line); canvas.setLineWidth(.5)
        canvas.line(left, y - row_height, left + width / 2, y - row_height)
    canvas.line(x + width / 2, top, x + width / 2, top - 3 * row_height)
    canvas.restoreState()
    footer(canvas, document)

destination = SOURCE.with_suffix('.pdf')
document = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=20*mm, leftMargin=20*mm,
                            topMargin=16*mm, bottomMargin=20*mm,
                            title=args.title, author='AI Pilot')
document.build(story, onFirstPage=first_page, onLaterPages=footer)
print(destination)
