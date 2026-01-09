"""
Convert CPPS Final Report Markdown to PDF
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
import re
import os

def create_pdf():
    # Read the markdown file
    input_file = 'CPPS_Final_Report.md'
    output_file = 'CPPS_Final_Report.pdf'

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Create PDF with proper margins
    doc = SimpleDocTemplate(
        output_file,
        pagesize=A4,
        rightMargin=2.5*cm,
        leftMargin=2.5*cm,
        topMargin=2.5*cm,
        bottomMargin=2.5*cm
    )

    elements = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=20,
        spaceBefore=60,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.HexColor('#444444'),
        spaceAfter=8,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )

    meta_style = ParagraphStyle(
        'Meta',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#333333'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )

    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold'
    )

    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#333333'),
        spaceAfter=8,
        spaceBefore=14,
        fontName='Helvetica-Bold'
    )

    h3_style = ParagraphStyle(
        'H3',
        parent=styles['Heading3'],
        fontSize=11,
        textColor=colors.HexColor('#444444'),
        spaceAfter=6,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        fontName='Helvetica',
        leading=14
    )

    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=20,
        spaceAfter=4
    )

    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#666666'),
        spaceAfter=12,
        spaceBefore=4,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )

    # Parse content
    lines = content.split('\n')
    in_title_section = True

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Main title
        if stripped.startswith('# ') and not stripped.startswith('## '):
            title_text = stripped[2:].strip()
            elements.append(Spacer(1, 1*inch))
            elements.append(Paragraph(title_text, title_style))

        # Subtitle (## A Cyber-Physical...)
        elif in_title_section and stripped.startswith('## '):
            subtitle_text = stripped[3:].strip()
            elements.append(Paragraph(subtitle_text, subtitle_style))
            elements.append(Spacer(1, 0.5*inch))

        # Meta info on title page
        elif in_title_section and stripped.startswith('**') and ':**' in stripped:
            meta_text = stripped.replace('**', '')
            elements.append(Paragraph(meta_text, meta_style))

        # First --- ends title section and creates page break
        elif stripped == '---':
            if in_title_section:
                elements.append(Spacer(1, 0.3*inch))
                in_title_section = False
            elements.append(PageBreak())

        # Section headers (## )
        elif stripped.startswith('## ') and not in_title_section:
            header_text = stripped[3:].strip()
            elements.append(Paragraph(header_text, h1_style))

        # Subsection headers (### )
        elif stripped.startswith('### '):
            header_text = stripped[4:].strip()
            elements.append(Paragraph(header_text, h2_style))

        # Sub-subsection headers (#### )
        elif stripped.startswith('#### '):
            header_text = stripped[5:].strip()
            elements.append(Paragraph(header_text, h3_style))

        # Screenshot placeholders
        elif stripped.startswith('[INSERT SCREENSHOT:'):
            caption = stripped.replace('[INSERT SCREENSHOT:', '').replace(']', '').strip()
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph(f"[Figure: {caption}]", caption_style))
            elements.append(Spacer(1, 0.2*inch))

        # Bullet points
        elif stripped.startswith('- ') or stripped.startswith('* '):
            bullet_text = stripped[2:].strip()
            bullet_text = format_text(bullet_text)
            elements.append(Paragraph(f'• {bullet_text}', bullet_style))

        # Numbered lists
        elif re.match(r'^\d+\.', stripped):
            list_text = format_text(stripped)
            elements.append(Paragraph(list_text, bullet_style))

        # Regular paragraphs
        elif stripped and not stripped.startswith('#') and not stripped.startswith('```') and not stripped.startswith('|'):
            para_text = format_text(stripped)
            elements.append(Paragraph(para_text, body_style))

        # Empty lines
        elif not stripped:
            elements.append(Spacer(1, 0.08*inch))

        i += 1

    # Build PDF
    doc.build(elements)
    print(f"PDF created successfully: {output_file}")
    print(f"Location: {os.path.abspath(output_file)}")

def format_text(text):
    """Convert markdown formatting to reportlab tags"""
    # Bold
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Italic
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # Remove markdown links, keep text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    return text

if __name__ == "__main__":
    create_pdf()
