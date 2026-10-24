"""Generate a synthetic (fictional) health-policy PDF that exercises the hard ingestion cases:
definitions section, a benefit table with a two-row header that spans two pages, a full-width
row-group label, footnote markers, a chart image, and an image-only (scanned) page.

Run:  python tests/fixtures/make_sample_policy.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import LongTable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, TableStyle

OUT = Path(__file__).with_name("sample_policy.pdf")
styles = getSampleStyleSheet()
H1, H2, BODY = styles["Heading1"], styles["Heading2"], styles["BodyText"]


def premium_chart_png() -> bytes:
    bands = [("18-35", 6200), ("36-45", 8900), ("46-55", 13400), ("56-65", 21800)]
    img = Image.new("RGB", (900, 560), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=22)
    d.text((40, 20), "Annual premium (Gold plan, Rs. 10 lakh sum insured) by age band", fill="black", font=font)
    base, scale = 500, 380 / 21800
    for i, (band, amount) in enumerate(bands):
        x = 110 + i * 190
        h = int(amount * scale)
        d.rectangle([x, base - h, x + 110, base], fill=(46, 104, 170))
        d.text((x + 10, base - h - 30), f"Rs. {amount:,}", fill="black", font=font)
        d.text((x + 25, base + 12), band, fill="black", font=font)
    d.line([80, base, 860, base], fill="black", width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def benefit_table() -> LongTable:
    rows = [
        ["Benefit", "Plan Variant", ""],
        ["", "Silver", "Gold"],
        ["In-patient Benefits", "In-patient Benefits", "In-patient Benefits"],
        ["Room Rent (per day)", "1% of Sum Insured*", "Single private room"],
        ["ICU Charges (per day)", "2% of Sum Insured*", "No limit"],
        ["Pre-hospitalisation", "30 days", "60 days"],
        ["Post-hospitalisation", "60 days", "180 days"],
        ["Day Care Procedures", "Covered", "Covered"],
        ["Road Ambulance", "Up to Rs. 2,000 per hospitalisation", "Up to Rs. 5,000 per hospitalisation"],
        ["AYUSH Treatment", "Up to 25% of Sum Insured", "Up to Sum Insured"],
        ["Cataract Surgery", "Rs. 40,000 per eye#", "Rs. 75,000 per eye#"],
        ["Modern Treatments", "50% of Sum Insured", "Up to Sum Insured"],
        ["Organ Donor Expenses", "Up to Rs. 1 lakh", "Up to Sum Insured"],
        ["Domiciliary Hospitalisation", "Not covered", "Up to 10% of Sum Insured"],
        ["Other Benefits", "Other Benefits", "Other Benefits"],
        ["Maternity Benefit", "Not covered", "Rs. 50,000 (normal), Rs. 75,000 (caesarean)#"],
        ["New Born Baby Cover", "Not covered", "Up to Rs. 50,000"],
        ["Health Check-up", "Once every 2 years", "Once every year"],
        ["Cumulative Bonus", "10% per claim-free year, max 50%", "25% per claim-free year, max 100%"],
        ["Co-payment", "20% for insured aged 61 and above", "No co-payment"],
        ["Restoration of Sum Insured", "Not available", "100% once per policy year"],
        ["Second Medical Opinion", "Not covered", "Covered"],
        ["Mental Illness Treatment", "Up to Rs. 50,000", "Up to Sum Insured"],
        ["Bariatric Surgery", "Not covered", "Up to Rs. 2 lakh"],
        ["Daily Hospital Cash", "Not covered", "Rs. 1,000 per day, max 30 days"],
        ["Air Ambulance", "Not covered", "Up to Rs. 2.5 lakh"],
        ["Emergency Treatment Abroad", "Not covered", "Up to Rs. 10 lakh"],
        ["OPD Consultations", "Not covered", "Up to Rs. 5,000 per year"],
        ["Dental Treatment (accidental)", "Covered", "Covered"],
        ["Home Care Treatment", "Not covered", "Up to Rs. 1 lakh"],
        ["Wellness Discount", "Up to 5% on renewal", "Up to 10% on renewal"],
        ["Pharmacy Discount", "Not available", "10% at network pharmacies"],
        ["Teleconsultation", "Unlimited", "Unlimited"],
        ["Vaccination Cover", "Not covered", "Up to Rs. 5,000"],
        ["Hearing Aid", "Not covered", "Up to Rs. 25,000 once in 3 years"],
        ["Spectacles", "Not covered", "Up to Rs. 5,000 once in 2 years"],
        ["Physiotherapy (post-surgery)", "Up to 10 sessions", "Up to 30 sessions"],
        ["Organ Transplant Recipient", "Covered", "Covered"],
        ["Convalescence Benefit", "Not covered", "Rs. 10,000 if hospitalised 10+ days"],
        ["Compassionate Visit", "Not covered", "Economy airfare up to Rs. 25,000"],
    ]
    style = TableStyle([
        ("SPAN", (1, 0), (2, 0)),
        ("SPAN", (0, 0), (0, 1)),
        ("SPAN", (0, 2), (2, 2)),
        ("SPAN", (0, 14), (2, 14)),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 1), colors.lightgrey),
        ("BACKGROUND", (0, 2), (-1, 2), colors.whitesmoke),
        ("BACKGROUND", (0, 14), (-1, 14), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ])
    return LongTable(rows, colWidths=[5.5 * cm, 5.5 * cm, 6.0 * cm], repeatRows=2, style=style)


def build_text_pdf() -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="Sample Health Shield Policy Wording")
    s: list = [
        Paragraph("Sample Health Shield - Policy Wording", styles["Title"]),
        Paragraph("Fictional policy wording generated for software testing only.", BODY),
        Paragraph("2. Definitions", H1),
        Paragraph("2.1 Hospital means any institution established for in-patient care and day care treatment of illness and/or injuries which has at least 10 in-patient beds and is registered with the local authorities.", BODY),
        Paragraph("2.2 Pre-existing Disease means any condition, ailment, injury or disease that is diagnosed by a physician within 36 months prior to the date of commencement of the policy.", BODY),
        Paragraph("2.3 Room Rent means the amount charged by a Hospital towards room and boarding expenses and shall include associated medical expenses.", BODY),
        Paragraph("2.4 Co-payment means a cost sharing requirement under a health insurance policy that provides that the policyholder will bear a specified percentage of the admissible claims amount.", BODY),
        Paragraph("3. Schedule of Benefits", H1),
        Paragraph("The following benefits are available under each plan variant, subject to the terms, conditions and exclusions of this policy.", BODY),
        Spacer(1, 6),
        benefit_table(),
        Paragraph("* Proportionate deduction applies to associated medical expenses if the insured opts for a room or ICU above the eligible limit.", BODY),
        Paragraph("# Subject to a waiting period of 24 months from the first policy inception.", BODY),
        Paragraph("4. Waiting Periods", H1),
        Paragraph("4.1 Initial Waiting Period: Expenses related to the treatment of any illness within 30 days from the first policy commencement date shall be excluded, except claims arising due to an accident.", BODY),
        Paragraph("4.2 Pre-existing Diseases: Expenses related to the treatment of a Pre-existing Disease and its direct complications shall be excluded until the expiry of 36 months of continuous coverage for the Silver plan and 24 months for the Gold plan.", BODY),
        Paragraph("4.3 Specific Diseases: Treatment of cataract, hernia, joint replacement and benign prostatic hypertrophy is covered only after 24 months of continuous coverage.", BODY),
        Paragraph("5. Permanent Exclusions", H1),
        Paragraph("5.1 Cosmetic or plastic surgery unless necessary for reconstruction following an accident, burns or cancer.", BODY),
        Paragraph("5.2 Treatment for obesity or weight control, except bariatric surgery under the Gold plan as listed in the Schedule of Benefits.", BODY),
        Paragraph("5.3 Expenses arising from hazardous or adventure sports, war, or nuclear contamination are not covered.", BODY),
        PageBreak(),
        Paragraph("6. Premium Illustration", H1),
        Paragraph("The chart below illustrates indicative annual premiums for the Gold plan.", BODY),
        RLImage(io.BytesIO(premium_chart_png()), width=15 * cm, height=9.3 * cm),
        Paragraph("Figure 1: Indicative Gold plan premium by age band (excluding GST).", BODY),
        PageBreak(),
        Paragraph("7. Claims Procedure", H1),
        Paragraph("7.1 For cashless claims, the insured must intimate the TPA at least 48 hours before a planned hospitalisation and within 24 hours of an emergency hospitalisation.", BODY),
        Paragraph("7.2 For reimbursement claims, all documents must be submitted within 30 days of discharge from the Hospital.", BODY),
    ]
    doc.build(s)
    return buf.getvalue()


def make() -> Path:
    src = pymupdf.open(stream=build_text_pdf(), filetype="pdf")
    out = pymupdf.open()
    last = src.page_count - 1
    for i, page in enumerate(src):
        if i == last:
            # Simulate a scanned page: rasterise the claims page so it has no text layer.
            pix = page.get_pixmap(dpi=200)
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=pix.tobytes("png"))
        else:
            out.insert_pdf(src, from_page=i, to_page=i)
    out.save(OUT)
    return OUT


if __name__ == "__main__":
    print(make())
