from pathlib import Path

from reportlab.pdfgen import canvas


def make_blank_pdf(path: Path) -> Path:
    c = canvas.Canvas(str(path))
    c.showPage()
    c.save()
    return path


def make_corrupted_pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4\nthis is not a valid pdf body, truncated")
    return path
