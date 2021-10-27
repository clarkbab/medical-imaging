from fpdf import FPDF

from mymi import types

def generate_dicom_parameters_report(
    dataset: str,
    clear_cache: bool = False,
    report_name: str = None) -> None:
    # Create PDF.
    report = FPDF()
    report.set_font('Arial', 'B', 16)