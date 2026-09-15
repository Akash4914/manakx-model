"""
MANAKX Model — AI Indian Standards Assistant core.

Public entry point:

    from manakx_model import ManakxPipeline

    pipeline = ManakxPipeline()          # build once, reuse across requests
    result = pipeline.analyze("Outdoor LED street light, 120W, IP66, highway installation")
    tender_audit = pipeline.analyze_tender(pdf_path="tender.pdf")
    revision = pipeline.check_revision("IS 9137:2010")

See README.md for the full architecture and example_usage.py for a
runnable demo of every mode.
"""

from .pipeline import ManakxPipeline

__all__ = ["ManakxPipeline"]
__version__ = "0.2.0"
