"""
Runnable demo of the MANAKX model, using the spec document's own examples.

    python example_usage.py
"""

import json

from manakx_model import ManakxPipeline


def show(title: str, obj: dict):
    print(f"\n{'=' * 10} {title} {'=' * 10}")
    print(json.dumps(obj, indent=2))


if __name__ == "__main__":
    pipeline = ManakxPipeline()  # build once; reuse for every call below

    # 1. Product description -> AI attribute extraction + recommended standards
    #    (exact example from the spec doc)
    show(
        "Mode 1: Analyze a product description",
        pipeline.analyze(input_text="Outdoor LED street light, 120W, IP66, highway installation"),
    )

    # 2. Detailed spec -> same flow, longer input
    show(
        "Mode 1: Analyze a detailed spec",
        pipeline.analyze(
            input_text="500 L polyethylene water storage tank for outdoor residential use"
        ),
    )

    # 3. Tender document -> audit mode (detects outdated + missing standards)
    tender_text = """
    Tender for supply of Centrifugal Water Pump for municipal water supply.
    The pump shall conform to IS 9137:2010.
    """
    show("Mode 2: Tender document audit", pipeline.analyze_tender(input_text=tender_text))

    # 4. Direct revision lookup
    show("Mode 3: Direct revision check", pipeline.check_revision("IS 9137:2010"))
