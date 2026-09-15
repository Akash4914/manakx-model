"""
Structured attribute extraction.

This reproduces the exact example from the spec doc:

    Input:  "Outdoor LED street light, 120W, IP66, highway installation"
    Output: Product     -> LED Luminaire
            Application -> Street/Highway lighting
            Environment -> Outdoor
            Protection  -> IP66
            Power       -> 120W

Single-value fields (product, application, environment, material,
protection, power) return the single best match, since a spec normally
describes one product. Multi-value fields (performance, electrical or
mechanical properties, safety requirements, testing requirements) return
every distinct match found, since a spec can list several of these.

This is rule/lexicon-based rather than a black-box classifier for the
same reason as the rest of this package: it is explainable ("why did the
AI say Product = LED Luminaire?" -> this pattern matched), needs no model
download, and is trivial for a teammate to extend by adding a pattern —
no retraining required. Swap in an actual NLP/NER model later without
touching any other module; only the return shape (`ExtractedAttributes`)
matters to the rest of the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# (regex, canonical output value) — first match wins for single-value fields.
_PRODUCT_PATTERNS: List[Tuple[str, str]] = [
    (r"led.{0,15}(street|highway|road).{0,15}light", "LED Luminaire"),
    (r"\bled\b.{0,10}\blight", "LED Luminaire"),
    (r"\blight.{0,10}\bled\b", "LED Luminaire"),
    (r"water\s*storage\s*tank|water\s*tank", "Water Storage Tank"),
    (r"safety\s*helmet|protective\s*headgear|hard\s*hat", "Industrial Safety Helmet"),
    (r"centrifugal\s+water\s+pump", "Centrifugal Water Pump"),
    (r"centrifugal.{0,20}pump", "Centrifugal Pump"),
    (r"\bwater\s*pump\b", "Water Pump"),
    (r"school\s*desk|student\s*desk", "School Desk"),
    (r"\bdesk\b", "Desk"),
]

_APPLICATION_PATTERNS: List[Tuple[str, str]] = [
    (r"highway", "Street/Highway Lighting"),
    (r"street\s*light|road\s*light", "Street/Highway Lighting"),
    (r"residential", "Residential Use"),
    (r"\bindustrial\b", "Industrial Use"),
    (r"construction\s*(site|worker)?", "Construction Site Use"),
    (r"school|classroom|student", "Educational/School Use"),
    (r"irrigation", "Irrigation"),
    (r"drinking\s*water|potable", "Potable Water Supply"),
    (r"hospital", "Healthcare Facility Use"),
]

_ENVIRONMENT_PATTERNS: List[Tuple[str, str]] = [
    (r"\boutdoor\b", "Outdoor"),
    (r"\bindoor\b", "Indoor"),
]

_MATERIAL_PATTERNS: List[Tuple[str, str]] = [
    (r"polyethylene", "Polyethylene"),
    (r"\bplastic\b", "Plastic"),
    (r"\bsteel\b", "Steel"),
    (r"aluminium|aluminum", "Aluminium"),
    (r"\bwood(en)?\b", "Wood"),
    (r"\bconcrete\b", "Concrete"),
]

# Multi-value fields: every distinct match is kept.
_PERFORMANCE_PATTERNS: List[Tuple[str, str]] = [
    (r"\bstrong\b|\bstrength\b", "High strength"),
    (r"\bdurable\b|\bdurability\b", "Durability"),
    (r"efficien\w*", "Energy efficiency"),
    (r"\blife\s*span|\blong\s*life", "Long service life"),
    (r"\bload\s*bearing|\bload\b", "Load-bearing capacity"),
    (r"uv\s*resist\w*|uv\s*stabili[sz]", "UV resistance"),
    (r"leak\s*proof|water\s*proof", "Leak/water proofing"),
    (r"corrosion\s*resist\w*", "Corrosion resistance"),
]

_ELECTRICAL_MECHANICAL_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(\d+(?:\.\d+)?)\s?v\b", "Voltage: {0}V"),
    (r"\b(\d+(?:\.\d+)?)\s?hz\b", "Frequency: {0}Hz"),
    (r"\b(\d+(?:\.\d+)?)\s?a\b(?!ny)", "Current: {0}A"),
    (r"\bsingle\s*phase\b", "Single phase"),
    (r"\bthree\s*phase\b", "Three phase"),
]

_SAFETY_PATTERNS: List[Tuple[str, str]] = [
    (r"\bsafety\b|\bsafe\b", "General safety compliance"),
    (r"\bhazard\w*", "Hazard mitigation"),
    (r"fire\s*resist\w*|fire\s*retard\w*", "Fire resistance"),
    (r"\bshock\b", "Shock/electrical safety"),
    (r"smooth\s*edge", "Edge/impact safety"),
    (r"\bchild\b|\bstudent\s*safety\b", "Child/student safety"),
]

_TESTING_PATTERNS: List[Tuple[str, str]] = [
    (r"\btest(ing|ed)?\b", "Testing required"),
    (r"certificat\w*", "Certification required"),
    (r"\bcompliance\b", "Compliance verification"),
    (r"\binspection\b", "Inspection required"),
]

_PROTECTION_RE = re.compile(r"\bIP\s?(\d{2})\b", re.IGNORECASE)
_POWER_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s?(k?w)\b", re.IGNORECASE)
_CAPACITY_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s?(ml|l|litre|liter)\b", re.IGNORECASE)


def _first_match(text: str, patterns: List[Tuple[str, str]]) -> Optional[str]:
    for pattern, value in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return value
    return None


def _all_matches(text: str, patterns: List[Tuple[str, str]]) -> List[str]:
    found: List[str] = []
    for pattern, value in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            out = value.format(*m.groups()) if "{0}" in value else value
            if out not in found:
                found.append(out)
    return found


@dataclass
class ExtractedAttributes:
    product: Optional[str] = None
    application: Optional[str] = None
    environment: Optional[str] = None
    material: Optional[str] = None
    protection: Optional[str] = None
    power: Optional[str] = None
    capacity: Optional[str] = None
    performance: List[str] = field(default_factory=list)
    electrical_mechanical: List[str] = field(default_factory=list)
    safety: List[str] = field(default_factory=list)
    testing: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "product": self.product,
            "application": self.application,
            "environment": self.environment,
            "material": self.material,
            "protection": self.protection,
            "power": self.power,
            "capacity": self.capacity,
            "performance": self.performance,
            "electrical_mechanical": self.electrical_mechanical,
            "safety": self.safety,
            "testing": self.testing,
        }


def extract_attributes(text: str) -> ExtractedAttributes:
    protection_match = _PROTECTION_RE.search(text)
    power_match = _POWER_RE.search(text)
    capacity_match = _CAPACITY_RE.search(text)

    return ExtractedAttributes(
        product=_first_match(text, _PRODUCT_PATTERNS),
        application=_first_match(text, _APPLICATION_PATTERNS),
        environment=_first_match(text, _ENVIRONMENT_PATTERNS),
        material=_first_match(text, _MATERIAL_PATTERNS),
        protection=f"IP{protection_match.group(1)}" if protection_match else None,
        power=(
            f"{power_match.group(1)}{power_match.group(2).upper()}"
            if power_match
            else None
        ),
        capacity=(
            f"{capacity_match.group(1)}{capacity_match.group(2).upper()}"
            if capacity_match
            else None
        ),
        performance=_all_matches(text, _PERFORMANCE_PATTERNS),
        electrical_mechanical=_all_matches(text, _ELECTRICAL_MECHANICAL_PATTERNS),
        safety=_all_matches(text, _SAFETY_PATTERNS),
        testing=_all_matches(text, _TESTING_PATTERNS),
    )
