"""
Certification recommendation.

The spec is explicit: "This information should preferably come from a
structured, verified rules database rather than letting an LLM invent
the requirement." So this reads straight off each Standard's
`certification` field (bis_mandatory, scheme) in the catalogue — no
generation involved, just a structured lookup keyed off whichever
standard the pipeline already matched as the main product standard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..retrieval.standards_repository import Standard


@dataclass
class CertificationRequirement:
    standard_id: str
    bis_mandatory: bool
    scheme: Optional[str]

    def to_dict(self) -> dict:
        return {
            "applicable_standard": self.standard_id,
            "bis_certification": "Mandatory" if self.bis_mandatory else "Not mandatory",
            "scheme": self.scheme,
        }


def get_certification_requirement(standard: Standard) -> CertificationRequirement:
    cert = standard.certification or {}
    return CertificationRequirement(
        standard_id=standard.id,
        bis_mandatory=bool(cert.get("bis_mandatory", False)),
        scheme=cert.get("scheme"),
    )
