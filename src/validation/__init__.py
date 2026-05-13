"""Scientific validation submodule.

Public API mirrors the rule canon documented in
``scientific_validator.py``.  Consumers should import from this
submodule rather than reaching into individual files:

    from mess_methods.validation import (
        ScientificValidator,
        Observation,
        PaperContext,
        Violation,
    )

The :class:`ScientificValidator` is the recommended entry point.  The
individual ``validate_*`` functions are exported for callers that want
to run a single rule in isolation (e.g. inline at extraction time).
"""

from .scientific_validator import (
    CITATION_LOGAN_2008_CH2,
    CITATION_LOGAN_2008_CH5,
    CITATION_LOGAN_2008_CH9,
    CITATION_LOGAN_HAMELERS_2006,
    CITATION_NEWMAN_TA_CH22,
    DUPLICATE_INCLUDED_SLUGS,
    F_FARADAY,
    H2_MOLAR_VOLUME_L,
    OCV_CEILING_V,
    POSITIVE_REQUIRED_SLUGS,
    REMOVAL_FAMILY_SLUGS,
    ScientificValidator,
    validate_ce_bounds,
    validate_ce_unit_interval,
    validate_faraday_h2_ceiling,
    validate_max_power_ohm,
    validate_non_positive,
    validate_ocv_thermodynamic_ceiling,
    validate_power_identity,
    validate_removal_unit_interval,
    validate_temperature_range,
    validate_voltage_ordering,
    validate_within_paper_duplicate,
)
from .types import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MED,
    CONFIDENCE_NEEDS_REVIEW,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MED,
    Observation,
    PaperContext,
    Violation,
    ViolationOrNone,
)

__all__ = [
    "ScientificValidator",
    "validate_power_identity",
    "validate_ce_bounds",
    "validate_ce_unit_interval",
    "validate_voltage_ordering",
    "validate_max_power_ohm",
    "validate_temperature_range",
    "validate_removal_unit_interval",
    "validate_non_positive",
    "validate_ocv_thermodynamic_ceiling",
    "validate_within_paper_duplicate",
    "validate_faraday_h2_ceiling",
    "Observation",
    "PaperContext",
    "Violation",
    "ViolationOrNone",
    "SEVERITY_HIGH",
    "SEVERITY_MED",
    "SEVERITY_LOW",
    "CONFIDENCE_HIGH",
    "CONFIDENCE_MED",
    "CONFIDENCE_LOW",
    "CONFIDENCE_NEEDS_REVIEW",
    "OCV_CEILING_V",
    "DUPLICATE_INCLUDED_SLUGS",
    "REMOVAL_FAMILY_SLUGS",
    "POSITIVE_REQUIRED_SLUGS",
    "F_FARADAY",
    "H2_MOLAR_VOLUME_L",
    "CITATION_LOGAN_HAMELERS_2006",
    "CITATION_LOGAN_2008_CH2",
    "CITATION_LOGAN_2008_CH5",
    "CITATION_LOGAN_2008_CH9",
    "CITATION_NEWMAN_TA_CH22",
]
