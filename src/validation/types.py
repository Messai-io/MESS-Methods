"""
Types for the MESS-Methods scientific validator.

These dataclasses are the public boundary between consumers (extractors,
hunter pipelines, external analysis code) and the rule canon in
``scientific_validator.py``.

A consumer constructs ``Observation`` objects (one per reported value
from a paper) plus an optional ``PaperContext`` (system class, materials,
microbes, operating conditions) and passes them to
``ScientificValidator.validate()``.  Each rule emits zero or more
``Violation`` objects which mirror the structure used by the hunter
``computationTrace`` payload, so callers can render the same kind of
chain-of-thought card.

References
----------
Logan & Hamelers et al. (2006) "Microbial Fuel Cells: Methodology and
Technology", Environ. Sci. Technol. 40(17), §3 — reporting conventions
(P = V·I, areal vs volumetric, V_cell vs V_oc).

Logan, B.E. (2008) "Microbial Fuel Cells", Wiley — Ch. 2 energy balance
and OCV thermodynamic ceilings; Ch. 5 Coulombic efficiency definition;
Ch. 9 hydrogen production (MEC) Faraday-law bound.

Newman & Thomas-Alyea (2004) "Electrochemical Systems", 3rd ed., Wiley —
Ch. 22 porous-electrode conventions, internal resistance / Thevenin
matched-load identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────
# Severity levels (mirrors the hunter pipeline so JSON payloads are
# interchangeable).
# ─────────────────────────────────────────────────────────────────────

# Severity = the *headline* importance of the finding (does the published
# claim survive?). HIGH = published claim is wrong; MED = published claim
# is suspect; LOW = a small inconsistency, likely cosmetic.
SEVERITY_HIGH = "HIGH"
SEVERITY_MED = "MED"
SEVERITY_LOW = "LOW"

# Confidence = how sure we are about the *finding itself* (did we have
# the right inputs to apply the rule cleanly?). Useful when downstream
# consumers want to filter to clean evidence vs. flag-for-review.
CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_MED = "MED"
CONFIDENCE_LOW = "LOW"
CONFIDENCE_NEEDS_REVIEW = "NEEDS_REVIEW"


@dataclass
class Observation:
    """One reported numeric value from a paper.

    The minimum useful payload is ``canonical_slug``, ``raw_value`` /
    ``si_value`` and ``unit``.  The provenance fields (``paper_id``,
    ``condition_set_id``, ``source_epd_id``) are recommended so
    violations can drill back into the source extraction.

    Attributes
    ----------
    paper_id : str
        Stable identifier for the paper. Used to group observations by
        publication for within-paper rules and for paper-level reports.
    condition_set_id : str | None
        Group identifier for observations recorded at the same operating
        point (same substrate, temperature, electrode pair, …).  Used by
        rules that compare V, I, P, OCV, R_int from one matched set
        (``power_identity``, ``voltage_ordering``, ``max_power_ohm``,
        ``within_paper_duplicate``).  ``None`` is acceptable but disables
        those rules for this observation.
    canonical_slug : str
        Normalised parameter slug from ``mess-parameters``.  Examples:
        ``powerDensity``, ``currentDensity``, ``coulombic_efficiency``,
        ``openCircuitVoltage``, ``cell_voltage``, ``internalResistance``,
        ``temperature``, ``specific_h2_production``,
        ``electrode_surface_area``, ``reactor_volume``.
    parameter_name : str | None
        Free-text label as it appears in the paper (e.g. "Peak Power
        Density").  Used to disambiguate when ``canonical_slug`` is
        ``voltage`` (could be OCV or operating V).
    raw_value : float | None
        Value as reported in the paper, in its native unit.  Rules that
        check unit-bound conditions (CE > 100%, removal > 1) consult
        this directly.
    si_value : float | None
        Value normalised to SI units (W/m² for power density, A/m² for
        current density, V for voltage, m³/m³/d for volumetric H2 rate,
        m² for electrode area, L for reactor volume — note: this
        matches the v2 ``normalize-to-si.py`` convention where
        ``reactor_volume`` stays in litres).  Rules that compose
        derived quantities (P = V·I, P_max = V_oc²/(4·R_int),
        Faraday-law bound) require SI values.
    unit : str | None
        Reported unit ("%", "mV", "V", "A/m²", "mW/m²" …).  Used by CE
        and removal rules to interpret the raw value's bound.
    source_epd_id : str | None
        Database identifier for the source row.  Echoed into the
        ``Violation.trace_inputs`` so a UI can link the card to the
        underlying extraction.
    snippet : str | None
        Short text snippet from the paper near the value.  Surfaced in
        the violation for human review.
    flags : list[str]
        Upstream consistency flags (``physics:power_identity:as_observed``
        etc.) attached by an earlier extractor pass.  Optional — the
        validator can rediscover the violations from raw values.

    Notes
    -----
    The dataclass intentionally accepts ``None`` for most fields so
    consumers can build partial ``Observation`` objects from messy
    upstream data.  Each rule defensively checks the fields it needs
    and returns ``None`` / NEEDS_REVIEW when inputs are missing rather
    than raising — making the validator safe to run on every row.
    """

    canonical_slug: str
    paper_id: Optional[str] = None
    condition_set_id: Optional[str] = None
    parameter_name: Optional[str] = None
    raw_value: Optional[float] = None
    si_value: Optional[float] = None
    unit: Optional[str] = None
    source_epd_id: Optional[str] = None
    snippet: Optional[str] = None
    flags: list[str] = field(default_factory=list)


@dataclass
class PaperContext:
    """Paper-level metadata used by rules that need it.

    ``system_class`` is the most important field — it parameterises the
    OCV thermodynamic ceiling and gates the Faraday-H₂ rule (MEC-only).

    Attributes
    ----------
    paper_id : str | None
        Same identifier as on the observations.
    system_class : str | None
        Coarse system taxonomy from the v2 extractor classifier.
        Accepted values mirror the hunter pipeline:
        ``"MFC"`` microbial fuel cell, ``"MEC"`` microbial electrolysis,
        ``"MDC"`` microbial desalination, ``"MES"`` microbial
        electrosynthesis, ``"MSC"`` microbial solar cell,
        ``"MBES"`` microbial bio-electrochemical system,
        ``"MRB"`` microbial reverse battery, ``"MERC"`` /
        ``"OTHER"`` fallback.
    primary_system_type : str | None
        Same field as the DB column ``ResearchPaper.primarySystemType``.
        ``system_class`` may be derived from this; if both are present
        consumers should prefer ``system_class``.
    materials : dict[str, Any]
        Free-form material dictionary (anode, cathode, membrane).  Not
        consumed by the current rule canon — reserved for future
        material-aware rules.
    microbes : list[str]
        Inoculum or pure-culture identifiers.  Reserved for future use.
    operating_conditions : dict[str, Any]
        pH, applied voltage (MEC), substrate, etc.  Reserved for future
        use.
    doi : str | None
        Useful for citation in human-readable violation messages.
    title : str | None
        Paper title, surfaced in reports.
    """

    paper_id: Optional[str] = None
    system_class: Optional[str] = None
    primary_system_type: Optional[str] = None
    materials: dict[str, Any] = field(default_factory=dict)
    microbes: list[str] = field(default_factory=list)
    operating_conditions: dict[str, Any] = field(default_factory=dict)
    doi: Optional[str] = None
    title: Optional[str] = None

    def effective_system_class(self) -> str:
        """Return the system class to use for system-class-gated rules,
        defaulting to ``"OTHER"`` when neither field is set."""
        return self.system_class or self.primary_system_type or "OTHER"


@dataclass
class Violation:
    """A single rule finding.

    The payload is shaped to mirror the hunter pipeline's
    ``computationTrace`` JSON so the same UI components that render the
    Next.js Hunter page can render validator output directly.

    Attributes
    ----------
    rule_name : str
        Canonical rule identifier (one of ``power_identity``,
        ``ce_bounds``, ``voltage_ordering``, ``max_power_ohm``,
        ``temperature_out_of_plausible_range``, ``non_positive``,
        ``removal_out_of_unit_interval``,
        ``ocv_thermodynamic_ceiling``, ``within_paper_duplicate``,
        ``faraday_h2_ceiling``).
    severity : str
        ``"HIGH" | "MED" | "LOW"``.
    confidence : str
        ``"HIGH" | "MED" | "LOW" | "NEEDS_REVIEW"``.
    plain_english : str
        Human-readable explanation suitable for a UI card or report.
    chain_of_thought : list[dict[str, Any]]
        Ordered list of computation steps.  Each step is
        ``{"text": str, "math"?: str, "value"?: float, "unit"?: str}``,
        matching ``_step()`` in the hunter pipeline.
    raw_values_used : list[dict[str, Any]]
        The reported (paper-native) values that fed the rule.  Each
        entry is ``{"name": str, "value": float, "unit": str}``.
    trace_inputs : dict[str, dict[str, Any]]
        SI-normalised inputs keyed by their canonical role
        (``"voltage"``, ``"currentDensity"``, …).  Each value includes
        ``value`` / ``unit`` / ``sourceEpdId`` / ``meta`` and matches
        ``computationTrace.inputs`` in the hunter JSON.
    predicted : float | None
        Value the rule predicts from inputs (e.g. V·I for
        ``power_identity``).
    observed : float | None
        Value reported in the paper.
    residual_pct : float | None
        ``(observed - predicted) / predicted`` when both are finite.
    confidence_reasons : list[str]
        Plain-English justifications for the chosen confidence band.
    citations : list[str]
        Literature references for the rule (Logan 2008 §, Hamelers
        2006 §, etc.).  Surfaced verbatim in reports so reviewers can
        check the canon.
    check_type : str
        Short symbolic form of the check ("P = V × I",
        "V_cell ≤ V_oc", …).  Useful as a card header.
    paper_id : str | None
        Echoed from the triggering observation.
    flagged_slug : str | None
        Canonical slug of the row that triggered the finding.
    """

    rule_name: str
    severity: str
    confidence: str
    plain_english: str
    chain_of_thought: list[dict[str, Any]] = field(default_factory=list)
    raw_values_used: list[dict[str, Any]] = field(default_factory=list)
    trace_inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    predicted: Optional[float] = None
    observed: Optional[float] = None
    residual_pct: Optional[float] = None
    confidence_reasons: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    check_type: str = ""
    paper_id: Optional[str] = None
    flagged_slug: Optional[str] = None

    def to_hunter_json(self) -> dict[str, Any]:
        """Render as the ``computationTrace`` payload shape used by the
        hunter pipeline / Next.js Hunter page.

        Returns a dict with the same field names as
        ``apps/web/public/data/computed/hunter/violations.json``'s
        ``items[*].computationTrace``.
        """
        return {
            "rule": self.rule_name,
            "checkType": self.check_type,
            "inputs": self.trace_inputs,
            "steps": self.chain_of_thought,
            "predicted": self.predicted,
            "observed": self.observed,
            "residualPct": self.residual_pct,
            "confidence": self.confidence,
            "confidenceReasons": self.confidence_reasons,
            "citations": self.citations,
        }


# Convenience: type aliases for callers using ``Optional`` patterns.
ViolationOrNone = Optional[Violation]
