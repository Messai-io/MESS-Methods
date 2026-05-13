"""
ScientificValidator — physics-violation rule canon for microbial
electrochemical systems papers.

This module is the **single source of truth** for the physics rules that
catch papers reporting physically-impossible values (P ≠ V·I, CE > 100%,
V_cell > V_oc, OCV above thermodynamic ceiling, Faraday-law-bounded H₂
rate exceeded, …).  It is imported by:

* the v2 extractor, for inline quality checks at extraction time
* the hunter pipeline (``scripts/hunter/build_hunter_jsons.py``), via
  re-export — the hunter previously hand-rolled these rules; future work
  is to delete the hand-rolled code and import from here
* external consumers via the public ``Messai-io/MESS-Methods`` mirror

The 10 rules implemented here mirror the hunter canon **byte-for-byte on
numerical bounds**.  Any drift between the hunter Python and this module
is a bug — there should be exactly one set of thresholds in the
codebase, and it lives in this file.

References
----------
Logan & Hamelers et al. (2006) "Microbial Fuel Cells: Methodology and
Technology", Environ. Sci. Technol. 40(17), §3 — reporting conventions
(P = V·I, areal vs volumetric, V_cell vs V_oc, Coulombic efficiency
definition).

Logan, B.E. (2008) "Microbial Fuel Cells", Wiley
    §2.3 — Thermodynamic OCV ceiling per system class (O₂ cathode at
    pH 7, typical anode redox couples).
    §5  — Coulombic efficiency definition and bound (0–1 or 0–100%).
    §9  — Microbial electrolysis cells; Faraday-law upper bound on H₂.

Newman & Thomas-Alyea (2004) "Electrochemical Systems", 3rd ed., Wiley
    Ch. 22 — Porous-electrode conventions, Thevenin matched-load
    P_max = V_oc² / (4·R_int).

Hunter pipeline reference: ``scripts/hunter/build_hunter_jsons.py``
(rev 2026-05-11, see ``docs/superpowers/specs/2026-05-11-hunter-next-handoff.md``).

License: MIT (see project root LICENSE).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Iterable, Optional

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

# ─────────────────────────────────────────────────────────────────────
# Citations — exposed on every violation so consumers can audit the
# rule canon against the published methodology literature.
# ─────────────────────────────────────────────────────────────────────

CITATION_LOGAN_HAMELERS_2006 = (
    "Logan, Hamelers, Rozendal et al. (2006) Environ. Sci. Technol. "
    "40(17) §3 — MFC reporting conventions."
)
CITATION_LOGAN_2008_CH2 = "Logan (2008) Microbial Fuel Cells, Wiley, §2.3 — Energy balance, OCV ceiling."
CITATION_LOGAN_2008_CH5 = "Logan (2008) Microbial Fuel Cells, Wiley, Ch. 5 — Coulombic efficiency definition."
CITATION_LOGAN_2008_CH9 = "Logan (2008) Microbial Fuel Cells, Wiley, Ch. 9 — MEC and Faraday-law H₂ ceiling."
CITATION_NEWMAN_TA_CH22 = "Newman & Thomas-Alyea (2004) Electrochemical Systems, 3e, Wiley, Ch. 22 — Thevenin matched-load."

# ─────────────────────────────────────────────────────────────────────
# Numeric constants (mirroring the hunter canon exactly)
# ─────────────────────────────────────────────────────────────────────

# Power-identity log-ratio bands (see hunter build_hunter_jsons.py).
# log10(2) ≈ 0.30  →  factor 2; log10(5) ≈ 0.70  →  factor 5;
# log10(10) = 1.00 → factor 10; log10(30) ≈ 1.48 → factor 30.
PI_PASS_LOG_RATIO = 0.30           # within ~2× → counts as match
PI_LOW_CONF_LOG_RATIO = 0.70       # 2–5× → keep, LOW confidence
PI_MED_CONF_LOG_RATIO = 1.00       # 5–10× → MED
PI_CROSS_COND_LOG_RATIO = 1.50     # >30× → downgrade to cross-cond artifact

# Per-system OCV thermodynamic ceiling. Values are an UPPER bound on the
# reversible cell potential under typical operating conditions, derived
# from standard redox couples in Logan 2008 §2.3 with ~20-30% slack.
#   MFC (acetate/O₂, pH 7):  1.1 V theoretical → 1.3 V ceiling
#   MSC (microbial solar):   1.2 V (photo-cathode can be slightly higher)
#   MDC (desalination):      1.1 V (similar redox to MFC)
#   MEC: applied bias system, OCV rarely meaningful → loose 1.5 V
#   MES: CO₂ reduction, varies widely with cathode → loose 1.5 V
#   OTHER/unknown:           1.5 V (very loose)
OCV_CEILING_V: dict[str, float] = {
    "MFC": 1.3,
    "MSC": 1.2,
    "MDC": 1.1,
    "MEC": 1.5,
    "MES": 1.5,
    "MBES": 1.3,
    "MRB": 1.3,
    "OTHER": 1.5,
}
# System classes the hunter pipeline explicitly skips for OCV-ceiling
# (the rule would always fire by construction on these).
OCV_CEILING_OUT_OF_SCOPE = {"not_BES", "REVIEW", "FUNDAMENTAL"}

# Within-paper duplicate allowlist. Restricted to slugs where wide
# variation within one paper is rarely legitimate (single-valued
# measurement summaries, not swept inputs).  See hunter notes:
#   "5× spread on a CE / OCV / R_int duplicate is much harder to
#    explain than 5× spread on substrate concentration."
DUPLICATE_INCLUDED_SLUGS: set[str] = {
    "coulombic_efficiency",
    "internalResistance",
    "openCircuitVoltage",
    "energy_efficiency",
}
DUPLICATE_MIN_SPREAD = 5.0           # max/min ratio that triggers the rule

# Faraday-law H₂ constants (Logan 2008 Ch. 9).
F_FARADAY = 96485.0                  # C / mol e⁻
H2_MOLAR_VOLUME_L = 22.414           # L / mol at STP (0 °C, 1 atm)
SECONDS_PER_DAY = 86400.0
FARADAY_PASS_RATIO = 1.3             # 30% slack before flagging

# Power-identity Thevenin matched-load slack.
MAX_POWER_OHM_PASS_RATIO = 1.3       # within 30% of V_oc²/(4·R_int)

# Temperature plausibility (BES is biological → 0–60 °C realistic;
# the hunter rule fires at >100 °C and <-20 °C).
TEMPERATURE_UPPER_C = 100.0
TEMPERATURE_LOWER_C = -20.0

# Voltage-ordering slack (V_cell allowed to be slightly above V_oc to
# tolerate measurement noise and reporting precision).
VOLTAGE_ORDERING_SLACK = 1.05        # V_cell ≤ V_oc × 1.05 → pass
VOLTAGE_ORDERING_UNIT_MIXUP_RATIO = 50.0  # >50× → probable mV/V mix-up


# ─────────────────────────────────────────────────────────────────────
# Slug helpers — match the hunter's intent-detection of OCV vs cell V
# ─────────────────────────────────────────────────────────────────────


def _looks_like_ocv(obs: Observation) -> bool:
    """True when an observation is open-circuit voltage (NOT to be used
    as a cell voltage in the V·I identity)."""
    if obs.canonical_slug == "openCircuitVoltage":
        return True
    name = (obs.parameter_name or "").lower()
    return "open" in name and ("circuit" in name or "ocv" in name)


def _looks_like_cell_voltage(obs: Observation) -> bool:
    """True when an observation is operating cell voltage at non-zero
    current.  ``voltage`` is accepted unless it smells like OCV."""
    if obs.canonical_slug in ("cell_voltage", "operating_voltage"):
        return True
    if obs.canonical_slug == "voltage":
        if _looks_like_ocv(obs):
            return False
        return True
    return False


def _step(
    text: str,
    math_str: Optional[str] = None,
    value: Optional[float] = None,
    unit: Optional[str] = None,
) -> dict[str, Any]:
    """Build one chain-of-thought step in the hunter payload shape."""
    step: dict[str, Any] = {"text": text}
    if math_str:
        step["math"] = math_str
    if value is not None and math.isfinite(value):
        step["value"] = float(value)
    if unit:
        step["unit"] = unit
    return step


def _raw_values(*obs: Observation) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for o in obs:
        if o is None or o.raw_value is None:
            continue
        out.append(
            {
                "name": o.parameter_name or (o.canonical_slug or ""),
                "value": float(o.raw_value),
                "unit": o.unit or "",
            }
        )
    return out


# ─────────────────────────────────────────────────────────────────────
# Rule 1: power_identity   P_observed ≈ V_cell · I
# ─────────────────────────────────────────────────────────────────────


def validate_power_identity(
    power: Observation,
    bucket: Iterable[Observation],
) -> ViolationOrNone:
    """Re-evaluate P = V · I with strict input typing.

    ``power`` is the powerDensity observation under test.  ``bucket`` is
    the set of observations sharing the same ``condition_set_id`` —
    the rule picks the best cell-voltage + current pair from the
    bucket.  When no cell voltage is present, an OCV-based upper bound
    is applied as a soft fallback (per the hunter strict-revalidator).

    References: Logan & Hamelers 2006 §3 (P = V·I reporting convention);
    Newman & Thomas-Alyea Ch. 22 (Thevenin upper bound).
    """
    if power.canonical_slug != "powerDensity":
        return None
    p_obs = power.si_value
    if p_obs is None or p_obs <= 0:
        return None

    bucket = list(bucket)
    voltages = [
        o for o in bucket
        if _looks_like_cell_voltage(o) and o.si_value is not None and o.si_value > 0
    ]
    ocvs = [
        o for o in bucket
        if _looks_like_ocv(o) and o.si_value is not None and o.si_value > 0
    ]
    currents = [
        o for o in bucket
        if o.canonical_slug == "currentDensity"
        and o.si_value is not None and o.si_value > 0
    ]

    citations = [CITATION_LOGAN_HAMELERS_2006, CITATION_NEWMAN_TA_CH22]

    if voltages and currents:
        # Best-pair search: minimise log-ratio against observed P.
        best = min(
            ((v, i) for v in voltages for i in currents),
            key=lambda vi: abs(math.log10(vi[0].si_value * vi[1].si_value) - math.log10(p_obs)),
        )
        v_obs_row, i_obs_row = best
        v_si, i_si = v_obs_row.si_value, i_obs_row.si_value
        predicted = v_si * i_si
        ratio = p_obs / predicted
        log_ratio = abs(math.log10(ratio))

        inputs = {
            "voltage": {
                "value": v_si, "unit": "V", "sourceEpdId": v_obs_row.source_epd_id,
                "meta": f"interpreted as cell voltage ({v_obs_row.canonical_slug or v_obs_row.parameter_name})",
            },
            "currentDensity": {
                "value": i_si, "unit": "A/m²", "sourceEpdId": i_obs_row.source_epd_id,
            },
            "powerDensity": {
                "value": p_obs, "unit": "W/m²", "sourceEpdId": power.source_epd_id,
            },
        }
        steps = [
            _step(f"Pick cell voltage: {v_si:.3g} V (from {v_obs_row.canonical_slug or v_obs_row.parameter_name}).",
                  value=v_si, unit="V"),
            _step(f"Pick current density: {i_si:.3g} A/m² (from {i_obs_row.canonical_slug or i_obs_row.parameter_name}).",
                  value=i_si, unit="A/m²"),
            _step("Apply identity P = V × I.",
                  math_str=f"V·I = {v_si:.3g} × {i_si:.3g} = {predicted:.3g} W/m²",
                  value=predicted, unit="W/m²"),
            _step(f"Observed P = {p_obs:.3g} W/m²; ratio observed/predicted = {ratio:.2g}× (|log₁₀| = {log_ratio:.2f}).",
                  value=p_obs, unit="W/m²"),
        ]

        # Match-band logic mirrors the hunter strict revalidator.
        if log_ratio < PI_PASS_LOG_RATIO:
            return None  # within ~2× → pass
        if log_ratio < PI_LOW_CONF_LOG_RATIO:
            severity, confidence = SEVERITY_LOW, CONFIDENCE_LOW
            reason = f"Mismatch is {ratio:.1f}× — could be measurement noise or value-selection mismatch."
            extra = " Mismatch is small enough that it could be measurement noise."
        elif log_ratio < PI_MED_CONF_LOG_RATIO:
            severity, confidence = SEVERITY_MED, CONFIDENCE_MED
            reason = f"Mismatch is {ratio:.1f}× — beyond typical reporting tolerance."
            extra = " Mismatch is meaningfully beyond measurement noise."
        elif log_ratio < PI_CROSS_COND_LOG_RATIO:
            severity, confidence = SEVERITY_HIGH, CONFIDENCE_HIGH
            reason = f"Mismatch is {ratio:.1f}× — strong evidence of a real violation."
            extra = " Mismatch is well beyond any plausible measurement noise."
        else:
            severity, confidence = SEVERITY_MED, CONFIDENCE_LOW
            reason = (
                f"Mismatch is {ratio:.1f}× — implausible at a single operating point; "
                "most likely cause: the conditionSet groups multiple operating points."
            )
            extra = " Mismatch this large is almost certainly a cross-condition artifact."

        return Violation(
            rule_name="power_identity",
            severity=severity,
            confidence=confidence,
            plain_english=(
                "Reported peak power doesn't match V·I from the same operating point."
                + extra
            ),
            chain_of_thought=steps,
            raw_values_used=_raw_values(power, v_obs_row, i_obs_row),
            trace_inputs=inputs,
            predicted=predicted,
            observed=p_obs,
            residual_pct=(p_obs - predicted) / predicted,
            confidence_reasons=[reason],
            citations=citations,
            check_type="P = V × I",
            paper_id=power.paper_id,
            flagged_slug=power.canonical_slug,
        )

    # No cell-voltage present — try OCV upper bound (Newman §22 Thevenin).
    if ocvs and currents:
        best = min(
            ((v, i) for v in ocvs for i in currents),
            key=lambda vi: abs(math.log10(vi[0].si_value * vi[1].si_value) - math.log10(p_obs)),
        )
        v_obs_row, i_obs_row = best
        v_si, i_si = v_obs_row.si_value, i_obs_row.si_value
        predicted = v_si * i_si
        ratio = p_obs / predicted
        log_ratio = abs(math.log10(ratio))

        inputs = {
            "openCircuitVoltage": {
                "value": v_si, "unit": "V", "sourceEpdId": v_obs_row.source_epd_id,
                "meta": "OCV used (no cell voltage in conditionSet)",
            },
            "currentDensity": {
                "value": i_si, "unit": "A/m²", "sourceEpdId": i_obs_row.source_epd_id,
            },
            "powerDensity": {
                "value": p_obs, "unit": "W/m²", "sourceEpdId": power.source_epd_id,
            },
        }
        steps = [
            _step("No cell/operating voltage in this conditionSet — fallback to OCV-based upper bound."),
            _step(f"OCV = {v_si:.3g} V", value=v_si, unit="V"),
            _step(f"Current density = {i_si:.3g} A/m²", value=i_si, unit="A/m²"),
            _step("Upper bound P_max ≤ V_oc · I (peak operating V is at most V_oc).",
                  math_str=f"V_oc·I = {v_si:.3g}·{i_si:.3g} = {predicted:.3g} W/m²",
                  value=predicted, unit="W/m²"),
            _step(f"Observed P = {p_obs:.3g} W/m²; ratio = {ratio:.2g}× the upper bound."),
        ]
        if ratio <= 1.15:
            return None  # fits within OCV upper bound
        severity = SEVERITY_MED if log_ratio < PI_MED_CONF_LOG_RATIO else SEVERITY_HIGH
        confidence = CONFIDENCE_MED if log_ratio < PI_CROSS_COND_LOG_RATIO else CONFIDENCE_LOW
        return Violation(
            rule_name="power_identity",
            severity=severity,
            confidence=confidence,
            plain_english=(
                "Reported peak power doesn't match V·I from the same operating point."
                " Reported P exceeds even the OCV-based upper bound; cell voltage at"
                " peak would have to exceed OCV (impossible)."
            ),
            chain_of_thought=steps,
            raw_values_used=_raw_values(power, v_obs_row, i_obs_row),
            trace_inputs=inputs,
            predicted=predicted,
            observed=p_obs,
            residual_pct=(p_obs - predicted) / predicted,
            confidence_reasons=[
                "Fell back to OCV upper bound (no operating cell voltage available).",
                f"Observed/predicted = {ratio:.2g}×; log-ratio = {log_ratio:.2f}.",
            ],
            citations=citations,
            check_type="P = V × I",
            paper_id=power.paper_id,
            flagged_slug=power.canonical_slug,
        )
    return None


# ─────────────────────────────────────────────────────────────────────
# Rule 2: ce_bounds   0 ≤ CE ≤ 100% (or 0 ≤ CE ≤ 1 fraction)
# Rule 2b: ce_out_of_unit_interval (same logic, different flag name in
# the hunter; both surface through validate_ce_bounds here).
# ─────────────────────────────────────────────────────────────────────


def validate_ce_bounds(obs: Observation) -> ViolationOrNone:
    """Coulombic efficiency must lie in its valid unit interval.

    Definition: CE = (charge actually recovered as current) /
    (charge theoretically available from oxidised substrate).  Bounded
    by 0–1 (fraction) or 0–100% by definition.  See Logan 2008 Ch. 5.
    """
    if obs.canonical_slug != "coulombic_efficiency":
        return None
    raw = obs.raw_value
    if raw is None:
        return None
    unit = (obs.unit or "").lower().strip()
    is_fraction = unit in ("", "fraction", "dimensionless", "ratio")
    is_percent = "%" in unit or "percent" in unit

    if is_fraction and raw > 1.0:
        bound, observed_val = 1.0, raw
        extra = " Reported as fraction but exceeds 1.0 (i.e., > 100%)."
    elif is_percent and raw > 100.0:
        bound, observed_val = 100.0, raw
        extra = " Reported as % but exceeds 100."
    elif raw > 100.0:
        bound, observed_val = 100.0, raw
        extra = " Numeric value exceeds the % bound; unit annotation is ambiguous."
    else:
        return None  # within bounds

    ratio = observed_val / bound
    severity = SEVERITY_HIGH if ratio < 5 else SEVERITY_MED
    confidence = CONFIDENCE_HIGH if severity == SEVERITY_HIGH else CONFIDENCE_MED
    inputs = {
        "coulombic_efficiency": {
            "value": observed_val, "unit": obs.unit or "%",
            "sourceEpdId": obs.source_epd_id,
        },
    }
    steps = [
        _step(f"Reported CE = {observed_val:.3g} {obs.unit or '%'}",
              value=observed_val, unit=obs.unit),
        _step(f"Physical bound: 0 ≤ CE ≤ {bound:.0f} {('fraction' if is_fraction else '%')}."),
        _step(f"Observed exceeds bound by {ratio:.1f}× — likely fraction-vs-percent unit confusion."),
    ]
    return Violation(
        rule_name="ce_bounds",
        severity=severity,
        confidence=confidence,
        plain_english=(
            "Coulombic efficiency exceeds 100% — physically impossible." + extra
        ),
        chain_of_thought=steps,
        raw_values_used=_raw_values(obs),
        trace_inputs=inputs,
        predicted=bound,
        observed=observed_val,
        residual_pct=(observed_val - bound) / bound,
        confidence_reasons=[f"CE = {observed_val:.3g} {obs.unit or ''} > bound of {bound}."],
        citations=[CITATION_LOGAN_2008_CH5],
        check_type="0 ≤ CE ≤ 100%",
        paper_id=obs.paper_id,
        flagged_slug=obs.canonical_slug,
    )


def validate_ce_unit_interval(obs: Observation) -> ViolationOrNone:
    """Alias for ``validate_ce_bounds`` — the hunter pipeline maintains
    a separate flag name (``ce_out_of_unit_interval``) for the same
    semantic check.  Exposed here so consumers can match either
    flag-name convention."""
    return validate_ce_bounds(obs)


# ─────────────────────────────────────────────────────────────────────
# Rule 3: voltage_ordering   V_cell ≤ V_oc
# ─────────────────────────────────────────────────────────────────────


def validate_voltage_ordering(
    cell_voltage: Observation,
    bucket: Iterable[Observation],
) -> ViolationOrNone:
    """V_cell at non-zero current must not exceed V_oc.  Required input:
    an OCV observation in the same conditionSet.

    Thevenin upper bound: an electrochemical cell under load cannot
    produce more than its open-circuit potential, since the load
    current necessarily drops voltage across the internal resistance
    (Newman & Thomas-Alyea Ch. 22)."""
    if not _looks_like_cell_voltage(cell_voltage):
        return None
    if cell_voltage.si_value is None or cell_voltage.si_value <= 0:
        return None

    ocvs = [
        o for o in bucket
        if _looks_like_ocv(o) and o.si_value is not None and o.si_value > 0
    ]
    if not ocvs:
        return None
    v_cell = cell_voltage.si_value
    v_oc = max(o.si_value for o in ocvs)
    best_ocv = next(o for o in ocvs if o.si_value == v_oc)

    if v_cell <= v_oc * VOLTAGE_ORDERING_SLACK:
        return None  # within 5% slack → pass

    ratio = v_cell / v_oc
    inputs = {
        "cell_voltage": {
            "value": v_cell, "unit": "V", "sourceEpdId": cell_voltage.source_epd_id,
        },
        "openCircuitVoltage": {
            "value": v_oc, "unit": "V", "sourceEpdId": best_ocv.source_epd_id,
        },
    }
    steps = [
        _step(f"V_cell = {v_cell:.3g} V (from {cell_voltage.canonical_slug or cell_voltage.parameter_name}).",
              value=v_cell, unit="V"),
        _step(f"V_oc = {v_oc:.3g} V (from {best_ocv.canonical_slug or best_ocv.parameter_name}).",
              value=v_oc, unit="V"),
        _step(f"V_cell / V_oc = {ratio:.2g}× — exceeds the Thevenin upper bound."),
    ]

    if ratio > VOLTAGE_ORDERING_UNIT_MIXUP_RATIO:
        severity, confidence = SEVERITY_MED, CONFIDENCE_LOW
        extra = " Probably a unit / column-swap error rather than a physics misclaim."
        reason = (
            f"V_cell is {ratio:.0f}× V_oc — most likely a mV↔V unit error or column swap."
        )
    else:
        severity, confidence = SEVERITY_HIGH, CONFIDENCE_HIGH
        extra = ""
        reason = f"V_cell = {v_cell:.3g} V > V_oc = {v_oc:.3g} V."

    return Violation(
        rule_name="voltage_ordering",
        severity=severity,
        confidence=confidence,
        plain_english=(
            "Cell voltage exceeds open-circuit voltage while current is flowing — "
            "violates the Thevenin upper bound." + extra
        ),
        chain_of_thought=steps,
        raw_values_used=_raw_values(cell_voltage, best_ocv),
        trace_inputs=inputs,
        predicted=v_oc,
        observed=v_cell,
        residual_pct=(v_cell - v_oc) / v_oc,
        confidence_reasons=[reason],
        citations=[CITATION_NEWMAN_TA_CH22, CITATION_LOGAN_HAMELERS_2006],
        check_type="V_cell ≤ V_oc",
        paper_id=cell_voltage.paper_id,
        flagged_slug=cell_voltage.canonical_slug,
    )


# ─────────────────────────────────────────────────────────────────────
# Rule 4: max_power_ohm   P_peak ≤ V_oc² / (4·R_int)
# ─────────────────────────────────────────────────────────────────────


def validate_max_power_ohm(
    power: Observation,
    bucket: Iterable[Observation],
) -> ViolationOrNone:
    """Thevenin matched-load ceiling for an electrochemical source.

    When the external load matches the internal resistance, P delivered
    to the external load is maximal at V_oc²/(4·R_int).  Reported peak
    power above this limit is physically impossible unless R_int is in
    different units than assumed (Ω vs Ω·m² — see notes; the hunter
    pipeline reports both interpretations and asks for human review).

    Reference: Newman & Thomas-Alyea Ch. 22.
    """
    if power.canonical_slug != "powerDensity":
        return None
    p_peak = power.si_value
    if p_peak is None or p_peak <= 0:
        return None

    bucket = list(bucket)
    ocvs = [o for o in bucket if _looks_like_ocv(o) and o.si_value and o.si_value > 0]
    rints = [
        o for o in bucket
        if o.canonical_slug == "internalResistance" and o.si_value and o.si_value > 0
    ]
    if not ocvs or not rints:
        return None
    v_oc = max(o.si_value for o in ocvs)
    r_int = min(o.si_value for o in rints)
    best_ocv = next(o for o in ocvs if o.si_value == v_oc)
    best_rint = next(o for o in rints if o.si_value == r_int)
    p_max_pred = (v_oc ** 2) / (4.0 * r_int)
    ratio = p_peak / p_max_pred if p_max_pred > 0 else math.inf
    if ratio <= MAX_POWER_OHM_PASS_RATIO:
        return None

    inputs = {
        "openCircuitVoltage": {
            "value": v_oc, "unit": "V", "sourceEpdId": best_ocv.source_epd_id,
        },
        "internalResistance": {
            "value": r_int, "unit": "Ω", "sourceEpdId": best_rint.source_epd_id,
        },
        "powerDensity": {
            "value": p_peak, "unit": "W/m²", "sourceEpdId": power.source_epd_id,
        },
    }
    steps = [
        _step(f"V_oc = {v_oc:.3g} V", value=v_oc, unit="V"),
        _step(f"R_int = {r_int:.3g} Ω (raw column; unit-area conversion not applied)",
              value=r_int, unit="Ω"),
        _step("Apply Thevenin matched-load: P_max = V_oc² / (4·R_int).",
              math_str=f"({v_oc:.3g}²) / (4·{r_int:.3g}) = {p_max_pred:.3g} W/m²",
              value=p_max_pred, unit="W/m²"),
        _step(f"Reported peak P = {p_peak:.3g} W/m²; ratio = {ratio:.2g}×."),
    ]
    return Violation(
        rule_name="max_power_ohm",
        severity=SEVERITY_MED,
        confidence=CONFIDENCE_MED,
        plain_english="Reported peak P exceeds the matched-load ceiling V_oc²/(4·R_int).",
        chain_of_thought=steps,
        raw_values_used=_raw_values(power, best_ocv, best_rint),
        trace_inputs=inputs,
        predicted=p_max_pred,
        observed=p_peak,
        residual_pct=(p_peak - p_max_pred) / p_max_pred,
        confidence_reasons=[
            f"Peak P = {ratio:.1f}× the matched-load ceiling.",
            "R_int unit may be Ω vs Ω·m²; check the area conversion.",
        ],
        citations=[CITATION_NEWMAN_TA_CH22],
        check_type="P_max ≤ V_oc² / (4 · R_int)",
        paper_id=power.paper_id,
        flagged_slug=power.canonical_slug,
    )


# ─────────────────────────────────────────────────────────────────────
# Rule 5: temperature_out_of_plausible_range
# ─────────────────────────────────────────────────────────────────────


def validate_temperature_range(obs: Observation) -> ViolationOrNone:
    """Operating temperature outside the biological-catalyst plausible
    range.

    The hunter rule fires when T > 100 °C or T < -20 °C — the upper
    bound matches the denaturation onset of mesophilic electroactive
    biofilms (~60 °C with significant safety margin to 100 °C).  Common
    failure mode: Kelvin value transcribed as °C (e.g. 298 K → 298 °C).
    Reference: Logan 2008 §2.
    """
    if obs.canonical_slug != "temperature":
        return None
    val = obs.raw_value
    if val is None:
        return None
    unit = (obs.unit or "").lower()
    inputs = {
        "temperature": {
            "value": val, "unit": obs.unit or "°C",
            "sourceEpdId": obs.source_epd_id,
        },
    }
    if val > TEMPERATURE_UPPER_C:
        kelvin_equiv = val - 273.15
        return Violation(
            rule_name="temperature_out_of_plausible_range",
            severity=SEVERITY_MED,
            confidence=CONFIDENCE_HIGH,
            plain_english=(
                "Operating temperature outside biological + typical-lab range; "
                "likely K-as-°C transcription."
            ),
            chain_of_thought=[
                _step(f"Reported T = {val:.3g} {obs.unit or '°C'}", value=val, unit=obs.unit),
                _step("Bio-catalyst denaturation onset ≈ 60 °C; T > 100 °C breaks living biofilm."),
                _step(f"Hypothesis: value was {val:.0f} K mistyped as °C; that's "
                      f"{kelvin_equiv:.1f} °C — typical lab T."),
            ],
            raw_values_used=_raw_values(obs),
            trace_inputs=inputs,
            predicted=None,
            observed=val,
            residual_pct=None,
            confidence_reasons=[
                f"T = {val:.3g} {unit or '°C'} is well above the biological denaturation threshold (~60 °C).",
                f"If interpreted as Kelvin: {kelvin_equiv:.1f} °C — much more plausible for a BES.",
            ],
            citations=[CITATION_LOGAN_2008_CH2],
            check_type="0 °C ≤ T ≤ 100 °C (biological)",
            paper_id=obs.paper_id,
            flagged_slug=obs.canonical_slug,
        )
    if val < TEMPERATURE_LOWER_C:
        return Violation(
            rule_name="temperature_out_of_plausible_range",
            severity=SEVERITY_MED,
            confidence=CONFIDENCE_HIGH,
            plain_english="Operating temperature below realistic lab range.",
            chain_of_thought=[
                _step(f"Reported T = {val:.3g} {obs.unit or '°C'} — below typical lab freezer range."),
            ],
            raw_values_used=_raw_values(obs),
            trace_inputs=inputs,
            predicted=None,
            observed=val,
            residual_pct=None,
            confidence_reasons=[f"T = {val:.3g} {unit or '°C'} is well below realistic operating range."],
            citations=[CITATION_LOGAN_2008_CH2],
            check_type="0 °C ≤ T ≤ 100 °C (biological)",
            paper_id=obs.paper_id,
            flagged_slug=obs.canonical_slug,
        )
    return None


# ─────────────────────────────────────────────────────────────────────
# Rule 6: removal_out_of_unit_interval
# ─────────────────────────────────────────────────────────────────────

REMOVAL_FAMILY_SLUGS: set[str] = {
    "removal_efficiency",
    "cod_removal",
    "bod_removal",
    "nitrogen_removal",
    "salt_removal",
    "heavy_metal_removal",
    "metal_removal_efficiency",
    "phosphorus_removal",
    "nitrate_removal",
    "ammonium_removal",
    "toc_removal",
    "tn_removal",
    "tss_removal",
    "sulfate_removal",
    "cathodic_h2_recovery",
    "energy_efficiency",
    "faradaic_efficiency",
}


def validate_removal_unit_interval(obs: Observation) -> ViolationOrNone:
    """A removal-family quantity (efficiencies, recoveries, removals)
    must lie in [0, 100%] or [0, 1] depending on the reported unit.

    Reference: Logan 2008 §5 — efficiency / yield conventions.
    """
    if obs.canonical_slug not in REMOVAL_FAMILY_SLUGS:
        return None
    val = obs.raw_value
    if val is None:
        return None
    unit = (obs.unit or "").lower().strip()
    if "%" in unit or "percent" in unit:
        bound = 100.0
    else:
        bound = 1.0
    if 0 <= val <= bound:
        return None
    inputs = {
        "removal_efficiency": {
            "value": val, "unit": obs.unit or "%", "sourceEpdId": obs.source_epd_id,
        },
    }
    residual = (val - bound) / bound if val > bound else val / bound
    return Violation(
        rule_name="removal_out_of_unit_interval",
        severity=SEVERITY_LOW,
        confidence=CONFIDENCE_MED,
        plain_english="Removal efficiency outside 0-100% / 0-1.",
        chain_of_thought=[
            _step(f"Reported removal = {val:.3g} {obs.unit or '%'}; bound is [0, {bound}]."),
        ],
        raw_values_used=_raw_values(obs),
        trace_inputs=inputs,
        predicted=bound,
        observed=val,
        residual_pct=residual,
        confidence_reasons=[
            f"Removal efficiency {val:.3g} outside [0, {bound}]; likely a unit-format mix-up.",
        ],
        citations=[CITATION_LOGAN_2008_CH5],
        check_type="0 ≤ removal ≤ 100%",
        paper_id=obs.paper_id,
        flagged_slug=obs.canonical_slug,
    )


# ─────────────────────────────────────────────────────────────────────
# Rule 7: non_positive   value > 0 for currentDensity / powerDensity
# ─────────────────────────────────────────────────────────────────────

POSITIVE_REQUIRED_SLUGS: set[str] = {
    "currentDensity",
    "powerDensity",
}


def validate_non_positive(obs: Observation) -> ViolationOrNone:
    """Reported value ≤ 0 for a quantity that must be > 0 by sign
    convention (current density, power density).

    Reference: Logan & Hamelers 2006 §3 (reporting conventions).
    """
    if obs.canonical_slug not in POSITIVE_REQUIRED_SLUGS:
        return None
    val = obs.si_value if obs.si_value is not None else obs.raw_value
    if val is None or val > 0:
        return None
    slug = obs.canonical_slug
    inputs = {
        slug: {
            "value": val, "unit": obs.unit or "",
            "sourceEpdId": obs.source_epd_id,
        },
    }
    # Use parameter_name as the key when present, mirroring hunter output.
    if obs.parameter_name:
        inputs = {
            obs.parameter_name: {
                "value": val, "unit": obs.unit or "",
                "sourceEpdId": obs.source_epd_id,
            },
        }
    return Violation(
        rule_name="non_positive",
        severity=SEVERITY_LOW,
        confidence=CONFIDENCE_MED,
        plain_english="Value reported as ≤ 0 for a quantity that must be > 0. Sign convention or extraction error.",
        chain_of_thought=[
            _step(f"Reported {slug} = {val:.3g} {obs.unit or ''}; expected > 0."),
        ],
        raw_values_used=_raw_values(obs),
        trace_inputs=inputs,
        predicted=0.0,
        observed=val,
        residual_pct=None,
        confidence_reasons=[f"{slug} = {val:.3g} ≤ 0, expected > 0."],
        citations=[CITATION_LOGAN_HAMELERS_2006],
        check_type="value > 0",
        paper_id=obs.paper_id,
        flagged_slug=obs.canonical_slug,
    )


# ─────────────────────────────────────────────────────────────────────
# Rule 8: ocv_thermodynamic_ceiling
# ─────────────────────────────────────────────────────────────────────


def validate_ocv_thermodynamic_ceiling(
    obs: Observation,
    paper_context: Optional[PaperContext] = None,
) -> ViolationOrNone:
    """Reported OCV exceeds the system-class thermodynamic ceiling.

    OCV is bounded above by the difference between the best cathode
    half-reaction (e.g. O₂/H₂O ≈ +0.81 V vs SHE at pH 7) and the
    anode half-reaction available to the chosen substrate (typically
    around -0.3 V vs SHE for acetate).  With reasonable slack this
    gives the per-system ceilings in ``OCV_CEILING_V``.

    Reference: Logan 2008 §2.3.  Catches mV/V unit mix-ups,
    series-stacked-cell measurements reported as single-cell OCV, and
    misclassified non-BES papers.
    """
    if obs.canonical_slug != "openCircuitVoltage":
        return None
    v_si = obs.si_value
    if v_si is None or v_si <= 0:
        return None

    system_class = (paper_context.effective_system_class() if paper_context else "OTHER")
    if system_class in OCV_CEILING_OUT_OF_SCOPE:
        return None
    ceiling = OCV_CEILING_V.get(system_class, OCV_CEILING_V["OTHER"])
    if v_si <= ceiling:
        return None

    ratio = v_si / ceiling
    if ratio > 50:
        severity, confidence = SEVERITY_MED, CONFIDENCE_LOW
        confidence_reasons = [
            f"OCV reported as {v_si:.3g} V — {ratio:.0f}× the {system_class} ceiling of {ceiling} V.",
            "Almost certainly a mV/V unit mix-up or a series-stacked-cell reading.",
        ]
        extra = " Probable unit-prefix error or series-stack reading, not a true single-cell OCV."
    elif ratio > 2:
        severity, confidence = SEVERITY_HIGH, CONFIDENCE_HIGH
        confidence_reasons = [
            f"OCV {v_si:.3g} V is {ratio:.1f}× the {system_class} thermodynamic ceiling.",
            "No standard redox couple in this system class produces OCV > ceiling.",
        ]
        extra = " No standard redox chemistry supports this OCV in this system class."
    else:
        severity, confidence = SEVERITY_MED, CONFIDENCE_MED
        confidence_reasons = [
            f"OCV {v_si:.3g} V exceeds the {system_class} ceiling ({ceiling} V) by {((ratio - 1) * 100):.0f}%.",
            "Within the range where novel cathode chemistry might justify it.",
        ]
        extra = " Within the range where unusual cathode chemistry could justify the reading."

    inputs = {
        "openCircuitVoltage": {
            "value": v_si, "unit": "V", "sourceEpdId": obs.source_epd_id,
            "meta": f"raw: {obs.raw_value} {obs.unit or ''}",
        },
        "system_class_ceiling": {
            "value": ceiling, "unit": "V",
            "meta": "Logan 2008 §2.3 + 20-30% slack",
        },
    }
    steps = [
        _step(f"Paper system class: {system_class}."),
        _step(
            f"Thermodynamic ceiling for {system_class}: {ceiling} V "
            "(O₂ cathode + typical anode redox at pH 7, with slack).",
            value=ceiling, unit="V",
        ),
        _step(f"Reported OCV = {v_si:.3g} V (from raw value {obs.raw_value} {obs.unit or ''}).",
              value=v_si, unit="V"),
        _step(f"Ratio observed/ceiling = {ratio:.2g}×."),
    ]
    return Violation(
        rule_name="ocv_thermodynamic_ceiling",
        severity=severity,
        confidence=confidence,
        plain_english=(
            f"OCV of {v_si:.3g} V exceeds the thermodynamic ceiling for {system_class} systems."
            + extra
        ),
        chain_of_thought=steps,
        raw_values_used=_raw_values(obs),
        trace_inputs=inputs,
        predicted=ceiling,
        observed=v_si,
        residual_pct=(v_si - ceiling) / ceiling,
        confidence_reasons=confidence_reasons,
        citations=[CITATION_LOGAN_2008_CH2],
        check_type=f"OCV ≤ {ceiling} V ({system_class} ceiling)",
        paper_id=obs.paper_id,
        flagged_slug=obs.canonical_slug,
    )


# ─────────────────────────────────────────────────────────────────────
# Rule 9: within_paper_duplicate
# ─────────────────────────────────────────────────────────────────────


def validate_within_paper_duplicate(
    observations: Iterable[Observation],
) -> list[Violation]:
    """Detect within-paper, within-conditionSet duplicates of the same
    canonical slug with > 5× spread, restricted to slugs where wide
    variation is rarely legitimate (CE, R_int, OCV, EE).

    Unlike most rules this is a *set-level* rule — it consumes a
    bucket of observations and returns zero or more violations (one
    per offending slug-in-conditionSet group).

    Reference: data-quality convention (Logan & Hamelers 2006 §3 on
    reporting consistency); the slug allowlist is empirically chosen
    to avoid false positives from intentional parameter sweeps.
    """
    grouped: dict[tuple[Optional[str], Optional[str], str], list[Observation]] = defaultdict(list)
    for o in observations:
        if not o.canonical_slug or o.canonical_slug not in DUPLICATE_INCLUDED_SLUGS:
            continue
        if o.si_value is None or o.si_value <= 0:
            continue
        if o.condition_set_id is None:
            continue
        grouped[(o.paper_id, o.condition_set_id, o.canonical_slug)].append(o)

    out: list[Violation] = []
    for (paper_id, cs_id, slug), rows in grouped.items():
        if len(rows) < 2:
            continue
        si_vals = [r.si_value for r in rows]
        minv, maxv = min(si_vals), max(si_vals)
        if minv <= 0 or maxv / minv <= DUPLICATE_MIN_SPREAD:
            continue
        ratio = maxv / minv
        idx_min = si_vals.index(minv)
        idx_max = si_vals.index(maxv)
        low_row, high_row = rows[idx_min], rows[idx_max]

        if ratio > 1000:
            severity, confidence = SEVERITY_MED, CONFIDENCE_LOW
            extra = " Spread is 1000× — almost certainly a unit-prefix mix-up."
        elif ratio > 100:
            severity, confidence = SEVERITY_HIGH, CONFIDENCE_MED
            extra = " Spread > 100× — likely involves a unit error in one of the reports."
        elif ratio > 30:
            severity, confidence = SEVERITY_HIGH, CONFIDENCE_HIGH
            extra = " Spread > 30× — too large to be explained by configuration differences."
        else:
            severity, confidence = SEVERITY_MED, CONFIDENCE_MED
            extra = " Spread is 5-30× — well beyond measurement noise."

        inputs = {
            f"{slug}_low": {
                "value": minv, "unit": low_row.unit or "",
                "sourceEpdId": low_row.source_epd_id,
                "meta": f"raw: {low_row.raw_value} ({low_row.parameter_name})",
            },
            f"{slug}_high": {
                "value": maxv, "unit": high_row.unit or "",
                "sourceEpdId": high_row.source_epd_id,
                "meta": f"raw: {high_row.raw_value} ({high_row.parameter_name})",
            },
        }
        steps = [
            _step(f"Found {len(rows)} measurements of {slug} in the same conditionSet."),
            _step(f"Range: [{minv:.3g}, {maxv:.3g}] (SI scale).",
                  value=minv, unit=low_row.unit or ""),
            _step(f"Spread: {ratio:.1f}× — beyond typical reporting tolerance (~2×)."),
            _step("Same-slug duplicates in a single conditionSet should agree "
                  "unless the paper explicitly distinguishes them."),
        ]
        plain = (
            f"Paper reports {slug} {len(rows)} times in the same conditionSet, "
            f"ranging from {minv:.3g} to {maxv:.3g} ({ratio:.1f}× spread)."
            + extra
        )
        out.append(Violation(
            rule_name="within_paper_duplicate",
            severity=severity,
            confidence=confidence,
            plain_english=plain,
            chain_of_thought=steps,
            raw_values_used=_raw_values(low_row, high_row),
            trace_inputs=inputs,
            predicted=(minv * maxv) ** 0.5,
            observed=maxv,
            residual_pct=ratio - 1.0,
            confidence_reasons=[
                f"{len(rows)} duplicates, spread {ratio:.1f}×.",
                f"Slug `{slug}` is in the wide-variation allowlist for within-paper duplicates.",
            ],
            citations=[CITATION_LOGAN_HAMELERS_2006],
            check_type="Same canonical slug, same conditionSet → values agree",
            paper_id=paper_id,
            flagged_slug=slug,
        ))
    return out


# ─────────────────────────────────────────────────────────────────────
# Rule 10: faraday_h2_ceiling (MEC only)
# ─────────────────────────────────────────────────────────────────────


def validate_faraday_h2_ceiling(
    observations: Iterable[Observation],
    paper_context: Optional[PaperContext] = None,
) -> ViolationOrNone:
    """For MEC papers reporting both current and H₂ production rate,
    check that reported H₂ ≤ Faraday-law theoretical maximum.

    Working in SI, the volumetric daily H₂ rate that can be produced
    by current I across electrode area A in a reactor volume V is::

        r_H2_max [m³/m³/d] = I [A/m²] · 86400 [s/d]
                             / (2 · F [C/mol])    # mol e⁻ → mol H₂
                             · 22.414e-3 [m³/mol] # mol H₂ → m³
                             · (A_electrode [m²] / V_reactor [m³])

    Required observations (paper-level grouping; the v2 corpus stores
    conditionSets per paper, so the validator groups by paper_id when
    a unifying condition_set_id is not available):
        * ``currentDensity`` (SI: A/m²)
        * ``specific_h2_production`` (SI: m³/m³/d)
        * ``electrode_surface_area`` (SI: m²)
        * ``reactor_volume`` (SI: L, per normalizer convention; converted to m³ here)

    System-class gate: this rule applies only to MEC papers (the only
    BES type whose primary product is H₂).  Reference: Logan 2008 Ch. 9.
    """
    if paper_context is None:
        # Default: still attempt the rule but mark as NEEDS_REVIEW if
        # we can't confirm system class.
        sys_class = "OTHER"
    else:
        sys_class = paper_context.effective_system_class()
    if sys_class != "MEC":
        return None

    observations = list(observations)
    buckets: dict[str, list[Observation]] = defaultdict(list)
    for o in observations:
        if o.canonical_slug in (
            "currentDensity",
            "specific_h2_production",
            "electrode_surface_area",
            "reactor_volume",
        ) and o.si_value is not None and o.si_value > 0:
            buckets[o.canonical_slug].append(o)

    needed = ("currentDensity", "specific_h2_production", "electrode_surface_area", "reactor_volume")
    if not all(buckets.get(s) for s in needed):
        return None

    # Charitable interpretation: pick highest I (largest theoretical max),
    # highest reported H2 rate (largest observed) — this gives the paper
    # the most generous benefit of the doubt before flagging.
    i_row = max(buckets["currentDensity"], key=lambda r: r.si_value)
    r_h2_row = max(buckets["specific_h2_production"], key=lambda r: r.si_value)
    a_row = buckets["electrode_surface_area"][0]
    v_row = buckets["reactor_volume"][0]

    i_si = float(i_row.si_value)
    r_h2_si = float(r_h2_row.si_value)
    a_si = float(a_row.si_value)
    v_si_L = float(v_row.si_value)
    v_si_m3 = v_si_L / 1000.0
    if v_si_m3 <= 0 or a_si <= 0 or i_si <= 0:
        return None

    r_max_vol = (
        (i_si * SECONDS_PER_DAY / (2.0 * F_FARADAY))
        * (a_si / v_si_m3)
        * (H2_MOLAR_VOLUME_L / 1000.0)
    )
    if r_max_vol <= 0:
        return None
    ratio = r_h2_si / r_max_vol
    if ratio <= FARADAY_PASS_RATIO:
        return None

    if ratio > 100:
        severity, confidence = SEVERITY_MED, CONFIDENCE_LOW
        extra = (
            " Reported rate is 100× the Faraday-law upper bound; "
            "likely a unit-conversion error in reactor volume or electrode area."
        )
    elif ratio > 10:
        severity, confidence = SEVERITY_HIGH, CONFIDENCE_HIGH
        extra = " Reported H₂ rate is well above the maximum possible from the reported current."
    else:
        severity, confidence = SEVERITY_MED, CONFIDENCE_MED
        extra = " Reported H₂ rate exceeds the Faraday-law upper bound by more than 30%."

    inputs = {
        "currentDensity": {
            "value": i_si, "unit": "A/m²", "sourceEpdId": i_row.source_epd_id,
            "meta": f"raw: {i_row.raw_value} {i_row.unit or ''}",
        },
        "electrode_surface_area": {
            "value": a_si, "unit": "m²", "sourceEpdId": a_row.source_epd_id,
            "meta": f"raw: {a_row.raw_value} {a_row.unit or ''}",
        },
        "reactor_volume": {
            "value": v_si_L, "unit": "L", "sourceEpdId": v_row.source_epd_id,
            "meta": f"raw: {v_row.raw_value} {v_row.unit or ''} (= {v_si_m3:.4g} m³ for the formula)",
        },
        "specific_h2_production": {
            "value": r_h2_si, "unit": "m³/m³/d", "sourceEpdId": r_h2_row.source_epd_id,
            "meta": f"raw: {r_h2_row.raw_value} {r_h2_row.unit or ''}",
        },
    }
    steps = [
        _step(f"Charge per day at I = {i_si:.3g} A/m²: I × 86400 s = {i_si * SECONDS_PER_DAY:.3g} C/(m²·day)."),
        _step(f"Moles e⁻ per day per m²: that ÷ F = {i_si * SECONDS_PER_DAY / F_FARADAY:.3g} mol/(m²·day).",
              math_str=f"{i_si:.3g} × 86400 / 96485"),
        _step(f"Moles H₂ per day per m² (½ × moles e⁻): {i_si * SECONDS_PER_DAY / (2 * F_FARADAY):.3g}."),
        _step(f"Per-area volume of H₂ per day (× 22.414 L/mol): "
              f"{(i_si * SECONDS_PER_DAY / (2 * F_FARADAY)) * H2_MOLAR_VOLUME_L:.3g} L/(m²·day)."),
        _step(f"Convert to per-reactor-volume (× A/V): A = {a_si:.3g} m², V = {v_si_L:.3g} L = {v_si_m3:.4g} m³.",
              math_str=f"× {a_si:.3g} / {v_si_m3:.4g}",
              value=r_max_vol, unit="m³/m³/d"),
        _step(f"Theoretical max r_H₂ = {r_max_vol:.3g} m³/m³/d.",
              value=r_max_vol, unit="m³/m³/d"),
        _step(f"Reported r_H₂ = {r_h2_si:.3g} m³/m³/d; ratio = {ratio:.2g}×.",
              value=r_h2_si, unit="m³/m³/d"),
    ]
    return Violation(
        rule_name="faraday_h2_ceiling",
        severity=severity,
        confidence=confidence,
        plain_english=(
            f"H₂ production rate {r_h2_si:.3g} m³/m³/d exceeds the Faraday-law maximum of "
            f"{r_max_vol:.3g} m³/m³/d derivable from the reported {i_si:.3g} A/m² current."
            + extra
        ),
        chain_of_thought=steps,
        raw_values_used=_raw_values(r_h2_row, i_row),
        trace_inputs=inputs,
        predicted=r_max_vol,
        observed=r_h2_si,
        residual_pct=(r_h2_si - r_max_vol) / r_max_vol,
        confidence_reasons=[
            f"Reported H₂ = {r_h2_si:.3g} m³/m³/d vs Faraday max = {r_max_vol:.3g} m³/m³/d.",
            f"Ratio = {ratio:.1f}× (1.3× slack allowed for round-tripping).",
        ],
        citations=[CITATION_LOGAN_2008_CH9],
        check_type="r_H₂ ≤ I·A/(2F·V) × 22.4 L/mol × 86400 s/day",
        paper_id=r_h2_row.paper_id,
        flagged_slug="specific_h2_production",
    )


# ─────────────────────────────────────────────────────────────────────
# Master validator
# ─────────────────────────────────────────────────────────────────────


class ScientificValidator:
    """Apply the full physics-violation rule canon to a set of
    observations from one paper.

    Typical use:

        >>> validator = ScientificValidator()
        >>> violations = validator.validate(observations, paper_context)

    where ``observations`` is a list of :class:`Observation` for a
    single paper (ideally tagged with ``condition_set_id`` so that the
    set-level rules can match V, I, P, OCV, R_int across one operating
    point).
    """

    # Rule registry — exposed as a class attribute so consumers can
    # introspect / disable rules.  Each entry is
    #   (rule_name, scope, fn)
    # where scope is one of:
    #   "single"            → fn(obs) -> ViolationOrNone (one row at a time)
    #   "single_bucket"     → fn(obs, bucket) -> ViolationOrNone
    #   "single_context"    → fn(obs, paper_context) -> ViolationOrNone
    #   "set"               → fn(observations, paper_context=None) ->
    #                          list[Violation] | ViolationOrNone
    RULES: list[tuple[str, str, Callable[..., Any]]] = [
        ("ce_bounds",                          "single",         validate_ce_bounds),
        ("temperature_out_of_plausible_range", "single",         validate_temperature_range),
        ("removal_out_of_unit_interval",       "single",         validate_removal_unit_interval),
        ("non_positive",                       "single",         validate_non_positive),
        ("ocv_thermodynamic_ceiling",          "single_context", validate_ocv_thermodynamic_ceiling),
        ("power_identity",                     "single_bucket",  validate_power_identity),
        ("voltage_ordering",                   "single_bucket",  validate_voltage_ordering),
        ("max_power_ohm",                      "single_bucket",  validate_max_power_ohm),
        ("within_paper_duplicate",             "set",            validate_within_paper_duplicate),
        ("faraday_h2_ceiling",                 "set_context",    validate_faraday_h2_ceiling),
    ]

    def __init__(self, enabled_rules: Optional[Iterable[str]] = None) -> None:
        """Construct a validator.

        Parameters
        ----------
        enabled_rules : iterable of str, optional
            If given, only rules whose name appears in this iterable
            are run.  Default: all 10 rules.
        """
        if enabled_rules is None:
            self.enabled = {name for name, _scope, _fn in self.RULES}
        else:
            self.enabled = set(enabled_rules)

    def validate(
        self,
        observations: Iterable[Observation],
        paper_context: Optional[PaperContext] = None,
    ) -> list[Violation]:
        """Apply all enabled rules to ``observations``.

        ``observations`` should cover a single paper; for the
        bucket-level rules the validator groups them by
        ``condition_set_id``.  Observations without a condition_set_id
        are placed in a single ``None`` bucket so that single-row
        rules still fire.

        Returns
        -------
        list[Violation]
            All violations, in rule-registration order.  Duplicates
            (same rule + same flagged slug + same paper) are not
            de-duplicated — that's the caller's choice (a row may
            legitimately fire on multiple rules).
        """
        obs_list = list(observations)
        # Group by condition_set_id for bucket rules.
        by_cs: dict[Optional[str], list[Observation]] = defaultdict(list)
        for o in obs_list:
            by_cs[o.condition_set_id].append(o)

        out: list[Violation] = []
        for name, scope, fn in self.RULES:
            if name not in self.enabled:
                continue
            if scope == "single":
                for o in obs_list:
                    v = fn(o)
                    if v is not None:
                        out.append(v)
            elif scope == "single_context":
                for o in obs_list:
                    v = fn(o, paper_context)
                    if v is not None:
                        out.append(v)
            elif scope == "single_bucket":
                for cs_id, bucket in by_cs.items():
                    for o in bucket:
                        v = fn(o, bucket)
                        if v is not None:
                            out.append(v)
            elif scope == "set":
                result = fn(obs_list)
                if isinstance(result, list):
                    out.extend(result)
                elif result is not None:
                    out.append(result)
            elif scope == "set_context":
                result = fn(obs_list, paper_context)
                if isinstance(result, list):
                    out.extend(result)
                elif result is not None:
                    out.append(result)
        return out


# ─────────────────────────────────────────────────────────────────────
# Legacy: the original generic-system validator that lived in this
# module before the rule canon was canonicalised here.  Kept for
# backwards compatibility; not part of the physics-violation rule set.
# Pre-existing consumers should migrate to ScientificValidator above.
# ─────────────────────────────────────────────────────────────────────

from dataclasses import dataclass as _dataclass
from enum import Enum as _Enum
from functools import wraps as _wraps


class SystemType(str, _Enum):
    """Electrochemical system types (legacy enum)."""

    MFC = "mfc"
    MEC = "mec"
    MDC = "mdc"
    PEM = "pem"
    SOFC = "sofc"


@_dataclass
class MaterialSpecification:
    """Material specifications for electrochemical systems (legacy)."""

    anode_material: str
    cathode_material: str
    anode_surface_area: float
    cathode_surface_area: float
    membrane_type: Optional[str] = None


@_dataclass
class OperatingConditions:
    """Operating conditions for electrochemical systems (legacy)."""

    temperature: float
    ph: float
    pressure: Optional[float] = None
    substrate_concentration: Optional[float] = None
    external_resistance: Optional[float] = None


@_dataclass
class SystemConfiguration:
    """System configuration parameters (legacy)."""

    reactor_volume: float
    electrode_spacing: float
    flow_mode: Optional[str] = None


class ValidationError(Exception):
    """Custom validation error with detailed field information (legacy)."""

    def __init__(self, field: str, message: str, value: Any = None) -> None:
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"{field}: {message}")


class GenericSystemValidator:
    """Domain-specific generic validation rules for system configuration
    (legacy; superseded by :class:`ScientificValidator` for
    physics-violation checks)."""

    COMPATIBLE_MATERIALS = {
        "carbon_cloth": ["graphite", "carbon_felt", "carbon_paper"],
        "stainless_steel": ["titanium", "nickel"],
        "platinum": ["platinum_carbon", "platinum_ruthenium"],
        "graphite": ["carbon_cloth", "carbon_felt"],
    }

    SYSTEM_CONSTRAINTS = {
        SystemType.MFC: {
            "temperature": (283.15, 318.15),
            "ph": (6.0, 8.5),
            "reactor_volume": (10, 5000),
            "electrode_spacing": (0.1, 10),
        },
        SystemType.MEC: {
            "temperature": (283.15, 318.15),
            "ph": (6.0, 8.5),
            "reactor_volume": (10, 5000),
            "electrode_spacing": (0.1, 10),
        },
        SystemType.MDC: {
            "temperature": (283.15, 318.15),
            "ph": (6.0, 8.5),
            "reactor_volume": (10, 5000),
        },
        SystemType.PEM: {
            "temperature": (323.15, 363.15),
            "ph": (0, 2),
            "pressure": (1, 5),
        },
        SystemType.SOFC: {
            "temperature": (873.15, 1273.15),
            "pressure": (1, 10),
        },
    }

    @classmethod
    def validate_materials(cls, materials: MaterialSpecification, system_type: SystemType) -> list[str]:
        errors: list[str] = []
        anode_base = materials.anode_material.lower().split("_")[0]
        cathode_base = materials.cathode_material.lower().split("_")[0]
        if anode_base in cls.COMPATIBLE_MATERIALS:
            compatible = cls.COMPATIBLE_MATERIALS[anode_base]
            if cathode_base not in compatible and cathode_base != anode_base:
                errors.append(
                    f"Anode material '{materials.anode_material}' may not be compatible "
                    f"with cathode material '{materials.cathode_material}'"
                )
        if materials.anode_surface_area > 10000:
            errors.append(f"Anode surface area {materials.anode_surface_area} cm² seems unrealistically large")
        if materials.cathode_surface_area > 10000:
            errors.append(f"Cathode surface area {materials.cathode_surface_area} cm² seems unrealistically large")
        if system_type == SystemType.SOFC and "carbon" in anode_base:
            errors.append("Carbon-based materials are not suitable for SOFC high temperatures")
        return errors

    @classmethod
    def validate_operating_conditions(cls, conditions: OperatingConditions, system_type: SystemType) -> list[str]:
        errors: list[str] = []
        if system_type not in cls.SYSTEM_CONSTRAINTS:
            return errors
        constraints = cls.SYSTEM_CONSTRAINTS[system_type]
        if "temperature" in constraints:
            min_temp, max_temp = constraints["temperature"]
            if not min_temp <= conditions.temperature <= max_temp:
                errors.append(
                    f"Temperature {conditions.temperature}K is outside typical range "
                    f"{min_temp}-{max_temp}K for {system_type.value}"
                )
        if "ph" in constraints:
            min_ph, max_ph = constraints["ph"]
            if not min_ph <= conditions.ph <= max_ph:
                errors.append(
                    f"pH {conditions.ph} is outside typical range "
                    f"{min_ph}-{max_ph} for {system_type.value}"
                )
        if "pressure" in constraints and conditions.pressure:
            min_pressure, max_pressure = constraints["pressure"]
            if not min_pressure <= conditions.pressure <= max_pressure:
                errors.append(
                    f"Pressure {conditions.pressure} atm is outside typical range "
                    f"{min_pressure}-{max_pressure} atm for {system_type.value}"
                )
        if system_type in [SystemType.MFC, SystemType.MEC, SystemType.MDC]:
            if conditions.substrate_concentration and conditions.substrate_concentration > 50:
                errors.append(f"Substrate concentration {conditions.substrate_concentration} g/L is very high")
            if conditions.external_resistance and conditions.external_resistance > 10000:
                errors.append(f"External resistance {conditions.external_resistance} Ohms is very high")
        return errors

    @classmethod
    def validate_configuration(cls, config: SystemConfiguration, system_type: SystemType) -> list[str]:
        errors: list[str] = []
        if system_type not in cls.SYSTEM_CONSTRAINTS:
            return errors
        constraints = cls.SYSTEM_CONSTRAINTS[system_type]
        if "reactor_volume" in constraints:
            min_vol, max_vol = constraints["reactor_volume"]
            if not min_vol <= config.reactor_volume <= max_vol:
                errors.append(
                    f"Reactor volume {config.reactor_volume} mL is outside typical range "
                    f"{min_vol}-{max_vol} mL for {system_type.value}"
                )
        if "electrode_spacing" in constraints:
            min_spacing, max_spacing = constraints["electrode_spacing"]
            if not min_spacing <= config.electrode_spacing <= max_spacing:
                errors.append(
                    f"Electrode spacing {config.electrode_spacing} cm is outside typical range "
                    f"{min_spacing}-{max_spacing} cm for {system_type.value}"
                )
        if config.flow_mode and config.flow_mode not in ["batch", "continuous", "fed-batch"]:
            errors.append(f"Unknown flow mode: {config.flow_mode}")
        return errors


def sanitize_string(value: str, max_length: int = 100) -> str:
    """Sanitize string inputs (legacy helper)."""
    if not value:
        return ""
    value = "".join(ch for ch in value if ch.isprintable())
    if len(value) > max_length:
        value = value[:max_length]
    return value.strip()


def validate_numeric_range(
    value: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    field_name: str = "value",
) -> float:
    """Validate numeric values are within expected ranges (legacy)."""
    if min_val is not None and value < min_val:
        raise ValidationError(
            field=field_name,
            message=f"Value {value} is below minimum {min_val}",
            value=value,
        )
    if max_val is not None and value > max_val:
        raise ValidationError(
            field=field_name,
            message=f"Value {value} exceeds maximum {max_val}",
            value=value,
        )
    return value


def validate_request(func: Callable) -> Callable:
    """Decorator for validating requests with scientific constraints (legacy)."""

    @_wraps(func)
    def wrapper(
        materials: MaterialSpecification,
        conditions: OperatingConditions,
        config: SystemConfiguration,
        system_type: SystemType,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        errors: list[str] = []
        errors.extend(GenericSystemValidator.validate_materials(materials, system_type))
        errors.extend(GenericSystemValidator.validate_operating_conditions(conditions, system_type))
        errors.extend(GenericSystemValidator.validate_configuration(config, system_type))
        if errors:
            raise ValidationError(
                field="request",
                message=f"Validation failed: {'; '.join(errors)}",
                value=None,
            )
        return func(materials, conditions, config, system_type, *args, **kwargs)

    return wrapper


__all__ = [
    # Rule canon (the new canonical API)
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
    # Types re-exports for convenience
    "Observation",
    "PaperContext",
    "Violation",
    "ViolationOrNone",
    # Constants
    "OCV_CEILING_V",
    "DUPLICATE_INCLUDED_SLUGS",
    "REMOVAL_FAMILY_SLUGS",
    "POSITIVE_REQUIRED_SLUGS",
    "F_FARADAY",
    "H2_MOLAR_VOLUME_L",
    # Legacy API (deprecated)
    "SystemType",
    "MaterialSpecification",
    "OperatingConditions",
    "SystemConfiguration",
    "ValidationError",
    "GenericSystemValidator",
    "sanitize_string",
    "validate_numeric_range",
    "validate_request",
]
