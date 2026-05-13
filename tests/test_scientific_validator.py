"""Tests for the MESS-Methods physics-violation rule canon.

Covers:
    1. One known-good observation set (no violations).
    2. One fixture per rule that triggers that rule and only that rule.
    3. Smoke test against the live hunter ``violations.json`` (≥80% of
       the 89 strict-revalidated violations re-detected).

The smoke test is skipped automatically when the live JSON is not
present, so the file can be run from the public mirror without DB
access.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from validation import (
    Observation,
    PaperContext,
    ScientificValidator,
    validate_ce_bounds,
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
from validation.types import SEVERITY_HIGH, SEVERITY_MED, SEVERITY_LOW


# ─────────────────────────────────────────────────────────────────────
# 1. Known-good set
# ─────────────────────────────────────────────────────────────────────


def test_known_good_observation_produces_no_violations() -> None:
    """A realistic MFC operating point: OCV 0.7 V, V_cell 0.4 V, I 5
    A/m², P 2 W/m², CE 30%, R_int 30 Ω, T 25 °C. P matches V·I exactly,
    every value lies within its physical bound."""
    cs = "cs-known-good"
    obs = [
        Observation(
            canonical_slug="openCircuitVoltage", paper_id="p1", condition_set_id=cs,
            parameter_name="Open-Circuit Voltage", raw_value=0.7, si_value=0.7, unit="V",
        ),
        Observation(
            canonical_slug="cell_voltage", paper_id="p1", condition_set_id=cs,
            parameter_name="Cell Voltage", raw_value=0.4, si_value=0.4, unit="V",
        ),
        Observation(
            canonical_slug="currentDensity", paper_id="p1", condition_set_id=cs,
            parameter_name="Current Density", raw_value=5.0, si_value=5.0, unit="A/m²",
        ),
        Observation(
            canonical_slug="powerDensity", paper_id="p1", condition_set_id=cs,
            parameter_name="Peak Power Density", raw_value=2.0, si_value=2.0, unit="W/m²",
        ),
        Observation(
            canonical_slug="coulombic_efficiency", paper_id="p1", condition_set_id=cs,
            parameter_name="CE", raw_value=30.0, si_value=0.3, unit="%",
        ),
        Observation(
            # R_int 0.04 Ω·m² gives P_max = V_oc²/(4·R) = 0.49/0.16 = 3.06 W/m²,
            # comfortably above the reported peak P of 2.0 W/m².
            canonical_slug="internalResistance", paper_id="p1", condition_set_id=cs,
            parameter_name="R_int", raw_value=0.04, si_value=0.04, unit="Ω·m²",
        ),
        Observation(
            canonical_slug="temperature", paper_id="p1", condition_set_id=cs,
            parameter_name="Operating T", raw_value=25.0, si_value=298.15, unit="°C",
        ),
    ]
    ctx = PaperContext(paper_id="p1", system_class="MFC")
    out = ScientificValidator().validate(obs, ctx)
    assert out == [], f"expected no violations, got {[v.rule_name for v in out]}"


# ─────────────────────────────────────────────────────────────────────
# 2a. CE > 100%
# ─────────────────────────────────────────────────────────────────────


def test_ce_bounds_triggers_only_ce_rule() -> None:
    """CE = 212 % > 100 % is a hard physical impossibility."""
    obs = Observation(
        canonical_slug="coulombic_efficiency", paper_id="p", parameter_name="CE",
        raw_value=212.0, si_value=2.12, unit="%",
    )
    v = validate_ce_bounds(obs)
    assert v is not None
    assert v.rule_name == "ce_bounds"
    assert v.severity == SEVERITY_HIGH
    assert v.observed == 212.0
    assert v.predicted == 100.0

    # Master validator with this single observation: only ce_bounds fires.
    out = ScientificValidator().validate([obs])
    assert {x.rule_name for x in out} == {"ce_bounds"}


# ─────────────────────────────────────────────────────────────────────
# 2b. Temperature out of plausible range
# ─────────────────────────────────────────────────────────────────────


def test_temperature_triggers_only_temperature_rule() -> None:
    """T = 700 °C is physically impossible for a biological catalyst."""
    obs = Observation(
        canonical_slug="temperature", paper_id="p", parameter_name="heating temperature",
        raw_value=700.0, si_value=973.15, unit="°C",
    )
    v = validate_temperature_range(obs)
    assert v is not None
    assert v.rule_name == "temperature_out_of_plausible_range"
    out = ScientificValidator().validate([obs])
    assert {x.rule_name for x in out} == {"temperature_out_of_plausible_range"}


# ─────────────────────────────────────────────────────────────────────
# 2c. Removal out of unit interval
# ─────────────────────────────────────────────────────────────────────


def test_removal_unit_interval_triggers_only_removal_rule() -> None:
    """Energy efficiency = 109 % violates the [0, 100] bound."""
    obs = Observation(
        canonical_slug="energy_efficiency", paper_id="p", parameter_name="Energy efficiency",
        raw_value=109.0, si_value=1.09, unit="%",
    )
    v = validate_removal_unit_interval(obs)
    assert v is not None
    assert v.rule_name == "removal_out_of_unit_interval"
    out = ScientificValidator().validate([obs])
    assert {x.rule_name for x in out} == {"removal_out_of_unit_interval"}


# ─────────────────────────────────────────────────────────────────────
# 2d. Non-positive
# ─────────────────────────────────────────────────────────────────────


def test_non_positive_triggers_only_non_positive_rule() -> None:
    """Negative current density (sign-convention error)."""
    obs = Observation(
        canonical_slug="currentDensity", paper_id="p",
        parameter_name="Diffusion Current Density Limit",
        raw_value=-35.0, si_value=-3.5, unit="µA/cm²",
    )
    v = validate_non_positive(obs)
    assert v is not None
    assert v.rule_name == "non_positive"
    assert v.observed == -3.5
    out = ScientificValidator().validate([obs])
    assert {x.rule_name for x in out} == {"non_positive"}


# ─────────────────────────────────────────────────────────────────────
# 2e. OCV thermodynamic ceiling
# ─────────────────────────────────────────────────────────────────────


def test_ocv_thermodynamic_ceiling_triggers_only_ocv_rule() -> None:
    """OCV 1.542 V exceeds the OTHER ceiling of 1.5 V."""
    obs = Observation(
        canonical_slug="openCircuitVoltage", paper_id="p",
        parameter_name="Open Circuit Voltage",
        raw_value=1542.0, si_value=1.542, unit="mV",
    )
    ctx = PaperContext(paper_id="p", system_class="OTHER")
    v = validate_ocv_thermodynamic_ceiling(obs, ctx)
    assert v is not None
    assert v.rule_name == "ocv_thermodynamic_ceiling"
    out = ScientificValidator().validate([obs], ctx)
    assert {x.rule_name for x in out} == {"ocv_thermodynamic_ceiling"}


def test_ocv_thermodynamic_ceiling_uses_system_class_specific_bound() -> None:
    """MFC ceiling = 1.3 V is tighter than the OTHER ceiling of 1.5 V."""
    obs = Observation(
        canonical_slug="openCircuitVoltage", paper_id="p",
        raw_value=1.4, si_value=1.4, unit="V",
    )
    # MFC ceiling 1.3 V → fires
    assert validate_ocv_thermodynamic_ceiling(
        obs, PaperContext(system_class="MFC"),
    ) is not None
    # OTHER ceiling 1.5 V → does not fire
    assert validate_ocv_thermodynamic_ceiling(
        obs, PaperContext(system_class="OTHER"),
    ) is None


# ─────────────────────────────────────────────────────────────────────
# 2f. Power identity (P ≠ V·I) — uses OCV fallback
# ─────────────────────────────────────────────────────────────────────


def test_power_identity_with_ocv_fallback() -> None:
    """OCV 0.092 V × 0.065 A/m² = 0.006 W/m² but reported P = 1.1 W/m²
    — physically impossible for a single operating point."""
    cs = "cs-pi"
    ocv = Observation(
        canonical_slug="openCircuitVoltage", paper_id="p", condition_set_id=cs,
        parameter_name="Stable OCV",
        raw_value=92.0, si_value=0.092, unit="mV",
    )
    i = Observation(
        canonical_slug="currentDensity", paper_id="p", condition_set_id=cs,
        parameter_name="Current Density",
        raw_value=6.5, si_value=0.065, unit="µA/cm²",
    )
    p = Observation(
        canonical_slug="powerDensity", paper_id="p", condition_set_id=cs,
        parameter_name="Peak Power Density",
        raw_value=1100.0, si_value=1.1, unit="mW/m²",
    )
    bucket = [ocv, i, p]
    v = validate_power_identity(p, bucket)
    assert v is not None
    assert v.rule_name == "power_identity"
    assert v.observed == pytest.approx(1.1)
    # predicted = V_oc × I = 0.092 × 0.065 = 0.00598
    assert v.predicted == pytest.approx(0.00598, rel=1e-3)


def test_power_identity_close_match_does_not_fire() -> None:
    """V_cell = 0.4 V, I = 5 A/m², predicted P = 2.0 W/m². Observed
    P = 2.1 W/m² is well within the 2× pass band."""
    cs = "cs-pi-pass"
    v = Observation(
        canonical_slug="cell_voltage", paper_id="p", condition_set_id=cs,
        raw_value=0.4, si_value=0.4, unit="V",
    )
    i = Observation(
        canonical_slug="currentDensity", paper_id="p", condition_set_id=cs,
        raw_value=5.0, si_value=5.0, unit="A/m²",
    )
    p = Observation(
        canonical_slug="powerDensity", paper_id="p", condition_set_id=cs,
        raw_value=2.1, si_value=2.1, unit="W/m²",
    )
    assert validate_power_identity(p, [v, i, p]) is None


# ─────────────────────────────────────────────────────────────────────
# 2g. Voltage ordering
# ─────────────────────────────────────────────────────────────────────


def test_voltage_ordering_triggers_only_voltage_rule() -> None:
    """V_cell = 1.0 V > V_oc = 0.77 V violates the Thevenin bound."""
    cs = "cs-vo"
    ocv = Observation(
        canonical_slug="openCircuitVoltage", paper_id="p", condition_set_id=cs,
        parameter_name="OCV", raw_value=0.77, si_value=0.77, unit="V",
    )
    cell = Observation(
        canonical_slug="voltage", paper_id="p", condition_set_id=cs,
        parameter_name="voltage", raw_value=1.0, si_value=1.0, unit="V",
    )
    bucket = [ocv, cell]
    v = validate_voltage_ordering(cell, bucket)
    assert v is not None
    assert v.rule_name == "voltage_ordering"
    out = ScientificValidator().validate([ocv, cell])
    # power_identity won't fire (no powerDensity), and OCV is below 1.5 V.
    assert {x.rule_name for x in out} == {"voltage_ordering"}


# ─────────────────────────────────────────────────────────────────────
# 2h. Max power ohm (Thevenin matched-load ceiling)
# ─────────────────────────────────────────────────────────────────────


def test_max_power_ohm_triggers() -> None:
    """V_oc 0.7 V, R_int 1 Ω → P_max = 0.7²/(4·1) = 0.1225 W/m². A
    reported peak P of 10 W/m² breaches the ceiling 80×."""
    cs = "cs-mpo"
    ocv = Observation(
        canonical_slug="openCircuitVoltage", paper_id="p", condition_set_id=cs,
        raw_value=0.7, si_value=0.7, unit="V",
    )
    rint = Observation(
        canonical_slug="internalResistance", paper_id="p", condition_set_id=cs,
        raw_value=1.0, si_value=1.0, unit="Ω",
    )
    p = Observation(
        canonical_slug="powerDensity", paper_id="p", condition_set_id=cs,
        raw_value=10.0, si_value=10.0, unit="W/m²",
    )
    bucket = [ocv, rint, p]
    v = validate_max_power_ohm(p, bucket)
    assert v is not None
    assert v.rule_name == "max_power_ohm"
    assert v.predicted == pytest.approx(0.1225, rel=1e-3)


# ─────────────────────────────────────────────────────────────────────
# 2i. Within-paper duplicate
# ─────────────────────────────────────────────────────────────────────


def test_within_paper_duplicate_triggers() -> None:
    """Two OCV readings in the same conditionSet, 0.46 V vs 128 V —
    278× spread, almost certainly a unit error."""
    cs = "cs-dup"
    low = Observation(
        canonical_slug="openCircuitVoltage", paper_id="p", condition_set_id=cs,
        parameter_name="Open-Circuit Voltage", raw_value=0.46, si_value=0.46, unit="V",
        source_epd_id="low-epd",
    )
    high = Observation(
        canonical_slug="openCircuitVoltage", paper_id="p", condition_set_id=cs,
        parameter_name="Open-Circuit Voltage", raw_value=128.0, si_value=128.0, unit="V",
        source_epd_id="high-epd",
    )
    out = validate_within_paper_duplicate([low, high])
    assert len(out) == 1
    v = out[0]
    assert v.rule_name == "within_paper_duplicate"
    assert v.observed == 128.0
    # geometric mean ~ sqrt(0.46 * 128) ≈ 7.67
    assert v.predicted == pytest.approx(math.sqrt(0.46 * 128.0), rel=1e-3)


def test_within_paper_duplicate_respects_allowlist() -> None:
    """``currentDensity`` is NOT in the duplicate allowlist (papers
    legitimately sweep it on polarisation curves)."""
    cs = "cs-dup"
    a = Observation(
        canonical_slug="currentDensity", paper_id="p", condition_set_id=cs,
        raw_value=0.1, si_value=0.1, unit="A/m²",
    )
    b = Observation(
        canonical_slug="currentDensity", paper_id="p", condition_set_id=cs,
        raw_value=10.0, si_value=10.0, unit="A/m²",
    )
    assert validate_within_paper_duplicate([a, b]) == []


# ─────────────────────────────────────────────────────────────────────
# 2j. Faraday-law H₂ ceiling (MEC only)
# ─────────────────────────────────────────────────────────────────────


def test_faraday_h2_ceiling_triggers_on_mec() -> None:
    """Reproduces the hunter's only Faraday H2 hit on the v2 corpus:
    I = 16.5 A/m², A = 0.06 m², V = 10 L, r_H2 = 1.67 m³/m³/d.

    Theoretical max: 16.5 × 86400 / (2 × 96485) × (0.06 / 0.010)
                    × (22.414/1000) ≈ 0.993 m³/m³/d.
    Reported 1.67 m³/m³/d is 1.68× over the ceiling.
    """
    cs = "cs-h2"
    i = Observation(
        canonical_slug="currentDensity", paper_id="p", condition_set_id=cs,
        parameter_name="Current Density (Cathode)",
        raw_value=16.5, si_value=16.5, unit="A/m²",
    )
    a = Observation(
        canonical_slug="electrode_surface_area", paper_id="p", condition_set_id=cs,
        raw_value=600.0, si_value=0.06, unit="cm²",
    )
    vol = Observation(
        canonical_slug="reactor_volume", paper_id="p", condition_set_id=cs,
        raw_value=10.0, si_value=10.0, unit="L",
    )
    h2 = Observation(
        canonical_slug="specific_h2_production", paper_id="p", condition_set_id=cs,
        parameter_name="Hydrogen Production Rate",
        raw_value=1.67, si_value=1.67, unit="L/L/D",
    )
    ctx = PaperContext(system_class="MEC")
    v = validate_faraday_h2_ceiling([i, a, vol, h2], ctx)
    assert v is not None
    assert v.rule_name == "faraday_h2_ceiling"
    assert v.predicted == pytest.approx(0.9935, rel=1e-2)
    assert v.observed == 1.67


def test_faraday_h2_ceiling_skipped_for_non_mec() -> None:
    """Rule gates on system_class == MEC."""
    cs = "cs-h2"
    obs = [
        Observation(canonical_slug="currentDensity", condition_set_id=cs,
                    raw_value=16.5, si_value=16.5, unit="A/m²"),
        Observation(canonical_slug="electrode_surface_area", condition_set_id=cs,
                    raw_value=600.0, si_value=0.06, unit="cm²"),
        Observation(canonical_slug="reactor_volume", condition_set_id=cs,
                    raw_value=10.0, si_value=10.0, unit="L"),
        Observation(canonical_slug="specific_h2_production", condition_set_id=cs,
                    raw_value=1.67, si_value=1.67, unit="L/L/D"),
    ]
    assert validate_faraday_h2_ceiling(obs, PaperContext(system_class="MFC")) is None


# ─────────────────────────────────────────────────────────────────────
# 3. Smoke test against live hunter violations
# ─────────────────────────────────────────────────────────────────────


HUNTER_VIOLATIONS_JSON = (
    Path(__file__).resolve().parents[3]
    / "apps" / "web" / "public" / "data" / "computed" / "hunter" / "violations.json"
)


def _violation_to_observations(item: dict) -> tuple[list[Observation], PaperContext]:
    """Reconstruct Observation objects from one item in the hunter
    violations JSON.  We use:
      * the SI inputs from ``computationTrace.inputs`` (preferred),
        and back-fill from ``rawValues`` for raw-value-only rules.
    """
    inputs = item.get("computationTrace", {}).get("inputs", {}) or {}
    paper_id = item["paperId"]
    cs_id = item.get("flaggedEpdId") or paper_id  # placeholder grouping
    system_class = item.get("systemClass") or "OTHER"
    ctx = PaperContext(paper_id=paper_id, system_class=system_class)

    obs: list[Observation] = []
    SLUG_MAP = {
        "voltage": "cell_voltage",
        "cell_voltage": "cell_voltage",
        "openCircuitVoltage": "openCircuitVoltage",
        "openCircuitVoltage_low": "openCircuitVoltage",
        "openCircuitVoltage_high": "openCircuitVoltage",
        "coulombic_efficiency_low": "coulombic_efficiency",
        "coulombic_efficiency_high": "coulombic_efficiency",
        "internalResistance_low": "internalResistance",
        "internalResistance_high": "internalResistance",
        "energy_efficiency_low": "energy_efficiency",
        "energy_efficiency_high": "energy_efficiency",
        "currentDensity": "currentDensity",
        "powerDensity": "powerDensity",
        "coulombic_efficiency": "coulombic_efficiency",
        "internalResistance": "internalResistance",
        "temperature": "temperature",
        "removal_efficiency": "energy_efficiency",  # hunter renames; map to allowlist member
        "specific_h2_production": "specific_h2_production",
        "electrode_surface_area": "electrode_surface_area",
        "reactor_volume": "reactor_volume",
    }
    # Track which (slug, conditionSet) pairs we've created — for duplicate
    # rule we need to keep low+high as two separate rows.
    counter = 0
    for key, payload in inputs.items():
        if key == "system_class_ceiling":
            continue  # not an observation
        slug = SLUG_MAP.get(key, key)
        # For *_low/_high we want separate rows in the same conditionSet
        # so the duplicate detector fires.
        if not isinstance(payload, dict):
            continue
        v_si = payload.get("value")
        unit = payload.get("unit") or ""
        meta = payload.get("meta") or ""
        # parse raw from meta when present (format "raw: X UNIT (...)")
        raw_value = None
        raw_unit = unit
        if isinstance(meta, str) and meta.startswith("raw:"):
            try:
                rest = meta[len("raw:"):].strip()
                tok = rest.split()
                raw_value = float(tok[0])
                raw_unit = tok[1] if len(tok) > 1 else unit
            except (ValueError, IndexError):
                raw_value = None
        if raw_value is None:
            raw_value = v_si
            raw_unit = unit
        obs.append(Observation(
            canonical_slug=slug,
            paper_id=paper_id,
            condition_set_id=cs_id,
            parameter_name=slug,
            raw_value=raw_value,
            si_value=v_si if isinstance(v_si, (int, float)) else None,
            unit=raw_unit,
            source_epd_id=payload.get("sourceEpdId"),
        ))
        counter += 1
    # Back-fill raw values for rules that only consume rawValues (CE,
    # removal): if computationTrace.inputs is empty, the rule still has
    # to fire from the rawValues list.
    if not obs:
        for rv in item.get("rawValues", []):
            slug = "coulombic_efficiency" if item["primaryRule"] == "ce_bounds" else "temperature"
            obs.append(Observation(
                canonical_slug=slug,
                paper_id=paper_id, condition_set_id=cs_id,
                parameter_name=rv.get("name"),
                raw_value=rv.get("value"),
                si_value=rv.get("value"),
                unit=rv.get("unit") or "",
            ))
    return obs, ctx


@pytest.mark.skipif(
    not HUNTER_VIOLATIONS_JSON.exists(),
    reason="live hunter violations.json not present (run scripts/hunter/build_hunter_jsons.py)",
)
def test_smoke_against_live_hunter_violations() -> None:
    """Pass each hunter-flagged paper's reconstructed observations
    through the canonical validator and confirm we re-detect at least
    one violation for ≥80% of the entries."""
    data = json.loads(HUNTER_VIOLATIONS_JSON.read_text())
    items = data.get("items", [])
    assert items, "violations.json had no items"

    validator = ScientificValidator()
    rediscovered = 0
    misses: list[tuple[str, str]] = []  # (rule, paperId)
    for item in items:
        obs, ctx = _violation_to_observations(item)
        if not obs:
            misses.append((item["primaryRule"], item["paperId"]))
            continue
        result = validator.validate(obs, ctx)
        if result:
            rediscovered += 1
        else:
            misses.append((item["primaryRule"], item["paperId"]))

    rate = rediscovered / len(items)
    print(f"\nSmoke test: {rediscovered}/{len(items)} ({rate:.0%}) re-detected.")
    if misses:
        # Aggregate by rule for diagnostic output.
        from collections import Counter
        rule_counts = Counter(rule for rule, _ in misses)
        print("Misses by rule:", dict(rule_counts))
    assert rate >= 0.80, f"only {rate:.0%} re-detected (target ≥ 80%)"
