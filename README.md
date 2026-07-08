# MESS-Methods

<!-- MIRROR_DISCLOSURE_START -->

> **This repository is a downstream mirror.** Source of truth lives in the
> `messai-ai` monorepo; this mirror is updated on each release. Issues and
> Discussions are welcome here. PRs against this mirror will be redirected — see
> [CONTRIBUTING.md](./CONTRIBUTING.md).
>
> History was reset as part of the 2026 monorepo consolidation. Versions tagged
> before that (e.g. `v0.2.0`) remain accessible as historical refs.

<!-- MIRROR_DISCLOSURE_END -->

**Research methodology and experimental design tools for MES research**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/mess-methods.svg)](https://pypi.org/project/mess-methods/)

## Overview

MESS-Methods provides tools for research methodology and experimental design:

- **ScientificValidator** - Physics-violation rule canon for MES papers (P =
  V·I, CE ≤ 100%, V*cell ≤ V_oc, OCV thermodynamic ceiling, Faraday H₂ ceiling,
  …) — \_the recommended entry point for data-quality checks*
- **Protocol Generator** - Generate lab-ready protocols from similar experiments
- **Sample Size Calculator** - Statistical power analysis
- **Reproducibility Checklist** - Materials/methods validation scoring
- **Paper Extractor** - PDF parsing and metrics extraction
- **Export Utilities** - CSV, JSON, PDF, BibTeX, Markdown

## Scientific Validator

The `ScientificValidator` is the **canonical source of truth** for the
physics-violation rules MESSAI uses to flag papers reporting physically-
impossible values. The same rule canon is consumed by the v2 extractor (for
inline quality checks at extraction time) and by the hunter pipeline (which
produces the public `/hunter` dashboard).

```python
from mess_methods.validation import (
    ScientificValidator,
    Observation,
    PaperContext,
)

obs = [
    Observation(canonical_slug="openCircuitVoltage", paper_id="p1",
                condition_set_id="cs1", raw_value=0.7, si_value=0.7, unit="V"),
    Observation(canonical_slug="cell_voltage", paper_id="p1",
                condition_set_id="cs1", raw_value=0.4, si_value=0.4, unit="V"),
    Observation(canonical_slug="currentDensity", paper_id="p1",
                condition_set_id="cs1", raw_value=5.0, si_value=5.0, unit="A/m²"),
    Observation(canonical_slug="powerDensity", paper_id="p1",
                condition_set_id="cs1", raw_value=10.0, si_value=10.0, unit="W/m²"),
]
ctx = PaperContext(paper_id="p1", system_class="MFC")

violations = ScientificValidator().validate(obs, ctx)
for v in violations:
    print(v.rule_name, v.severity, v.plain_english)
    # → power_identity HIGH "Reported peak power doesn't match V·I …"
```

Each `Violation` carries:

- `rule_name`, `severity` (`HIGH | MED | LOW`), `confidence`
  (`HIGH | MED | LOW | NEEDS_REVIEW`)
- `plain_english` summary suitable for a UI card or PDF supplement
- `chain_of_thought` — an ordered list of computation steps with math and units
- `trace_inputs` / `raw_values_used` — the values that fed the rule, with
  `sourceEpdId` provenance pointers for drill-back
- `predicted`, `observed`, `residual_pct` for quantitative residuals
- `citations` — literature references (see below)

`Violation.to_hunter_json()` returns the same JSON shape used by the hunter
pipeline's `computationTrace`, so existing UI components can render it
unchanged.

**See also:** [cross-package joins](../../docs/data-join-patterns.md) —
`Observation.canonical_slug` uses the mess-parameters parameter-slug vocabulary;
the guide shows how to assemble observations and join across the MESS-\*
datasets.

### Rule canon (10 rules)

| Rule                                    | Check                                                                 | Reference                                |
| --------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------- |
| `power_identity`                        | P_observed ≈ V_cell · I (OCV fallback)                                | Logan-Hamelers 2006 §3; Newman-TA Ch. 22 |
| `ce_bounds` / `ce_out_of_unit_interval` | 0 ≤ CE ≤ 100% (or 0 ≤ CE ≤ 1)                                         | Logan 2008 Ch. 5                         |
| `voltage_ordering`                      | V_cell ≤ V_oc (5% slack)                                              | Newman-TA Ch. 22                         |
| `max_power_ohm`                         | P_peak ≤ V_oc²/(4·R_int)                                              | Newman-TA Ch. 22                         |
| `temperature_out_of_plausible_range`    | -20 °C ≤ T ≤ 100 °C (biological catalyst)                             | Logan 2008 §2                            |
| `removal_out_of_unit_interval`          | 0 ≤ removal ≤ 100% (or [0, 1])                                        | Logan 2008 Ch. 5                         |
| `non_positive`                          | currentDensity, powerDensity > 0                                      | Logan-Hamelers 2006 §3                   |
| `ocv_thermodynamic_ceiling`             | OCV ≤ system-class ceiling                                            | Logan 2008 §2.3                          |
| `within_paper_duplicate`                | Same slug, same conditionSet, < 5× spread (CE/R_int/OCV/EE allowlist) | Logan-Hamelers 2006 §3                   |
| `faraday_h2_ceiling`                    | r_H₂ ≤ I·A/(2F·V) × 22.414 L/mol × 86400 s/d (MEC only)               | Logan 2008 Ch. 9                         |

All numerical thresholds (log-ratio bands, 5% / 30% slacks, system-class OCV
ceilings, the duplicate allowlist) are documented in
`src/validation/scientific_validator.py` and mirror the hunter pipeline
byte-for-byte.

### Citations

- **Logan & Hamelers et al. (2006)** "Microbial Fuel Cells: Methodology and
  Technology", _Environ. Sci. Technol._ 40(17), §3 — reporting conventions (P =
  V·I, areal vs volumetric, V_cell vs V_oc, Coulombic-efficiency definition).
- **Logan, B.E. (2008)** _Microbial Fuel Cells_, Wiley. §2.3 — thermodynamic OCV
  ceiling per system class; Ch. 5 — Coulombic efficiency; Ch. 9 — MEC and
  Faraday-law H₂ bound.
- **Newman & Thomas-Alyea (2004)** _Electrochemical Systems_, 3rd ed., Wiley,
  Ch. 22 — porous-electrode conventions, Thevenin matched-load identity.

### Consumers

- **v2 extractor** — imports `validate_*` functions for inline quality checks at
  extraction time; flagged rows surface in the `consistency_flags` column.
- **Hunter pipeline** (`scripts/hunter/build_hunter_jsons.py`) — produces the
  `/hunter` page's four-column dashboard. The hunter currently re-implements the
  rules; the planned migration imports from this package so there is exactly one
  set of thresholds in the codebase.
- **External consumers** — the public `Messai-io/MESS-Methods` mirror exposes
  the same API.

## Installation

```bash
pip install mess-methods
```

## Features

### Protocol Generation

```python
from mess_methods import ProtocolGenerator

generator = ProtocolGenerator()

# Generate protocol from experiment parameters
protocol = generator.generate(
    system_type='MFC',
    electrode_material='carbon_cloth',
    inoculum='wastewater',
    substrate='acetate',
    target_metric='power_density'
)

print(protocol.steps)
print(protocol.materials_list)
print(protocol.expected_results)
```

### Sample Size Calculator

```python
from mess_methods import SampleSizeCalculator

calc = SampleSizeCalculator()

# Calculate required sample size
n = calc.calculate(
    effect_size=0.5,      # Cohen's d
    alpha=0.05,           # Significance level
    power=0.8,            # Statistical power
    test_type='t-test'    # Two-sample t-test
)

print(f"Required samples per group: {n}")
```

### Reproducibility Scoring

```python
from mess_methods import ReproducibilityChecker

checker = ReproducibilityChecker()

# Score experiment reproducibility
score = checker.score(
    materials_specified=True,
    methods_detailed=True,
    data_available=True,
    code_available=False,
    stats_reported=True
)

print(f"Reproducibility score: {score}/100")
print(checker.recommendations)
```

### PDF Paper Extraction

```python
from mess_methods import PaperExtractor

extractor = PaperExtractor()

# Extract data from research paper
data = extractor.extract('paper.pdf')

print(data.title)
print(data.authors)
print(data.performance_metrics)  # Power density, CE, etc.
print(data.operating_conditions)
```

### Export Utilities

```python
from mess_methods import Exporter

exporter = Exporter()

# Export to multiple formats
exporter.to_csv(data, 'results.csv')
exporter.to_json(data, 'results.json')
exporter.to_bibtex(references, 'refs.bib')
exporter.to_pdf(report, 'report.pdf')
```

## Command Line Interface

```bash
# Generate protocol
mess-methods protocol --type MFC --electrode carbon_cloth

# Extract from PDF
mess-methods extract paper.pdf --output data.json

# Validate data
mess-methods validate data.csv --schema mfc_performance

# Calculate sample size
mess-methods sample-size --effect 0.5 --alpha 0.05 --power 0.8
```

## API Reference

See [API Documentation](docs/API.md) for complete reference.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Links

- [MESSAI Platform](https://messai.io)
- [Documentation](https://docs.messai.io/methods)
- [Examples](examples/)
