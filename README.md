# MESS-Methods

**Research methodology and experimental design tools for MES research**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/mess-methods.svg)](https://pypi.org/project/mess-methods/)

## Overview

MESS-Methods provides tools for research methodology and experimental design:

- **Protocol Generator** - Generate lab-ready protocols from similar experiments
- **Sample Size Calculator** - Statistical power analysis
- **Reproducibility Checklist** - Materials/methods validation scoring
- **Paper Extractor** - PDF parsing and metrics extraction
- **Data Validator** - Quality checking and error reporting
- **Export Utilities** - CSV, JSON, PDF, BibTeX, Markdown

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

### Data Validation

```python
from mess_methods import DataValidator

validator = DataValidator()

# Validate experimental data
results = validator.validate(
    data=experimental_data,
    schema='mfc_performance'
)

if not results.is_valid:
    print(results.errors)
    print(results.warnings)
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
