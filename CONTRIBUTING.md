# Contributing

This repository is maintained as a research inference package. Contributions
should preserve the public output schema and numeric behavior unless a change is
explicitly documented as a model or schema change.

## Development Setup

```bash
./run_sample.sh
pip install -e ".[dev]"
```

## Checks

Run fast tests:

```bash
python -m pytest
```

Run the slow sample parity test when the local checkpoint is available:

```bash
PHILOSOPHER_RUN_SLOW=1 python -m pytest tests/test_sample_parity.py
```

Build the package:

```bash
python -m build
```

## Compatibility Expectations

The default summary CSV schema is part of the public interface. The legacy
`phi_utils` import path and root `philosopher.py` wrapper are retained for
compatibility and should not be removed without a deprecation release.
