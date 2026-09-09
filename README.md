# SDCF Engine V4

This repository contains a compact, vendor-neutral implementation of the
stochastic discounted cash-flow method described by Bottazzi, Cordoni, Livieri,
and Marmi (2023). It exposes one method only: SDCF Engine V4, identified in
outputs and random streams as
`sdcf_engine_candidate_v4_sequential_quarterly_ttm`.

V4 is a controlled clean-room interpretation. The article and related thesis do
not disclose every operational or numerical detail, so this repository does not
claim to reproduce the authors' private code or original data exactly.

## Quick start

Python 3.12 is required. A standard virtual environment is sufficient:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
sdcf-v4 run \
  --fundamentals examples/synthetic_fundamentals.csv \
  --snapshot examples/synthetic_snapshot.yaml \
  --output output/example
```

The command writes `output/example/summary.json`. Add `--save-paths` to retain
the complete fair-value-per-share vector in a compressed NumPy file. Output
directories are never overwritten.

For a fully locked development environment:

```bash
make setup
make demo
make check
```

## What the engine does

For one firm and one fiscal snapshot, the engine:

1. selects only accounting observations available by the information cutoff;
2. reconstructs quarterly operating cash flow and estimates the operating
   margin `alpha` with moving-average errors;
3. estimates working-capital margin `beta` from four complete fiscal years;
4. constructs quarterly trailing-twelve-month revenue and selects one of three
   revenue models;
5. simulates 50,000 twenty-quarter revenue paths with deterministic random
   streams;
6. values 20 quarterly-updated TTM cash-flow nodes, a terminal value, and the
   common-equity bridge; and
7. reports the fair-value distribution and log-price z-score.

The engine does not download, map, repair, impute, clip, or winsorize data. It
expects normalized inputs with explicit fiscal identity and availability.

## Python API

```python
from sdcf_engine import CapitalInputs, V4Method, evaluate_snapshot
```

The public interface consists of `V4Method`, `CapitalInputs`, `SnapshotResult`,
`evaluate_snapshot(...)`, and `compute_daily_z_scores(...)`. The command-line
interface uses the same functions; it is not a separate implementation.

Run the complete synthetic example through the Python API with:

```bash
uv run --locked python examples/run_python_api.py
```

The script loads the documented inputs, constructs `CapitalInputs`, calls
`evaluate_snapshot(...)`, rejects an invalid or unconverged result, and prints
the principal estimates and fair-value quantiles as JSON.

## Reading the result

`summary.json` records input SHA-256 fingerprints, alpha and beta estimates,
the selected revenue model, sequential-bootstrap evidence when applicable,
random-stream seeds, fair-value quantiles, z-score, invalid-path counts, and the
25,000-versus-50,000 convergence check.

A result is suitable for interpretation only when both `valid` and
`convergence_passed` are true. A positive z-score means market price is high
relative to the simulated log fair-value distribution; a negative score means
it is low relative to that distribution.

## Documentation

- [Methodology](docs/methodology.md) follows the complete calculation.
- [Methodological choices](docs/methodological_choices.md) separates published
  instructions from implementation assumptions and gives their advantages and
  limitations.
- [Input contract](docs/input_contract.md) defines every required field, unit,
  sign, and time rule.
- [Verification](docs/verification.md) records tests and equivalence checks
  against the source V4 implementation.

## References

G. Bottazzi, F. Cordoni, G. Livieri, and S. Marmi (2023), “Uncertainty in firm
valuation and a cross-sectional misvaluation measure,” *Annals of Finance*, 19,
63–93. <https://doi.org/10.1007/s10436-022-00423-w>

F. Cordoni (2021), *From Macro to Micro: Causal Inference, Firm Valuation and
Trading Conditions*, PhD thesis, Scuola Normale Superiore, Financial
Mathematics. <https://hdl.handle.net/20.500.14242/166912>

No license is granted by this repository at this stage.
