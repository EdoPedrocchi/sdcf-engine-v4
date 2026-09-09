"""Command-line interface for one vendor-neutral SDCF V4 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sdcf_engine.core.method import load_v4_method
from sdcf_engine.engine import CapitalInputs, SnapshotResult, evaluate_snapshot
from sdcf_engine.exceptions import ConfigurationError, DataContractError


def _sha256(path: Path) -> str:
    """Return a lowercase SHA-256 fingerprint."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_fundamentals(path: Path) -> pd.DataFrame:
    """Load typed quarterly fundamentals without imputing or changing values."""
    frame = pd.read_csv(path)
    for column in ("fiscal_period_end", "available_at"):
        if column not in frame:
            raise DataContractError(f"Missing fundamentals column: {column}.")
    frame["fiscal_period_end"] = pd.to_datetime(
        frame["fiscal_period_end"], errors="raise"
    )
    frame["available_at"] = pd.to_datetime(
        frame["available_at"], utc=True, errors="raise"
    )
    return frame


def _load_snapshot(path: Path) -> tuple[dict[str, Any], CapitalInputs]:
    """Load the exact snapshot YAML schema and construct capital inputs."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "security_id",
        "fiscal_period_end",
        "information_cutoff",
        "reference_market_price_per_share",
        "capital",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ConfigurationError(
            "Snapshot YAML requires exactly the documented fields."
        )
    capital_payload = payload["capital"]
    capital_fields = {
        "short_term_wacc_annual_decimal",
        "terminal_wacc_annual_decimal",
        "perpetual_growth_annual_decimal",
        "total_debt",
        "cash_and_short_term_investments",
        "minority_interest",
        "preferred_stock",
        "shares_outstanding",
        "available_at",
    }
    if not isinstance(capital_payload, dict) or set(capital_payload) != capital_fields:
        raise ConfigurationError(
            "Snapshot capital block has missing or unknown fields."
        )
    capital = CapitalInputs(
        short_term_wacc_annual_decimal=float(
            capital_payload["short_term_wacc_annual_decimal"]
        ),
        terminal_wacc_annual_decimal=float(
            capital_payload["terminal_wacc_annual_decimal"]
        ),
        perpetual_growth_annual_decimal=float(
            capital_payload["perpetual_growth_annual_decimal"]
        ),
        total_debt=float(capital_payload["total_debt"]),
        cash_and_short_term_investments=float(
            capital_payload["cash_and_short_term_investments"]
        ),
        minority_interest=float(capital_payload["minority_interest"]),
        preferred_stock=float(capital_payload["preferred_stock"]),
        shares_outstanding=float(capital_payload["shares_outstanding"]),
        available_at=pd.Timestamp(capital_payload["available_at"]),
    )
    return payload, capital


def _optional_number(value: float | None) -> float | None:
    """Convert NumPy-compatible finite scalars while preserving unavailable values."""
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("A reported statistic is non-finite.")
    return number


def _summary(
    result: SnapshotResult,
    *,
    fundamentals_sha256: str,
    snapshot_sha256: str,
    method_sha256: str,
) -> dict[str, Any]:
    """Build the stable human-readable JSON summary schema."""
    comparison = result.revenue.sequential_bootstrap
    return {
        "method_id": result.method_id,
        "security_id": result.security_id,
        "fiscal_period_end": result.fiscal_period_end.date().isoformat(),
        "information_cutoff": result.available_at.isoformat(),
        "input_sha256": {
            "fundamentals": fundamentals_sha256,
            "snapshot": snapshot_sha256,
            "method": method_sha256,
        },
        "alpha": {
            "estimate": result.alpha.alpha,
            "ma_order": result.alpha.ma_order,
            "aic": result.alpha.aic,
            "standard_error": result.alpha.alpha_standard_error,
            "uncentered_r_squared": result.alpha.uncentered_r_squared,
            "eligible": result.eligibility.eligible,
        },
        "beta": {
            "estimate": result.beta.beta,
            "fiscal_years": [
                int(value) for value in result.beta.observations["fiscal_year"]
            ],
            "available_at": result.beta.available_at.isoformat(),
        },
        "revenue_model": {
            "selected": result.revenue.fit.model_name,
            "adf_statistic": result.revenue.adf_statistic,
            "adf_p_value": result.revenue.adf_p_value,
            "adf_selected_lag": result.revenue.adf_selected_lag,
            "likelihood_ratio_statistic": result.revenue.likelihood_ratio_statistic,
            "sequential_looks": (
                []
                if comparison is None
                else [
                    {
                        "replicates": look.replicates,
                        "exceedances": look.exceedances,
                        "lower": look.lower,
                        "upper": look.upper,
                        "decision": look.decision,
                    }
                    for look in comparison.looks
                ]
            ),
        },
        "valuation": {
            "paths": result.valuation.diagnostic_path_count,
            "valid": result.valuation.valid,
            "convergence_passed": result.valuation.convergence_passed,
            "mean_log_fair_value": _optional_number(
                result.valuation.mean_log_fair_value
            ),
            "standard_deviation_log_fair_value": _optional_number(
                result.valuation.standard_deviation_log_fair_value
            ),
            "z_score": _optional_number(result.valuation.z_score),
            "fair_value_per_share_quantiles": {
                "q05": _optional_number(result.valuation.level_quantile_05),
                "q50": _optional_number(result.valuation.level_quantile_50),
                "q95": _optional_number(result.valuation.level_quantile_95),
            },
            "terminal_value_share_of_enterprise_value": _optional_number(
                result.valuation.terminal_value_share_of_enterprise_value
            ),
            "invalid_paths": {
                "count": result.valuation.invalid_path_count,
                "share": result.valuation.invalid_path_share,
                "component": result.valuation.invalid_component,
                "reason": result.valuation.invalid_reason,
            },
            "convergence_differences": result.valuation.convergence_differences,
        },
        "stream_seeds": result.stream_seeds,
    }


def _run(arguments: argparse.Namespace) -> int:
    """Execute one exact V4 snapshot and write a new output directory."""
    fundamentals_path = arguments.fundamentals.resolve(strict=True)
    snapshot_path = arguments.snapshot.resolve(strict=True)
    method_path = arguments.method.resolve(strict=True)
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    method = load_v4_method(method_path)
    payload, capital = _load_snapshot(snapshot_path)
    result = evaluate_snapshot(
        _load_fundamentals(fundamentals_path),
        capital,
        security_id=str(payload["security_id"]),
        fiscal_period_end=pd.Timestamp(payload["fiscal_period_end"]),
        information_cutoff=pd.Timestamp(payload["information_cutoff"]),
        reference_market_price_per_share=float(
            payload["reference_market_price_per_share"]
        ),
        method=method,
    )
    summary = _summary(
        result,
        fundamentals_sha256=_sha256(fundamentals_path),
        snapshot_sha256=_sha256(snapshot_path),
        method_sha256=_sha256(method_path),
    )
    output.mkdir(parents=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if arguments.save_paths:
        np.savez_compressed(
            output / "fair_value_paths.npz",
            fair_value_per_share=result.valuation.fair_value_per_share,
        )
    print(
        f"completed: model={result.revenue.fit.model_name}; "
        f"valid={result.valuation.valid}; "
        f"converged={result.valuation.convergence_passed}; "
        f"summary={output / 'summary.json'}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the documented command-line parser."""
    parser = argparse.ArgumentParser(description="Run SDCF Engine V4.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="value one firm at one fiscal snapshot")
    run.add_argument("--fundamentals", type=Path, required=True)
    run.add_argument("--snapshot", type=Path, required=True)
    run.add_argument("--method", type=Path, default=Path("config/v4.yaml"))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--save-paths", action="store_true")
    run.set_defaults(handler=_run)
    return parser


def main() -> int:
    """Run the selected command and return a process exit status."""
    arguments = build_parser().parse_args()
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
