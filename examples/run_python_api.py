"""Run the complete synthetic SDCF V4 example through the public Python API."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from sdcf_engine import CapitalInputs, V4Method, evaluate_snapshot


def main() -> None:
    """Load the bundled inputs, evaluate one snapshot, and print key results."""
    repository = Path(__file__).resolve().parents[1]
    fundamentals_path = repository / "examples" / "synthetic_fundamentals.csv"
    snapshot_path = repository / "examples" / "synthetic_snapshot.yaml"

    fundamentals = pd.read_csv(fundamentals_path)
    fundamentals["fiscal_period_end"] = pd.to_datetime(
        fundamentals["fiscal_period_end"]
    )
    fundamentals["available_at"] = pd.to_datetime(
        fundamentals["available_at"], utc=True
    )

    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    capital_data = snapshot["capital"]
    capital = CapitalInputs(
        short_term_wacc_annual_decimal=float(
            capital_data["short_term_wacc_annual_decimal"]
        ),
        terminal_wacc_annual_decimal=float(
            capital_data["terminal_wacc_annual_decimal"]
        ),
        perpetual_growth_annual_decimal=float(
            capital_data["perpetual_growth_annual_decimal"]
        ),
        total_debt=float(capital_data["total_debt"]),
        cash_and_short_term_investments=float(
            capital_data["cash_and_short_term_investments"]
        ),
        minority_interest=float(capital_data["minority_interest"]),
        preferred_stock=float(capital_data["preferred_stock"]),
        shares_outstanding=float(capital_data["shares_outstanding"]),
        available_at=pd.Timestamp(capital_data["available_at"]),
    )

    result = evaluate_snapshot(
        fundamentals,
        capital,
        security_id=str(snapshot["security_id"]),
        fiscal_period_end=pd.Timestamp(snapshot["fiscal_period_end"]),
        information_cutoff=pd.Timestamp(snapshot["information_cutoff"]),
        reference_market_price_per_share=float(
            snapshot["reference_market_price_per_share"]
        ),
        method=V4Method(),
    )

    valuation = result.valuation
    if not valuation.valid:
        raise RuntimeError(f"Invalid valuation: {valuation.invalid_reason}")
    if not valuation.convergence_passed:
        raise RuntimeError("The 25,000-versus-50,000 convergence check failed.")

    print(
        json.dumps(
            {
                "method_id": result.method_id,
                "security_id": result.security_id,
                "revenue_model": result.revenue.fit.model_name,
                "alpha": result.alpha.alpha,
                "beta": result.beta.beta,
                "paths": valuation.diagnostic_path_count,
                "valid": valuation.valid,
                "convergence_passed": valuation.convergence_passed,
                "invalid_paths": valuation.invalid_path_count,
                "fair_value_quantiles": {
                    "q05": valuation.level_quantile_05,
                    "q50": valuation.level_quantile_50,
                    "q95": valuation.level_quantile_95,
                },
                "z_score": valuation.z_score,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
