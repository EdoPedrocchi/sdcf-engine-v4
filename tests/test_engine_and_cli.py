"""End-to-end tests for the public API and command-line interface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from sdcf_engine import CapitalInputs, V4Method, evaluate_snapshot

ROOT = Path(__file__).resolve().parents[1]


def _inputs() -> tuple[pd.DataFrame, dict[str, Any], CapitalInputs]:
    frame = pd.read_csv(ROOT / "examples/synthetic_fundamentals.csv")
    frame["fiscal_period_end"] = pd.to_datetime(frame["fiscal_period_end"])
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    payload = yaml.safe_load((ROOT / "examples/synthetic_snapshot.yaml").read_text())
    capital = payload["capital"]
    return (
        frame,
        payload,
        CapitalInputs(
            float(capital["short_term_wacc_annual_decimal"]),
            float(capital["terminal_wacc_annual_decimal"]),
            float(capital["perpetual_growth_annual_decimal"]),
            float(capital["total_debt"]),
            float(capital["cash_and_short_term_investments"]),
            float(capital["minority_interest"]),
            float(capital["preferred_stock"]),
            float(capital["shares_outstanding"]),
            pd.Timestamp(capital["available_at"]),
        ),
    )


def test_public_api_runs_exact_v4_example() -> None:
    """Run normalized accounting through the full 50,000-path valuation."""
    frame, payload, capital = _inputs()
    result = evaluate_snapshot(
        frame,
        capital,
        security_id=str(payload["security_id"]),
        fiscal_period_end=pd.Timestamp(payload["fiscal_period_end"]),
        information_cutoff=pd.Timestamp(payload["information_cutoff"]),
        reference_market_price_per_share=float(
            payload["reference_market_price_per_share"]
        ),
        method=V4Method(),
    )
    assert result.method_id == V4Method().method_id
    assert result.revenue.fit.model_name == "model_1_ar_growth"
    assert result.valuation.diagnostic_path_count == 50_000
    assert result.valuation.valid


def test_cli_writes_fingerprinted_summary_and_refuses_overwrite(tmp_path: Path) -> None:
    """Expose a reproducible command without silently replacing an earlier run."""
    output = tmp_path / "result"
    command = [
        sys.executable,
        "-m",
        "sdcf_engine.cli",
        "run",
        "--fundamentals",
        str(ROOT / "examples/synthetic_fundamentals.csv"),
        "--snapshot",
        str(ROOT / "examples/synthetic_snapshot.yaml"),
        "--method",
        str(ROOT / "config/v4.yaml"),
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    summary = json.loads((output / "summary.json").read_text())
    assert completed.returncode == 0
    assert summary["method_id"] == V4Method().method_id
    assert summary["input_sha256"]["fundamentals"]
    repeated = subprocess.run(command, check=False, text=True, capture_output=True)
    assert repeated.returncode != 0
    assert "Output already exists" in repeated.stderr


def test_future_capital_inputs_fail_closed() -> None:
    """Prevent information released after the valuation cutoff from entering."""
    frame, payload, capital = _inputs()
    future = CapitalInputs(
        **{**vars(capital), "available_at": pd.Timestamp("2010-01-01", tz="UTC")}
    )
    with pytest.raises(Exception, match="no later than the cutoff"):
        evaluate_snapshot(
            frame,
            future,
            security_id=str(payload["security_id"]),
            fiscal_period_end=pd.Timestamp(payload["fiscal_period_end"]),
            information_cutoff=pd.Timestamp(payload["information_cutoff"]),
            reference_market_price_per_share=float(
                payload["reference_market_price_per_share"]
            ),
            method=V4Method(),
        )
