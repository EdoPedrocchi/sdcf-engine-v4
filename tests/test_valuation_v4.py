"""Tests for V4's quarterly cash-flow valuation and failure policy."""

import numpy as np
import pytest

from sdcf_engine.core.valuation import (
    ValuationInputs,
    compute_fair_value,
    derive_stream_seed,
)


def _inputs() -> ValuationInputs:
    return ValuationInputs(1_000, 0.2, 0.05, 0.09, 0.09, 0.03, 300, 100, 10, 5, 100)


def test_seed_identity_is_stable() -> None:
    """Keep firm-date streams independent of scheduling order."""
    first = derive_stream_seed(
        master_seed=2023,
        method_id="sdcf_engine_candidate_v4_sequential_quarterly_ttm:test",
        security_id="SYNTHETIC_FIRM",
        valuation_date_iso="2008-12-31",
    )
    repeated = derive_stream_seed(
        master_seed=2023,
        method_id="sdcf_engine_candidate_v4_sequential_quarterly_ttm:test",
        security_id="SYNTHETIC_FIRM",
        valuation_date_iso="2008-12-31",
    )
    assert first == repeated


def test_twenty_quarterly_ttm_nodes_are_required_and_valued() -> None:
    """Exercise the exact V4 horizon, fractional discounting and terminal value."""
    rng = np.random.default_rng(51)
    paths = np.log(1_000) + rng.normal(0, 0.01, size=(20, 50_000))
    result = compute_fair_value(
        paths,
        _inputs(),
        reference_market_price_per_share=20,
        reporting_paths=50_000,
    )
    assert result.valid
    assert result.diagnostic_path_count == 50_000
    assert result.level_quantile_50 is not None
    with pytest.raises(ValueError, match="exactly 20"):
        compute_fair_value(
            paths[:5],
            _inputs(),
            reference_market_price_per_share=20,
            reporting_paths=50_000,
        )


def test_one_nonpositive_path_invalidates_the_distribution() -> None:
    """Do not discard, clip or replace economically invalid simulations."""
    paths = np.full((20, 50_000), np.log(1.0))
    inputs = ValuationInputs(**{**vars(_inputs()), "total_debt": 1_000_000.0})
    result = compute_fair_value(
        paths,
        inputs,
        reference_market_price_per_share=20,
        reporting_paths=50_000,
    )
    assert not result.valid
    assert result.invalid_path_count == 50_000
    assert result.mean_log_fair_value is None


def test_discounting_terminal_value_and_equity_bridge_match_manual_formula() -> None:
    """Verify every deterministic valuation sign and quarterly exponent."""
    paths = np.full((20, 50_000), np.log(1_000.0))
    inputs = ValuationInputs(1_000, 0.2, 0.0, 0.10, 0.10, 0.02, 300, 100, 10, 5, 100)
    result = compute_fair_value(
        paths,
        inputs,
        reference_market_price_per_share=20,
        reporting_paths=50_000,
    )
    cash_flow = 200.0
    explicit = sum(cash_flow / 1.10 ** (quarter / 4) for quarter in range(1, 21))
    terminal = cash_flow * 1.02 / (1.10**5 * (0.10 - 0.02))
    expected_per_share = (explicit + terminal - 300 + 100 - 10 - 5) / 100
    assert result.fair_value_per_share == pytest.approx(expected_per_share)


def test_terminal_wacc_must_exceed_growth() -> None:
    """Reject a zero or negative Gordon-growth denominator."""
    paths = np.full((20, 50_000), np.log(1_000.0))
    inputs = ValuationInputs(1_000, 0.2, 0.0, 0.09, 0.02, 0.02, 0, 0, 0, 0, 100)
    with pytest.raises(ValueError, match="must exceed"):
        compute_fair_value(
            paths,
            inputs,
            reference_market_price_per_share=20,
            reporting_paths=50_000,
        )
