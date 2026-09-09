"""Unit tests for alpha and beta estimation."""

import numpy as np
import pandas as pd
import pytest

from sdcf_engine.core.margins import (
    build_beta_annual_window,
    build_calendar_bounded_ttm_flows,
    estimate_alpha,
    estimate_beta,
)


def _quarterly_frame(periods: int) -> pd.DataFrame:
    """Return positive synthetic quarterly flows on a fiscal sequence."""
    dates = pd.date_range("1992-03-31", periods=periods, freq="QE-DEC")
    return pd.DataFrame(
        {
            "revenue": np.arange(periods, dtype=float) + 100.0,
            "ebitda": np.arange(periods, dtype=float) + 30.0,
        },
        index=dates,
    )


def test_alpha_ttm_window_excludes_three_structural_warmup_rows() -> None:
    """Treat raw TTM warm-up as inputs, not failed alpha observations."""
    quarterly = _quarterly_frame(66)

    ttm = build_calendar_bounded_ttm_flows(quarterly)

    assert len(ttm) == 63
    assert ttm.index[0] == quarterly.index[3]
    assert ttm.iloc[0]["revenue"] == pytest.approx(quarterly.iloc[:4]["revenue"].sum())


def test_alpha_ttm_window_ignores_gap_and_missingness_before_calendar_bound() -> None:
    """Apply the 66-TTM calendar limit before checking required inputs."""
    relevant = _quarterly_frame(69)
    older = pd.DataFrame(
        {"revenue": [np.nan], "ebitda": [np.nan]},
        index=pd.to_datetime(["1989-03-31"]),
    )

    ttm = build_calendar_bounded_ttm_flows(pd.concat([older, relevant]))

    assert len(ttm) == 66
    assert ttm.index.equals(relevant.index[3:])
    assert not ttm.isna().any(axis=None)


def test_alpha_ttm_window_rejects_gap_inside_calendar_bound() -> None:
    """Keep an in-window missing fiscal quarter fail-closed under V4."""
    quarterly = _quarterly_frame(69).drop(index=pd.Timestamp("2000-03-31"))

    with pytest.raises(ValueError, match="alpha raw window.*consecutive"):
        build_calendar_bounded_ttm_flows(quarterly)


def test_beta_window_checks_only_four_required_annual_ttm_levels() -> None:
    """Ignore older gaps while validating the exact 16 relevant raw quarters."""
    relevant = _quarterly_frame(16)
    older_revenue = pd.Series([np.nan], index=pd.to_datetime(["1989-03-31"]))
    older_working_capital = pd.Series([np.nan], index=pd.to_datetime(["1989-03-31"]))
    revenue = pd.concat([older_revenue, relevant["revenue"]])
    working_capital = pd.concat([older_working_capital, relevant["revenue"] * 0.05])

    annual = build_beta_annual_window(revenue, working_capital)

    assert len(annual) == 4
    annual_dates = pd.DatetimeIndex(annual.index)
    assert np.all(np.diff(annual_dates.year * 4 + annual_dates.quarter) == 4)
    assert estimate_beta(
        annual["working_capital"], annual["ttm_revenue"]
    ) == pytest.approx(np.mean(annual["working_capital"] / annual["ttm_revenue"]))


def test_beta_window_rejects_missing_revenue_inside_required_quarters() -> None:
    """Do not fill a raw revenue needed by one of beta's four TTM levels."""
    relevant = _quarterly_frame(16)
    relevant.loc[relevant.index[5], "revenue"] = np.nan
    annual = build_beta_annual_window(
        relevant["revenue"],
        relevant["revenue"].fillna(100.0) * 0.05,
    )

    with pytest.raises(ValueError, match="finite"):
        estimate_beta(annual["working_capital"], annual["ttm_revenue"])


def test_estimate_alpha_recovers_synthetic_through_origin_margin() -> None:
    """Check that a 66-quarter TTM estimator recovers a known alpha."""
    rng = np.random.default_rng(814)
    revenue = np.linspace(80.0, 180.0, 66)
    cash_flow = 0.22 * revenue + rng.normal(0.0, 0.5, size=revenue.size)

    estimate = estimate_alpha(cash_flow, revenue)

    assert estimate.alpha == pytest.approx(0.22, abs=0.01)
    assert estimate.ma_order in range(5)
    assert len(estimate.candidates) == 5
    assert estimate.uncentered_r_squared > 0.99


def test_estimate_alpha_accepts_later_listing_history() -> None:
    """Allow a firm-specific initial history shorter than the 66-quarter maximum."""
    rng = np.random.default_rng(2032)
    revenue = np.linspace(90.0, 150.0, 32)
    cash_flow = 0.18 * revenue + rng.normal(0.0, 0.2, size=revenue.size)

    estimate = estimate_alpha(
        cash_flow,
        revenue,
        expected_observations=32,
    )

    assert estimate.alpha == pytest.approx(0.18, abs=0.01)


def test_estimate_beta_is_mean_of_four_annual_ratios() -> None:
    """Keep beta distinct from a ratio of aggregate working capital to revenue."""
    revenue = np.linspace(100.0, 160.0, 4)
    ratios = np.linspace(0.04, 0.07, 4)
    working_capital = revenue * ratios

    assert estimate_beta(working_capital, revenue) == pytest.approx(ratios.mean())


def test_estimate_alpha_retains_degenerate_candidate_failure() -> None:
    """Expose zero-variance candidate failure rather than inventing uncertainty."""
    revenue = np.linspace(100.0, 200.0, 20)
    cash_flow = 0.175 * revenue
    manual_alpha = float(np.dot(revenue, cash_flow) / np.dot(revenue, revenue))

    assert manual_alpha == pytest.approx(0.175, rel=1e-12, abs=1e-10)
    with pytest.raises(ValueError, match="Every alpha candidate failed"):
        estimate_alpha(cash_flow, revenue, expected_observations=20)


@pytest.mark.parametrize("observed", [19, 21])
def test_estimate_alpha_rejects_wrong_common_window(observed: int) -> None:
    """Do not resize, pad, or truncate an alpha estimation window."""
    revenue = np.linspace(100.0, 200.0, observed)
    with pytest.raises(ValueError, match="requires 20 observations"):
        estimate_alpha(0.20 * revenue, revenue, expected_observations=20)


def test_estimate_beta_rejects_invalid_window_and_nonpositive_revenue() -> None:
    """Keep the four-observation beta identity fail-closed."""
    with pytest.raises(ValueError, match="requires 4 observations"):
        estimate_beta([4.0, 5.0, 6.0], [100.0, 100.0, 100.0])
    with pytest.raises(ValueError, match="strictly positive"):
        estimate_beta([4.0, 5.0, 6.0, 7.0], [100.0, 100.0, 0.0, 100.0])
