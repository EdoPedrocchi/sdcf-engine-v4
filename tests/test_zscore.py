"""Unit tests for V4 point-in-time daily z-score alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdcf_engine.core.zscore import (
    align_quarterly_fair_values_to_daily_prices,
    compute_z_score,
)


def _prices() -> pd.DataFrame:
    """Return synthetic trading sessions with explicit UTC closes."""
    dates = pd.to_datetime(
        ["2020-04-01", "2020-04-02", "2020-06-30", "2020-07-01", "2020-07-02"]
    )
    return pd.DataFrame(
        {
            "security_id": "SYNTHETIC",
            "price_date": dates,
            "market_close_at": [
                f"{date.date().isoformat()}T20:00:00Z" for date in dates
            ],
            "price_close": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )


def _summary(
    periods: list[str],
    availability: list[str],
    precision: list[str],
) -> pd.DataFrame:
    """Return valid synthetic quarterly log-fair-value summaries."""
    return pd.DataFrame(
        {
            "security_id": "SYNTHETIC",
            "fiscal_period_end": periods,
            "available_at": availability,
            "availability_precision": precision,
            "mean_log_fair_value": [np.log(10.0)] * len(periods),
            "standard_deviation_log_fair_value": [0.2] * len(periods),
        }
    )


def test_compute_z_score_uses_log_market_price() -> None:
    """Distinguish log-price standardization from a level-price mistake."""
    assert compute_z_score(20.0, np.log(10.0), np.log(2.0)) == pytest.approx(1.0)


def test_date_only_evidence_starts_on_next_observed_session() -> None:
    """Prevent an unknown intraday release time from entering the same close."""
    result = align_quarterly_fair_values_to_daily_prices(
        _summary(["2020-03-31"], ["2020-04-01"], ["date"]),
        _prices(),
    )

    assert result.rows.iloc[0]["price_date"] == pd.Timestamp("2020-04-02")
    assert result.rows.iloc[0]["source_fiscal_period"] == pd.Timestamp("2020-03-31")


def test_timestamp_before_close_enters_same_session() -> None:
    """Allow same-session use only when an exact release precedes the close."""
    result = align_quarterly_fair_values_to_daily_prices(
        _summary(["2020-06-30"], ["2020-07-01T19:00:00Z"], ["timestamp"]),
        _prices(),
    )

    assert result.rows.iloc[0]["price_date"] == pd.Timestamp("2020-07-01")
    assert result.first_effective_at == pd.Timestamp("2020-07-01T20:00:00Z")


def test_timestamp_after_close_starts_on_next_session() -> None:
    """Move an after-close release to the following observed trading day."""
    result = align_quarterly_fair_values_to_daily_prices(
        _summary(["2020-06-30"], ["2020-07-01T21:00:00Z"], ["timestamp"]),
        _prices(),
    )

    assert result.rows.iloc[0]["price_date"] == pd.Timestamp("2020-07-02")


def test_missing_quarter_fails_before_step_alignment() -> None:
    """Never carry a stale distribution across an absent required quarter."""
    summaries = _summary(
        ["2020-03-31", "2020-09-30"],
        ["2020-04-01", "2020-10-01"],
        ["date", "date"],
    )

    with pytest.raises(ValueError, match="consecutive"):
        align_quarterly_fair_values_to_daily_prices(summaries, _prices())


def test_future_distribution_does_not_change_earlier_z_scores() -> None:
    """Prove that future fair-value information cannot perturb earlier closes."""
    first = _summary(["2020-03-31"], ["2020-04-01"], ["date"])
    extended = _summary(
        ["2020-03-31", "2020-06-30"],
        ["2020-04-01", "2020-07-01T19:00:00Z"],
        ["date", "timestamp"],
    )
    before = align_quarterly_fair_values_to_daily_prices(first, _prices()).rows
    after = align_quarterly_fair_values_to_daily_prices(extended, _prices()).rows

    cutoff = pd.Timestamp("2020-06-30")
    pd.testing.assert_frame_equal(
        before.loc[before["price_date"] <= cutoff].reset_index(drop=True),
        after.loc[after["price_date"] <= cutoff].reset_index(drop=True),
    )
