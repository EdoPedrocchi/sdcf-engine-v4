"""Compute and point-in-time align the article's daily mispricing z-score.

Bottazzi et al. (2023), Equation (6), page 70, and Cordoni (2021), Equation
(4.1), page 75, define the log-price/log-fair-value z-score. Neither source
specifies the operational join from quarterly accounting-based distributions
to daily closes. V4 therefore supplies the point-in-time step-alignment
rule as a project no-look-ahead convention.

The module receives validated quarterly fair-value summaries and compatible
adjusted daily prices. It does not estimate fair values, infer missing source
availability, interpolate distributions, or repair a missing quarterly update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

AvailabilityPrecision = Literal["date", "timestamp"]


@dataclass(frozen=True)
class DailyZScorePanel:
    """Validated daily z-score rows and their quarterly lineage."""

    rows: pd.DataFrame
    first_effective_at: pd.Timestamp
    last_effective_at: pd.Timestamp


def compute_z_score(
    market_price_per_share: float,
    mean_log_fair_value: float,
    standard_deviation_log_fair_value: float,
) -> float:
    """Compute the uncertainty-adjusted mispricing indicator.

    Parameters
    ----------
    market_price_per_share:
        Positive adjusted market close on a compatible share basis.
    mean_log_fair_value:
        Monte Carlo mean of natural-log fair value per share.
    standard_deviation_log_fair_value:
        Positive sample standard deviation of natural-log fair value.

    Returns
    -------
    float
        ``(log(price) - mean_log_fair_value) / standard_deviation``.

    Raises
    ------
    ValueError
        If an input is non-finite, price is non-positive, or standard
        deviation is non-positive.

    Notes
    -----
    Implements Bottazzi et al. (2023), Equation (6), page 70, and Cordoni
    (2021), Equation (4.1), page 75. Input validation is an engineering
    guardrail. Market price is intentionally downstream and cannot affect a
    simulated fair-value path.
    """
    values = (
        market_price_per_share,
        mean_log_fair_value,
        standard_deviation_log_fair_value,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Z-score inputs must be finite.")
    if market_price_per_share <= 0.0:
        raise ValueError("market_price_per_share must be strictly positive.")
    if standard_deviation_log_fair_value <= 0.0:
        raise ValueError("standard_deviation_log_fair_value must be strictly positive.")
    return float(
        (np.log(market_price_per_share) - mean_log_fair_value)
        / standard_deviation_log_fair_value
    )


def _require_columns(
    frame: pd.DataFrame,
    required: tuple[str, ...],
    context: str,
) -> pd.DataFrame:
    """Return a copy with required columns or raise an explicit error."""
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{context} lacks columns: {sorted(missing)}.")
    return frame.loc[:, list(required)].copy()


def _quarter_ordinals(values: pd.Series) -> np.ndarray:
    """Return sortable quarter ordinals for fiscal-period dates."""
    dates = pd.DatetimeIndex(values)
    return np.asarray(dates.year * 4 + dates.quarter, dtype=np.int64)


def _validate_quarterly_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate complete consecutive quarterly fair-value summaries."""
    required = (
        "security_id",
        "fiscal_period_end",
        "available_at",
        "availability_precision",
        "mean_log_fair_value",
        "standard_deviation_log_fair_value",
    )
    summaries = _require_columns(frame, required, "fair_value_summaries")
    summaries["fiscal_period_end"] = pd.to_datetime(
        summaries["fiscal_period_end"], errors="raise"
    )
    summaries["available_at"] = pd.to_datetime(
        summaries["available_at"], errors="raise", format="mixed", utc=True
    )
    summaries = summaries.sort_values("fiscal_period_end").reset_index(drop=True)
    if summaries.empty:
        raise ValueError("fair_value_summaries must not be empty.")
    if summaries["fiscal_period_end"].duplicated().any():
        raise ValueError("Fair-value fiscal periods must be unique.")
    if len(set(summaries["security_id"].astype(str))) != 1:
        raise ValueError("Fair-value summaries must describe one security.")
    ordinals = _quarter_ordinals(summaries["fiscal_period_end"])
    if len(ordinals) > 1 and np.any(np.diff(ordinals) != 1):
        raise ValueError(
            "Fair-value summaries must contain every consecutive required quarter."
        )
    allowed_precision = {"date", "timestamp"}
    observed_precision = set(summaries["availability_precision"].astype(str))
    if not observed_precision.issubset(allowed_precision):
        raise ValueError("availability_precision must be 'date' or 'timestamp'.")
    if not summaries["available_at"].is_monotonic_increasing:
        raise ValueError("Quarterly distribution availability must be non-decreasing.")
    means = summaries["mean_log_fair_value"].to_numpy(dtype=np.float64)
    standard_deviations = summaries["standard_deviation_log_fair_value"].to_numpy(
        dtype=np.float64
    )
    if not np.all(np.isfinite(means)):
        raise ValueError("Fair-value means must be finite.")
    if not np.all(np.isfinite(standard_deviations)) or np.any(
        standard_deviations <= 0.0
    ):
        raise ValueError("Fair-value standard deviations must be finite and positive.")
    return summaries


def _validate_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate one-security adjusted daily closes."""
    required = ("security_id", "price_date", "price_close")
    prices = _require_columns(frame, required, "daily_prices")
    if "market_close_at" in frame.columns:
        prices["market_close_at"] = frame["market_close_at"]
    prices["price_date"] = pd.to_datetime(prices["price_date"], errors="raise")
    if "market_close_at" in prices.columns:
        prices["market_close_at"] = pd.to_datetime(
            prices["market_close_at"], errors="raise", format="mixed", utc=True
        )
    prices = prices.sort_values("price_date").reset_index(drop=True)
    if prices.empty:
        raise ValueError("daily_prices must not be empty.")
    if prices["price_date"].duplicated().any():
        raise ValueError("Daily price dates must be unique.")
    if len(set(prices["security_id"].astype(str))) != 1:
        raise ValueError("Daily prices must describe one security.")
    values = prices["price_close"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Daily adjusted prices must be finite and positive.")
    return prices


def _effective_price_row(
    summary: pd.Series,
    prices: pd.DataFrame,
) -> int:
    """Return the first eligible price row under V4."""
    available_at = pd.Timestamp(summary["available_at"])
    precision = str(summary["availability_precision"])
    if precision == "date":
        available_date = (
            available_at.tz_convert(None).normalize()
            if available_at.tzinfo is not None
            else available_at.normalize()
        )
        eligible = prices.index[prices["price_date"] > available_date]
    else:
        if "market_close_at" not in prices.columns:
            raise ValueError(
                "Timestamp availability requires official market_close_at values."
            )
        available_utc = available_at
        if available_utc.tzinfo is None:
            available_utc = available_utc.tz_localize("UTC")
        else:
            available_utc = available_utc.tz_convert("UTC")
        eligible = prices.index[prices["market_close_at"] >= available_utc]
    if len(eligible) == 0:
        raise ValueError(
            "No trading session exists after one valuation distribution became "
            "available."
        )
    return int(eligible[0])


def align_quarterly_fair_values_to_daily_prices(
    fair_value_summaries: pd.DataFrame,
    daily_prices: pd.DataFrame,
) -> DailyZScorePanel:
    """Align complete quarterly distributions to daily closes without look-ahead.

    Date-only inputs become effective on the next observed trading session.
    Timestamped inputs may enter the same close only when the timestamp is no
    later than that close. A missing fiscal quarter raises before any panel is
    returned, so the previous distribution cannot be silently extended.

    Notes
    -----
    The sources compare a daily close with a fair-value distribution at the
    same date but do not disclose this temporal bridge. Same-session cutoffs,
    the next-session rule, step carry-forward, lineage fields, and the ban on
    interpolation/backfill are project decision V4, not author settings.
    """
    summaries = _validate_quarterly_summaries(fair_value_summaries)
    prices = _validate_prices(daily_prices)
    security_id = str(summaries.iloc[0]["security_id"])
    if set(prices["security_id"].astype(str)) != {security_id}:
        raise ValueError("Fair-value summaries and prices identify different firms.")

    effective_rows = [
        _effective_price_row(summary, prices) for _, summary in summaries.iterrows()
    ]
    if len(effective_rows) > 1 and np.any(np.diff(effective_rows) < 0):
        raise ValueError(
            "Quarterly distributions must not become effective out of order."
        )

    # Multiple distributions may already be available before the first price
    # retained for a configured panel, or may become available on the same
    # close. V4 uses the newest eligible distribution at that close, so the
    # last complete fiscal period for each effective session supersedes earlier
    # periods without interpolating or concealing a missing quarter.
    active_positions = [
        position
        for position, effective_row in enumerate(effective_rows)
        if position == len(effective_rows) - 1
        or effective_rows[position + 1] != effective_row
    ]
    active_effective_rows = [effective_rows[position] for position in active_positions]

    output_rows: list[dict[str, object]] = []
    for active_position, summary_position in enumerate(active_positions):
        effective_row = effective_rows[summary_position]
        next_row = (
            active_effective_rows[active_position + 1]
            if active_position + 1 < len(active_effective_rows)
            else len(prices)
        )
        summary = summaries.iloc[summary_position]
        effective_price = prices.iloc[effective_row]
        effective_at = (
            pd.Timestamp(effective_price["market_close_at"])
            if "market_close_at" in prices.columns
            else pd.Timestamp(effective_price["price_date"])
        )
        for price_position in range(effective_row, next_row):
            price = prices.iloc[price_position]
            price_date = pd.Timestamp(price["price_date"])
            available_at = pd.Timestamp(summary["available_at"])
            output_rows.append(
                {
                    "security_id": security_id,
                    "price_date": price_date,
                    "price_close": float(price["price_close"]),
                    "source_fiscal_period": pd.Timestamp(summary["fiscal_period_end"]),
                    "valuation_effective_timestamp": (effective_at),
                    "source_available_at": available_at,
                    "source_availability_precision": str(
                        summary["availability_precision"]
                    ),
                    "valuation_age_days": (
                        price_date.normalize()
                        - (
                            available_at.tz_convert(None).normalize()
                            if available_at.tzinfo is not None
                            else available_at.normalize()
                        )
                    ).days,
                    "mean_log_fair_value": float(summary["mean_log_fair_value"]),
                    "standard_deviation_log_fair_value": float(
                        summary["standard_deviation_log_fair_value"]
                    ),
                    "z_score": compute_z_score(
                        float(price["price_close"]),
                        float(summary["mean_log_fair_value"]),
                        float(summary["standard_deviation_log_fair_value"]),
                    ),
                }
            )
    output = pd.DataFrame(output_rows)
    if output.empty:
        raise ValueError("Point-in-time alignment produced no daily z-scores.")
    if output.duplicated(["security_id", "price_date"]).any():
        raise AssertionError("Point-in-time alignment produced duplicate daily keys.")
    for _, row in output.iterrows():
        available_at = pd.Timestamp(row["source_available_at"])
        effective_at = pd.Timestamp(row["valuation_effective_timestamp"])
        if str(row["source_availability_precision"]) == "date":
            available_date = (
                available_at.tz_convert(None).normalize()
                if available_at.tzinfo is not None
                else available_at.normalize()
            )
            if available_date >= pd.Timestamp(row["price_date"]).normalize():
                raise AssertionError(
                    "Date-only evidence entered before the next trading session."
                )
        else:
            available_utc = (
                available_at.tz_localize("UTC")
                if available_at.tzinfo is None
                else available_at.tz_convert("UTC")
            )
            effective_utc = (
                effective_at.tz_localize("UTC")
                if effective_at.tzinfo is None
                else effective_at.tz_convert("UTC")
            )
            if available_utc > effective_utc:
                raise AssertionError(
                    "Timestamped evidence entered before its eligible market close."
                )
    return DailyZScorePanel(
        rows=output,
        first_effective_at=pd.Timestamp(
            output.iloc[0]["valuation_effective_timestamp"]
        ),
        last_effective_at=pd.Timestamp(
            output.loc[
                output["source_fiscal_period"]
                == summaries.iloc[-1]["fiscal_period_end"],
                "valuation_effective_timestamp",
            ].iloc[0]
        ),
    )
