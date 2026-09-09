"""Estimate the article's operating and working-capital margins.

Bottazzi et al. (2023), Equation (3), pages 66-67, specifies a through-origin
cash-flow regression with moving-average errors, AIC lag selection, and an
initial FQ4 1992-FQ1 2009 calendar. Page 67 describes the annual beta average.
Cordoni (2021), Section 3.1.1, page 63, clarifies quarterly TTM cash flow,
annual working capital, and rolling-window movement.

Exact Gaussian likelihood, candidate orders zero through four, the 20-66
observation boundary, the uncentered R-squared rule, and candidate-failure
semantics are project decisions V4 and V4; they are not attributed to
undisclosed author code. Vendor mapping, unit normalization, and missing-value
treatment remain outside this module.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA


@dataclass(frozen=True)
class AlphaCandidate:
    """Diagnostic record for one candidate MA order."""

    ma_order: int
    valid: bool
    alpha: float | None
    aic: float | None
    innovation_standard_deviation: float | None
    uncentered_r_squared: float | None
    centered_r_squared: float | None
    ks_p_value: float | None
    ljung_box_p_value_lag_1: float | None
    ljung_box_p_value_lag_10: float | None
    converged: bool
    failure_reason: str | None
    monetary_scale: float = 1.0
    innovation_r_squared: float | None = None
    alpha_standard_error: float | None = None
    optimizer_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AlphaEstimate:
    """Selected alpha estimate and complete candidate diagnostics."""

    alpha: float
    ma_order: int
    aic: float
    innovation_standard_deviation: float
    uncentered_r_squared: float
    centered_r_squared: float
    ks_p_value: float
    ljung_box_p_value_lag_1: float
    ljung_box_p_value_lag_10: float
    candidates: tuple[AlphaCandidate, ...]
    monetary_scale: float = 1.0
    innovation_r_squared: float | None = None
    alpha_standard_error: float | None = None


def _fiscal_quarter_ordinals(index: pd.DatetimeIndex) -> NDArray[np.int64]:
    """Return calendar-quarter ordinals used for fiscal-sequence checks."""
    return np.asarray(index.year * 4 + index.quarter, dtype=np.int64)


def _sequence_ordinals(
    index: pd.DatetimeIndex,
    fiscal_ordinals: pd.Series | None,
) -> NDArray[np.int64]:
    """Return provider ordinals when present, otherwise calendar ordinals.

    Provider identity is accepted only when it is a complete one-to-one
    alignment to actual fiscal-period-end dates. This prevents a reindex from
    silently discarding an identity row or attaching it to the wrong quarter.
    """
    if fiscal_ordinals is None:
        return _fiscal_quarter_ordinals(index)
    if not isinstance(fiscal_ordinals.index, pd.DatetimeIndex):
        raise ValueError("fiscal_ordinals must use a DatetimeIndex.")
    if fiscal_ordinals.index.has_duplicates:
        raise ValueError("fiscal_ordinals contains duplicate period dates.")
    aligned = fiscal_ordinals.sort_index()
    if not aligned.index.equals(index):
        raise ValueError("fiscal_ordinals must align exactly to the source index.")
    try:
        values = pd.to_numeric(aligned, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("fiscal_ordinals must be numeric.") from exc
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError("fiscal_ordinals must be complete finite integers.")
    return values.astype(np.int64)


def build_calendar_bounded_ttm_flows(
    quarterly_flows: pd.DataFrame,
    *,
    minimum_observations: int = 20,
    maximum_observations: int = 66,
    fiscal_ordinals: pd.Series | None = None,
) -> pd.DataFrame:
    """Build V4's bounded quarterly-updated TTM estimation frame.

    Parameters
    ----------
    quarterly_flows:
        Quarterly flow columns indexed by fiscal-period-end timestamps. Values
        must already share one currency, scale, and source sign convention.
    minimum_observations:
        Minimum number of TTM observations permitted for a later-listed firm.
    maximum_observations:
        Maximum number of TTM observations in the rolling alpha window.
    fiscal_ordinals:
        Optional provider fiscal sequence aligned to the actual-date index.
        Direct low-level callers may omit it for calendar-quarter fiscal years.

    Returns
    -------
    pandas.DataFrame
        Between 20 and 66 TTM flow rows. The first three raw quarters are used
        only as TTM warm-up and are not returned as incomplete observations.

    Raises
    ------
    ValueError
        If the index is invalid, the relevant raw window has a fiscal-quarter
        gap, or fewer than the configured minimum TTM dates can be formed.

    Notes
    -----
    V4 define a maximum 66-observation calendar window and permit a
    firm's shorter complete history. A 66-observation TTM window can require
    69 raw quarters: three warm-up quarters plus 66 estimation dates. The
    calendar bound is therefore applied before gap and completeness checks,
    so an older gap cannot invalidate a window it does not enter. Missing
    values inside the returned frame remain missing for the caller's explicit
    fail-closed validation; they are never dropped or filled here.
    """
    if not isinstance(quarterly_flows.index, pd.DatetimeIndex):
        raise ValueError("quarterly_flows must use a DatetimeIndex.")
    if quarterly_flows.shape[1] == 0:
        raise ValueError("quarterly_flows must contain at least one flow column.")
    if not 20 <= minimum_observations <= maximum_observations <= 66:
        raise ValueError(
            "Alpha TTM observation bounds must satisfy 20 <= min <= max <= 66."
        )

    ordered = quarterly_flows.sort_index()
    if ordered.empty:
        raise ValueError("No quarterly flow history is available.")
    values = ordered.apply(pd.to_numeric, errors="raise").astype(float)
    index = pd.DatetimeIndex(values.index)
    ordinals = _sequence_ordinals(index, fiscal_ordinals)
    last_ordinal = int(ordinals[-1])
    raw_quarters = maximum_observations + 3
    selected = values.iloc[ordinals >= last_ordinal - raw_quarters + 1]
    selected_index = pd.DatetimeIndex(selected.index)
    selected_ordinals = ordinals[
        np.flatnonzero(ordinals >= last_ordinal - raw_quarters + 1)
    ]
    if selected_index.has_duplicates or (
        len(selected_ordinals) > 1 and np.any(np.diff(selected_ordinals) != 1)
    ):
        raise ValueError(
            "V4 alpha raw window must contain consecutive fiscal quarters."
        )

    # The first three raw rows are inputs to the first TTM observation, not
    # failed observations in the 20--66-row estimation sample.
    ttm = selected.rolling(window=4, min_periods=4).sum().iloc[3:]
    if len(ttm) < minimum_observations:
        raise ValueError(
            "V4 requires at least "
            f"{minimum_observations} firm-specific quarterly TTM observations "
            f"inside the alpha calendar window; observed {len(ttm)}."
        )
    if len(ttm) > maximum_observations:
        raise AssertionError("The bounded alpha TTM window exceeded its maximum.")
    return cast(pd.DataFrame, ttm)


def build_beta_annual_window(
    quarterly_revenue: pd.Series,
    quarterly_working_capital: pd.Series,
    *,
    observations: int = 4,
    fiscal_ordinals: pd.Series | None = None,
) -> pd.DataFrame:
    """Construct the four exact annual inputs required by V4 beta.

    Parameters
    ----------
    quarterly_revenue:
        Quarterly revenue flows indexed by fiscal-period end.
    quarterly_working_capital:
        Working-capital levels on the same fiscal-quarter index and monetary
        scale as revenue.
    observations:
        Annual level count. V4 fixes four observations over three years.
    fiscal_ordinals:
        Optional provider fiscal sequence aligned to the actual-date index.
        It governs consecutivity and annual spacing when supplied.

    Returns
    -------
    pandas.DataFrame
        Four rows one fiscal year apart with ``working_capital`` and
        ``ttm_revenue`` columns.

    Raises
    ------
    ValueError
        If the two indexes differ or the 16 relevant raw quarters needed to
        form four annual TTM levels are incomplete or nonconsecutive.

    Notes
    -----
    The four TTM revenue levels require four disjoint four-quarter blocks, or
    16 relevant raw revenue observations. Older dates and gaps cannot affect
    those four ratios and are intentionally outside this validation boundary.
    No missing value is filled, interpolated, or replaced.
    """
    if observations != 4:
        raise ValueError("V4 beta requires exactly 4 annual observations.")
    if not isinstance(quarterly_revenue.index, pd.DatetimeIndex) or not isinstance(
        quarterly_working_capital.index, pd.DatetimeIndex
    ):
        raise ValueError("Beta quarterly inputs must use DatetimeIndex objects.")
    revenue = pd.to_numeric(quarterly_revenue.sort_index(), errors="raise").astype(
        float
    )
    working_capital = pd.to_numeric(
        quarterly_working_capital.sort_index(), errors="raise"
    ).astype(float)
    if not revenue.index.equals(working_capital.index):
        raise ValueError("Beta revenue and working-capital indexes must match exactly.")
    if revenue.empty:
        raise ValueError("No quarterly beta history is available.")

    index = pd.DatetimeIndex(revenue.index)
    ordinals = _sequence_ordinals(index, fiscal_ordinals)
    required_raw_quarters = observations * 4
    last_ordinal = int(ordinals[-1])
    relevant_mask = ordinals >= last_ordinal - required_raw_quarters + 1
    relevant_positions = np.flatnonzero(relevant_mask)
    relevant_revenue = revenue.iloc[relevant_positions]
    relevant_working_capital = working_capital.iloc[relevant_positions]
    relevant_index = pd.DatetimeIndex(relevant_revenue.index)
    relevant_ordinals = ordinals[relevant_positions]
    if len(relevant_revenue) != required_raw_quarters or (
        relevant_index.has_duplicates or np.any(np.diff(relevant_ordinals) != 1)
    ):
        raise ValueError(
            "Beta requires 16 consecutive raw fiscal quarters to form four "
            "exact annual TTM observations."
        )

    ttm_revenue = relevant_revenue.rolling(window=4, min_periods=4).sum()
    annual_positions = np.arange(3, required_raw_quarters, 4, dtype=np.int64)
    annual = pd.DataFrame(
        {
            "working_capital": relevant_working_capital.iloc[
                annual_positions
            ].to_numpy(),
            "ttm_revenue": ttm_revenue.iloc[annual_positions].to_numpy(),
        },
        index=relevant_index.take(annual_positions),
    )
    annual_ordinals = relevant_ordinals[annual_positions]
    if len(annual) != observations or np.any(np.diff(annual_ordinals) != 4):
        raise AssertionError("The beta annual-window construction is inconsistent.")
    return annual


def _as_finite_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return one finite float vector."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector


def _fit_alpha_candidate(
    cash_flow: NDArray[np.float64],
    revenue: NDArray[np.float64],
    ma_order: int,
) -> AlphaCandidate:
    """Fit one normalized exact-likelihood alpha candidate."""
    scale = float(np.max(np.abs(revenue)))
    captured_warnings: tuple[str, ...] = ()
    cash_flow = cash_flow / scale
    revenue = revenue / scale
    try:
        # A constant cash-flow/revenue ratio leaves no stochastic innovation
        # variance to estimate.  Optimizer rounding can otherwise turn the
        # same degenerate input into a tiny positive variance on one platform
        # and an invalid fit on another.  The machine-precision tolerance makes
        # this fail-closed decision portable without affecting empirical noise.
        margins = cash_flow / revenue
        margin_scale = max(1.0, float(np.max(np.abs(margins))))
        if float(np.ptp(margins)) <= (
            32.0 * float(np.finfo(np.float64).eps) * margin_scale
        ):
            raise ValueError("innovation variance is numerically zero")
        # Revenue is supplied as the sole exogenous regressor and ``trend='n'``
        # prevents statsmodels from adding an intercept to the paper's
        # through-origin specification.
        model = ARIMA(
            endog=cash_flow,
            exog=revenue[:, np.newaxis],
            order=(0, 0, ma_order),
            trend="n",
            enforce_invertibility=True,
        )
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            result = model.fit()
        captured_warnings = tuple(str(record.message) for record in warning_records)
        converged = bool(result.mle_retvals.get("converged", False))
        parameter_names = list(result.param_names)
        alpha_index = parameter_names.index("x1")
        alpha = float(np.asarray(result.params)[alpha_index])
        covariance = np.asarray(result.cov_params(), dtype=np.float64)
        residuals = np.asarray(result.resid, dtype=np.float64)
        # Residual tests use standardized one-step-ahead forecast errors, not
        # raw residuals, because the MA likelihood can imply time-varying
        # forecast-error variance during initialization.
        standardized = np.asarray(
            result.filter_results.standardized_forecasts_error[0],
            dtype=np.float64,
        )
        standardized = standardized[np.isfinite(standardized)]
        sigma2_index = parameter_names.index("sigma2")
        sigma2 = float(np.asarray(result.params)[sigma2_index])

        valid = (
            converged
            and math_is_finite_positive(sigma2)
            and math_is_finite(alpha)
            and np.all(np.isfinite(covariance))
            and np.all(np.isfinite(residuals))
            and math_is_finite(float(result.aic))
            and math_is_finite(float(result.llf))
            and standardized.size > 10
        )
        if not valid:
            raise ValueError(
                "non-convergence, non-finite covariance, or invalid innovation variance"
            )

        innovation_sum_squares = float(np.dot(residuals, residuals))
        errors = cash_flow - alpha * revenue
        residual_sum_squares = float(np.dot(errors, errors))
        # V4 uses the through-origin total for eligibility. The centered
        # total is retained beside it only to make the convention auditable.
        uncentered_total = float(np.dot(cash_flow, cash_flow))
        centered_total = float(np.sum((cash_flow - cash_flow.mean()) ** 2))
        if uncentered_total <= 0.0 or centered_total <= 0.0:
            raise ValueError("cash-flow variation is insufficient for R-squared")

        # KS and Ljung-Box results are diagnostics, never extra sample filters.
        ks_p_value = float(stats.kstest(standardized, "norm").pvalue)
        ljung_box = acorr_ljungbox(
            standardized,
            lags=[1, 10],
            return_df=True,
        )
        return AlphaCandidate(
            ma_order=ma_order,
            valid=True,
            alpha=alpha,
            # A change of monetary units contributes n*log(scale) to the
            # negative likelihood; restore original-unit AIC consistently.
            aic=float(result.aic) + 2.0 * cash_flow.size * float(np.log(scale)),
            innovation_standard_deviation=float(np.sqrt(sigma2)) * scale,
            uncentered_r_squared=1.0 - residual_sum_squares / uncentered_total,
            centered_r_squared=1.0 - residual_sum_squares / centered_total,
            ks_p_value=ks_p_value,
            ljung_box_p_value_lag_1=float(ljung_box.loc[1, "lb_pvalue"]),
            ljung_box_p_value_lag_10=float(ljung_box.loc[10, "lb_pvalue"]),
            converged=True,
            failure_reason=None,
            monetary_scale=scale,
            innovation_r_squared=1.0 - innovation_sum_squares / uncentered_total,
            alpha_standard_error=(
                float(np.sqrt(covariance[alpha_index, alpha_index]))
                if covariance[alpha_index, alpha_index] >= 0.0
                else None
            ),
            optimizer_warnings=captured_warnings,
        )
    except Exception as exc:
        # Candidate-level failure is data: V4 requires every attempted
        # order to remain visible instead of silently falling back to q=0.
        return AlphaCandidate(
            ma_order=ma_order,
            valid=False,
            alpha=None,
            aic=None,
            innovation_standard_deviation=None,
            uncentered_r_squared=None,
            centered_r_squared=None,
            ks_p_value=None,
            ljung_box_p_value_lag_1=None,
            ljung_box_p_value_lag_10=None,
            converged=False,
            failure_reason=f"{type(exc).__name__}: {exc}",
            monetary_scale=scale,
            optimizer_warnings=captured_warnings,
        )


def math_is_finite(value: float) -> bool:
    """Return whether one scalar is finite."""
    return bool(np.isfinite(value))


def math_is_finite_positive(value: float) -> bool:
    """Return whether one scalar is finite and strictly positive."""
    return bool(np.isfinite(value) and value > 0.0)


def estimate_alpha(
    cash_flow: ArrayLike,
    revenue: ArrayLike,
    *,
    candidate_ma_orders: tuple[int, ...] = (0, 1, 2, 3, 4),
    expected_observations: int = 66,
) -> AlphaEstimate:
    """Estimate alpha on one calendar-bounded quarterly TTM window.

    Parameters
    ----------
    cash_flow:
        Reconstructed quarterly operating cash flow in one monetary scale.
    revenue:
        Quarterly revenue in the same currency and scale.
    candidate_ma_orders:
        Candidate MA orders compared by AIC. V4 fixes ``0`` through ``4``.
    expected_observations:
        Required common observation count for the firm-date. V4 permits
        20--66 observations inside the fixed initial calendar window.

    Returns
    -------
    AlphaEstimate
        Lowest-AIC valid candidate and diagnostics for every attempted order.

    Raises
    ------
    ValueError
        If inputs violate the common-window contract or every candidate fails.

    Notes
    -----
    The regression is through the origin. The uncentered R-squared is the
    paper-style eligibility statistic; centered R-squared is diagnostic only.

    Source boundary: the article specifies Equation (3), AIC selection, the
    initial calendar, and the 10 percent filter. V4 fix the candidate
    range, likelihood implementation, minimum firm history, common sample,
    exact R-squared convention, threshold equality, and failure behavior.
    """
    cash_flow_vector = _as_finite_vector(cash_flow, "cash_flow")
    revenue_vector = _as_finite_vector(revenue, "revenue")
    if not 20 <= expected_observations <= 66:
        raise ValueError("Alpha expected_observations must lie in [20, 66].")
    if cash_flow_vector.shape != revenue_vector.shape:
        raise ValueError("cash_flow and revenue must have identical shapes.")
    if cash_flow_vector.size != expected_observations:
        raise ValueError(
            f"Alpha estimation requires {expected_observations} observations; "
            f"observed {cash_flow_vector.size}."
        )
    if np.any(revenue_vector <= 0.0):
        raise ValueError("Alpha-estimation revenue must be strictly positive.")
    if candidate_ma_orders != tuple(range(5)):
        raise ValueError("V4 requires candidate MA orders (0, 1, 2, 3, 4).")

    # Every order receives the identical calendar-bounded sample so its AIC is
    # comparable; no candidate may gain observations by using fewer MA lags.
    candidates = tuple(
        _fit_alpha_candidate(cash_flow_vector, revenue_vector, order)
        for order in candidate_ma_orders
    )
    valid = [candidate for candidate in candidates if candidate.valid]
    if not valid:
        reasons = "; ".join(
            f"q={candidate.ma_order}: {candidate.failure_reason}"
            for candidate in candidates
        )
        raise ValueError(f"Every alpha candidate failed: {reasons}")

    selected = min(
        valid,
        key=lambda candidate: (
            candidate.aic if candidate.aic is not None else float("inf")
        ),
    )
    # ``valid`` establishes these optional diagnostic fields as present. The
    # assertions document that internal invariant for the type checker.
    assert selected.alpha is not None
    assert selected.aic is not None
    assert selected.innovation_standard_deviation is not None
    assert selected.uncentered_r_squared is not None
    assert selected.centered_r_squared is not None
    assert selected.ks_p_value is not None
    assert selected.ljung_box_p_value_lag_1 is not None
    assert selected.ljung_box_p_value_lag_10 is not None
    return AlphaEstimate(
        alpha=selected.alpha,
        ma_order=selected.ma_order,
        aic=selected.aic,
        innovation_standard_deviation=selected.innovation_standard_deviation,
        uncentered_r_squared=selected.uncentered_r_squared,
        centered_r_squared=selected.centered_r_squared,
        ks_p_value=selected.ks_p_value,
        ljung_box_p_value_lag_1=selected.ljung_box_p_value_lag_1,
        ljung_box_p_value_lag_10=selected.ljung_box_p_value_lag_10,
        candidates=candidates,
        monetary_scale=selected.monetary_scale,
        innovation_r_squared=selected.innovation_r_squared,
        alpha_standard_error=selected.alpha_standard_error,
    )


def estimate_beta(
    working_capital: ArrayLike,
    revenue: ArrayLike,
    *,
    observations: int = 4,
) -> float:
    """Estimate beta as the mean annual working-capital-to-revenue ratio.

    Parameters
    ----------
    working_capital:
        Four annual working-capital observations spanning three years.
    revenue:
        Corresponding annual or TTM revenues in the same currency and scale.
    observations:
        Required observation count. The inclusive 2006--2009 initial window
        contains four annual levels over a three-year span.

    Returns
    -------
    float
        Arithmetic mean of annual ``working_capital / revenue`` ratios.

    Raises
    ------
    ValueError
        If inputs are non-finite, differ in shape, contain non-positive
        revenue, or do not have the required length.

    Notes
    -----
    Bottazzi et al. (2023), page 67, and Cordoni (2021), page 63, describe an
    annual average over the inclusive 2006-2009 initial period. V4 makes the
    operational interpretation explicit: four annual level ratios span that
    three-year interval, and their arithmetic mean is used rather than a ratio
    of aggregates.
    """
    working_capital_vector = _as_finite_vector(working_capital, "working_capital")
    revenue_vector = _as_finite_vector(revenue, "revenue")
    if working_capital_vector.shape != revenue_vector.shape:
        raise ValueError("working_capital and revenue must have identical shapes.")
    if revenue_vector.size != observations:
        raise ValueError(
            f"Beta estimation requires {observations} observations; "
            f"observed {revenue_vector.size}."
        )
    if np.any(revenue_vector <= 0.0):
        raise ValueError("Beta-estimation revenue must be strictly positive.")
    # Average the four annual ratios themselves. A ratio of aggregate working
    # capital to aggregate revenue would implicitly revenue-weight the years.
    return float(np.mean(working_capital_vector / revenue_vector))
