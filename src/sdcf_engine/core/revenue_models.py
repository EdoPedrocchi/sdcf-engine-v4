"""Fit and simulate the article's three quarterly TTM revenue models.

Bottazzi et al. (2023), Section 2.2, page 68, defines Model 1 for log-revenue
growth and the Gaussian local-level and local-linear-trend Models 2 and 3. It
states the AIC/Kalman-filter/LR selection sequence and use of TTM data. Cordoni
(2021), Section 3.1.2, pages 63-65, provides secondary clarification that the
stationarity target is the log-revenue increment.

V4 fix the undisclosed TTM construction, ADF settings, candidate
ranges, LR boundary, failure policy, and diagnostic software convention.
V4 fixes the independent Gaussian simulation interface. The article says
Monte Carlo sampling and the thesis says bootstrap, but neither discloses the
authors' complete sampler; the documented source ambiguity therefore remains open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from statsmodels.tsa.stattools import adfuller

from sdcf_engine.core.sequential_bootstrap import SequentialPolicy

if TYPE_CHECKING:
    from sdcf_engine.core.state_likelihood import SequentialComparison


RevenueModelName = Literal[
    "model_1_ar_growth",
    "model_2_local_level",
    "model_3_local_trend",
]


@dataclass(frozen=True)
class RevenueCandidate:
    """Diagnostic record for one estimated revenue-model candidate."""

    model_name: str
    valid: bool
    log_likelihood: float | None
    aic: float | None
    parameters: tuple[float, ...]
    converged: bool
    failure_reason: str | None


@dataclass(frozen=True)
class RevenueModelFit:
    """Portable parameters required to simulate one selected revenue model."""

    model_name: RevenueModelName
    current_log_ttm_revenue: float
    coefficients: tuple[float, ...]
    innovation_standard_deviation: float
    growth_lags_most_recent_first: tuple[float, ...]
    filtered_state: tuple[float, ...]
    observation_standard_deviation: float
    level_standard_deviation: float
    trend_standard_deviation: float
    log_likelihood: float
    aic: float
    residuals: tuple[float, ...]
    filtered_state_covariance: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class RevenueSelection:
    """Selected revenue model and pre-selection diagnostic evidence."""

    fit: RevenueModelFit
    adf_statistic: float
    adf_p_value: float
    adf_selected_lag: int
    likelihood_ratio_statistic: float | None
    candidates: tuple[RevenueCandidate, ...]
    sequential_bootstrap: SequentialComparison | None = None


def construct_ttm_revenue(
    quarterly_revenue: pd.Series,
    *,
    fiscal_ordinals: pd.Series | None = None,
) -> pd.Series:
    """Construct log-ready TTM revenue from consecutive quarterly flows.

    Parameters
    ----------
    quarterly_revenue:
        Series indexed by fiscal-period-end timestamps. Values are quarterly
        revenue flows in one common currency and scale.
    fiscal_ordinals:
        Optional validated provider fiscal ordinals aligned exactly to the
        revenue index. When supplied, they define quarter consecutivity while
        timestamps remain the actual fiscal-period-end dates. Direct low-level
        callers may omit them only when calendar quarters are the fiscal identity.

    Returns
    -------
    pandas.Series
        Four-quarter rolling sums indexed by the latest contributing fiscal
        period.

    Raises
    ------
    ValueError
        If dates are duplicated or non-consecutive, values are missing or
        non-positive, or fewer than four quarters are supplied.

    Notes
    -----
    The article, page 68, and Cordoni (2021), pages 63-64, state that TTM data
    are used. V4 supplies the operational rule not stated by the sources:
    sum four complete firm-specific quarterly flows before taking logarithms,
    and do not silently substitute a vendor TTM field.
    """
    if not isinstance(quarterly_revenue.index, pd.DatetimeIndex):
        raise ValueError("quarterly_revenue must use a DatetimeIndex.")
    # Source-file order has no temporal meaning; sorting makes the completeness
    # test and rolling TTM window depend only on fiscal dates.
    ordered = quarterly_revenue.sort_index()
    if ordered.index.has_duplicates:
        raise ValueError("quarterly_revenue contains duplicate fiscal periods.")
    values = pd.to_numeric(ordered, errors="raise").astype(float)
    if values.size < 4:
        raise ValueError("At least four fiscal quarters are required for TTM revenue.")
    if values.isna().any() or not np.all(np.isfinite(values.to_numpy())):
        raise ValueError("quarterly_revenue must be complete and finite.")

    date_index = pd.DatetimeIndex(ordered.index)
    if fiscal_ordinals is None:
        # Legacy diagnostics do not carry provider fiscal identity.
        ordinal = np.asarray(date_index.year * 4 + date_index.quarter, dtype=np.int64)
    else:
        if not isinstance(fiscal_ordinals.index, pd.DatetimeIndex):
            raise ValueError("fiscal_ordinals must use a DatetimeIndex.")
        if fiscal_ordinals.index.has_duplicates:
            raise ValueError("fiscal_ordinals contains duplicate period dates.")
        aligned = fiscal_ordinals.sort_index()
        if not aligned.index.equals(date_index):
            raise ValueError("fiscal_ordinals must align exactly to quarterly_revenue.")
        try:
            ordinal_values = pd.to_numeric(aligned, errors="raise").to_numpy(
                dtype=float
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("fiscal_ordinals must be numeric.") from exc
        if not np.all(np.isfinite(ordinal_values)) or not np.all(
            ordinal_values == np.floor(ordinal_values)
        ):
            raise ValueError("fiscal_ordinals must be complete finite integers.")
        ordinal = ordinal_values.astype(np.int64)
    if np.any(np.diff(ordinal) != 1):
        raise ValueError(
            "Fiscal-quarter history must be consecutive; interpolation is prohibited."
        )

    # ``min_periods=4`` enforces V4's complete-window rule and prevents a
    # partial first year from being annualized or interpolated.
    ttm = values.rolling(window=4, min_periods=4).sum().dropna()
    if (ttm <= 0.0).any():
        raise ValueError(
            "Four-quarter TTM revenue must be strictly positive before logarithm."
        )
    ttm.name = "ttm_revenue"
    return ttm


def _fit_model_1_candidates(
    log_ttm_revenue: NDArray[np.float64],
    candidate_orders: tuple[int, ...],
) -> tuple[tuple[RevenueCandidate, ...], RevenueModelFit]:
    """Fit Model 1 candidates by common-sample conditional Gaussian likelihood."""
    growth = np.diff(log_ttm_revenue)
    maximum_order = max(candidate_orders)
    if growth.size <= maximum_order + 2:
        raise ValueError("Insufficient log-TTM growth history for AR selection.")
    # Trimming every candidate by the maximum order creates one common target
    # sample, so AIC differences reflect the model rather than sample length.
    target = growth[maximum_order:]
    candidates: list[RevenueCandidate] = []
    fitted: dict[int, RevenueModelFit] = {}

    for order in candidate_orders:
        try:
            if order == 0:
                # Model 1 with p=0 is zero-mean white-noise growth because the
                # paper equation contains no deterministic drift term.
                coefficients = np.empty(0, dtype=np.float64)
                residuals = target.copy()
            else:
                design = np.column_stack(
                    [
                        growth[maximum_order - lag : growth.size - lag]
                        for lag in range(1, order + 1)
                    ]
                )
                coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
                residuals = target - design @ coefficients
            variance = float(np.mean(residuals**2))
            if not np.isfinite(variance) or variance <= 0.0:
                raise ValueError("non-positive innovation variance")
            n_observations = residuals.size
            log_likelihood = float(
                -0.5 * n_observations * (np.log(2.0 * np.pi * variance) + 1.0)
            )
            # The conditional Gaussian AIC counts the AR coefficients and the
            # innovation variance; no intercept parameter is present.
            parameter_count = order + 1
            aic = float(2 * parameter_count - 2.0 * log_likelihood)
            candidate = RevenueCandidate(
                model_name=f"model_1_ar_growth_p{order}",
                valid=True,
                log_likelihood=log_likelihood,
                aic=aic,
                parameters=tuple(float(value) for value in coefficients),
                converged=True,
                failure_reason=None,
            )
            candidates.append(candidate)
            fitted[order] = RevenueModelFit(
                model_name="model_1_ar_growth",
                current_log_ttm_revenue=float(log_ttm_revenue[-1]),
                coefficients=candidate.parameters,
                innovation_standard_deviation=float(np.sqrt(variance)),
                growth_lags_most_recent_first=tuple(
                    float(value) for value in growth[-order:][::-1]
                ),
                filtered_state=(),
                observation_standard_deviation=0.0,
                level_standard_deviation=0.0,
                trend_standard_deviation=0.0,
                log_likelihood=log_likelihood,
                aic=aic,
                residuals=tuple(float(value) for value in residuals),
            )
        except Exception as exc:
            # Preserve a failed order in the selection record rather than
            # substituting a simpler model without evidence.
            candidates.append(
                RevenueCandidate(
                    model_name=f"model_1_ar_growth_p{order}",
                    valid=False,
                    log_likelihood=None,
                    aic=None,
                    parameters=(),
                    converged=False,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                )
            )

    valid = [candidate for candidate in candidates if candidate.valid]
    if not valid:
        raise ValueError("Every Model 1 AR candidate failed.")
    selected = min(
        valid,
        key=lambda candidate: (
            candidate.aic if candidate.aic is not None else float("inf")
        ),
    )
    selected_order = int(selected.model_name.rsplit("p", maxsplit=1)[1])
    return tuple(candidates), fitted[selected_order]


def select_revenue_model(
    ttm_revenue: pd.Series,
    *,
    candidate_ar_orders: tuple[int, ...] = (0, 1, 2, 3, 4),
    adf_significance_level: float = 0.05,
    maximum_adf_lag: int = 4,
    bootstrap_rng: np.random.Generator,
    bootstrap_replicates: int = 9999,
    sequential_policy: SequentialPolicy,
) -> RevenueSelection:
    """Fit and select the paper-compatible revenue model.

    Parameters
    ----------
    ttm_revenue:
        Complete positive TTM revenue series in one currency and scale.
    candidate_ar_orders:
        Model 1 orders compared by common-sample Gaussian AIC.
    adf_significance_level:
        Strict p-value threshold for selecting Model 1.
    maximum_adf_lag:
        Maximum ADF augmentation lag; AIC selects the actual lag.
    bootstrap_rng:
        Independent random stream used for parametric bootstrap replication.
    sequential_policy:
        Predeclared finite-look stopping policy.
    bootstrap_replicates:
        V4 fixes 9,999 as the maximum count.

    Returns
    -------
    RevenueSelection
        Selected portable fit and all selection diagnostics.

    Raises
    ------
    ValueError
        If the input is invalid or a required candidate family fails.

    Notes
    -----
    Bottazzi et al. (2023), Section 2.2, page 68, supplies the three equations
    and general selection sequence. Its stationarity prose is ambiguous with
    its Model 1 equation; Cordoni (2021), pages 64-65, explicitly refers to
    log-revenue increments. V4 therefore tests first-difference log TTM
    revenue. V4 independently fixes the ADF specification, orders zero
    through four, strict 5 percent rule, chi-bar-square boundary, numerical
    tolerance, required-fit failure policy, and software convention.
    """
    values = pd.to_numeric(ttm_revenue, errors="raise").to_numpy(dtype=np.float64)
    if values.ndim != 1 or values.size < 16:
        raise ValueError("Revenue selection requires at least 16 TTM observations.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("TTM revenue must be finite and strictly positive.")
    if candidate_ar_orders != tuple(range(5)):
        raise ValueError("V4 requires candidate AR orders (0, 1, 2, 3, 4).")
    sequential_policy.validate()
    if (
        sequential_policy.looks[-1] != bootstrap_replicates
        or sequential_policy.significance != 0.05
    ):
        raise ValueError("Sequential selection requires a matching bootstrap policy.")

    # V4 takes logs only after four-quarter aggregation and applies the ADF
    # gate to log-TTM growth, the series modeled by Model 1.
    log_ttm = np.log(values)
    growth = np.diff(log_ttm)
    adf = adfuller(
        growth,
        maxlag=min(maximum_adf_lag, max(0, growth.size // 2 - 2)),
        regression="c",
        autolag="AIC",
    )
    adf_statistic = float(adf[0])
    adf_p_value = float(adf[1])
    adf_selected_lag = int(adf[2])

    # Equality does not reject the unit-root null under V4; the threshold is
    # intentionally strict rather than chosen from downstream fit.
    if adf_p_value < adf_significance_level:
        candidates, selected = _fit_model_1_candidates(
            log_ttm,
            candidate_ar_orders,
        )
        return RevenueSelection(
            fit=selected,
            adf_statistic=adf_statistic,
            adf_p_value=adf_p_value,
            adf_selected_lag=adf_selected_lag,
            likelihood_ratio_statistic=None,
            candidates=candidates,
        )

    from sdcf_engine.core.state_likelihood import (
        compare_state_models,
        filter_increment_fit,
    )

    comparison = compare_state_models(
        log_ttm,
        rng=bootstrap_rng,
        replicates=bootstrap_replicates,
        sequential_policy=sequential_policy,
    )
    fits = (comparison.null, comparison.alternative)
    candidates = tuple(
        RevenueCandidate(
            model_name=(
                "model_3_local_trend" if fitted.local_trend else "model_2_local_level"
            ),
            valid=True,
            log_likelihood=fitted.log_likelihood,
            aic=-2 * fitted.log_likelihood + 2 * (4 if fitted.local_trend else 2),
            parameters=(
                fitted.observation_variance,
                fitted.level_variance,
                fitted.slope_variance,
                fitted.initial_slope,
            ),
            converged=True,
            failure_reason=None,
        )
        for fitted in fits
    )
    return RevenueSelection(
        filter_increment_fit(log_ttm, comparison.selected),
        adf_statistic,
        adf_p_value,
        adf_selected_lag,
        comparison.statistic,
        candidates,
        comparison,
    )


def simulate_log_ttm_paths(
    fit: RevenueModelFit,
    *,
    steps: int,
    paths: int,
    rng: np.random.Generator,
    state_rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Simulate quarterly log-TTM revenue paths from one selected fit.

    Parameters
    ----------
    fit:
        Portable selected revenue-model parameters.
    steps:
        Number of future fiscal quarters. V4 fixes 20.
    paths:
        Independent Monte Carlo path count.
    rng:
        Explicit NumPy generator. The caller owns deterministic seed derivation.
    state_rng:
        Independent stream for one filtered latent-state draw per path.

    Returns
    -------
    numpy.ndarray
        Array shaped ``(steps, paths)`` containing log TTM revenue.

    Raises
    ------
    ValueError
        If dimensions are invalid or the selected fit lacks required state.

    Notes
    -----
    The sources disclose the model equations and future revenue sampling, but
    not the state origin, generator, seed, or path count. V4 uses parametric
    Gaussian innovations implied by the equations and draws the filtered latent
    state once per path. PCG64, 20 steps, 50,000 paths, hash-derived streams and
    no variance reduction are implementation choices.
    """
    if steps <= 0 or paths <= 0:
        raise ValueError("Simulation steps and paths must be positive.")

    def draw_innovations(scale: float) -> NDArray[np.float64]:
        """Draw one Gaussian innovation component."""
        return rng.normal(loc=0.0, scale=scale, size=paths)

    output = np.empty((steps, paths), dtype=np.float64)

    if fit.model_name == "model_1_ar_growth":
        current = np.full(paths, fit.current_log_ttm_revenue, dtype=np.float64)
        order = len(fit.coefficients)
        if order:
            if len(fit.growth_lags_most_recent_first) != order:
                raise ValueError("Model 1 growth-lag state does not match AR order.")
            # Each row is one path and columns run from most-recent to oldest
            # growth, matching the coefficient ordering used during fitting.
            lag_state = np.tile(
                np.asarray(fit.growth_lags_most_recent_first, dtype=np.float64),
                (paths, 1),
            )
            coefficients = np.asarray(fit.coefficients, dtype=np.float64)
        for step in range(steps):
            conditional_mean = (
                lag_state @ coefficients if order else np.zeros(paths, dtype=np.float64)
            )
            growth = conditional_mean + draw_innovations(
                fit.innovation_standard_deviation
            )
            current = current + growth
            output[step] = current
            if order:
                # Shift after observing the new innovation so the next quarter
                # conditions on the just-simulated growth without look-ahead.
                lag_state[:, 1:] = lag_state[:, :-1]
                lag_state[:, 0] = growth
        return output

    if not fit.filtered_state:
        raise ValueError("State-space model lacks its terminal filtered state.")
    level = np.full(paths, fit.filtered_state[0], dtype=np.float64)
    trend = np.zeros(paths, dtype=np.float64)
    if fit.model_name == "model_3_local_trend":
        if len(fit.filtered_state) < 2:
            raise ValueError("Local-trend model lacks a filtered slope state.")
        trend.fill(fit.filtered_state[1])

    covariance = np.asarray(fit.filtered_state_covariance, dtype=np.float64)
    dimension = len(fit.filtered_state)
    if covariance.shape != (dimension, dimension) or not np.all(
        np.isfinite(covariance)
    ):
        raise ValueError("Missing or invalid filtered-state covariance.")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
        raise ValueError("Filtered-state covariance must be symmetric.")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(covariance))))
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("Filtered-state covariance is not positive semidefinite.")
    root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
    states = np.asarray(fit.filtered_state)[:, None] + root @ state_rng.normal(
        size=(dimension, paths)
    )
    level = states[0]
    if fit.model_name == "model_3_local_trend":
        trend = states[1]

    for step in range(steps):
        # State innovations move the latent economic process; observation
        # noise affects reported log revenue but is not carried into the state.
        level = level + trend + draw_innovations(fit.level_standard_deviation)
        if fit.model_name == "model_3_local_trend":
            trend = trend + draw_innovations(fit.trend_standard_deviation)
        output[step] = level + draw_innovations(fit.observation_standard_deviation)
    return output
