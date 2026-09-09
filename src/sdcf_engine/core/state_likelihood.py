"""Compare article local-level/trend models on a common increment likelihood.

V4 eliminates the unknown initial level by differencing observations. The
initial slope is a deterministic fitted nuisance parameter, zero under the null.
For zero-based increment indices, covariance is Q I + H L + S min(i,j), where
L has diagonal 2 and off-diagonal -1. This is the exact Gaussian marginal law
of the article's equations, not a comparison of differently burned likelihoods.
Variance scale and (under trend) mean slope are profiled analytically. Bootstrap
replicas refit both models; no failed replica is discarded. No vendor data or
paper result enters these calculations.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, minimize_scalar
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.structural import UnobservedComponents

from sdcf_engine.core.revenue_models import RevenueModelFit
from sdcf_engine.core.sequential_bootstrap import SequentialLook, SequentialPolicy

# V4 numerical, not economic, precision: convergence and nested optimum check.
OPTIMIZER_TOLERANCE = 1e-9
NEGATIVE_LR_TOLERANCE = 1e-7


@dataclass(frozen=True)
class IncrementFit:
    """Common-sample variance estimates and their explicit initial-state model."""

    local_trend: bool
    observations: int
    log_likelihood: float
    initial_slope: float
    observation_variance: float
    level_variance: float
    slope_variance: float
    optimizer_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SequentialComparison:
    """State fits and complete stopping evidence without a stopped-sample p-value."""

    null: IncrementFit
    alternative: IncrementFit
    statistic: float
    replicate_statistics: tuple[float, ...]
    policy: SequentialPolicy
    looks: tuple[SequentialLook, ...]

    @property
    def selected(self) -> IncrementFit:
        """Block dependent forecasts when the maximum budget cannot resolve selection."""
        decision = self.looks[-1].decision
        if decision == "unresolved":
            raise UnresolvedBootstrapError(self)
        return self.alternative if decision == "local_trend" else self.null


class UnresolvedBootstrapError(ValueError):
    """Retain inconclusive comparison evidence for callers instead of forcing a model."""

    def __init__(self, comparison: SequentialComparison) -> None:
        """Attach the full result, including all completed statistics and intervals."""
        self.comparison = comparison
        super().__init__(
            "Sequential bootstrap unresolved at its cap; dependent valuation blocked."
        )


def increment_covariance(
    observations: int,
    observation_variance: float,
    level_variance: float,
    slope_variance: float,
) -> NDArray[np.float64]:
    """Return the exact covariance of n successive log-revenue increments.

    Parameters
    ----------
    observations:
        Increment count, one less than the number of level observations.
    observation_variance, level_variance, slope_variance:
        Nonnegative variances in log-revenue units, per quarterly step.

    Returns
    -------
    numpy.ndarray
        Symmetric n-by-n covariance; singular cases are rejected by the fitter.

    Raises
    ------
    ValueError
        If a variance or the dimension is invalid.
    """
    variances = np.asarray([observation_variance, level_variance, slope_variance])
    if observations < 1 or not np.all(np.isfinite(variances)) or np.any(variances < 0):
        raise ValueError(
            "Increment covariance requires positive size and finite nonnegative variances."
        )
    identity = np.eye(observations)
    differenced_noise = (
        2 * identity - np.eye(observations, k=1) - np.eye(observations, k=-1)
    )
    positions = np.arange(observations)
    return (
        level_variance * identity
        + observation_variance * differenced_noise
        + slope_variance * np.minimum.outer(positions, positions)
    )


def fit_increment_model(
    log_revenue: NDArray[np.float64], *, local_trend: bool
) -> IncrementFit:
    """Fit the article's structural model on an identical proper increment law.

    Parameters
    ----------
    log_revenue:
        Complete finite one-dimensional observed log-TTM history, at least 16.
    local_trend:
        Include stochastic slope and estimate its initial constant mean.

    Returns
    -------
    IncrementFit
        Profile maximum-likelihood estimates with all n-1 increments included.

    Raises
    ------
    ValueError
        If observations are degenerate or all deterministic optimizer starts fail.

    Notes
    -----
    The two shape parameters allocate variance to observation noise and slope;
    overall variance is profiled by ML with divisor n. Initial level drops out.
    No arbitrary diffuse constant or observation burn enters this likelihood.
    """
    values = np.asarray(log_revenue, dtype=float)
    if values.ndim != 1 or len(values) < 16 or not np.all(np.isfinite(values)):
        raise ValueError(
            "Increment estimation requires at least 16 finite log revenues."
        )
    increments = np.diff(values)
    n = len(increments)
    numerical_scale = float(np.max(np.abs(increments)))
    if numerical_scale == 0 or float(np.ptp(increments)) <= np.finfo(float).eps:
        raise ValueError("Increment likelihood is degenerate for constant increments.")
    data = increments / numerical_scale
    identity = np.eye(n)
    noise = 2 * identity - np.eye(n, k=1) - np.eye(n, k=-1)
    trend = np.minimum.outer(np.arange(n), np.arange(n)) / n
    ones = np.ones(n)

    def profile(shape: NDArray[np.float64]) -> tuple[float, float, float]:
        """Profile common scale and mean, retaining variance-boundary cases."""
        h, t = float(shape[0]), float(shape[1]) if local_trend else 0.0
        covariance = (1 - t) * ((1 - h) * identity + h * noise) + t * trend
        try:
            factor = cho_factor(covariance, lower=True, check_finite=False)
            slope = 0.0
            if local_trend:
                solved = cho_solve(
                    factor, np.column_stack((data, ones)), check_finite=False
                )
                slope = float(ones @ solved[:, 0] / (ones @ solved[:, 1]))
            residual = data - slope
            variance = float(
                residual @ cho_solve(factor, residual, check_finite=False) / n
            )
            if not np.isfinite(variance) or variance <= 0:
                return float("inf"), slope, variance
            logdet = 2 * float(np.log(np.diag(factor[0])).sum())
            nll = 0.5 * (n * (np.log(2 * np.pi) + 1 + np.log(variance)) + logdet)
            return float(nll), slope, variance
        except np.linalg.LinAlgError:
            return float("inf"), 0.0, 0.0

    def objective(shape: NDArray[np.float64]) -> float:
        """Expose a typed scalar objective to the numerical optimizer."""
        return profile(shape)[0]

    candidates: list[tuple[float, NDArray[np.float64]]] = []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        if not local_trend:
            optimized = minimize_scalar(
                lambda h: profile(np.asarray([h, 0.0]))[0],
                bounds=(0, 1),
                method="bounded",
                options={"xatol": OPTIMIZER_TOLERANCE},
            )
            if not optimized.success:
                raise ValueError(
                    f"Local-level profile optimization failed: {optimized.message}"
                )
            for h in (0.0, float(optimized.x), 1.0):
                shape = np.asarray([h, 0.0])
                candidates.append((profile(shape)[0], shape))
        else:
            # Explicit t=0 optimization includes deterministic nonzero slope;
            # it must not be confused with the zero-slope local-level null.
            boundary = minimize_scalar(
                lambda h: profile(np.asarray([h, 0.0]))[0],
                bounds=(0, 1),
                method="bounded",
                options={"xatol": OPTIMIZER_TOLERANCE},
            )
            if boundary.success:
                for h in (0.0, float(boundary.x), 1.0):
                    shape = np.asarray([h, 0.0])
                    candidates.append((profile(shape)[0], shape))
            # Multiple fixed starts are retries on the same data, not new draws.
            for start in ([0.1, 0.01], [0.8, 0.2], [0.5, 0.7]):
                optimized_trend = minimize(
                    objective,
                    np.asarray(start, dtype=np.float64),
                    method="Nelder-Mead",
                    bounds=((0.0, 1.0), (0.0, 1.0)),
                    options={
                        "xatol": 1e-7,
                        "fatol": OPTIMIZER_TOLERANCE,
                        "maxiter": 500,
                    },
                )
                if optimized_trend.success and np.isfinite(optimized_trend.fun):
                    candidates.append(
                        (float(optimized_trend.fun), np.asarray(optimized_trend.x))
                    )
    finite = [(value, shape) for value, shape in candidates if np.isfinite(value)]
    if not finite:
        raise ValueError("All deterministic profile-likelihood starts failed.")
    nll, shape = min(finite, key=lambda item: item[0])
    _, slope, scale_variance = profile(shape)
    scale_variance *= numerical_scale**2
    h, t = shape
    return IncrementFit(
        local_trend,
        n,
        -nll - n * np.log(numerical_scale),
        slope * numerical_scale,
        float(scale_variance * (1 - t) * h),
        float(scale_variance * (1 - t) * (1 - h)),
        float(scale_variance * t / n),
        tuple(str(record.message) for record in records),
    )


def compare_state_models(
    log_revenue: NDArray[np.float64],
    *,
    rng: np.random.Generator,
    replicates: int = 9999,
    sequential_policy: SequentialPolicy,
) -> SequentialComparison:
    """Calibrate the LR using V4's sequential parametric-null bootstrap.

    Parameters
    ----------
    log_revenue:
        Observed finite log-TTM series.
    rng:
        Explicit purpose-isolated bootstrap stream.
    replicates:
        V4 production count is 9999; smaller counts are synthetic diagnostics.
    sequential_policy:
        Finite-look rule whose final look must match the replicate budget.

    Returns
    -------
    SequentialComparison
        Fits, completed replica statistics and simultaneous interval evidence.

    Raises
    ------
    ValueError
        If any required fit fails; failed draws are never replaced or dropped.
    """
    if isinstance(replicates, bool) or replicates < 1:
        raise ValueError("Bootstrap requires a positive replicate count.")
    sequential_policy.validate()
    if sequential_policy.looks[-1] != replicates:
        raise ValueError("Sequential policy must match the bootstrap cap.")
    null = fit_increment_model(log_revenue, local_trend=False)
    alternative = fit_increment_model(log_revenue, local_trend=True)

    def lr(first: IncrementFit, second: IncrementFit) -> float:
        """Reject optimization defects rather than hiding a negative LR."""
        value = 2 * (second.log_likelihood - first.log_likelihood)
        if value < -NEGATIVE_LR_TOLERANCE:
            raise ValueError(
                f"Common-sample LR is negative: {value}; inspect optimizer."
            )
        return max(0.0, value)

    observed = lr(null, alternative)
    simulated_statistics: list[float] = []
    looks: list[SequentialLook] = []
    exceedances = 0
    for index in range(replicates):
        # Under LL, differences contain a level innovation and two consecutive
        # observation errors. Reintegrating is a likelihood-preserving transform.
        noise = rng.normal(0, np.sqrt(null.observation_variance), null.observations + 1)
        increments = rng.normal(
            0, np.sqrt(null.level_variance), null.observations
        ) + np.diff(noise)
        sample = np.r_[0.0, np.cumsum(increments)]
        try:
            first = fit_increment_model(sample, local_trend=False)
            second = fit_increment_model(sample, local_trend=True)
            simulated_statistics.append(lr(first, second))
        except ValueError as exc:
            raise ValueError(
                f"Bootstrap replica {index} failed; retained completed count={len(simulated_statistics)}; {exc}"
            ) from exc
        exceedances += int(simulated_statistics[-1] >= observed)
        if index + 1 in sequential_policy.looks:
            look = sequential_policy.inspect(index + 1, exceedances)
            looks.append(look)
            if look.decision != "unresolved" or index + 1 == replicates:
                return SequentialComparison(
                    null,
                    alternative,
                    observed,
                    tuple(simulated_statistics),
                    sequential_policy,
                    tuple(looks),
                )
    raise AssertionError("The final sequential look must return a decision record.")


def filter_increment_fit(
    log_revenue: NDArray[np.float64], fit: IncrementFit
) -> RevenueModelFit:
    """Filter the observed levels conditional on common-likelihood parameters.

    Parameters
    ----------
    log_revenue:
        Same log-TTM observations used in estimation.
    fit:
        Estimated increment law, including deterministic initial slope.

    Returns
    -------
    RevenueModelFit
        Last filtered mean and covariance for data-conditional future simulation.

    Raises
    ------
    ValueError
        If filtering yields invalid states or covariance.

    Notes
    -----
    Only the initial level is exact diffuse in both models. The slope, when
    present, starts at its fitted value with zero parameter-conditional variance.
    Subsequent filtered covariance remains part of the predictive distribution.
    """
    dimension = 2 if fit.local_trend else 1
    initialization = Initialization(dimension)
    initialization.set(0, "diffuse")
    if fit.local_trend:
        initialization.set(
            1, "known", constant=[fit.initial_slope], stationary_cov=[[0.0]]
        )
    model = UnobservedComponents(
        log_revenue,
        level=True,
        trend=fit.local_trend,
        irregular=True,
        stochastic_level=True,
        stochastic_trend=fit.local_trend,
        initialization=initialization,
        use_exact_diffuse=True,
    )
    parameters = [fit.observation_variance, fit.level_variance]
    if fit.local_trend:
        parameters.append(fit.slope_variance)
    result = model.smooth(parameters)
    state = np.asarray(result.filtered_state[:, -1], dtype=float)
    covariance = np.asarray(result.filtered_state_cov[:, :, -1], dtype=float)
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(covariance)):
        raise ValueError(
            "Increment-parameter filter returned non-finite state or covariance."
        )
    parameter_count = 4 if fit.local_trend else 2
    return RevenueModelFit(
        "model_3_local_trend" if fit.local_trend else "model_2_local_level",
        float(log_revenue[-1]),
        (),
        0.0,
        (),
        tuple(state),
        np.sqrt(fit.observation_variance),
        np.sqrt(fit.level_variance),
        np.sqrt(fit.slope_variance),
        fit.log_likelihood,
        -2 * fit.log_likelihood + 2 * parameter_count,
        tuple(np.asarray(result.resid, dtype=float)),
        tuple(tuple(float(value) for value in row) for row in covariance),
    )
