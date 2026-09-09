"""Convert simulated revenues into SDCF enterprise and equity values.

Bottazzi et al. (2023), Equation (1), page 66, defines stochastic enterprise
value; page 69 defines the equity bridge and per-share distribution. Cordoni
(2021), Equation (3.1), page 61, and Section 3.3, page 69, provide secondary
five-year and terminal-value context.

V4 supplies the undisclosed executable grid: 20 quarterly TTM cash-flow
nodes, fractional-year discount exponents, 50,000 PCG64 paths, the first 1,000
publication-facing paths, and nested convergence prefixes. V4 define
strict all-path validity and keep market price downstream. Those are explicit
project decisions, not numerical settings attributed to the authors. The
module receives canonical values and does not judge vendor semantics.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sdcf_engine.core.cash_flows import cash_flow_from_margins


@dataclass(frozen=True)
class ValuationInputs:
    """Canonical deterministic inputs for one firm-date valuation."""

    current_ttm_revenue: float
    alpha: float
    beta: float
    short_term_wacc_annual_decimal: float
    terminal_wacc_annual_decimal: float
    perpetual_growth_annual_decimal: float
    total_debt: float
    cash_and_short_term_investments: float
    minority_interest: float
    preferred_stock: float
    shares_outstanding: float


@dataclass(frozen=True)
class PrefixSummary:
    """Distribution summary for one nested Monte Carlo prefix."""

    paths: int
    valid: bool
    invalid_path_count: int
    invalid_path_share: float
    mean_log_fair_value: float | None
    standard_deviation_log_fair_value: float | None
    log_quantile_05: float | None
    log_quantile_50: float | None
    log_quantile_95: float | None
    z_score: float | None


@dataclass(frozen=True)
class InvalidPathDiagnostics:
    """Typed V4 evidence for all generated and publication-prefix paths."""

    generated_paths: int
    invalid_path_count: int
    invalid_path_share: float
    minimum_fair_value: float
    publication_paths: int
    publication_invalid_path_count: int
    publication_invalid_path_share: float
    affected_component: str | None
    reason: str | None


@dataclass(frozen=True)
class FairValueSummary:
    """Typed publication-facing fair-value distribution summary."""

    paths: int
    valid: bool
    mean_log_fair_value: float | None
    standard_deviation_log_fair_value: float | None
    level_quantile_05: float | None
    level_quantile_50: float | None
    level_quantile_95: float | None
    terminal_value_share_of_enterprise_value: float | None


@dataclass(frozen=True)
class FairValueResult:
    """Complete one-firm-date SDCF valuation and convergence evidence."""

    reporting_paths: int
    reporting_invalid_path_count: int
    reporting_invalid_path_share: float
    reporting_minimum_fair_value: float
    valid: bool
    invalid_path_count: int
    invalid_path_share: float
    minimum_fair_value: float
    invalid_component: str | None
    invalid_reason: str | None
    diagnostic_path_count: int
    diagnostic_invalid_path_count: int
    diagnostic_invalid_path_share: float
    diagnostic_minimum_fair_value: float
    enterprise_value: NDArray[np.float64]
    discounted_terminal_value: NDArray[np.float64]
    equity_value: NDArray[np.float64]
    fair_value_per_share: NDArray[np.float64]
    mean_log_fair_value: float | None
    standard_deviation_log_fair_value: float | None
    z_score: float | None
    level_quantile_05: float | None
    level_quantile_50: float | None
    level_quantile_95: float | None
    terminal_value_share_of_enterprise_value: float | None
    prefixes: tuple[PrefixSummary, ...]
    convergence_passed: bool
    convergence_differences: dict[str, float | None]

    @property
    def convergence_available(self) -> bool:
        """Return whether the fixed V4 25,000/50,000 test can be applied."""
        prefix_counts = {prefix.paths for prefix in self.prefixes}
        return {25_000, 50_000}.issubset(prefix_counts)

    @property
    def invalid_paths(self) -> InvalidPathDiagnostics:
        """Return the authoritative all-path V4 diagnostic structure."""
        return InvalidPathDiagnostics(
            generated_paths=self.diagnostic_path_count,
            invalid_path_count=self.invalid_path_count,
            invalid_path_share=self.invalid_path_share,
            minimum_fair_value=self.minimum_fair_value,
            publication_paths=self.reporting_paths,
            publication_invalid_path_count=self.reporting_invalid_path_count,
            publication_invalid_path_share=self.reporting_invalid_path_share,
            affected_component=self.invalid_component,
            reason=self.invalid_reason,
        )

    @property
    def publication_summary(self) -> FairValueSummary:
        """Return the separate first-1,000 publication-facing summary."""
        return FairValueSummary(
            paths=self.reporting_paths,
            valid=(
                self.reporting_invalid_path_count == 0
                and self.mean_log_fair_value is not None
            ),
            mean_log_fair_value=self.mean_log_fair_value,
            standard_deviation_log_fair_value=(self.standard_deviation_log_fair_value),
            level_quantile_05=self.level_quantile_05,
            level_quantile_50=self.level_quantile_50,
            level_quantile_95=self.level_quantile_95,
            terminal_value_share_of_enterprise_value=(
                self.terminal_value_share_of_enterprise_value
            ),
        )


def derive_stream_seed(
    *,
    master_seed: int,
    method_id: str,
    security_id: str,
    valuation_date_iso: str,
) -> int:
    """Derive the schedule-independent firm-date PCG64 seed from V4.

    Parameters
    ----------
    master_seed:
        Non-negative method-level master seed.
    method_id:
        Stable method-version identifier.
    security_id:
        Stable security identifier used for stream identity.
    valuation_date_iso:
        ISO ``YYYY-MM-DD`` valuation date.

    Returns
    -------
    int
        Unsigned big-endian integer represented by the SHA-256 digest.

    Notes
    -----
    Neither the article nor thesis discloses a generator, seed, or stream
    construction. This hash-based identity is a project reproducibility rule
    that prevents worker scheduling from changing firm-date paths.
    """
    if master_seed < 0:
        raise ValueError("master_seed must be non-negative.")
    components = (str(master_seed), method_id, security_id, valuation_date_iso)
    if any(component == "" for component in components):
        raise ValueError("Seed stream-key components must be non-empty.")
    # A reserved unit separator prevents ambiguous concatenations such as
    # ("12", "3") and ("1", "23") from sharing a stream key.
    payload = "\x1f".join(components).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big")


def _validate_inputs(inputs: ValuationInputs) -> None:
    """Validate deterministic economic preconditions before path valuation."""
    values = tuple(vars(inputs).values())
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Every valuation input must be finite.")
    if inputs.current_ttm_revenue <= 0.0:
        raise ValueError("current_ttm_revenue must be strictly positive.")
    if inputs.shares_outstanding <= 0.0:
        raise ValueError("shares_outstanding must be strictly positive.")
    # The Gordon-growth denominator must be strictly positive; repairing it
    # would silently impose a different terminal-value assumption.
    if inputs.terminal_wacc_annual_decimal <= inputs.perpetual_growth_annual_decimal:
        raise ValueError(
            "terminal_wacc_annual_decimal must exceed perpetual_growth_annual_decimal."
        )
    if inputs.short_term_wacc_annual_decimal <= -1.0:
        raise ValueError("short-term WACC must be greater than -1.")
    if inputs.terminal_wacc_annual_decimal <= -1.0:
        raise ValueError("terminal WACC must be greater than -1.")


def _summarize_prefix(
    fair_value_per_share: NDArray[np.float64],
    market_price_per_share: float,
    paths: int,
) -> PrefixSummary:
    """Summarize one nested prefix under the strict V4 rule."""
    sample = fair_value_per_share[:paths]
    # V4 evaluates the distribution before taking logs. Even one invalid
    # draw invalidates this prefix instead of being dropped or resampled.
    invalid_mask = ~np.isfinite(sample) | (sample <= 0.0)
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        return PrefixSummary(
            paths=paths,
            valid=False,
            invalid_path_count=invalid_count,
            invalid_path_share=invalid_count / paths,
            mean_log_fair_value=None,
            standard_deviation_log_fair_value=None,
            log_quantile_05=None,
            log_quantile_50=None,
            log_quantile_95=None,
            z_score=None,
        )

    log_fair_value = np.log(sample)
    mean = float(np.mean(log_fair_value))
    # V4 fixes the empirical sample standard deviation rather than the
    # population convention used by NumPy's default.
    standard_deviation = float(np.std(log_fair_value, ddof=1))
    quantiles = np.quantile(log_fair_value, [0.05, 0.5, 0.95])
    z_score = (
        float((np.log(market_price_per_share) - mean) / standard_deviation)
        if standard_deviation > 0.0
        else None
    )
    return PrefixSummary(
        paths=paths,
        valid=standard_deviation > 0.0,
        invalid_path_count=0,
        invalid_path_share=0.0,
        mean_log_fair_value=mean,
        standard_deviation_log_fair_value=standard_deviation,
        log_quantile_05=float(quantiles[0]),
        log_quantile_50=float(quantiles[1]),
        log_quantile_95=float(quantiles[2]),
        z_score=z_score,
    )


def _assess_convergence(
    prefixes: tuple[PrefixSummary, ...],
) -> tuple[bool, dict[str, float | None]]:
    """Apply the fixed V4 comparison between 25,000 and 50,000 paths."""
    # Both summaries are nested prefixes of one deterministic stream, which
    # isolates Monte Carlo precision from differences in random draws.
    by_count = {prefix.paths: prefix for prefix in prefixes}
    if not {25_000, 50_000}.issubset(by_count):
        # A reduced-path sensitivity cannot satisfy or fail V4's numerical
        # convergence gate because neither required comparison is available.
        return False, {
            "mean_log_fair_value_absolute_change": None,
            "maximum_log_quantile_absolute_change": None,
            "z_score_absolute_change": None,
            "invalid_path_share_absolute_change": None,
            "log_standard_deviation_relative_change": None,
        }
    low = by_count[25_000]
    high = by_count[50_000]
    if not low.valid or not high.valid:
        # Log-moment convergence is undefined for an invalid V4 prefix; only
        # the invalid-path-share comparison remains meaningful.
        return False, {
            "mean_log_fair_value_absolute_change": None,
            "maximum_log_quantile_absolute_change": None,
            "z_score_absolute_change": None,
            "invalid_path_share_absolute_change": abs(
                high.invalid_path_share - low.invalid_path_share
            ),
            "log_standard_deviation_relative_change": None,
        }

    assert low.mean_log_fair_value is not None
    assert high.mean_log_fair_value is not None
    assert low.standard_deviation_log_fair_value is not None
    assert high.standard_deviation_log_fair_value is not None
    assert low.z_score is not None
    assert high.z_score is not None
    low_quantiles = np.asarray(
        [low.log_quantile_05, low.log_quantile_50, low.log_quantile_95],
        dtype=np.float64,
    )
    high_quantiles = np.asarray(
        [high.log_quantile_05, high.log_quantile_50, high.log_quantile_95],
        dtype=np.float64,
    )
    mean_change = abs(high.mean_log_fair_value - low.mean_log_fair_value)
    quantile_change = float(np.max(np.abs(high_quantiles - low_quantiles)))
    z_score_change = abs(high.z_score - low.z_score)
    invalid_share_change = abs(high.invalid_path_share - low.invalid_path_share)
    standard_deviation_change = (
        abs(
            high.standard_deviation_log_fair_value
            - low.standard_deviation_log_fair_value
        )
        / low.standard_deviation_log_fair_value
    )
    differences: dict[str, float | None] = {
        "mean_log_fair_value_absolute_change": mean_change,
        "maximum_log_quantile_absolute_change": quantile_change,
        "z_score_absolute_change": z_score_change,
        "invalid_path_share_absolute_change": invalid_share_change,
        "log_standard_deviation_relative_change": standard_deviation_change,
    }
    passed = bool(
        mean_change <= 0.005
        and quantile_change <= 0.01
        and z_score_change <= 0.02
        and invalid_share_change <= 0.001
        and standard_deviation_change <= 0.01
    )
    return passed, differences


def compute_fair_value(
    log_ttm_revenue_paths: NDArray[np.float64],
    inputs: ValuationInputs,
    *,
    reference_market_price_per_share: float,
    reporting_paths: int = 1_000,
    prefix_paths: tuple[int, ...] = (1_000, 5_000, 10_000, 25_000, 50_000),
) -> FairValueResult:
    """Compute enterprise, equity, and per-share fair-value distributions.

    Parameters
    ----------
    log_ttm_revenue_paths:
        Quarterly simulated log-TTM revenue shaped ``(20, generated_paths)``.
        V4 fixes 50,000; separately configured sensitivities may use less.
    inputs:
        Canonical firm-date valuation inputs in consistent units.
    reference_market_price_per_share:
        Positive adjusted market close used only for the V4 nested-prefix
        z-score convergence diagnostic. Market price is not a deterministic
        valuation input and does not affect any fair-value path.
    reporting_paths:
        Publication-facing Monte Carlo sample. Figure 1's histogram counts are
        consistent with 1,000 draws; larger nested prefixes remain numerical
        diagnostics and do not silently redefine the reported distribution.
    prefix_paths:
        Nested Monte Carlo prefixes used for convergence evidence. A schedule
        without 25,000 and 50,000 marks V4 convergence unavailable.

    Returns
    -------
    FairValueResult
        Path arrays, strict validity diagnostics, distribution summaries,
        z-score, terminal-value share, and convergence evidence.

    Raises
    ------
    ValueError
        If dimensions or deterministic economic preconditions are invalid.

    Notes
    -----
    The enterprise-value formula is Bottazzi et al. (2023), Equation (1),
    page 66; the bridge and share division are stated on page 69. The 20-node
    quarterly interpretation, no TTM division, fractional exponents,
    path/reporting counts, ``ddof=1``, convergence limits, and any-path invalid
    rule are project decisions V4. The five-node branch is kept
    only as a labelled sensitivity and cannot silently replace the baseline.
    """
    _validate_inputs(inputs)
    if (
        not np.isfinite(reference_market_price_per_share)
        or reference_market_price_per_share <= 0.0
    ):
        raise ValueError(
            "reference_market_price_per_share must be finite and positive."
        )
    paths = np.asarray(log_ttm_revenue_paths, dtype=np.float64)
    if paths.ndim != 2:
        raise ValueError("log_ttm_revenue_paths must be two-dimensional.")
    if paths.shape[0] != 20:
        raise ValueError("V4 requires exactly 20 quarterly forecast observations.")
    if not prefix_paths:
        raise ValueError("prefix_paths must not be empty.")
    if any(count <= 0 for count in prefix_paths) or any(
        right <= left
        for left, right in zip(prefix_paths, prefix_paths[1:], strict=False)
    ):
        raise ValueError("prefix_paths must be positive and strictly increasing.")
    if paths.shape[1] != prefix_paths[-1]:
        raise ValueError(
            "Monte Carlo path count must equal the final convergence prefix."
        )
    if reporting_paths not in prefix_paths:
        raise ValueError("reporting_paths must be one of the nested prefixes.")

    # Do not clip explosive revenue paths. Overflow is retained as non-finite
    # valuation evidence and is classified by the strict invalid-path rule.
    with np.errstate(over="ignore", invalid="ignore"):
        forecast_revenue = np.exp(paths)

    # V4 recognizes every quarterly-updated TTM cash-flow estimate. These TTM
    # amounts are not divided by four; only their discount exponents are quarterly.
    cash_flow_revenue = forecast_revenue
    previous_revenue = np.vstack(
        [
            np.full((1, paths.shape[1]), inputs.current_ttm_revenue),
            forecast_revenue[:-1],
        ]
    )
    discount_years = np.arange(1, paths.shape[0] + 1, dtype=np.float64) / 4.0
    terminal_year = float(paths.shape[0]) / 4.0

    cash_flow = cash_flow_from_margins(
        cash_flow_revenue,
        previous_revenue,
        inputs.alpha,
        inputs.beta,
        allow_nonfinite_paths=True,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        discount_factors = (
            1.0 + inputs.short_term_wacc_annual_decimal
        ) ** discount_years[:, np.newaxis]
        discounted_cash_flow = cash_flow / discount_factors
        terminal_discount = (1.0 + inputs.terminal_wacc_annual_decimal) ** terminal_year
        # Gordon growth is applied to the final forecast cash flow and discounted
        # from the same terminal year used by the explicit forecast grid.
        discounted_terminal_value = (
            cash_flow[-1]
            * (1.0 + inputs.perpetual_growth_annual_decimal)
            / (
                terminal_discount
                * (
                    inputs.terminal_wacc_annual_decimal
                    - inputs.perpetual_growth_annual_decimal
                )
            )
        )
        enterprise_value = discounted_cash_flow.sum(axis=0) + discounted_terminal_value
        # Enterprise value belongs to all capital providers. The bridge removes
        # non-common claims and adds cash available to common shareholders.
        equity_value = (
            enterprise_value
            - inputs.total_debt
            + inputs.cash_and_short_term_investments
            - inputs.minority_interest
            - inputs.preferred_stock
        )
        fair_value_per_share = equity_value / inputs.shares_outstanding

    # The configured prefix defines the publication-facing distribution. The
    # complete generated array remains available for tail diagnostics; V4's
    # 1,000-path sensitivity deliberately has no larger convergence prefixes.
    reporting_sample = fair_value_per_share[:reporting_paths]
    invalid_mask = ~np.isfinite(reporting_sample) | (reporting_sample <= 0.0)
    invalid_count = int(invalid_mask.sum())
    finite_reporting = reporting_sample[np.isfinite(reporting_sample)]
    minimum = float(np.min(finite_reporting)) if finite_reporting.size else float("nan")
    diagnostic_invalid_mask = ~np.isfinite(fair_value_per_share) | (
        fair_value_per_share <= 0.0
    )
    diagnostic_invalid_count = int(diagnostic_invalid_mask.sum())
    finite_all_paths = fair_value_per_share[np.isfinite(fair_value_per_share)]
    diagnostic_minimum = (
        float(np.min(finite_all_paths)) if finite_all_paths.size else float("nan")
    )
    prefix_summaries = tuple(
        _summarize_prefix(
            fair_value_per_share,
            reference_market_price_per_share,
            count,
        )
        for count in prefix_paths
    )
    convergence_passed, convergence_differences = _assess_convergence(prefix_summaries)

    reporting_mean_log: float | None = None
    reporting_standard_deviation_log: float | None = None
    reporting_z_score: float | None = None
    reporting_level_quantiles: NDArray[np.float64] | None = None
    reporting_terminal_share: float | None = None
    if invalid_count == 0:
        reporting_log_fair_value = np.log(reporting_sample)
        reporting_mean_log = float(np.mean(reporting_log_fair_value))
        reporting_standard_deviation_log = float(
            np.std(reporting_log_fair_value, ddof=1)
        )
        if (
            np.isfinite(reporting_standard_deviation_log)
            and reporting_standard_deviation_log > 0.0
        ):
            reporting_z_score = float(
                (np.log(reference_market_price_per_share) - reporting_mean_log)
                / reporting_standard_deviation_log
            )
            reporting_level_quantiles = np.quantile(reporting_sample, [0.05, 0.5, 0.95])
            reporting_terminal_share = float(
                np.mean(discounted_terminal_value[:reporting_paths])
                / np.mean(enterprise_value[:reporting_paths])
            )

    # V4 applies to the entire generated distribution. The first 1,000
    # paths remain a publication-facing prefix, but a later invalid draw must
    # still invalidate the formal firm-date result.
    invalid_component: str | None = None
    invalid_reason: str | None = None
    component_checks = (
        ("forecast_revenue", forecast_revenue, False),
        ("cash_flow", cash_flow, False),
        ("discounted_cash_flow", discounted_cash_flow, False),
        ("discounted_terminal_value", discounted_terminal_value, False),
        ("enterprise_value", enterprise_value, False),
        ("equity_value", equity_value, True),
        ("fair_value_per_share", fair_value_per_share, True),
    )
    for component_name, component_values, require_positive in component_checks:
        component_array = np.asarray(component_values, dtype=np.float64)
        if np.any(~np.isfinite(component_array)):
            invalid_component = component_name
            invalid_reason = "non_finite"
            break
        if require_positive and np.any(component_array <= 0.0):
            invalid_component = component_name
            invalid_reason = "non_positive"
            break

    if diagnostic_invalid_count:
        # Return the complete failed-path evidence but leave every log-derived
        # statistic absent because no compliant distribution exists.
        return FairValueResult(
            reporting_paths=reporting_paths,
            reporting_invalid_path_count=invalid_count,
            reporting_invalid_path_share=invalid_count / reporting_paths,
            reporting_minimum_fair_value=minimum,
            valid=False,
            invalid_path_count=diagnostic_invalid_count,
            invalid_path_share=(diagnostic_invalid_count / fair_value_per_share.size),
            minimum_fair_value=diagnostic_minimum,
            invalid_component=invalid_component,
            invalid_reason=invalid_reason,
            diagnostic_path_count=fair_value_per_share.size,
            diagnostic_invalid_path_count=diagnostic_invalid_count,
            diagnostic_invalid_path_share=(
                diagnostic_invalid_count / fair_value_per_share.size
            ),
            diagnostic_minimum_fair_value=diagnostic_minimum,
            enterprise_value=enterprise_value,
            discounted_terminal_value=discounted_terminal_value,
            equity_value=equity_value,
            fair_value_per_share=fair_value_per_share,
            mean_log_fair_value=reporting_mean_log,
            standard_deviation_log_fair_value=reporting_standard_deviation_log,
            z_score=reporting_z_score,
            level_quantile_05=(
                None
                if reporting_level_quantiles is None
                else float(reporting_level_quantiles[0])
            ),
            level_quantile_50=(
                None
                if reporting_level_quantiles is None
                else float(reporting_level_quantiles[1])
            ),
            level_quantile_95=(
                None
                if reporting_level_quantiles is None
                else float(reporting_level_quantiles[2])
            ),
            terminal_value_share_of_enterprise_value=reporting_terminal_share,
            prefixes=prefix_summaries,
            convergence_passed=False,
            convergence_differences=convergence_differences,
        )

    if (
        reporting_mean_log is None
        or reporting_standard_deviation_log is None
        or reporting_level_quantiles is None
        or reporting_terminal_share is None
    ):
        raise ValueError("Log fair-value standard deviation must be positive.")
    return FairValueResult(
        reporting_paths=reporting_paths,
        reporting_invalid_path_count=0,
        reporting_invalid_path_share=0.0,
        reporting_minimum_fair_value=minimum,
        valid=True,
        invalid_path_count=0,
        invalid_path_share=0.0,
        minimum_fair_value=diagnostic_minimum,
        invalid_component=None,
        invalid_reason=None,
        diagnostic_path_count=fair_value_per_share.size,
        diagnostic_invalid_path_count=diagnostic_invalid_count,
        diagnostic_invalid_path_share=(
            diagnostic_invalid_count / fair_value_per_share.size
        ),
        diagnostic_minimum_fair_value=diagnostic_minimum,
        enterprise_value=enterprise_value,
        discounted_terminal_value=discounted_terminal_value,
        equity_value=equity_value,
        fair_value_per_share=fair_value_per_share,
        mean_log_fair_value=reporting_mean_log,
        standard_deviation_log_fair_value=reporting_standard_deviation_log,
        # The paper standardizes log market price against the simulated
        # log-fair-value distribution; level prices must never enter directly.
        z_score=float(
            (np.log(reference_market_price_per_share) - reporting_mean_log)
            / reporting_standard_deviation_log
        ),
        level_quantile_05=float(reporting_level_quantiles[0]),
        level_quantile_50=float(reporting_level_quantiles[1]),
        level_quantile_95=float(reporting_level_quantiles[2]),
        terminal_value_share_of_enterprise_value=reporting_terminal_share,
        prefixes=prefix_summaries,
        convergence_passed=convergence_passed,
        convergence_differences=convergence_differences,
    )
