"""Compose the vendor-neutral SDCF V4 method from its scientific modules.

Inputs are already normalized one-security accounting observations and capital
assumptions with explicit availability. This reusable numerical orchestrator
has no raw-vendor reader, empirical runner, authorization flag or release action.
It cannot turn supplied metadata into evidence of a validated data contract.
The entry point accepts only the immutable V4 contract and exports full-sample
scientific moments together with explicit numerical diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sdcf_engine.core.cash_flows import reconstruct_operating_cash_flow
from sdcf_engine.core.margins import (
    AlphaEstimate,
    build_calendar_bounded_ttm_flows,
    estimate_alpha,
)
from sdcf_engine.core.method import V4Method
from sdcf_engine.core.revenue_models import (
    RevenueSelection,
    construct_ttm_revenue,
    select_revenue_model,
    simulate_log_ttm_paths,
)
from sdcf_engine.core.sequential_bootstrap import SequentialPolicy
from sdcf_engine.core.valuation import (
    FairValueResult,
    PrefixSummary,
    ValuationInputs,
    compute_fair_value,
    derive_stream_seed,
)
from sdcf_engine.core.zscore import (
    DailyZScorePanel,
    align_quarterly_fair_values_to_daily_prices,
)
from sdcf_engine.data.annual_fundamentals import (
    AnnualBetaEstimate,
    available_fiscal_history,
    estimate_available_annual_beta,
)
from sdcf_engine.exceptions import DataContractError


@dataclass(frozen=True)
class CapitalInputs:
    """Available annual-decimal rates and monetary equity-bridge inputs.

    All monetary values share the accounting input currency and scale. The
    timestamp is the latest availability of these inputs, not period end.
    """

    short_term_wacc_annual_decimal: float
    terminal_wacc_annual_decimal: float
    perpetual_growth_annual_decimal: float
    total_debt: float
    cash_and_short_term_investments: float
    minority_interest: float
    preferred_stock: float
    shares_outstanding: float
    available_at: pd.Timestamp


@dataclass(frozen=True)
class AlphaEligibility:
    """Initial-window decision carried explicitly through subsequent snapshots."""

    method_id: str
    security_id: str
    information_cutoff: pd.Timestamp
    fiscal_period_end: pd.Timestamp
    r_squared: float
    eligible: bool


@dataclass(frozen=True)
class SnapshotResult:
    """Estimates, annual-beta lineage, full scientific output and display prefix."""

    method_id: str
    security_id: str
    fiscal_period_end: pd.Timestamp
    available_at: pd.Timestamp
    alpha: AlphaEstimate
    beta: AnnualBetaEstimate
    eligibility: AlphaEligibility
    revenue: RevenueSelection
    valuation: FairValueResult
    display_summary: PrefixSummary
    stream_seeds: dict[str, int]


def evaluate_snapshot(
    fundamentals: pd.DataFrame,
    capital: CapitalInputs,
    *,
    security_id: str,
    fiscal_period_end: pd.Timestamp,
    information_cutoff: pd.Timestamp,
    reference_market_price_per_share: float,
    method: V4Method,
    initial_eligibility: AlphaEligibility | None = None,
) -> SnapshotResult:
    """Estimate and value one available firm-date under the complete V4 method.

    Parameters
    ----------
    fundamentals:
        Fiscal-identity/availability frame with revenue, EBITDA, D&A, CAPEX,
        working_capital and marginal_tax_rate_annual_decimal. Discrete quarterly
        flows, canonical CAPEX sign, common currency and units are prerequisites.
    capital:
        Available rates and equity bridge. Market price is kept downstream.
    security_id:
        Stable one-security identity present on all accounting rows.
    fiscal_period_end, information_cutoff:
        Accounting-period boundary and timezone-aware information boundary.
    reference_market_price_per_share:
        Positive observed price used for z-score and convergence, not valuation.
    method:
        Fully validated V4 method contract.
    initial_eligibility:
        None on the initial scheduled snapshot; its returned decision must be
        supplied for later snapshots so eligibility is not reapplied each quarter.

    Returns
    -------
    SnapshotResult
        Auditable scientific distribution, estimates, lineage and separate plot
        summary. Numerical validity and convergence remain explicit flags.

    Raises
    ------
    DataContractError
        If input availability, fiscal history or initial eligibility is invalid.
    ValueError
        If required estimation, simulation or valuation fails.
    """
    method.validate()
    cutoff = pd.Timestamp(information_cutoff)
    capital_time = pd.Timestamp(capital.available_at)
    if cutoff.tzinfo is None or capital_time.tzinfo is None or capital_time > cutoff:
        raise DataContractError(
            "Capital inputs require known timestamps no later than the cutoff."
        )
    history = available_fiscal_history(
        fundamentals,
        security_id=security_id,
        information_cutoff=cutoff,
        fiscal_period_end=fiscal_period_end,
    )
    if pd.Timestamp(history.iloc[-1]["fiscal_period_end"]) != pd.Timestamp(
        fiscal_period_end
    ):
        raise DataContractError(
            "The required current accounting quarter is not available; do not substitute an older snapshot."
        )
    indexed = history.set_index("fiscal_period_end")
    flow_names = [
        "revenue",
        "ebitda",
        "depreciation_and_amortization",
        "capital_expenditure",
    ]
    if not set([*flow_names, "marginal_tax_rate_annual_decimal"]).issubset(
        indexed.columns
    ):
        raise DataContractError(
            "Candidate alpha lacks required accounting flow/tax columns."
        )
    ttm = build_calendar_bounded_ttm_flows(
        indexed[flow_names],
        minimum_observations=method.alpha_minimum_observations,
        maximum_observations=method.alpha_maximum_observations,
        fiscal_ordinals=indexed["fiscal_ordinal"],
    )
    tax = indexed.loc[ttm.index, "marginal_tax_rate_annual_decimal"]
    cash_flow = reconstruct_operating_cash_flow(
        ttm["ebitda"],
        ttm["depreciation_and_amortization"],
        ttm["capital_expenditure"],
        tax,
    )
    alpha = estimate_alpha(
        cash_flow,
        ttm["revenue"],
        expected_observations=len(ttm),
    )
    eligibility = initial_eligibility
    if eligibility is None:
        eligibility = AlphaEligibility(
            method.method_id,
            security_id,
            cutoff,
            pd.Timestamp(fiscal_period_end),
            alpha.uncentered_r_squared,
            alpha.uncentered_r_squared >= method.alpha_eligibility_threshold,
        )
    if (
        eligibility.method_id != method.method_id
        or eligibility.security_id != security_id
        or eligibility.information_cutoff > cutoff
        or eligibility.fiscal_period_end > pd.Timestamp(fiscal_period_end)
    ):
        raise DataContractError(
            "Initial alpha eligibility belongs to another method, security or future date."
        )
    if not np.isfinite(eligibility.r_squared) or eligibility.eligible != (
        eligibility.r_squared >= method.alpha_eligibility_threshold
    ):
        raise DataContractError(
            "Initial alpha eligibility disagrees with its recorded R-squared."
        )
    if not eligibility.eligible:
        raise DataContractError(
            f"Initial alpha R-squared {eligibility.r_squared} is below {method.alpha_eligibility_threshold}."
        )
    beta = estimate_available_annual_beta(history)
    ttm_revenue = construct_ttm_revenue(
        indexed["revenue"], fiscal_ordinals=indexed["fiscal_ordinal"]
    )
    # Actual cutoff, not just fiscal period, enters identity: a later public
    # revision creates a new stream/input identity rather than rewriting history.
    seeds = {
        purpose: derive_stream_seed(
            master_seed=method.master_seed,
            method_id=f"{method.method_id}:{purpose}:{cutoff.isoformat()}",
            security_id=security_id,
            valuation_date_iso=pd.Timestamp(fiscal_period_end).date().isoformat(),
        )
        for purpose in ("bootstrap", "filtered_state", "future_innovations")
    }
    revenue = select_revenue_model(
        ttm_revenue,
        adf_significance_level=method.adf_significance,
        maximum_adf_lag=method.adf_maximum_lag,
        bootstrap_rng=np.random.Generator(np.random.PCG64(seeds["bootstrap"])),
        bootstrap_replicates=method.bootstrap_replicates,
        sequential_policy=SequentialPolicy(
            looks=method.bootstrap_looks,
            resampling_risk=method.bootstrap_resampling_risk,
            significance=method.bootstrap_significance,
        ),
    )
    paths = simulate_log_ttm_paths(
        revenue.fit,
        steps=method.forecast_steps,
        paths=method.scientific_paths,
        rng=np.random.Generator(np.random.PCG64(seeds["future_innovations"])),
        state_rng=np.random.Generator(np.random.PCG64(seeds["filtered_state"])),
    )
    valuation = compute_fair_value(
        paths,
        ValuationInputs(
            float(ttm_revenue.iloc[-1]),
            alpha.alpha,
            beta.beta,
            capital.short_term_wacc_annual_decimal,
            capital.terminal_wacc_annual_decimal,
            capital.perpetual_growth_annual_decimal,
            capital.total_debt,
            capital.cash_and_short_term_investments,
            capital.minority_interest,
            capital.preferred_stock,
            capital.shares_outstanding,
        ),
        reference_market_price_per_share=reference_market_price_per_share,
        reporting_paths=method.scientific_paths,
        prefix_paths=method.prefix_paths,
    )
    display = next(
        prefix for prefix in valuation.prefixes if prefix.paths == method.display_paths
    )
    # Valuation is not made retrospectively tradable at an earlier accounting
    # release when its capital inputs or information cutoff are later.
    return SnapshotResult(
        method.method_id,
        security_id,
        pd.Timestamp(fiscal_period_end),
        cutoff,
        alpha,
        beta,
        eligibility,
        revenue,
        valuation,
        display,
        seeds,
    )


def compute_daily_z_scores(
    snapshots: tuple[SnapshotResult, ...],
    daily_prices: pd.DataFrame,
) -> DailyZScorePanel:
    """Align only valid converged full-sample summaries to daily market closes.

    Parameters
    ----------
    snapshots:
        Ordered required quarterly valuations from the candidate pipeline.
    daily_prices:
        Canonical price table including actual market-close timestamps; dates
        must be bounded by the study's separately declared valuation schedule.

    Returns
    -------
        DailyZScorePanel using scientific, never display, moments.

    Raises
    ------
    DataContractError
        If a snapshot is invalid, unconverged or belongs to another method.
    """
    if not snapshots:
        raise DataContractError("Candidate daily alignment requires snapshots.")
    rows: list[dict[str, object]] = []
    for snapshot in snapshots:
        result = snapshot.valuation
        if (
            snapshot.method_id != V4Method().method_id
            or snapshot.method_id != snapshots[0].method_id
            or not result.valid
            or not result.convergence_passed
        ):
            raise DataContractError(
                "Candidate daily output requires valid converged scientific distributions."
            )
        rows.append(
            {
                "security_id": snapshot.security_id,
                "fiscal_period_end": snapshot.fiscal_period_end,
                "available_at": snapshot.available_at,
                "availability_precision": "timestamp",
                "mean_log_fair_value": result.mean_log_fair_value,
                "standard_deviation_log_fair_value": result.standard_deviation_log_fair_value,
            }
        )
    return align_quarterly_fair_values_to_daily_prices(pd.DataFrame(rows), daily_prices)
