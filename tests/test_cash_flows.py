"""Unit tests for the paper's canonical cash-flow equations."""

import numpy as np
import pytest

from sdcf_engine.core.cash_flows import (
    cash_flow_from_margins,
    reconstruct_operating_cash_flow,
)


def test_reconstruct_operating_cash_flow_matches_manual_calculation() -> None:
    """Protect the tax, D&A add-back, and CAPEX signs in paper Phase 1."""
    result = reconstruct_operating_cash_flow(
        ebitda=[120.0],
        depreciation_and_amortization=[20.0],
        capital_expenditure=[30.0],
        tax_rate_annual_decimal=[0.25],
    )

    # Manual result: (120 - 20) * 0.75 + 20 - 30 = 65.
    assert result == pytest.approx(np.array([65.0]))


def test_reconstruct_operating_cash_flow_rejects_negative_canonical_capex() -> None:
    """Prevent a vendor CAPEX sign from being normalized twice."""
    with pytest.raises(ValueError, match="positive canonical expenditure"):
        reconstruct_operating_cash_flow(120.0, 20.0, -30.0, 0.25)


def test_cash_flow_zero_tax_and_zero_investment_edges_are_exact() -> None:
    """Pin economically interpretable boundary cases without hidden repair."""
    zero_tax = reconstruct_operating_cash_flow(120.0, 20.0, 30.0, 0.0)
    zero_da_and_capex = reconstruct_operating_cash_flow(120.0, 0.0, 0.0, 0.25)

    assert zero_tax == pytest.approx(np.array([90.0]), rel=1e-12, abs=1e-10)
    assert zero_da_and_capex == pytest.approx(np.array([90.0]), rel=1e-12, abs=1e-10)


@pytest.mark.parametrize("tax_rate", [-0.01, 1.01, np.nan])
def test_reconstruct_operating_cash_flow_rejects_invalid_tax_rate(
    tax_rate: float,
) -> None:
    """Reject invalid tax inputs instead of clipping them into the formula."""
    with pytest.raises(ValueError, match="tax_rate_annual_decimal"):
        reconstruct_operating_cash_flow(120.0, 20.0, 30.0, tax_rate)


def test_cash_flow_signs_isolate_capex_and_da_tax_shield() -> None:
    """Verify that CAPEX subtracts and D&A contributes only its tax shield."""
    baseline = float(reconstruct_operating_cash_flow(120.0, 20.0, 30.0, 0.25))
    higher_capex = float(reconstruct_operating_cash_flow(120.0, 20.0, 31.0, 0.25))
    higher_da = float(reconstruct_operating_cash_flow(120.0, 21.0, 30.0, 0.25))

    assert baseline - higher_capex == pytest.approx(1.0, rel=1e-12, abs=1e-10)
    assert higher_da - baseline == pytest.approx(0.25, rel=1e-12, abs=1e-10)


def test_margin_cash_flow_forms_are_identical() -> None:
    """Detect a sign error in the working-capital adjustment."""
    revenue_current = np.array([110.0, 90.0])
    revenue_previous = np.array([100.0, 100.0])
    alpha = 0.20
    beta = 0.05

    paper_form = cash_flow_from_margins(
        revenue_current,
        revenue_previous,
        alpha,
        beta,
    )
    identity_form = alpha * revenue_current - beta * (
        revenue_current - revenue_previous
    )

    assert paper_form == pytest.approx(identity_form)
