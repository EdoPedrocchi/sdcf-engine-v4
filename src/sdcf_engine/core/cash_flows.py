"""Implement the article's canonical operating-cash-flow transformations.

Bottazzi et al. (2023), Equation (4), page 67, supplies the operating-cash-flow
reconstruction; Equation (2), page 66, supplies the revenue-to-cash-flow
identity. Cordoni (2021), Chapter 3, Section 3.1.1, pages 62-63, provides
secondary frequency and margin context. V4, V4, V4, and V4 define
the implementation details that those sources do not uniquely specify.

The module receives already normalized arrays and does not map vendor fields,
infer units, repair missing data, or estimate margins. Monetary inputs supplied
to one call must share a currency and scale. CAPEX is a positive expenditure in
the canonical interface; that source-sign normalization is a project data
contract rather than an author-specified vendor convention.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return a finite floating-point array or raise a targeted error."""
    # Validate before applying an economic identity so NaN propagation cannot
    # disguise which canonical input breached the data contract.
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def reconstruct_operating_cash_flow(
    ebitda: ArrayLike,
    depreciation_and_amortization: ArrayLike,
    capital_expenditure: ArrayLike,
    tax_rate_annual_decimal: ArrayLike,
) -> NDArray[np.float64]:
    """Reconstruct operating cash flow before working-capital adjustment.

    Parameters
    ----------
    ebitda:
        Quarterly EBITDA in one canonical monetary currency and scale.
    depreciation_and_amortization:
        Quarterly D&A expense in the same currency and scale as EBITDA.
    capital_expenditure:
        Quarterly CAPEX expressed as a positive expenditure in the same
        currency and scale.
    tax_rate_annual_decimal:
        Tax rate as a decimal in the closed interval ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Reconstructed cash flow in the input monetary scale.

    Raises
    ------
    ValueError
        If an input is non-finite, arrays cannot be broadcast together, CAPEX
        is negative, or a tax rate lies outside ``[0, 1]``.

    Notes
    -----
    Implements Bottazzi et al. (2023), Equation (4), page 67:

    ``(EBITDA - D&A) * (1 - tax) + D&A - CAPEX``.

    The equation comes from the article. Requiring canonical CAPEX to be a
    positive expenditure, rejecting invalid tax values, and refusing source
    sign inference are project guardrails under V4 and the documented source ambiguity.
    """
    ebitda_array = _finite_array(ebitda, "ebitda")
    da_array = _finite_array(
        depreciation_and_amortization,
        "depreciation_and_amortization",
    )
    capex_array = _finite_array(capital_expenditure, "capital_expenditure")
    tax_array = _finite_array(tax_rate_annual_decimal, "tax_rate_annual_decimal")

    # CAPEX is normalized once at the vendor boundary. Accepting a negative
    # value here would risk reversing the source sign a second time.
    if np.any(capex_array < 0.0):
        raise ValueError(
            "capital_expenditure must be a positive canonical expenditure; "
            "normalize the verified source sign at the data boundary."
        )
    if np.any((tax_array < 0.0) | (tax_array > 1.0)):
        raise ValueError("tax_rate_annual_decimal must lie in [0, 1].")

    try:
        # After-tax EBIT is converted back to operating cash flow by adding
        # non-cash D&A and then deducting the positive CAPEX expenditure.
        return (ebitda_array - da_array) * (1.0 - tax_array) + da_array - capex_array
    except ValueError as exc:
        raise ValueError(
            "Cash-flow inputs must have mutually broadcastable shapes."
        ) from exc


def cash_flow_from_margins(
    revenue_current: ArrayLike,
    revenue_previous: ArrayLike,
    alpha: ArrayLike,
    beta: ArrayLike,
    *,
    allow_nonfinite_paths: bool = False,
) -> NDArray[np.float64]:
    """Convert revenue into valuation cash flow using alpha and beta.

    Parameters
    ----------
    revenue_current:
        Current-period revenue in one canonical currency and scale.
    revenue_previous:
        Previous-period revenue in the same currency and scale.
    alpha:
        Operating-cash-flow margin as a decimal.
    beta:
        Working-capital-to-revenue margin as a decimal.
    allow_nonfinite_paths:
        Retain overflow or non-finite simulated paths for downstream V4
        classification. Canonical deterministic callers must keep the default
        strict validation.

    Returns
    -------
    numpy.ndarray
        Cash flow in the revenue currency and scale.

    Raises
    ------
    ValueError
        If an input is non-finite or arrays cannot be broadcast together.

    Notes
    -----
    Implements Bottazzi et al. (2023), Equation (2), page 66, and Cordoni
    (2021), Equation (3.3), page 63:

    ``CF_t = (alpha - beta) * REV_t + beta * REV_(t-1)``.

    Retaining a non-finite simulated path for downstream classification is a
    V4 engineering rule. It does not repair the path or alter this identity.
    """
    revenue_array = (
        np.asarray(revenue_current, dtype=np.float64)
        if allow_nonfinite_paths
        else _finite_array(revenue_current, "revenue_current")
    )
    lagged_array = (
        np.asarray(revenue_previous, dtype=np.float64)
        if allow_nonfinite_paths
        else _finite_array(revenue_previous, "revenue_previous")
    )
    alpha_array = _finite_array(alpha, "alpha")
    beta_array = _finite_array(beta, "beta")
    try:
        # Equation 2 charges beta only on the change in revenue: expanding the
        # expression gives alpha * REV_t - beta * (REV_t - REV_(t-1)).
        with np.errstate(over="ignore", invalid="ignore"):
            return (alpha_array - beta_array) * revenue_array + (
                beta_array * lagged_array
            )
    except ValueError as exc:
        raise ValueError(
            "Revenue and margin inputs must have mutually broadcastable shapes."
        ) from exc
