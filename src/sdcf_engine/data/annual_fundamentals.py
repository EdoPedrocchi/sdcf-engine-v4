"""Select available fiscal accounting vintages and annual beta inputs for V4.

The caller supplies validated fiscal identities and timestamp semantics; this
module cannot manufacture vendor evidence. Annual revenue is the sum of four
identified quarterly flows and working capital is the fiscal-year-end stock.
Latest-vintage selection is point-in-time, never a revision backfill. Missing
required years fail rather than silently selecting an older substitute year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sdcf_engine.core.margins import estimate_beta
from sdcf_engine.exceptions import DataContractError


@dataclass(frozen=True)
class AnnualBetaEstimate:
    """Annual margin with the four contributing observations and availability."""

    beta: float
    available_at: pd.Timestamp
    observations: pd.DataFrame


def available_fiscal_history(
    fundamentals: pd.DataFrame,
    *,
    security_id: str,
    information_cutoff: pd.Timestamp,
    fiscal_period_end: pd.Timestamp,
) -> pd.DataFrame:
    """Select latest eligible revisions while retaining provider fiscal identity.

    Parameters
    ----------
    fundamentals:
        One firm's rows with security_id, fiscal_period_end, fiscal_year,
        fiscal_quarter and timezone-aware available_at. Revisions are allowed
        only with distinct availability timestamps.
    security_id:
        Stable identifier expected on every row, not a ticker lookup.
    information_cutoff:
        Timezone-aware information boundary. Later releases cannot enter.
    fiscal_period_end:
        Latest accounting period permitted for this snapshot.

    Returns
    -------
    pandas.DataFrame
        One latest available vintage per period, with fiscal_ordinal attached.

    Raises
    ------
    DataContractError
        If identity, timestamp, fiscal sequence or revision keys are invalid.
    """
    required = {
        "security_id",
        "fiscal_period_end",
        "fiscal_year",
        "fiscal_quarter",
        "available_at",
    }
    if not required.issubset(fundamentals.columns):
        raise DataContractError(
            f"Candidate requires fiscal/availability columns: {sorted(required - set(fundamentals.columns))}."
        )
    cutoff = pd.Timestamp(information_cutoff)
    if cutoff.tzinfo is None:
        raise DataContractError(
            "Information cutoff requires a timezone-aware timestamp."
        )
    frame = fundamentals.copy()
    if frame.empty or not frame["security_id"].eq(security_id).all():
        raise DataContractError(
            "Candidate accounting rows require one matching stable security ID."
        )
    if any(pd.Timestamp(value).tzinfo is None for value in frame["available_at"]):
        raise DataContractError(
            "Candidate available_at requires evidenced timezone-aware timestamps."
        )
    frame["available_at"] = pd.to_datetime(
        frame["available_at"], utc=True, errors="raise"
    )
    frame["fiscal_period_end"] = pd.to_datetime(
        frame["fiscal_period_end"], errors="raise"
    )
    if frame[["available_at", "fiscal_period_end"]].isna().any(axis=None):
        raise DataContractError("Accounting dates cannot be missing.")
    frame = frame.loc[
        (frame["available_at"] <= cutoff)
        & (frame["fiscal_period_end"] <= pd.Timestamp(fiscal_period_end))
    ].copy()
    if frame.empty:
        raise DataContractError(
            "No accounting observation is available at this cutoff."
        )
    if frame[["available_at", "fiscal_period_end"]].isna().any(axis=None):
        raise DataContractError("Accounting dates cannot be missing.")
    if (frame["available_at"].dt.tz_localize(None) < frame["fiscal_period_end"]).any():
        raise DataContractError(
            "Accounting availability precedes its fiscal period end."
        )
    for column in ("fiscal_year", "fiscal_quarter"):
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise DataContractError(f"Invalid integral fiscal identity: {column}.")
        frame[column] = values.astype(np.int64)
    if not frame["fiscal_quarter"].isin([1, 2, 3, 4]).all():
        raise DataContractError(
            "Fiscal quarters must be provider-labelled 1 through 4."
        )
    if frame.duplicated(["fiscal_period_end", "available_at"]).any():
        raise DataContractError(
            "Ambiguous accounting vintages at the same availability timestamp."
        )
    identities = frame.groupby("fiscal_period_end")[
        ["fiscal_year", "fiscal_quarter"]
    ].nunique()
    if (identities > 1).any(axis=None):
        raise DataContractError(
            "A revision changes fiscal identity; explicit reconciliation is required."
        )
    frame = frame.sort_values("available_at").drop_duplicates(
        "fiscal_period_end", keep="last"
    )
    frame = frame.sort_values("fiscal_period_end").reset_index(drop=True)
    frame["fiscal_ordinal"] = frame["fiscal_year"] * 4 + frame["fiscal_quarter"] - 1
    if frame["fiscal_ordinal"].duplicated().any() or np.any(
        np.diff(frame["fiscal_ordinal"]) <= 0
    ):
        raise DataContractError(
            "Provider fiscal identities must have unique chronological order."
        )
    return frame


def estimate_available_annual_beta(history: pd.DataFrame) -> AnnualBetaEstimate:
    """Estimate beta from the latest four consecutive available fiscal years.

    Parameters
    ----------
    history:
        Output of available_fiscal_history, with revenue and working_capital in
        common monetary units. Quarter 4 identifies each annual WC stock.

    Returns
    -------
    AnnualBetaEstimate
        Mean ratio, latest contributing availability, and source-year lineage.

    Raises
    ------
    DataContractError
        If any required quarter, annual stock or annual revenue is unavailable.

    Notes
    -----
    V4 retains four observations as the declared resolution of the sources'
    ambiguous three-years/2006--2009 wording. No annual WC stock is summed.
    """
    required = {
        "fiscal_year",
        "fiscal_quarter",
        "fiscal_period_end",
        "available_at",
        "revenue",
        "working_capital",
    }
    if not required.issubset(history.columns):
        raise DataContractError("Annual beta lacks required fiscal/accounting columns.")
    annual_ends = history.loc[history["fiscal_quarter"] == 4]
    if annual_ends.empty:
        raise DataContractError(
            "No available fiscal-year-end working-capital observation."
        )
    last_year = int(annual_ends["fiscal_year"].max())
    rows: list[dict[str, object]] = []
    for year in range(last_year - 3, last_year + 1):
        quarters = history.loc[history["fiscal_year"] == year].sort_values(
            "fiscal_quarter"
        )
        if quarters["fiscal_quarter"].tolist() != [1, 2, 3, 4]:
            raise DataContractError(
                f"Annual beta requires all four quarters in fiscal year {year}."
            )
        revenue = pd.to_numeric(quarters["revenue"], errors="raise").to_numpy(
            dtype=float
        )
        stock = float(quarters.iloc[-1]["working_capital"])
        if not np.all(np.isfinite(revenue)) or not np.isfinite(stock):
            raise DataContractError(
                f"Annual beta has missing/nonfinite inputs in fiscal year {year}."
            )
        annual_revenue = float(revenue.sum())
        if annual_revenue <= 0:
            raise DataContractError(
                f"Annual beta revenue must be positive in fiscal year {year}."
            )
        rows.append(
            {
                "fiscal_year": year,
                "fiscal_period_end": quarters.iloc[-1]["fiscal_period_end"],
                "available_at": quarters["available_at"].max(),
                "working_capital": stock,
                "annual_revenue": annual_revenue,
                "ratio": stock / annual_revenue,
            }
        )
    observations = pd.DataFrame(rows)
    beta = estimate_beta(
        observations["working_capital"], observations["annual_revenue"]
    )
    return AnnualBetaEstimate(
        beta, pd.Timestamp(observations["available_at"].max()), observations
    )
