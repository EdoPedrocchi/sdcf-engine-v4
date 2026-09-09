# Input contract

The command accepts one quarterly fundamentals CSV and one snapshot YAML. It
does not infer missing fields, units, signs, currencies, fiscal labels or
availability times.

## Quarterly fundamentals CSV

Each row is one vintage of one fiscal quarter. Revisions may repeat a fiscal
period only when `available_at` differs. The engine selects the latest eligible
vintage at the requested cutoff.

| Field | Type | Required meaning |
|---|---|---|
| `security_id` | string | Stable security identity; identical on every row |
| `fiscal_period_end` | ISO date | Actual fiscal-quarter end |
| `fiscal_year` | integer | Source-supported fiscal year |
| `fiscal_quarter` | integer 1–4 | Source-supported fiscal quarter |
| `available_at` | timezone-aware ISO timestamp | Time at which this exact vintage was publicly usable |
| `revenue` | finite number | Discrete quarterly revenue flow |
| `ebitda` | finite number | Discrete quarterly EBITDA |
| `depreciation_and_amortization` | finite number | Discrete quarterly D&A expense |
| `capital_expenditure` | finite nonnegative number | Quarterly CAPEX as a positive expenditure |
| `working_capital` | finite number | Quarter-end working-capital stock |
| `marginal_tax_rate_annual_decimal` | number in [0, 1] | Annual tax rate in decimal form |

All monetary columns must use the same currency and scale. Flow columns must be
discrete quarterly values rather than year-to-date or TTM values. Missing,
duplicate, nonfinite, nonconsecutive or conflicting observations fail.

## Snapshot YAML

The top level requires exactly:

| Field | Meaning |
|---|---|
| `security_id` | Security to evaluate |
| `fiscal_period_end` | Required current accounting quarter |
| `information_cutoff` | Timezone-aware boundary for all information |
| `reference_market_price_per_share` | Positive adjusted price used only for z-score and convergence |
| `capital` | Deterministic valuation inputs below |

The `capital` block requires exactly:

| Field | Unit and sign |
|---|---|
| `short_term_wacc_annual_decimal` | Annual decimal discount rate for explicit cash flows |
| `terminal_wacc_annual_decimal` | Annual decimal terminal discount rate; must exceed growth |
| `perpetual_growth_annual_decimal` | Annual decimal perpetual growth rate |
| `total_debt` | Nonnegative amount subtracted from enterprise value |
| `cash_and_short_term_investments` | Nonnegative amount added to common equity |
| `minority_interest` | Nonnegative amount subtracted |
| `preferred_stock` | Nonnegative amount subtracted |
| `shares_outstanding` | Positive shares in a scale compatible with monetary inputs |
| `available_at` | Timezone-aware timestamp no later than `information_cutoff` |

## Time and adjustment rules

- `available_at` means usable public availability, not fiscal period end or the
  date on which a database was queried.
- A later revision cannot enter an earlier snapshot.
- Market price and shares must be on compatible split-adjustment and scale
  bases.
- Ticker is not a substitute for a stable security identifier.
- Currency conversion, source-sign normalization and corporate-action handling
  occur before this interface and must be documented by the caller.

## Failure behavior

The engine stops on an unresolved required choice, missing quarter, invalid
timestamp, failed required fit, inconclusive structural bootstrap, nonpositive
terminal spread, nonpositive shares, or any invalid fair-value path. It does not
fall back to an older observation or alternative model to obtain a result.
