# Methodological choices, advantages, and limitations

The published article is the primary authority. Cordoni's thesis is secondary
context. “V4 choice” below identifies an operational decision that should not be
attributed to undisclosed author code.

| Component | Published basis | V4 choice | Advantages | Limitations |
|---|---|---|---|---|
| Accounting cash flow | Article Equation (4) | Positive canonical CAPEX; annual-decimal tax | Prevents double sign reversal and scale mistakes | Requires external validation of source signs and tax semantics |
| Alpha errors | Article specifies MA errors and AIC | Exact Gaussian fits for orders 0–4 after common scaling | Unit-invariant alpha and comparable candidate samples | Order range, optimizer and equation-error R² convention are operational assumptions |
| Alpha eligibility | Article gives an initial 10% filter | Uncentered equation-error R²; equality passes; decision carried forward | Explicit and reproducible | The precise R² convention and later-period treatment are not fully disclosed |
| Beta | Article and thesis describe annual working-capital ratios | Equal average of four latest consecutive fiscal-year ratios | Uses annual flow/stock units consistently | Four observations and equal weighting are interpretations of ambiguous source wording |
| ADF routing | Article gives the model-selection sequence; thesis clarifies log increments | Test log-TTM growth with constant, max lag 4 and AIC; strict 5% threshold | Fully specified and repeatable | Test form, lag rule and boundary equality can affect routing in small samples |
| Structural likelihood | Article gives local-level and local-trend equations | Proper common Gaussian likelihood for every increment | Removes unequal diffuse-likelihood constants and compares identical samples | Conditions on a fitted initial slope and selected likelihood specification |
| Structural selection | Article calls for an LR comparison | Parametric-null bootstrap with sequential exact intervals | Replaces an asymptotic boundary approximation and often stops early | Conditional Monte Carlo control is not proof of uniform statistical size; a case may remain unresolved |
| Initial forecast state | Sources do not fully specify initialization | Draw from the filtered Gaussian state and covariance | Propagates latent-state uncertainty and horizon dependence | Parameter-estimation uncertainty is excluded |
| Forecast sampling | Sources disclose Gaussian equations but not generator or path count | PCG64, purpose-separated deterministic seeds, 20 quarters, 50,000 paths | Reproducible and resistant to scheduling changes | Computationally heavier; results remain conditional on Gaussian innovations |
| Cash-flow grid | Sources specify TTM forecasting and a five-year valuation, but not exact payment timing | Value all 20 overlapping quarterly-updated TTM cash-flow estimates without dividing by four | Matches the V4 interpretation and uses each forecast update | This is the largest ambiguity: overlapping annualized TTM amounts may economically overcount cash flow and materially raise value relative to five annual payments |
| Discounting | Article supplies the stochastic DCF and terminal formula | Quarterly exponents for explicit nodes; terminal at year 5 | Internally consistent with the 20-node grid | Inherits the unresolved economic interpretation of that grid |
| Invalid paths | Sources do not specify tail handling | One invalid path invalidates the whole 50,000-path result | Prevents silent deletion, clipping or favorable resampling | Conservative; the probability of finding a rare invalid draw rises with path count |
| Scientific summary | Article figures are consistent with 1,000 simulations | Use all 50,000 paths for moments; retain nested convergence prefixes | Lower Monte Carlo error and explicit stability evidence | Costs more and cannot recover undisclosed author seeds or sampling choices |
| WACC and growth | Article treats them as valuation inputs | Require externally supplied short-term WACC, terminal WACC and growth | Keeps assumptions explicit and testable | Small changes can dominate value; the engine does not estimate or validate them |
| Point-in-time timing | Sources do not fully disclose the accounting-to-price bridge | Require timezone-aware availability and capital timestamps; forbid future revisions | Prevents mechanical look-ahead | Reliable use depends on evidence that supplied timestamps represent genuine public availability |

## Central interpretation risk

The quarterly TTM payment grid is the defining V4 choice and its main financial
risk. A TTM cash-flow estimate represents approximately one year of activity;
including a refreshed TTM amount every quarter can count overlapping activity
multiple times. The alternative five-annual-payment reading avoids overlap but
produces materially different valuations. Neither available source uniquely
settles the authors' implementation. V4 therefore exposes its choice in the
method identity rather than presenting it as a fact from the paper.

## Statistical limits

The sequential procedure controls resampling decision error conditional on the
fitted null distribution. It does not establish that the structural test has
exact 5% size for every nuisance-parameter configuration. Fixed fitted
parameters also understate predictive uncertainty when estimation error is
material. These limits should accompany any empirical interpretation.

## Data limits

Code-level validation cannot prove that accounting fields share definitions,
currency, units, restatement vintage, split basis, or genuine first-availability
timestamps. The caller owns that evidence. Passing the input contract proves
internal consistency, not economic equivalence across data sources.
