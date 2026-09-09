# Methodology

## Scope

SDCF Engine V4 values one common-equity security at one fiscal snapshot. Inputs
must already be normalized into one currency and monetary scale and must carry
provider-independent fiscal identities and publication timestamps. The engine
does not decide whether a source field is economically appropriate.

## 1. Available accounting history

For an information cutoff, the engine excludes later observations and later
revisions. For each fiscal period it retains the latest vintage available by
that cutoff. Fiscal year and quarter labels define sequence; calendar quarter
boundaries do not override them.

Quarterly operating cash flow is reconstructed as

\[
\widetilde{CF}_t=(EBITDA_t-D\&A_t)(1-\tau_t)+D\&A_t-CAPEX_t.
\]

CAPEX is a positive expenditure in the input contract. No sign inference occurs
inside the engine.

## 2. Operating margin alpha

Four consecutive quarterly flows are summed to create quarterly-updated TTM
revenue and cash flow. The estimation window contains 20 to 66 complete TTM
observations. For each moving-average order \(q=0,\ldots,4\), V4 estimates

\[
\widetilde{CF}_t=\alpha REV_t+u_t,
\]

without an intercept and with Gaussian MA errors. Revenue and cash flow are
divided by the same positive revenue scale before estimation; alpha is therefore
unchanged by monetary units. AIC selects among valid fits on the same sample.

Initial eligibility uses the equation-error uncentered statistic

\[
R^2_u=1-\frac{\sum_t(\widetilde{CF}_t-\hat\alpha REV_t)^2}
                    {\sum_t\widetilde{CF}_t^2}.
\]

The firm is eligible at equality or above 0.10. This initial decision is carried
forward rather than retested at every later snapshot.

## 3. Working-capital margin beta

For each of the latest four consecutive complete fiscal years, annual revenue
is the sum of four quarterly revenue flows and working capital is the Q4 stock.
V4 estimates

\[
\beta=\frac{1}{4}\sum_{y=1}^{4}\frac{WC_y}{REV_y}.
\]

Missing years or quarters fail. Older nonconsecutive years are not substituted.

## 4. Revenue-model selection

TTM revenue is the rolling sum of four consecutive quarterly flows. Logarithms
are taken only after aggregation and require positive TTM revenue. An augmented
Dickey–Fuller test is applied to log-TTM growth with a constant, maximum lag 4,
and AIC lag selection.

If the ADF p-value is strictly below 0.05, Model 1 is selected from AR orders
0–4 by common-sample Gaussian AIC. Otherwise V4 compares:

- Model 2: Gaussian local level;
- Model 3: Gaussian local linear trend.

Both structural models use the same proper Gaussian likelihood for all
log-revenue increments. Differencing removes the unknown initial level. The
local-level null imposes zero initial slope and zero slope variance. A
parametric bootstrap refits both models to every null simulation.

The bootstrap is inspected after 100, 250, 500, 1,000, 2,000, 4,000, 7,000 and
9,999 replicas. Simultaneous exact binomial intervals control the probability
of making the wrong Monte Carlo decision at 0.001, conditional on the fitted
bootstrap law. If the final interval still crosses 0.05, selection is unresolved
and valuation stops.

## 5. Revenue simulation

V4 generates 50,000 paths over 20 fiscal quarters using independent PCG64
streams for bootstrap, filtered state and future innovations. Stream seeds are
SHA-256 functions of the master seed, method ID, security ID, fiscal date,
cutoff and purpose, so processing order cannot change a result.

Model 1 uses Gaussian AR-growth innovations. Structural models draw the final
filtered latent state, including its covariance, once per path and then apply
Gaussian state and observation innovations. Model parameters are held fixed.

## 6. Valuation

At each of the 20 quarterly forecast nodes, V4 converts TTM revenue into cash
flow using the article's identity

\[
CF_t=(\alpha-\beta)REV_t+\beta REV_{t-1}
    =\alpha REV_t-\beta(REV_t-REV_{t-1}).
\]

Every quarterly-updated TTM amount is included without division by four and is
discounted at \(t/4\) years. The five-year terminal value is

\[
TV=\frac{CF_{20}(1+g)}{(1+k_{TV})^5(k_{TV}-g)},
\qquad k_{TV}>g.
\]

Enterprise value is the sum of discounted explicit cash flows and terminal
value. Common-equity value is

\[
V^{Eq}=V-TD+Cash-MI-PS,
\]

and fair value per share divides by positive shares outstanding.

## 7. Distribution and z-score

Any nonfinite or nonpositive fair-value path invalidates the complete result;
paths are never dropped or replaced. V4 reports log-fair-value moments and level
quantiles from all 50,000 scientific paths. Nested prefixes at 25,000 and 50,000
must satisfy fixed stability tolerances.

For market price \(p\), mean log fair value \(\mu\), and sample standard
deviation \(\sigma\),

\[
z=\frac{\log p-\mu}{\sigma}.
\]

Market price is used only for this statistic and the convergence diagnostic; it
does not enter fair-value construction.
