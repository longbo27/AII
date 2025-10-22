# Metrics Reference (D1)

All annualizations assume **252 trading days per year**.

## Formulas

- **Total Return**: \( \text{equity}[-1] / \text{equity}[0] - 1 \)
- **CAGR**: \( (1 + \text{total\_return})^{252 / \text{trading\_days}} - 1 \)
- **Sharpe Ratio** (daily, risk-free \(\approx 0\)): \( \frac{\operatorname{mean}(\text{daily\_returns})}{\operatorname{std}(\text{daily\_returns})} \times \sqrt{252} \)
- **Sortino Ratio**: \( \frac{\operatorname{mean}(\text{daily\_returns})}{\operatorname{std}(\text{negative\_daily\_returns})} \times \sqrt{252} \)
- **Calmar Ratio**: \( \frac{\text{CAGR}}{|\text{max\_drawdown}|} \)
- **Annualized Volatility**: \( \operatorname{std}(\text{daily\_returns}) \times \sqrt{252} \)

## Conventions

- Transaction fees must be recorded as **negative contributions** to PnL; the
  risk configuration supplies both `slippage_bps` and `commission_per_share`.
- `cooldown_on` is a boolean. During cooldown periods **exposure must be 0**
  (flat book).
- Rebalance cadence definitions:
  - `daily`: every trading day.
  - `weekly`: Monday’s session (aligned with Day-0 semantics).
  - `monthly`: first trading day of the month.

These conventions are referenced by the README and all downstream reports.
