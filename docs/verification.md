# Verification

The repository uses synthetic data only. Verification covers accounting
identities, alpha scaling and MA selection, annual beta, revenue routing,
increment likelihood, sequential stopping, filtered-state simulation, the
20-node quarterly valuation, terminal value, equity bridge, invalid paths,
z-score alignment, configuration validation, CLI output and no-look-ahead
behavior.

## Source-V4 equivalence

The extracted implementation was compared directly with the source V4 under
the same Python environment and deterministic inputs.

| Controlled case | Comparison |
|---|---|
| Full Model 1 snapshot | Method ID, three stream seeds, alpha, beta, selected model, all 50,000 enterprise-value paths, all 50,000 fair-value paths and z-score matched exactly |
| Local-level structural history | Observed LR, all 100 completed bootstrap statistics, decision, filtered fit and simulated paths matched exactly |
| Local-trend structural history | Observed LR, all 250 completed bootstrap statistics, decision, filtered fit and simulated paths matched exactly |

The reduced 100/250-look structural cases verify refactoring equivalence; they
do not replace V4's production maximum of 9,999 bootstrap replicas.

## Reproduction boundary

These checks establish behavioral equivalence of the extracted scientific code
on controlled inputs. They do not validate any empirical dataset, recover the
authors' private implementation, or prove that V4's assumptions are the only
valid reading of the published method.
