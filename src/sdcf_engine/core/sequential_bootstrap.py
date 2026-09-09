"""V4 finite-look Monte Carlo decisions with simultaneous exact intervals.

Bonferroni coverage over predeclared looks controls simulation decision error
conditional on the fitted bootstrap law. It does not establish the statistical
size of that law under unknown nuisance parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scipy.stats import beta

SequentialDecision = Literal["local_level", "local_trend", "unresolved"]


@dataclass(frozen=True)
class SequentialLook:
    """Retained binomial evidence; the interval, not the fraction, drives selection."""

    replicates: int
    exceedances: int
    lower: float
    upper: float
    decision: SequentialDecision


@dataclass(frozen=True)
class SequentialPolicy:
    """Predeclared inspection schedule and per-comparison Monte Carlo risk."""

    looks: tuple[int, ...] = (100, 250, 500, 1000, 2000, 4000, 7000, 9999)
    resampling_risk: float = 0.001
    significance: float = 0.05

    def validate(self) -> None:
        """Reject adaptive, duplicated, unordered or nonpositive look counts."""
        if (
            not isinstance(self.looks, tuple)
            or not self.looks
            or any(type(n) is not int or n < 1 for n in self.looks)
            or tuple(sorted(set(self.looks))) != self.looks
            or not 0 < self.resampling_risk < 1
            or not 0 < self.significance < 1
        ):
            raise ValueError("Invalid predeclared sequential bootstrap policy.")

    def inspect(self, replicates: int, exceedances: int) -> SequentialLook:
        """Construct an exact interval with error risk divided across all looks.

        Parameters
        ----------
        replicates, exceedances:
            Completed iid draws and inclusive LR exceedances at a declared look.

        Returns
        -------
        SequentialLook
            Decision relative to the ideal conditional tail probability.
        """
        self.validate()
        if (
            type(replicates) is not int
            or type(exceedances) is not int
            or replicates not in self.looks
            or not 0 <= exceedances <= replicates
        ):
            raise ValueError("Sequential inspection requires valid declared counts.")
        tail = self.resampling_risk / (2 * len(self.looks))
        lower = (
            0.0
            if exceedances == 0
            else float(beta.ppf(tail, exceedances, replicates - exceedances + 1))
        )
        upper = (
            1.0
            if exceedances == replicates
            else float(beta.ppf(1 - tail, exceedances + 1, replicates - exceedances))
        )
        decision: SequentialDecision = "unresolved"
        if upper < self.significance:
            decision = "local_trend"
        elif lower >= self.significance:
            decision = "local_level"
        return SequentialLook(replicates, exceedances, lower, upper, decision)
