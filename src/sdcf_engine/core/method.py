"""Define and validate the single supported SDCF V4 method contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

import yaml

from sdcf_engine.exceptions import ConfigurationError


@dataclass(frozen=True)
class V4Method:
    """Immutable settings for the only engine version exposed by this package."""

    method_id: str = "sdcf_engine_candidate_v4_sequential_quarterly_ttm"
    alpha_convention: str = "normalized_equation_errors"
    alpha_minimum_observations: int = 20
    alpha_maximum_observations: int = 66
    alpha_eligibility_threshold: float = 0.10
    beta_observations: int = 4
    beta_schedule: str = "available_fiscal_year_end"
    revenue_history: str = "complete_available_history"
    adf_significance: float = 0.05
    adf_maximum_lag: int = 4
    state_likelihood: str = "exact_log_increment_gaussian"
    bootstrap_replicates: int = 9999
    bootstrap_significance: float = 0.05
    bootstrap_looks: tuple[int, ...] = (100, 250, 500, 1000, 2000, 4000, 7000, 9999)
    bootstrap_resampling_risk: float = 0.001
    state_origin: str = "filtered_distribution"
    forecast_steps: int = 20
    cash_flow_frequency: Literal["quarterly_ttm"] = "quarterly_ttm"
    scientific_paths: int = 50000
    display_paths: int = 1000
    prefix_paths: tuple[int, ...] = (1000, 5000, 10000, 25000, 50000)
    master_seed: int = 2023

    def validate(self) -> None:
        """Reject any change made under the stable V4 method identity."""
        expected = V4Method()
        if type(self) is not V4Method or any(
            type(getattr(self, field.name)) is not type(getattr(expected, field.name))
            or getattr(self, field.name) != getattr(expected, field.name)
            for field in fields(self)
        ):
            raise ConfigurationError(
                "Settings differ from SDCF V4; create a new method version."
            )


def load_v4_method(path: Path) -> V4Method:
    """Load an exact V4 YAML contract and reject missing or unknown fields."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = asdict(V4Method())
    if not isinstance(payload, dict) or set(payload) != set(expected):
        raise ConfigurationError(
            "V4 configuration requires exactly the declared fields."
        )
    values: dict[str, object] = {}
    for name, reference in expected.items():
        observed = payload[name]
        if isinstance(reference, tuple):
            if not isinstance(observed, list) or any(
                type(item) is not int for item in observed
            ):
                raise ConfigurationError(f"V4 {name} requires an integer sequence.")
            observed = tuple(observed)
        if type(observed) is not type(reference) or observed != reference:
            raise ConfigurationError(
                f"V4 configuration differs at {name}; create a new method version."
            )
        values[name] = observed
    method = V4Method(**values)  # type: ignore[arg-type]
    method.validate()
    return method
