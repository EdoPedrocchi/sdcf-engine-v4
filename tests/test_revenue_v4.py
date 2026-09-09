"""Tests for V4 revenue routing, sequential selection and simulation."""

import numpy as np
import pandas as pd
import pytest

from sdcf_engine.core.revenue_models import (
    RevenueModelFit,
    select_revenue_model,
    simulate_log_ttm_paths,
)
from sdcf_engine.core.sequential_bootstrap import SequentialPolicy
from sdcf_engine.core.state_likelihood import (
    UnresolvedBootstrapError,
    compare_state_models,
)


def test_stationary_growth_routes_to_model_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the ADF gate before fitting the five common-sample AR candidates."""
    monkeypatch.setattr(
        "sdcf_engine.core.revenue_models.adfuller",
        lambda *args, **kwargs: (-4.0, 0.01, 1, 30, {}, 0.0),
    )
    values = pd.Series(np.exp(6 + np.cumsum(np.sin(np.arange(36)) * 0.01)))
    selected = select_revenue_model(
        values,
        bootstrap_rng=np.random.default_rng(1),
        sequential_policy=SequentialPolicy(),
    )
    assert selected.fit.model_name == "model_1_ar_growth"
    assert len(selected.candidates) == 5
    assert selected.sequential_bootstrap is None


def test_sequential_extremes_and_unresolved_boundary() -> None:
    """Resolve clear evidence and retain inconclusive maximum-budget evidence."""
    policy = SequentialPolicy()
    assert policy.inspect(250, 0).decision == "local_trend"
    assert policy.inspect(100, 100).decision == "local_level"
    assert policy.inspect(9999, 500).decision == "unresolved"


def test_actual_sequential_fit_stops_on_a_declared_look() -> None:
    """Refit both structural models for every parametric-null replicate."""
    rng = np.random.default_rng(7501)
    values = np.cumsum(rng.normal(0, 0.02, 20)) + rng.normal(0, 0.01, 20)
    result = compare_state_models(
        values,
        rng=np.random.default_rng(17501),
        replicates=250,
        sequential_policy=SequentialPolicy(looks=(100, 250)),
    )
    assert result.looks[-1].replicates in {100, 250}
    assert len(result.replicate_statistics) == result.looks[-1].replicates
    if result.looks[-1].decision == "unresolved":
        with pytest.raises(UnresolvedBootstrapError):
            _ = result.selected


def test_model_one_simulation_is_seed_reproducible() -> None:
    """Preserve the same explicit future-innovation stream."""
    fit = RevenueModelFit(
        "model_1_ar_growth",
        np.log(1_000.0),
        (0.2,),
        0.01,
        (0.005,),
        (),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        (),
    )
    first = simulate_log_ttm_paths(
        fit,
        steps=20,
        paths=100,
        rng=np.random.default_rng(2),
        state_rng=np.random.default_rng(3),
    )
    repeated = simulate_log_ttm_paths(
        fit,
        steps=20,
        paths=100,
        rng=np.random.default_rng(2),
        state_rng=np.random.default_rng(3),
    )
    np.testing.assert_array_equal(first, repeated)


def test_structural_simulation_draws_filtered_state_uncertainty() -> None:
    """Do not collapse a nonzero filtered-state covariance to its mean."""
    fit = RevenueModelFit(
        "model_2_local_level",
        np.log(1_000.0),
        (),
        0.0,
        (),
        (np.log(1_000.0),),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        (),
        ((0.04,),),
    )
    paths = simulate_log_ttm_paths(
        fit,
        steps=20,
        paths=100_000,
        rng=np.random.default_rng(6),
        state_rng=np.random.default_rng(7),
    )
    assert np.var(paths[0]) == pytest.approx(0.04, rel=0.02)
    np.testing.assert_array_equal(paths[0], paths[-1])
