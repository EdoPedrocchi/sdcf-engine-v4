"""Tests for the immutable V4 method contract."""

from dataclasses import replace
from pathlib import Path

import pytest

from sdcf_engine import V4Method, load_v4_method
from sdcf_engine.exceptions import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]


def test_exact_configuration_loads() -> None:
    """Load the single complete method definition."""
    assert load_v4_method(ROOT / "config/v4.yaml") == V4Method()


def test_relabelled_or_modified_v4_fails() -> None:
    """Require a new version identity for every scientific change."""
    with pytest.raises(ConfigurationError):
        replace(V4Method(), scientific_paths=1_000).validate()
