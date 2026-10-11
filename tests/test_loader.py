"""
tests/test_loader.py
====================
Unit tests for src/data/loader.py (split_bounds function).
"""

import pytest
from data.loader import split_bounds


def test_split_bounds_240_steps():
    train_bound, val_bound, test_bound = split_bounds(240, fracs=(0.70, 0.15, 0.15))
    assert train_bound == (0, 168)
    assert val_bound == (168, 204)
    assert test_bound == (204, 240)


def test_split_bounds_480_steps():
    train_bound, val_bound, test_bound = split_bounds(480, fracs=(0.70, 0.15, 0.15))
    assert train_bound == (0, 336)
    assert val_bound == (336, 408)
    assert test_bound == (408, 480)


def test_split_bounds_custom_fraction():
    train_bound, val_bound, test_bound = split_bounds(100, fracs=(0.80, 0.10, 0.10))
    assert train_bound == (0, 80)
    assert val_bound == (80, 90)
    assert test_bound == (90, 100)


def test_split_bounds_invalid_fractions():
    with pytest.raises(ValueError, match="Fractions must sum to 1.0"):
        split_bounds(100, fracs=(0.6, 0.2, 0.1))
