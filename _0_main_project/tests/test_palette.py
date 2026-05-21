"""Sanity check that the palette contract (PROJECT_STATE / refactor brief)
is enforced exactly. Any drift here is a review-blocker.
"""
from __future__ import annotations

from src.utils.plotting import (
    ACCENT,
    CLASS_COLORS,
    PALETTE,
    PRIMARY,
    REFERENCE,
    SECONDARY,
    TERTIARY_A,
    TERTIARY_B,
)


EXPECTED = {
    "primary": "#882255",     # Wine
    "secondary": "#4477AA",   # Blue
    "tertiary_a": "#44AA99",  # Teal
    "tertiary_b": "#DDCC77",  # Sand
    "accent": "#CC6677",      # Rose
    "reference": "#98A4B0",   # Grey
}


def test_palette_constants_match_contract():
    assert PRIMARY == EXPECTED["primary"]
    assert SECONDARY == EXPECTED["secondary"]
    assert TERTIARY_A == EXPECTED["tertiary_a"]
    assert TERTIARY_B == EXPECTED["tertiary_b"]
    assert ACCENT == EXPECTED["accent"]
    assert REFERENCE == EXPECTED["reference"]


def test_palette_dict_complete():
    assert PALETTE == EXPECTED


def test_class_colors_use_primary_for_positive():
    # Positive class (1 = stimulus) is mapped to PRIMARY (Wine).
    # Negative class (0 = control) is mapped to SECONDARY (Blue).
    assert CLASS_COLORS[0] == SECONDARY
    assert CLASS_COLORS[1] == PRIMARY
