import math

import pytest

from cartesian import Cartesian3D
from cartesian.operations import dot_product

from dispersion4b.potential import FourBodyDispersionPotential

# TODO:
# - try out other unit geometries, ones that are easy to calculate by hand
# - 4 particles in a straight line?
# - a square of unit side length?


def get_collinear_points(sidelen: float) -> list[Cartesian3D]:
    p0 = sidelen * Cartesian3D(0.0, 0.0, 0.0)
    p1 = sidelen * Cartesian3D(1.0, 0.0, 0.0)
    p2 = sidelen * Cartesian3D(2.0, 0.0, 0.0)
    p3 = sidelen * Cartesian3D(3.0, 0.0, 0.0)

    return [p0, p1, p2, p3]


def unit_collinear_energy_by_hand() -> float:
    """
    Calculate the 4-body dispersion interaction energy, for an interaction with
    a coefficient and a sidelength of 1.0, and for points in a collinear geometry.

    The calculation here was done by hand.
    """
    # distances:
    #    R_10 = R_20 = R_30 = 1
    #    R_20 = R_31 =        2
    #    R_30 =               3
    total_pair_contrib = \
        3 * (1.0/1.0)**12 + \
        2 * (1.0/2.0)**12 + \
        1 * (1.0/3.0)**12

    # there are 4 triplet terms
    #
    # the dot product between the unit vectors is always -1 or 1, but it gets squared,
    # so it doesn't matter
    # thus the numerator is always 2
    #
    # for (0-1-2) and (1-2-3) -> denominator is 1
    # for (0-1-3) and (0-2-3) -> denominator is 2^6
    total_triplet_contrib = \
        2 * (2.0 / (1.0**6)) + \
        2 * (2.0 / (2.0**6))

    # there are 3 quadruplet terms
    #
    # for the squared terms:
    # - each has a coefficient of +1
    # - the dot product between the unit vectors is always -1 or 1, but it gets squared,
    #   so it doesn't matter; the result is always 1
    #
    # for the "product of 3 dot product" terms:
    # - each has a coefficient of -3
    # - each unit vector appears twice in each term; the result is always 1
    #
    # for the "product of 4 dot product" term:
    # - it has a coefficient of +9
    # - again, each unit vector appears twice in each term; the result is always 1
    #
    quadruplet_numerator = \
        -1 + \
        6 * 1.0 * 1.0 + \
        4 * (-3.0) * 1.0 + \
        1 * (9.0) * 1.0
    
    # the denominators:
    # f(0-1-2-3) -> 27
    # f(0-1-3-2) -> 64
    # f(0-2-1-3) -> 1728
    
    total_quadruplet_contrib = \
        2.0 * quadruplet_numerator * (
        (1.0/27.0) + (1.0/64.0) + (1.0/1728.0)
    )

    return -(total_pair_contrib + total_triplet_contrib + total_quadruplet_contrib)


def test_unit_collinear_energy():
    c12_coeff = 1.0
    pot = FourBodyDispersionPotential(c12_coeff)

    sidelen = 1.0
    p0, p1, p2, p3 = get_collinear_points(sidelen)

    expect_energy = unit_collinear_energy_by_hand()
    actual_energy = pot(p0, p1, p2, p3)

    assert expect_energy == pytest.approx(actual_energy)
