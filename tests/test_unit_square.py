import math

import pytest

from cartesian import Cartesian3D
from cartesian.operations import dot_product

from dispersion4b.potential import FourBodyDispersionPotential


def get_square_points(sidelen: float) -> list[Cartesian3D]:
    p0 = sidelen * Cartesian3D(0.0, 0.0, 0.0)
    p1 = sidelen * Cartesian3D(1.0, 0.0, 0.0)
    p2 = sidelen * Cartesian3D(1.0, 1.0, 0.0)
    p3 = sidelen * Cartesian3D(0.0, 1.0, 0.0)

    return [p0, p1, p2, p3]


def unit_square_energy_by_hand() -> float:
    """
    Calculate the 4-body dispersion interaction energy, for an interaction with
    a coefficient and a sidelength of 1.0, and for points in a collinear geometry.

    The calculation here was done by hand.
    """
    # distances:
    #    R_01 = R_12 = R_23 = R_30 = 1
    #    R_02 = R_13 = sqrt(2)
    total_pair_contrib = 4 * (1.0 / 1.0) ** 12 + 2 * (1.0 / math.sqrt(2.0)) ** 12

    # there are 4 triplet terms
    #
    # for (0-1-2) and (1-2-3)
    # - the unit vectors are orthogonal
    # - the pair distances are 1
    # for (0-1-3) and (0-2-3)
    # - the angle between the unit vectors is cos(135deg)
    # - one pair distance is 1, the other is sqrt(2)
    cos135 = math.cos(math.pi * (3.0 / 4.0))
    contrib_012 = (1.0 + (0.0) ** 2) / (1.0**12)
    contrib_013 = (1.0 + (cos135) ** 2) / ((1.0**6) * (math.sqrt(2.0) ** 6))

    total_triplet_contrib = 2 * contrib_012 + 2 * contrib_013

    # there are 3 quadruplet terms
    #
    # f(0-1-2-3)
    # - all pair distances are 1; the denominator is 1
    # - two of the dot products (u_01 * u_23) and (u_12 * u_30) are -1
    # - the rest are 0
    #
    # square terms: +2
    # triplet terms: 0
    # quad term: 0
    term_0123_denom = 1.0
    term_0123_numer = -1 + (1) * (2.0) + (-3) * (0.0) + (9) * (0.0)
    term_0123 = term_0123_numer / term_0123_denom

    # f(0-1-3-2)
    # - two pair distances are 1, the other two are sqrt(2); the denominator is 8
    # - four of the dot products give cos(135)
    # - one dot product gives -1
    # - one dot product gives 0
    #
    # square terms: 1 + 4*(cos135^2)
    # triplet terms: -6*(cos135^2)
    # quad term: 9*(cos135^4)
    #
    # f(0-2-1-3)
    # - very similar to f(0-1-3-2)
    # - two of the dot products are swapped in value, but the final result works out to be the same
    term_0132_denom = 8.0
    term_0132_numer = -2 * cos135**2 + 9 * cos135**4
    term_0132 = term_0132_numer / term_0132_denom

    total_quadruplet_contrib = 2.0 * (term_0123 + 2 * term_0132)

    return -(total_pair_contrib + total_triplet_contrib + total_quadruplet_contrib)


def test_unit_square_energy():
    c12_coeff = 1.0
    pot = FourBodyDispersionPotential(c12_coeff)

    sidelen = 1.0
    p0, p1, p2, p3 = get_square_points(sidelen)

    expect_energy = unit_square_energy_by_hand()
    actual_energy = pot(p0, p1, p2, p3)

    assert expect_energy == pytest.approx(actual_energy)
