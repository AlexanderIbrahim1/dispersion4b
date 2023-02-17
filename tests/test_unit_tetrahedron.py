import math

import pytest

from cartesian import Cartesian3D
from cartesian.operations import dot_product

from dispersion4b.potential import FourBodyDispersionPotential


def get_tetrahedron_points(sidelen: float) -> list[Cartesian3D]:
    p0 = sidelen * Cartesian3D(-0.5, 0.0, 0.0)
    p1 = sidelen * Cartesian3D(0.5, 0.0, 0.0)
    p2 = sidelen * Cartesian3D(0.0, math.sqrt(3.0 / 4.0), 0.0)
    p3 = sidelen * Cartesian3D(0.0, math.sqrt(1.0 / 12.0), math.sqrt(2.0 / 3.0))

    return [p0, p1, p2, p3]


def unit_tetrahedron_energy_by_hand() -> float:
    """
    Calculate the 4-body dispersion interaction energy, for an interaction with
    a coefficient and a sidelength of 1.0, and for points in a tetrahedron geometry.

    The calculation here was done by hand.
    """
    cos120 = math.cos((2.0 / 3.0) * math.pi)

    # there are 6 pair terms, and they all give 1.0
    indiv_pair_contrib = 1.0
    total_pair_contrib = 6 * indiv_pair_contrib

    # there are 4 triplet terms
    # the denominator of each is 1.0
    # the numerator of each is `1.0 + cos(120deg)^2`
    indiv_triplet_contrib = 1.0 + cos120**2
    total_triplet_contrib = 4 * indiv_triplet_contrib

    # there are 3 quadruplet terms
    # each of them is calculated using dot products of unit vectors
    #
    # consider a product where there is one index shared between the two vectors, like
    #   u_ij * u_jk
    # this is cos(120)
    #
    # consider a product where there is no index shared between the two vectors, like
    #   u_ij * u_kl
    # this is 0.0
    #
    # 4 of the "squared" terms contribute
    # none of the "groups of 3" terms contribute
    # the last term contributes, with a coefficient of 9
    indiv_quadruplet_contrib = 2.0 * (-1.0 + 4 * (cos120**2) + 9 * (cos120**4))
    total_quadruplet_contrib = 3 * indiv_quadruplet_contrib

    return -(total_pair_contrib + total_triplet_contrib + total_quadruplet_contrib)


def sidelen2_tetrahedron_energy_by_hand() -> float:
    """
    Calculate the 4-body dispersion interaction energy, for an interaction with
    a coefficient and a sidelength of 2.0, and for points in a tetrahedron geometry.

    The calculation here was done by hand.
    """
    cos120 = math.cos((2.0 / 3.0) * math.pi)

    # there are 6 pair terms, and they all give (1.0/2.0)**12
    indiv_pair_contrib = (1.0 / 2.0) ** 12
    total_pair_contrib = 6 * indiv_pair_contrib

    # there are 4 triplet terms
    # the denominator of each is (2.0)**12
    # the numerator of each is `1.0 + cos(120deg)^2`
    indiv_triplet_contrib = (1.0 + cos120**2) / (2.0**12)
    total_triplet_contrib = 4 * indiv_triplet_contrib

    # there are 3 quadruplet terms
    # each of them is calculated using dot products of unit vectors
    #
    # consider a product where there is one index shared between the two vectors, like
    #   u_ij * u_jk
    # this is cos(120)
    #
    # consider a product where there is no index shared between the two vectors, like
    #   u_ij * u_kl
    # this is 0.0
    #
    # 4 of the "squared" terms contribute
    # none of the "groups of 3" terms contribute
    # the last term contributes, with a coefficient of 9
    indiv_quadruplet_contrib = (
        2.0 * (-1.0 + 4 * (cos120**2) + 9 * (cos120**4)) / (2.0**12)
    )
    total_quadruplet_contrib = 3 * indiv_quadruplet_contrib

    return -(total_pair_contrib + total_triplet_contrib + total_quadruplet_contrib)


@pytest.mark.parametrize(
    "sidelen, energy_func_by_hand",
    [
        (1.0, unit_tetrahedron_energy_by_hand),
        (2.0, sidelen2_tetrahedron_energy_by_hand),
    ],
)
def test_unit_tetrahedron_energy(sidelen, energy_func_by_hand):
    c12_coeff = 1.0
    pot = FourBodyDispersionPotential(c12_coeff)

    p0, p1, p2, p3 = get_tetrahedron_points(sidelen)

    expect_energy = energy_func_by_hand()
    actual_energy = pot(p0, p1, p2, p3)

    assert expect_energy == pytest.approx(actual_energy)
