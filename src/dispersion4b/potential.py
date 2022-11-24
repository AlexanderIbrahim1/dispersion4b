"""
Calculate the quadruple-dipole (typo warning: quadruPLE, and *not* quadruPOLE")
dispersion interaction energy between four identical pointwise particles.

This code is based on equations (1) and (2), with N == 4, taken from:
    W. L. Bade. "Drude-Model calculation of dispersion forces. III. The fourth-order
    contribution", J. Chem Phys., 28 (1957).

TODO: are there better names for this?
    : there was no confusion with "triple-dipole" for the three-body interaction, but
    : "quadruple-dipole" looks too similar to "quadrupole-dipole".
    :
    : "dipole-dipole-dipole-dipole"?
    : "fourth-order dipole"?
"""

from __future__ import annotations

from dataclasses import dataclass

from cartesian import CartesianND
from cartesian.measure import euclidean_norm
from cartesian.operations import dot_product


@dataclass(frozen=True)
class MagnitudeAndDirection:
    magnitude: float
    direction: CartesianND


class FourBodyDispersionPotential:
    """
    Calculate the dipole^4 dispersion interaction energy between four identical
    pointwise particles.
    """

    _c12_coeff: float  # coefficient determining interaction strength

    def __init__(self, c12_coeff: float) -> None:
        self._check_c12_coeff_positive(c12_coeff)

        self._c12_coeff = c12_coeff

    def __call__(
        self, p0: CartesianND, p1: CartesianND, p2: CartesianND, p3: CartesianND
    ) -> float:
        # calculate the distances and unit vectors between each pair of points
        # i.e. describe the vector as an arrow with a magnitude and direction
        vec10 = _distance_and_unit_vector(p1, p0)
        vec20 = _distance_and_unit_vector(p2, p0)
        vec30 = _distance_and_unit_vector(p3, p0)
        vec21 = _distance_and_unit_vector(p2, p1)
        vec31 = _distance_and_unit_vector(p3, p1)
        vec32 = _distance_and_unit_vector(p3, p2)

        total_energy = 0.0

        # the pair contribution
        for vec in [vec10, vec20, vec30, vec21, vec31, vec32]:
            total_energy += _pair_contribution(vec.magnitude)

        # the triplet contribution
        total_energy += _triplet_contribution(vec21, vec10)
        total_energy += _triplet_contribution(vec31, vec10)
        total_energy += _triplet_contribution(vec32, vec20)
        total_energy += _triplet_contribution(vec32, vec21)

        # calculate all the dot product combinations
        # NOTE: the equation in the paper has some of the indices swapped compared to
        #       what I use; however, the unit vectors in each term of the quadruplet
        #       contribution come in pairs, so I don't think swapping the direction
        #       of a unit vector matters.
        #
        #     : physically, this would be like saying "swapping the positions of any
        #       two of the four particles should not change the energy", which makes
        #       sense.
        #
        #     : I should unit test it anyways
        total_energy += 2.0 * _quadruplet_contribution(vec30, vec32, vec21, vec10)
        total_energy += 2.0 * _quadruplet_contribution(vec20, vec32, vec31, vec10)
        total_energy += 2.0 * _quadruplet_contribution(vec20, vec21, vec31, vec30)

        return self._c12_coeff * total_energy

    def _check_c12_coeff_positive(self, c12_coeff: float) -> None:
        if c12_coeff <= 0.0:
            raise ValueError(
                "The C12 coefficient for the interaction must be positive.\n"
                f"Entered: c12_coeff = {c12_coeff}"
            )


def _distance_and_unit_vector(
    p_i: CartesianND, p_j: CartesianND
) -> MagnitudeAndDirection:
    """Calculate the distance and unit vector of the separation 'p_i - p_j'."""
    p_ij = p_i - p_j
    distance = euclidean_norm(p_ij)
    unit_vec = p_ij / distance

    return MagnitudeAndDirection(distance, unit_vec)


def _pair_contribution(distance: float) -> float:
    """The two-particle contribution to the 4-body dispersion energy."""
    return 1.0 / (distance**12)


def _triplet_contribution(
    vec_ij: MagnitudeAndDirection,
    vec_jk: MagnitudeAndDirection,
) -> float:
    """The three-particle contribution to the 4-body dispersion energy."""
    cosine_ijk = dot_product(vec_ij.direction, vec_jk.direction)

    numer = 1.0 + cosine_ijk**2
    denom = (vec_ij.magnitude * vec_jk.magnitude) ** 6

    return numer / denom


def _quadruplet_contribution(
    vec_ij: MagnitudeAndDirection,
    vec_jk: MagnitudeAndDirection,
    vec_kl: MagnitudeAndDirection,
    vec_li: MagnitudeAndDirection,
) -> float:
    """The four-particle contribution to the 4-body dispersion energy."""

    # the distance term
    denom = (
        vec_ij.magnitude * vec_jk.magnitude * vec_kl.magnitude * vec_li.magnitude
    ) ** 3

    prod_ijjk = dot_product(vec_ij.direction, vec_jk.direction)
    prod_ijkl = dot_product(vec_ij.direction, vec_kl.direction)
    prod_ijli = dot_product(vec_ij.direction, vec_li.direction)
    prod_jkkl = dot_product(vec_jk.direction, vec_kl.direction)
    prod_jkli = dot_product(vec_jk.direction, vec_li.direction)
    prod_klli = dot_product(vec_kl.direction, vec_li.direction)

    # begin with the constant contribution
    numer = -1.0

    # the squared pair prods
    numer += (
        prod_ijjk**2
        + prod_ijkl**2
        + prod_ijli**2
        + prod_jkkl**2
        + prod_jkli**2
        + prod_klli**2
    )

    # the triplets
    numer -= 3.0 * (
        (prod_ijjk * prod_jkkl * prod_ijkl)
        + (prod_ijjk * prod_jkli * prod_ijli)
        + (prod_ijkl * prod_klli * prod_ijli)
        + (prod_jkkl * prod_klli * prod_jkli)
    )

    # the quadruplet term
    numer += 9.0 * (prod_ijjk * prod_jkkl * prod_klli * prod_ijli)

    return numer / denom
