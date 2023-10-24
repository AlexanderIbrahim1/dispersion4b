"""
Calculate the quadruple-dipole (typo warning: quadruPLE, and *not* quadruPOLE")
dispersion interaction energy between four identical pointwise particles.

This code is based on equations (1) and (2), with N == 4, taken from:
    W. L. Bade. "Drude-Model calculation of dispersion forces. III. The fourth-order
    contribution", J. Chem Phys., 28 (1957).

This version of the function removes the pair and triplet components of the Bade
interaction potential, leaving only the quadruplet components.
"""

from __future__ import annotations

from cartesian import CartesianND
from cartesian.operations import dot_product

from dispersion4b.magnitude_and_direction import MagnitudeAndDirection
from dispersion4b.utils import distance_and_unit_vector


class QuadrupletDispersionPotential:
    """
    Calculate the quadruplet contribution to the dipole^4 dispersion
    interaction energy between four identical pointwise particles.
    """

    _coeff: float  # coefficient determining interaction strength

    def __init__(self, coeff: float) -> None:
        self._check_coeff_positive(coeff)
        self._coeff = coeff

    def __call__(
        self, p0: CartesianND, p1: CartesianND, p2: CartesianND, p3: CartesianND
    ) -> float:
        # calculate the distances and unit vectors between each pair of points
        # i.e. describe the vector as an arrow with a magnitude and direction
        vec10 = distance_and_unit_vector(p1, p0)
        vec20 = distance_and_unit_vector(p2, p0)
        vec30 = distance_and_unit_vector(p3, p0)
        vec21 = distance_and_unit_vector(p2, p1)
        vec31 = distance_and_unit_vector(p3, p1)
        vec32 = distance_and_unit_vector(p3, p2)

        # calculate all the dot product combinations
        # NOTE: the equation in the paper has some of the indices swapped compared to
        #       what I use; however, the unit vectors in each term of the quadruplet
        #       contribution come in pairs, so I don't think swapping the direction
        #       of a unit vector matters.
        #
        #     : physically, this would be like saying "swapping the positions of any
        #       two of the four identical particles should not change the energy",
        #       which makes sense.
        total_energy = 2.0 * (
            _quadruplet_contribution(vec30, vec32, vec21, vec10)
            + _quadruplet_contribution(vec20, vec32, vec31, vec10)
            + _quadruplet_contribution(vec20, vec21, vec31, vec30)
        )

        return -self._coeff * total_energy

    def _check_coeff_positive(self, coeff: float) -> None:
        if coeff <= 0.0:
            raise ValueError(
                "The C12 coefficient for the interaction must be positive.\n"
                f"Entered: coeff = {coeff}"
            )


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
