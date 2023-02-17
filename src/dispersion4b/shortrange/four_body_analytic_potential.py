"""
This module contains the FourBodyAnalyticPotential class, which is an analytic
version (as opposed to an 'ab initio' or a 'machine-learned' version) of the
four-body interaction potential energy surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated
from typing import Callable
from typing import Sequence

from cartesian import Cartesian3D
from dispersion4b.potential import FourBodyDispersionPotential


FourPoints = Annotated[Sequence[Cartesian3D], 4]


@dataclass
class FourBodyAnalyticPotential:
    """
    An analytic approximation to the true four-body parahydrogen interaction potential.

    dispersion_potential
    - the Bade potential, used at long intermolecular separations
    short_range_potential
    - the potential form used at short intermolecular separations; what counts as "short"
      is not strictly enforced; it is up to the user to make sure the "short_range_potential"
      dies off before the dispersion part of the Bade potential.
    short_long_attentuation
    - the Bade potential is not meant to be used at short distances; unfortunately, it becomes
      stronger in magnitude at short distances anyways. The attenuation function is a function
      that is equal to 1 when greater than a certain distance, and decays to zero as the
      intermolecular separations decrease.
    """

    dispersion_potential: FourBodyDispersionPotential
    short_range_potential: Callable[[FourPoints], float]
    short_long_attenuation: Callable[[FourPoints], float]

    def __call__(self, points: FourPoints) -> float:
        short_range_energy = self.short_range_potential(points)
        short_long_att_factor = self.short_long_attenuation(points)
        dispersion_energy = self.dispersion_potential(*points)

        return short_range_energy + (dispersion_energy * short_long_att_factor)
