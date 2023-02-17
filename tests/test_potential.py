import math

import numpy as np
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


def test_raises_negative_c12_coeff():
    with pytest.raises(ValueError) as exc_info:
        FourBodyDispersionPotential(-1.0)

    assert "The C12 coefficient for the interaction must be positive.\n" in str(
        exc_info.value
    )


def test_inverse_r12_trend():
    """
    If the potential falls off as R^{-12}, then multiplying it by R^12 should
    make the result constant.
    """
    c12_coeff = 1.0
    pot = FourBodyDispersionPotential(c12_coeff)

    sidelengths = np.linspace(1.0, 5.0, 128)
    energies = np.array(
        [pot(*get_tetrahedron_points(sidelen)) for sidelen in sidelengths]
    )
    energies_times_r12 = energies * (sidelengths**12)

    mean_energy_times_r12 = np.mean(energies_times_r12)

    for eng_times_r12 in energies_times_r12:
        assert mean_energy_times_r12 == pytest.approx(eng_times_r12)
