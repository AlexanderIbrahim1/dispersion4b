import math

import pytest

from cartesian import Cartesian3D
from cartesian.measure import approx_eq
from cartesian.measure import euclidean_norm

from dispersion4b.magnitude_and_direction import MagnitudeAndDirection
from dispersion4b.utils import distance_and_unit_vector


def test_unit_vector():
    p0 = Cartesian3D(0.0, 0.0, 0.0)
    p1 = Cartesian3D(1.0, 0.0, 0.0)
    mad = distance_and_unit_vector(p0, p1)

    expected_magnitude = 1.0

    assert mad.magnitude == pytest.approx(expected_magnitude)
    assert approx_eq(mad.direction, Cartesian3D(-1.0, 0.0, 0.0))
    assert euclidean_norm(mad.direction) == pytest.approx(1.0)


def test_corner_vector():
    p0 = Cartesian3D(0.0, 0.0, 0.0)
    p1 = Cartesian3D(1.0, 2.0, 3.0)
    mad = distance_and_unit_vector(p0, p1)

    expected_magnitude = euclidean_norm(p1 - p0)

    assert mad.magnitude == pytest.approx(expected_magnitude)
    assert approx_eq(mad.direction, Cartesian3D(-1.0, -2.0, -3.0) / expected_magnitude)
    assert euclidean_norm(mad.direction) == pytest.approx(1.0)
