import math

import pytest
from cartesian import Cartesian3D
from cartesian.operations import centroid
from cartesian.measure import euclidean_distance as distance

from dispersion4b.shortrange.distance_parameter_function import sum_of_sidelengths
from dispersion4b.shortrange.distance_parameter_function import sum_of_com_distances

@pytest.fixture(scope='function')
def unit_tetrahedron_points():
    lat_const = 1.0
    points = [
        lat_const * Cartesian3D(-0.5, 0.0, 0.0),
        lat_const * Cartesian3D( 0.5, 0.0, 0.0),
        lat_const * Cartesian3D( 0.0, math.sqrt(3.0/4.0), 0.0),
        lat_const * Cartesian3D( 0.0, math.sqrt(1.0/12.0), math.sqrt(2.0/3.0)),
    ]

    yield points
    

@pytest.mark.usefixtures('unit_tetrahedron_points')
def test_sum_of_sidelengths(unit_tetrahedron_points):
    assert sum_of_sidelengths(unit_tetrahedron_points) == pytest.approx(6.0)


@pytest.mark.usefixtures('unit_tetrahedron_points')
def test_sum_of_com_distances(unit_tetrahedron_points):
    com = centroid(unit_tetrahedron_points)
    dist = distance(com, unit_tetrahedron_points[0])

    assert sum_of_com_distances(unit_tetrahedron_points) == pytest.approx(4 * dist)
