"""
The 'DistanceParameterFunction' is a callable that takes a collection of points
in cartesian space, and uses their relative separations to calculation a single
floating point value.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Callable
from typing import Sequence

from cartesian import Cartesian3D
from cartesian.measure import euclidean_distance as distance
from cartesian.operations import centroid


@dataclass
class DistanceParameterFunction:
    """
    function:
        - calculates some value using the distance parameter as the only argument
    dist_param_calculator:
        - calculates some value from the four Cartesian3D points that represent
          the positions of the particles
    """

    function: Callable[[float], float]
    dist_param_calculator: Callable[[Sequence[Cartesian3D]], float]

    def __call__(self, points: Sequence[Cartesian3D]) -> float:
        dist_param = self.dist_param_calculator(points)
        return self.function(dist_param)


def sum_of_sidelengths(points: Sequence[Cartesian3D]) -> float:
    return sum([distance(p0, p1) for (p0, p1) in combinations(points, 2)])


def sum_of_com_distances(points: Sequence[Cartesian3D]) -> float:
    com = centroid(points)
    return sum([distance(p, com) for p in points])
