from cartesian import CartesianND
from cartesian.measure import euclidean_norm

from dispersion4b.magnitude_and_direction import MagnitudeAndDirection


def distance_and_unit_vector(
    p_i: CartesianND, p_j: CartesianND
) -> MagnitudeAndDirection:
    """Calculate the distance and unit vector of the separation 'p_i - p_j'."""
    p_ij = p_i - p_j
    distance = euclidean_norm(p_ij)
    unit_vec = p_ij / distance

    return MagnitudeAndDirection(distance, unit_vec)
