from dataclasses import dataclass

from cartesian import CartesianND


@dataclass(frozen=True)
class MagnitudeAndDirection:
    magnitude: float
    direction: CartesianND
