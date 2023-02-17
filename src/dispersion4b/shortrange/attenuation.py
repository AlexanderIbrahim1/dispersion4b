import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SilveraGoldmanAttenuation:
    """
    The short-long attenuation function matching the form given in the
    Silvera-Goldman potential.
    """

    r_cutoff: float
    expon_coeff: float

    def __post_init__(self) -> None:
        if self.r_cutoff <= 0.0:
            raise ValueError(
                "The cutoff distance for the attenuation function must be positive.\n"
                f"Entered: {self.r_cutoff: .12f}"
            )

        if self.expon_coeff <= 0.0:
            raise ValueError(
                "The coefficient in the exponent must be positive.\n"
                f"Entered: {self.expon_coeff: .12f}"
            )

    def __call__(self, r: float) -> float:
        if r >= self.r_cutoff:
            return 1.0
        else:
            exponent = ((self.r_cutoff / r) - 1.0) ** 2
            return math.exp(-self.expon_coeff * exponent)
