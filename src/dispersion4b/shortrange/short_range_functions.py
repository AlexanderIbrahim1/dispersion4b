"""
This module contains functions used as the short-range potential portion of the
'FourBodyAnalyticPotential'.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ExponentialDecay:
    coeff: float
    expon: float

    def __post_init__(self) -> None:
        if self.expon <= 0.0:
            raise ValueError(
                "The coefficient in the exponent for the exponential decay must be positive.\n"
                f"Entered: {self.expon: .12f}"
            )

    def __call__(self, x: float) -> float:
        return self.coeff * math.exp(-self.expon * x)


@dataclass(frozen=True)
class ExponentialDecayOrder2:
    coeff: float
    expon_lin: float
    expon_sq: float

    def __post_init__(self) -> None:
        # it is *technically* fine for convergence if the coefficient for the linear term in
        # the exponent is nonpositive; as long as the coefficient for the squared term is
        # positive, the function will eventually converge
        if self.expon_sq <= 0.0:
            raise ValueError(
                "The coefficient in the exponent for the squared term must be positive.\n"
                f"Entered: {self.expon_sq: .12f}"
            )

    def __call__(self, x: float) -> float:
        exponent = (self.expon_lin * x) + (self.expon_sq * x**2)
        return self.coeff * math.exp(-exponent)
