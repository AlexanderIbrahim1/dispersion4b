"""
This module contains functions used as the short-range potential portion of the
'FourBodyAnalyticPotential'.
"""

import numpy as np


def exponential_decay(r: float, a: float, b: float) -> float:
    return a * np.exp(-b * r)


def exponential_decay_order2(r: float, a: float, b0: float, b1: float) -> float:
    exponent = (b0 * r) + (b1 * r**2)
    return a * np.exp(-exponent)
