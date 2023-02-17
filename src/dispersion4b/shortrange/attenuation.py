import numpy as np


def silvera_goldman_attenuation(r: float, b0: float, r_cutoff: float) -> float:
    """
    The short-long attenuation function matching the form given in the
    Silvera-Goldman potential.
    """
    if r > r_cutoff:
        return 1.0
    else:
        exponent = ((r_cutoff / r) - 1.0) ** 2
        return np.exp(-b0 * exponent)
