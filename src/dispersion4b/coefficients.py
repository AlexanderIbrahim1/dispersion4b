"""
This module contains functions that return the coefficients for the pair-, triple-,
and quadruple-dipole dispersion interactions between parahydrogen molecules.
"""


def c6_parahydrogen() -> float:
    """
    The C_6 coefficient for the dipole-dipole dispersion interaction between two
    parahydrogen molecules.

    Units: [cm^{-1}] [Angstrom]^{-6}

    Taken from table 6 in:
        M. Schmidt et al. "Raman vibrational shifts of small clusters of
        hydrogen isotopologues." J. Phys. Chem. A. 119, p. 12551-12561 (2015).
    """
    return 58203.64


def c9_parahydrogen() -> float:
    """
    The C_9 coefficient for the triple-dipole dispersion interaction between three
    parahydrogen molecules (i.e. the Axilrod-Teller-Muto potential).

    Units: [cm^{-1}] [Angstrom]^{-9}

    Converted from the value given at the bottom of paragraph 1 in section 3
    (Results and discussion section) of:
        R. J. Hinde. "Three-body interactions in solid parahydrogen." Chem. Phys.
        Lett. 460, p. 141-145 (2008).
    """
    return 34336.220013464925


def c12_parahydrogen_midzuno_kihara() -> float:
    """
    The C_12 coefficient for the quadruple-dipole dispersion interaction between
    four parahydrogen molecules.

    Units: [cm^{-1}] [Angstrom]^{-12}

    This is an approximation of the C_12 coefficient using a the Midzuno-Kihara
    approximation. We use the C_6 and C_9 coefficients to estimate the C_12
    coefficient.
    """
    c6_coeff = c6_parahydrogen()
    c9_coeff = c9_parahydrogen()

    return (5.0 * c9_coeff**2) / (12.0 * c6_coeff)


def c12_parahydrogen_avdz_approx() -> float:
    """
    The C_12 coefficient for the quadruple-dipole dispersion interaction between
    four parahydrogen molecules.

    Units: [cm^{-1}] [Angstrom]^{-12}

    This is an approximation of the coefficient based on the preliminary electronic
    structure calculations done for the four-body parahydrogen interaction.

    We calculate the four-body interaction energy for a tetrahedron with a parahydrogen
    molecule at each corner. The energies are calculated at the CCSD(T) level, using
    an AVDZ basis set. The energies are averaged using the L=3 lebedev quadrature,
    aligned along the xyz axes.

    The estimate of C_12 is based on the value of the interaction energy when the
    tetrahedron's side length is 4.25 Angstroms. This value was chosen because it is
    the turning point in the curve when the interaction energy is multiplied by the
    sidelength to the 12th power.

    The estimate is given as a ratio of the Midzuno-Kihara estimate.
    """
    mk_to_avdz_ratio = 0.8822

    return mk_to_avdz_ratio * c12_parahydrogen_midzuno_kihara()
