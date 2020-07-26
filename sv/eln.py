import numpy as np
from scipy.interpolate import Rbf


def _calc_sveln_st():
    """Calculates the

    Args:
        Tn (float): takes Tn in keV

    Returns:
        sveln (float):
    """
    # populate interpolation data
    eln = np.array([-1.4569E+01, -1.4167E+01, -1.3796E+01])

    tnnt = np.array([-3, -2, -1])

    return Rbf(tnnt, eln)
