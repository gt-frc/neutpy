import numpy as np
from scipy.interpolate import Rbf


def _calc_svel_st():
    """Calculates the

    Args:
        Ti (float): takes Ti in keV
        Tn (float): takes Tn in keV

    Returns:
        svel (float):

    """
    # populate interpolation data
    el = np.array([[-1.3569E+01, -1.3337E+01, -1.3036E+01, -1.3569E+01, -1.3337E+01],
                   [-1.3036E+01, -1.3337E+01, -1.3167E+01, -1.3046E+01, -1.3036E+01],
                   [-1.3046E+01, -1.2796E+01, -1.3036E+01, -1.3046E+01, -1.2796E+01]])

    tint, tnnt = np.meshgrid(np.array([-4, -3, -2, -1, 0]),
                             np.array([-3, -2, -1]))

    return Rbf(tint, tnnt, el)
