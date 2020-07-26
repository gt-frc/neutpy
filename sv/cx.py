import numpy as np
from scipy.interpolate import Rbf


def _calc_svcx_st():
    """
    Calculates the

    Args:
        Ti (float): takes Ti in keV
        Tn (float): takes Tn in keV

    Returns:
        svcx (float): m^3/s
    """
    # populate interpolation data
    cx = np.array([[-1.4097E+01, -1.3921E+01, -1.3553E+01, -1.4097E+01, -1.3921E+01],
                   [-1.3553E+01, -1.3921E+01, -1.3824E+01, -1.3538E+01, -1.3553E+01],
                   [-1.3538E+01, -1.3432E+01, -1.3553E+01, -1.3538E+01, -1.3432E+01]])

    tint, tnnt = np.meshgrid(np.array([-4, -3, -2, -1, 0]),
                             np.array([-3, -2, -1]))

    return Rbf(tint, tnnt, cx)
