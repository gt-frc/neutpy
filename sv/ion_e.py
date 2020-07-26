import numpy as np
from scipy.interpolate import Rbf


def _calc_svione_st():
    """Calculates the ionization cross section using stacey thomas data

    Args:
        ne (float): takes ne in m^-3
        Te (float): takes Te in keV

    Returns:
        svione (float):

    """
    # populate interpolation data
    eion = np.array([[-2.8523E+01, -2.8523E+01, -2.8523E+01, -2.8523E+01, -2.8523E+01],
                     [-1.7745E+01, -1.7745E+01, -1.7745E+01, -1.7745E+01, -1.7745E+01],
                     [-1.3620E+01, -1.3620E+01, -1.3620E+01, -1.3620E+01, -1.3620E+01],
                     [-1.3097E+01, -1.3097E+01, -1.3097E+01, -1.3097E+01, -1.3097E+01],
                     [-1.3301E+01, -1.3301E+01, -1.3301E+01, -1.3301E+01, -1.3301E+01]])

    nint, tint, = np.meshgrid(np.array([16, 18, 20, 21, 22]),
                              np.array([-4, -3, -2, -1, 0]))

    return Rbf(nint, tint, eion)
