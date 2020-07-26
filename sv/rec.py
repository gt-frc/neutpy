import numpy as np
from scipy.interpolate import Rbf


def _calc_svrec_st():
    """Calculates the

    Args:
        ne (float): takes ne in m^-3
        Te (float): takes Te in keV

    Returns:
        svrec (float): m^3 / s

    """
    # populate interpolation data
    rec = np.array([[-1.7523E+01, -1.6745E+01, -1.5155E+01, -1.4222E+01, -1.3301E+01],
                    [-1.8409E+01, -1.8398E+01, -1.8398E+01, -1.7886E+01, -1.7000E+01],
                    [-1.9398E+01, -1.9398E+01, -1.9398E+01, -1.9398E+01, -1.9398E+01],
                    [-2.0155E+01, -2.0155E+01, -2.0155E+01, -2.0155E+01, -2.0155E+01],
                    [-2.1000E+01, -2.1000E+01, -2.1000E+01, -2.1000E+01, -2.1000E+01]])

    nint, tint = np.meshgrid(np.array([16, 18, 20, 21, 22]),
                             np.array([-4, -3, -2, -1, 0]))

    return Rbf(nint, tint, rec)
