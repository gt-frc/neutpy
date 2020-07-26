from scipy import integrate
from scipy.interpolate import interp1d
import numpy as np


def calc_Ki3(x):
    return integrate.quad(lambda theta: (np.sin(theta)) ** 2 * np.exp(-x / np.sin(theta)), 0, np.pi / 2)[0]


# create bickley-naylor fit (much faster than evaluating Ki3 over and over)
xs = np.linspace(0, 100, 200)
Ki3 = interp1d(xs, np.array([calc_Ki3(x) for x in xs]))
