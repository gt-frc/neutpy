import numpy as np
from helpers import midpoint2D, Ki3


def li(phi, xi):
    x_coords = x_coords - xi

    vert_phis = np.arctan2(y_coords, x_coords)
    vert_phis[0] = 0
    vert_phis[-1] = np.pi

    if phi < np.pi:
        reg = np.searchsorted(vert_phis, phi, side='right') - 1
    else:
        reg = np.searchsorted(vert_phis, phi, side='right') - 2

    # points defining the side of the cell we're going to intersect with
    # eq of line is y = ((y2-y2)/(x2-x1))(x-x1)+y1
    x1, y1 = x_coords[reg], y_coords[reg]
    x2, y2 = x_coords[reg + 1], y_coords[reg + 1]

    # calculate intersection point
    if np.isclose(x2, x1):  # then line is vertical
        x_int = x1
        y_int = np.tan(phi) * x_int
    else:
        # eq of the intersecting line is y= tan(phi)x ( + 0 because of coordinate system choice)
        # set two equations equal and solve for x, then solve for y
        x_int = ((y2 - y1) / (x2 - x1) * x1 - y1) / ((y2 - y1) / (x2 - x1) - np.tan(phi))
        y_int = np.tan(phi) * x_int

    return np.sqrt(x_int ** 2 + y_int ** 2)


def calc_t_coef(length):
    def integrand(phi, xi, x_comp, length, y_coords, reg, mfp):
        value =  np.sin(phi) \
                * Ki3(li(phi, xi, x_coords, y_coords, reg) / mfp)

        return value

    def phi_limits(xi, x_coords, y_coords, reg):
        x_coords = x_coords - xi
        vert_phis = np.arctan2(y_coords, x_coords)
        vert_phis[0] = 0
        vert_phis[-1] = np.pi
        return [vert_phis[reg], vert_phis[reg + 1]]

    def xi_limits(x_comp):
        return [0, -1 * x_comp[-1]]

    return 2 / (np.pi * length) * midpoint2D(integrand, phi_limits, xi_limits)
