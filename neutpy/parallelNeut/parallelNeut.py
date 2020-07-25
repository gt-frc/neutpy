#!/usr/bin/python
from math import pi, sin, tan, sqrt
import numpy as np
from scipy import integrate
import sys

def f(phi, xi, x_comp, y_comp, x_coords, y_coords, reg, mfp, fromcell, tocell, throughcell, Ki3_fit, li):
    try:
        result = (2.0 / (pi * -1 * x_comp[-1])) * sin(phi) * Ki3_fit(li(phi, xi, x_coords, y_coords, reg) / mfp)
        return result
    except:
        print()
        print('something went wrong when evaluating A transmission coefficient:')
        print('li = ', li(phi, xi, x_coords, y_coords, reg))
        print('mfp = ', mfp)
        print('li/mfp = ', li(phi, xi, x_coords, y_coords, reg) / mfp)
        print('fromcell = ', fromcell)
        print('tocell = ', tocell)
        print('throughcell = ', throughcell)
        print()
        if li(phi, xi, x_coords, y_coords, reg) / mfp > 100:
            result = (2.0 / (pi * -1 * x_comp[-1])) * sin(phi) * Ki3_fit(100.0)
            return result


def li(phi, xi, x_coords, y_coords, reg):
    x_coords = x_coords - xi

    vert_phis = np.arctan2(y_coords, x_coords)
    vert_phis[0] = 0
    vert_phis[-1] = pi

    if phi < pi:
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
        y_int = tan(phi) * x_int
    else:
        # eq of the intersecting line is y= tan(phi)x ( + 0 because of coordinate system choice)
        # set two equations equal and solve for x, then solve for y
        x_int = ((y2 - y1) / (x2 - x1) * x1 - y1) / ((y2 - y1) / (x2 - x1) - tan(phi))
        y_int = tan(phi) * x_int

    return sqrt(x_int ** 2 + y_int ** 2)


def phi_limits(xi, x_comp, y_comp, x_coords, y_coords, reg, mfp, fromcell, tocell, throughcell, Ki3_fit, li):
    x_coords = x_coords - xi
    vert_phis = np.arctan2(y_coords, x_coords)
    vert_phis[0] = 0
    vert_phis[-1] = pi
    return [vert_phis[reg], vert_phis[reg + 1]]


def xi_limits(x_comp, y_comp, x_coords, y_coords, reg, mfp, fromcell, tocell, throughcell, Ki3_fit, li):
    return [0, -1 * x_comp[-1]]


def coeff_calc(inputs, *args, **kwargs):
    i = inputs[0][0]
    j = inputs[0][1]
    k = inputs[0][2]

    def midpoint2D(f, f_limx, f_limy, nx, ny, **kwargs):
        """calculates a double integral using the midpoint rule"""
        I = 0
        # start with outside (y) limits of integration
        c, d = f_limy(**kwargs)
        hy = (d - c) / float(ny)
        for j in range(ny):
            yj = c + hy / 2 + j * hy
            # for each j, calculate inside limits of integration
            a, b = f_limx(yj, **kwargs)
            hx = (b - a) / float(nx)
            for i in range(nx):
                xi = a + hx / 2 + i * hx
                I += hx * hy * f(xi, yj, **kwargs)
        return I

    nSides = kwargs['nSides']
    adjCell = kwargs['adjCell']
    lsides = kwargs['lsides']
    T_from = kwargs['T_from']
    T_to = kwargs['T_to']
    T_via = kwargs['T_via']
    int_method = kwargs['int_method']
    T_coef_s = kwargs['T_coef_s']
    T_coef_t = kwargs['T_coef_t']
    face_mfp_t = kwargs['face_mfp_t']
    face_mfp_s = kwargs['face_mfp_s']
    print_progress = kwargs['print_progress']
    outof = kwargs['outof']
    selfAngles = kwargs['angles']
    Ki3_fit = kwargs['Ki3_fit']
    li = kwargs['li']

    # progress = nSides[i] ** 2 * i  # + self.nSides[i]*j + k

    L_sides = np.roll(lsides[i, :nSides[i]], -(j + 1))  # begins with length of the current "from" side
    adj_cells = np.roll(adjCell[i, :nSides[i]], -j)
    angles = np.roll(selfAngles[i, :nSides[i]], -j) * 2 * pi / 360  # converted to radians
    angles[1:] = 2 * pi - (pi - angles[1:])

    if k < adj_cells.size and j < adj_cells.size:

        T_from[i, j, k] = adj_cells[0]
        T_to[i, j, k] = adj_cells[k - j]
        T_via[i, j, k] = i
        if j == k:
            # All flux from a side back through itself must have at least one collision
            T_coef_s[i, j, k] = 0.0
            T_coef_t[i, j, k] = 0.0
            # trans_coef_file.write(
            #     ('{:>6d}' * 3 + '{:>12.3E}' * 4 + '\n').format(int(T_from[i, j, k]), int(T_to[i, j, k]),
            #                                                    int(T_via[i, j, k]), T_coef_s[i, j, k],
            #                                                    T_coef_t[i, j, k], face.mfp.s[i, k], face.mfp.t[i, k]))
        else:
            side_thetas = np.cumsum(angles)

            x_comp = np.cos(side_thetas) * L_sides
            y_comp = np.sin(side_thetas) * L_sides

            y_coords = np.roll(np.flipud(np.cumsum(y_comp)), -1)
            x_coords = np.roll(np.flipud(np.cumsum(x_comp)),
                               -1)  # this gets adjusted for xi later, as part of the integration process

            reg = np.where(np.flipud(adj_cells[1:]) == T_to[i, j, k])[0][0]

            if int_method == 'midpoint':

                kwargs_s = {"x_comp": x_comp,
                            "y_comp": y_comp,
                            "x_coords": x_coords,
                            "y_coords": y_coords,
                            "reg": reg,
                            "mfp": face_mfp_s[i, j],  # not sure if this is j or k
                            "fromcell": adj_cells[0],
                            "tocell": adj_cells[k - j],
                            "throughcell": i}

                kwargs_t = {"x_comp": x_comp,
                            "y_comp": y_comp,
                            "x_coords": x_coords,
                            "y_coords": y_coords,
                            "reg": reg,
                            "mfp": face_mfp_t[i, j],
                            "fromcell": adj_cells[0],
                            "tocell": adj_cells[k - j],
                            "throughcell": i}
                nx = 10
                ny = 10

                T_coef_t[i, j, k] = midpoint2D(f, phi_limits, xi_limits, nx, ny, **kwargs_t)
                T_coef_s[i, j, k] = midpoint2D(f, phi_limits, xi_limits, nx, ny, **kwargs_s)

            elif int_method == 'quad':
                # T_coef_s[i, j, k] = 0
                # T_coef_t[i, j, k] = 0

                T_coef_s[i, j, k] = integrate.nquad(f, [phi_limits, xi_limits],
                                                    args=(x_comp,
                                                          y_comp,
                                                          x_coords,
                                                          y_coords,
                                                          reg,
                                                          face_mfp_s[i, j],
                                                          adj_cells[0],
                                                          adj_cells[k - j],
                                                          i,
                                                          Ki3_fit,
                                                          li),
                                                    opts=dict([('epsabs', 1.49e-2),
                                                               ('epsrel', 10.00e-4),
                                                               ('limit', 2)]))[0]

                T_coef_t[i, j, k] = integrate.nquad(f, [phi_limits, xi_limits],
                                                    args=(x_comp,
                                                          y_comp,
                                                          x_coords,
                                                          y_coords,
                                                          reg,
                                                          face_mfp_t[i, j],
                                                          adj_cells[0],
                                                          adj_cells[k - j],
                                                          i,
                                                          Ki3_fit,
                                                          li),
                                                    opts=dict([('epsabs', 1.49e-2),
                                                               ('epsrel', 10.00e-4),
                                                               ('limit', 2)]))[0]
            # stop if nan is detected
            if np.isnan(T_coef_t[i, j, k]) or np.isnan(T_coef_s[i, j, k]):
                print('T_coef = nan detected')
                print('i, j, k = ', i, j, k)
                print('T_coef_t[i, j, k] = ', (T_coef_t[i, j, k]))
                print('T_coef_s[i, j, k] = ', (T_coef_s[i, j, k]))
                print()
                print('x_comp = ', x_comp)
                print('y_comp = ', y_comp)
                print('x_coords = ', x_coords)
                print('y_coords = ', y_coords)
                print('reg = ', reg)
                # =    print('face.mfp.t[i, j] = ', face.mfp.t[i, j]
                print('adj_cells[0] = ', adj_cells[0])
                print('adj_cells[k-j] = ', adj_cells[k - j])
                sys.exit()
            else:
                pass
            #
            # trans_coef_file.write(('{:>6d}' * 3 + '{:>12.3E}' * 4 + '\n').format(int(T_from[i, j, k]),
            #                                                                      int(T_to[i, j, k]),
            #                                                                      int(T_via[i, j, k]),
            #                                                                      T_coef_s[i, j, k],
            #                                                                      T_coef_t[i, j, k],
            #                                                                      face.mfp.s[i, j],
            #                                                                      face.mfp.t[i, j]))

    return i, j, k, T_coef_s[i, j, k], T_coef_t[i, j, k], T_from[i, j, k], T_to[i, j, k], T_via[i, j, k]