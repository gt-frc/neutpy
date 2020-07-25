#!/usr/bin/python
# coding=utf-8

from __future__ import division
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import sys
from math import pi, sqrt, degrees, acos
import os
import pandas as pd
import _warnings
from scipy.constants import physical_constants
from shapely.geometry import LineString, Point

m_p = physical_constants['proton mass'][0]

def isnamedtupleinstance(x):
    """

    :param x:
    :return:
    """
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


def iterate_namedtuple(object, df):
    if isnamedtupleinstance(object):
        for key, item in object._asdict().iteritems():
            if isnamedtupleinstance(item):
                iterate_namedtuple(item, df)
            else:
                df[key] = pd.Series(item.flatten(), name=key)
    else:
        pass
    return df

def print_progress(iteration, total, prefix='', suffix='', decimals=0, bar_length=50):
    """creates a progress bar

    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * ((iteration + 1) / float(total)))
    filled_length = int(round(bar_length * (iteration + 1) / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix))

def calc_cell_pts(neut):
    sys.setrecursionlimit(100000)

    def loop(neut, oldcell, curcell, cellscomplete, xcoords, ycoords):
        beta[curcell, :neut.nSides[curcell]] = np.cumsum(
            np.roll(neut.angles[curcell, :neut.nSides[curcell]], 1) - 180) + 180

        # if first cell:
        if oldcell == 0 and curcell == 0:
            # rotate cell by theta0 value (specified)
            beta[curcell, :neut.nSides[curcell]] = beta[curcell, :neut.nSides[curcell]] + neut.cell1_theta0
            x_comp = np.cos(np.radians(beta[curcell, :neut.nSides[curcell]])) * neut.lsides[curcell,
                                                                                :neut.nSides[curcell]]
            y_comp = np.sin(np.radians(beta[curcell, :neut.nSides[curcell]])) * neut.lsides[curcell,
                                                                                :neut.nSides[curcell]]
            xcoords[curcell, :neut.nSides[curcell]] = np.roll(np.cumsum(x_comp), 1) + neut.cell1_ctr_x
            ycoords[curcell, :neut.nSides[curcell]] = np.roll(np.cumsum(y_comp), 1) + neut.cell1_ctr_y

        # for all other cells:
        else:

            # adjust all values in beta for current cell such that the side shared
            # with oldcell has the same beta as the oldcell side
            oldcell_beta = beta[oldcell, :][np.where(neut.adjCell[oldcell, :] == curcell)][0]
            delta_beta = beta[curcell, np.where(neut.adjCell[curcell, :] == oldcell)] + 180 - oldcell_beta
            beta[curcell, :neut.nSides[curcell]] = beta[curcell, :neut.nSides[curcell]] - delta_beta

            # calculate non-shifted x- and y- coordinates
            x_comp = np.cos(np.radians(beta[curcell, :neut.nSides[curcell]])) * neut.lsides[curcell,
                                                                                :neut.nSides[curcell]]
            y_comp = np.sin(np.radians(beta[curcell, :neut.nSides[curcell]])) * neut.lsides[curcell,
                                                                                :neut.nSides[curcell]]
            xcoords[curcell, :neut.nSides[curcell]] = np.roll(np.cumsum(x_comp),
                                                              1)  # xcoords[oldcell,np.where(neut.adjCell[oldcell,:]==curcell)[0][0]]
            ycoords[curcell, :neut.nSides[curcell]] = np.roll(np.cumsum(y_comp),
                                                              1)  # ycoords[oldcell,np.where(neut.adjCell[oldcell,:]==curcell)[0][0]]

            cur_in_old = np.where(neut.adjCell[oldcell, :] == curcell)[0][0]
            old_in_cur = np.where(neut.adjCell[curcell, :] == oldcell)[0][0]
            mdpt_old_x = (xcoords[oldcell, cur_in_old] + np.roll(xcoords[oldcell, :], -1)[cur_in_old]) / 2
            mdpt_old_y = (ycoords[oldcell, cur_in_old] + np.roll(ycoords[oldcell, :], -1)[cur_in_old]) / 2
            mdpt_cur_x = (xcoords[curcell, old_in_cur] + np.roll(xcoords[curcell, :], -1)[old_in_cur]) / 2
            mdpt_cur_y = (ycoords[curcell, old_in_cur] + np.roll(ycoords[curcell, :], -1)[old_in_cur]) / 2

            xshift = mdpt_old_x - mdpt_cur_x
            yshift = mdpt_old_y - mdpt_cur_y

            xcoords[curcell, :] = xcoords[curcell,
                                  :] + xshift  # xcoords[oldcell,np.where(neut.adjCell[oldcell,:]==curcell)[0][0]]
            ycoords[curcell, :] = ycoords[curcell,
                                  :] + yshift  # ycoords[oldcell,np.where(neut.adjCell[oldcell,:]==curcell)[0][0]]

        # continue looping through adjacent cells
        for j, newcell in enumerate(neut.adjCell[curcell, :neut.nSides[curcell]]):
            # if the cell under consideration is a normal cell (>3 sides) and not complete, then move into that cell and continue
            if neut.nSides[newcell] >= 3 and cellscomplete[newcell] == 0:
                cellscomplete[newcell] = 1
                loop(neut, curcell, newcell, cellscomplete, xcoords, ycoords)

        return xcoords, ycoords

    xcoords = np.zeros(neut.adjCell.shape)
    ycoords = np.zeros(neut.adjCell.shape)
    beta = np.zeros(neut.adjCell.shape)  # beta is the angle of each side with respect to the +x axis.

    ## Add initial cell to the list of cells that are complete
    cellscomplete = np.zeros(neut.nCells)
    cellscomplete[0] = 1
    xs, ys = loop(neut, 0, 0, cellscomplete, xcoords, ycoords)
    return xs, ys

class NeutpyTools:

    def __init__(self, neut=None):

        # get vertices in R, Z geometry
        self.xs, self.ys = self.calc_cell_pts(neut)

        # localize densities, ionization rates, and a few other parameters that might be needed.
        self.n_n_slow = neut.nn.s
        self.n_n_thermal = neut.nn.t
        self.n_n_total = neut.nn.tot
        self.izn_rate_slow = neut.izn_rate.s
        self.izn_rate_thermal = neut.izn_rate.t
        self.izn_rate_total = neut.izn_rate.tot
        self.flux_in_s = neut.flux.inc.s
        self.flux_in_t = neut.flux.inc.t
        self.flux_in_tot = self.flux_in_s + self.flux_in_t
        self.flux_out_s = neut.flux.out.s
        self.flux_out_t = neut.flux.out.t
        self.flux_out_tot = self.flux_out_s + self.flux_out_t

        self.create_flux_outfile()
        self.create_cell_outfile()

        flux_s_xcomp, flux_s_ycomp, flux_s_mag = self.calc_flow('slow', norm=True)
        flux_t_xcomp, flux_t_ycomp, flux_t_mag = self.calc_flow('thermal', norm=True)
        flux_tot_xcomp, flux_tot_ycomp, flux_tot_mag = self.calc_flow('total', norm=True)

        self.vars = {}
        self.vars['n_n_slow'] = neut.nn.s
        self.vars['n_n_thermal'] = neut.nn.t
        self.vars['n_n_total'] = neut.nn.tot

        self.vars['flux_s_xcomp'] = flux_s_xcomp
        self.vars['flux_s_ycomp'] = flux_s_ycomp
        self.vars['flux_s_mag'] = flux_s_mag

        self.vars['flux_t_xcomp'] = flux_t_xcomp
        self.vars['flux_t_ycomp'] = flux_t_ycomp
        self.vars['flux_t_mag'] = flux_t_mag

        self.vars['flux_tot_xcomp'] = flux_tot_xcomp
        self.vars['flux_tot_ycomp'] = flux_tot_ycomp
        self.vars['flux_tot_mag'] = flux_tot_mag

        print('attempting to start plot_cell_vals')
        self.plot_cell_vals()



    def create_cell_outfile(self):
        df = pd.DataFrame()
        df['R'] = pd.Series(np.mean(self.xs, axis=1), name='R')
        df['Z'] = pd.Series(np.mean(self.ys, axis=1), name='Z')
        df['n_n_slow'] = pd.Series(self.n_n_slow, name='n_n_slow')
        df['n_n_thermal'] = pd.Series(self.n_n_thermal, name='n_n_thermal')
        df['n_n_total'] = pd.Series(self.n_n_total, name='n_n_total')
        df['izn_rate_slow'] = pd.Series(self.izn_rate_slow, name='izn_rate_slow')
        df['izn_rate_thermal'] = pd.Series(self.izn_rate_thermal, name='izn_rate_thermal')
        df['izn_rate_total'] = pd.Series(self.izn_rate_thermal, name='izn_rate_total')
        #cell_df = iterate_namedtuple(neut.cell, df)
        df.to_csv(os.getcwd() + '/outputs/neutpy_cell_values.txt')


    def interp_RZ(self, var):
        x = np.average(self.xs, axis=1)
        y = np.average(self.ys, axis=1)
        d = self.vars[var]
        return Rbf(x, y, d)

    def calc_flow(self, ntrl_pop='tot', norm=True):
        """Creates interpolation functions for net flux directions and magnitudes

           flux_in: fluxes coming into cells. Can be slow, thermal, total or any other fluxes. Array of size (nCells, 3)
           flux_in: fluxes leaving cells. Can be slow, thermal, total or any other fluxes. Array of size (nCells, 3)
           norm: returns normalized x- and y-component interpolation functions. Useful for plotting quiver plots
                    with equally sized arrows or if you only care about the direction of the flux (You can still get
                    the magnitude from "flux_net_av_mag" interpolation object.)
            """

        if ntrl_pop is 'slow':
            flux_in = self.flux_in_s[:, :-1]
            flux_out = self.flux_out_s[:, :-1]
        elif ntrl_pop is 'thermal':
            flux_in = self.flux_in_t[:, :-1]
            flux_out = self.flux_out_t[:, :-1]
        elif ntrl_pop is 'total':
            flux_in = self.flux_in_tot[:, :-1]
            flux_out = self.flux_out_tot[:, :-1]

        flux_net = flux_out - flux_in

        cent_pts_x = np.average(self.xs, axis=1)
        cent_pts_y = np.average(self.ys, axis=1)

        x_comp = np.roll(self.xs, -1, axis=1) - self.xs
        y_comp = np.roll(self.ys, -1, axis=1) - self.ys
        lside = np.sqrt(x_comp**2 + y_comp**2)
        perim = np.sum(lside, axis=1).reshape((-1, 1))
        l_frac = lside / perim

        side_angles = np.arctan2(y_comp, x_comp)
        side_angles = np.where(side_angles < 0, side_angles+2*pi, side_angles)

        outwd_nrmls = side_angles + pi/2
        outwd_nrmls = np.where(outwd_nrmls < 0, outwd_nrmls+2*pi, outwd_nrmls)
        outwd_nrmls = np.where(outwd_nrmls >= 2*pi, outwd_nrmls-2*pi, outwd_nrmls)

        flux_net_dir = np.where(flux_net < 0, outwd_nrmls+pi, outwd_nrmls)
        flux_net_dir = np.where(flux_net_dir < 0, flux_net_dir+2*pi, flux_net_dir)
        flux_net_dir = np.where(flux_net_dir >= 2*pi, flux_net_dir-2*pi, flux_net_dir)

        # x- and y-component of fluxes
        flux_net_xcomp = np.abs(flux_net)*np.cos(flux_net_dir)
        flux_net_ycomp = np.abs(flux_net)*np.sin(flux_net_dir)

        # side-length weighted average of x- and y-components of flux
        flux_net_xcomp_av = np.sum(flux_net_xcomp * l_frac, axis=1)
        flux_net_ycomp_av = np.sum(flux_net_ycomp * l_frac, axis=1)

        # normalized x- and y-components
        flux_net_xcomp_av_norm = flux_net_xcomp_av / np.sqrt(flux_net_xcomp_av**2 + flux_net_ycomp_av**2)
        flux_net_ycomp_av_norm = flux_net_ycomp_av / np.sqrt(flux_net_xcomp_av**2 + flux_net_ycomp_av**2)

        # side-length weighted average flux magnitude
        flux_net_av_mag = np.sqrt(flux_net_xcomp_av**2 + flux_net_ycomp_av**2)

        # create averaged net x- and y-component interpolation functions
        if norm:
            flux_xcomp = flux_net_xcomp_av_norm
            flux_ycomp = flux_net_ycomp_av_norm
        else:
            flux_xcomp = flux_net_xcomp_av
            flux_ycomp = flux_net_ycomp_av

        return flux_xcomp, flux_ycomp, flux_net_av_mag

    def common_plots(self):
        pass

def cut(line, distance):
    """Cuts a shapely line in two at a distance(normalized) from its starting point"""
    if distance <= 0.0 or distance >= 1.0:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p), normalized=True)
        if pd == distance:
            return [LineString(coords[:i + 1]), LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance, normalized=True)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]

def isinline(pt, line):
    pt_s = Point(pt)
    dist = line.distance(pt_s)
    if dist < 1E-6:
        return True
    else:
        return False

def draw_line(R, Z, array, val, pathnum):
    res = plt.contour(R, Z, array, [val]).collections[0].get_paths()[pathnum]
    # res = cntr.contour(R, Z, array).trace(val)[pathnum]
    x = res.vertices[:, 0]
    y = res.vertices[:, 1]
    return x, y

def getangle(p1, p2):
    if isinstance(p1, Point) and isinstance(p2, Point):
        p1 = [p1.coords.xy[0][0], p1.coords.xy[1][0]]
        p2 = [p2.coords.xy[0][0], p2.coords.xy[1][0]]
    p1 = np.asarray(p1)
    p1 = np.reshape(p1, (-1, 2))
    p2 = np.asarray(p2)
    p2 = np.reshape(p2, (-1, 2))
    theta = np.arctan2(p1[:, 1] - p2[:, 1], p1[:, 0] - p2[:, 0])
    theta_mod = np.where(theta < 0, theta + pi,
                         theta)  # makes it so the angle is always measured counterclockwise from the horizontal
    return theta

def getangle3ptsdeg(p1, p2, p3):
    a = sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    b = sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2)
    c = sqrt((p1[0] - p3[0]) ** 2 + (p1[1] - p3[1]) ** 2)
    theta = degrees(acos((c ** 2 - a ** 2 - b ** 2) / (-2 * a * b)))  # returns degree in radians
    return theta

def listToFloatChecker(val, message, verbose=False):
    if type(val) == np.ndarray:
        if len(val) > 1:
            raise ValueError(message)
        elif len(val) == 1:
            if verbose: _warnings.warn("List value of len 1 found")
            return val[0]
        else:
            if verbose: _warnings.warn("List value of len 0 found")
            return val
    else:
        return val

def calc_fsa(x, R, Z):


    R1 = R[:, :-1]
    R2 = np.roll(R[:, :-1], -1, axis=1)
    Z1 = Z[:, :-1]
    Z2 = np.roll(Z[:, :-1], -1, axis=1)
    x1 = x[:, :-1]
    x2 = np.roll(x[:, :-1], -1, axis=1)

    dl = np.sqrt((R2 - R1) ** 2 + (Z2 - Z1) ** 2)

    R_av = (R1 + R2)/2

    dA = dl * (2 * pi * R_av)

    x_av = (x1 + x2)/2

    fsa = np.sum(x_av * dA, axis=1) / np.sum(dA, axis=1)
    fsa[0] = x[0, 0]
    return fsa