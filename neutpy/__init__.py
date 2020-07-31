# !/usr/bin/env python2
# -*- coding: utf-8 -*-

""" NeutPy - A neutrals code for tokamak fusion reactors

NeutPy is the Python 2.x port of GTNEUT, which was written by John Mandrekas.

GTNEUT is a two-dimensional code for the calculation of the transport of neutral particles in fusion plasmas.
It is based on the Transmission and Escape Probabilities (TEP) method and can be considered a computationally efficient
alternative to traditional Monte Carlo methods. The code has been benchmarked extensively against Monte Carlo and
has been used to model the distribution of neutrals in fusion experiments.

The original physics background can be found at

Mandrekas, John. (2004). GTNEUT: A code for the calculation of neutral particle transport in plasmas based on the
    Transmission and Escape Probability method. Computer Physics Communications.
    161. 36-64. 10.1016/j.cpc.2004.04.009.

The original FORTRAN 95 GTNEUT code can be found at The Fusion Research Center GitHub at
https://github.com/gt-frc/GTNEUT

The initial port to Python 2.x was done entirely by Maxwell D. Hill. Parallelization was implemented by
Jonathan J. Roveto, who also performed packaging and refactoring for PyPi distribution. NeutPy can be found at
https://github.com/gt-frc/neutpy

Usage:
    Import the neutrals class.

    >>> from neutpy import neutrals

    There are two main entry points into NeutPy: from_file and from_mesh. The usage of from_file is as follows:

    >>> neuts = neutrals()
    INITIALIZING NEUTPY
    >>> neuts.from_file('filename')

    where the filename is relative to the CWD.

TODO:
    Implement from_mesh method

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

from __future__ import division

from math import tan
import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from neutpy.parallelNeut import coeff_calc
from collections import namedtuple
from neutpy.crosssections import calc_svrec, calc_svcx, calc_svel, calc_sveln, calc_svione, calc_xsec
from neutpy.physics import calc_mfp, calc_X_i, calc_P_0i, calc_P_i, calc_c_i, calc_Tn_intocell_t, calc_refl_alb, calc_Ki3, \
    calc_ext_src
from neutpy.tools import cut, isclose, isinline, draw_line, getangle, getangle3ptsdeg, listToFloatChecker, calc_fsa, \
    calc_cell_pts
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import cpu_count
from scipy.interpolate import griddata, UnivariateSpline
from scipy.constants import elementary_charge
from shapely.geometry import Point, LineString, LinearRing
from shapely.ops import polygonize, linemerge
from math import degrees, sqrt, pi
import sys, os, re, json, yaml
from subprocess import call
import time

import ConfigParser

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError("Matplotlib failed to import and is a required package.")


class neutrals:


    def __init__(self, verbose=False):

        self.lsides = None
        self.elecTemp = None
        self.nCells = None
        self.ionTemp = None
        self.elecDens = None
        self.ionDens = None
        self.cpu_cores = 1
        self.config_loc = "neutpy.conf"
        print 'INITIALIZING NEUTPY'

        sys.dont_write_bytecode = True
        self.verbose = verbose
        self.sv = calc_xsec()

    def from_mesh(self, filename):
        """
        Runs NeutPy assuming a meshing has already been generated
        :param filename:
        """
        # TODO: Allow neutpy to be run from a mesh JSON
        # inp = read_infile(filename)
        raise NotImplementedError

    def from_file(self, infile):
        """
        Instantiate NeutPy using the main configuration file as well as a file containing profile and plasma data.
        To set the configuration file, use the set_config method.

        :param infile: The shot data, including profiles and plasma parameters (e.g., BT0)
        :type infile: str
        """
        self._read_config(infile)
        if self.verbose:
            print 'Generating separatrix lines'
        self._get_sep_lines()

        if self.verbose:
            print 'Generating core lines'
        self._get_core_lines()

        if self.verbose:
            print 'Generating scrape-off layer lines'
        self._get_sol_lines()

        if self.verbose:
            print 'Generating private flux region'
        self._pfr_lines()

        if self.verbose:
            print 'Generating core background plasma'
        self._core_nT()

        if self.verbose:
            print 'Generating scrape-off layer'
        self._sol_nT()

        if self.verbose:
            print 'Generating private flux region'
        self._pfr_nT()

        if self.verbose:
            print 'Running Triangle meshing routine'

        self._triangle_prep()
        self._read_triangle()

        if self.verbose:
            print 'Running neutrals calculation'
        self._run()

        # get vertices in R, Z geometry
        self.xs, self.ys = calc_cell_pts(self)
        return self

    def from_gt3(self, gt3):

        config = ConfigParser.RawConfigParser()
        config.read(self.config_loc)
        # Collect configuration from main configuration file

        self.verbose = config.getint('Data', 'verbose')
        self.cpu_cores = config.getint('Data', 'cpu_cores')
        self.corelines_begin = config.getfloat('Data', 'corelines_begin')
        self.num_corelines = config.getint('Data', 'num_corelines')
        self.sollines_psi_max = config.getfloat('Data', 'sollines_psi_max')
        self.num_sollines = config.getint('Data', 'num_sollines')
        self.xi_sep_pts = config.getint('Data', 'xi_sep_pts')
        self.ib_trim_off = config.getfloat('Data', 'ib_trim_off')
        self.ob_trim_off = config.getfloat('Data', 'ob_trim_off')
        self.xi_ib_pts = config.getint('Data', 'xi_ib_pts')
        self.xi_ob_pts = config.getint('Data', 'xi_ob_pts')
        self.core_pol_pts = config.getint('Data', 'core_pol_pts')
        self.ib_div_pol_pts = config.getint('Data', 'ib_div_pol_pts')
        self.ob_div_pol_pts = config.getint('Data', 'ob_div_pol_pts')
        self.pfr_ni_val = config.getfloat('Data', 'pfr_ni_val')
        self.pfr_ne_val = config.getfloat('Data', 'pfr_ne_val')
        self.pfr_Ti_val = config.getfloat('Data', 'pfr_Ti_val')
        self.pfr_Te_val = config.getfloat('Data', 'pfr_Te_val')
        self.tri_min_area = config.getfloat('Data', 'tri_min_area')
        self.tri_min_angle = config.getfloat('Data', 'tri_min_angle')

        # Pull in data from the input file

        self.ne_data = gt3.core.n.e[:, 0]
        self.ni_data = gt3.core.n.i[:, 0]
        self.Te_data = gt3.core.T.e[:, 0]
        self.Ti_data = gt3.core.T.i[:, 0]

        # Get plasma parameters

        self.BT0 = gt3.inp.BT0

        self.wall_line = LineString(self.wall_exp)
        self.R = gt3.core.psi_data.R
        self.Z = gt3.core.psi_data.Z
        self.psi = gt3.core.psi_data.psi

    def _get_sep_lines(self):
        """
        Generate the separatric lines

        :rtype: None

        """
        # find x-point location
        dpsidR = np.gradient(self.psi, self.R[0, :], axis=1)
        dpsidZ = np.gradient(self.psi, self.Z[:, 0], axis=0)
        d2psidR2 = np.gradient(dpsidR, self.R[0, :], axis=1)
        d2psidZ2 = np.gradient(dpsidZ, self.Z[:, 0], axis=0)

        # find line(s) where dpsidR=0
        csR = plt.contour(self.R, self.Z, dpsidR, [0])
        csZ = plt.contour(self.R, self.Z, dpsidZ, [0])

        self.dpsidR_0 = csR.collections[0].get_paths()
        # self.dpsidR_0 = cntr.contour(self.R, self.Z, dpsidR).trace(0.0)

        # find line(s) where dpsidZ=0
        # self.dpsidZ_0 = cntr.contour(self.R, self.Z, dpsidZ).trace(0.0)
        self.dpsidZ_0 = csZ.collections[0].get_paths()

        for i, path1 in enumerate(self.dpsidR_0):
            for j, path2 in enumerate(self.dpsidZ_0):
                try:
                    # find intersection points between curves for dpsidR=0 and dpsidZ=0
                    ints = LineString(path1.vertices).intersection(LineString(path2.vertices))
                    # if there is only one intersection ('Point'), then we're probably not
                    # dealing with irrelevant noise in psi
                    if ints.type == 'Point':
                        # check if local maximum or minimum
                        d2psidR2_pt = griddata(np.column_stack((self.R.flatten(), self.Z.flatten())),
                                               d2psidR2.flatten(),
                                               [ints.x, ints.y],
                                               method='cubic')
                        d2psidZ2_pt = griddata(np.column_stack((self.R.flatten(), self.Z.flatten())),
                                               d2psidZ2.flatten(),
                                               [ints.x, ints.y],
                                               method='cubic')

                        if d2psidR2_pt > 0 and d2psidZ2_pt > 0:
                            # we've found the magnetic axis
                            self.m_axis = np.array([ints.x, ints.y])
                        elif d2psidR2_pt < 0 and d2psidZ2_pt < 0:
                            # we've found a magnet. Do nothing.
                            pass
                        elif ints.y < 0:
                            # we've probably found our x-point, although this isn't super robust
                            # and obviously only applies to a single-diverted, lower-null configuration
                            # TODO: make this more robust, I could easily see this failing on some shots
                            self.xpt = np.array([ints.x, ints.y])

                        # uncomment this line when debugging
                        # print list(ints.coords), d2psidR2(ints.x, ints.y), d2psidZ2(ints.x, ints.y)
                except:
                    pass

        # normalize psi
        psi_shift = self.psi + abs(np.amin(self.psi))  # set center to zero
        psi_shift_xpt = griddata(np.column_stack((self.R.flatten(), self.Z.flatten())),
                                 psi_shift.flatten(),
                                 self.xpt,
                                 method='cubic')
        # psi_shift_xpt = interp2d(R, Z, psi_shift, kind='linear')(xpt[0], xpt[1]) # get new value at sep
        self.psi_norm = psi_shift / psi_shift_xpt

        # create lines for seperatrix and divertor legs of seperatrix
        plt.figure()
        plt.contour(self.R, self.Z, self.psi_norm, [1])

        num_lines = int(len(plt.contour(self.R, self.Z, self.psi_norm, [1]).collections[0].get_paths()))
        if num_lines == 1:
            # in this case, the contour points that matplotlib returned constitute
            # a single line from inboard divertor to outboard divertor. We need to
            # add in the x-point in at the appropriate locations and split into a
            # main and a lower seperatrix line, each of which will include the x-point.
            x_psi, y_psi = draw_line(self.R, self.Z, self.psi_norm, 1.0, 0)

            loc1 = np.argmax(y_psi > self.xpt[1])
            loc2 = len(y_psi) - np.argmin(y_psi[::-1] < self.xpt[1])

            x_psi = np.insert(x_psi, (loc1, loc2), self.xpt[0])
            y_psi = np.insert(y_psi, (loc1, loc2), self.xpt[1])

            psi_1_pts = np.column_stack((x_psi, y_psi))
            self.main_sep_pts = psi_1_pts[loc1:loc2 + 1, :]
            self.main_sep_line = LineString(self.main_sep_pts[:-1])
            self.main_sep_line_closed = LineString(self.main_sep_pts)

            # get the inboard and outboard divertor legs seperately. This is so that
            # everything that includes the x-point can start with the x-point, which
            # elliminates the risk of tiny triangles in the vicinity of the x-point
            self.inboard_div_sep = np.flipud(psi_1_pts[:loc1 + 1])
            self.outboard_div_sep = psi_1_pts[loc2 + 1:]

            # cut inboard line at the wall and add intersection point to wall_line
            line = LineString(self.inboard_div_sep)
            int_pt = line.intersection(self.wall_line)
            self.ib_div_line = line
            self.ib_div_line_cut = cut(line, line.project(int_pt, normalized=True))[0]
            # self.ib_div_line_cut = line
            # TODO: add point to wall line

            # cut inboard line at the wall and add intersection point to wall_line
            line = LineString(self.outboard_div_sep)
            int_pt = line.intersection(self.wall_line)
            self.ob_div_line = line
            self.ob_div_line_cut = cut(line, line.project(int_pt, normalized=True))[0]

            ib_div_pts = np.flipud(np.asarray(self.ib_div_line_cut.xy).T)
            sep_pts = np.asarray(self.main_sep_line.xy).T
            ob_div_pts = np.asarray(self.ob_div_line_cut.xy).T

            entire_sep_pts = np.vstack((ib_div_pts, sep_pts[1:, :], ob_div_pts))
            self.entire_sep_line = LineString(entire_sep_pts)

            # TODO: add point to wall line

        elif num_lines == 2:
            # in this case, we have a lower seperatrix trace (line 0), and a main
            # seperatrix trace (line 1).

            # first do lower seperatrix line
            x_psi, y_psi = draw_line(self.R, self.Z, self.psi_norm, 1.0, 0)
            loc = np.argmax(x_psi > self.xpt[0])

            x_psi = np.insert(x_psi, loc, self.xpt[0])
            y_psi = np.insert(y_psi, loc, self.xpt[1])
            psi_1_pts = np.column_stack((x_psi, y_psi))

            self.inboard_div_sep = np.flipud(psi_1_pts[:loc + 1])
            self.outboard_div_sep = psi_1_pts[loc + 1:]

            # cut inboard line at the wall and add intersection point to wall_line
            line = LineString(self.inboard_div_sep)
            int_pt = line.intersection(self.wall_line)
            self.ib_div_line = line
            self.ib_div_line_cut = cut(line, line.project(int_pt, normalized=True))[0]

            # cut inboard line at the wall and add intersection point to wall_line
            line = LineString(self.outboard_div_sep)
            int_pt = line.intersection(self.wall_line)
            self.ob_div_line = line
            self.ob_div_line_cut = cut(line, line.project(int_pt, normalized=True))[0]
            # TODO: add point to wall line

            # now to main seperatrix line
            x_psi, y_psi = draw_line(self.R, self.Z, self.psi_norm, 1.0, 1)
            self.main_sep_pts = np.insert(np.column_stack((x_psi, y_psi)), 0, self.xpt, axis=0)
            self.main_sep_line = LineString(self.main_sep_pts[:-1])
            self.main_sep_line_closed = LineString(self.main_sep_pts)

            entire_sep_pts = np.vstack((self.ib_div_pts, self.sep_pts[1:, :], self.ob_div_pts))
            self.entire_sep_line = LineString(entire_sep_pts)
            # now clean up the lines by removing any points that are extremely close
            # to the x-point
            # TODO:

    def _get_core_lines(self):
        self.core_lines = []
        # psi_pts = np.concatenate((np.linspace(0, 0.8, 5, endpoint=False), np.linspace(0.8, 1.0, 4, endpoint=False)))
        psi_pts = np.linspace(self.corelines_begin, 1, self.num_corelines, endpoint=False)
        for i, v in enumerate(psi_pts):
            num_lines = int(len(plt.contour(self.R, self.Z, self.psi_norm, [v]).collections[0].get_paths()))
            # num_lines = int(len(cntr.contour(self.R, self.Z, self.psi_norm).trace(v))/2)
            if num_lines == 1:
                # then we're definitely dealing with a surface inside the seperatrix
                x, y = draw_line(self.R, self.Z, self.psi_norm, v, 0)
                self.core_lines.append(LineString(np.column_stack((x[:-1], y[:-1]))))
            else:
                # we need to find which of the surfaces is inside the seperatrix
                for j, line in enumerate(
                        plt.contour(self.R, self.Z, self.psi_norm, [v]).collections[0].get_paths()[:num_lines]):
                    # for j, line in enumerate(cntr.contour(self.R, self.Z, self.psi_norm).trace(v)[:num_lines]):
                    # for j, line in enumerate(cntr.contour(R, Z, self.psi_norm).trace(v)):
                    x, y = draw_line(self.R, self.Z, self.psi_norm, v, j)
                    if (np.amax(x) < np.amax(self.main_sep_pts[:, 0]) and \
                            np.amin(x) > np.amin(self.main_sep_pts[:, 0]) and \
                            np.amax(y) < np.amax(self.main_sep_pts[:, 1]) and \
                            np.amin(y) > np.amin(self.main_sep_pts[:, 1])):
                        # then it's an internal flux surface
                        self.core_lines.append(LineString(np.column_stack((x[:-1], y[:-1]))))
                        break

    def _get_sol_lines(self):
        # find value of psi at outside of what we're going to call the SOL
        self.sol_lines = []
        self.sol_lines_cut = []

        sol_width_obmp = 0.02
        psi_pts = np.linspace(1, self.sollines_psi_max, self.num_sollines + 1, endpoint=True)[1:]
        for i, v in enumerate(psi_pts):
            num_lines = int(len(plt.contour(self.R, self.Z, self.psi_norm, [v]).collections[0].get_paths()))
            if num_lines == 1:
                # then we're definitely dealing with a surface inside the seperatrix
                x, y = draw_line(self.R, self.Z, self.psi_norm, v, 0)
                self.sol_lines.append(LineString(np.column_stack((x, y))))
            else:
                # TODO:
                pass
        for line in self.sol_lines:
            # find intersection points with the wall
            int_pts = line.intersection(self.wall_line)
            # cut line at intersection points
            cut_line = cut(line, line.project(int_pts[0], normalized=True))[1]
            cut_line = cut(cut_line, cut_line.project(int_pts[1], normalized=True))[0]
            self.sol_lines_cut.append(cut_line)

        # add wall intersection points from divertor legs and sol lines to wall_line.
        # This is necessary to prevent thousands of tiny triangles from forming if the
        # end of the flux line isn't exactly on top of the wall line.

        # add inboard seperatrix strike point
        union = self.wall_line.union(self.ib_div_line)
        result = [geom for geom in polygonize(union)][0]
        self.wall_line = LineString(result.exterior.coords)

        # add outboard seperatrix strike point
        union = self.wall_line.union(self.ob_div_line)
        result = [geom for geom in polygonize(union)][0]
        self.wall_line = LineString(result.exterior.coords)

        # add sol line intersection points on inboard side
        # for some reason, union freaks out when I try to do inboard and outboard
        # at the same time.
        for num, line in enumerate(self.sol_lines):
            union = self.wall_line.union(cut(line, 0.5)[0])
            result = [geom for geom in polygonize(union)][0]
            self.wall_line = LineString(result.exterior.coords)

        # add sol line intersection points on outboard side
        for num, line in enumerate(self.sol_lines):
            union = self.wall_line.union(cut(line, 0.5)[1])
            result = [geom for geom in polygonize(union)][0]
            self.wall_line = LineString(result.exterior.coords)

    def _pfr_lines(self):
        num_lines = int(len(plt.contour(self.R, self.Z, self.psi_norm, [.999]).collections[0].get_paths()))
        # num_lines = int(len(cntr.contour(self.R, self.Z, self.psi_norm).trace(0.999))/2)
        if num_lines == 1:
            # then we're definitely dealing with a surface inside the seperatrix
            print 'Did not find PFR flux surface. Stopping.'
            sys.exit()
        else:
            # we need to find the surface that is contained within the private flux region
            for j, line in enumerate(
                    plt.contour(self.R, self.Z, self.psi_norm, [.99]).collections[0].get_paths()[:num_lines]):
                # for j, line in enumerate(cntr.contour(self.R, self.Z, self.psi_norm).trace(0.99)[:num_lines]):
                # for j, line in enumerate(cntr.contour(R, Z, self.psi_norm).trace(v)):
                x, y = draw_line(self.R, self.Z, self.psi_norm, 0.99, j)
                if (np.amax(y) < np.amin(self.main_sep_pts[:, 1])):
                    # then it's a pfr flux surface, might need to add additional checks later
                    pfr_line_raw = LineString(np.column_stack((x, y)))
                    # find cut points
                    cut_pt1 = pfr_line_raw.intersection(self.wall_line)[0]
                    dist1 = pfr_line_raw.project(cut_pt1, normalized=True)
                    cutline_temp = cut(pfr_line_raw, dist1)[1]

                    # reverse line point order so we can reliably find the second intersection point
                    cutline_temp_rev = LineString(np.flipud(np.asarray(cutline_temp.xy).T))

                    cut_pt2 = cutline_temp_rev.intersection(self.wall_line)
                    dist2 = cutline_temp_rev.project(cut_pt2, normalized=True)
                    cutline_final_rev = cut(cutline_temp_rev, dist2)[1]

                    # reverse again for final pfr flux line
                    pfr_flux_line = LineString(np.flipud(np.asarray(cutline_final_rev.xy).T))

                    # add pfr_line intersection points on inboard side
                    # for some reason, union freaks out when I try to do inboard and outboard
                    # at the same time.
                    union = self.wall_line.union(cut(pfr_line_raw, 0.5)[0])
                    result = [geom for geom in polygonize(union)][0]
                    self.wall_line = LineString(result.exterior.coords)

                    # add pfr line intersection points on outboard side
                    union = self.wall_line.union(cut(pfr_line_raw, 0.5)[1])
                    result = [geom for geom in polygonize(union)][0]
                    self.wall_line = LineString(result.exterior.coords)

                    # cut out pfr section of wall line
                    wall_pts = np.asarray(self.wall_line.xy).T

                    # ib_int_pt = np.asarray(self.ib_div_line.intersection(self.wall_line).xy).T
                    # ob_int_pt = self.ob_div_line.intersection(self.wall_line)
                    wall_start_pos = np.where((wall_pts == cut_pt2).all(axis=1))[0][0]
                    wall_line_rolled = LineString(np.roll(wall_pts, -wall_start_pos, axis=0))
                    wall_line_cut_pfr = cut(wall_line_rolled,
                                            wall_line_rolled.project(cut_pt1, normalized=True))[0]

                    # create LineString with pfr line and section of wall line
                    self.pfr_line = linemerge((pfr_flux_line, wall_line_cut_pfr))
                    break
        # plt.axis('equal')
        # plt.plot(np.asarray(self.wall_line.xy).T[:, 0], np.asarray(self.wall_line.xy).T[:, 1], color='black', lw=0.5)
        # plt.plot(np.asarray(self.pfr_line.xy).T[:, 0], np.asarray(self.pfr_line.xy).T[:, 1], color='red', lw=0.5)

    def _core_nT(self):

        # Master arrays that will contain all the points we'll use to get n, T
        # throughout the plasma chamber via 2-D interpolation
        self.ni_pts = np.zeros((0, 3), dtype='float')
        self.ne_pts = np.zeros((0, 3), dtype='float')
        self.Ti_kev_pts = np.zeros((0, 3), dtype='float')
        self.Te_kev_pts = np.zeros((0, 3), dtype='float')

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Calculate n, T throughout the core plasma using radial profile input files, uniform on flux surface

        ni = UnivariateSpline(self.ni_data[:, 0], self.ni_data[:, 1], k=5, s=2.0)
        ne = UnivariateSpline(self.ne_data[:, 0], self.ne_data[:, 1], k=5, s=2.0)
        Ti_kev = UnivariateSpline(self.Ti_data[:, 0], self.Ti_data[:, 1], k=5, s=2.0)
        Te_kev = UnivariateSpline(self.Te_data[:, 0], self.Te_data[:, 1], k=5, s=2.0)

        # get approximate rho values associated with the psi values we're using
        # draw line between magnetic axis and the seperatrix at the outboard midplane
        self.obmp_pt = self.main_sep_pts[np.argmax(self.main_sep_pts, axis=0)[0]]
        self.ibmp_pt = self.main_sep_pts[np.argmin(self.main_sep_pts, axis=0)[0]]
        self.top_pt = self.main_sep_pts[np.argmax(self.main_sep_pts, axis=0)[1]]
        self.bot_pt = self.main_sep_pts[np.argmin(self.main_sep_pts, axis=0)[1]]

        rho_line = LineString([Point(self.m_axis), Point(self.obmp_pt)])
        # for several points on the rho line specified above:

        # To get smooth gradients for use in the SOL calculation, you need around
        # 50-100 radial points in the far edge and around 100 or so theta points
        # TODO: There is almost certainly a faster way to get these gradients.
        rho_pts = np.concatenate((np.linspace(0, 0.95, 20, endpoint=False),
                                  np.linspace(0.95, 1, 50, endpoint=False)), axis=0)

        thetapts = np.linspace(0, 1, 100, endpoint=False)
        for i, rho in enumerate(rho_pts):
            # get n, T information at the point by interpolating the rho-based input file data
            ni_val = ni(rho)
            ne_val = ne(rho)
            Ti_kev_val = Ti_kev(rho)
            Te_kev_val = Te_kev(rho)
            # get R, Z coordinates of each point along the rho_line
            pt_coords = np.asarray(rho_line.interpolate(rho, normalized=True).coords)[0]

            # get psi value at that point
            psi_val = griddata(np.column_stack((self.R.flatten(), self.Z.flatten())),
                               self.psi_norm.flatten(),
                               pt_coords,
                               method='linear')
            # map this n, T data to every point on the corresponding flux surface
            num_lines = int(len(plt.contour(self.R, self.Z, self.psi_norm, [psi_val]).collections[0].get_paths()))
            # num_lines = int(len(cntr.contour(self.R, self.Z, self.psi_norm).trace(psi_val))/2)

            if num_lines == 1:
                # then we're definitely dealing with a surface inside the seperatrix
                x, y = draw_line(self.R, self.Z, self.psi_norm, psi_val, 0)
                surf = LineString(np.column_stack((x, y)))
            else:
                # we need to find which of the surfaces is inside the seperatrix
                for j, line in enumerate(
                        plt.contour(self.R, self.Z, self.psi_norm, [psi_val]).collections[0].get_paths()[:num_lines]):
                    # for j, line in enumerate(cntr.contour(self.R, self.Z, self.psi_norm).trace(psi_val)[:num_lines]):
                    # for j, line in enumerate(cntr.contour(R, Z, self.psi_norm).trace(v)):
                    x, y = draw_line(self.R, self.Z, self.psi_norm, psi_val, j)
                    if (np.amax(x) < np.amax(self.main_sep_pts[:, 0]) and \
                            np.amin(x) > np.amin(self.main_sep_pts[:, 0]) and \
                            np.amax(y) < np.amax(self.main_sep_pts[:, 1]) and \
                            np.amin(y) > np.amin(self.main_sep_pts[:, 1])):
                        # then it's an internal flux surface
                        surf = LineString(np.column_stack((x, y)))
                        break

            for j, theta_norm in enumerate(thetapts):
                pt = np.asarray(surf.interpolate(theta_norm, normalized=True).coords).T
                self.ni_pts = np.vstack((self.ni_pts, np.append(pt, ni_val)))
                self.ne_pts = np.vstack((self.ne_pts, np.append(pt, ne_val)))
                self.Ti_kev_pts = np.vstack((self.Ti_kev_pts, np.append(pt, Ti_kev_val)))
                self.Te_kev_pts = np.vstack((self.Te_kev_pts, np.append(pt, Te_kev_val)))

        # Do seperatrix separately so we don't accidentally assign the input n, T data to the divertor legs
        self.ni_sep_val = ni(1.0)
        self.ne_sep_val = ne(1.0)
        self.Ti_kev_sep_val = Ti_kev(1.0)
        self.Te_kev_sep_val = Te_kev(1.0)
        self.Ti_J_sep_val = self.Ti_kev_sep_val * 1.0E3 * 1.6021E-19
        self.Te_J_sep_val = self.Te_kev_sep_val * 1.0E3 * 1.6021E-19
        for j, theta_norm in enumerate(thetapts):
            pt = np.asarray(self.main_sep_line.interpolate(theta_norm, normalized=False).coords, dtype='float').T
            self.ni_pts = np.vstack((self.ni_pts, np.append(pt, self.ni_sep_val)))
            self.ne_pts = np.vstack((self.ne_pts, np.append(pt, self.ne_sep_val)))
            self.Ti_kev_pts = np.vstack((self.Ti_kev_pts, np.append(pt, self.Ti_kev_sep_val)))
            self.Te_kev_pts = np.vstack((self.Te_kev_pts, np.append(pt, self.Te_kev_sep_val)))

    def _sol_nT(self):
        # Calculate n, T in SOL using Bohm diffusion, core data from radial profile input files, and input
        # divertor target densities and temperatures (replace with 2-pt divertor model later)

        # draw core line just inside the seperatrix (seperatrix would be too noisy absent SOL data, which is what we're trying to calculate)
        psi_val = 0.98
        num_lines = int(len(plt.contour(self.R, self.Z, self.psi_norm, [psi_val]).collections[0].get_paths()))
        # num_lines = int(len(cntr.contour(self.R, self.Z, self.psi_norm).trace(psi_val))/2)
        if num_lines == 1:
            # then we're definitely dealing with a surface inside the seperatrix
            x, y = draw_line(self.R, self.Z, self.psi_norm, psi_val, 0)
        else:
            # we need to find which of the surfaces is inside the seperatrix
            for j, line in enumerate(
                    plt.contour(self.R, self.Z, self.psi_norm, [psi_val]).collections[0].get_paths()[:num_lines]):
                # for j, line in enumerate(cntr.contour(self.R, self.Z, self.psi_norm).trace(psi_val)[:num_lines]):
                # for j, line in enumerate(cntr.contour(R, Z, self.psi_norm).trace(v)):
                x, y = draw_line(self.R, self.Z, self.psi_norm, psi_val, j)
                if (np.amax(x) < np.amax(self.main_sep_pts[:, 0]) and \
                        np.amin(x) > np.amin(self.main_sep_pts[:, 0]) and \
                        np.amax(y) < np.amax(self.main_sep_pts[:, 1]) and \
                        np.amin(y) > np.amin(self.main_sep_pts[:, 1])):
                    # then it's an internal flux surface
                    break

        # get quantities on a fairly fine R, Z grid for the purpose of taking gradients, etc.
        R_temp, Z_temp = np.meshgrid(np.linspace(0.95 * self.ibmp_pt[0], 1.05 * self.obmp_pt[0], 500),
                                     np.linspace(1.05 * self.top_pt[1], 1.05 * self.bot_pt[1], 500))

        ni_grid = griddata(self.ni_pts[:, :-1],
                           self.ni_pts[:, -1],
                           (R_temp, Z_temp),
                           method='cubic',
                           fill_value=self.ni_sep_val)
        ne_grid = griddata(self.ne_pts[:, :-1],
                           self.ne_pts[:, -1],
                           (R_temp, Z_temp),
                           method='cubic',
                           fill_value=self.ne_sep_val)
        Ti_grid = griddata(self.Ti_kev_pts[:, :-1],
                           self.Ti_kev_pts[:, -1] * 1.0E3 * 1.6021E-19,
                           (R_temp, Z_temp),
                           method='cubic',
                           fill_value=self.Ti_J_sep_val)
        Te_grid = griddata(self.Te_kev_pts[:, :-1],
                           self.Te_kev_pts[:, -1] * 1.0E3 * 1.6021E-19,
                           (R_temp, Z_temp),
                           method='cubic',
                           fill_value=self.Te_J_sep_val)

        dnidr = -1.0 * (np.abs(np.gradient(ni_grid, Z_temp[:, 0], axis=1)) + np.abs(
            np.gradient(ni_grid, R_temp[0, :], axis=0)))
        dnedr = -1.0 * (np.abs(np.gradient(ne_grid, Z_temp[:, 0], axis=1)) + np.abs(
            np.gradient(ne_grid, R_temp[0, :], axis=0)))
        dTidr = -1.0 * (np.abs(np.gradient(Ti_grid, Z_temp[:, 0], axis=1)) + np.abs(
            np.gradient(Ti_grid, R_temp[0, :], axis=0)))
        dTedr = -1.0 * (np.abs(np.gradient(Te_grid, Z_temp[:, 0], axis=1)) + np.abs(
            np.gradient(Te_grid, R_temp[0, :], axis=0)))

        # Get densities, temperatures, and other quantities along the flux surface we just drew
        # note densities and temperatures on seperatrix are assumed to be constant for all theta and are
        # obtained above, i.e. self.ni_sep_val, etc.
        dnidr_sep_raw = griddata(np.column_stack((R_temp.flatten(), Z_temp.flatten())),
                                 dnidr.flatten(),
                                 np.column_stack((x, y)),
                                 method='cubic'
                                 )
        dnedr_sep_raw = griddata(np.column_stack((R_temp.flatten(), Z_temp.flatten())),
                                 dnedr.flatten(),
                                 np.column_stack((x, y)),
                                 method='cubic'
                                 )

        dTidr_sep_raw = griddata(np.column_stack((R_temp.flatten(), Z_temp.flatten())),
                                 dTidr.flatten(),
                                 np.column_stack((x, y)),
                                 method='cubic'
                                 )

        dTedr_sep_raw = griddata(np.column_stack((R_temp.flatten(), Z_temp.flatten())),
                                 dTedr.flatten(),
                                 np.column_stack((x, y)),
                                 method='cubic'
                                 )

        BT_sep_raw = griddata(np.column_stack((R_temp.flatten(), Z_temp.flatten())),
                              (self.m_axis[0] * self.BT0 / R_temp).flatten(),
                              np.column_stack((x, y)),
                              method='cubic'
                              )

        # Get densities, temperatures, and other quantities along the inboard divertor leg of seperatrix
        # doing linear interpolation in the absense of a 1D model

        # Get densities, temperatures, and other quantities along the outboard divertor leg of seperatrix

        # norm factor used to divide by the order of magnitude to facilitate easier smoothing
        ni_norm_factor = 1.0  # *10**(int(np.log10(np.average(dnidr_sep)))-1)
        ne_norm_factor = 1.0  # *10**(int(np.log10(np.average(dnedr_sep)))-1)
        Ti_norm_factor = 1.0  # *10**(int(np.log10(np.average(dTidr_sep)))-1)
        Te_norm_factor = 1.0  # *10**(int(np.log10(np.average(dTedr_sep)))-1)

        # specify the number of xi (parallel) points in the seperatrix region of the 2 point divertor model

        ni_sep = np.zeros(self.xi_sep_pts) + self.ni_sep_val
        ne_sep = np.zeros(self.xi_sep_pts) + self.ne_sep_val
        Ti_sep = np.zeros(self.xi_sep_pts) + self.Ti_J_sep_val
        Te_sep = np.zeros(self.xi_sep_pts) + self.Te_J_sep_val

        dnidr_sep = UnivariateSpline(np.linspace(0, 1, len(dnidr_sep_raw)),
                                     dnidr_sep_raw / ni_norm_factor,
                                     k=5,
                                     s=0.0)(
            np.linspace(self.ib_trim_off, 1.0 - self.ob_trim_off, self.xi_sep_pts)) * ni_norm_factor

        dnedr_sep = UnivariateSpline(np.linspace(0, 1, len(dnedr_sep_raw)),
                                     dnedr_sep_raw / ne_norm_factor,
                                     k=5,
                                     s=0.0)(
            np.linspace(self.ib_trim_off, 1.0 - self.ob_trim_off, self.xi_sep_pts)) * ne_norm_factor

        dTidr_sep = UnivariateSpline(np.linspace(0, 1, len(dTidr_sep_raw)),
                                     dTidr_sep_raw / Ti_norm_factor,
                                     k=5,
                                     s=0.0)(
            np.linspace(self.ib_trim_off, 1.0 - self.ob_trim_off, self.xi_sep_pts)) * Ti_norm_factor

        dTedr_sep = UnivariateSpline(np.linspace(0, 1, len(dTedr_sep_raw)),
                                     dTedr_sep_raw / Te_norm_factor,
                                     k=5,
                                     s=0.0)(
            np.linspace(self.ib_trim_off, 1.0 - self.ob_trim_off, self.xi_sep_pts)) * Te_norm_factor

        BT_sep = UnivariateSpline(np.linspace(0, 1, len(dnidr_sep_raw)),
                                  BT_sep_raw,
                                  k=5,
                                  s=0.0)(np.linspace(self.ib_trim_off, 1.0 - self.ob_trim_off, self.xi_sep_pts))

        ni_ib_wall = self.ni_sep_val * 1.0
        ni_ob_wall = self.ni_sep_val * 1.0
        ni_ib = np.linspace(ni_ib_wall, self.ni_sep_val, self.xi_ib_pts, endpoint=False)
        ni_ob = np.linspace(self.ni_sep_val, ni_ob_wall, self.xi_ob_pts, endpoint=True)

        ne_ib_wall = self.ne_sep_val * 1.0
        ne_ob_wall = self.ne_sep_val * 1.0
        ne_ib = np.linspace(ne_ib_wall, self.ne_sep_val, self.xi_ib_pts, endpoint=False)
        ne_ob = np.linspace(self.ne_sep_val, ne_ob_wall, self.xi_ob_pts, endpoint=True)

        Ti_ib_wall = self.Ti_J_sep_val * 1.0
        Ti_ob_wall = self.Ti_J_sep_val * 1.0
        Ti_ib = np.linspace(Ti_ib_wall, self.Ti_J_sep_val, self.xi_ib_pts, endpoint=False)
        Ti_ob = np.linspace(self.Ti_J_sep_val, Ti_ob_wall, self.xi_ob_pts, endpoint=True)

        Te_ib_wall = self.Te_J_sep_val * 1.0
        Te_ob_wall = self.Te_J_sep_val * 1.0
        Te_ib = np.linspace(Te_ib_wall, self.Te_J_sep_val, self.xi_ib_pts, endpoint=False)
        Te_ob = np.linspace(self.Te_J_sep_val, Te_ob_wall, self.xi_ob_pts, endpoint=True)

        dnidr_ib_wall = dnidr_sep[0]
        dnidr_ob_wall = dnidr_sep[-1]
        dnidr_ib = np.linspace(dnidr_ib_wall, dnidr_sep[0], self.xi_ib_pts, endpoint=False)
        dnidr_ob = np.linspace(dnidr_sep[-1], dnidr_ob_wall, self.xi_ob_pts, endpoint=True)

        dnedr_ib_wall = dnedr_sep[0]
        dnedr_ob_wall = dnedr_sep[-1]
        dnedr_ib = np.linspace(dnedr_ib_wall, dnedr_sep[0], self.xi_ib_pts, endpoint=False)
        dnedr_ob = np.linspace(dnedr_sep[-1], dnedr_ob_wall, self.xi_ob_pts, endpoint=True)

        dTidr_ib_wall = dTidr_sep[0]
        dTidr_ob_wall = dTidr_sep[-1]
        dTidr_ib = np.linspace(dTidr_ib_wall, dTidr_sep[0], self.xi_ib_pts, endpoint=False)
        dTidr_ob = np.linspace(dTidr_sep[-1], dTidr_ob_wall, self.xi_ob_pts, endpoint=True)

        dTedr_ib_wall = dTedr_sep[0]
        dTedr_ob_wall = dTedr_sep[-1]
        dTedr_ib = np.linspace(dTedr_ib_wall, dTedr_sep[0], self.xi_ib_pts, endpoint=False)
        dTedr_ob = np.linspace(dTedr_sep[-1], dTedr_ob_wall, self.xi_ob_pts, endpoint=True)

        BT_ib_wall = BT_sep[0]
        BT_ob_wall = BT_sep[-1]
        BT_ib = np.linspace(BT_ib_wall, BT_sep[0], self.xi_ib_pts, endpoint=False)
        BT_ob = np.linspace(BT_sep[-1], BT_ob_wall, self.xi_ob_pts, endpoint=True)

        ni_xi = np.concatenate((ni_ib, ni_sep, ni_ob))
        ne_xi = np.concatenate((ne_ib, ne_sep, ne_ob))
        Ti_xi = np.concatenate((Ti_ib, Ti_sep, Ti_ob))
        Te_xi = np.concatenate((Te_ib, Te_sep, Te_ob))
        dnidr_xi = np.concatenate((dnidr_ib, dnidr_sep, dnidr_ob))
        dnedr_xi = np.concatenate((dnedr_ib, dnedr_sep, dnedr_ob))
        dTidr_xi = np.concatenate((dTidr_ib, dTidr_sep, dTidr_ob))
        dTedr_xi = np.concatenate((dTedr_ib, dTedr_sep, dTedr_ob))
        BT_xi = np.concatenate((BT_ib, BT_sep, BT_ob))

        ib_leg_length = self.ib_div_line_cut.length
        ob_leg_length = self.ob_div_line_cut.length
        sep_length = self.main_sep_line_closed.length
        ib_frac = ib_leg_length / (ib_leg_length + sep_length + ob_leg_length)
        sep_frac = sep_length / (ib_leg_length + sep_length + ob_leg_length)
        ob_frac = ob_leg_length / (ib_leg_length + sep_length + ob_leg_length)

        xi_ib_div = np.linspace(0,
                                ib_frac + sep_frac * self.ib_trim_off,
                                self.xi_ib_pts,
                                endpoint=False)

        xi_sep = np.linspace(ib_frac + sep_frac * self.ib_trim_off,
                             ib_frac + sep_frac * self.ib_trim_off + sep_frac - (self.ib_trim_off + self.ob_trim_off),
                             self.xi_sep_pts,
                             endpoint=False)

        xi_ob_div = np.linspace(
            ib_frac + sep_frac * self.ib_trim_off + sep_frac - (self.ib_trim_off + self.ob_trim_off),
            1,
            self.xi_ob_pts,
            endpoint=True)

        xi_pts = np.concatenate((xi_ib_div, xi_sep, xi_ob_div))

        # model perpendicular particle and heat transport using Bohm Diffusion
        D_perp = Ti_xi / (16.0 * elementary_charge * BT_xi)
        Chi_perp = 5.0 * Ti_xi / (32.0 * elementary_charge * BT_xi)

        Gamma_perp = -D_perp * dnidr_xi
        Q_perp = -ni_xi * Chi_perp * dTidr_xi - \
                 3.0 * Ti_xi * D_perp * dnidr_xi

        delta_sol_n = D_perp * ni_xi / Gamma_perp
        delta_sol_T = Chi_perp / \
                      (Q_perp / (ni_xi * Ti_xi) \
                       - 3.0 * D_perp / delta_sol_n)
        delta_sol_E = 2 / 7 * delta_sol_T
        # plt.plot(delta_sol_n)

        # plt.axis('equal')
        # plt.contourf(R_temp,
        #     Z_temp,
        #     ni_grid,
        #     500)
        # plt.colorbar()

        # plt.plot(xi_pts, ni_xi)
        # plt.plot(dnidr_sep_raw)
        # plt.plot(xi_pts, dTidr_sep_smooth)
        # plt.plot(xi_pts, np.nan_to_num(-Ti_J_sep_val/dTidr_sep_smooth))
        # plt.plot(xi_pts, BT_sep)
        # plt.plot(xi_pts, BT_sep_smooth)
        # plt.plot(xi_pts, D_perp, label='D_perp')
        # plt.plot(xi_pts, Chi_perp, label='Chi_perp')
        # plt.plot(xi_pts, Gamma_perp, label='Gamma_perp')
        # plt.plot(xi_pts, Q_perp, label='Q_perp')
        # plt.plot(xi_pts, -ni_sep_val * Chi_perp * dTidr_sep_smooth, label='Q term 1')
        # plt.plot(xi_pts, 3.0 * Ti_J_sep_val * D_perp * dnidr_sep_smooth, label='Q term 2')
        # plt.plot(xi_pts, Q_perp, label='Q_perp')
        # plt.plot(xi_pts, 3.0*D_perp*ni_sep_val*Ti_J_sep_val/delta_sol_n, label='term2')
        # plt.plot(xi_pts, delta_sol_n, label='delta_sol_n')
        # plt.plot(xi_pts, delta_sol_T, label='delta_sol_T')
        # plt.plot(xi_pts, delta_sol_E, label='delta_sol_E')
        # plt.legend()
        # plt.plot()
        # sys.exit()
        # delta_n_xi_ib  = np.array([0])
        # delta_n_xi_ib  = np.array([0])

        # pts = np.concatenate((np.array([0.0]),
        #              np.linspace(ib_frac, ib_frac+sep_frac, xi_sep_pts),
        #              np.array([1.0])))
        # vals = np.concatenate((delta_n_xi_ib, delta_sol_n_trim, delta_n_xi_ib))

        # delta_n_xi_sep = griddata(pts,
        # vals,
        # xi_pts,
        # method='linear',
        # fill_value='np.nan')

        r_max = 0.45
        twoptdiv_r_pts = 20

        r_pts = np.linspace(0, r_max, twoptdiv_r_pts)
        xi, r = np.meshgrid(xi_pts, r_pts)
        sol_ni = ni_xi * np.exp(-r / delta_sol_n)
        sol_ne = ne_xi * np.exp(-r / delta_sol_n)
        sol_Ti = Ti_xi * np.exp(-r / delta_sol_T)
        sol_Te = Te_xi * np.exp(-r / delta_sol_T)

        # draw sol lines through 2d strip model to get n, T along the lines
        sol_line_dist = np.zeros((len(xi_pts), len(self.sol_lines_cut)))
        sol_nT_pts = np.zeros((len(xi_pts), 2, len(self.sol_lines_cut)))
        for i, sol_line in enumerate(self.sol_lines_cut):
            for j, xi_val in enumerate(xi_pts):
                sol_pt = sol_line.interpolate(xi_val, normalized=True)
                sol_nT_pts[j, :, i] = np.asarray(sol_pt.xy).T
                sep_pt_pos = self.entire_sep_line.project(sol_pt, normalized=True)
                sep_pt = self.entire_sep_line.interpolate(sep_pt_pos, normalized=True)
                sol_line_dist[j, i] = sol_pt.distance(sep_pt)

        sol_line_ni = np.zeros((len(xi_pts), len(self.sol_lines_cut)))
        sol_line_ne = np.zeros((len(xi_pts), len(self.sol_lines_cut)))
        sol_line_Ti = np.zeros((len(xi_pts), len(self.sol_lines_cut)))
        sol_line_Te = np.zeros((len(xi_pts), len(self.sol_lines_cut)))
        for i, sol_line in enumerate(self.sol_lines_cut):
            sol_line_ni[:, i] = griddata(np.column_stack((xi.flatten(), r.flatten())),
                                         sol_ni.flatten(),
                                         np.column_stack((np.linspace(0, 1, len(xi_pts)), sol_line_dist[:, i])),
                                         method='linear')
            sol_line_ne[:, i] = griddata(np.column_stack((xi.flatten(), r.flatten())),
                                         sol_ne.flatten(),
                                         np.column_stack((np.linspace(0, 1, len(xi_pts)), sol_line_dist[:, i])),
                                         method='linear')
            sol_line_Ti[:, i] = griddata(np.column_stack((xi.flatten(), r.flatten())),
                                         sol_Ti.flatten(),
                                         np.column_stack((np.linspace(0, 1, len(xi_pts)), sol_line_dist[:, i])),
                                         method='linear')
            sol_line_Te[:, i] = griddata(np.column_stack((xi.flatten(), r.flatten())),
                                         sol_Te.flatten(),
                                         np.column_stack((np.linspace(0, 1, len(xi_pts)), sol_line_dist[:, i])),
                                         method='linear')

        # append to master arrays
        for i, line in enumerate(self.sol_lines_cut):
            pts_ni_sol = np.column_stack((sol_nT_pts[:, :, i], sol_line_ni[:, i]))
            pts_ne_sol = np.column_stack((sol_nT_pts[:, :, i], sol_line_ne[:, i]))
            pts_Ti_sol = np.column_stack(
                (sol_nT_pts[:, :, i], sol_line_Ti[:, i] / 1.0E3 / 1.6021E-19))  # converting back to kev
            pts_Te_sol = np.column_stack(
                (sol_nT_pts[:, :, i], sol_line_Te[:, i] / 1.0E3 / 1.6021E-19))  # converting back to kev

            self.ni_pts = np.vstack((self.ni_pts, pts_ni_sol))
            self.ne_pts = np.vstack((self.ne_pts, pts_ne_sol))
            self.Ti_kev_pts = np.vstack((self.Ti_kev_pts, pts_Ti_sol))
            self.Te_kev_pts = np.vstack((self.Te_kev_pts, pts_Te_sol))

        # draw wall line through 2d strip model to get n, T along the line
        wall_pts = np.asarray(self.wall_line.xy).T
        ib_int_pt = np.asarray(self.ib_div_line.intersection(self.wall_line).xy).T
        ob_int_pt = self.ob_div_line.intersection(self.wall_line)
        wall_start_pos = np.where((wall_pts == ib_int_pt).all(axis=1))[0][0]
        wall_line_rolled = LineString(np.roll(wall_pts, -wall_start_pos, axis=0))
        wall_line_cut = cut(wall_line_rolled,
                            wall_line_rolled.project(ob_int_pt, normalized=True))[0]
        # add points to wall line for the purpose of getting n, T along the wall. These points
        # won't be added to the main wall line or included in the triangulation.
        # for i, v in enumerate(np.linspace(0, 1, 300)):
        # # interpolate along wall_line_cut to find point to add
        # pt = wall_line_cut.interpolate(v, normalized=True)
        # # add point to wall_line_cut
        # union = wall_line_cut.union(pt)
        # result = [geom for geom in polygonize(union)][0]
        # wall_line_cut = LineString(result.exterior.coords)

        wall_nT_pts = np.asarray(wall_line_cut)
        num_wall_pts = len(wall_nT_pts)
        wall_pos_norm = np.zeros(num_wall_pts)
        wall_dist = np.zeros(num_wall_pts)

        for i, pt in enumerate(wall_nT_pts):
            wall_pt = Point(pt)
            sep_pt_pos = self.entire_sep_line.project(Point(wall_pt), normalized=True)
            sep_pt = self.entire_sep_line.interpolate(sep_pt_pos, normalized=True)
            wall_pos_norm[i] = wall_line_cut.project(wall_pt, normalized=True)
            wall_dist[i] = wall_pt.distance(sep_pt)

        wall_ni = griddata(np.column_stack((xi.flatten(), r.flatten())),
                           sol_ni.flatten(),
                           np.column_stack((wall_pos_norm, wall_dist)),
                           method='linear')
        wall_ne = griddata(np.column_stack((xi.flatten(), r.flatten())),
                           sol_ne.flatten(),
                           np.column_stack((wall_pos_norm, wall_dist)),
                           method='linear')
        wall_Ti = griddata(np.column_stack((xi.flatten(), r.flatten())),
                           sol_Ti.flatten(),
                           np.column_stack((wall_pos_norm, wall_dist)),
                           method='linear')
        wall_Te = griddata(np.column_stack((xi.flatten(), r.flatten())),
                           sol_Te.flatten(),
                           np.column_stack((wall_pos_norm, wall_dist)),
                           method='linear')

        # set minimum wall densities and temperatures
        # TODO: this needs to be more robust
        wall_ni_min = 1.0E15
        wall_ne_min = 1.0E15
        wall_Ti_min = 0.02 * 1.0E3 * 1.6021E-19
        wall_Te_min = 0.02 * 1.0E3 * 1.6021E-19

        wall_ni[wall_ni < wall_ni_min] = wall_ni_min
        wall_ne[wall_ne < wall_ne_min] = wall_ne_min
        wall_Ti[wall_Ti < wall_Ti_min] = wall_Ti_min
        wall_Te[wall_Te < wall_Te_min] = wall_Te_min

        # append to master arrays
        pts_ni_wall = np.column_stack((wall_nT_pts, wall_ni))
        pts_ne_wall = np.column_stack((wall_nT_pts, wall_ne))
        pts_Ti_wall = np.column_stack((wall_nT_pts, wall_Ti / 1.0E3 / 1.6021E-19))  # converting back to kev
        pts_Te_wall = np.column_stack((wall_nT_pts, wall_Te / 1.0E3 / 1.6021E-19))  # converting back to kev

        self.ni_pts = np.vstack((self.ni_pts, pts_ni_wall))
        self.ne_pts = np.vstack((self.ne_pts, pts_ne_wall))
        self.Ti_kev_pts = np.vstack((self.Ti_kev_pts, pts_Ti_wall))
        self.Te_kev_pts = np.vstack((self.Te_kev_pts, pts_Te_wall))

        # plt.contourf(xi, r, np.log10(sol_ni), 500)
        # plt.colorbar()
        # for i, v in enumerate(self.sol_lines_cut):
        # plt.plot(xi_pts, sol_line_dist[:, i])
        # plt.plot(np.linspace(0, 1, num_wall_pts), wall_dist, color='black')

    def _pfr_nT(self):

        pfr_pts = np.asarray(self.pfr_line.xy).T
        pfr_ni = np.zeros(len(pfr_pts)) + self.pfr_ni_val
        pfr_ne = np.zeros(len(pfr_pts)) + self.pfr_ne_val
        pfr_Ti = np.zeros(len(pfr_pts)) + self.pfr_Ti_val
        pfr_Te = np.zeros(len(pfr_pts)) + self.pfr_Te_val

        pts_ni_pfr = np.column_stack((pfr_pts, pfr_ni))
        pts_ne_pfr = np.column_stack((pfr_pts, pfr_ne))
        pts_Ti_pfr = np.column_stack((pfr_pts, pfr_Ti))
        pts_Te_pfr = np.column_stack((pfr_pts, pfr_Te))

        self.ni_pts = np.vstack((self.ni_pts, pts_ni_pfr))
        self.ne_pts = np.vstack((self.ne_pts, pts_ne_pfr))
        self.Ti_kev_pts = np.vstack((self.Ti_kev_pts, pts_Ti_pfr))
        self.Te_kev_pts = np.vstack((self.Te_kev_pts, pts_Te_pfr))

        # grid_x, grid_y = np.mgrid[1:2.5:500j, -1.5:1.5:500j]
        # ni_for_plot = griddata(self.ni_pts[:, :-1], self.ni_pts[:, -1], (grid_x, grid_y))
        # Ti_for_plot = griddata(self.Ti_kev_pts[:, :-1], self.Ti_kev_pts[:, -1], (grid_x, grid_y))
        # plt.contourf(grid_x, grid_y, np.log10(Ti_for_plot), 500)
        # plt.colorbar()
        # sys.exit()

    def _triangle_prep(self):

        sol_pol_pts = self.core_pol_pts + self.ib_div_pol_pts + self.ob_div_pol_pts

        # GET POINTS FOR TRIANGULATION
        # main seperatrix

        print 'Number of poloidal points in core: ', self.core_pol_pts
        print 'Number of inbooard poloidal points at the divertor: ', self.ib_div_pol_pts
        print 'Number of outboard poloidal points at the divertor', self.ob_div_pol_pts

        sep_pts = np.zeros((self.core_pol_pts, 2))
        for i, v in enumerate(np.linspace(0, 1, self.core_pol_pts, endpoint=False)):
            sep_pts[i] = np.asarray(self.main_sep_line.interpolate(v, normalized=True).xy).T[0]

        # inboard divertor leg
        ib_div_pts = np.zeros((self.ib_div_pol_pts, 2))
        for i, v in enumerate(np.linspace(0, 1, self.ib_div_pol_pts, endpoint=True)):  # skipping the x-point (point 0)
            ib_div_pts[i] = np.asarray(self.ib_div_line_cut.interpolate(v, normalized=True).xy).T[0]

        # outboard divertor leg
        ob_div_pts = np.zeros((self.ob_div_pol_pts, 2))
        for i, v in enumerate(np.linspace(0, 1, self.ob_div_pol_pts, endpoint=True)):  # skipping the x-point (point 0)
            ob_div_pts[i] = np.asarray(self.ob_div_line_cut.interpolate(v, normalized=True).xy).T[0]

        # core
        core_pts = np.zeros((self.core_pol_pts * len(self.core_lines), 2))
        for num, line in enumerate(self.core_lines):
            for i, v in enumerate(np.linspace(0, 1, self.core_pol_pts, endpoint=False)):
                core_pts[num * self.core_pol_pts + i] = np.asarray(line.interpolate(v, normalized=True).xy).T[0]

        self.core_ring = LinearRing(core_pts[:self.core_pol_pts])

        # sol
        sol_pts = np.zeros((sol_pol_pts * len(self.sol_lines_cut), 2))
        for num, line in enumerate(self.sol_lines_cut):
            for i, v in enumerate(np.linspace(0, 1, sol_pol_pts, endpoint=True)):
                sol_pts[num * sol_pol_pts + i] = np.asarray(line.interpolate(v, normalized=True).xy).T[0]

                # wall
        wall_pts = np.asarray(self.wall_line.coords)[:-1]
        self.wall_ring = LinearRing(wall_pts)

        all_pts = np.vstack((sep_pts,
                             ib_div_pts,
                             ob_div_pts,
                             core_pts,
                             sol_pts,
                             wall_pts))

        # CREATE SEGMENTS FOR TRIANGULATION
        # WHEN DOING WALL, CHECK EACH POINT TO SEE IF IT HAS ALREADY BEEN
        # CREATED. IF SO, USE THE NUMBER OF THAT POINT AND DELETE THE WALL
        # VERSION OF IT IN THE ALL_PTS ARRAY.

        sep_segs = np.column_stack((np.arange(self.core_pol_pts),
                                    np.roll(np.arange(self.core_pol_pts), -1)))

        ib_div_segs = np.column_stack((np.arange(self.ib_div_pol_pts),
                                       np.roll(np.arange(self.ib_div_pol_pts), -1)))[:-1]

        ob_div_segs = np.column_stack((np.arange(self.ob_div_pol_pts),
                                       np.roll(np.arange(self.ob_div_pol_pts), -1)))[:-1]

        core_segs = np.zeros((0, 2), dtype='int')
        for i, v in enumerate(self.core_lines):
            new_segs = np.column_stack((np.arange(self.core_pol_pts),
                                        np.roll(np.arange(self.core_pol_pts), -1))) \
                       + self.core_pol_pts * i
            core_segs = np.vstack((core_segs, new_segs))

        sol_segs = np.zeros((0, 2), dtype='int')
        for i, v in enumerate(self.sol_lines):
            new_segs = np.column_stack((np.arange(sol_pol_pts),
                                        np.roll(np.arange(sol_pol_pts), -1)))[:-1] \
                       + sol_pol_pts * i
            sol_segs = np.vstack((sol_segs, new_segs))

        wall_segs = np.column_stack((np.arange(len(wall_pts)),
                                     np.roll(np.arange(len(wall_pts)), -1)))

        all_segs = np.vstack((sep_segs,
                              ib_div_segs + len(sep_segs),
                              ob_div_segs + len(ib_div_segs) + len(sep_segs) + 1,
                              core_segs + len(ob_div_segs) + len(ib_div_segs) + len(sep_segs) + 1 + 1,
                              sol_segs + len(core_segs) + len(ob_div_segs) + len(ib_div_segs) + len(sep_segs) + 1 + 1,
                              wall_segs + len(sol_segs) + len(core_segs) + len(ob_div_segs) + len(ib_div_segs) + len(
                                  sep_segs) + 1 + 1 + self.num_sollines
                              ))

        all_pts_unique = np.unique(all_pts, axis=0)

        # CLEANUP
        # NOTE: this process will result in a segments array that looks fairly chaotic,
        # but will ensure that the triangulation goes smoothly.

        # loop over each point in all_segs
        # look up the point's coordinates in all_pts
        # find the location of those coordinates in all_pts_unique
        # put that location in the corresponding location in all_segs_unique

        all_segs_unique = np.zeros(all_segs.flatten().shape, dtype='int')
        for i, pt in enumerate(all_segs.flatten()):
            pt_coords = all_pts[pt]
            loc_unique = np.where((all_pts_unique == pt_coords).all(axis=1))[0][0]
            all_segs_unique[i] = loc_unique
        all_segs_unique = all_segs_unique.reshape(-1, 2)

        # # OUTPUT .poly FILE AND RUN TRIANGLE PROGRAM
        if not os.path.exists('tri_tmp'):
            os.makedirs('tri_tmp')
        open('tri_tmp/exp_mesh.poly', 'w').close()
        outfile = open('tri_tmp/exp_mesh.poly', 'ab')
        filepath = os.path.realpath(outfile.name)
        np.savetxt(outfile,
                   np.array([all_pts_unique.shape[0], 2, 0, 0])[None],
                   fmt='%i %i %i %i')
        np.savetxt(outfile,
                   np.column_stack((np.arange(len(all_pts_unique)),
                                    all_pts_unique)),
                   fmt='%i %f %f')
        np.savetxt(outfile,
                   np.array([all_segs_unique.shape[0], 0])[None],
                   fmt='%i %i')
        np.savetxt(outfile,
                   np.column_stack((np.arange(len(all_segs_unique)),
                                    all_segs_unique,
                                    np.zeros(len(all_segs_unique), dtype='int'))),
                   fmt='%i %i %i %i')
        np.savetxt(outfile,
                   np.array([1])[None],
                   fmt='%i')
        np.savetxt(outfile,
                   np.array([1, self.m_axis[0], self.m_axis[1]])[None],
                   fmt='%i %f %f')
        np.savetxt(outfile,
                   np.array([0])[None],
                   fmt='%i')
        outfile.close()

        # construct options to pass to triangle, as specified in input file
        # refer to https://www.cs.cmu.edu/~quake/triangle.html



        tri_options = '-p'
        try:
            tri_options = tri_options + 'q' + str(self.tri_min_angle)
        except:
            pass

        try:
            print self.tri_min_area
            tri_options = tri_options + 'a' + str(self.tri_min_area)
        except:
            pass

        tri_options = tri_options + 'nz'
        # call triangle
        try:
            print tri_options
            call(['triangle', tri_options, filepath])
        except AttributeError:
            try:
                call(['triangle', tri_options, filepath])
            except:
                print 'triangle could not be found. Stopping.'
                sys.exit

    def _read_triangle(self):
        # # READ TRIANGLE OUTPUT

        # # DECLARE FILE PATHS
        nodepath = os.getcwd() + '/tri_tmp/exp_mesh.1.node'
        elepath = os.getcwd() + '/tri_tmp/exp_mesh.1.ele'
        neighpath = os.getcwd() + '/tri_tmp/exp_mesh.1.neigh'

        # # GET NODE DATA
        with open(nodepath, 'r') as node:
            # dummy = next(mil_mesh)
            nodecount = re.findall(r'\d+', next(node))
            nNodes = int(nodecount[0])
            nodenum = np.zeros(nNodes)
            nodesx = np.zeros(nNodes)
            nodesy = np.zeros(nNodes)

            for i in range(0, nNodes):
                data1 = re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?', next(node))
                nodenum[i] = int(data1[0])
                nodesx[i] = data1[1]
                nodesy[i] = data1[2]

        # # GET TRIANGLE DATA
        with open(elepath, 'r') as tri_file:
            tricount = re.findall(r'\d+', next(tri_file))
            nTri = int(tricount[0])
            print 'number of triangles = ', nTri
            triangles = np.zeros((nTri, 3))
            tri_regions = np.zeros(nTri)
            for i in range(0, nTri):
                data1 = re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?', next(tri_file))
                triangles[i, 0] = data1[1]
                triangles[i, 1] = data1[2]
                triangles[i, 2] = data1[3]
                # tri_regions[i] = data1[4]
        triangles = triangles.astype('int')
        tri_regions = tri_regions.astype('int')

        # # GET NEIGHBOR DATA
        with open(neighpath, 'r') as neigh_file:
            neighcount = re.findall(r'\d+', next(neigh_file))
            nNeigh = int(neighcount[0])
            neighbors = np.zeros((nNeigh, 3))
            for i in range(0, nNeigh):
                data1 = re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)?', next(neigh_file))
                neighbors[i, 0] = data1[1]
                neighbors[i, 1] = data1[2]
                neighbors[i, 2] = data1[3]
        neighbors = neighbors.astype('int')

        # # REARRANGE TRIANGLES TO CONFORM TO GTNEUT CONVENTION
        triangles = np.fliplr(triangles)  # triangle vertices are given counterclockwise, but we want clockwise
        neighbors = np.fliplr(neighbors)  # neighbor 1 is opposite vertex 1, so also counterclockwise

        y = np.zeros(3)
        for i, tri in enumerate(triangles):
            # Find lowest value of y component of vertices
            y[0] = nodesy[tri[0]]
            y[1] = nodesy[tri[1]]
            y[2] = nodesy[tri[2]]
            miny = np.amin(y)
            miny_count = np.sum(y == miny)
            if miny_count == 1:
                # identify position of minimum and roll array accordingly
                miny_index = np.where(y == miny)[0][0]
            else:
                # identify which points are the two minima and determine
                # which of them is farthest to the left (or right if I change it)
                miny_index = np.where(y == miny)[0][
                    1]  # change this 1 to a zero to choose the rightmost of the two bottom vertices
            triangles[i] = np.roll(triangles[i], -1 * miny_index)
            neighbors[i] = np.roll(neighbors[i],
                                   -1 * miny_index - 2)  # the -2 is because the side 1 is opposite vertex 1. We want side 1 to start at vertex 1

        # # GET VALUES TO ORIENT THE FIRST CELL WHEN PLOTTING
        point1_x = nodesx[triangles[0, 0]]
        point1_y = nodesy[triangles[0, 0]]
        point2_x = nodesx[triangles[0, 1]]
        point2_y = nodesy[triangles[0, 1]]
        point3_x = nodesx[triangles[0, 2]]
        point3_y = nodesy[triangles[0, 2]]

        cell1_ctr_x = (point1_x + point2_x + point3_x) / 3
        cell1_ctr_y = (point1_y + point2_y + point3_y) / 3

        # # CALCULATE ANGLE BY WHICH TO ROTATE THE FIRST CELL WHEN PLOTTING
        cell1_theta0 = degrees(getangle([point3_x, point3_y], [point1_x, point1_y]))

        # # GET VALUES TO ORIENT THE FIRST CELL WHEN PLOTTING
        point1_x = nodesx[triangles[0, 0]]
        point1_y = nodesy[triangles[0, 0]]
        point2_x = nodesx[triangles[0, 1]]
        point2_y = nodesy[triangles[0, 1]]
        point3_x = nodesx[triangles[0, 2]]
        point3_y = nodesy[triangles[0, 2]]

        cell1_ctr_x = (point1_x + point2_x + point3_x) / 3
        cell1_ctr_y = (point1_y + point2_y + point3_y) / 3

        # # CALCULATE ANGLE BY WHICH TO ROTATE THE FIRST CELL WHEN PLOTTING
        cell1_theta0 = degrees(getangle([point3_x, point3_y], [point1_x, point1_y]))

        # # CALCULATE MID POINTS OF TRIANGLES, AS WELL AS MIDPOINTS FOR EACH FACE
        ptsx = np.zeros((nTri, 3))
        ptsy = np.zeros((nTri, 3))
        # for index, tri in ndenumerate(triangles):
        for i in range(0, nTri):
            ptsx[i, 0] = nodesx[triangles[i, 0]]
            ptsy[i, 0] = nodesy[triangles[i, 0]]
            ptsx[i, 1] = nodesx[triangles[i, 1]]
            ptsy[i, 1] = nodesy[triangles[i, 1]]
            ptsx[i, 2] = nodesx[triangles[i, 2]]
            ptsy[i, 2] = nodesy[triangles[i, 2]]

        mid_x = np.mean(ptsx, axis=1)
        mid_y = np.mean(ptsy, axis=1)
        self.midpts = np.column_stack((mid_x, mid_y))

        side1_midx = (ptsx[:, 0] + ptsx[:, 1]) / 2
        side2_midx = (ptsx[:, 1] + ptsx[:, 2]) / 2
        side3_midx = (ptsx[:, 2] + ptsx[:, 0]) / 2

        side1_midy = (ptsy[:, 0] + ptsy[:, 1]) / 2
        side2_midy = (ptsy[:, 1] + ptsy[:, 2]) / 2
        side3_midy = (ptsy[:, 2] + ptsy[:, 0]) / 2

        side1_midpt = np.column_stack((side1_midx, side1_midy))
        side2_midpt = np.column_stack((side2_midx, side2_midy))
        side3_midpt = np.column_stack((side3_midx, side3_midy))

        # COMBINE POINTS FOR THE PLASMA, SOL, AND DIVERTOR REGIONS
        # first fill in plasma cells
        plasmacells = np.zeros((1, 2))
        pcellnum = nTri
        pcellcount = 0

        for index, nei in enumerate(neighbors):
            # for each face of the cell, find the mid-point and check if it falls in line
            side1inline = isinline(side1_midpt[index], self.core_ring)
            side2inline = isinline(side2_midpt[index], self.core_ring)
            side3inline = isinline(side3_midpt[index], self.core_ring)

            if side1inline or side2inline or side3inline:
                nb = (nei == -1).sum()  # count number of times -1 occurs in nei
                if nb == 1:  # cell has one plasma border

                    # create plasma cell
                    plasmacells[pcellcount, 0] = pcellnum
                    plasmacells[pcellcount, 1] = index
                    plasmacells = np.vstack((plasmacells, [0, 0]))
                    # update neighbors
                    nei[np.argmax(nei == -1)] = pcellnum
                    # get ready for next run
                    pcellnum += 1
                    pcellcount += 1
                elif nb == 2:
                    # cell has two plasma borders (this will probably never happen. It would require a local
                    # concavity in the inner-most meshed flux surface)
                    # create plasma cell # 1
                    plasmacells[pcellcount, 0] = pcellnum
                    plasmacells[pcellcount, 1] = index
                    plasmacells = np.vstack((plasmacells, [0, 0]))
                    # update neighbors
                    nei[np.argmax(nei == -1)] = pcellnum
                    # get ready for next run
                    pcellnum += 1
                    pcellcount += 1

                    # create plasma cell # 2
                    plasmacells[pcellcount, 0] = pcellnum
                    plasmacells[pcellcount, 1] = index
                    plasmacells = np.vstack((plasmacells, [0, 0]))
                    # update neighbors
                    nei[np.argmax(nei == -1)] = pcellnum
                    # get ready for next run
                    pcellnum += 1
                    pcellcount += 1
        plasmacells = np.delete(plasmacells, -1, 0)
        plasmacells = plasmacells.astype('int')

        # now fill in wall cells
        wallcells = np.zeros((1, 6))
        wcellnum = pcellnum  # was already advanced in the plasmacell loop. Don't add 1.
        wcellcount = 0

        for index, nei in enumerate(neighbors):
            # for each face of the cell, find the mid-point and check if it falls in line
            side1inline = isinline(side1_midpt[index], self.wall_ring)
            side2inline = isinline(side2_midpt[index], self.wall_ring)
            side3inline = isinline(side3_midpt[index], self.wall_ring)

            if side1inline or side2inline or side3inline:
                nb = (nei == -1).sum()  # count number of times -1 occurs in nei
                if nb == 1:  # cell has one wall border
                    # identify the side that is the wall cell
                    sidenum = np.where(np.asarray([side1inline, side2inline, side3inline]))[0][0]
                    if sidenum == 0:
                        pt = side1_midpt[index]
                    elif sidenum == 1:
                        pt = side2_midpt[index]
                    elif sidenum == 2:
                        pt = side3_midpt[index]

                    # create wall cell
                    wallcells[wcellcount, 0] = wcellnum
                    wallcells[wcellcount, 1] = index
                    wallcells[wcellcount, 2] = griddata(self.ni_pts[:, :-1], self.ni_pts[:, -1], pt, method='nearest',
                                                        rescale=True)
                    wallcells[wcellcount, 3] = griddata(self.ne_pts[:, :-1], self.ne_pts[:, -1], pt, method='nearest',
                                                        rescale=True)
                    wallcells[wcellcount, 4] = griddata(self.Ti_kev_pts[:, :-1], self.Ti_kev_pts[:, -1], pt,
                                                        method='nearest', rescale=True)
                    wallcells[wcellcount, 5] = griddata(self.Te_kev_pts[:, :-1], self.Te_kev_pts[:, -1], pt,
                                                        method='nearest', rescale=True)
                    wallcells = np.vstack((wallcells, [0, 0, 0, 0, 0, 0]))
                    # update neighbors
                    nei[np.argmax(nei == -1)] = wcellnum
                    # get ready for next run
                    wcellnum += 1
                    wcellcount += 1
                elif nb == 2:  # cell has two wall borders (This can easily happen because the wall has many concave points.)
                    # create wall cell # 1
                    wallcells[wcellcount, 0] = wcellnum
                    wallcells[wcellcount, 1] = index
                    wallcells = np.vstack((wallcells, [0, 0, 0, 0, 0, 0]))
                    # update neighbors
                    nei[np.argmax(nei == -1)] = wcellnum
                    # get ready for next run
                    wcellnum += 1
                    wcellcount += 1

                    # create wall cell # 2
                    wallcells[wcellcount, 0] = wcellnum
                    wallcells[wcellcount, 1] = index
                    wallcells = np.vstack((wallcells, [0, 0, 0, 0, 0, 0]))
                    # update neighbors
                    nei[np.argmax(nei == -1)] = wcellnum
                    # get ready for next run
                    wcellnum += 1
                    wcellcount += 1
        wallcells = np.delete(wallcells, -1, 0)
        wallcells = wallcells.astype('int')

        # # POPULATE CELL DENSITIES AND TEMPERATURES
        # create array of all points in plasma, sol, id, and od
        # tri_param = np.vstack((plasma_param, sol_param, id_param, od_param))

        ni_tri = griddata(self.ni_pts[:, :-1],
                          self.ni_pts[:, -1],
                          (mid_x, mid_y),
                          method='linear',
                          fill_value=0,
                          rescale=True)
        ne_tri = griddata(self.ne_pts[:, :-1],
                          self.ne_pts[:, -1],
                          (mid_x, mid_y),
                          method='linear',
                          fill_value=0,
                          rescale=True)
        Ti_tri = griddata(self.Ti_kev_pts[:, :-1],
                          self.Ti_kev_pts[:, -1],
                          (mid_x, mid_y),
                          method='linear',
                          fill_value=0,
                          rescale=True)
        Te_tri = griddata(self.Te_kev_pts[:, :-1],
                          self.Te_kev_pts[:, -1],
                          (mid_x, mid_y),
                          method='linear',
                          fill_value=0,
                          rescale=True)

        # ni_tri[ni_tri<1.0E16] = 1.0E16
        # ne_tri[ne_tri<1.0E16] = 1.0E16
        # Ti_tri[Ti_tri<0.002] = 0.002
        # Te_tri[Te_tri<0.002] = 0.002

        # # CALCULATE LENGTHS OF SIDES
        lsides = np.zeros((nTri, 3))
        for i in range(0, nTri):
            lsides[i, 0] = sqrt((ptsx[i, 0] - ptsx[i, 1]) ** 2 + (ptsy[i, 0] - ptsy[i, 1]) ** 2)
            lsides[i, 1] = sqrt((ptsx[i, 1] - ptsx[i, 2]) ** 2 + (ptsy[i, 1] - ptsy[i, 2]) ** 2)
            lsides[i, 2] = sqrt((ptsx[i, 2] - ptsx[i, 0]) ** 2 + (ptsy[i, 2] - ptsy[i, 0]) ** 2)

        # # CALCULATE CELL ANGLES
        angles = np.zeros((nTri, 3))
        for i in range(0, nTri):
            p1 = np.array([ptsx[i, 0], ptsy[i, 0]])
            p2 = np.array([ptsx[i, 1], ptsy[i, 1]])
            p3 = np.array([ptsx[i, 2], ptsy[i, 2]])
            angles[i, 0] = getangle3ptsdeg(p1, p2, p3)
            angles[i, 1] = getangle3ptsdeg(p2, p3, p1)
            angles[i, 2] = getangle3ptsdeg(p3, p1, p2)

        # # WRITE NEUTPY JSON INPUT FILE!
        f = open(os.getcwd() + '/neutpy_in_generated.json', 'w')
        out_dict = {}
        out_dict['nCells'] = nTri
        out_dict['nPlasmReg'] = pcellcount
        out_dict['nWallSegm'] = wcellcount
        out_dict["aneut"] = 2
        out_dict["zion"] = 1
        out_dict["aion"] = 2
        out_dict["tslow"] = 0.002
        out_dict["int_method"] = 'midpoint'
        out_dict["phi_int_pts"] = 10
        out_dict["xi_int_pts"] = 10
        out_dict["xsec_ioni"] = 'degas'
        out_dict["xsec_ione"] = 'degas'
        out_dict["xsec_cx"] = 'degas'
        out_dict["xsec_rec"] = 'degas'
        out_dict["xsec_el"] = 'stacey_thomas'
        out_dict["xsec_eln"] = 'stacey_thomas'
        out_dict["refmod_e"] = 'stacey'
        out_dict["refmod_n"] = 'stacey'

        out_dict["iType"] = np.asarray([0] * nTri + [1] * pcellcount + [2] * wcellcount).tolist()
        out_dict["nSides"] = np.asarray([3] * nTri + [1] * (pcellcount + wcellcount)).tolist()
        out_dict["zwall"] = np.asarray([0] * (nTri + pcellcount) + [6] * wcellcount).tolist()
        out_dict["awall"] = np.asarray([0] * (nTri + pcellcount) + [12] * wcellcount).tolist()
        out_dict["elecTemp"] = Te_tri[:nTri].tolist()
        out_dict["ionTemp"] = Ti_tri[:nTri].tolist()
        out_dict["elecDens"] = ne_tri[:nTri].tolist()
        out_dict["ionDens"] = ni_tri[:nTri].tolist()
        out_dict["twall"] = np.asarray([0] * nTri + [5000] * pcellcount + [0.002] * wcellcount).tolist()
        out_dict["f_abs"] = np.asarray([0] * (nTri + pcellcount) + [0] * wcellcount).tolist()
        out_dict["alb_s"] = np.asarray([0] * nTri + [0] * pcellcount + [0] * wcellcount).tolist()
        out_dict["alb_t"] = np.asarray([0] * nTri + [0] * pcellcount + [0] * wcellcount).tolist()
        out_dict["s_ext"] = np.asarray([0.0] * nTri + [0.0] * pcellcount + [0.0] * wcellcount).tolist()

        out_dict["adjCell"] = neighbors.tolist()
        out_dict["lsides"] = lsides.tolist()
        out_dict["angles"] = angles.tolist()
        out_dict["cell1_ctr_x"] = cell1_ctr_x
        out_dict["cell1_ctr_y"] = cell1_ctr_y
        out_dict["cell1_theta0"] = cell1_theta0

        # Generate the neutpy input JSON

        json.dump(out_dict, f, indent=4)
        f.close()

    def _read_config(self, input_file):
        """
        Read the configuration files when running neutpy without a mesh given
        :param infile: The input configuration file containing radial profiles and plasma magnetic field strength (in T)
        :type infile: str
        """
        config = ConfigParser.RawConfigParser()
        inFile = ConfigParser.RawConfigParser()

        config.read(self.config_loc)
        inFile.read(input_file)
        # Collect configuration from main configuration file

        self.verbose = config.getint('Data', 'verbose')
        self.cpu_cores = config.getint('Data', 'cpu_cores')
        self.corelines_begin = config.getfloat('Data', 'corelines_begin')
        self.num_corelines = config.getint('Data', 'num_corelines')
        self.sollines_psi_max = config.getfloat('Data', 'sollines_psi_max')
        self.num_sollines = config.getint('Data', 'num_sollines')
        self.xi_sep_pts = config.getint('Data', 'xi_sep_pts')
        self.ib_trim_off = config.getfloat('Data', 'ib_trim_off')
        self.ob_trim_off = config.getfloat('Data', 'ob_trim_off')
        self.xi_ib_pts = config.getint('Data', 'xi_ib_pts')
        self.xi_ob_pts = config.getint('Data', 'xi_ob_pts')
        self.core_pol_pts = config.getint('Data', 'core_pol_pts')
        self.ib_div_pol_pts = config.getint('Data', 'ib_div_pol_pts')
        self.ob_div_pol_pts = config.getint('Data', 'ob_div_pol_pts')
        self.pfr_ni_val = config.getfloat('Data', 'pfr_ni_val')
        self.pfr_ne_val = config.getfloat('Data', 'pfr_ne_val')
        self.pfr_Ti_val = config.getfloat('Data', 'pfr_Ti_val')
        self.pfr_Te_val = config.getfloat('Data', 'pfr_Te_val')
        self.tri_min_area = config.getfloat('Data', 'tri_min_area')
        self.tri_min_angle = config.getfloat('Data', 'tri_min_angle')

        # Pull in data from the input file

        self.ne_file = inFile.get('Profiles', 'ne_file')
        self.ni_file = inFile.get('Profiles', 'ni_file')
        self.Te_file = inFile.get('Profiles', 'Te_file')
        self.Ti_file = inFile.get('Profiles', 'Ti_file')
        self.psirz_file = inFile.get('Profiles', 'psirz_file')
        self.wall_file = inFile.get('Profiles', 'wall_file')
        cwd = os.getcwd() + '/'
        input_dir = os.path.dirname(cwd + input_file)
        self.ne_data = np.loadtxt(os.path.join(input_dir, self.ne_file))
        self.ni_data = np.loadtxt(os.path.join(input_dir, self.ni_file))
        self.Te_data = np.loadtxt(os.path.join(input_dir, self.Te_file))
        self.Ti_data = np.loadtxt(os.path.join(input_dir, self.Ti_file))
        self.psirz_exp = np.loadtxt(os.path.join(input_dir, self.psirz_file))
        self.wall_exp = np.loadtxt(os.path.join(input_dir, self.wall_file))

        # Get plasma parameters

        self.BT0 = inFile.getfloat('Plasma', 'BT0')

        self.wall_line = LineString(self.wall_exp)
        self.R = self.psirz_exp[:, 0].reshape(-1, 65)
        self.Z = self.psirz_exp[:, 1].reshape(-1, 65)
        self.psi = self.psirz_exp[:, 2].reshape(-1, 65)

    def _read_neutpy_in(self, infile):
        print ('READING NEUTPY INPUT FILE')

        # populate 0d variables. Need to do this first to calculate the sizes of the other arrays
        with open(infile, 'r') as toneut:
            l = yaml.safe_load(toneut)
            self.nCells = l['nCells']
            self.nPlasmReg = l['nPlasmReg']
            self.nWallSegm = l['nWallSegm']
            self.aneut = l['aneut']
            self.zion = l['zion']
            self.aion = l['aion']
            self.tslow = l['tslow']
            self.int_method = l['int_method']
            self.phi_int_pts = l['phi_int_pts']
            self.xi_int_pts = l['xi_int_pts']
            self.xsec_ioni = l['xsec_ioni']
            self.xsec_ione = l['xsec_ione']
            self.xsec_cx = l['xsec_cx']
            self.xsec_rec = l['xsec_rec']
            self.xsec_el = l['xsec_el']
            self.xsec_eln = l['xsec_eln']
            self.refmod_e = l['refmod_e']
            self.refmod_n = l['refmod_n']
            self.cell1_ctr_x = l['cell1_ctr_x']
            self.cell1_ctr_y = l['cell1_ctr_y']
            self.cell1_theta0 = l['cell1_theta0']
        self.nCells_tot = self.nCells + self.nPlasmReg + self.nWallSegm

        # now we can do the same thing for the 1d and 2d arrays

        self.iType = np.asarray(l['iType'])
        self.nSides = np.asarray(l['nSides'])
        self.zwall = np.asarray(l['zwall'])
        self.awall = np.asarray(l['awall'])
        self.elecTemp = np.asarray(l['elecTemp'])
        self.ionTemp = np.asarray(l['ionTemp'])
        """The Ion Temperature"""
        self.elecDens = np.asarray(l['elecDens'])
        self.ionDens = np.asarray(l['ionDens'])
        self.twall = np.asarray(l['twall'])
        self.f_abs = np.asarray(l['f_abs'])
        self.alb_s = np.asarray(l['alb_s'])
        self.alb_t = np.asarray(l['alb_t'])
        self.s_ext = np.asarray(l['s_ext'])
        self.adjCell = np.array(l['adjCell'])
        self.lsides = np.array(l['lsides'])
        self.angles = np.array(l['angles'])

    def _run(self):

        """
        Run NEUTPY
        """

        # TODO: This and the actual running of neutpy needs to be moved out so that one can pass in an input file.
        self._read_neutpy_in(os.getcwd() + '/neutpy_in_generated.json')

        time0 = time.time()
        try:
            self.cpu_cores
        except:
            self.cpu_cores = 1

        Densities = namedtuple('Densities', 'i e n')
        """A namedtuple of ion, electron, and neutral (i, e, and n) densities"""

        Temperatures = namedtuple('Temperatures', 'i e')
        """A namedtuple of ion and electron (i and e) temperatures"""

        NeutralsTemperatures = namedtuple('NeutralsTemperatures', 's t')
        """A namedtuple of neutrals (s and t) temperatures"""

        Geometry = namedtuple('Geometry', 'area perim')
        """A namedtuple of cell geometry parameters (area and perimeter)"""

        SigmaV = namedtuple('SigmaV', 'ion rec cx_s cx_t el_s el_t eln_s eln_t')
        """A namedtuple containing sigma*V values"""

        MFP = namedtuple('MFP', 's t')
        """A namedtuple containing MFP values for slow (s) and thermal (t) neutrals"""

        CI = namedtuple('CI', 's t')
        """An unknown named tuple!"""

        Xi = namedtuple('Xi', 's t')
        """A namedtuple containing xi values for slow (s) and thermal (t) neutrals"""

        P0i = namedtuple('P0i', 's t')
        """A namedtuple containing P0i values for slow (s) and thermal (t) neutrals"""

        Pi = namedtuple('Pi', 's t')
        """A namedtuple containing Pi values for slow (s) and thermal (t) neutrals"""

        Cell = namedtuple('Cell', 'n T Tn area perim sv mfp ci Xi P0i Pi')
        """A named tuple containing information for cells"""

        Face = namedtuple('Face', 'lside lfrac adj alb f_abs refl Tn_in s_ext sv ci mfp')
        """A namedtuple containing information for faces"""

        FaceGeometry = namedtuple('FaceGeometry', 'lside lfrac')
        """A namedtuple containing information for the geometry of faces"""


        cell_n = Densities(self.ionDens[:self.nCells],
                           self.elecDens[:self.nCells],
                           np.zeros(self.nCells))

        # initialize cell ion and electron temperatures

        cell_T = Temperatures(self.ionTemp[:self.nCells],
                              self.elecTemp[:self.nCells])

        # initialize cell neutral temperatures
        cell_Tn = NeutralsTemperatures(self.ionTemp[:self.nCells],
                                       np.full(self.ionTemp[:self.nCells].shape, 0.002))

        # initialize cell areas and perimeters

        geom = self._calc_cell_geom()
        cell_geom = Geometry(geom[0], geom[1])

        # initialize cell electron ionization cross sections

        cell_sv = SigmaV(calc_svione(self.sv, cell_n, cell_T),
                         calc_svrec(self.sv, cell_n, cell_T),
                         calc_svcx(self.sv, cell_T, cell_Tn, 'slow'),
                         calc_svcx(self.sv, cell_T, cell_Tn, 'thermal'),
                         calc_svel(self.sv, cell_T, cell_Tn, 'slow'),
                         calc_svel(self.sv, cell_T, cell_Tn, 'thermal'),
                         calc_sveln(self.sv, cell_Tn, 'slow'),
                         calc_sveln(self.sv, cell_Tn, 'thermal'))

        # initialize cell mfp values
        cell_mfp = MFP(calc_mfp(cell_Tn, cell_n, cell_sv, 'slow'),
                       calc_mfp(cell_Tn, cell_n, cell_sv, 'thermal'))

        # initialize cell c_i values

        cell_ci = CI(calc_c_i(cell_n, cell_sv, 'slow'),
                     calc_c_i(cell_n, cell_sv, 'thermal'))

        # initialize cell X_i values

        cell_Xi = Xi(calc_X_i(cell_geom, cell_mfp, 'slow'),
                     calc_X_i(cell_geom, cell_mfp, 'thermal'))


        # initialize cell P_0i values

        cell_P0i = P0i(calc_P_0i(cell_Xi, 'slow'),
                       calc_P_0i(cell_Xi, 'thermal'))

        # initialize cell P_i values
        cell_Pi = Pi(calc_P_i(cell_n, cell_sv, cell_P0i, 'slow'),
                     calc_P_i(cell_n, cell_sv, cell_P0i, 'thermal'))


        # combine into 'cell' dictionary and namedtuple

        cell = Cell(cell_n,
                    cell_T,
                    cell_Tn,
                    cell_geom.area,
                    cell_geom.perim,
                    cell_sv,
                    cell_mfp,
                    cell_ci,
                    cell_Xi,
                    cell_P0i,
                    cell_Pi)

        #########################################################################

        # initialize face geometry and adjacent cell parameters
        face_geom = FaceGeometry(self.lsides[:self.nCells],
                                 self.lsides[:self.nCells] / np.sum(self.lsides[:self.nCells], axis=1).reshape((-1, 1)))



        face_adj = self._calc_adj_cell_prop()  # instance method that already has everything it needs

        # initialize neutral temperatures for neutrals entering the cell
        face_alb, face_refl, face_f_abs = calc_refl_alb(cell_T, face_adj)

        face_Tn_in = NeutralsTemperatures(np.full(face_adj.int_type.shape, 0.002),
                                          calc_Tn_intocell_t(face_adj, cell_T, face_refl))

        face_s_ext = calc_ext_src(face_adj, self.s_ext)
        face_sv = SigmaV(cell_sv.ion,
                         cell_sv.rec,
                         calc_svcx(self.sv, cell_T, face_Tn_in, 'slow'),
                         calc_svcx(self.sv, cell_T, face_Tn_in, 'thermal'),
                         calc_svel(self.sv, cell_T, face_Tn_in, 'slow'),
                         calc_svel(self.sv, cell_T, face_Tn_in, 'thermal'),
                         calc_sveln(self.sv, face_Tn_in, 'slow'),
                         calc_sveln(self.sv, face_Tn_in, 'thermal'))

        face_ci = CI(calc_c_i(cell_n, face_sv, 'slow'),
                     calc_c_i(cell_n, face_sv, 'thermal'))

        face_mfp = MFP(calc_mfp(face_Tn_in, cell_n, face_sv, 'slow'),
                       calc_mfp(face_Tn_in, cell_n, face_sv, 'thermal'))

        # combine into a face dictionary and namedtuple

        self.face = Face(face_geom.lside,
                         face_geom.lfrac,
                         face_adj,
                         face_alb,
                         face_f_abs,
                         face_refl,
                         face_Tn_in,
                         face_s_ext,
                         face_sv,
                         face_ci,
                         face_mfp)

        # compute transmission coefficients
        self.T_coef = self._calc_tcoefs(self.face, int_method='quad', cpu_cores=self.cpu_cores)

        # construct and solve the matrix to obtain the fluxes
        self.flux = self._solve_matrix(self.face, cell, self.T_coef)

        # compute ionization rates and densities
        self.izn_rate, self.nn = self._calc_neutral_dens(cell, self.face, self.T_coef, self.flux)

        time1 = time.time()
        minutes, seconds = divmod(time1 - time0, 60)
        if self.verbose: print 'NEUTPY TIME = {} min, {} sec'.format(minutes, seconds)
        self.nn_s_raw = self.cell_nn_s
        self.nn_t_raw = self.cell_nn_t
        self.nn_raw = self.nn_s_raw + self.nn_t_raw

        self.iznrate_s_raw = self.cell_izn_rate_s
        self.iznrate_t_raw = self.cell_izn_rate_t
        self.iznrate_raw = self.iznrate_s_raw + self.iznrate_t_raw

    def gen_outfile(self, filename):
        """
        Write a Neutpy output file.

        This function takes as input a filename and prints out the NeutPy values. The file
        contains R,Z coordinates and the values of several calculated parameters at each of these points.

        :param filename: The filename for the output
        :type filename: str
        """
        f = open(filename, 'w')
        f.write(('{:^18s}' * 8).format('R', 'Z', 'n_n_slow', 'n_n_thermal', 'n_n_total', 'izn_rate_slow',
                                       'izn_rate_thermal', 'izn_rate_total'))
        for i, pt in enumerate(self.midpts):
            f.write(('\n' + '{:>18.5f}' * 2 + '{:>18.5E}' * 6).format(
                self.midpts[i, 0],
                self.midpts[i, 1],
                self.nn_s_raw[i],
                self.nn_t_raw[i],
                self.nn_raw[i],
                self.iznrate_s_raw[i],
                self.iznrate_t_raw[i],
                self.iznrate_raw[i]))
        f.close()

    def _calc_cell_geom(self):
        cell_area = np.zeros(self.nCells)
        cell_perim = np.zeros(self.nCells)

        for i in range(0, self.nCells):
            L_sides = self.lsides[i, :self.nSides[i]]

            angles2 = self.angles[i, :self.nSides[i]] * 2 * pi / 360  # in radians
            theta = np.zeros(angles2.shape)
            for j in range(0, int(self.nSides[i])):
                if j == 0:
                    theta[j] = 0.0
                elif j == 1:
                    theta[j] = angles2[0]
                else:
                    theta[j] = theta[j - 1] + angles2[j - 1] - pi

            x_comp = L_sides * np.cos(theta)
            y_comp = L_sides * np.sin(theta)

            x_comp[0] = 0
            y_comp[0] = 0

            xs = np.cumsum(x_comp)
            ys = np.cumsum(y_comp)

            # calculate cell area and perimeter
            cell_area[i] = 1.0 / 2.0 * abs(np.sum(xs * np.roll(ys, -1) - ys * np.roll(xs, -1)))
            cell_perim[i] = np.sum(L_sides)
        return cell_area, cell_perim

    def _calc_adj_cell_prop(self):
        """
        Determines the type of interface for each face of a cell
        :param adjCells:
        :param iType:
        :return:
        """

        face_adjcell = self.adjCell[:self.nCells]

        face_int_type = np.zeros(face_adjcell.shape)
        face_awall = np.zeros(face_adjcell.shape)
        face_zwall = np.zeros(face_adjcell.shape)
        face_twall = np.zeros(face_adjcell.shape)
        face_f_abs = np.zeros(face_adjcell.shape)
        face_s_ext = np.zeros(face_adjcell.shape)
        for (i, j), val in np.ndenumerate(face_adjcell):
            if val != -1:
                face_int_type[i, j] = self.iType[val]
                face_awall[i, j] = self.awall[val]
                face_zwall[i, j] = self.zwall[val]
                face_twall[i, j] = self.twall[val]
                face_f_abs[i, j] = self.f_abs[val]
                face_s_ext[i, j] = self.s_ext[val]
        FaceAdj = namedtuple('FaceAdj', 'cellnum int_type awall zwall f_abs s_ext')

        face_adj = FaceAdj(face_adjcell,
                           face_int_type,
                           face_awall,
                           face_zwall,
                           face_f_abs,
                           face_s_ext)

        return face_adj

    def _calc_tcoefs(self, face, int_method='quad', cpu_cores=1):
        # create bickley-naylor fit (much faster than evaluating Ki3 over and over)
        Ki3_x = np.linspace(0, 100, 200)
        Ki3 = np.zeros(Ki3_x.shape)
        for i, x in enumerate(Ki3_x):
            Ki3[i] = calc_Ki3(x)
        Ki3_fit = interp1d(Ki3_x, Ki3)

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
            if isclose(x2, x1):  # then line is vertical
                x_int = x1
                y_int = tan(phi) * x_int
            else:
                # eq of the intersecting line is y= tan(phi)x ( + 0 because of coordinate system choice)
                # set two equations equal and solve for x, then solve for y
                x_int = ((y2 - y1) / (x2 - x1) * x1 - y1) / ((y2 - y1) / (x2 - x1) - tan(phi))
                y_int = tan(phi) * x_int

            return sqrt(x_int ** 2 + y_int ** 2)

        # arrays to be filled in and returned
        T_coef_s = np.zeros((self.nCells, 4, 4), dtype='float')
        T_coef_t = np.zeros((self.nCells, 4, 4), dtype='float')
        T_from = np.zeros((self.nCells, 4, 4), dtype='int')
        T_to = np.zeros((self.nCells, 4, 4), dtype='int')
        T_via = np.zeros((self.nCells, 4, 4), dtype='int')


        outof = np.sum(self.nSides[:self.nCells] ** 2)

        start_time = time.time()
        print "Start time: %s" % start_time

        cord_list = list(np.ndenumerate(T_coef_s))
        # for (i, j, k), val in np.ndenumerate(T_coef_s):
        # nSides, adjCell, lsides, T_from, T_to, T_via, int_method, T_coef_s, T_coef_t, midpoint2D, face_mfp_t, face_mfp_s, print_progress, outof, selfAngles
        kwargs = {'nSides': self.nSides,
                  'adjCell': self.adjCell,
                  'lsides': self.lsides,
                  'T_from': T_from,
                  'T_to': T_to,
                  'T_via': T_via,
                  'int_method': int_method,
                  'T_coef_s': T_coef_s,
                  'T_coef_t': T_coef_t,
                  'face_mfp_t': face.mfp.t,
                  'face_mfp_s': face.mfp.s,
                  'print_progress': None,
                  'outof': outof,
                  'angles': self.angles,
                  'Ki3_fit': Ki3_fit,
                  'li': li,
                  }

        # Use all but one CPU by default
        if cpu_cores > cpu_count():
            pool = Pool(cpu_count() - 1)
            print "T_coef calculation running on %s cores." % cpu_count() - 1
        else:
            pool = Pool(cpu_cores)
            print "T_coef calculation running on %s cores." % cpu_cores

        self.coef_results = pool.map(partial(coeff_calc, **kwargs), cord_list)

        # result = [(0, 0, 0, 0, 0)] * (self.nCells * 4 * 4)

        end_time = time.time()
        print "Total: %s" % (end_time - start_time)

        for (i, j, k, s, t, f, to, via) in self.coef_results:
            T_from[i, j, k] = f
            T_to[i, j, k] = to
            T_via[i, j, k] = via
            T_coef_s[i, j, k] = s
            T_coef_t[i, j, k] = t


        # create t_coef_sum arrays for use later
        tcoef_sum_s = np.zeros((self.nCells, 4))
        tcoef_sum_t = np.zeros((self.nCells, 4))
        for i in range(0, self.nCells):
            for k in range(0, int(self.nSides[i])):
                tcoef_sum_s[i, k] = np.sum(T_coef_s[np.where((T_via == i) & (T_from == self.adjCell[i, k]))])
                tcoef_sum_t[i, k] = np.sum(T_coef_t[np.where((T_via == i) & (T_from == self.adjCell[i, k]))])

        TransmissionCoefficients = namedtuple('TransmissionCoefficients', 's t from_cell to_cell via_cell sum_s sum_t')
        T_coef = TransmissionCoefficients(T_coef_s,
                                          T_coef_t,
                                          T_from,
                                          T_to,
                                          T_via,
                                          tcoef_sum_s,
                                          tcoef_sum_t)

        return T_coef

    def _solve_matrix(self, face, cell, T_coef):
        # calculate size of matrix and initialize

        num_fluxes = int(np.sum(self.nSides[:self.nCells]))
        num_en_groups = 2
        M_size = int(num_fluxes * num_en_groups)
        M_matrix = np.zeros((M_size, M_size))

        # set the order of the fluxes in the solution vector
        flux_pos = np.zeros((M_size, 3), dtype=int)
        pos_count = 0
        for g in range(0, num_en_groups):
            for i in range(0, self.nCells):
                for j in range(0, (self.nSides[i])):
                    flux_pos[pos_count, 0] = g
                    flux_pos[pos_count, 1] = i
                    flux_pos[pos_count, 2] = self.adjCell[i, j]
                    pos_count += 1

        m_sparse = 1

        if m_sparse == 1 or m_sparse == 2:
            # construct sparse matrix using coordinate list (COO) method
            m_row = []
            m_col = []
            m_data = []

        # M_row = 0
        for i in range(0, self.nCells):
            for j in range(0, int(self.nSides[i])):
                for k in range(0, int(self.nSides[i])):

                    curcell = int(i)
                    tocell = int(self.adjCell[i, j])
                    fromcell = int(self.adjCell[i, k])

                    # curcell_type = int(inp.iType[curcell])
                    # tocell_type = int(inp.iType[tocell])
                    fromcell_type = int(self.iType[fromcell])

                    T_coef_loc = np.where((T_coef.via_cell == curcell) &
                                          (T_coef.from_cell == fromcell) &
                                          (T_coef.to_cell == tocell))

                    face_loc = T_coef_loc[:2]

                    # FROM NORMAL CELL
                    if fromcell_type == 0:  # if fromcell is normal cell
                        # determine row and column of matrix

                        M_row_s = np.where((flux_pos[:, 1] == curcell) & (flux_pos[:, 2] == tocell))[0][0]
                        M_col_s = np.where((flux_pos[:, 1] == fromcell) & (flux_pos[:, 2] == curcell))[0][0]
                        M_row_t = M_row_s + num_fluxes
                        M_col_t = M_col_s + num_fluxes

                        if fromcell == tocell:
                            uncoll_ss = 0
                            uncoll_tt = 0
                        else:
                            uncoll_ss = T_coef.s[T_coef_loc]
                            uncoll_tt = T_coef.t[T_coef_loc]

                        coll_ss = 0
                        coll_tt = (1 - T_coef.sum_t[i, k]) * face.ci.t[i, k] * \
                                  (cell.P0i.t[i] * face.lfrac[i, j] + (1 - cell.P0i.t[i]) * cell.ci.t[i] * cell.Pi.t[
                                      i] * face.lfrac[i, j])
                        coll_ts = 0
                        coll_st = (1 - T_coef.sum_s[i, k]) * face.ci.s[i, k] * \
                                  (cell.P0i.t[i] * face.lfrac[i, j] + (1 - cell.P0i.t[i]) * cell.ci.t[i] * cell.Pi.t[
                                      i] * face.lfrac[i, j])
                        # SLOW GROUP EQUATIONS
                        if m_sparse == 0 or m_sparse == 2:
                            # matrix: from slow group into slow group
                            M_matrix[M_row_s, M_col_s] = uncoll_ss + coll_ss
                            # matrix: from thermal group into slow group
                            M_matrix[M_row_s, M_col_t] = coll_ts
                            # matrix: from slow group into thermal group
                            M_matrix[M_row_t, M_col_s] = coll_st
                            # matrix: from thermal group into thermal group
                            M_matrix[M_row_t, M_col_t] = uncoll_tt + coll_tt

                        if m_sparse == 1 or m_sparse == 2:
                            # matrix: from slow group into slow group
                            m_row.append(M_row_s)
                            m_col.append(M_col_s)
                            m_data.append(listToFloatChecker(uncoll_ss + coll_ss,
                                                             "Multi-dimensional m_data entry for uncoll_ss + coll_ss at row_s %s, col_s %s" % (
                                                                 M_row_s, M_col_s), verbose=self.verbose))

                            # matrix: from thermal group into slow group
                            m_row.append(M_row_s)
                            m_col.append(M_col_t)
                            m_data.append(listToFloatChecker(coll_ts,
                                                             "Multi-dimensional m_data entry for coll_ts at row_s %s, col_t %s" % (
                                                                 M_row_s, M_col_t), verbose=self.verbose))

                            # matrix: from slow group into thermal group
                            m_row.append(M_row_t)
                            m_col.append(M_col_s)
                            m_data.append(listToFloatChecker(coll_st,
                                                             "Multi-dimensional m_data entry for coll_st at row_t %s, col_s %s" % (
                                                                 M_row_t, M_col_s), verbose=self.verbose))

                            # matrix: from thermal group into thermal group
                            m_row.append(M_row_t)
                            m_col.append(M_col_t)
                            m_data.append(listToFloatChecker(uncoll_tt + coll_tt,
                                                             "Multi-dimensional m_data entry for uncoll_tt + coll_tt at row_t %s, col_t %s" % (
                                                                 M_row_t, M_col_t), verbose=self.verbose))

                    # FROM PLASMA CORE
                    elif fromcell_type == 1:  # if fromcell is plasma core cell
                        # column from, to swapped because incoming is defined as a function of outgoing via albedo condition
                        M_row_s = np.where((flux_pos[:, 1] == curcell) & (flux_pos[:, 2] == tocell))[0][0]
                        M_col_s = np.where((flux_pos[:, 1] == curcell) & (flux_pos[:, 2] == fromcell))[0][0]
                        M_row_t = M_row_s + num_fluxes
                        M_col_t = M_col_s + num_fluxes

                        # uncollided flux from slow group to slow group
                        if fromcell == tocell:
                            uncoll_ss = 0.0
                            uncoll_tt = 0.0
                        else:
                            uncoll_ss = face.alb.s[face_loc] * T_coef.s[T_coef_loc]
                            uncoll_tt = face.alb.t[face_loc] * T_coef.t[T_coef_loc]

                        coll_ss = 0
                        coll_tt = 0
                        coll_ts = 0
                        coll_st = 0

                        if m_sparse == 0 or m_sparse == 2:
                            # matrix: from slow group into slow group
                            M_matrix[M_row_s, M_col_s] = uncoll_ss + coll_ss
                            # matrix: from thermal group into slow group
                            M_matrix[M_row_s, M_col_t] = coll_ts
                            # matrix: from slow group into thermal group
                            M_matrix[M_row_t, M_col_s] = coll_st
                            # matrix: from thermal group into thermal group
                            M_matrix[M_row_t, M_col_t] = uncoll_tt + coll_tt

                        if m_sparse == 1 or m_sparse == 2:
                            # matrix: from slow group into slow group
                            m_row.append(M_row_s)
                            m_col.append(M_col_s)
                            m_data.append(listToFloatChecker(uncoll_ss + coll_ss,
                                                             "Multi-dimensional m_data entry for uncoll_ss + coll_ss at row_s %s, col_s %s" % (
                                                                 M_row_s, M_col_s), verbose=self.verbose))

                            # matrix: from thermal group into slow group
                            m_row.append(M_row_s)
                            m_col.append(M_col_t)
                            m_data.append(listToFloatChecker(coll_ts,
                                                             "Multi-dimensional m_data entry for coll_ts at row_s %s, col_t %s" % (
                                                                 M_row_s, M_col_t), verbose=self.verbose))

                            # matrix: from slow group into thermal group
                            m_row.append(M_row_t)
                            m_col.append(M_col_s)
                            m_data.append(listToFloatChecker(coll_st,
                                                             "Multi-dimensional m_data entry for coll_st at row_t %s, col_s %s" % (
                                                                 M_row_t, M_col_s), verbose=self.verbose))

                            # matrix: from thermal group into thermal group
                            m_row.append(M_row_t)
                            m_col.append(M_col_t)
                            m_data.append(listToFloatChecker(uncoll_tt + coll_tt,
                                                             "Multi-dimensional m_data entry for uncoll_tt + coll_tt at row_t %s, col_t %s" % (
                                                                 M_row_t, M_col_t), verbose=self.verbose))

                    # FROM WALL
                    elif fromcell_type == 2:  # if fromcell is wall cell
                        # column from, to swapped because incoming is defined as a function of outgoing via reflection coefficient
                        M_row_s = np.where((flux_pos[:, 1] == curcell) & (flux_pos[:, 2] == tocell))[0][0]
                        M_col_s = np.where((flux_pos[:, 1] == curcell) & (flux_pos[:, 2] == fromcell))[0][0]
                        M_row_t = M_row_s + num_fluxes
                        M_col_t = M_col_s + num_fluxes

                        # UNCOLLIDED FLUX
                        if fromcell == tocell:
                            # a particle going back to the cell it came from implies a collision
                            uncoll_refl_ss = 0
                            uncoll_refl_tt = 0
                            uncoll_refl_ts = 0
                            uncoll_refl_st = 0

                            uncoll_reem_ss = 0
                            uncoll_reem_tt = 0
                            uncoll_reem_ts = 0
                            uncoll_reem_st = 0
                        else:
                            # slow neutrals can hit the wall, reflect as slow, and stream uncollided into adjacent cells
                            uncoll_refl_ss = face.refl.n.s[face_loc] * T_coef.s[T_coef_loc]
                            # thermal neutrals can hit the wall, reflect as thermals, and stream uncollided into adjacent cells
                            uncoll_refl_tt = face.refl.n.t[face_loc] * T_coef.t[T_coef_loc]
                            # thermal neutrals can reflect back as slow and then stay slow , but we'll treat this later
                            uncoll_refl_ts = 0
                            # slow neutrals can reflect back as slow neutrals, and end up as thermal, but that implies a collision
                            uncoll_refl_st = 0

                            # slow neutrals can hit the wall, be reemitted as slow, and stream uncollided into adjacent cells
                            uncoll_reem_ss = (1.0 - face.refl.n.s[face_loc]) * (1 - face.f_abs[face_loc]) * T_coef.s[
                                T_coef_loc]
                            # thermal neutrals can hit the wall, be reemitted as slow, and then end up thermal again, but that implies a collision.
                            uncoll_reem_tt = 0
                            # thermal neutrals can hit the wall, be reemitted as slow, and then stream uncollided into adjacent cells
                            uncoll_reem_ts = (1.0 - face.refl.n.t[face_loc]) * (1 - face.f_abs[face_loc]) * T_coef.s[
                                T_coef_loc]
                            # slow neutrals can hit the wall, be reemitted as slow, and then end up thermal again, but that implies a collision.
                            uncoll_reem_st = 0

                            # COLLIDED FLUX
                        # slow neutrals can hit the wall, reflect as slow neutrals, but a collision removes them from the slow group
                        coll_refl_ss = 0
                        # thermal neutrals can hit the wall, reflect as thermal neutrals, and then have a collision and stay thermal afterward
                        coll_refl_tt = face.refl.n.t[face_loc] * (1 - T_coef.sum_t[i, k]) * \
                                       face.ci.t[i, k] * (
                                               cell.P0i.t[i] * face.lfrac[i, j] + (1 - cell.P0i.t[i]) * cell.ci.t[
                                           i] * cell.Pi.t[i] * face.lfrac[i, j])
                        # thermal neutrals can hit the wall, reflect as thermal neutrals, but they won't reenter the slow group
                        coll_refl_ts = 0
                        # slow neutrals can hit the wall, reflect as slow neutrals, and have a collision to enter and stay in the thermal group

                        coll_refl_st = face.refl.n.s[face_loc] * (1 - T_coef.sum_s[i, k]) * \
                                       face.ci.s[i, k] * (
                                               cell.P0i.t[i] * face.lfrac[i, j] + (1 - cell.P0i.t[i]) * cell.ci.t[
                                           i] * cell.Pi.t[i] * face.lfrac[i, j])

                        # slow neutrals can hit the wall, be reemitted as slow, but a collision removes them from the slow group
                        coll_reem_ss = 0

                        # thermal neutrals can hit the wall, be reemitted as slow, and then collide to enter and stay in the thermal group
                        coll_reem_tt = (1 - face.refl.n.t[face_loc]) * (1 - face.f_abs[face_loc]) * (
                                1 - T_coef.sum_s[i, k]) * \
                                       face.ci.s[i, k] * (
                                               cell.P0i.t[i] * face.lfrac[i, j] + (1 - cell.P0i.t[i]) * cell.ci.t[
                                           i] * cell.Pi.t[i] * face.lfrac[i, j])
                        # thermal neutrals can hit the wall, be reemitted as slow, but a collision removes them from the slow group
                        coll_reem_ts = 0

                        # slow neutrals can hit the wall, be reemitted as slow, and then collide to enter and stay in the thermal group
                        coll_reem_st = (1 - face.refl.n.s[face_loc]) * (1 - face.f_abs[face_loc]) * (
                                1 - T_coef.sum_s[i, k]) * \
                                       face.ci.s[i, k] * (
                                               cell.P0i.t[i] * face.lfrac[i, j] + (1 - cell.P0i.t[i]) * cell.ci.t[
                                           i] * cell.Pi.t[i] * face.lfrac[i, j])

                        if m_sparse == 0 or m_sparse == 2:
                            # matrix: from slow group into slow group
                            M_matrix[M_row_s, M_col_s] = uncoll_refl_ss + uncoll_reem_ss + coll_refl_ss + coll_reem_ss
                            # matrix: from thermal group into slow group
                            M_matrix[M_row_s, M_col_t] = uncoll_refl_ts + uncoll_reem_ts + coll_refl_ts + coll_reem_ts
                            # matrix: from slow group into thermal group
                            M_matrix[M_row_t, M_col_s] = uncoll_refl_st + uncoll_reem_st + coll_refl_st + coll_reem_st
                            # matrix: from thermal group into thermal group
                            M_matrix[M_row_t, M_col_t] = uncoll_refl_tt + uncoll_reem_tt + coll_refl_tt + coll_reem_tt
                        if m_sparse == 1 or m_sparse == 2:
                            # matrix: from slow group into slow group
                            m_row.append(M_row_s)
                            m_col.append(M_col_s)
                            m_data.append(
                                listToFloatChecker(uncoll_refl_ss + uncoll_reem_ss + coll_refl_ss + coll_reem_ss,
                                                   "Multi-dimensional m_data entry for uncoll_refl_ss + uncoll_reem_ss + coll_refl_ss + coll_reem_ss"
                                                   " at row_s %s, col_s %s" % (M_row_s, M_col_s), verbose=self.verbose))

                            # matrix: from thermal group into slow group
                            m_row.append(M_row_s)
                            m_col.append(M_col_t)
                            m_data.append(
                                listToFloatChecker(uncoll_refl_ts + uncoll_reem_ts + coll_refl_ts + coll_reem_ts,
                                                   "Multi-dimensional m_data entry for uncoll_refl_ts + uncoll_reem_ts + coll_refl_ts + coll_reem_ts"
                                                   " at row_s %s, col_t %s" % (M_row_s, M_col_t), verbose=self.verbose))

                            # matrix: from slow group into thermal group
                            m_row.append(M_row_t)
                            m_col.append(M_col_s)
                            m_data.append(
                                listToFloatChecker(uncoll_refl_st + uncoll_reem_st + coll_refl_st + coll_reem_st,
                                                   "Multi-dimensional m_data entry for uncoll_refl_st + uncoll_reem_st + coll_refl_st + coll_reem_st"
                                                   " at row_t %s, col_s %s" % (M_row_t, M_col_s), verbose=self.verbose))

                            # matrix: from thermal group into thermal group
                            m_row.append(M_row_t)
                            m_col.append(M_col_t)
                            m_data.append(
                                listToFloatChecker(uncoll_refl_tt + uncoll_reem_tt + coll_refl_tt + coll_reem_tt,
                                                   "Multi-dimensional m_data entry for uncoll_refl_tt + uncoll_reem_tt + coll_refl_tt + coll_reem_tt"
                                                   " at row_t %s, col_t %s" % (M_row_t, M_col_t), verbose=self.verbose))

        # create source vector

        # account for external sources
        flux_cells = self.adjCell[np.where(face.s_ext > 0)]
        flux_vals = face.s_ext[np.where(face.s_ext > 0)]

        source = np.zeros(M_size)
        for i, v in enumerate(flux_pos):
            group = v[0]
            cell_io = v[1]
            cell_to = v[2]
            if group == 0:
                # ADD CONTRIBUTION FROM VOLUMETRIC SOURCE (I.E. RECOMBINATION)
                source[i] = source[i] + 0  # assumed that all neutrals from recombination are thermal

                # ADD CONTRIBUTION FROM EXTERNAL SOURCES (all external sources are assumed to be slow for now)
                # if our cell has a wall that has an external source
                # TODO: I'm not confident that that this calculation is quite right. In the event that there are
                # multiple sides with external fluxes, it should loop over them and treat each of them individually.
                # this essentially requires an improved treatment of "cell_from"
                if np.any(np.in1d(flux_cells, self.adjCell[cell_io])):
                    cell_from = list(set(flux_cells) & set(self.adjCell[cell_io]))[0]
                    incoming_flux = flux_vals[np.where(flux_cells == cell_from)][0]

                    T_coef_loc = np.where((T_coef.via_cell == cell_io) &
                                          (T_coef.from_cell == cell_from) &
                                          (T_coef.to_cell == cell_to))
                    face_from_loc = T_coef_loc[:2]
                    face_to_loc = [T_coef_loc[0], T_coef_loc[-1]]

                    # add uncollided slow flux from slow external source
                    source[i] = source[i] + incoming_flux * T_coef.s[T_coef_loc][0]
                    # add collided slow flux from slow external source
                    source[i] = source[i] + incoming_flux * 0
                    # add collided thermal flux from slow external source
                    source[i + num_fluxes] = source[i + num_fluxes] + incoming_flux * (
                            1 - T_coef.sum_s[face_from_loc]) * \
                                             face.ci.s[face_from_loc] * (
                                                     cell.P0i.t[cell_io] * face.lfrac[face_to_loc] + (
                                                     1 - cell.P0i.t[cell_io]) * cell.ci.t[cell_io] * cell.Pi.t[
                                                         cell_io] * face.lfrac[face_to_loc])

            if group == 1:
                # ADD CONTRIBUTION FROM VOLUMETRIC SOURCE (I.E. RECOMBINATION)
                source[i] = source[
                                i] + 0  # cell_area[cell_io]*cell_ni[cell_io]*cell_sv_rec[cell_io]*P_i_t[cell_io]*0.25 # assumed that all neutrals from recombination are thermal

                # ADD CONTRIBUTION FROM IMPINGING IONS REFLECTING AS THERMAL NEUTRALS
                # loop over "from cells and determine if any are wall cells
                for count, cell_from in enumerate(self.adjCell[cell_io, :self.nSides[cell_io]]):
                    # if it's a wall cell
                    if self.iType[cell_from] == 2:

                        # calculate flux to wall segment
                        ni = cell.n.i[cell_io]
                        Ti = cell.T.i[cell_io] * 1.0E3 * 1.6021E-19
                        vel = sqrt(2.0 * Ti / 3.343583719E-27)
                        wall_length = face.lside[cell_io, count]
                        R0_wall = 1.4  # TODO: Read in R values for each wall segment
                        flux2wall = 2.0 * pi * ni * vel * wall_length * R0_wall

                        # calculate returning neutral flux
                        refl_coef = face.refl.n.t[cell_io, count]
                        incoming_flux = flux2wall * refl_coef

                        # calculate uncollided source to cell_to
                        T_coef_loc = np.where((T_coef.via_cell == cell_io) & (T_coef.from_cell == cell_from) & (
                                T_coef.to_cell == cell_to))
                        face_from_loc = T_coef_loc[:2]
                        face_to_loc = [T_coef_loc[0], T_coef_loc[-1]]

                        source[i] = source[i] + incoming_flux * T_coef.t[T_coef_loc]
                        #TODO: Check deprecation warnings for this.
                        # calculate collided source to cell_to
                        source[i] = source[i] + incoming_flux * (1.0 - T_coef.sum_t[face_from_loc]) * \
                                    face.ci.t[face_from_loc] * (cell.P0i.t[cell_io] * face.lfrac[face_to_loc] + (
                                1.0 - cell.P0i.t[cell_io]) * cell.ci.t[cell_io] * cell.Pi.t[cell_io] * face.lfrac[
                                                                    face_to_loc])
                        if incoming_flux < 0:
                            print 'incoming flux less than zero'
                            print 'stopping'
                            sys.exit()

        # CREATE FINAL MATRIX AND SOLVE
        if m_sparse == 0 or m_sparse == 2:
            M_matrix = np.identity(M_size) - M_matrix
            flux_out = spsolve(M_matrix, source)
        if m_sparse == 1 or m_sparse == 2:
            # multiply m_data by -1 and append "identity matrix"
            # note: we're taking advantage of the way coo_matrix handles duplicate
            # entries to achieve the same thing as I - M
            m_row = np.concatenate((np.asarray(m_row), np.arange(0, M_size)))
            m_col = np.concatenate((np.asarray(m_col), np.arange(0, M_size)))
            m_data = np.concatenate((-np.asarray(m_data), np.ones(M_size)))
            m_sp_final = coo_matrix((m_data, (m_row, m_col))).tocsc()
            flux_out = spsolve(m_sp_final, source)

        # create outgoing flux arrays, dictionary, and namedtuple
        flux_out_s = np.zeros((self.nCells, 4))
        flux_out_t = np.zeros((self.nCells, 4))
        flux_counter = 0
        for g, g1 in enumerate(range(num_en_groups)):
            for i, v1 in enumerate(range(self.nCells)):
                for j, v2 in enumerate(range(self.nSides[i])):
                    if g == 0:
                        flux_out_s[i, j] = flux_out[flux_counter]
                    if g == 1:
                        flux_out_t[i, j] = flux_out[flux_counter]
                    flux_counter += 1

        Flux = namedtuple('Flux', 'inc out')
        DirectedFlux = namedtuple('DirectedFlux', 's t tot')

        flux_out = DirectedFlux(flux_out_s,
                                flux_out_t,
                                flux_out_s + flux_out_t)


        # create incoming flux arrays, dictionary, and namedtuple
        flux_inc_s = np.zeros(flux_out_s.shape)
        flux_inc_t = np.zeros(flux_out_t.shape)

        flux_out_dest = self.adjCell[:self.nCells, :]
        flux_out_src = flux_out_dest * 0
        for i in range(self.nCells):
            flux_out_src[i] = i

        for i in range(0, self.nCells):
            for k in range(0, self.nSides[i]):
                if face.adj.int_type[i, k] == 0:
                    flux_inc_s[i, k] = flux_out_s[np.where((flux_out_dest == i) & (flux_out_src == self.adjCell[i, k]))]
                    flux_inc_t[i, k] = flux_out_t[np.where((flux_out_dest == i) & (flux_out_src == self.adjCell[i, k]))]
                if face.adj.int_type[i, k] == 1:
                    flux_inc_s[i, k] = flux_out_s[i, k] * face.alb.s[i, k]
                    flux_inc_t[i, k] = flux_out_t[i, k] * face.alb.t[i, k]
                    flux_inc_s[i, k] = flux_inc_s[i, k] + face.s_ext[i, k]
                if face.adj.int_type[i, k] == 2:
                    flux_inc_s[i, k] = flux_out_s[i, k] * (
                            face.refl.n.s[i, k] + (1.0 - face.refl.n.s[i, k]) * (1 - face.f_abs[i, k])) + \
                                       flux_out_t[i, k] * (1.0 - face.refl.n.t[i, k]) * (1 - face.f_abs[i, k])
                    flux_inc_t[i, k] = flux_out_t[i, k] * face.refl.n.t[i, k]
                    flux_inc_s[i, k] = flux_inc_s[i, k] + face.s_ext[i, k]

        flux_inc = DirectedFlux(flux_inc_s,
                                flux_inc_t,
                                flux_inc_s + flux_inc_t)

        flux = Flux(flux_inc,
                    flux_out)


        return flux

    def _calc_neutral_dens(self, cell, face, T_coef, flux):
        """
        Calculates the neutral density
        Args:
            cell:
            face:
            T_coef:
            flux:

        Returns:

        """
        # first calculate the fluxes into cells from the fluxes leaving the cells

        # calculate ionization rate
        self.cell_izn_rate_s = np.zeros(self.nCells)
        self.cell_izn_rate_t = np.zeros(self.nCells)
        for i in range(0, self.nCells):
            # add contribution to ionization rate from fluxes streaming into cell
            for k in range(0, self.nSides[i]):
                self.cell_izn_rate_s[i] = self.cell_izn_rate_s[i] + flux.inc.s[i, k] * (1.0 - T_coef.sum_s[i, k]) * \
                                          ((1.0 - face.ci.s[i, k]) + face.ci.s[i, k] * (1.0 - cell.ci.t[i]) * (
                                                  1.0 - cell.P0i.t[i]) / (1 - cell.ci.t[i] * (1.0 - cell.P0i.t[i])))

                self.cell_izn_rate_t[i] = self.cell_izn_rate_t[i] + flux.inc.t[i, k] * (1.0 - T_coef.sum_t[i, k]) * \
                                          ((1.0 - face.ci.t[i, k]) + face.ci.t[i, k] * (1.0 - cell.ci.t[i]) * (
                                                  1.0 - cell.P0i.t[i]) / (1 - cell.ci.t[i] * (1.0 - cell.P0i.t[i])))

            # add contribution to ionization rate from volumetric recombination within the cell
            self.cell_izn_rate_s[i] = self.cell_izn_rate_s[
                                          i] + 0  # all recombination neutrals are assumed to be thermal
            self.cell_izn_rate_t[i] = self.cell_izn_rate_t[i] + 0 * (1 - cell.P0i.t[i]) * cell.area[i] * cell.n.i[i] * \
                                      cell.sv.rec[i] * \
                                      (1.0 - cell.ci.t[i] + cell.ci.t[i] * (1.0 - cell.ci.t[i]) * (
                                              1 - cell.P0i.t[i]) / (1.0 - cell.ci.t[i] * (cell.P0i.t[i])))

        self.cell_izn_rate = self.cell_izn_rate_s + self.cell_izn_rate_t

        # calculate neutral densities from ionization rates
        self.cell_nn_s = self.cell_izn_rate_s / (cell.n.i * cell.sv.ion * cell.area)
        self.cell_nn_t = self.cell_izn_rate_t / (cell.n.i * cell.sv.ion * cell.area)
        self.cell_nn = self.cell_izn_rate / (cell.n.i * cell.sv.ion * cell.area)

        # # fix negative values (usually won't be necessary)
        for i, v1 in enumerate(self.cell_nn):
            if v1 < 0:
                # get neutral densities of adjacent cells
                # average the positive values
                print i, 'Found a negative density. Fixing by using average of surrounding cells.'
                # print 'You probably need to adjust the way you are calculating the transmission coefficients.'
                nn_sum = 0
                nn_count = 0
                for j, v2 in enumerate(self.adjCell[i]):
                    if self.iType[v2] == 0:
                        if self.cell_nn[j] > 0:
                            nn_sum = nn_sum + self.cell_nn[j]
                            nn_count += 1
                self.cell_nn[i] = nn_sum / nn_count

        IonizationRate = namedtuple('IonizationRate', 's t tot')
        NeutralDensities = namedtuple('NeutralDensities', 's t tot')

        izn_rate = IonizationRate(self.cell_izn_rate_s,
                                  self.cell_izn_rate_t,
                                  self.cell_izn_rate)

        nn = NeutralDensities(self.cell_nn_s,
                              self.cell_nn_t,
                              self.cell_nn)

        return izn_rate, nn

        # print 'Checking particle balance...'
        # relerr = abs(np.sum(flux_out_tot)+np.sum(cell_izn_rate)-np.sum(flux_in_tot))/ \
        # ((np.sum(flux_out_tot)+np.sum(cell_izn_rate)+np.sum(flux_in_tot))/2)
        # if relerr<0.001: # this is arbitrary. It's just here to alert if there's a problem in the particle balance.
        # print 'Particle balance passes. {5.3f}% relative error.'.format(relerr*100)
        # else:
        # print 'Particle balance failed. {}% relative error.'.format(relerr*100)
        # for i, (v1, v2) in enumerate(zip(flux_in_tot, flux_out_tot)):
        # print i, np.sum(v1), np.sum(v2)+cell_izn_rate[i]

    def set_config(self, f):
        """
        Set the configuration file.
        Args:
            f (str): The configuration filename with directory

        Returns:
            neutpy : The neutpy instance
        """
        self.config_loc = f
        return self

    def save_Tcoefs(self, f):
        """
        Save the transmission coefficients to a file.

        :param f: The file to which to save the coefficients
        :type f: str
        """
        trans_coef_file = open(f, 'w')
        trans_coef_file.write(
            ('{:^6s}' * 3 + '{:^12s}' * 4 + '\n').format("from", "to", "via", "T_slow", "T_thermal", "mfp_s", "mfp_t"))

        T_coef_s = np.zeros((self.nCells, 4, 4), dtype='float')
        T_coef_t = np.zeros((self.nCells, 4, 4), dtype='float')
        T_from = np.zeros((self.nCells, 4, 4), dtype='int')
        T_to = np.zeros((self.nCells, 4, 4), dtype='int')
        T_via = np.zeros((self.nCells, 4, 4), dtype='int')

        for (i, j, k, s, t, f, to, via) in self.coef_results:
            T_from[i, j, k] = f
            T_to[i, j, k] = to
            T_via[i, j, k] = via
            T_coef_s[i, j, k] = s
            T_coef_t[i, j, k] = t

        for (i, j, k), val in np.ndenumerate(T_coef_s):
            adj_cells = np.roll(self.adjCell[i, :self.nSides[i]], -j)
            if k < adj_cells.size and j < adj_cells.size:
                if j == k:
                    trans_coef_file.write(
                        ('{:>6d}' * 3 + '{:>12.3E}' * 4 + '\n').format(int(T_from[i, j, k]), int(T_to[i, j, k]),
                                                                       int(T_via[i, j, k]), T_coef_s[i, j, k],
                                                                       T_coef_t[i, j, k], self.face.mfp.s[i, k],
                                                                       self.face.mfp.t[i, k]))
                else:
                    trans_coef_file.write(('{:>6d}' * 3 + '{:>12.3E}' * 4 + '\n').format(int(T_from[i, j, k]),
                                                                                         int(T_to[i, j, k]),
                                                                                         int(T_via[i, j, k]),
                                                                                         T_coef_s[i, j, k],
                                                                                         T_coef_t[i, j, k],
                                                                                         self.face.mfp.s[i, j],
                                                                                         self.face.mfp.t[i, j]))

        trans_coef_file.close()

    def save_flux_outfile(self, filename):
        # create face output data file
        f = open(filename, 'w')
        f.write(('{:^18s}' * 18).format('x1', 'x2', 'x3', 'y1', 'y2', 'y3',
                                        'flxout_s1', 'flxout_s2', 'flxout_s3',
                                        'flxin_s1', 'flxin_s2', 'flxin_s3',
                                        'flxout_t1', 'flxout_t2', 'flxout_t3',
                                        'flxin_t1', 'flxin_t2', 'flxin_t3'))
        for i, pt in enumerate(self.xs):
            f.write(('\n' + '{:>18.5f}' * 6 + '{:>18.5E}' * 12).format(
                self.xs[i, 0], self.xs[i, 1], self.xs[i, 2],
                self.ys[i, 0], self.ys[i, 1], self.ys[i, 2],
                self.flux.out.s[i, 0],
                self.flux.out.s[i, 1],
                self.flux.out.s[i, 2],
                self.flux.inc.s[i, 0],
                self.flux.inc.s[i, 1],
                self.flux.inc.s[i, 2],
                self.flux.out.t[i, 0],
                self.flux.out.t[i, 1],
                self.flux.out.t[i, 2],
                self.flux.inc.t[i, 0],
                self.flux.inc.t[i, 1],
                self.flux.inc.t[i, 2]))
        f.close()

    def plot_cell_lines(self):

        # plot cell diagram
        # grid = plt.figure(figsize=(160,240))
        grid = plt.figure(figsize=(8, 12))
        ax1 = grid.add_subplot(111)
        ax1.set_title('Neutrals Mesh', fontsize=30)
        ax1.set_ylabel(r'Z ($m$)', fontsize=30)
        ax1.set_xlabel(r'R ($m$)', fontsize=30)
        ax1.tick_params(labelsize=15)
        ax1.axis('equal')
        for i, (v1, v2) in enumerate(zip(self.xs, self.ys)):
            # if neut.nSides[i] == len(v1):
            v1 = np.append(v1, v1[0])
            v2 = np.append(v2, v2[0])
            # elif neut.nSides[i] == 3 and len(v1) == 4:
            #    v1[3] = v1[0]
            # elif neut.nSides[i]==3 and len(v1)==3:
            #    v1 = np.append(v1,v1[0])
            #    v2 = np.append(v2,v2[0])
            ax1.plot(v1, v2, color='black', lw=1)

    def _plot_cell_val(self, var, title, nSides=None, logscale=False, cmap='viridis'):
        try:
            import matplotlib.colors as colors
        except ImportError:
            print ImportError("Matplotlib Colors module not found.")
            return
        try:
            from matplotlib.patches import Polygon
        except ImportError:
            print ImportError("Matplotlib Polygon module not found")
            return
        try:
            from matplotlib.collections import PatchCollection
        except ImportError:
            print ImportError("Matplotlib PatchCollection module not found")
            return

        if logscale:
            colors = np.log10(var)
        else:
            colors = var

        patches = []
        for i, v in enumerate(var):
            if nSides is not None:
                verts = np.column_stack((self.xs[i, :nSides[i]], self.ys[i, :nSides[i]]))
            else:
                verts = np.column_stack((self.xs[i, :3], self.ys[i, :3]))

            polygon = Polygon(verts, closed=True)
            patches.append(polygon)

        collection1 = PatchCollection(patches, cmap=cmap)
        collection1.set_array(np.array(colors))

        fig, ax1 = plt.subplots(figsize=(8, 12))
        cax = ax1.add_collection(collection1)
        ax1.axis('equal')
        ax1.set_title(title, fontsize=24)
        ax1.set_ylabel(r'Z ($m$)', fontsize=24)
        ax1.set_xlabel(r'R ($m$)', fontsize=24)
        ax1.tick_params(labelsize=24)
        cb = fig.colorbar(cax)
        cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=24)

    def plot_2D_dens_total(self):
        self._plot_cell_val(self.nn.tot, "Total Neutral Density")

    def plot_2D_dens_therm(self):
        self._plot_cell_val(self.nn.t, "Thermal Neutral Density")

    def plot_2D_dens_slow(self):
        self._plot_cell_val(self.nn.s, "Slow Neutral Density")

if __name__ == "__main__":
    neuts = neutrals()
    neuts.from_file("inputs/144977_3000/toneutpy.conf")
    neuts.plot_dens_slow()
    neuts.plot_dens_thermal()
    neuts.plot_dens_total()
    neuts.plot_cell_lines()
