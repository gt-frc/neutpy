from __future__ import division
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import sys
from math import pi
import os


class NeutpyTools:

    def __init__(self, neut=None):

        # get vertices in R, Z geometry
        self.xs, self.ys = self.calc_cell_pts(neut)

        # localize densities, ionization rates, and a few other parameters that might be needed.
        self.n_n_slow = neut.nn.s
        self.n_n_thermal = neut.nn.t
        self.n_n_total = neut.nn.tot
        self.flux_in_s = neut.flux.inc.s
        self.flux_in_t = neut.flux.inc.t
        self.flux_in_tot = self.flux_in_s + self.flux_in_t
        self.flux_out_s = neut.flux.out.s
        self.flux_out_t = neut.flux.out.t
        self.flux_out_tot = self.flux_out_s + self.flux_out_t

        self.create_flux_outfile()

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

        print 'attempting to start plot_cell_vals'
        self.plot_cell_vals()

    def create_flux_outfile(self):
        # create face output data file
        f = open('neutpy_face_data.dat', 'w')
        f.write(('{:^18s}' * 18).format('x1', 'x2', 'x3', 'y1', 'y2', 'y3',
                                        'flxout_s1', 'flxout_s2', 'flxout_s3',
                                        'flxin_s1', 'flxin_s2', 'flxin_s3',
                                        'flxout_t1', 'flxout_t2', 'flxout_t3',
                                        'flxin_t1', 'flxin_t2', 'flxin_t3'))
        for i, pt in enumerate(self.xs):
            f.write(('\n' + '{:>18.5f}' * 6 + '{:>18.5E}' * 12).format(
                self.xs[i, 0], self.xs[i, 1], self.xs[i, 2],
                self.ys[i, 0], self.ys[i, 1], self.ys[i, 2],
                self.flux_out_s[i, 0],
                self.flux_out_s[i, 1],
                self.flux_out_s[i, 2],
                self.flux_in_s[i, 0],
                self.flux_in_s[i, 1],
                self.flux_in_s[i, 2],
                self.flux_out_t[i, 0],
                self.flux_out_t[i, 1],
                self.flux_out_t[i, 2],
                self.flux_in_t[i, 0],
                self.flux_in_t[i, 1],
                self.flux_in_t[i, 2]))
        f.close()

    @staticmethod
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
                ycoords[curcell, :neut.nSides[curcell]] = np.roll(np.cumsum(y_comp), 1) + neut.cell1_ctr_x

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

    def plot_cell_lines(self, dir=''):

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
        grid.savefig(dir+'neutpy_mesh.png', dpi=300, transparent=True, bbox_inches="tight")

    def plot_cell_vals(self, title='n_n_total', dir=os.getcwd(), nSides=None, logscale=False, cmap='viridis'):
        print 'beginning plot_cell_cals'
        var = self.n_n_total

        if logscale:
            colors = np.log10(var)
        else:
            colors = var

        patches = []
        for i,v in enumerate(var):
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
        ax1.set_title(title, fontsize=30)
        ax1.set_ylabel(r'Z ($m$)', fontsize=30)
        ax1.set_xlabel(r'R ($m$)', fontsize=30)
        ax1.tick_params(labelsize=30)
        cb = fig.colorbar(cax)
        cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=30)
        print 'attempting to show plot'
        plt.show()
        # print 'attempting to save plot in '+dir+'/'+title+'.png'
        # fig.savefig('figure.png', dpi=300, transparent=True, bbox_inches="tight")

    def interp_RZ(self, var):
        x = np.average(xs, axis=1)
        y = np.average(ys, axis=1)
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
