from collections import defaultdict, deque
import numpy as np
from scipy.constants import e, m_p

import sv
from helpers import area_triangle, calc_t_coef


class CellEnergyGroup:
    def __init__(self):
        return


class Interface:
    def __init__(self, adjcell, length, albedo):
        # set main interface attributes
        self.length = length
        self.adjcell = adjcell
        self.albedo = albedo
        self.flux = None


class Cell:
    def __init__(self, name=None, cell_type=None, ni=None, ne=None, ti=None, te=None, **kwargs):
        self.name = name
        self.cell_type = cell_type  # internal or bdry
        self.ni = ni
        self.ne = ne
        self.ti = ti
        self.te = te
        self.interfaces = None  # list of instances of the Interface class
        self.area = None
        self.perimeter = None

        self.nn = None
        self.tn = None

        self.angles = None

        self.sv_el = None
        self.sv_eln = None
        self.sv_cx = None
        self.sv_rec = None
        self.sv_ion_e = None
        self.sv_ion_i = None
        self.sv = None

        self.mfp = None
        self.c_i = None
        self.x_i = None
        self.p_0i = None
        self.p_i = None

        self.t_coefs = None  # each cell with have n*(n-1)/2 transmission coefficients, where n = n_sides

        # set cell cross sections
        # self.set_sv()

        # set cell transport related properties
        # self.set_mfp()
        # self.set_c_i()
        # self.set_x_i()
        # self.set_p_0i()
        # self.set_p_i()

    def set_angles(self):
        l_sides = deque([interface.length for interface in self.interfaces])
        l_sides.rotate(1)

    def set_interfaces(self):
        return

    def set_geom(self):
        l_sides = [interface.length for interface in self.interfaces]
        self.area = area_triangle(l_sides)
        self.perimeter = sum(l_sides)

    def set_sv(self, degas=False):
        # TODO: figure out which temperatures, etc. need to be passed to each of the cross section interpolators
        self.sv_el = sv.el(self.ti, self.tn)
        self.sv_eln = sv.eln()

        if degas:
            self.sv_ion_e = 10 ** sv.ion_e_degas()
            self.sv_ion_i = 10 ** sv.ion_i_degas()
            self.sv_rec = 10 ** sv.rec_degas()
            self.sv_cx = 10 ** sv.cx_degas
        else:
            self.sv_ion_e = 10 ** sv.ion_e()
            self.sv_ion_i = 10 ** sv.ion_i_degas()  # degas library is the only one available
            self.sv_rec = 10 ** sv.rec()
            self.sv_cx = 10 ** sv.cx()
        return

    def set_mfp(self):
        """
        Calculates the mean free path of a neutral particle through a background plasma
        :return:
        """

        vn = np.sqrt(2 * self.tn * 1E3 * e / (2 * m_p))
        self.mfp = vn / (self.ne * self.sv_ion_e + self.ni * self.sv_cx + self.ni * self.sv_el)

    def set_c_i(self):
        self.c_i = (self.sv_cx + self.sv_el) / (self.ne / self.ni * self.sv_ion_e + self.sv_cx + self.sv_el)

    def set_x_i(self):
        self.x_i = 4.0 * self.area / (self.mfp * self.perimeter)

    def set_p_0i(self, n_sauer=2.0931773):
        self.p_0i = 1 / self.x_i * (1 - (1 + self.x_i / n_sauer) ** -n_sauer)

    def set_p_i(self):
        self.p_i = self.p_0i / (1 - self.c_i * (1 - self.p_0i))

    def set_t_coefs(self):
        t_coef = defaultdict(dict)
        adjcells = [_.adjcell for _ in self.interfaces]

        for cell1 in adjcells:
            for cell2 in adjcells:
                if cell2 in t_coef and cell1 in t_coef[cell2]:
                    t_coef[cell1][cell2] = t_coef[cell2][cell1]
                else:
                    t_coef = calc_t_coef()
