import json
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import cpu_count
import time

from cell import Cell


class Neutpy:
    def __init__(self, input_path):
        # read input file
        with open(input_path) as f:
            cell_data = json.load(f)

        # populate cells (this could be parallelized, but probably doesn't need it)
        self.cells = []

        for cell in cell_data.keys():
            self.cells.append(Cell(**cell_data[cell]))

        # create interfaces (this could be parallelized, but probably doesn't need it)
        # for cell in self.cells:
        #    cell.set_interfaces()

        # calculate transmission coefficients (this will need to be parallelized)
        # for cell in self.cells:
        #    cell.set_t_coefs()

        # construct matrix

        # solve matrix

        # calculate densities from fluxes and cross sections

        #


if __name__ == '__main__':
    neuts = Neutpy('example_input.json')
