import json
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import cpu_count

from cell import Cell


class Neutpy:
    def __init__(self, input_path):
        # read input file
        with open(input_path) as f:
            cell_data = json.load(f)

        # populate cells (this could be parallelized, but probably doesn't need it)
        self.cells = {}

        for cellname in cell_data.keys():
            # instantiate the cell
            self.cells[cellname] = Cell(cellname, **cell_data[cellname])

            # create interfaces for the cell
            adjcells = cell_data[cellname]['adjcells']
            lengths = cell_data[cellname]['lengths']
            self.cells[cellname].set_interfaces(adjcells, lengths)

        # calculate transmission coefficients (this will need to be parallelized)
        # for cell in self.cells:
        #    cell.set_t_coefs()

        # construct matrix

        # solve matrix

        # calculate densities from fluxes and cross sections

        #


if __name__ == '__main__':
    neuts = Neutpy('example_input.json')
