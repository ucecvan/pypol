from typing import Union
from PyPol.walls import Wall


class _MetaCV(object):
    """
    General Class for Collective Variables.
    Attributes:\n
    - name: name of the CV.
    - cv_type: Type of the CV.
    - sigma: the bandwidth for kernel density estimation.
    - grid_min: the lower bounds for the grid.
    - grid_max: the upper bounds for the grid.
    - grid_bins: the number of bins for the grid.
    - grid_space: the approximate grid spacing for the grid.
    """

    _name = None
    _sigma = None
    _grid_bins = None
    _grid_max = None
    _grid_min = None

    def __init__(self, name: str,
                 cv_type: str = "",
                 sigma: Union[float, list, tuple] = None,
                 grid_min: Union[float, list, tuple] = None,
                 grid_max: Union[float, list, tuple] = None,
                 grid_bins: Union[int, list, tuple] = None,
                 grid_space: Union[float, list, tuple] = None):
        """
        General Class for Collective Variables.
        :param name: name of the CV.
        :param cv_type: Type of the CV.
        :param sigma: the bandwidth for kernel density estimation.
        :param grid_min: the lower bounds for the grid.
        :param grid_max: the upper bounds for the grid.
        :param grid_bins: the number of bins for the grid.
        :param grid_space: the approximate grid spacing for the grid.
        """
        self._name = name
        self._type = cv_type
        self._grid_min = grid_min
        self._grid_max = grid_max
        self._grid_bins = grid_bins
        self._grid_space = grid_space
        self._sigma = sigma
        self._stride = 100

    @property
    def stride(self):
        return self._stride

    @stride.setter
    def stride(self, value: int):
        self._stride = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma: float):
        if self._grid_space < sigma * 0.5:
            self._sigma = sigma
        else:
            print("""
The bin size must be smaller than half the sigma. Choose a sigma greater than {}. 
Alternatively, you can change the bin space or the number of bins.""".format(self._grid_space * 2))

    @property
    def grid_min(self):
        return self._grid_min

    @grid_min.setter
    def grid_min(self, grid_min: float):
        self._grid_min = grid_min
        self.grid_space = self._grid_space

    @property
    def grid_max(self):
        return self._grid_max

    @grid_max.setter
    def grid_max(self, grid_max: float):
        self._grid_max = grid_max
        self.grid_space = self._grid_space

    @property
    def grid_bins(self):
        return self._grid_bins

    @grid_bins.setter
    def grid_bins(self, grid_bins: int):
        self._grid_bins = grid_bins
        if self._grid_max:
            self._grid_space = (self._grid_max - self._grid_min) / float(self._grid_bins)
            if self._grid_space > self._sigma * 0.5:
                print("The bin size must be smaller than half the bandwidth. Please change the bandwidth accordingly.")

    @property
    def grid_space(self):
        return self._grid_space

    @grid_space.setter
    def grid_space(self, grid_space: float):
        self._grid_space = grid_space
        if self._grid_space > self._sigma * 0.5:
            print("The bin size must be smaller than half the bandwidth. Please change the bandwidth accordingly.")
        if self._grid_max:
            self._grid_bins = int((self._grid_max - self._grid_min) / self._grid_space)

    # Read-only properties
    @property
    def name(self):
        return self._name

    def __str__(self):
        txt = """
CV: {0._name} ({0._type})
SIGMA={0._sigma:.3f} GRID_BIN={0._grid_bins} GRID_MAX={0._grid_max:.3f} GRID_MIN={0._grid_min:.3f}""".format(self)
        return txt

    def _write_output(self, path_output):
        file_output = open(path_output, "a")
        file_output.write(self.__str__())
        file_output.close()


class Density(_MetaCV):
    """
    Use the density of the crystal as a collective variable
    Attributes:\n
    - name: name of the CV.
    - cv_type: Type of the CV.
    - sigma: the bandwidth for kernel density estimation.
    - grid_min: the lower bounds for the grid.
    - grid_max: the upper bounds for the grid.
    - grid_bins: the number of bins for the grid.
    - grid_space: the approximate grid spacing for the grid.
    - use_walls: Use walls at the upper and lower bounds of the grid to force the system not to escape from it
    - walls: return a list with the upper and lower walls
    - uwall: return the upper wall
    - lwall: return the lower wall
    """

    _type = "Density"
    _short_type = "density"

    def __init__(self, name):
        """
        Use the density of the crystal as a collective variable
        :param name: name of the CV
        """
        super().__init__(name, "Density", sigma=10.)
        self._use_walls = False
        self._walls = []

    @property
    def use_walls(self):
        return self._use_walls

    @use_walls.setter
    def use_walls(self, value: bool, offset=50.):
        self._use_walls = value
        if self._use_walls:
            if self._grid_max and self._grid_min:
                self._walls = [Wall(self._name + "_upper", "UPPER"), Wall(self._name + "_lower", "LOWER")]
                self.lwall.add_arg(self._name, at=self._grid_min + offset, kappa=1000)
                self.uwall.add_arg(self._name, at=self._grid_max - offset, kappa=1000)
            else:
                print("Error: Define grid_max and grid_min before walls.")
                exit()

    @property
    def walls(self):
        return self._walls

    @property
    def uwall(self):
        if self._walls:
            return self._walls[0]

    @property
    def lwall(self):
        if self._walls:
            return self._walls[1]

    def _metad(self, value, print_output=True):
        txt = f"""
# Density
{self._name}_vol: VOLUME
{self._name}: MATHEVAL ARG={self._name}_vol FUNC={value:.3f}/x PERIODIC=NO # FUNC = NMOLS*MW*CONVERSIONFACTOR/VOLUME
"""
        if self._use_walls:
            for wall in self._walls:
                txt += wall._metad(False)

        if print_output:
            if self._use_walls:
                args = self._walls[0]._name + ".bias," + self._walls[1]._name + ".bias"
                txt += f"PRINT ARG={self._name},{args} FILE={self._name}_COLVAR STRIDE={self._stride}"
            else:
                txt += f"PRINT ARG={self._name} FILE={self._name}_COLVAR STRIDE={self._stride}"
        return txt


class PotentialEnergy(_MetaCV):
    """
    Use the Potential Energy of the crystal as a collective variable
    Attributes:\n
    - name: name of the CV.
    - cv_type: Type of the CV.
    - sigma: the bandwidth for kernel density estimation.
    - grid_min: the lower bounds for the grid.
    - grid_max: the upper bounds for the grid.
    - grid_bins: the number of bins for the grid.
    - grid_space: the approximate grid spacing for the grid.
    """

    _type = "Potential Energy"
    _short_type = "energy"

    def __init__(self, name):
        super(PotentialEnergy, self).__init__(name, "Potential Energy", sigma=2.)

    def _metad(self, nmols, imp, remove_bias: list = None, print_output=True):
        if not remove_bias:
            txt = f"""
# Potential Energy Difference
{self._name}_pot: ENERGY
{self._name}: MATHEVAL ARG={self._name}_pot VAR=a FUNC=a/{nmols}-{imp} PERIODIC=NO"""
        else:
            list_var = list("bcdefghijklmnopqrstuvwxyz")
            arg = ",".join([i._name + ".bias" for i in remove_bias])
            var = ",".join([list_var[i] for i in range(len(remove_bias))])
            func = "-".join([list_var[i] for i in range(len(remove_bias))])
            txt = f"""
{self._name}_pot: ENERGY
{self._name}: MATHEVAL ARG={self._name}_pot,{arg} VAR=a,{var} FUNC=(a-{func})/{nmols}-{imp} PERIODIC=NO\n"""

        if print_output:
            txt += f"PRINT ARG={self._name} FILE={self._name}_COLVAR STRIDE={self._stride}\n"
        return txt
