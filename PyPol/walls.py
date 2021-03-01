import os
from typing import Union

from PyPol.utilities import get_list_crystals
from PyPol.gromacs import MolecularDynamics, Metadynamics


class Wall(object):
    """
    Creates a plumed input file for upper and lower lower WALLS
    Attributes:\n
    - name: label for the WALL.
    - type: Wall.
    - arg: Inputs for the wall
    - position: Position of the wall
    - kappa: Force constant of the wall
    - offset: The offset for the start of the wall
    - exp: powers for the wall
    - eps: the rescaling factor
    - stride: If possible, the stride used for printing the bias potential in the output file
    - collective_variable_line: String to be added above the Wall command. This can be used to specify the inputs for
      the wall (ARG)

    Methods:\n
    - add_arg(name, kappa=100000, offset=0.0, exp=2, eps=1, at=0.): Add a new input for the wall
    - reset_arg(name, kappa=100000, offset=0.0, exp=2, eps=1, at=0.): Modify an existing input for the wall
    """

    def __init__(self, name, position="upper"):
        """
        Wall object.
        :param name: Name given to the variable Wall
        :param position: "upper" or "lower"
        """

        position = position.upper()
        if position not in ("LOWER", "UPPER"):
            print("Error: Position of the wall not recognized, choose between 'upper' and 'lower'.")
            exit()
        self._position = position
        self._name = name
        self._type = "Wall"
        self._arg = list()
        self._kappa = list()
        self._offset = list()
        self._exp = list()
        self._eps = list()
        self._at = list()
        self._stride = 100
        self._collective_variable_line = ""

    @property
    def collective_variable_line(self):
        return self._collective_variable_line

    @collective_variable_line.setter
    def collective_variable_line(self, values: str):
        self._collective_variable_line = values

    @property
    def kappa(self):
        return self._kappa

    @kappa.setter
    def kappa(self, values: list):
        self._kappa = values

    @property
    def at(self):
        return self._at

    @at.setter
    def at(self, values: list):
        self._at = values

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, values: list):
        self._offset = values

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, values: list):
        self._eps = values

    @property
    def exp(self):
        return self._exp

    @exp.setter
    def exp(self, values: list):
        self._exp = values

    @property
    def stride(self):
        return self._stride

    @stride.setter
    def stride(self, value: int):
        self._stride = value

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value: str):
        if value.upper() in ("LOWER", "UPPER"):
            self._position = value
        else:
            print("Error: Position of the wall not recognized, choose between 'upper' and 'lower'.")
            exit()

    def add_arg(self, name, kappa=100000, offset=0.0, exp=2, eps=1, at=0.):
        """
        Add an argument to the Wall object.
        :param name: Inputs (arg) for the wall
        :param at: Position of the wall
        :param kappa: Force constant of the wall
        :param offset: The offset for the start of the wall
        :param exp: powers for the wall
        :param eps: the rescaling factor
        :return:
        """
        self._arg.append(name)
        self._kappa.append(kappa)
        self._offset.append(offset)
        self._exp.append(exp)
        self._eps.append(eps)
        self._at.append(at)

    def reset_arg(self, name, kappa=100000, offset=0.0, exp=2, eps=1, at=0.):
        """
        Modify an existing argument of the Wall object with the default ones, unless they are specified.
        :param name: Inputs (arg) for the wall
        :param at: Position of the wall
        :param kappa: Force constant of the wall
        :param offset: The offset for the start of the wall
        :param exp: powers for the wall
        :param eps: the rescaling factor
        :return:
        """

        if name not in self._arg:
            print("Error: No ARG with name {}".format(name))
            exit()
        i = self._arg.index(name)
        self._kappa[i] = kappa
        self._offset[i] = offset
        self._exp[i] = exp
        self._eps[i] = eps
        self._at[i] = at

    def __str__(self):
        txt = "\nCV: {0._name} ({0._type})\nWall position: {0._position}".format(self)
        for i in range(len(self._arg)):
            txt += f"ARG={self._arg[i]} AT={self._at[i]} KAPPA={self._kappa[i]} EXP={self._exp[i]} " \
                   f"EPS={self._eps[i]} OFFSET={self._offset[i]}\n"
        return txt

    def _write_output(self, path_output):
        file_output = open(path_output, "a")
        file_output.write(self.__str__())
        file_output.close()

    def _metad(self, print_output=True):
        if self._collective_variable_line:
            txt = "\n" + "# Wall\n" + self._collective_variable_line + "\n"
        else:
            txt = "\n" + "# Wall\n"
        args = ",".join(self._arg)
        at = ",".join([str(a) for a in self._at])
        kappa = ",".join([str(a) for a in self._kappa])
        exp = ",".join([str(a) for a in self._exp])
        eps = ",".join([str(a) for a in self._eps])
        offset = ",".join([str(a) for a in self._offset])
        txt += f"""{self._position}_WALLS ...
ARG={args}
AT={at}
KAPPA={kappa}
EXP={exp}
EPS={eps}
OFFSET={offset}
LABEL={self._name}
... {self._position}_WALLS
"""

        if print_output:
            txt += f"\nPRINT ARG={args},{self._name}.bias FILE={self._name}_COLVAR STRIDE={self._stride}\n"
        return txt


class AvoidScrewedBox(Wall):
    """
    Creates a plumed input file forcing the non-diagonal element of the box matrix to stay within a certain range:
    abs(bx) <= 0.5*ax
    abs(cx) <= 0.5*ax
    abs(cy) <= 0.5*by
    This is done by creating three upper walls with the following attributes:
    Attributes:\n
    - name: label for the WALL.
    - type: "Avoid Screwed Box (Wall)"
    - arg: [bx,cx,cy]
    - position: [0.,0.,0.]
    - kappa: [100000,100000,100000]
    - offset: [0.1, 0.1, 0.1]
    - exp: [2,2,2]
    - eps: [1,1,1]
    - stride: If possible, the stride used for printing the bias potential in the output file
    - collective_variable_line: "
    cell: CELL
    bx: MATHEVAL ARG=cell.bx,cell.ax FUNC=abs(x)-0.5*y PERIODIC=NO
    cx: MATHEVAL ARG=cell.cx,cell.ax FUNC=abs(x)-0.5*y PERIODIC=NO
    cy: MATHEVAL ARG=cell.cy,cell.by FUNC=abs(x)-0.5*y PERIODIC=NO"

    Methods:\n
    - add_arg(name, kappa=100000, offset=0.0, exp=2, eps=1, at=0.): Add a new input for the wall
    - reset_arg(name, kappa=100000, offset=0.0, exp=2, eps=1, at=0.): Modify an existing input for the wall
    """
    _type = "Avoid Screwed Box (Wall)"
    _short_type = "asb"

    def __init__(self, name):
        """

        :param name:
        """
        super(AvoidScrewedBox, self).__init__(name, "UPPER")
        self._type = "Avoid Screwed Box (Wall)"
        self._collective_variable_line = """cell: CELL
bx: MATHEVAL ARG=cell.bx,cell.ax FUNC=abs(x)-0.5*y PERIODIC=NO
cx: MATHEVAL ARG=cell.cx,cell.ax FUNC=abs(x)-0.5*y PERIODIC=NO
cy: MATHEVAL ARG=cell.cy,cell.by FUNC=abs(x)-0.5*y PERIODIC=NO"""

        self.add_arg("bx", offset=0.1)
        self.add_arg("cx", offset=0.1)
        self.add_arg("cy", offset=0.1)

    def generate_input(self, simulation: Union[MolecularDynamics, Metadynamics],
                       crystals="all",
                       catt=None):

        """
        Generate the plumed input files. This is particularly useful for crystals with tilted boxes.
        If the catt option is used, only crystals with the specified attribute are used.
        Attributes must be specified in the form of a python dict, menaning catt={"AttributeLabel": "AttributeValue"}.
        NB: The <simulation>.mdrun_options attribute is modified to include "-plumed plumed_<name>.dat"
        :param catt: Use crystal attributes to select the crystal list
        :param simulation: Simulation object
        :param crystals: It can be either "all", use all non-melted Crystal objects from the previous simulation or
                         "centers", use only cluster centers from the previous simulation. Alternatively, you can select
                         a specific subset of crystals by listing crystal names.
        :return:
        """
        list_crystals = get_list_crystals(simulation._crystals, crystals, attributes=catt)
        add_plumed_file = False
        file_plumed = None
        if "-plumed" in simulation._mdrun_options:
            add_plumed_file = input("A plumed file has been found in the mdrun options. \n"
                                    "Do you want to add it the plumed input (NB: if not, it will be ignored for this "
                                    "simulation)? [y/n] ")
            if add_plumed_file.lower() in ("yes", "y", "true"):
                add_plumed_file = True
                it = iter(simulation._mdrun_options.split())
                for i in it:
                    if i == "-plumed":
                        file_plumed = next(it)
            else:
                add_plumed_file = False
        simulation._mdrun_options = " -plumed plumed_{}.dat ".format(self._name)
        for crystal in list_crystals:
            txt = self._metad()
            f = open(crystal._path + "plumed_{}", "w")
            f.write(txt)
            if add_plumed_file:
                if os.path.exists(crystal._path + file_plumed):
                    f2 = open(crystal._path + file_plumed, "r")
                    f.write("".join(f2.readlines()))
                    f2.close()
            f.close()
