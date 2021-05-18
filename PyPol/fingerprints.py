import numpy as np
import os
import matplotlib.pyplot as plt
import progressbar
from typing import Union
from sklearn.neighbors import KernelDensity as KDE

from PyPol.utilities import get_list_crystals, hellinger, generate_atom_list
from PyPol.crystals import Molecule
from PyPol.gromacs import EnergyMinimization, MolecularDynamics, CellRelaxation, Metadynamics


# TODO Correct Docstrings

#
# Distributions
#


class _Distribution(object):
    """
    General Class for Collective Variables.
    Attributes:\n
    - name: name of the CV.
    - type: Type of the CV.
    - clustering_type: How is it treated by clustering algorithms.
    - kernel: kernel function to use in the histogram generation.
    - bandwidth: the bandwidths for kernel density estimation.
    - grid_min: the lower bounds for the grid.
    - grid_max: the upper bounds for the grid.
    - grid_bins: the number of bins for the grid.
    - grid_space: the approximate grid spacing for the grid.
    - timeinterval: Simulation time interval to generate the distribution.
    """

    def __init__(self, name: str, cv_type: str, cv_short_type: str,
                 plumed: Union[str, None], plumed_version: Union[str, None] = "master",
                 clustering_type="distribution", kernel="GAUSSIAN",
                 bandwidth: Union[float, list, tuple] = None,
                 grid_min: Union[float, list, tuple] = None,
                 grid_max: Union[float, list, tuple] = None,
                 grid_bins: Union[int, list, tuple] = None,
                 grid_space: Union[float, list, tuple] = None,
                 timeinterval: Union[int, float, tuple] = None):
        """
        General Class for Collective Variables.
        :param name: name of the CV.
        :param cv_type: Type of the CV.
        :param plumed: Command line for plumed.
        :param clustering_type: How is it treated by clustering algorithms.
        :param kernel: kernel function to use in the histogram generation.
        :param bandwidth: the bandwidths for kernel density estimation.
        :param grid_min: the lower bounds for the grid.
        :param grid_max: the upper bounds for the grid.
        :param grid_bins: the number of bins for the grid.
        :param grid_space: the approximate grid spacing for the grid.
        :param timeinterval: Simulation time interval to generate the distribution.
        """
        self._name = name
        self._type = cv_type
        self._short_type = cv_short_type
        self._clustering_type = clustering_type

        self._kernel = kernel
        self._bandwidth = bandwidth

        self._grid_min = grid_min
        self._grid_max = grid_max
        self._grid_bins = grid_bins
        self._timeinterval = timeinterval
        self._grid_space = grid_space
        self._plumed = plumed
        self._plumed_version = plumed_version

        self._matt = None

    @property
    def molecular_attributes(self):
        return self._matt

    @molecular_attributes.setter
    def molecular_attributes(self, att: dict):
        self._matt = att

    def set_molecular_attribute(self, att, val):
        """
        Create a custom attribute for the Crystal.
        :param att: Attribute label
        :param val: Attribute value
        :return:
        """
        if self._matt is None:
            self._matt = {}
        self._matt[att] = val

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, kernel: str):
        if kernel.upper() in ("GAUSSIAN", "TRIANGULAR"):
            self._kernel = kernel
        else:
            print("Kernel function not recognized. Choose between 'GAUSSIAN' and 'TRIANGULAR'.")

    @property
    def timeinterval(self):
        return self._timeinterval

    @timeinterval.setter
    def timeinterval(self, time: float, time2: float = None):
        if time2:
            self._timeinterval = (time, time2)
        else:
            self._timeinterval = time

    @property
    def bandwidth(self):
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, bandwidth: float):
        if self._grid_space < bandwidth * 0.5:
            self._bandwidth = bandwidth
        else:
            print("""
The bin size must be smaller than half the bandwidth. Choose a bandwidth greater than {}. 
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
            if self._grid_space > self._bandwidth * 0.5:
                print("The bin size must be smaller than half the bandwidth. Please change the bandwidth accordingly.")

    @property
    def grid_space(self):
        return self._grid_space

    @grid_space.setter
    def grid_space(self, grid_space: float):
        self._grid_space = grid_space
        if self._grid_space > self._bandwidth * 0.5:
            print("The bin size must be smaller than half the bandwidth. Please change the bandwidth accordingly.")
        if self._grid_max:
            self._grid_bins = int((self._grid_max - self._grid_min) / self._grid_space)

    # Read-only properties
    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    @property
    def clustering_type(self):
        return self._clustering_type

    def __str__(self):
        if self._grid_max:
            txt = """
CV: {0._name} ({0._type})
Clustering Type: {0._clustering_type}
Plumed command: {0._plumed}      
KERNEL={0._kernel} BANDWIDTH={0._bandwidth:.3f} 
NBINS={0._grid_bins} GRIDSPACE={0._grid_space:.3f} UPPER={0._grid_max:.3f} LOWER={0._grid_min:.3f}""".format(self)
        else:
            txt = """
CV: {0._name} ({0._type})
Clustering Type: {0._clustering_type}
Plumed command: {0._plumed}      
KERNEL={0._kernel} BANDWIDTH={0._bandwidth:.3f} GRIDSPACE={0._grid_space:.3f}""".format(self)
        return txt

    def check_attributes(self):
        print(f"No attributes-check set for distribution {self._name}")

    def generate_input(self, crystal, input_name="", output_name=""):
        print(f"No plumed input generation available for distribution {self._name}."
              f"Cannot create a plumed file '{input_name}' for crystal {crystal._name}")

    def generate_inputs(self,
                        simulation: Union[EnergyMinimization, CellRelaxation, MolecularDynamics, Metadynamics],
                        bash_script=True,
                        crystals="all",
                        catt=None):
        """
        Generate the plumed input files. If the catt option is used, only crystals with the specified attribute are
        used. In both cases,
        attributes must be specified in the form of a python dict, meaning catt={"AttributeLabel": "AttributeValue"}.

        :param simulation: Simulation object
        :param bash_script: If True, generate a bash script to run simulations
        :param crystals: It can be either "all", use all non-melted Crystal objects from the previous simulation or
                         "centers", use only cluster centers from the previous simulation. Alternatively, you can select
                         a specific subset of crystals by listing crystal names.
        :param catt: Use crystal attributes to select the crystal list
        :return:
        """

        self.check_attributes()
        print("=" * 100)
        print(self.__str__())

        list_crystals = get_list_crystals(simulation._crystals, crystals, catt)

        for crystal in list_crystals:
            self.generate_input(crystal,
                                crystal._path + f"/plumed_{self._name}.dat",
                                crystal._path + f"/plumed_{simulation._name}_{self._name}.dat")

        if bash_script:
            dt, nsteps, traj_stride, traj_start, traj_end = (None, None, None, None, None)

            file_mdp = open(simulation._path_mdp)
            for line in file_mdp:
                if line.startswith('dt '):
                    dt = float(line.split()[2])
                elif line.startswith(("nstxout", "nstxout-compressed")):
                    traj_stride = int(line.split()[2])
                elif line.startswith('nsteps '):
                    nsteps = float(line.split()[2])
            file_mdp.close()

            traj_time = int(nsteps * dt)
            if traj_time > 0:
                if isinstance(self._timeinterval, tuple):
                    traj_start = self._timeinterval[0]
                    traj_end = self._timeinterval[1]
                elif isinstance(self._timeinterval, int):
                    traj_start = traj_time - self._timeinterval
                    traj_end = traj_time
                else:
                    print("Error: No suitable time interval.")
                    exit()

            file_script = open(simulation._path_data + "/run_plumed_" + self._name + ".sh", "w")
            file_script.write('#!/bin/bash\n\n'
                              'crystal_paths="\n')
            for crystal in list_crystals:
                file_script.write(crystal._path + "\n")

            if isinstance(simulation, Metadynamics):
                file_script.write('"\n\n'
                                  'for crystal in $crystal_paths ; do\n'
                                  'cd "$crystal" || exit \n'
                                  '#{0} trjconv -f {1}.xtc -o plumed_{1}.xtc -s {1}.tpr -b {2} -e {3} <<< 0\n'
                                  '{4} driver --mf_xtc plumed_{1}.xtc --plumed plumed_{5}.dat --timestep {6} '
                                  '--trajectory-stride {7} --mc mc.dat\n'
                                  '#rm plumed_{1}.xtc\n'
                                  'done\n'
                                  ''.format(simulation._gromacs, simulation._name, traj_start, traj_end,
                                            self._plumed, self._name, dt, traj_stride))
                file_script.close()
            else:
                file_script.write('"\n\n'
                                  'for crystal in $crystal_paths ; do\n'
                                  'cd "$crystal" || exit \n'
                                  '{0} trjconv -f {1}.xtc -o plumed_{1}.xtc -s {1}.tpr -b {2} -e {3} <<< 0\n'
                                  '{4} driver --mf_xtc plumed_{1}.xtc --plumed plumed_{5}.dat --timestep {6} '
                                  '--trajectory-stride {7} --mc mc.dat\n'
                                  'rm plumed_{1}.xtc\n'
                                  'done\n'
                                  ''.format(simulation._gromacs, simulation._name, traj_start, traj_end,
                                            self._plumed, self._name, dt, traj_stride))
                file_script.close()
        print("=" * 100)

    def get_from_file(self, crystal, input_file, output_label="", plot=True):
        print(f"No instructions were given to upload data from file '{input_file}' of crystal {crystal._name}")

    def get_results(self,
                    simulation: Union[EnergyMinimization, CellRelaxation, MolecularDynamics, Metadynamics],
                    crystals: Union[str, list, tuple] = "all",
                    plot: bool = True, catt=None, suffix=""):
        """
        Verify if the distribution has been correctly generated and store the result. If the distribution is taken over
        different frames, the average is calculated.
        :param simulation: Simulation object
        :param crystals: It can be either "all", use all non-melted Crystal objects from the previous simulation or
                         "centers", use only cluster centers from the previous simulation. Alternatively, you can select
                         a specific subset of crystals by listing crystal names.
        :param plot: If true, generate a plot of the distribution.
        :param catt: Use crystal attributes to select the crystal list
        :param suffix: suffix to add to the cv name.
        :return:
        """
        list_crystals = get_list_crystals(simulation._crystals, crystals, catt)
        print("\n" + str(self._name))
        bar = progressbar.ProgressBar(maxval=len(list_crystals)).start()
        nbar = 1
        for crystal in list_crystals:
            path_plumed_output = crystal._path + "plumed_{}_{}.dat".format(simulation._name, self._name)
            if os.path.exists(path_plumed_output):
                crystal._cvs[self._name + suffix] = self.get_from_file(crystal, path_plumed_output, simulation._name,
                                                                       plot)
                bar.update(nbar)
                nbar += 1
            else:
                print("An error has occurred with Plumed. Check file {} in folder {}."
                      "".format(path_plumed_output, crystal._path))
        bar.finish()

    def _write_output(self, path_output):
        file_output = open(path_output, "a")
        file_output.write(self.__str__())
        file_output.close()


class Torsions(_Distribution):
    """
    Generates a distribution of the torsional angles of the selected atoms.
    Attributes:\n
    - name: name of the CV.
    - type: Type of the CV.
    - plumed: Command line for plumed.
    - clustering_type: How is it treated by clustering algorithms.
    - kernel: kernel function to use in the histogram generation.
    - bandwidth: the bandwidths for kernel density estimation.
    - grid_min: the lower bounds for the grid.
    - grid_max: the upper bounds for the grid.
    - grid_bins: the number of bins for the grid.
    - grid_space: the approximate grid spacing for the grid.
    - timeinterval: Simulation time interval to generate the distribution.
    - atoms: the 4 atom index of the molecular forcefield object used to generate the set of torsional angles
    - molecule: the molecular forcefield object from which atoms are selected

    Methods:\n
    - help(): Print attributes and methods
    - set_atoms(atoms, molecule): Select the 4 atom index from the Molecule obj to generate the set of torsional angles
    - generate_input(simulation, bash_script=True): Generate the plumed input files
    - get_results(simulation, crystal='all', plot=True): Check if the plumed driver analysis is ended and store results
    """
    _type = "Torsional Angle"
    _short_type = "tor"
    _plumed_version = "hack-the-tree"
    _clustering_type = "distribution"

    def __init__(self, name: str, plumed: str):
        """
        Generates a distribution of the torsional angles of the selected atoms.
        :param name: str, name of the collective variable. Default output and variables will have this name.
        :param plumed: command line for the plumed file.
        """
        super().__init__(name=name,
                         cv_type="Torsional Angle",
                         cv_short_type="tor",
                         plumed=plumed,
                         plumed_version="hack-the-tree",
                         clustering_type="distribution",
                         kernel="GAUSSIAN",
                         bandwidth=0.25,
                         grid_bins=73,
                         grid_min=-np.pi,
                         grid_max=np.pi,
                         grid_space=2 * np.pi / 73,
                         timeinterval=200)

        self._atoms = list()
        self._molecule = None

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        if len(atoms) == 4:
            self._atoms = atoms
        else:
            print("Error: Torsional Angles needs 4 atoms as input")

    @property
    def molecule(self):
        return self._molecule

    @molecule.setter
    def molecule(self, molecule):
        self._molecule = molecule

    @staticmethod
    def help():
        return """
Calculate the distribution of a set of torsional angles.
It creates the inputs for plumed and stores the results.

Attributes:
- name: name of the CV.
- type: Type of the CV (Torsional Angle).
- plumed: Command line for plumed.
- clustering_type: How is it treated by clustering algorithms (distribution). 
- kernel: kernel function to use in the histogram generation. It can be "TRIANGULAR" or "GAUSSIAN"
- bandwidth: the bandwidths for kernel density estimation. The bin size must be smaller than half the bandwidth.
- grid_min: the lower bounds for the grid.
- grid_max: the upper bounds for the grid.
- grid_bins: the number of bins for the grid.
- grid_space: the approximate grid spacing for the grid.
- timeinterval: Simulation time interval to generate the distribution.
                If a single value is given, t, frames corresponding to the last "t" picoseconds are used.
                If two values are given, t1 and t2, frames from time t1 to time t2 are used.
- atoms: the 4 atom index of the molecular forcefield object used to generate the set of torsional angles.
                The same torsional angle in each molecule of the crystal will be considered for the distribution.
- molecule: the molecular forcefield object from which atoms are selected.

Methods:
- help(): Print attributes and methods
- set_atoms(atoms, molecule): Select the 4 atom index from the Molecule obj to generate the set of torsional angles. 
                The atom index in PyPol starts from 0 and can be seen in the
- generate_input(simulation, bash_script=True): Generate the plumed input files
- get_results(simulation, crystal='all', plot=True): Check if the plumed driver analysis is ended and store results
                If crystal="all", results are stored for all crystals. Alternatively, you can select a subset of 
                crystals by specifying their IDs in an iterable object.
                If plot=True, a plot of the distribution is created. This could be slow for large sets.

Examples:
- Select atoms of the torsional angles and create plumed inputs:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
molecule = gaff.get_molecule("MOL")                           # Use molecular forcefield info for the CV 
tor = gaff.get_cv("tor")                                      # Retrieve the CV Object
tor.set_atoms((0, 1, 2, 3), molecule)                         # Use the first four atoms to define the torsional angle
npt = gaff.get_simulation("npt")                              # Retrieve a completed simulation
tor.generate_input(npt)                                       # Generate plumed driver input for the selected simulation
project.save()                                                # Save project

- Import distributions once the plumed driver analysis is finished:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
molecule = gaff.get_molecule("MOL")                           # Use molecular forcefield info for the CV 
tor = gaff.get_cv("tor")                                      # Retrieve the CV Object
npt = gaff.get_simulation("npt")                              # Retrieve a completed simulation
tor.get_results(npt, plot=False)                              # Generate plumed driver input for the selected simulation
project.save()                                                # Save project"""

    def set_atoms(self, atoms: Union[list, tuple], molecule: Molecule):
        """
        Select atom indices of the reference molecule. This is used to identify the torsions of each molecule in the
        crystal.
        :param atoms: list, Atom indices. All atoms indices are available in the project output file after the topology
        is defined.
        :param molecule: obj, Reference molecule
        :return:
        """
        self.atoms = atoms
        self.molecule = molecule

    def check_attributes(self):
        if not self._atoms:
            print("Error: no atoms found. select atoms with the set_atoms module.")
            exit()

    def generate_input(self, crystal, input_name="", output_name=""):

        lines_atoms = generate_atom_list(self._atoms, self._molecule, crystal, keyword="ATOMS", lines=[],
                                         attributes=self._matt)
        file_plumed = open(input_name, "w")
        file_plumed.write("TORSIONS ...\n")
        for line in lines_atoms:
            file_plumed.write(line)

        file_plumed.write("HISTOGRAM={{{{{0._kernel} NBINS={0._grid_bins} BANDWIDTH={0._bandwidth:.3f} "
                          "UPPER={0._grid_max:.3f} LOWER={0._grid_min:.3f}}}}}\n".format(self))

        file_plumed.write("LABEL={0}\n... TORSIONS\n\n"
                          "PRINT ARG={0}.* FILE={1}\n".format(self._name, output_name))
        file_plumed.close()

    def get_from_file(self, crystal, input_file, output_label="", plot=True):
        cv = np.genfromtxt(input_file, skip_header=1)
        if cv.ndim == 2:
            cv = cv[:, 1:]
            if np.isnan(cv).any():
                cv = np.nanmean(cv, axis=0)
                if np.isnan(cv).any():
                    print(f"\nError: NaN values present in final distribution of crystal {crystal._name}. "
                          f"Check {input_file}")
                    exit()
                print(f"\nWarning: NaN values present in some frames of crystal {crystal._name}. Check {input_file}")
            else:
                cv = np.average(cv, axis=0)
        else:
            cv = cv[1:]
            if np.isnan(cv).any():
                print(f"\nError: NaN values present in final distribution of crystal {crystal._name}. "
                      f"Check {input_file}")
                exit()
        cv /= cv.sum()
        # Save output and plot distribution
        x = np.linspace(self._grid_min, self._grid_max, len(cv))
        np.savetxt(os.path.dirname(input_file) + "/plumed_{}_{}_data.dat".format(output_label, self._name),
                   np.column_stack((x, cv)), fmt=("%1.3f", "%1.5f"),
                   header="Angle ProbabilityDensity")
        if plot:
            plt.plot(x, cv, "-")
            plt.xlabel("Torsional Angle / rad")
            plt.xlim(self._grid_min, self._grid_max)
            plt.ylabel("Probability Density")
            plt.savefig(os.path.dirname(input_file) + "/plumed_{}_{}_plot.png".format(output_label, self._name),
                        dpi=300)
            plt.close("all")
        return cv

    def __str__(self):
        txt = super(Torsions, self).__str__()
        if self._atoms:
            txt += "\nAtoms:  "
            for atom in self._atoms:
                txt += "{}({})  ".format(atom, self._molecule._atoms[atom]._label)
        else:
            txt += "No atoms found in CV {}. Select atoms with the 'set_atoms' module.\n".format(self._name)
        txt += "\n"
        return txt


class MolecularOrientation(_Distribution):
    """
    Generates a distribution of the intermolecular torsional angles of the selected atoms.
    Attributes:\n
    - name: name of the CV.
    - type: Type of the CV.
    - plumed: Command line for plumed.
    - clustering_type: How is it treated by clustering algorithms.
    - kernel: kernel function to use in the histogram generation.
    - bandwidth: the bandwidths for kernel density estimation.
    - grid_min: the lower bounds for the grid.
    - grid_max: the upper bounds for the grid.
    - grid_bins: the number of bins for the grid.
    - grid_space: the approximate grid spacing for the grid.
    - timeinterval: Simulation time interval to generate the distribution.
    - atoms: the 2 atom index of the molecular forcefield object used to generate the set of orientational vectors
    - molecules: the molecular forcefield object from which atoms are selected

    Methods:\n
    - help(): Print attributes and methods
    - set_atoms(atoms, molecule): Select the 4 atom index from the Molecule obj to generate the set of torsional angles
    - generate_input(simulation, bash_script=True): Generate the plumed input files
    - get_results(simulation, crystal='all', plot=True): Check if the plumed driver analysis is ended and store results
    """

    _type = "Molecular Orientation"
    _short_type = "mo"
    _plumed_version = "hack-the-tree"
    _clustering_type = "distribution"

    def __init__(self, name, plumed):
        """
        Generates a distribution of the intermolecular torsional angles of the selected atoms.
        :param name: str, name of the collective variable. Default output and variables will have this name.
        """

        super().__init__(name=name,
                         cv_type="Molecular Orientation",
                         cv_short_type="mo",
                         plumed=plumed,
                         plumed_version="hack-the-tree",
                         clustering_type="distribution",
                         kernel="GAUSSIAN",
                         bandwidth=0.25,
                         grid_min=0.0,
                         grid_max=np.pi,
                         grid_bins=37,
                         grid_space=np.pi / 37,
                         timeinterval=200)

        self._atoms = list()
        self._molecules = list()

    @property
    def atoms(self):
        return self._atoms

    @property
    def molecules(self):
        return self._molecules

    def __str__(self):
        txt = super(MolecularOrientation, self).__str__()
        if self._atoms:
            for idx_mol in range(len(self._molecules)):
                txt += "\nMolecule '{}': ".format(self._molecules[idx_mol]._residue)
                for atom in self._atoms[idx_mol]:
                    txt += "{}({})    ".format(atom, self._molecules[idx_mol]._atoms[atom]._label)
            txt += "\n"
        else:
            txt += "No atoms found in CV {}. Select atoms with the 'set_atoms' module.\n".format(self._name)
        return txt

    @staticmethod
    def help():
        return """
Calculate the distribution of a set of torsional angles.
It creates the inputs for plumed and stores the results.

Attributes:
- name: name of the CV.
- type: Type of the CV (Torsional Angle).
- plumed: Command line for plumed.
- clustering_type: How is it treated by clustering algorithms (distribution). 
- kernel: kernel function to use in the histogram generation. It can be "TRIANGULAR" or "GAUSSIAN"
- bandwidth: the bandwidths for kernel density estimation. The bin size must be smaller than half the bandwidth.
- grid_min: the lower bounds for the grid.
- grid_max: the upper bounds for the grid.
- grid_bins: the number of bins for the grid.
- grid_space: the approximate grid spacing for the grid.
- timeinterval: Simulation time interval to generate the distribution.
                If a single value is given, t, frames corresponding to the last "t" picoseconds are used.
                If two values are given, t1 and t2, frames from time t1 to time t2 are used.
- atoms: the 2 atom index of the molecular forcefield object used to generate the set of orientational vectors
- molecules: the molecular forcefield object from which atoms are selected

Methods:
- help(): Print attributes and methods
- set_atoms(atoms, molecule): Select the 2 atom index from the Molecule obj to generate the orientational vectors. 
                The atom index in PyPol starts from 0 and can be seen in the
- generate_input(simulation, bash_script=True): Generate the plumed input files
- get_results(simulation, crystal='all', plot=True): Check if the plumed driver analysis is ended and store results
                If crystal="all", results are stored for all crystals. Alternatively, you can select a subset of 
                crystals by specifying their IDs in an iterable object.
                If plot=True, a plot of the distribution is created. This could be slow for large sets.

Examples:
- Select atoms of the orientational vectors and create plumed inputs:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
molecule = gaff.get_molecule("MOL")                           # Use molecular forcefield info for the CV 
mo = gaff.get_cv("mo")                                        # Retrieve the CV Object
mo.set_atoms((0, 1), molecule)                                # Use the first two atoms to define the orientational vect
npt = gaff.get_simulation("npt")                              # Retrieve a completed simulation
mo.generate_input(npt)                                        # Generate plumed driver input for the selected simulation
project.save()                                                # Save project

- Import distributions once the plumed driver analysis is finished:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
molecule = gaff.get_molecule("MOL")                           # Use molecular forcefield info for the CV 
mo = gaff.get_cv("mo")                                        # Retrieve the CV Object
npt = gaff.get_simulation("npt")                              # Retrieve a completed simulation
mo.get_results(npt, plot=False)                               # Generate plumed driver input for the selected simulation
project.save()                                                # Save project

- Check orientational disorder:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
mo = gaff.get_cv("mo")                                        # Retrieve the CV Object
npt = gaff.get_simulation("npt")                              # Retrieve a completed simulation
mo.identify_orientational_disorder(npt)                       # Identify melted structures 
project.save()                                                # Save project"""

    def set_atoms(self, atoms: Union[list, tuple], molecule: Molecule):
        """
        :param atoms:
        :param molecule:
        :return:
        """
        if len(atoms) == 2 and molecule not in self._molecules:
            self._atoms.append(list(atoms))
            self._molecules.append(molecule)

    # TODO Introduce this method after multiple molecular forcefields can be used
    # def remove_atoms(self, index="all"):
    #     if index == "all":
    #         self._atoms.clear()
    #         self._molecules.clear()
    #     elif isinstance(index, int):
    #         del self._atoms[index]
    #         del self._molecules[index]
    #     else:
    #         print("Error: not clear which set of atoms you want to delete.")

    def check_attributes(self):
        if not self._atoms:
            print("Error: no atoms found. select atoms with the set_atoms module.")
            exit()

    def generate_input(self, crystal, input_name="", output_name=""):
        # Select atoms and molecules
        lines_atoms = []
        for idx_mol in range(len(self._molecules)):
            lines_atoms = generate_atom_list(self._atoms[idx_mol], self._molecules[idx_mol], crystal,
                                             keyword="ATOMS", lines=lines_atoms, attributes=self._matt)
        file_plumed = open(input_name, "w")

        file_plumed.write("DISTANCE ...\n")
        for line in lines_atoms:
            file_plumed.write(line)
        file_plumed.write("LABEL=dd_{0}\n"
                          "COMPONENTS\n"
                          "... DISTANCE\n\n"
                          "vv_{0}: NORMALIZE ARG1=dd_{0}.x ARG2=dd_{0}.y ARG3=dd_{0}.z\n"
                          "dp_mat_{0}: DOTPRODUCT_MATRIX GROUP1=vv_{0}.x GROUP2=vv_{0}.y GROUP3=vv_{0}.z\n"
                          "ang_mat_{0}: MATHEVAL ARG1=dp_mat_{0} FUNC=acos(x) PERIODIC=NO\n"
                          "valg_{0}: KDE ARG1=ang_mat_{0} GRID_MIN={1} GRID_MAX={2} "
                          "GRID_BIN={3} BANDWIDTH={4} KERNEL={5}\n\n"
                          "PRINT ARG=valg_{0} FILE={6}\n"
                          "".format(self._name, self._grid_min, self._grid_max,
                                    self._grid_bins, self._bandwidth, self._kernel, output_name))
        file_plumed.close()

    def get_from_file(self, crystal, input_file, output_label="", plot=True):
        cv = np.genfromtxt(input_file, skip_header=2)
        if cv.ndim == 2:
            cv = cv[:, 1:]
            if np.isnan(cv).any():
                cv = np.nanmean(cv, axis=0)
                if np.isnan(cv).any():
                    print(f"\nError: NaN values present in final distribution of crystal {crystal._name}. "
                          f"Check {input_file} ")
                    exit()
                print(f"\nWarning: NaN values present in some frames of crystal {crystal._name}. Check {input_file} ")
            else:
                cv = np.average(cv, axis=0)
        else:
            cv = cv[1:]
            if np.isnan(cv).any():
                print(f"\nError: NaN values present in final distribution of crystal {crystal._name}. "
                      f"Check {input_file} ")
                exit()

            # Save output and plot distribution
        x = np.linspace(self._grid_min, self._grid_max, len(cv))
        np.savetxt(os.path.dirname(input_file) + "/plumed_{}_{}_data.dat".format(output_label, self._name),
                   np.column_stack((x, cv)), fmt=("%1.3f", "%1.5f"),
                   header="Angle ProbabilityDensity")
        if plot:
            plt.plot(x, cv, "-")
            plt.xlabel("Intermolecular Angle / rad")
            plt.xlim(self._grid_min, self._grid_max)
            plt.ylabel("Probability Density")
            plt.title(crystal._name)
            plt.savefig(os.path.dirname(input_file) + "/plumed_{}_{}_plot.png".format(output_label, self._name),
                        dpi=300)
            plt.close("all")
        return cv

    def identify_orientational_disorder(self,
                                        simulation: Union[EnergyMinimization, CellRelaxation,
                                                          MolecularDynamics, Metadynamics],
                                        crystals: Union[str, list, tuple] = "all",
                                        cutoff: float = 0.1, catt=None):
        """
        Given the intermolecular angle distribution obtained for each crystal in a simulation, it compares
        it with an homogeneous distribution (typical of melted systems) to identify possible orientational disorder.
        :param simulation: Simulation object
        :param crystals: It can be either "all", use all non-melted Crystal objects from the previous simulation or
                         "centers", use only cluster centers from the previous simulation. Alternatively, you can select
                         a specific subset of crystals by listing crystal names.
        :param cutoff: Distance cutoff from melted to be used for identifying melted structures
        :param catt: Use crystal attributes to select the crystal list
        :return:
        """
        if self._grid_min != 0. and self._grid_max != np.pi:
            print("Error: A range between 0 and pi must be used to identify melted structures.")
            exit()
        include_melted = False
        if crystals == "all":
            include_melted = True

        list_crystals = get_list_crystals(simulation._crystals, crystals, catt, _include_melted=include_melted)

        file_hd = open("{}/HD_{}.dat".format(simulation._path_output, simulation._name), "w")
        file_hd.write("# Tolerance = {}\n#\n# Structures HD\n".format(round(cutoff, 5)))
        ref = np.sin(np.linspace(0., np.pi, self._grid_bins + 1))
        for crystal in list_crystals:
            if not (self._name in crystal._cvs):
                print("Error: A distribution for this simulation has not been generated.\n"
                      "Remember to run the check_normal_termination after running plumed.")
                exit()
            hd = hellinger(crystal._cvs[self._name], ref)
            file_hd.write("{:35} {:3.3f}\n".format(crystal._name, hd))
            if hd < cutoff:
                crystal._state = "melted"
        file_hd.close()


class Planes(_Distribution):
    """
    TODO Change docstrings
    Generates a distribution of the torsional angles of the selected atoms.
    Attributes:\n
    - name: name of the CV.
    - type: Type of the CV.
    - plumed: Command line for plumed.
    - clustering_type: How is it treated by clustering algorithms.
    - kernel: kernel function to use in the histogram generation.
    - bandwidth: the bandwidths for kernel density estimation.
    - grid_min: the lower bounds for the grid.
    - grid_max: the upper bounds for the grid.
    - grid_bins: the number of bins for the grid.
    - grid_space: the approximate grid spacing for the grid.
    - timeinterval: Simulation time interval to generate the distribution.
    - atoms: the 4 atom index of the molecular forcefield object used to generate the set of torsional angles
    - molecule: the molecular forcefield object from which atoms are selected

    Methods:\n
    - help(): Print attributes and methods
    - set_atoms(atoms, molecule): Select the 4 atom index from the Molecule obj to generate the set of torsional angles
    - generate_input(simulation, bash_script=True): Generate the plumed input files
    - get_results(simulation, crystal='all', plot=True): Check if the plumed driver analysis is ended and store results
    """

    _type = "Planes"
    _short_type = "planes"
    _plumed_version = "master"
    _clustering_type = "distribution"

    def __init__(self, name: str, plumed: str):
        """
        Generates a distribution of the torsional angles of the selected atoms.
        :param name: str, name of the collective variable. Default output and variables will have this name.
        :param plumed: command line for the plumed file.
        """
        super().__init__(name=name,
                         cv_type="Planes",
                         cv_short_type="planes",
                         plumed=plumed,
                         plumed_version="master",
                         clustering_type="distribution",
                         kernel="GAUSSIAN",
                         bandwidth=0.25,
                         grid_bins=73,
                         grid_min=-np.pi,
                         grid_max=np.pi,
                         grid_space=2 * np.pi / 73,
                         timeinterval=200)

        self._atoms = list()
        self._molecule = None

        self._r_0 = 0.1
        self._d_0 = 2.0
        self._d_max = 2.5
        self._normalization = "false"

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        if len(atoms) == 4:
            self._atoms = atoms
        else:
            print("Error: Planes needs 4 atoms as input")

    @property
    def molecule(self):
        return self._molecule

    @molecule.setter
    def molecule(self, molecule):
        self._molecule = molecule

    @staticmethod
    def help():
        # TODO Modify from torsions to planes
        return """
Calculate the distribution of a set of torsional angles.
It creates the inputs for plumed and stores the results.

Attributes:
- name: name of the CV.
- type: Type of the CV (Torsional Angle).
- plumed: Command line for plumed.
- clustering_type: How is it treated by clustering algorithms (distribution). 
- kernel: kernel function to use in the histogram generation. It can be "TRIANGULAR" or "GAUSSIAN"
- bandwidth: the bandwidths for kernel density estimation. The bin size must be smaller than half the bandwidth.
- grid_min: the lower bounds for the grid.
- grid_max: the upper bounds for the grid.
- grid_bins: the number of bins for the grid.
- grid_space: the approximate grid spacing for the grid.
- timeinterval: Simulation time interval to generate the distribution.
                If a single value is given, t, frames corresponding to the last "t" picoseconds are used.
                If two values are given, t1 and t2, frames from time t1 to time t2 are used.
- atoms: the 4 atom index of the molecular forcefield object used to generate the set of torsional angles.
                The same torsional angle in each molecule of the crystal will be considered for the distribution.
- molecule: the molecular forcefield object from which atoms are selected.

Methods:
- help(): Print attributes and methods
- set_atoms(atoms, molecule): Select the 4 atom index from the Molecule obj to generate the set of torsional angles. 
                The atom index in PyPol starts from 0 and can be seen in the
- generate_input(simulation, bash_script=True): Generate the plumed input files
- get_results(simulation, crystal='all', plot=True): Check if the plumed driver analysis is ended and store results
                If crystal="all", results are stored for all crystals. Alternatively, you can select a subset of 
                crystals by specifying their IDs in an iterable object.
                If plot=True, a plot of the distribution is created. This could be slow for large sets.

Examples:
- Select atoms of the torsional angles and create plumed inputs:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
molecule = gaff.get_molecule("MOL")                           # Use molecular forcefield info for the CV 
tor = gaff.get_cv("tor")                                      # Retrieve the CV Object
tor.set_atoms((0, 1, 2, 3), molecule)                         # Use the first four atoms to define the torsional angle
npt = gaff.get_simulation("npt")                              # Retrieve a completed simulation
tor.generate_input(npt)                                       # Generate plumed driver input for the selected simulation
project.save()                                                # Save project

- Import distributions once the plumed driver analysis is finished:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
molecule = gaff.get_molecule("MOL")                           # Use molecular forcefield info for the CV 
tor = gaff.get_cv("tor")                                      # Retrieve the CV Object
npt = gaff.get_simulation("npt")                              # Retrieve a completed simulation
tor.get_results(npt, plot=False)                              # Generate plumed driver input for the selected simulation
project.save()                                                # Save project"""

    def set_atoms(self, atoms: Union[list, tuple], molecule: Molecule):
        """
        Select atom indices of the reference molecule. This is used to identify the torsions of each molecule in the
        crystal.
        :param atoms: list, Atom indices. All atoms indices are available in the project output file after the topology
        is defined.
        :param molecule: obj, Reference molecule
        :return:
        """
        self.atoms = atoms
        self.molecule = molecule

    def check_attributes(self):
        if not self._atoms:
            print("Error: no atoms found. select atoms with the set_atoms module.")
            exit()

    def generate_input(self, crystal, input_name="", output_name=""):
        lines_atoms = generate_atom_list(self._atoms, self._molecule, crystal, keyword="MOL", lines=[],
                                         attributes=self._matt)
        file_plumed = open(input_name, "w")
        file_plumed.write("PLANES ...\n")
        for line in lines_atoms:
            file_plumed.write(line)

        file_plumed.write("LABEL=planes_{0._name}\n... PLANES\n\n"
                          "int_tor_{0._name}: INTERMOLECULARTORSIONS MOLS=planes_{0._name} "
                          "SWITCH={{RATIONAL R_0={0._r_0} D_0={0._d_0} D_MAX={0._d_max}}}\n"
                          "hist_{0._name}: HISTOGRAM DATA=int_tor_{0._name} GRID_MIN={0._grid_min:.3f} "
                          "GRID_MAX={0._grid_max:.3f} BANDWIDTH={0._bandwidth:.3f} "
                          "GRID_BIN={0._grid_bins} KERNEL={0._kernel} NORMALIZATION={0._normalization}\n"
                          "DUMPGRID GRID=hist_{0._name} FILE={1}\n"
                          "".format(self, output_name))
        file_plumed.close()

    def get_from_file(self, crystal, input_file, output_label="", plot=True):
        cv = np.genfromtxt(input_file, skip_header=1)[:, 1]
        if np.isnan(cv).any():
            print(f"\nError: NaN values present in final distribution of crystal {crystal._name}. Check {input_file} ")
            exit()

        cv /= cv.sum()
        # Save output and plot distribution
        x = np.linspace(self._grid_min, self._grid_max, len(cv))
        np.savetxt(os.path.dirname(input_file) + "/plumed_{}_{}_data.dat".format(output_label, self._name),
                   np.column_stack((x, cv)), fmt=("%1.3f", "%1.5f"),
                   header="Angle ProbabilityDensity")
        if plot:
            plt.plot(x, cv, "-")
            plt.xlabel("Intermolecular Angle / rad")
            plt.xlim(self._grid_min, self._grid_max)
            plt.ylabel("Probability Density")
            plt.title(crystal._name)
            plt.savefig(os.path.dirname(input_file) + "/plumed_{}_{}_plot.png".format(output_label, self._name),
                        dpi=300)
            plt.close("all")

        return cv

    def __str__(self):
        txt = super(Planes, self).__str__()
        if self._atoms:
            txt += "\nAtoms:  "
            for atom in self._atoms:
                txt += "{}({})  ".format(atom, self._molecule._atoms[atom]._label)
        else:
            txt += "No atoms found in CV {}. Select atoms with the 'set_atoms' module.\n".format(self._name)
        txt += "\n"
        return txt


class RDF(_Distribution):
    """
    Generates a distribution of the intermolecular torsional angles of the selected atoms.
    Attributes:\n
    - name: name of the CV.
    - type: Type of the CV.
    - plumed: Command line for plumed.
    - clustering_type: How is it treated by clustering algorithms.
    - kernel: kernel function to use in the histogram generation.
    - bandwidth: the bandwidths for kernel density estimation.
    - grid_min: the lower bounds for the grid.
    - grid_bins: the number of bins for the grid.
    - grid_space: the approximate grid spacing for the grid.
    - timeinterval: Simulation time interval to generate the distribution.
    - atoms: the atoms index of the molecular forcefield object used to calculate molecule position
    - molecules: the molecular forcefield object from which atoms are selected
    - center: Calculate the molecule position based on geometrical center or center of mass
    - r_0: R_0 parameter

    Methods:\n
    - help(): Print attributes and methods
    - set_atoms(atoms, molecule): Select the atoms from the Molecule obj to generate calculate the molecule position
    - generate_input(simulation, bash_script=True): Generate the plumed input files
    - get_results(simulation, crystal='all', plot=True): Check if the plumed driver analysis is ended and store results
    """

    _type = "Radial Distribution Function"
    _short_type = "rdf"
    _plumed_version = "master"
    _clustering_type = "distribution"

    def __init__(self, name, plumed, center="geometrical"):
        """

        :param name:
        :param center:
        """
        super().__init__(name=name,
                         cv_type="Radial Distribution Function",
                         cv_short_type="rdf",
                         plumed=plumed,
                         plumed_version="master",
                         clustering_type="distribution",
                         kernel="GAUSSIAN",
                         grid_space=0.01,
                         bandwidth=0.01,
                         timeinterval=200)

        self._center = center
        self._atoms = list()
        self._molecules = list()

        self._switching_function = "RATIONAL"
        self._r_0 = 0.01

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, center):
        if center.lower() in ("com", "geometrical"):
            self._center = center.lower()
        else:
            print("Method to evaluate molecule position, not available, choose between:"
                  "\n - 'com': Calculate the center of mass of each molecule."
                  "\n - 'geometrical': Calculate the centroid of each molecule.")

    @property
    def r_0(self):
        return self._r_0

    @r_0.setter
    def r_0(self, r_0=None):
        if not r_0:
            self._r_0 = self._grid_space
        elif r_0 > 0:
            self._r_0 = r_0
        else:
            print("R_0 cannot have values lower than 0.")

    @property
    def atoms(self):
        return self._atoms

    @property
    def molecules(self):
        return self._molecules

    def __str__(self):
        txt = super().__str__()
        if self._atoms:
            for idx_mol in range(len(self._molecules)):
                txt += "\nMolecule '{}': ".format(self._molecules[idx_mol]._residue)
                for atom in self._atoms[idx_mol]:
                    txt += "{}({})  ".format(atom, self._molecules[idx_mol]._atoms[atom]._label)
                txt += "\n"
        else:
            txt += "No atoms found in CV {}. Select atoms with the 'set_atoms' module.".format(self._name)
        return txt

    @staticmethod
    def help():
        return """
Calculate the radial distribution function of the molecules
Attributes:\n
- name: name of the CV.
- type: Type of the CV.
- plumed: Command line for plumed.
- clustering_type: How is it treated by clustering algorithms.
- kernel: kernel function to use in the histogram generation.
- bandwidth: the bandwidths for kernel density estimation.
- grid_min: the lower bounds for the grid.
- grid_bins: the number of bins for the grid.
- grid_space: the approximate grid spacing for the grid.
- timeinterval: Simulation time interval to generate the distribution.
- atoms: the atoms index of the molecular forcefield object used to calculate molecule position
- molecules: the molecular forcefield object from which atoms are selected
- center: Calculate the molecule position based on geometrical center or center of mass
- r_0: R_0 parameter

Methods:\n
- help(): Print attributes and methods
- set_atoms(atoms="all", molecule): Select the atoms from the Molecule obj to generate calculate the molecule position.
                You can use atoms="all" to select all atoms or atoms="non-hydrogen" to select all atoms except H.
- generate_input(simulation, bash_script=True): Generate the plumed input files for the distribution of distances.
- get_results(simulation, crystal='all', plot=True): Check if the plumed driver analysis is ended and calculate the 
                radial distribution function from the distribution of distances.

Examples:\n
- Generate inputs for the RDF of the centers of mass:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
rdf = gaff.get_cv("rdf")                                      # Retrieve the CV Object
rdf.center = "com"                                            # Select how to calculate molecules position
molecule = gaff.get_molecule("MOL")                           # Use molecular forcefield info for the CV 
rdf.set_atoms("all", molecule)                                # Set all the atoms in the molecule
npt = gaff.get_simulation("npt")                              # Retrieve a completed simulation
rdf.generate_input(npt)                                       # Generate plumed driver input for the selected simulation
project.save()                                                # Save project

- Generate inputs for the RDF of molecule geometrical center:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
rdf = gaff.get_cv("rdf")                                      # Retrieve the CV Object
rdf.center = "geometrical"                                    # Select how to calculate molecules position
molecule = gaff.get_molecule("MOL")                           # Use molecular forcefield info for the CV 
rdf.set_atoms("non-hydrogen", molecule)                       # Select the atoms in the molecule
npt = gaff.get_simulation("npt")                              # Retrieve a completed simulation
rdf.generate_input(npt)                                       # Generate plumed driver input for the selected simulation
project.save()                                                # Save project

- Import distributions once the plumed driver analysis is finished:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
molecule = gaff.get_molecule("MOL")                           # Use molecular forcefield info for the CV 
rdf = gaff.get_cv("rdf")                                      # Retrieve the CV Object
npt = gaff.get_simulation("npt")                              # Retrieve a completed simulation
rdf.get_results(npt, plot=False)                              # Generate plumed driver input for the selected simulation
project.save()                                                # Save project
    """

    def set_atoms(self, atoms: str, molecule: Molecule, overwrite: bool = True):
        """
        Select the atoms from the Molecule obj to generate calculate the molecule position.
        :param atoms: str. You can use atoms="all" to select all atoms or atoms="non-hydrogen"
                      to select all atoms except H.
        :param molecule: Molecular forcfield Molecule object
        :param overwrite: If True, ignores previous atom settings
        :return:
        """
        for idx_mol in range(len(self._molecules)):
            ref_mol = self._molecules[idx_mol]
            if ref_mol._residue == molecule._residue and overwrite:
                del self._molecules[idx_mol]
                del self._atoms[idx_mol]

        if atoms == "all":
            atoms = list()
            for atom in molecule._atoms:
                atoms.append(atom._index)
        elif atoms == "non-hydrogen":
            atoms = list()
            for atom in molecule._atoms:
                if atom._element.upper() != "H":
                    atoms.append(atom._index)
        self._atoms.append(list(atoms))
        self._molecules.append(molecule)

    def del_atoms(self, index: Union[str, int] = "all"):
        if index == "all":
            self._atoms.clear()
            self._molecules.clear()
        elif isinstance(index, int):
            del self._atoms[index]
            del self._molecules[index]
        else:
            print("Error: not clear which set of atoms you want to delete.")

    def check_attributes(self):
        if not self._atoms:
            print("Error: no atoms found. Select atoms with the set_atoms module.")
            exit()

    def generate_input(self, crystal, input_name="", output_name=""):
        d_max = 0.5 * np.min(np.array([crystal._box[0, 0], crystal._box[1, 1], crystal._box[2, 2]]))
        nbins = int(round((d_max - self._r_0) / self._grid_space, 0))

        lines_atoms = []
        for idx_mol in range(len(self._molecules)):
            lines_atoms = generate_atom_list(self._atoms[idx_mol], self._molecules[idx_mol], crystal,
                                             keyword="ATOMS", lines=lines_atoms, index_lines=False,
                                             attributes=self._matt)

        file_plumed = open(input_name, "w")
        idx_com = 1
        str_group = ""
        if self._center == "geometrical":
            for line in lines_atoms:
                file_plumed.write("{}_c{}: CENTER {}".format(self._name, idx_com, line))
                str_group += "{}_c{},".format(self._name, idx_com)
                idx_com += 1
        elif self._center.upper() == "COM":
            for line in lines_atoms:
                file_plumed.write("{}_c{}: COM {}".format(self._name, idx_com, line))
                str_group += "{}_c{},".format(self._name, idx_com)
                idx_com += 1

        str_group = str_group[:-1]
        file_plumed.write("{0}_g: GROUP ATOMS={1}\n"
                          "{0}_d: DISTANCES GROUP={0}_g MORE_THAN={{RATIONAL R_0={2} D_0={3} D_MAX={3}}} "
                          "HISTOGRAM={{{6} NBINS={5} BANDWIDTH={4} UPPER={3} LOWER={2}}}\n"
                          "PRINT ARG={0}_d.* FILE={7}\n\n"
                          "".format(self._name, str_group, self._r_0, d_max, self._bandwidth,
                                    nbins, self._kernel, output_name))
        file_plumed.close()

    def get_from_file(self, crystal, input_file, output_label="", plot=True):
        dn_r = np.genfromtxt(input_file, skip_header=1)
        if dn_r.ndim == 2:
            dn_r = dn_r[:, 2:]
            if np.isnan(dn_r).any():
                dn_r = np.nanmean(dn_r, axis=0)
                if np.isnan(dn_r).any():
                    print(f"\nError: NaN values present in final distribution of crystal {crystal._name}. "
                          f"Check {input_file} ")
                    exit()
                print(f"\nWarning: NaN values present in some frames of crystal {crystal._name}. Check {input_file} ")
            else:
                dn_r = np.average(dn_r, axis=0)
        else:
            dn_r = dn_r[2:]
            if np.isnan(dn_r).any():
                print(f"\nError: NaN values present in final distribution of crystal {crystal._name}. "
                      f"Check {input_file} ")
                exit()

        d_max = 0.5 * np.min(np.array([crystal._box[0, 0], crystal._box[1, 1], crystal._box[2, 2]]))
        nbins = int(round((d_max - self._r_0) / self._grid_space, 0))
        r = np.linspace(self._r_0, d_max, nbins)
        rho = crystal._density

        cv = np.where(r > 0, dn_r / (4 * np.pi * rho * r ** 2 * self._grid_space) / crystal._Z * 2.0, 0.)
        # Save output and plot distribution
        np.savetxt(os.path.dirname(input_file) + "/plumed_{}_{}_data.dat".format(output_label, self._name),
                   np.column_stack((r, cv)), fmt=("%1.4f", "%1.5f"),
                   header="r RDF")
        if plot:
            plt.plot(r, cv, "-")
            plt.xlabel("r / nm")
            plt.xlim(self._r_0, d_max)
            plt.ylabel("Probability Density")
            plt.savefig(os.path.dirname(input_file) + "/plumed_{}_{}_plot.png".format(output_label, self._name),
                        dpi=300)
            plt.close("all")
        return cv


class Combine(_Distribution):
    """
    Combine torsional or intermolecular torsional angles in multidimensional distributions.

    Attributes:\n
    - name: name of the CV.
    - type: Type of the CV.
    - clustering_type: How is it treated by clustering algorithms.
    - kernel: kernel function to use in the histogram generation.
    - timeinterval: Simulation time interval to generate the distribution.
    - cvs: List CVs names

    Methods:\n
    - help(): Print attributes and methods
    - generate_input(simulation, bash_script=True): Generate the plumed input files
    - get_results(simulation, crystal='all', plot=True): Check if the plumed driver analysis is ended and store results
    """

    def __init__(self, name: str, cvs: iter):
        """
        Combine torsional or intertorsional angles in multidimensional distributions.\n
        :param name: CV name
        :param cvs: list of 1-D CVs objects
        """
        super().__init__(name=name,
                         cv_type="{} ({}D)".format(cvs[0]._type, len(cvs)),
                         cv_short_type="nd-dist",
                         plumed=cvs[0]._plumed,
                         plumed_version=cvs[0]._plumed_version,
                         kernel=cvs[0]._kernel,
                         timeinterval=cvs[0]._timeinterval)

        # CVs properties
        self._cvs = cvs
        self._grid_min = []
        self._grid_max = []
        self._grid_bins = []
        self._bandwidth = []

        self._str_grid_min = ""
        self._str_grid_max = ""
        self._str_grid_bins = ""
        self._str_bandwidth = ""
        self._str_args = ""
        idx_cv = 0
        for cv in self._cvs:
            self._grid_min.append(cv._grid_min)
            self._grid_max.append(cv._grid_max)
            self._grid_bins.append(cv._grid_bins)
            self._bandwidth.append(cv._bandwidth)
            self._str_grid_min += "{:.3f},".format(cv.grid_min)
            self._str_grid_max += "{:.3f},".format(cv.grid_max)
            self._str_bandwidth += "{:.3f},".format(cv.bandwidth)

            # if self._type.startswith("Molecular Orientation"):
            #     self._str_grid_bins += "{},".format(cv.grid_bins + 1)
            # else:
            #     self._str_grid_bins += "{},".format(cv.grid_bins)
            self._str_grid_bins += "{},".format(cv.grid_bins)

            if self._type.startswith("Molecular Orientation"):
                self._str_args += "ARG{}=ang_mat_{} ".format(idx_cv + 1, cv._name)
            else:
                self._str_args += "ARG{}={} ".format(idx_cv + 1, cv._name)
            idx_cv += 1

    @property
    def cvs(self):
        txt = "CVs:\n"
        for cv in self._cvs:
            txt += cv._name + "\n"
        return txt

    def __str__(self):
        txt = ""
        idx_cv = 0
        txt += "\nCV: {} ({})\n".format(self._name, self._type)
        for cv in self._cvs:
            txt += "CV{}: {} ({})\n".format(idx_cv, cv._name, cv._type)
            idx_cv += 1

        txt += "Clustering type: {5}-D Distribution\n" \
               "Parameters: KERNEL={0} NBINS={1} BANDWIDTH={2} UPPER={3} LOWER={4}\n" \
               "".format(self._kernel, self._str_grid_bins, self._str_bandwidth, self._str_grid_max, self._str_grid_min,
                         len(self._cvs))
        return txt

    @staticmethod
    def help():
        return """
Combine torsional or intertorsional angles in multidimensional distributions.

Attributes:\n
- name: name of the CV.
- type: Type of the CV.
- clustering_type: How is it treated by clustering algorithms.
- kernel: kernel function to use in the histogram generation.
- timeinterval: Simulation time interval to generate the distribution.
- cvs: List CVs names

Methods:\n
- help(): Print attributes and methods
- generate_input(simulation, bash_script=True): Generate the plumed input files
- get_results(simulation, crystal='all', plot=True): Check if the plumed driver analysis is ended and store results

Examples:
- Create plumed inputs:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
mo = gaff.get_cv("mo")                                        # Retrieve the CV Object
npt = gaff.get_simulation("npt")                              # Retrieve a completed simulation
mo.generate_input(npt)                                        # Generate plumed driver input for the selected simulation
project.save()                                                # Save project

- Import distributions once the plumed driver analysis is finished:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
molecule = gaff.get_molecule("MOL")                           # Use molecular forcefield info for the CV 
mo = gaff.get_cv("mo")                                        # Retrieve the CV Object
npt = gaff.get_simulation("npt")                              # Retrieve a completed simulation
mo.get_results(npt, plot=False)                               # Generate plumed driver input for the selected simulation
project.save()                                                # Save project"""

    def check_attributes(self):
        self._grid_min = []
        self._grid_max = []
        self._grid_bins = []
        self._bandwidth = []

        self._str_grid_min = ""
        self._str_grid_max = ""
        self._str_grid_bins = ""
        self._str_bandwidth = ""
        self._str_args = ""
        idx_cv = 0
        for cv in self._cvs:
            if not cv._atoms:
                print("Error: no atoms found in CV {}. select atoms with the set_atoms module.".format(cv._name))
                exit()
            self._grid_min.append(cv._grid_min)
            self._grid_max.append(cv._grid_max)
            self._grid_bins.append(cv._grid_bins)
            self._bandwidth.append(cv._bandwidth)
            self._str_grid_min += "{:.3f},".format(cv.grid_min)
            self._str_grid_max += "{:.3f},".format(cv.grid_max)
            self._str_bandwidth += "{:.3f},".format(cv.bandwidth)

            # if self._type.startswith("Molecular Orientation"):
            #     self._str_grid_bins += "{},".format(cv.grid_bins + 1)
            # else:
            #     self._str_grid_bins += "{},".format(cv.grid_bins)
            self._str_grid_bins += "{},".format(cv.grid_bins)
            if self._type.startswith("Molecular Orientation"):
                self._str_args += "ARG{}=ang_mat_{} ".format(idx_cv + 1, cv._name)
            else:
                self._str_args += "ARG{}={} ".format(idx_cv + 1, cv._name)
            idx_cv += 1

    def generate_input(self, crystal, input_name="", output_name=""):
        if self._type.startswith("Molecular Orientation"):
            file_plumed = open(input_name, "w")
            for cv in self._cvs:
                # Select atoms and molecules
                lines_atoms = []
                for idx_mol in range(len(cv._molecules)):
                    lines_atoms = generate_atom_list(cv._atoms[idx_mol], cv._molecules[idx_mol], crystal,
                                                     keyword="ATOMS", lines=lines_atoms, attributes=self._matt)

                file_plumed.write("DISTANCE ...\n")
                for line in lines_atoms:
                    file_plumed.write(line)
                file_plumed.write("LABEL=dd_{0}\n"
                                  "COMPONENTS\n"
                                  "... DISTANCE\n\n"
                                  "vv_{0}: NORMALIZE ARG1=dd_{0}.x ARG2=dd_{0}.y ARG3=dd_{0}.z\n"
                                  "dp_mat_{0}: DOTPRODUCT_MATRIX GROUP1=vv_{0}.x GROUP2=vv_{0}.y GROUP3=vv_{0}.z\n"
                                  "ang_mat_{0}: MATHEVAL ARG1=dp_mat_{0} FUNC=acos(x) PERIODIC=NO\n\n"
                                  "".format(cv._name))

            file_plumed.write("valg_{0}: KDE {7} GRID_MIN={1} GRID_MAX={2} "
                              "GRID_BIN={3} BANDWIDTH={4} KERNEL={5}\n\n"
                              "PRINT ARG=valg_{0} FILE={6}\n"
                              "".format(self._name, self._str_grid_min, self._str_grid_max,
                                        self._str_grid_bins, self._str_bandwidth, self._kernel, output_name,
                                        self._str_args))
            file_plumed.close()

        if self._type.startswith("Torsional Angle"):
            file_plumed = open(input_name, "w")
            for cv in self._cvs:
                # Select atoms and molecules
                lines_atoms = generate_atom_list(cv._atoms, cv.molecule, crystal, keyword="ATOMS", lines=[],
                                                 attributes=self._matt)

                file_plumed.write("TORSIONS ...\n")
                for line in lines_atoms:
                    file_plumed.write(line)

                file_plumed.write("LABEL={0}\n... TORSIONS\n\n".format(cv._name))
            file_plumed.write("kde_{0}: KDE {7} GRID_MIN={1} GRID_MAX={2} "
                              "GRID_BIN={3} BANDWIDTH={4} KERNEL={5}\n\n"
                              "PRINT ARG=kde_{0} FILE={6}\n"
                              "".format(self._name, self._str_grid_min, self._str_grid_max,
                                        self._str_grid_bins, self._str_bandwidth, self._kernel, output_name,
                                        self._str_args))
            file_plumed.close()

    def get_from_file(self, crystal, input_file, output_label="", plot=True):
        cv_dist = np.genfromtxt(input_file, skip_header=1)
        if cv_dist.ndim == 2:
            cv_dist = cv_dist[:, 1:]
            if np.isnan(cv_dist).any():
                cv_dist = np.nanmean(cv_dist, axis=0)
                if np.isnan(cv_dist).any():
                    print("\nError: NaN values present in final distribution of crystal {0._name}. Check {0._path} "
                          "".format(crystal))
                    exit()
                print("\nWarning: NaN values present in some frames of crystal {0._name}. Check {0._path} "
                      "".format(crystal))
            else:
                cv_dist = np.average(cv_dist, axis=0)
        else:
            cv_dist = cv_dist[1:]
            if np.isnan(cv_dist).any():
                print("\nError: NaN values present in final distribution of crystal {0._name}. Check {0._path} "
                      "".format(crystal))
                exit()
        try:
            cv_dist = cv_dist.reshape(self._grid_bins)
        except ValueError:
            nbins = []
            for b in self._grid_bins:
                nbins.append(b + 1)
            cv_dist = cv_dist.reshape(tuple(nbins))

        if len(self._cvs) == 2:
            np.savetxt(os.path.dirname(input_file) + "/plumed_{}_{}_data.dat".format(output_label, self._name),
                       cv_dist,
                       header="Probability Density Grid.")
            if plot:
                extent = self._cvs[0].grid_min, self._cvs[0].grid_max, self._cvs[1].grid_min, self._cvs[1].grid_max
                plt.imshow(cv_dist, interpolation="nearest", cmap="viridis", extent=extent)
                plt.xlabel("{} / rad".format(self._cvs[0]._name))
                plt.ylabel("{} / rad".format(self._cvs[1]._name))
                plt.savefig(os.path.dirname(input_file) + "/plumed_{}_{}_plot.png".format(output_label, self._name),
                            dpi=300)
                plt.close("all")
        else:
            np.save(os.path.dirname(input_file) + "/plumed_{}_{}_data.npy".format(output_label, self._name),
                    crystal._cvs[self._name])

        return cv_dist


class _OwnDistributions(_Distribution):
    """
    General Class for Collective Variables.
    Attributes:\n
    - name: name of the CV.
    - type: Type of the CV.
    - clustering_type: How is it treated by clustering algorithms.
    - kernel: kernel function to use in the histogram generation.
    - bandwidth: the bandwidths for kernel density estimation.
    - grid_min: the lower bounds for the grid.
    - grid_max: the upper bounds for the grid.
    - grid_bins: the number of bins for the grid.
    - grid_space: the approximate grid spacing for the grid.
    - timeinterval: Simulation time interval to generate the distribution.
    """

    def __init__(self, name: str, cv_type: str, cv_short_type: str, clustering_type="distribution",
                 kernel="gaussian", timeinterval: Union[int, float, tuple] = None):
        """
        General Class for Collective Variables.
        :param name: name of the CV.
        :param cv_type: Type of the CV.
        :param clustering_type: How is it treated by clustering algorithms.
        :param kernel: kernel function to use in the histogram generation.
        :param timeinterval: Simulation time interval to generate the distribution.
        """
        super().__init__(name=name, cv_type=cv_type, cv_short_type=cv_short_type, plumed=None, plumed_version=None,
                         clustering_type=clustering_type, kernel=kernel, timeinterval=timeinterval)

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, kernel: str):
        if kernel.lower() in ("gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"):
            self._kernel = kernel
        else:
            print('Kernel function not recognized. Available formats from sklearn: "gaussian", "tophat", '
                  '"epanechnikov", "exponential", "linear", "cosine".')

    def __str__(self):
        txt = """
CV: {0._name} ({0._type})
Clustering Type: {0._clustering_type}""".format(self)
        return txt

    @staticmethod
    def _kde_ovect_rvect(data, r_grid_min=0., r_grid_max=4., r_bw=0.05, r_bins=100j,
                         o_grid_min=0., o_grid_max=np.pi, o_bw=0.05, o_bins=100j, mirror=False):
        kde = KDE()
        data_scaled = data / np.array([o_bw, r_bw])
        kde.fit(data_scaled)
        xx, yy = np.mgrid[int(o_grid_min / o_bw):int(o_grid_max / o_bw):o_bins,
                 int(r_grid_min / r_bw):int(r_grid_max / r_bw): r_bins]
        zz = np.reshape(np.exp(kde.score_samples(np.vstack([xx.ravel(), yy.ravel()]).T)), xx.shape).T
        if mirror:
            zz = (zz + np.flip(zz, axis=1))
            zz = zz[:, :int(zz.shape[1] / 2)]
        return zz

    def generate_input(self, crystal, input_name="", output_label=""):
        pass

    def generate_inputs(self,
                        simulation: Union[EnergyMinimization, CellRelaxation, MolecularDynamics, Metadynamics],
                        bash_script=True,
                        crystals="all",
                        catt=None):
        print("Info: No plumed input needed to generate this distribution")

    def get_results(self,
                    simulation: Union[EnergyMinimization, CellRelaxation, MolecularDynamics, Metadynamics],
                    crystals: Union[str, list, tuple] = "all",
                    plot: bool = True,
                    catt=None,
                    suffix=""):
        """
        Verify if the distribution has been correctly generated and store the result. If the distribution is taken over
        different frames, the average is calculated.
        :param simulation: Simulation object
        :param crystals: It can be either "all", use all non-melted Crystal objects from the previous simulation or
                         "centers", use only cluster centers from the previous simulation. Alternatively, you can select
                         a specific subset of crystals by listing crystal names.
        :param plot: If true, generate a plot of the distribution.
        :param catt: Use crystal attributes to select the crystal list
        :param suffix: suffix to add to the cv name.
        :return:
        """
        list_crystals = get_list_crystals(simulation._crystals, crystals, catt)
        print("\n" + str(self._name))
        bar = progressbar.ProgressBar(maxval=len(list_crystals)).start()
        nbar = 1
        for crystal in list_crystals:
            os.chdir(crystal._path)

            traj_start, traj_end = (None, None)
            traj_time = int(float(simulation._mdp["dt"]) * float(simulation._mdp["nsteps"]))
            if isinstance(self._timeinterval, tuple):
                traj_start = self._timeinterval[0]
                traj_end = self._timeinterval[1]
            elif isinstance(self._timeinterval, (int, float)):
                traj_start = traj_time - self._timeinterval
                traj_end = traj_time
            else:
                print("Error: No suitable time interval.")
                exit()

            os.system('{0} trjconv -f {1}.xtc -o PYPOL_TMP_{1}.xtc -s {1}.tpr -b {2} -e {3} <<< '
                      '0 &> /dev/null '.format(simulation._gromacs, simulation._name, traj_start, traj_end))

            if os.path.exists(f"PYPOL_TMP_{simulation._name}.xtc"):
                crystal._cvs[self._name + suffix] = self.gen_from_traj(
                    crystal=crystal,
                    simulation=simulation,
                    input_traj=crystal._path + f"PYPOL_TMP_{simulation._name}.xtc",
                    output_label=simulation._name,
                    plot=plot)
                bar.update(nbar)
                nbar += 1
            else:
                print("An error has occurred with PyPol. Check file {}.xtc in folder {}."
                      "".format(simulation._name, crystal._path))
        bar.finish()

    def gen_from_traj(self, crystal, simulation, input_traj, output_label="", plot=True):
        pass

    def get_from_file(self, crystal, input_file, output_label="", plot=True):
        print(f"Info: This distribution is generated directly from the trajectory. No plumed output needed")


# Change Name in DistancesPlanes
class RDFPlanes(_OwnDistributions):
    """
    TODO Change docstrings
    Generates a distribution of the torsional angles of the selected atoms.
    Attributes:\n
    - name: name of the CV.
    - type: Type of the CV.
    - plumed: Command line for plumed.
    - clustering_type: How is it treated by clustering algorithms.
    - kernel: kernel function to use in the histogram generation.
    - bandwidth: the bandwidths for kernel density estimation.
    - grid_min: the lower bounds for the grid.
    - grid_max: the upper bounds for the grid.
    - grid_bins: the number of bins for the grid.
    - grid_space: the approximate grid spacing for the grid.
    - timeinterval: Simulation time interval to generate the distribution.
    - atoms: the 4 atom index of the molecular forcefield object used to generate the set of torsional angles
    - molecule: the molecular forcefield object from which atoms are selected

    Methods:\n
    - help(): Print attributes and methods
    - set_atoms(atoms, molecule): Select the 4 atom index from the Molecule obj to generate the set of torsional angles
    - generate_input(simulation, bash_script=True): Generate the plumed input files
    - get_results(simulation, crystal='all', plot=True): Check if the plumed driver analysis is ended and store results
    """

    _type = "RDF-Planes"
    _short_type = "rdf-planes"
    _plumed_version = None
    _clustering_type = "distribution"

    def __init__(self, name: str):
        """
        Generates a distribution of the torsional angles of the selected atoms.
        :param name: str, name of the collective variable. Default output and variables will have this name.
        """
        super().__init__(name=name,
                         cv_type="RDF-Planes",
                         cv_short_type="rdf-planes",
                         clustering_type="distribution",
                         kernel="gaussian",
                         timeinterval=200)

        self._atoms = list()
        self._molecule = None

        self._r_grid_min = 0.
        self._r_grid_max = None
        self._r_bw = 0.05
        self._r_bins = 100j
        self._r_grid_space = 0.01
        self._o_grid_min = 0.
        self._o_grid_max = np.pi
        self._o_bw = 0.05
        self._o_bins = 72j
        self._mirror = False

    @property
    def rdf_grid_min(self):
        return self._r_grid_min

    @rdf_grid_min.setter
    def rdf_grid_min(self, value: float):
        self._r_grid_min = value

    @property
    def rdf_grid_max(self):
        return self._r_grid_max

    @rdf_grid_max.setter
    def rdf_grid_max(self, value: float):
        self._r_grid_max = value

    @property
    def rdf_grid_bins(self):
        return self._r_bins

    @rdf_grid_bins.setter
    def rdf_grid_bins(self, value: Union[float, int, complex]):
        if isinstance(value, (float, int)):
            value = complex(0, value)
        self._r_grid_space = (self._r_grid_max - self._r_grid_max) / value.imag
        self._r_bins = value

    @property
    def rdf_bandwidth(self):
        return self._r_bw

    @rdf_bandwidth.setter
    def rdf_bandwidth(self, value: float):
        self._r_bw = value

    @property
    def planes_grid_min(self):
        return self._o_grid_min

    @planes_grid_min.setter
    def planes_grid_min(self, value: float):
        self._o_grid_min = value

    @property
    def planes_grid_max(self):
        return self._o_grid_max

    @planes_grid_max.setter
    def planes_grid_max(self, value: float):
        self._o_grid_max = value

    @property
    def planes_grid_bins(self):
        return self._o_bins

    @planes_grid_bins.setter
    def planes_grid_bins(self, value: Union[float, int, complex]):
        if isinstance(value, (float, int)):
            value = complex(0, value)
        self._o_bins = value

    @property
    def planes_bandwidth(self):
        return self._o_bw

    @planes_bandwidth.setter
    def planes_bandwidth(self, value: float):
        self._o_bw = value

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        if len(atoms) == 3:
            self._atoms = atoms
        else:
            print("Error: RDF-Planes needs 3 atoms as input")

    @property
    def molecule(self):
        return self._molecule

    @molecule.setter
    def molecule(self, molecule):
        self._molecule = molecule

    @property
    def mirror(self):
        return self._mirror

    @mirror.setter
    def mirror(self, value: bool):
        self._mirror = value

    @staticmethod
    def help():
        # TODO Modify from torsions to planes
        return """
Calculate the distribution of a set of torsional angles.
It creates the inputs for plumed and stores the results.

Attributes:
- name: name of the CV.
- type: Type of the CV (Torsional Angle).
- plumed: Command line for plumed.
- clustering_type: How is it treated by clustering algorithms (distribution). 
- kernel: kernel function to use in the histogram generation. It can be "TRIANGULAR" or "GAUSSIAN"
- bandwidth: the bandwidths for kernel density estimation. The bin size must be smaller than half the bandwidth.
- grid_min: the lower bounds for the grid.
- grid_max: the upper bounds for the grid.
- grid_bins: the number of bins for the grid.
- grid_space: the approximate grid spacing for the grid.
- timeinterval: Simulation time interval to generate the distribution.
                If a single value is given, t, frames corresponding to the last "t" picoseconds are used.
                If two values are given, t1 and t2, frames from time t1 to time t2 are used.
- atoms: the 4 atom index of the molecular forcefield object used to generate the set of torsional angles.
                The same torsional angle in each molecule of the crystal will be considered for the distribution.
- molecule: the molecular forcefield object from which atoms are selected.

Methods:
- help(): Print attributes and methods
- set_atoms(atoms, molecule): Select the 4 atom index from the Molecule obj to generate the set of torsional angles. 
                The atom index in PyPol starts from 0 and can be seen in the
- generate_input(simulation, bash_script=True): Generate the plumed input files
- get_results(simulation, crystal='all', plot=True): Check if the plumed driver analysis is ended and store results
                If crystal="all", results are stored for all crystals. Alternatively, you can select a subset of 
                crystals by specifying their IDs in an iterable object.
                If plot=True, a plot of the distribution is created. This could be slow for large sets.

Examples:
- Select atoms of the torsional angles and create plumed inputs:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
molecule = gaff.get_molecule("MOL")                           # Use molecular forcefield info for the CV 
tor = gaff.get_cv("tor")                                      # Retrieve the CV Object
tor.set_atoms((0, 1, 2, 3), molecule)                         # Use the first four atoms to define the torsional angle
npt = gaff.get_simulation("npt")                              # Retrieve a completed simulation
tor.generate_input(npt)                                       # Generate plumed driver input for the selected simulation
project.save()                                                # Save project

- Import distributions once the plumed driver analysis is finished:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
molecule = gaff.get_molecule("MOL")                           # Use molecular forcefield info for the CV 
tor = gaff.get_cv("tor")                                      # Retrieve the CV Object
npt = gaff.get_simulation("npt")                              # Retrieve a completed simulation
tor.get_results(npt, plot=False)                              # Generate plumed driver input for the selected simulation
project.save()                                                # Save project"""

    def set_atoms(self, atoms: Union[list, tuple], molecule: Molecule):
        """
        Select atom indices of the reference molecule. This is used to identify the torsions of each molecule in the
        crystal.
        :param atoms: list, Atom indices. All atoms indices are available in the project output file after the topology
        is defined.
        :param molecule: obj, Reference molecule
        :return:
        """
        self.atoms = atoms
        self.molecule = molecule

    def gen_from_traj(self, crystal, simulation, input_traj, output_label="cv", plot=True):
        os.chdir(os.path.dirname(input_traj))

        mols = []
        for mol in crystal._load_coordinates():
            if self._molecule._residue == mol._residue:
                if self._matt:
                    if self._matt.items() <= mol._attributes.items():
                        mols.append(mol._index)
                else:
                    mols.append(mol._index)

        if self._r_grid_max:
            crystal_grid_max = self._r_grid_max
            crystal_grid_bins = self._r_bins
        else:
            crystal_grid_max = 0.5 * np.min(np.array([crystal._box[0, 0], crystal._box[1, 1], crystal._box[2, 2]]))
            crystal_grid_bins = complex(0,
                                        int(round((crystal_grid_max - self._r_grid_min) / self._r_grid_space, 0)))

        file_ndx = open(os.path.dirname(input_traj) + f"/PYPOL_TMP_{output_label}.ndx", "w")
        file_ndx.write("[ System ] \n")
        space = 0
        for mol in mols:
            for atom in self._atoms:
                file_ndx.write("{:5} ".format(atom + mol * self._molecule._natoms + 1))
                space += 1
                if space >= 20:
                    file_ndx.write("\n")
                    space = 0
        file_ndx.close()

        os.system('{0} trjconv -f {1} -o PYPOL_TMP_{2}.gro -n PYPOL_TMP_{2}.ndx -s {3}.tpr '
                  '-pbc mol -ur tric -center <<< "2 2"  &> /dev/null'
                  ''.format(simulation._gromacs, input_traj, output_label, simulation._name))

        planes = {}
        r_plane = {}
        box_param = {}
        file_gro = open(os.path.dirname(input_traj) + f"/PYPOL_TMP_{output_label}.gro")
        frame, plane = 0, 0
        for line in file_gro:
            if "t=" in line:
                frame = line.split()[-1]
                planes[frame] = {}
                planes[frame][0] = np.zeros((len(mols), 3))
                r_plane[frame] = {}
                r_plane[frame][0] = np.zeros((len(mols), 3))
                plane = 0
                next(file_gro)
            elif line[5:8] == self._molecule._residue:
                a1 = np.array([float(line[20:28].strip()), float(line[28:36].strip()), float(line[36:44].strip())])
                line = next(file_gro)
                a2 = np.array([float(line[20:28].strip()), float(line[28:36].strip()), float(line[36:44].strip())])
                line = next(file_gro)
                a3 = np.array([float(line[20:28].strip()), float(line[28:36].strip()), float(line[36:44].strip())])
                planes[frame][0][plane, :] = np.cross(a2 - a1, a2 - a3)
                planes[frame][0][plane, :] /= np.linalg.norm(planes[frame][0][plane, :])
                r_plane[frame][0][plane, :] = np.mean([a1, a2, a3], axis=0)
                # vmd arrows:
                # print(frame, "draw arrow {{ {0[0]} {0[1]} {0[2]} }} {{ {1[0]} {1[1]} {1[2]} }}
                # ".format(r_plane[frame][0][plane, :]*10,
                #          (planes[frame][0][plane, :] * 0.2 + r_plane[frame][0][plane, :])*10))
                plane += 1
            else:
                idx_gromacs = [0, 5, 7, 3, 1, 8, 4, 6, 2]
                box_param[frame] = np.array([float(line.split()[ii]) for ii in idx_gromacs]).reshape((3, 3))
        file_gro.close()

        nbox = 0
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for c in (-1, 0, 1):
                    if (a, b, c) != (0, 0, 0):
                        nbox += 1
                        for frame in planes.keys():
                            planes[frame][nbox] = np.zeros((len(mols), 3))
                            r_plane[frame][nbox] = np.zeros((len(mols), 3))
                            for i in range(len(mols)):
                                r_i = r_plane[frame][0][i, :]
                                r_plane[frame][nbox][i, :] = np.sum([r_i,
                                                                     a * box_param[frame][:, 0],
                                                                     b * box_param[frame][:, 1],
                                                                     c * box_param[frame][:, 2], ], axis=0)
                                planes[frame][nbox][i, :] = planes[frame][0][i, :]
        data = np.full((int(len(mols) * (len(mols) - 1) / 2) * len(planes.keys()) * 27, 2), np.nan)
        d = 0
        for frame in planes.keys():
            for i in range(len(mols) - 1):
                ax, ay, az = planes[frame][0][i, :]
                r_i = r_plane[frame][0][i, :]
                for j in range(i + 1, len(mols)):
                    for nbox in range(27):
                        r_j = r_plane[frame][nbox][j, :]
                        distance = np.linalg.norm(r_i - r_j)
                        if self._r_grid_min <= distance <= crystal_grid_max:
                            bx, by, bz = planes[frame][nbox][j, :]
                            angle = np.arccos(
                                (ax * bx + ay * by + az * bz) / np.sqrt(
                                    (ax * ax + ay * ay + az * az) * (bx * bx + by * by + bz * bz)))
                            data[d, :] = np.array([angle, distance])
                        d += 1

        data = data[~np.isnan(data).any(axis=1)]

        cv = super()._kde_ovect_rvect(data,
                                      self._r_grid_min, crystal_grid_max,
                                      self._r_bw, crystal_grid_bins,
                                      self._o_grid_min, self._o_grid_max,
                                      self._o_bw, self._o_bins, mirror=self._mirror)

        r = np.linspace(start=self._r_grid_min, stop=crystal_grid_max, num=int(crystal_grid_bins.imag))
        N = 4 * np.pi * crystal._density * r ** 2 * crystal._Z * (crystal_grid_max - self._r_grid_min) / \
            (crystal_grid_bins.imag * 2.0)
        cv /= N.reshape(-1, 1)
        cv /= np.sum(cv)

        # Save output and plot distribution
        np.savetxt(os.path.dirname(input_traj) + "/pypol_{}_{}_data.dat".format(output_label, self._name),
                   data,
                   header="Angle / rad      Distance / nm")
        np.savetxt(os.path.dirname(input_traj) + "/pypol_{}_{}_data_grid.dat".format(output_label, self._name),
                   cv,
                   header="Probability Density Grid.")
        if plot:
            if self._mirror:
                extent = [self._o_grid_min, self._o_grid_max / 2, crystal_grid_max, self._r_grid_min]
            else:
                extent = [self._o_grid_min, self._o_grid_max, crystal_grid_max, self._r_grid_min]
            plt.imshow(cv, extent=extent, cmap="viridis")
            plt.colorbar()
            # plt.scatter(data[:, 0], data[:, 1], s=1, facecolor=None, edgecolors='white', alpha=0.025)
            plt.xlim(self._o_grid_min, self._o_grid_max)
            plt.ylim(self._r_grid_min, crystal_grid_max)
            plt.ylabel("RDF / nm")
            plt.xlabel("Molecular orientation / rad")
            plt.savefig(os.path.dirname(input_traj) + "/pypol_{}_{}_plot.png".format(output_label, self._name),
                        dpi=300)
            plt.close("all")
        return cv

    def __str__(self):
        txt = super(RDFPlanes, self).__str__()
        if self._r_grid_max:
            txt += f"""
Grid Parameters:
Distances: GRID_MAX={self._r_grid_max} GRID_MIN={self._r_grid_min} GRID_BINS={self._r_bins} BANDWIDTH={self._r_bins}
Planes: GRID_MAX={self._o_grid_max} GRID_MIN={self._o_grid_min} GRID_BINS={self._o_bins} BANDWIDTH={self._o_bw}
"""
        else:
            txt += f"""
Grid Parameters:
Distances: GRID_MIN={self._r_grid_min} GRID_SPACING={self._r_grid_space} BANDWIDTH={self._r_bins}
Planes: GRID_MAX={self._o_grid_max} GRID_MIN={self._o_grid_min} GRID_BINS={self._o_bins} BANDWIDTH={self._o_bw}
"""

        if self._atoms:
            txt += "\nAtoms:  "
            for atom in self._atoms:
                txt += "{}({})  ".format(atom, self._molecule._atoms[atom]._label)
        else:
            txt += "No atoms found in CV {}. Select atoms with the 'set_atoms' module.\n".format(self._name)
        txt += "\n"
        return txt
