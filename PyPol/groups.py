import numpy as np
import progressbar
import pandas as pd
import itertools as its
import copy
from typing import Union

from PyPol.utilities import get_list_crystals, hellinger
from PyPol.fingerprints import Torsions, MolecularOrientation
from PyPol.metad import Density, PotentialEnergy
from PyPol.gromacs import EnergyMinimization, MolecularDynamics, CellRelaxation, Metadynamics


class _GG(object):
    """
    General Class for Groups objects.
    Attributes:\n
    - name: name of the CV.
    - type: Type of the CV.
    - clustering_type: "classification"
    """

    def __init__(self, name: str, gtype: str):
        """
        Generate Groups From Distribution
        :param name:
        :param gtype:
        """

        # Grouping Properties
        self._name = name
        self._type = gtype
        self._clustering_type = "classification"

    @property
    def type(self):
        return self._type

    @property
    def name(self):
        return self._name

    @property
    def clustering_type(self):
        return self._clustering_type

    def _run(self, simulation, groups, crystals="all", catt=None, suffix=""):

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_seq_items', None)

        list_crystals = get_list_crystals(simulation._crystals, crystals, catt)

        cvg = {}
        for i in groups.keys():
            cvg[i] = 0

        for crystal in list_crystals:
            crystal._cvs[self._name + suffix] = copy.deepcopy(cvg)
            for group in groups.keys():
                if crystal._name in groups[group]:
                    crystal._cvs[self._name + suffix][group] += 1
                    break

        file_hd = open("{}/Groups_{}_{}.dat".format(simulation._path_output, self._name, simulation._name), "w")
        file_hd.write("# Group_name             Crystal_IDs\n")
        for group in groups.keys():
            file_hd.write("{:<25}: {}\n".format(str(group), groups[group]))
        file_hd.close()


class GGFD(_GG):
    """
    Classify structures based on their structural fingerprint.
    Attributes:\n
    - name: name of the CV.
    - type: Type of the CV.
    - clustering_type: How is it treated by clustering algorithms.
    - grouping_method: Classification method. It can be based on the distribution similarity or specific patterns.
    - integration_type: For similarity methods, you can select how the hellinger distance is calculate.
    - group_threshold: For groups method, the tolerance to define the belonging to a group
    - kernel: kernel function to use in the histogram generation.
    - bandwidth: the bandwidths for kernel density estimation.
    - grid_min: the lower bounds for the grid.
    - grid_max: the upper bounds for the grid.
    - grid_bins: the number of bins for the grid.
    - grid_space: the approximate grid spacing for the grid.
    - timeinterval: Simulation time interval to generate the distribution.

    Methods:\n
    - help(): returns available attributes and methods.
    - set_group_bins(*args, periodic=True, threshold="auto): select boundaries for the groups grouping method.
    - run(simulation): Creates groups from the crystal distributions in the simulation object.
    """

    def __init__(self, name: str, cv: Union[Torsions, MolecularOrientation, Density, PotentialEnergy]):
        """
        Generate Groups From Distribution
        :param name:
        :param cv:
        """
        if not cv._type.startswith(("Torsional Angle", "Molecular Orientation", "Density", "Potential Energy")):
            print("CV not suitable for creating groups.")
            exit()
        super(GGFD, self).__init__(name, cv._type)
        # Grouping Method Properties
        self._int_type = "discrete"
        self._grouping_method = "similarity"  # Alternatively, "groups"
        self._group_threshold = 0.1
        self._group_bins = {}

        # Original CV Properties (Read-only)
        self._dist_cv = cv
        self._kernel = cv._kernel
        self._bandwidth = cv._bandwidth
        self._timeinterval = cv._timeinterval
        if isinstance(cv._grid_bins, int):
            self._grid_min = [cv._grid_min]
            self._grid_max = [cv._grid_max]
            self._grid_bins = [cv._grid_bins]
            self._D: int = 1
        else:
            self._grid_min: Union[list, tuple] = cv._grid_min
            self._grid_max: Union[list, tuple] = cv._grid_max
            self._grid_bins: Union[list, tuple] = cv._grid_bins
            self._D: int = len(cv._grid_bins)

    # Read-only properties
    @property
    def cv_kernel(self):
        return self._kernel

    @property
    def cv_timeinterval(self):
        return self._timeinterval

    @property
    def cv_bandwidth(self):
        return self._bandwidth

    @property
    def cv_grid_min(self):
        return self._grid_min

    @property
    def cv_grid_max(self):
        return self._grid_max

    @property
    def cv_grid_bins(self):
        return self._grid_bins

    @property
    def grouping_method(self):
        return self._grouping_method

    @grouping_method.setter
    def grouping_method(self, grouping_method: str):
        if grouping_method.lower() in ("similarity", "groups"):
            self._grouping_method = grouping_method.lower()
        else:
            print("""
Error: Grouping selection method not recognized. Choose between:
- 'similarity': Calculate the Hellinger distance between each pair of distributions and group together the similar ones.
- 'groups':     Group together structures that have non-zero probability density in specified regions of space.""")
            exit()

    @property
    def integration_type(self):
        return self._int_type

    @integration_type.setter
    def integration_type(self, int_type: str):
        if int_type.lower() in ("discrete", "simps", "trapz"):
            self._int_type = int_type.lower()
        else:
            print('Error: Hellinger integration type not recognized. Choose between "discrete", "simps" or "trapz"')
            exit()

    @property
    def group_threshold(self):
        return self._group_threshold

    @group_threshold.setter
    def group_threshold(self, value: float):
        if 0. < value < 1.:
            self._group_threshold = value
        else:
            print("Group threshold must be between 0 and 1")
            exit()

    @staticmethod
    def _non_periodic_dist(step, ibins, bins, grid_bins, grid_min, grid_max, bins_space, threshold):
        if abs(grid_min - bins[0]) > threshold:
            bins = [grid_min] + bins
        if abs(grid_max - bins[-1]) > threshold:
            bins = bins[:-1]
        ibins[(step, 0)] = [j for j in range(grid_bins) if j * bins_space + grid_min < bins[1]]
        ibins[(step, len(bins) - 1)] = [j for j in range(grid_bins) if bins[-1] <= j * bins_space + grid_min]
        return ibins, bins

    @staticmethod
    def _periodic_dist(step, ibins, bins, grid_bins, grid_min, bins_space):
        bins = [grid_min] + bins
        ibins[(step, 0)] = [j for j in range(grid_bins) if j * bins_space + grid_min < bins[1]] + \
                           [j for j in range(grid_bins) if bins[-1] <= j * bins_space + grid_min]
        return ibins, bins

    def __str__(self):
        txt = """
CV: {0._name} ({0._type})
Clustering Type: {0._clustering_type}
Grouping Method: {0._grouping_method} 
Threshold: {0._group_threshold}\n""".format(self)
        if self._grouping_method == "groups" and self._group_bins:
            for k, item in self._group_bins.items():
                txt += "{}: {}\n".format(k, item)
        return txt

    def _write_output(self, path_output):
        file_output = open(path_output, "a")
        file_output.write(self.__str__())
        file_output.close()

    @staticmethod
    def help():
        return """    
Classify structures based on their structural fingerprint .
Attributes:
- name: name of the CV.
- type: Type of the CV.
- clustering_type: How is it treated by clustering algorithms. 
- grouping_method: Classification method. It can be based on the distribution similarity or specific patterns:
            - 'similarity': Calculate the Hellinger distance between each pair of distributions and group 
                            together the similar ones.
            - 'groups':     Group together structures that have non-zero probability density in specified regions 
                            of space.
- integration_type: For similarity methods, you can select how the hellinger distance is calculate.
            Choose between "discrete", "simps" or "trapz"
- group_threshold: For groups method, if the probability density in a specific region of space is lower than this 
            value, then it is assumed that no molecules are that area. 
- kernel: kernel function to use in the histogram generation.
- bandwidth: the bandwidths for kernel density estimation.
- grid_min: the lower bounds for the grid.
- grid_max: the upper bounds for the grid.
- grid_bins: the number of bins for the grid.
- grid_space: the approximate grid spacing for the grid.
- timeinterval: Simulation time interval to generate the distribution.
    
Methods:
- help(): returns available attributes and methods.
- set_group_bins(*args, periodic=True, threshold="auto): select boundaries for the groups grouping method. args must be 
                iterable objects. If periodic = False, an additional boundary is put at the grid max and min. 
- run(simulation): Creates groups from the crystal distributions in the simulation object.
    
Examples:
- Group structures by similarity:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
tor = gaff.get_cv("tor")                                      # Retrieve the CV Object
conf = gaff.ggfd("conf", tor)                                 # Create the GGFD object
conf.grouping_method = "similarity"                           # Use the similarity grouping method
conf.integration_type = "simps"                               # Use the simps method to calculate the hellonger distance
npt = gaff.get_simulation("npt")                              # Retrieve a completed simulation
conf.run(tor)                                                 # Generates groups
project.save() 

- Group structures by similarity:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
tor = gaff.get_cv("tor")                                      # Retrieve the CV Object (2D torsional angle)
conf = gaff.ggfd("conf", tor)                                 # Create the GGFD object
conf.grouping_method = "groups"                               # Use the "groups" grouping method
conf.group_threshold = 0.1                                    # Cutoff for determining which areas are occupied
conf.set_group_bins((-2., 0, 2.),(-1., 2.), periodic=True)    # Define the group boundaries in the (2D) CV space
npt = gaff.get_simulation("npt")                              # Retrieve a completed simulation
conf.run(tor)                                                 # Generates groups
project.save() 
"""

    def set_group_bins(self, *args: Union[list, tuple], periodic: bool = True):  # , threshold="auto"
        # TODO Change args to dict with key == names of the variables.
        #  As it is right now you have to remember the order in which ND dimensional distribution are added together.
        """
        Select boundaries for the groups grouping method. args must be iterable objects.
        If periodic = False, an additional boundary is put at the grid max and min.
        :param args: A list or a tuple of the dividing boundaries in a distribution. The number of boundaries must be
               equal to the dimension of the distribution.
        :param periodic: If True, periodic conditions are applied to the grouping algorithm. Mixing periodic and
               non-periodic boundaries can be done by setting periodic=True and adding a 0. to the boundary list in
               which it is not true
        :return:
        """
        args = list(args)
        if len(args) != self._D:
            print("Error: incorrect number of args, {} instead of {}.".format(len(args), self._D))
            exit()

        self._group_bins = {}
        for i in range(self._D):
            bins = list(args[i])
            bins_space = (self._grid_max[i] - self._grid_min[i]) / self._grid_bins[i]

            # if threshold == "auto":
            threshold = 0.5 * (self._grid_max[i] - self._grid_min[i]) / self._grid_bins[i]

            if not periodic:
                self._group_bins, bins = self._non_periodic_dist(i, self._group_bins, bins, self._grid_bins[i],
                                                                 self._grid_min[i], self._grid_max[i], bins_space,
                                                                 threshold)

            else:
                if abs(self._grid_min[i] - bins[0]) < threshold or abs(self._grid_max[i] - bins[-1]) < threshold:
                    self._group_bins, bins = self._non_periodic_dist(i, self._group_bins, bins, self._grid_bins[i],
                                                                     self._grid_min[i], self._grid_max[i], bins_space,
                                                                     threshold)
                else:
                    self._group_bins, bins = self._periodic_dist(i, self._group_bins, bins, self._grid_bins[i],
                                                                 self._grid_min[i], bins_space)

            for b in range(1, len(bins) - 1):
                self._group_bins[(i, b)] = [j for j in range(self._grid_bins[i])
                                            if bins[b] < j * bins_space + self._grid_min[i] <= bins[b + 1]]

    def run(self,
            simulation: Union[EnergyMinimization, CellRelaxation, MolecularDynamics, Metadynamics],
            crystals="all",
            catt=None,
            suffix=""):
        """
        Creates groups from the crystal distributions in the simulation object.
        :param simulation: Simulation Object (EnergyMinimization, CellRelaxation, MolecularDynamics, Metadynamics)
        :param crystals: It can be either "all", use all non-melted Crystal objects from the previous simulation or
               "centers", use only cluster centers from the previous simulation. Alternatively, you can select
               a specific subset of crystals by listing crystal names.
        :param catt: Use crystal attributes to select the crystal list
        :param suffix: TODO
        :return:
        """

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_seq_items', None)

        groups = {}
        list_crystals = get_list_crystals(simulation._crystals, crystals, catt)

        if self._grouping_method == "groups":
            combinations: list = []
            for i in range(self._D):
                combinations.append([c for c in self._group_bins.keys() if c[0] == i])

            # noinspection PyTypeChecker
            dataset = np.full((len(list_crystals), len((list(its.product(*combinations)))) + 1), np.nan)
            index = []
            for cidx in range(len(list_crystals)):
                crystal = list_crystals[cidx]

                index.append(crystal._name)
                dist = crystal._cvs[self._dist_cv._name + suffix] / np.sum(crystal._cvs[self._dist_cv._name + suffix])
                c = 0
                for i in its.product(*combinations):
                    dataset[cidx, c] = np.sum(dist[np.ix_(self._group_bins[i[0]], self._group_bins[i[1]])])
                    c += 1

            # noinspection PyTypeChecker
            dataset = pd.DataFrame(np.where(dataset > self._group_threshold, 1, 0), index=index,
                                   columns=[(i[0][1], i[1][1]) for i in its.product(*combinations)] + ["Others"])

            groups = dataset.groupby(dataset.columns.to_list()).groups
            # noinspection PyUnresolvedReferences
            groups = {k: groups[k].to_list() for k in sorted(groups.keys(), key=lambda x: np.sum(x))}

        elif self._grouping_method == "similarity":
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import breadth_first_order
            index = []
            for crystal in list_crystals:
                index.append(crystal._name)

            dmat = pd.DataFrame(np.zeros((len(index), len(index))), columns=index, index=index)
            bar = progressbar.ProgressBar(maxval=int(len(crystals) * (len(crystals) - 1) / 2)).start()
            nbar = 1

            for i in range(len(list_crystals) - 1):
                from copy import deepcopy
                di = deepcopy(list_crystals[i]._cvs[self._dist_cv._name + suffix])
                ci = list_crystals[i]._name
                for j in range(i + 1, len(crystals)):
                    dj = deepcopy(list_crystals[j]._cvs[self._dist_cv._name + suffix])
                    cj = list_crystals[j]._name
                    bar.update(nbar)
                    nbar += 1
                    if self._dist_cv._type == "Radial Distribution Function":
                        if len(di) > len(dj):
                            hd = hellinger(di.copy()[:len(dj)], dj.copy(), self._int_type)
                        else:
                            hd = hellinger(di.copy(), dj.copy()[:len(di)], self._int_type)
                    else:
                        hd = hellinger(di.copy(), dj.copy(), self._int_type)
                    dmat.at[ci, cj] = dmat.at[cj, ci] = hd
            bar.finish()

            dmat = pd.DataFrame(np.where(dmat.values < self._group_threshold, 1., 0.), columns=index, index=index)

            graph = csr_matrix(dmat)
            removed = []
            for c in range(len(dmat.index)):
                if dmat.index.to_list()[c] in removed:
                    continue
                bfs = breadth_first_order(graph, c, False, False)
                group_index = [index[i] for i in range(len(index)) if i in bfs]
                removed = removed + group_index
                groups[group_index[0]] = group_index

        self._run(simulation, groups, crystals, catt, suffix)


class GGFA(_GG):
    """
    Classify structures based on their attributes.
    Attributes:\n
    - name: name of the CV.
    - type: Type of the CV.
    - clustering_type: "classification"
    - attribute: Attribute label to be used for classification

    Methods:\n
    - help(): returns available attributes and methods. TODO
    - run(simulation): Creates groups looking at the crystal attributes in the simulation object
    """

    def __init__(self, name: str, attribute: str):
        """
        Generate Groups From Attributes
        :param name: Group Label.
        :param attribute: Attribute label to be used for classification.
        """

        # Grouping Properties
        super().__init__(name, "Attribute")

        self._attribute = attribute

    @property
    def attribute(self):
        return self._attribute

    def run(self,
            simulation: Union[EnergyMinimization, CellRelaxation, MolecularDynamics, Metadynamics],
            crystals="all",
            catt=None,
            suffix=""):
        """
        Creates groups from the crystal attributes in the simulation object.
        :param simulation: Simulation Object (EnergyMinimization, CellRelaxation, MolecularDynamics, Metadynamics)
        :param crystals: It can be either "all", use all non-melted Crystal objects from the previous simulation or
               "centers", use only cluster centers from the previous simulation. Alternatively, you can select
               a specific subset of crystals by listing crystal names.
        :param catt: Use crystal attributes to select the crystal list
        :param suffix: TODO
        :return:
        """

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_seq_items', None)

        groups = {}
        list_crystals = get_list_crystals(simulation._crystals, crystals, catt)

        if not all(self._attribute in crystal._attributes for crystal in list_crystals):
            print(f"Error: some of the Crystals do not have attribute '{self._attribute}'")
            exit()

        for crystal in list_crystals:
            gatt = crystal._attributes[self._attribute]
            if gatt in groups.keys():
                groups[gatt].append(crystal._name)
            else:
                groups[gatt] = [crystal._name]

        self._run(simulation, groups, crystals, catt, suffix)

    def __str__(self):
        return """
CV: {0._name} ({0._type})
Clustering Type: {0._clustering_type}
""".format(self)

    def _write_output(self, path_output):
        file_output = open(path_output, "a")
        file_output.write(self.__str__())
        file_output.close()
