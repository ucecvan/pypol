import numpy as np
import os
import matplotlib.pyplot as plt
import progressbar
import pandas as pd
import itertools as its
import copy
from typing import Union

from PyPol.utilities import get_list_crystals, hellinger


class Clustering(object):

    def __init__(self, name, cvs):
        self._name = name
        self._cvp = cvs

        self._int_type = "discrete"
        self._algorithm = "fsfdp"  # Only method available for now

        # FSFDP parameters
        self._kernel = "gaussian"
        self._centers = "energy"
        # self._d_c = []
        self._d_c_fraction = 0.01
        self._sigma_cutoff = False

    # Read-only Properties
    # @property
    # def d_c(self):
    #     if isinstance(self._d_c, float):
    #         return self._d_c
    #     else:
    #         print("d_c has not been calculated yet. Generate the distance matrix to visualize it.")

    # @property
    # def distance_matrix(self):
    #     return self._distance_matrix

    @property
    def clustering_algorithm(self):
        return self._algorithm

    #
    # @property
    # def clusters(self):
    #     return self._clusters

    @property
    def name(self):
        return self._name

    @property
    def cvs(self):
        txt = "Collective Variables:\n"
        for cv in self._cvp:
            txt += cv._name + "\n"
        return txt

    # Properties

    @property
    def sigma_cutoff(self):
        return self._sigma_cutoff

    @sigma_cutoff.setter
    def sigma_cutoff(self, value):
        if 0. < value < 1.:
            self._sigma_cutoff = value
        else:
            print(r"The cutoff must be between 0 and 1.")

    @property
    def d_c_neighbors_fraction(self):
        return self._d_c_fraction

    @d_c_neighbors_fraction.setter
    def d_c_neighbors_fraction(self, value: float):
        if 0. < value < 1.:
            self._d_c_fraction = value
            if value > 0.05:
                print("The average number of neighbors should be between 1% and 5% of the total number of points."
                      "Fractions higher than 0.05 could cause problems in the analysis.")
        else:
            print(r"The neighbors fraction must be between 0 and 1. The average number of neighbors should be between "
                  r"1% and 5% of the total number of points.")

    @property
    def hellinger_integration_type(self):
        return self._int_type

    @hellinger_integration_type.setter
    def hellinger_integration_type(self, int_type):
        if int_type.lower() in ("discrete", "simps", "trapz"):
            self._int_type = int_type.lower()
        else:
            print('Error: Hellinger integration type not recognized. Choose between "discrete", "simps" or "trapz"')
            exit()

    @property
    def center_selection(self):
        return self._centers

    @center_selection.setter
    def center_selection(self, center_selection):
        if center_selection.lower() in ("energy", "cluster_center"):
            self._centers = center_selection.lower()
        else:
            print("Error: Center selection method not recognized. Choose between:\n"
                  "'energy'        : select structure with the lower potential energy in the group as cluster center.\n"
                  "'cluster_center': select the cluster center resulting from the clustering algorithm.")
            exit()

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, kernel):
        if kernel.lower() in ("gaussian", "cutoff"):
            self._kernel = kernel.lower()
        else:
            print("Error: Kernel function not recognized. Choose between 'gaussian' and 'cutoff'")
            exit()

    @staticmethod
    def _sort_crystal(crystal, combinations, threshold=0.8, suffix=""):
        for i in combinations.index[:-1]:
            for j in combinations.columns[:-2]:
                if crystal._cvs[j + suffix][combinations.loc[i, j]] > threshold and j == combinations.columns[-3]:
                    combinations.loc[i, "Structures"].append(crystal)
                    combinations.loc[i, "Number of structures"] += 1
                    return combinations
                elif crystal._cvs[j + suffix][combinations.loc[i, j]] < threshold:
                    break
        combinations.loc["Others", "Structures"].append(crystal)
        combinations.loc["Others", "Number of structures"] += 1
        return combinations

    def run(self,
            simulation,
            crystals="all",
            group_threshold: float = 0.8,
            catt=None,
            suffix="",
            path_output="",
            _check=True):

        from PyPol.gromacs import EnergyMinimization, MolecularDynamics, CellRelaxation, Metadynamics

        if type(simulation) not in [EnergyMinimization, MolecularDynamics, CellRelaxation, Metadynamics]:
            print("Error: simulation object not suitable for clustering analysis")

        # TODO Simplify script. Rewrite crystal sorting in groups and clustering
        #      ---> too complicated and probably inefficient to use pandas Dataframe in this context
        #      Use crystal attributes or dict?
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.max_seq_items', None)

        simulation._clusters = {}
        simulation._cluster_data = {}

        list_crystals = get_list_crystals(simulation._crystals, crystals, catt)

        if not simulation._completed and not _check:
            print("Simulation {} is not completed yet. Run simulation.get_results() to check termination and import "
                  "results.".format(simulation._name))

        if not path_output:
            path_output = simulation._path_output
            path_output_data = simulation._path_output + str(self._name) + f"{suffix}_data/"
        else:
            path_output_data = path_output + str(self._name) + f"{suffix}_data/"

        if not os.path.exists(path_output_data):
            os.mkdir(path_output_data)
        if not os.path.exists(path_output):
            os.mkdir(path_output)

        d_c = np.array([])

        # Sort crystals into groups
        group_options = []
        group_names = []
        for cv in self._cvp:
            if cv.clustering_type == "classification":
                crystal = list_crystals[0]
                group_options.append(list(crystal._cvs[cv._name + suffix].keys()))
                group_names.append(cv._name)
        if group_options:
            if len(group_names) == 1:
                combinations = group_options[0] + [None]
                index = [str(i) for i in range(len(combinations) - 1)] + ["Others"]
                combinations = pd.concat((pd.Series(combinations, name=group_names[0], index=index),
                                          pd.Series([0 for _ in range(len(combinations))],
                                                    name="Number of structures", dtype=int, index=index),
                                          pd.Series([[] for _ in range(len(combinations))], name="Structures",
                                                    index=index)), axis=1)
            else:
                combinations = list(its.product(*group_options)) + \
                               [tuple([None for _ in range(len(group_options[0]))])]
                index = [str(i) for i in range(len(combinations) - 1)] + ["Others"]
                combinations = pd.concat((pd.DataFrame(combinations, columns=group_names, index=index),
                                          pd.Series([0 for _ in range(len(combinations))],
                                                    name="Number of structures", dtype=int, index=index),
                                          pd.Series([[] for _ in range(len(combinations))], name="Structures",
                                                    index=index)), axis=1)
            combinations.index.name = "Combinations"
            for crystal in list_crystals:
                combinations = self._sort_crystal(crystal, combinations, group_threshold, suffix)
        else:
            combinations = pd.DataFrame([[len(list_crystals), list_crystals]],
                                        columns=["Number of structures", "Structures"],
                                        dtype=None, index=["all"])
            combinations.index.name = "Combinations"
            # for crystal in list_crystals:
            #     combinations.loc["all", "Structures"].append(crystal)
            #     combinations.loc["all", "Number of structures"] += 1

        slist = [np.full((combinations.loc[i, "Number of structures"],
                          combinations.loc[i, "Number of structures"]), 0.0) for i in combinations.index]
        combinations = pd.concat((combinations,
                                  pd.Series(slist, name="Distance Matrix", index=combinations.index)), axis=1)

        # Generate Distance Matrix of each set of distributions
        distributions = [cv for cv in self._cvp if cv.clustering_type != "classification"]
        n_factors = {}
        for cv in distributions:
            combinations[cv._name + suffix] = pd.Series(copy.deepcopy(combinations["Distance Matrix"].to_dict()),
                                                        index=combinations.index)
            n_factors[cv._name + suffix] = 0.

            print("\nCV: {}".format(cv._name))
            bar = progressbar.ProgressBar(maxval=len(list_crystals)).start()
            nbar = 1

            for index in combinations.index:
                if combinations.at[index, "Number of structures"] > 1:
                    crystals = combinations.at[index, "Structures"]

                    for i in range(len(crystals) - 1):
                        di = crystals[i]._cvs[cv._name + suffix]
                        bar.update(nbar)
                        nbar += 1
                        for j in range(i + 1, len(crystals)):
                            dj = crystals[j]._cvs[cv._name + suffix]
                            if di.shape != dj.shape:
                                di = di.copy()[tuple(map(slice, dj.shape))]
                                dj = dj.copy()[tuple(map(slice, di.shape))]
                            hd = hellinger(di.copy(), dj.copy(), self._int_type)
                            combinations.loc[index, cv._name + suffix][i, j] = combinations.loc[
                                index, cv._name + suffix][j, i] = hd

                            if hd > n_factors[cv._name + suffix]:
                                n_factors[cv._name + suffix] = hd

            bar.finish()

        # Normalize distances
        print("Normalization...", end="")
        normalization = []
        for cv in distributions:
            normalization.append(1. / n_factors[cv._name + suffix])
            for index in combinations.index:
                if combinations.at[index, "Number of structures"] > 1:
                    combinations.at[index, cv._name + suffix] /= n_factors[cv._name + suffix]
        print("done\nGenerating Distance Matrix...", end="")

        # Generate Distance Matrix
        normalization = np.linalg.norm(np.array(normalization))
        for index in combinations.index:
            if combinations.at[index, "Number of structures"] > 1:
                if len(distributions) > 1:
                    combinations.at[index, "Distance Matrix"][:, :] = np.linalg.norm(
                        np.dstack(tuple([k for k in combinations.loc[index, [
                            cv._name + suffix for cv in distributions]]])), axis=2) / normalization
                else:
                    combinations.at[index, "Distance Matrix"][:, :] = combinations.loc[
                                                                          index, distributions[0]._name + suffix] / \
                                                                      normalization
                d_c = np.append(d_c, combinations.at[
                    index, "Distance Matrix"][np.triu_indices(combinations.at[index, "Distance Matrix"].shape[0], 1)])

        # combinations.at[index, "Distance Matrix"][:, :] = np.linalg.norm(
        #             np.dstack(set([k for k in combinations.loc[index, [cv._name for cv in distributions]]])),
        #             axis=2) / normalization
            #     for i in range(combinations.at[index, "Number of structures"] - 1):
            #         for j in range(i + 1, combinations.at[index, "Number of structures"]):
            #             dist_ij = np.linalg.norm([k[i, j] for k in
            #                                       combinations.loc[index, [cv._name for cv in distributions]]])
            #             combinations.at[index, "Distance Matrix"][i, j] = \
            #                 combinations.at[index, "Distance Matrix"][j, i] = dist_ij / normalization
            #             d_c.append(dist_ij)

        for index in combinations.index:
            if combinations.at[index, "Number of structures"] > 1:
                idx = [i._name for i in combinations.at[index, "Structures"]]
                for mat in combinations.loc[index, "Distance Matrix":].index:
                    combinations.at[index, mat] = pd.DataFrame(combinations.at[index, mat], index=idx, columns=idx)
                    combinations.at[index, mat].to_csv(path_output_data + mat.replace(" ", "") + "_" + index + ".dat")
                    # with open(path_output + mat.replace(" ", "") + "_" + index + ".dat", 'w') as fo:
                    #     fo.write(combinations.loc[index, mat].__str__())

        print("done\nSaving Distance Matrix...", end="")
        for i in combinations.loc[:, "Distance Matrix":].columns:
            total = pd.concat([m for m in combinations.loc[:, i] if not isinstance(m, np.ndarray)])
            total.to_csv(path_output + str(self._name) + "_" + i.replace(" ", "") + ".dat")
            plt.imshow(total, interpolation="nearest", cmap="viridis")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(path_output + str(self._name) + "_" + i.replace(" ", "") + ".png", dpi=300)
            plt.close('all')

        list_crys = [[i._name for i in row["Structures"]] for index, row in combinations.iterrows()]
        file_output = pd.concat((combinations.loc[:, :"Number of structures"],
                                 pd.Series(list_crys, name="IDs", index=combinations.index)), axis=1)

        with open(path_output + str(self._name) + "_Groups.dat", 'w') as fo:
            fo.write("Normalization Factors:\n")
            for n in n_factors.keys():
                fo.write("{:15}: {:<1.3f}\n".format(n, n_factors[n]))
            fo.write(file_output.__str__())
        d_c = np.sort(np.array(d_c))[int(float(len(d_c)) * self._d_c_fraction)]
        print("done")

        # Remove structures that are not cluster centers
        print("Clustering...", end="")
        changes_string = ""
        with open(path_output + str(self._name) + "_FSFDP.dat", 'w') as fo:
            fo.write("# FSFDP parameters for every group:\n")

        for index in combinations.index:
            if int(combinations.at[index, "Number of structures"]) == 0:
                continue
            elif int(combinations.at[index, "Number of structures"]) == 1:
                nc = combinations.at[index, "Structures"][0]._name
                columns = ["rho", "sigma", "NN", "cluster", "distance"]
                simulation._cluster_data[index] = pd.DataFrame([[0, 0, pd.NA, nc, 0]], index=[nc], columns=columns)
                simulation._clusters[index] = {nc: [nc]}
            elif int(combinations.at[index, "Number of structures"]) == 2:
                nc1 = combinations.at[index, "Structures"][0]._name
                nc2 = combinations.at[index, "Structures"][1]._name
                columns = ["rho", "sigma", "NN", "cluster", "distance"]
                d_12 = combinations.at[index, "Distance Matrix"].values[0, 1]
                if d_12 > d_c:
                    simulation._cluster_data[index] = pd.DataFrame([[0, 0, nc2, nc1, 0], [0, 0, nc1, nc2, 0]],
                                                                   index=[nc1, nc2], columns=columns)
                    simulation._clusters[index] = {nc1: [nc1], nc2: [nc2]}
                else:
                    simulation._cluster_data[index] = pd.DataFrame([[0, 0, nc2, nc1, 0], [0, 0, nc1, nc1, d_12]],
                                                                   index=[nc1, nc2], columns=columns)
                    simulation._clusters[index] = {nc1: [nc1, nc2]}
            elif int(combinations.at[index, "Number of structures"]) > 2:
                if self._algorithm == "fsfdp":
                    simulation._cluster_data[index], sc = FSFDP(combinations.at[index, "Distance Matrix"],
                                                                kernel=self._kernel,
                                                                d_c=d_c,
                                                                d_c_neighbors_fraction=self._d_c_fraction,
                                                                sigma_cutoff=self._sigma_cutoff)
                    _save_decision_graph(simulation._cluster_data[index].loc[:, "rho"].values,
                                         simulation._cluster_data[index].loc[:, "sigma"].values,
                                         sigma_cutoff=sc,
                                         path=path_output_data + "Decision_graph_" + str(index) + ".png")

                    with open(path_output_data + "FSFDP_" + str(index) + ".dat", 'w') as fo:
                        fo.write(simulation._cluster_data[index].__str__())

                    with open(path_output + str(self._name) + "_FSFDP.dat", 'a') as fo:
                        fo.write("\n# Group {}\n".format(str(index)))
                        fo.write(simulation._cluster_data[index].__str__())

                simulation._clusters[index] = {
                    k: simulation._cluster_data[index].index[simulation._cluster_data[index]["cluster"] == k].tolist()
                    for k in list(simulation._cluster_data[index]["cluster"].unique())}

            if self._centers.lower() == "energy":
                new_clusters = copy.deepcopy(simulation._clusters[index])
                energies = {k._name: k._energy for k in combinations.at[index, "Structures"]}
                for center in simulation._clusters[index].keys():
                    changes = [center, None]
                    emin = energies[center]
                    for crystal in simulation._clusters[index][center]:
                        if energies[crystal] < emin:
                            changes[1] = crystal
                            emin = energies[crystal]
                    if changes[1]:
                        new_clusters[changes[1]] = new_clusters.pop(changes[0])
                        changes_string += "{:>25} ---> {:25}\n".format(changes[0], changes[1])
                simulation._clusters[index] = new_clusters

            for crystal in combinations.at[index, "Structures"]:
                for cc in simulation._clusters[index].keys():
                    if crystal._name in simulation._clusters[index][cc]:
                        crystal._state = cc
                        break
        cluster_groups = [g for g in simulation._clusters.keys() for _ in simulation._clusters[g].keys()]
        simulation._clusters = {k: v for g in simulation._clusters.keys() for k, v in simulation._clusters[g].items()}
        simulation._clusters = pd.concat((
            pd.Series(data=[len(simulation._clusters[x]) for x in simulation._clusters.keys()],
                      index=simulation._clusters.keys(),
                      name="Number of Structures"),
            pd.Series(data=cluster_groups, index=simulation._clusters.keys(), name="Group"),
            pd.Series(data=[", ".join(str(y) for y in simulation._clusters[x]) for x in simulation._clusters.keys()],
                      index=simulation._clusters.keys(), name="Structures")),
            axis=1).sort_values(by="Number of Structures", ascending=False)

        with open(path_output + str(self._name) + "_Clusters.dat", 'w') as fo:
            if changes_string:
                fo.write("Cluster centers changed according to potential energy:\n")
                fo.write(changes_string)
            fo.write(simulation._clusters.__str__())

        with open(path_output + str(self._name) + "_DFC.dat", 'w') as fo:
            if changes_string:
                fo.write("Cluster centers changed according to potential energy:\n")
                fo.write(changes_string)
            total = pd.concat([m for m in combinations.loc[:, "Distance Matrix"]
                               if not isinstance(m, np.ndarray)])
            index = []
            centers = []
            distances = []
            for crystal in get_list_crystals(simulation._crystals, crystals=total.index.to_list()):
                index.append(crystal._name)
                centers.append(crystal._state)
                distances.append(total.at[crystal._name, crystal._state])
            dfc = pd.DataFrame({"Center": centers, "Distance": distances}, index=index).sort_values(by="Distance")
            fo.write(dfc.__str__())
        print("done")


#
# Clustering Algorithms
#

# 1. Fast Search and Find of Density Peaks
# noinspection PyPep8Naming
def FSFDP(dmat: Union[pd.DataFrame, np.ndarray],
          kernel: str = "gaussian",
          d_c: Union[str, float] = "auto",
          d_c_neighbors_fraction: float = 0.02,
          sigma_cutoff: Union[bool, float] = False):
    """
    Simplified FSFDP algorithm. Instead of halo, the distance of crystals from center is printed in the output to check
    possible errors. No use of density cutoff.
    :param dmat: Distance Matrix, expressed as a pandas DataFrame or Numpy array. If an indexed DataFrame is used,
                 results are shown using structures' IDs
    :param kernel: Kernel function to calculate the distance, it can be 'cutoff' or 'gaussian'
    :param d_c: Distance cutoff used to calculate the density (rho)
    :param d_c_neighbors_fraction: Average number of neighbors with respect to total used to calculate d_c
    :param sigma_cutoff: Sigma cutoff for the decision graph. If false you can select it from the plot.
    :return:
    """

    if isinstance(dmat, np.ndarray):
        dmat = pd.DataFrame(dmat)

    if d_c == "auto":
        d_c = np.sort(dmat.values.flatten())[int(dmat.values.size * d_c_neighbors_fraction) + dmat.values.shape[0]]

    # Find density vector
    rho = np.zeros(dmat.values.shape[0])
    if kernel == "cutoff":
        def kernel_function(d_ij):
            return 1 if d_ij < d_c else 0
    else:
        def kernel_function(d_ij):
            return np.exp(-(d_ij / d_c) * (d_ij / d_c))

    for i in range(dmat.values.shape[0] - 1):
        for j in range(i + 1, dmat.values.shape[0]):
            rho[i] += kernel_function(dmat.values[i][j])
            rho[j] += kernel_function(dmat.values[i][j])

    rho = pd.Series(rho, index=dmat.index, name="rho")

    # Find sigma vector
    sigma = pd.Series(np.full(rho.shape, -1.0), dmat.index, name="sigma")
    nn = pd.Series(np.full(rho.shape, pd.NA), dmat.index, dtype="string", name="NN")
    for i in sigma.index:
        if rho[i] == np.max(rho.values):
            continue
        else:
            sigma[i] = np.nanmin(np.where(rho > rho[i], dmat[i].values, np.nan))
            nn[i] = str(dmat.index[np.nanargmin(np.where(rho > rho[i], dmat[i].values, np.nan))])
    sigma[rho.idxmax()] = np.nanmax(sigma.values)

    # plot results
    if not sigma_cutoff:
        sigma_cutoff = _decision_graph(rho, sigma)

    # Assign structures to cluster centers
    dataset = pd.concat((rho, sigma, nn,
                         pd.Series(np.full(rho.shape, pd.NA), dmat.index, name="cluster"),
                         pd.Series(np.full(rho.shape, pd.NA), dmat.index, name="distance")),
                        axis=1).sort_values(by="rho", ascending=False)

    for i in dataset.index:
        if dataset.loc[i, "sigma"] >= sigma_cutoff or i == dataset.index[0]:
            dataset.at[i, "cluster"] = i
            dataset.at[i, "distance"] = 0.0

        else:
            dataset.at[i, "cluster"] = dataset.loc[dataset.loc[i]["NN"]]["cluster"]
            dataset.at[i, "distance"] = dmat.loc[i, dataset.loc[i, "cluster"]]

    return dataset, sigma_cutoff


def _decision_graph(x, y):
    class PointPicker(object):
        def __init__(self, ppax, ppscat, clicklim=0.05):
            self.fig = ppax.figure
            self.ax = ppax
            self.scat = ppscat
            self.clicklim = clicklim
            self.sigma_cutoff = 0.1
            self.horizontal_line = ppax.axhline(y=.1, color='red', alpha=0.5)
            self.text = ppax.text(0, 0.5, "")
            self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        def onclick(self, event):
            if event.inaxes == self.ax:
                self.sigma_cutoff = event.ydata
                xlim0, xlim1 = ax.get_xlim()
                self.horizontal_line.set_ydata(self.sigma_cutoff)
                self.text.set_text(str(round(self.sigma_cutoff, 5)))
                self.text.set_position((xlim0, self.sigma_cutoff))
                colors = []
                for i in self.scat.get_offsets():
                    if i[1] >= self.sigma_cutoff:
                        colors.append("C0")
                    else:
                        colors.append("C1")
                self.scat.set_color(colors)
                self.fig.canvas.draw()

    fig = plt.figure()

    ax = fig.add_subplot(111)
    scat = ax.scatter(x, y, color="C0", alpha=0.25)

    plt.title(r"Select $\sigma$-cutoff and quit to continue", fontsize=20)
    plt.xlabel(r"$\rho$", fontsize=20)
    plt.ylabel(r"$\delta$", fontsize=20)
    p = PointPicker(ax, scat)
    plt.show()
    return p.sigma_cutoff


def _save_decision_graph(rho, sigma, sigma_cutoff, path):
    for i in range(len(rho)):
        if sigma[i] >= sigma_cutoff:
            if rho[i] < rho.max() / 100.0:
                plt.scatter(rho[i], sigma[i], s=20, marker='o', c="black", edgecolor='face')
            else:
                plt.scatter(rho[i], sigma[i], s=20, marker='o', c="C0", edgecolor='face')
        else:
            plt.scatter(rho[i], sigma[i], s=20, marker='o', c="C1", edgecolor='face')

    plt.fill_between(np.array([-max(rho), max(rho) + 0.25]), np.array([sigma_cutoff, sigma_cutoff]),
                     color="C1", alpha=0.1)

    plt.xlim(0.0, max(rho) + 0.25)
    plt.ylim(0.0, max(sigma) + 0.1)

    plt.xlabel(r"$\rho$", fontsize=20)
    plt.ylabel(r"$\sigma$", fontsize=20)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close('all')
