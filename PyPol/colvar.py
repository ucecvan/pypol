class Torsions(object):

    def __init__(self, name, method):
        """
        Generates a distribution of the torsional angles of the selected atoms.
        :param name: str, name of the collective variable. Default output and variables will have this name.
        :param method: obj, method used to identify topology parameters and crystal structures.
        """
        import numpy as np
        self.type = "Torsional Angle"
        self.clustering_type = "regression"
        self.method = method
        self.name = name
        self.atoms = list()
        self.molecule = None

        self.kernel = "GAUSSIAN"
        self.bandwidth = 0.5

        self.grid_min = -np.pi
        self.grid_max = np.pi
        self.grid_bin = 36

        self.groups = {}

    def set_atoms(self, atoms, molecule):
        """
        Select atom indices of the reference molecule. This is used to identify the torsions of each molecule in the
        crystal.
        :param atoms: list, Atom indices. All atoms indices are available in the project output file after the topology
        is defined.
        :param molecule: obj, Reference molecule
        :return:
        """
        self.atoms = list(atoms)
        self.molecule = molecule
        self.generate_input()

    def set_grid(self, grid_min, grid_max, grid_bin):
        """

        :param grid_min:
        :param grid_max:
        :param grid_bin:
        :return:
        """
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.grid_bin = grid_bin
        self.generate_input()

    def set_bandwidth(self, bandwidth=0.5):
        """

        :param bandwidth:
        :return:
        """
        self.bandwidth = bandwidth
        self.generate_input()

    def set_kernel(self, kernel="GAUSSIAN"):
        """

        :param kernel:
        :return:
        """
        self.kernel = kernel
        self.generate_input()

    def set_clusteringtype(self, clusteringtype):
        """

        :param clusteringtype:
        :return:
        """
        if clusteringtype not in ["classification", "regression"]:
            print("Clustering types available: 'classification' or 'regression'")
            exit()
        self.clustering_type = clusteringtype
        if clusteringtype == "classification" and not self.groups:
            self.add_group((self.grid_min, self.grid_max), "Others", sort_group=False)

    def add_group(self, groups, name=None, sort_group=True):
        """

        :param groups:
        :param name:
        :param sort_group:
        :return:
        """
        if name is None:
            name = len(self.groups)

        if all(isinstance(group, tuple) for group in groups):
            for group in groups:
                self.add_group(group, name)

        elif isinstance(groups, tuple):
            for group_name in self.groups.keys():
                if name == group_name:
                    self.groups[name].append(groups)
                    if sort_group:
                        self.groups = sort_groups(self.grid_min, self.grid_max, self.groups)
                    self.generate_input()
                    return
            self.groups[name] = [groups]
        if sort_group:
            self.groups = sort_groups(self.grid_min, self.grid_max, self.groups)
        self.generate_input()

    def generate_input(self):
        """

        :return:
        """
        if not self.atoms:
            print("Error: no atoms found. select atoms with the set_atoms module.")
            exit()
        print("=" * 100)
        print("Generate plumed input files")
        print("CV: {} ({})".format(self.name, self.type))
        print("Atoms:", end=" ")
        for atom in self.atoms:
            print("{}({})".format(atom, self.molecule.atoms[atom].label), end="  ")

        if self.clustering_type == "regression":
            print("\nClustering type: Regression\n"
                  "Parameters: KERNEL={0} NBINS={1} BANDWIDTH={2:.3f} UPPER={3:.3f} LOWER={4:.3f}"
                  "".format(self.kernel, self.grid_bin, self.bandwidth, self.grid_max, self.grid_min))
        elif self.clustering_type == "classification":
            print("\nClustering type: Classification")
            for group in self.groups.keys():
                print("Group {}:".format(group), end=" ")
                for boundary in self.groups[group]:
                    print(boundary, end=" ")
                print()

        for crystal in self.method.initial_crystals:
            print(crystal.name)
            lines_atoms = generate_atom_list(self.atoms, self.molecule, crystal, keyword="ATOMS", lines=[])
            file_plumed = open(crystal.path + "plumed_" + self.name + ".dat", "w")
            file_plumed.write("TORSIONS ...\n")
            for line in lines_atoms:
                file_plumed.write(line)

            if self.clustering_type == "regression":
                file_plumed.write("HISTOGRAM={{{{{0} NBINS={1} BANDWIDTH={2:.3f} UPPER={3:.3f} LOWER={4:.3f}}}}}\n"
                                  "".format(self.kernel, self.grid_bin, self.bandwidth, self.grid_max, self.grid_min))
            elif self.clustering_type == "classification":
                idx_boundaries = 1
                for group in self.groups.keys():
                    for boundary in self.groups[group]:
                        file_plumed.write("BETWEEN{0}={{{{{1} LOWER={2:.3f} UPPER={3:.3f}}}}}\n"
                                          "".format(idx_boundaries, self.kernel, boundary[0], boundary[1]))
                        idx_boundaries += 1
            file_plumed.write("LABEL={0}\n... TORSIONS\n\n"
                              "PRINT ARG={0}.* FILE=plumed_SimulationName_{0}.dat\n".format(self.name))
            file_plumed.close()
        print("=" * 100)


class MolecularOrientation(object):

    def __init__(self, name, method):
        """
        Generates a distribution of the intermolecular torsional angles of the selected atoms.
        :param name: str, name of the collective variable. Default output and variables will have this name.
        :param method: obj, method used to identify topology parameters and crystal structures.
        """
        import numpy as np
        self.type = "Molecular Orientation"
        self.clustering_type = "regression"  # No "classification" possible
        self.method = method
        self.name = name
        self.atoms = list()
        self.molecules = list()

        self.kernel = "GAUSSIAN"
        self.bandwidth = 0.5

        self.grid_min = 0.0
        self.grid_max = np.pi
        self.grid_bin = 18

    def set_atoms(self, atoms, molecule):
        """

        :param atoms:
        :param molecule:
        :return:
        """
        self.atoms.append(list(atoms))
        self.molecules.append(molecule)
        self.generate_input()

    def set_grid(self, grid_min, grid_max, grid_bin):
        """

        :param grid_min:
        :param grid_max:
        :param grid_bin:
        :return:
        """
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.grid_bin = grid_bin
        self.generate_input()

    def set_bandwidth(self, bandwidth=0.5):
        """

        :param bandwidth:
        :return:
        """
        self.bandwidth = bandwidth
        self.generate_input()

    def set_kernel(self, kernel="GAUSSIAN"):
        """

        :param kernel:
        :return:
        """
        self.kernel = kernel.upper()
        self.generate_input()

    def generate_input(self):
        """

        :return:
        """
        # Change creation of parameters. Depends on molecules and number of vectors for each of them! CAMBIARE!!!
        if not self.atoms:
            print("Error: no atoms found. select atoms with the set_atoms module.")
            exit()
        print("=" * 100)
        print("Generate plumed input files")
        print("CV: {} ({})".format(self.name, self.type))
        print("Atoms:", end=" ")
        for idx_mol in range(len(self.molecules)):
            for atom in self.atoms[idx_mol]:
                print("{}({})".format(atom, self.molecules[idx_mol].atoms[atom].label), end="  ")

        print("\nClustering type: Regression\n"
              "Parameters: KERNEL={0} NBINS={1} BANDWIDTH={2:.3f} UPPER={3:.3f} LOWER={4:.3f}"
              "".format(self.kernel, self.grid_bin, self.bandwidth, self.grid_max, self.grid_min))

        for crystal in self.method.initial_crystals:
            print(crystal.name)
            # Select atoms and molecules
            lines_atoms = []
            for idx_mol in range(len(self.molecules)):
                lines_atoms = generate_atom_list(self.atoms[idx_mol], self.molecules[idx_mol], crystal,
                                                 keyword="ATOMS", lines=lines_atoms)
            file_plumed = open(crystal.path + "plumed_" + self.name + ".dat", "w")

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
                              "PRINT ARG=valg_{0} FILE=plumed_SimulationName_{0}.dat\n"
                              "".format(self.name, self.grid_min, self.grid_max,
                                        self.grid_bin, self.bandwidth, self.kernel))
            file_plumed.close()
        print("=" * 100)


class RDF(object):

    def __init__(self, name, center="geometrical"):
        """

        :param name:
        :param center:
        """
        self.name = name
        self.center = center
        self.atoms = None

        self.switching_function = "RATIONAL"
        self.r_0 = 0.0
        self.d_0 = 10.0
        self.d_max = 10.0

        self.grid_min = 0.0
        self.grid_max = 10.0
        self.grid_bin = 100

        self.bandwidth = 0.01


def sort_groups(grid_min, grid_max, groups, tolerance=0.01):
    """

    :param grid_min:
    :param grid_max:
    :param groups:
    :param tolerance:
    :return:
    """
    new_groups = {k: groups[k] for k in sorted(groups, key=groups.get) if k != "Others"}

    ranges = list()
    for cvranges in new_groups.values():
        for cvrange in cvranges:
            ranges.append(cvrange)
    ranges.sort(key=lambda x: x[0])
    gmin = grid_min
    new_groups["Others"] = list()
    for cvrange in ranges:
        if cvrange[0] > gmin and cvrange[0] - gmin > tolerance:
            new_groups["Others"].append((gmin, cvrange[0]))
            gmin = cvrange[1]
        else:
            gmin = cvrange[1]
        if cvrange == ranges[-1] and cvrange[1] < grid_max and grid_max - cvrange[1] > tolerance:
            new_groups["Others"].append((gmin, grid_max))

    return new_groups


def generate_atom_list(atoms, molecule, crystal, keyword="ATOMS", lines=None):
    """

    :param atoms:
    :param molecule:
    :param crystal:
    :param keyword:
    :param lines:
    :return:
    """
    if lines is None:
        lines = []
    idx_mol = len(lines) + 1
    for mol in crystal.load_molecules():
        if molecule.residue == mol.residue:
            line = "{}{}=".format(keyword, idx_mol)
            for atom in atoms:
                atom_idx = atom + mol.index * mol.natoms
                line += str(atom_idx) + ","
            line = line[:-1] + "\n"
            lines.append(line)
            idx_mol += 1
    crystal.molecules = list()
    return lines
