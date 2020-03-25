class Torsions(object):

    def __init__(self, name, method):
        """
        Generates a distribution of the torsional angles of the selected atoms.
        :param name: str, name of the collective variable. Default output and variables will have this name.
        :param method: obj, method used to identify topology parameters and crystal structures.
        """
        import numpy as np
        self.type = "Torsional Angle"
        self.clustering_type = "distribution"
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
        Error: check if 2 atoms are involved.
        Select atom indices of the reference molecule. This is used to identify the torsions of each molecule in the
        crystal.
        :param atoms: list, Atom indices. All atoms indices are available in the project output file after the topology
        is defined.
        :param molecule: obj, Reference molecule
        :return:
        """
        self.atoms = list(atoms)
        self.molecule = molecule

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

    def set_bandwidth(self, bandwidth=0.5):
        """

        :param bandwidth:
        :return:
        """
        self.bandwidth = bandwidth

    def set_kernel(self, kernel="GAUSSIAN"):
        """

        :param kernel:
        :return:
        """
        self.kernel = kernel

    def set_clusteringtype(self, clusteringtype):
        """

        :param clusteringtype:
        :return:
        """
        if clusteringtype not in ["classification", "distribution"]:
            print("Clustering types available: 'classification' or 'distribution'")
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
                    return
            self.groups[name] = [groups]
        if sort_group:
            self.groups = sort_groups(self.grid_min, self.grid_max, self.groups)

    def generate_input(self, simulation):
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

        if self.clustering_type == "distribution":
            print("\nClustering type: Distribution\n"
                  "Parameters: KERNEL={0} NBINS={1} BANDWIDTH={2:.3f} UPPER={3:.3f} LOWER={4:.3f}"
                  "".format(self.kernel, self.grid_bin, self.bandwidth, self.grid_max, self.grid_min))
        elif self.clustering_type == "classification":
            print("\nClustering type: Classification")
            for group in self.groups.keys():
                print("Group {}:".format(group), end=" ")
                for boundary in self.groups[group]:
                    print(boundary, end=" ")
                print()

        for crystal in simulation.crystals:
            print(crystal.name)
            lines_atoms = generate_atom_list(self.atoms, self.molecule, crystal, keyword="ATOMS", lines=[])
            file_plumed = open(crystal.path + "plumed_" + self.name + ".dat", "w")
            file_plumed.write("TORSIONS ...\n")
            for line in lines_atoms:
                file_plumed.write(line)

            if self.clustering_type == "distribution":
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
                              "PRINT ARG={0}.* FILE=plumed_{1}_{0}.dat\n".format(self.name, simulation.name))
            file_plumed.close()
        print("=" * 100)


class MolecularOrientation(object):

    def __init__(self, name, method):
        """
        Error: Be more specific in creating N-Dimensional CV.
        Generates a distribution of the intermolecular torsional angles of the selected atoms.
        :param name: str, name of the collective variable. Default output and variables will have this name.
        :param method: obj, method used to identify topology parameters and crystal structures.
        """
        import numpy as np
        self.type = "Molecular Orientation"
        self.clustering_type = "distribution"  # No "classification" possible, unless classification based on the grid?
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
        Error: check that 2 atoms are involved
        :param atoms:
        :param molecule:
        :return:
        """
        self.atoms.append(list(atoms))
        self.molecules.append(molecule)

    def remove_atoms(self, index="all"):
        if index == "all":
            self.atoms.clear()
            self.molecules.clear()
        elif isinstance(index, int):
            del self.atoms[index]
            del self.molecules[index]
        else:
            print("Error: not clear which set of atoms you want to delete.")

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

    def set_bandwidth(self, bandwidth=0.5):
        """

        :param bandwidth:
        :return:
        """
        self.bandwidth = bandwidth

    def set_kernel(self, kernel="GAUSSIAN", bandwidth=None):
        """

        :param bandwidth:
        :param kernel:
        :return:
        """
        self.kernel = kernel.upper()

        if bandwidth:
            self.bandwidth = bandwidth

    def generate_input(self, simulation):
        """
        Error: Modify for N-Dimensional CV
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
            print("\nMolecule '{}': ".format(self.molecules[idx_mol].residue))
            for atom in self.atoms[idx_mol]:
                print("{}({})".format(atom, self.molecules[idx_mol].atoms[atom].label), end="  ")

        print("\nClustering type: Distribution\n"
              "Parameters: KERNEL={0} NBINS={1} BANDWIDTH={2:.3f} UPPER={3:.3f} LOWER={4:.3f}"
              "".format(self.kernel, self.grid_bin, self.bandwidth, self.grid_max, self.grid_min))

        for crystal in simulation.crystals:
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
                              "PRINT ARG=valg_{0} FILE=plumed_{6}_{0}.dat\n"
                              "".format(self.name, self.grid_min, self.grid_max,
                                        self.grid_bin, self.bandwidth, self.kernel, simulation.name))
            file_plumed.close()
        print("=" * 100)


class RDF(object):

    def __init__(self, name, method, center="geometrical"):
        """

        :param name:
        :param center:
        """
        self.name = name
        self.center = center
        method.cv.append(self)  # Check if cv exists
        self.method = method

        self.atoms = list()
        self.molecules = list()
        self.type = "Radial Distribution Function"
        self.clustering_type = "distribution"

        self.switching_function = "RATIONAL"
        self.r_0 = 0.01

        self.kernel = "GAUSSIAN"
        self.binspace = 0.01
        self.bandwidth = 0.01

    def set_kernel(self, kernel="GAUSSIAN", bandwidth=None, binspace=None):
        """

        :param kernel:
        :param bandwidth:
        :param binspace:
        :return:
        """
        self.kernel = kernel.upper()

        if bandwidth:
            self.bandwidth = bandwidth
        if binspace:
            self.binspace = binspace

    def set_switching_function(self, switching_function="RATIONAL", r_0=None):
        """

        :param switching_function:
        :param r_0:
        :return:
        """
        self.switching_function = switching_function.upper()
        if r_0:
            self.r_0 = r_0

    def set_atoms(self, atoms, molecule, overwrite=True):
        """

        :param overwrite:
        :param atoms:
        :param molecule:
        :return:
        """
        for idx_mol in range(len(self.molecules)):
            ref_mol = self.molecules[idx_mol]
            if ref_mol.residue == molecule.residue and overwrite:
                del self.molecules[idx_mol]
                del self.atoms[idx_mol]

        if atoms == "all":
            atoms = list()
            for atom in molecule.atoms:
                atoms.append(atom.index)
        elif atoms == "non-hydrogen":
            atoms = list()
            for atom in molecule.atoms:
                if atom.element.upper() != "H":
                    atoms.append(atom.index)
        self.atoms.append(list(atoms))
        self.molecules.append(molecule)

    def remove_atoms(self, index="all"):
        if index == "all":
            self.atoms.clear()
            self.molecules.clear()
        elif isinstance(index, int):
            del self.atoms[index]
            del self.molecules[index]
        else:
            print("Error: not clear which set of atoms you want to delete.")

    def generate_input(self, simulation):
        """

        :param simulation:
        :return:
        """
        import numpy as np

        if not self.atoms:
            print("Error: no atoms found. Select atoms with the set_atoms module.")
            exit()
        if simulation not in self.method.energy_minimisation+self.method.molecular_dynamics+self.method.metadynamics:
            print("Error: simulation {} not found in method {}.".format(simulation.name, self.method.name))
            exit()

        print("=" * 100)
        print("Generate plumed input files")
        print("CV: {} ({})".format(self.name, self.type))
        print("Atoms:", end=" ")
        for idx_mol in range(len(self.molecules)):
            print("\nMolecule '{}': ".format(self.molecules[idx_mol].residue))
            for atom in self.atoms[idx_mol]:
                print("{}({})".format(atom, self.molecules[idx_mol].atoms[atom].label), end="  ")

        print("\nClustering type: Distribution\n"
              "Parameters: KERNEL={0} BANDWIDTH={1:.3f} LOWER={2:.3f} BIN_SPACE={3:.3f}"
              "".format(self.kernel, self.bandwidth, self.r_0, self.binspace))

        for crystal in simulation.crystals:
            print(crystal.name)

            d_max = 0.5 * np.min(np.array([crystal.box[0:0], crystal.box[1:1], crystal.box[2:2]]))
            nbins = int(round((d_max - self.r_0)/self.binspace, 0))

            lines_atoms = []
            for idx_mol in range(len(self.molecules)):
                lines_atoms = generate_atom_list(self.atoms[idx_mol], self.molecules[idx_mol], crystal,
                                                 keyword="ATOMS", lines=lines_atoms)

            file_plumed = open(crystal.path + "plumed_" + self.name + ".dat", "w")
            idx_com = 1
            str_group = ""
            if self.center == "geometrical":
                for line in lines_atoms:
                    file_plumed.write("{}_c{}: CENTER {}".format(self.name, idx_com, line))
                    str_group += "{}_c{},".format(self.name, idx_com)
            elif self.center.upper() == "COM":
                for line in lines_atoms:
                    file_plumed.write("{}_c{}: COM {}".format(self.name, idx_com, line))
                    str_group += "{}_c{},".format(self.name, idx_com)

            str_group = str_group[:-1]
            file_plumed.write("{0}_g: GROUP ATOMS={1}\n"
                              "{0}_d: DISTANCES GROUP={0}_g MORE_THAN={{RATIONAL R_0={2} D_0={3} D_MAX={3}}} "
                              "HISTOGRAM={{{6} NBINS={5} BANDWIDTH={4} UPPER={3} LOWER={2}}}\n"
                              "PRINT ARG={0}_d.* FILE=plumed_{7}_{0}.dat\n\n"
                              "".format(self.name, str_group, self.r_0, d_max, self.bandwidth,
                                        nbins, self.kernel, simulation.name))
            file_plumed.close()


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

    if not new_groups["Others"]:
        del new_groups["Others"]

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
