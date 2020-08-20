#
# Collective Variables
#


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
        self.bandwidth = 0.3

        self.grid_min = -np.pi
        self.grid_max = np.pi
        self.grid_bin = 37
        self.timeinterval = 200

        self.groups = {}
        self.group_bins = {}

        for cv in self.method.cvs:
            if cv.name == self.name:
                print("Error: CV with label {} already present in this method. Remove it with 'del method.cvs[{}]' or "
                      "change CV label".format(self.name, self.method.cvs.index(cv)))
                exit()
        self.method.cvs.append(self)
        self.method.project.save()

    def set_time_interval(self, time, time2=False):
        if time2:
            self.timeinterval = (time, time2)
        else:
            self.timeinterval = time
        self.method.project.save()

    def set_atoms(self, atoms, molecule):
        """
        Error: check if 4 atoms are involved.
        Select atom indices of the reference molecule. This is used to identify the torsions of each molecule in the
        crystal.
        :param atoms: list, Atom indices. All atoms indices are available in the project output file after the topology
        is defined.
        :param molecule: obj, Reference molecule
        :return:
        """
        self.atoms = list(atoms)
        self.molecule = molecule
        self.method.project.save()

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
        self.method.project.save()

    def set_bandwidth(self, bandwidth=0.3):
        """

        :param bandwidth:
        :return:
        """
        self.bandwidth = bandwidth
        self.method.project.save()

    def set_kernel(self, kernel="GAUSSIAN"):
        """

        :param kernel:
        :return:
        """
        self.kernel = kernel
        self.method.project.save()

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
            self.groups["Others"] = list()
        self.method.project.save()

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
                    self.method.project.save()
                    return

            self.groups[name] = [groups]

        if sort_group:
            self.groups = sort_groups(self.grid_min, self.grid_max, self.groups)
        self.method.project.save()

    def write_output(self, path_output):
        file_output = open(path_output, "a")
        file_output.write("\nCV: {} ({})".format(self.name, self.type))
        if self.atoms:
            file_output.write("Atoms:  ")
            for atom in self.atoms:
                file_output.write("{}({})  ".format(atom, self.molecule.atoms[atom].label))
        else:
            file_output.write("No atoms found in CV {}. Select atoms with the 'set_atoms' module.\n"
                              "".format(self.name))

        if self.clustering_type == "distribution":
            file_output.write("\nClustering type: Distribution\n"
                              "Parameters: KERNEL={0} NBINS={1} BANDWIDTH={2:.3f} UPPER={3:.3f} LOWER={4:.3f}\n"
                              "".format(self.kernel, self.grid_bin, self.bandwidth, self.grid_max, self.grid_min))
        elif self.clustering_type == "classification":
            file_output.write("\nClustering type: Classification\n")
            for group in self.groups.keys():
                file_output.write("Group {}: ".format(group))
                for boundary in self.groups[group]:
                    file_output.write("{} ".format(boundary))
                file_output.write("\n")
        file_output.close()

    def generate_input(self, simulation, bash_script=True):
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
                    self.group_bins[group] = list()
                    for boundary in self.groups[group]:
                        self.group_bins[group].append(idx_boundaries - 1)
                        file_plumed.write("BETWEEN{0}={{{{{1} LOWER={2:.3f} UPPER={3:.3f} SMEAR=0.1}}}}\n"
                                          "".format(idx_boundaries, self.kernel, boundary[0], boundary[1]))
                        idx_boundaries += 1
            file_plumed.write("LABEL={0}\n... TORSIONS\n\n"
                              "PRINT ARG={0}.* FILE=plumed_{1}_{0}.dat\n".format(self.name, simulation.name))
            file_plumed.close()

        if bash_script:
            dt, nsteps, traj_stride, traj_start, traj_end = (None, None, None, None, None)

            file_mdp = open(simulation.mdp)
            for line in file_mdp:
                if line.startswith('dt '):
                    dt = float(line.split()[2])
                elif line.startswith(("nstxout", "nstxout-compressed")):
                    traj_stride = int(line.split()[2])
                elif line.startswith('nsteps '):
                    nsteps = float(line.split()[2])
            file_mdp.close()

            traj_time = int(nsteps * dt)
            if isinstance(self.timeinterval, tuple):
                traj_start = self.timeinterval[0]
                traj_end = self.timeinterval[1]
            elif isinstance(self.timeinterval, int):
                traj_start = traj_time - self.timeinterval
                traj_end = traj_time
            else:
                print("Error: No suitable time interval.")
                exit()

            file_script = open(simulation.path_data + "/run_plumed_" + self.name + ".sh", "w")
            file_script.write('#!/bin/bash\n\n'
                              'crystal_paths="\n')
            for crystal in simulation.crystals:
                file_script.write(crystal.path + "\n")
            file_script.write('"\n\n'
                              'for crystal in $crystal_paths ; do\n'
                              'cd "$crystal" || exit \n'
                              '{0} trjconv -f {1}.xtc -o plumed_{1}.xtc -s {1}.tpr -b {2} -e {3} <<< 0\n'
                              '{4} driver --mf_xtc plumed_{1}.xtc --plumed plumed_{5}.dat --timestep {6} '
                              '--trajectory-stride {7} --mc mc.dat\n'
                              'rm plumed_{1}.xtc\n'
                              'done\n'
                              ''.format(simulation.command, simulation.name, traj_start, traj_end,
                                        self.method.project.plumed, self.name, dt, traj_stride))
            file_script.close()
        self.method.project.save()
        print("=" * 100)

    def check_normal_termination(self, simulation, crystals="all"):
        import numpy as np
        import os
        from PyPol.utilities import get_list

        if crystals == "all":
            list_crystals = list()
            for crystal in simulation.crystals:
                if crystal.completed:
                    list_crystals.append(crystal)
        else:
            list_crystals = get_list(crystals)

        for crystal in list_crystals:
            path_output = crystal.path + "plumed_{}_{}.dat".format(simulation.name, self.name)
            if os.path.exists(path_output):
                cv = np.genfromtxt(path_output, skip_header=1)[:, 1:]
                cv = np.average(cv, axis=0)
                cv /= cv.sum()
                if self.clustering_type == "distribution":
                    crystal.cvs[self.name] = cv
                elif self.clustering_type == "classification":
                    crystal.cvs[self.name] = {}
                    for group_name in self.group_bins.keys():
                        crystal.cvs[self.name][group_name] = np.sum(cv[self.group_bins[group_name]])

            else:
                print("An error has occurred with Plumed. Check file {} in folder {}."
                      "".format(path_output, crystal.path))
        self.method.project.save()


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
        self.bandwidth = 0.3

        self.grid_min = 0.0
        self.grid_max = np.pi
        self.grid_bin = 19

        self.timeinterval = 200

        for cv in self.method.cvs:
            if cv.name == self.name:
                print("Error: CV with label {} already present in this method. Remove it with 'del method.cvs[{}]' or "
                      "change CV label".format(self.name, self.method.cvs.index(cv)))
                exit()
        self.method.cvs.append(self)
        self.method.project.save()

    def set_time_interval(self, time, time2=False):
        if time2:
            self.timeinterval = (time, time2)
        else:
            self.timeinterval = time
        self.method.project.save()

    def set_atoms(self, atoms, molecule):
        """
        # TODO check that 2 atoms are involved and that molecule is not already present
        :param atoms:
        :param molecule:
        :return:
        """
        self.atoms.append(list(atoms))
        self.molecules.append(molecule)
        self.method.project.save()

    def remove_atoms(self, index="all"):
        if index == "all":
            self.atoms.clear()
            self.molecules.clear()
        elif isinstance(index, int):
            del self.atoms[index]
            del self.molecules[index]
        else:
            print("Error: not clear which set of atoms you want to delete.")
        self.method.project.save()

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
        self.method.project.save()

    def set_bandwidth(self, bandwidth=0.3):
        """

        :param bandwidth:
        :return:
        """
        self.bandwidth = bandwidth
        self.method.project.save()

    def set_kernel(self, kernel="GAUSSIAN", bandwidth=None):
        """

        :param bandwidth:
        :param kernel:
        :return:
        """
        self.kernel = kernel.upper()

        if bandwidth:
            self.bandwidth = bandwidth
        self.method.project.save()

    def write_output(self, path_output):
        file_output = open(path_output, "a")
        file_output.write("CV: {} ({})\n".format(self.name, self.type))
        if self.atoms:
            file_output.write("Atoms: ")
            for idx_mol in range(len(self.molecules)):
                file_output.write("\nMolecule '{}': ".format(self.molecules[idx_mol].residue))
                for atom in self.atoms[idx_mol]:
                    file_output.write("{}({})".format(atom, self.molecules[idx_mol].atoms[atom].label))
        else:
            file_output.write("No atoms found in CV {}. Select atoms with the 'set_atoms' module.\n"
                              "".format(self.name))

        file_output.write("\nClustering type: Distribution\n"
                          "Parameters: KERNEL={0} NBINS={1} BANDWIDTH={2:.3f} UPPER={3:.3f} LOWER={4:.3f}\n"
                          "".format(self.kernel, self.grid_bin, self.bandwidth, self.grid_max, self.grid_min))
        file_output.close()

    def generate_input(self, simulation, bash_script=True):
        """

        :param simulation:
        :param bash_script:
        :return:
        """

        # TODO Change creation of parameters. Depends on molecules and number of vectors for each of them!
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

        if bash_script:
            dt, nsteps, traj_stride, traj_start, traj_end = (None, None, None, None, None)

            file_mdp = open(simulation.mdp)
            for line in file_mdp:
                if line.startswith('dt '):
                    dt = float(line.split()[2])
                elif line.startswith(("nstxout", "nstxout-compressed")):
                    traj_stride = int(line.split()[2])
                elif line.startswith('nsteps '):
                    nsteps = float(line.split()[2])
            file_mdp.close()

            traj_time = int(nsteps * dt)
            if isinstance(self.timeinterval, tuple):
                traj_start = self.timeinterval[0]
                traj_end = self.timeinterval[1]
            elif isinstance(self.timeinterval, int):
                traj_start = traj_time - self.timeinterval
                traj_end = traj_time
            else:
                print("Error: No suitable time interval.")
                exit()

            file_script = open(simulation.path_data + "/run_plumed_" + self.name + ".sh", "w")
            file_script.write('#!/bin/bash\n\n'
                              'crystal_paths="\n')
            for crystal in simulation.crystals:
                if crystal.completed:
                    file_script.write(crystal.path + "\n")
            file_script.write('"\n\n'
                              'for crystal in $crystal_paths ; do\n'
                              'cd "$crystal" || exit \n'
                              '{0} trjconv -f {1}.xtc -o plumed_{1}.xtc -s {1}.tpr -b {2} -e {3} <<< 0\n'
                              '{4} driver --mf_xtc plumed_{1}.xtc --plumed plumed_{5}.dat --timestep {6} '
                              '--trajectory-stride {7} --mc mc.dat\n'
                              'rm plumed_{1}.xtc\n'
                              'done\n'
                              ''.format(simulation.command, simulation.name, traj_start, traj_end,
                                        self.method.project.htt_plumed, self.name, dt, traj_stride))
            file_script.close()
        self.method.project.save()
        print("=" * 100)

    def check_normal_termination(self, simulation, crystals="all"):
        import numpy as np
        import os
        from PyPol.utilities import get_list

        if crystals == "all":
            list_crystals = list()
            for crystal in simulation.crystals:
                if crystal.completed:
                    list_crystals.append(crystal)
        else:
            list_crystals = get_list(crystals)

        for crystal in list_crystals:
            path_output = crystal.path + "plumed_{}_{}.dat".format(simulation.name, self.name)
            if os.path.exists(path_output):
                cv = np.genfromtxt(path_output, skip_header=2)[:, 1:]
                cv = np.average(cv, axis=0)
                crystal.cvs[self.name] = cv
            else:
                print("An error has occurred with Plumed. Check file {} in folder {}."
                      "".format(path_output, crystal.path))
        self.method.project.save()

    def identify_melted_structures(self, simulation, crystals="all", tolerance=0.1):
        import numpy as np
        from PyPol.utilities import get_list

        if self.grid_min != 0. and self.grid_max != np.pi:
            print("Error: A range between 0 and pi must be used to identify melted structures.")
            exit()

        if crystals == "all":
            list_crystals = list()
            for crystal in simulation.crystals:
                if crystal.completed:
                    list_crystals.append(crystal)
        else:
            list_crystals = get_list(crystals)

        file_hd = open("{}/HD_{}.dat".format(simulation.path_output, simulation.name), "w")
        file_hd.write("# Tolerance = {}\n#\n# Structures HD".format(round(tolerance, 5)))
        ref = np.sin(np.linspace(0., np.pi, self.grid_bin))  # Add "+ 1" ?
        for crystal in list_crystals:
            if not self.name in crystal.cvs:
                print("Error: A distribution for this simulation has not been generated.\n"
                      "Remember to run the check_normal_termination after running plumed.")
                exit()
            hd = hellinger(crystal.cvs[self.name], ref)
            file_hd.write("{:35} {:3.3f}\n".format(crystal.name, hd))
            if hd < tolerance:
                crystal.melted = True
            else:
                crystal.melted = False
        file_hd.close()
        self.method.project.save()


class Combine(object):

    def __init__(self, name, method, cvs):
        self.name = name
        self.type = "N-Dimensional CV"
        self.method = method
        self.cvs = cvs

        self.kernel = cvs[0].kernel
        self.timeinterval = cvs[0].timeinterval
        self.list_bins = ()

        for cv in self.method.cvs:
            if cv.name == self.name:
                print("Error: CV with label {} already present in this method. Remove it with 'del method.cvs[{}]' or "
                      "change CV label".format(self.name, self.method.cvs.index(cv)))
                exit()
        self.method.cvs.append(self)
        self.method.project.save()

    def set_time_interval(self, time, time2=False):
        if time2:
            self.timeinterval = (time, time2)
        else:
            self.timeinterval = time
        self.method.project.save()

    def set_kernel(self, kernel="GAUSSIAN"):
        """

        :param kernel:
        :return:
        """
        self.kernel = kernel.upper()
        self.method.project.save()

    def write_output(self, path_output):
        idx_cv = 0
        grid_min, grid_max, grid_bin, bandwidth, args = ("", "", "", "", "")
        file_output = open(path_output, "a")
        file_output.write("\nCV: {} ({})".format(self.name, self.type))
        for cv in self.cvs:
            file_output.write("CV{}: {} ({})".format(idx_cv, cv.name, cv.type))
            if not cv.atoms:
                file_output.write("No atoms found in CV {}. Select atoms with the 'set_atoms' module.\n"
                                  "".format(cv.name))
            else:
                file_output.write("Atoms:\n")
                for idx_mol in range(len(cv.molecules)):
                    file_output.write("\nMolecule '{}': ".format(cv.molecules[idx_mol].residue))
                    for atom in cv.atoms[idx_mol]:
                        file_output.write("{}({})  ".format(atom, cv.molecules[idx_mol].atoms[atom].label))
            grid_min += "{:.3f},".format(cv.grid_min)
            grid_max += "{:.3f},".format(cv.grid_max)
            grid_bin += "{},".format(cv.grid_bin)
            bandwidth += "{:.3f},".format(cv.grid_min)
            idx_cv += 1

        file_output.write("\nClustering type: {5}-D Distribution\n"
                          "Parameters: KERNEL={0} NBINS={1} BANDWIDTH={2} UPPER={3} LOWER={4}"
                          "".format(self.kernel, grid_bin, bandwidth, grid_max, grid_min, len(self.cvs)))
        file_output.close()

    def generate_input(self, simulation, bash_script=True):
        """
        TODO output format, only one cv is printed!
        :param simulation:
        :param bash_script:
        :return:
        """
        idx_cv = 0
        list_bins = list()
        grid_min, grid_max, grid_bin, bandwidth, args = ("", "", "", "", "")
        print("=" * 100)
        print("Generate plumed input files")
        print("CV: {} ({})".format(idx_cv, self.name, self.type))
        for cv in self.cvs:
            if not cv.atoms:
                print("Error: no atoms found in CV {}. select atoms with the set_atoms module.".format(cv.name))
                exit()

            print("CV{}: {} ({})".format(idx_cv, cv.name, cv.type))
            print("Atoms:", end=" ")
            for idx_mol in range(len(cv.molecules)):
                print("\nMolecule '{}': ".format(cv.molecules[idx_mol].residue))
                for atom in cv.atoms[idx_mol]:
                    print("{}({})".format(atom, cv.molecules[idx_mol].atoms[atom].label), end="  ")

            grid_min += "{:.3f},".format(cv.grid_min)
            grid_max += "{:.3f},".format(cv.grid_max)
            grid_bin += "{},".format(cv.grid_bin)
            list_bins.append(int(cv.grid_bin))
            bandwidth += "{:.3f},".format(cv.bandwidth)
            args += "ARG{}=ang_mat_{} ".format(idx_cv + 1, cv.name)
            idx_cv += 1
        self.list_bins = tuple(list_bins)
        print("\nClustering type: {5}-D Distribution\n"
              "Parameters: KERNEL={0} NBINS={1} BANDWIDTH={2} UPPER={3} LOWER={4}"
              "".format(self.kernel, grid_bin, bandwidth, grid_max, grid_min, len(self.cvs)))

        for crystal in simulation.crystals:
            print(crystal.name)
            file_plumed = open(crystal.path + "plumed_" + self.name + ".dat", "w")
            for cv in self.cvs:
                # Select atoms and molecules
                lines_atoms = []
                for idx_mol in range(len(cv.molecules)):
                    lines_atoms = generate_atom_list(cv.atoms[idx_mol], cv.molecules[idx_mol], crystal,
                                                     keyword="ATOMS", lines=lines_atoms)

                file_plumed.write("DISTANCE ...\n")
                for line in lines_atoms:
                    file_plumed.write(line)
                file_plumed.write("LABEL=dd_{0}\n"
                                  "COMPONENTS\n"
                                  "... DISTANCE\n\n"
                                  "vv_{0}: NORMALIZE ARG1=dd_{0}.x ARG2=dd_{0}.y ARG3=dd_{0}.z\n"
                                  "dp_mat_{0}: DOTPRODUCT_MATRIX GROUP1=vv_{0}.x GROUP2=vv_{0}.y GROUP3=vv_{0}.z\n"
                                  "ang_mat_{0}: MATHEVAL ARG1=dp_mat_{0} FUNC=acos(x) PERIODIC=NO\n\n"
                                  "".format(cv.name))

            file_plumed.write("valg_{0}: KDE {7} GRID_MIN={1} GRID_MAX={2} "
                              "GRID_BIN={3} BANDWIDTH={4} KERNEL={5}\n\n"
                              "PRINT ARG=valg_{0} FILE=plumed_{6}_{0}.dat\n"
                              "".format(self.name, grid_min, grid_max,
                                        grid_bin, bandwidth, self.kernel, simulation.name, args))
            file_plumed.close()

        if bash_script:
            dt, nsteps, traj_stride, traj_start, traj_end = (None, None, None, None, None)

            file_mdp = open(simulation.mdp)
            for line in file_mdp:
                if line.startswith('dt '):
                    dt = float(line.split()[2])
                elif line.startswith(("nstxout", "nstxout-compressed")):
                    traj_stride = int(line.split()[2])
                elif line.startswith('nsteps '):
                    nsteps = float(line.split()[2])
            file_mdp.close()

            traj_time = int(nsteps * dt)
            if isinstance(self.timeinterval, tuple):
                traj_start = self.timeinterval[0]
                traj_end = self.timeinterval[1]
            elif isinstance(self.timeinterval, int):
                traj_start = traj_time - self.timeinterval
                traj_end = traj_time
            else:
                print("Error: No suitable time interval.")
                exit()

            file_script = open(simulation.path_data + "/run_plumed_" + self.name + ".sh", "w")
            file_script.write('#!/bin/bash\n\n'
                              'crystal_paths="\n')
            for crystal in simulation.crystals:
                file_script.write(crystal.path + "\n")
            file_script.write('"\n\n'
                              'for crystal in $crystal_paths ; do\n'
                              'cd "$crystal" || exit \n'
                              '{0} trjconv -f {1}.xtc -o plumed_{1}.xtc -s {1}.tpr -b {2} -e {3} <<< 0\n'
                              '{4} driver --mf_xtc plumed_{1}.xtc --plumed plumed_{5}.dat --timestep {6} '
                              '--trajectory-stride {7} --mc mc.dat\n'
                              'rm plumed_{1}.xtc\n'
                              'done\n'
                              ''.format(simulation.command, simulation.name, traj_start, traj_end,
                                        self.method.project.htt_plumed, self.name, dt, traj_stride))
            file_script.close()
        self.method.project.save()

        print("=" * 100)

    def check_normal_termination(self, simulation, crystals="all"):
        import numpy as np
        import os
        from PyPol.utilities import get_list

        if crystals == "all":
            list_crystals = list()
            for crystal in simulation.crystals:
                if crystal.completed:
                    list_crystals.append(crystal)
        else:
            list_crystals = get_list(crystals)

        for crystal in list_crystals:
            path_output = crystal.path + "plumed_{}_{}.dat".format(simulation.name, self.name)
            if os.path.exists(path_output):
                cv_dist = np.genfromtxt(path_output, skip_header=1)[:, 1:]
                cv_dist = np.average(cv_dist, axis=0)
                crystal.cvs[self.name] = cv_dist.reshape(self.list_bins)
            else:
                print("An error has occurred with Plumed. Check file {} in folder {}."
                      "".format(path_output, crystal.path))
        self.method.project.save()


class RDF(object):

    def __init__(self, name, method, center="geometrical"):
        """

        :param name:
        :param center:
        """
        self.name = name
        self.center = center
        self.method = method

        for cv in self.method.cvs:
            if cv.name == self.name:
                print("Error: CV with label {} already present in this method. Remove it with 'del method.cvs[{}]' or "
                      "change CV label".format(self.name, self.method.cvs.index(cv)))
                exit()

        self.atoms = list()
        self.molecules = list()
        self.type = "Radial Distribution Function"
        self.clustering_type = "distribution"

        self.switching_function = "RATIONAL"
        self.r_0 = 0.01

        self.kernel = "GAUSSIAN"
        self.binspace = 0.01
        self.bandwidth = 0.01

        self.timeinterval = 200

        self.method.cvs.append(self)
        self.method.project.save()

    def set_time_interval(self, time, time2=False):
        if time2:
            self.timeinterval = (time, time2)
        else:
            self.timeinterval = time
        self.method.project.save()

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
        self.method.project.save()

    def set_switching_function(self, switching_function="RATIONAL", r_0=None):
        """

        :param switching_function:
        :param r_0:
        :return:
        """
        self.switching_function = switching_function.upper()
        if r_0:
            self.r_0 = r_0
        self.method.project.save()

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
        self.method.project.save()

    def remove_atoms(self, index="all"):
        if index == "all":
            self.atoms.clear()
            self.molecules.clear()
        elif isinstance(index, int):
            del self.atoms[index]
            del self.molecules[index]
        else:
            print("Error: not clear which set of atoms you want to delete.")
        self.method.project.save()

    def write_output(self, path_output):
        file_output = open(path_output, "a")
        file_output.write("CV: {} ({})".format(self.name, self.type))
        if self.atoms:
            file_output.write("Atoms: ")
            for idx_mol in range(len(self.molecules)):
                file_output.write("\nMolecule '{}': ".format(self.molecules[idx_mol].residue))
                for atom in self.atoms[idx_mol]:
                    file_output.write("{}({})".format(atom, self.molecules[idx_mol].atoms[atom].label))
        else:
            file_output.write("No atoms found in CV {}. Select atoms with the 'set_atoms' module."
                              "".format(self.name))
        file_output.write("\nClustering type: Distribution\n"
                          "Parameters: KERNEL={0} BANDWIDTH={1:.3f} LOWER={2:.3f} BIN_SPACE={3:.3f}\n"
                          "".format(self.kernel, self.bandwidth, self.r_0, self.binspace))
        file_output.close()

    def generate_input(self, simulation, bash_script=True):
        """

        :param bash_script:
        :param simulation:
        :return:
        """
        import numpy as np

        if not self.atoms:
            print("Error: no atoms found. Select atoms with the set_atoms module.")
            exit()
        if simulation not in self.method.energy_minimisation + self.method.molecular_dynamics + \
                self.method.metadynamics:
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

            d_max = 0.5 * np.min(np.array([crystal.box[0, 0], crystal.box[1, 1], crystal.box[2, 2]]))
            nbins = int(round((d_max - self.r_0) / self.binspace, 0))

            lines_atoms = []
            for idx_mol in range(len(self.molecules)):
                lines_atoms = generate_atom_list(self.atoms[idx_mol], self.molecules[idx_mol], crystal,
                                                 keyword="ATOMS", lines=lines_atoms, index_lines=False)

            file_plumed = open(crystal.path + "plumed_" + self.name + ".dat", "w")
            idx_com = 1
            str_group = ""
            if self.center == "geometrical":
                for line in lines_atoms:
                    file_plumed.write("{}_c{}: CENTER {}".format(self.name, idx_com, line))
                    str_group += "{}_c{},".format(self.name, idx_com)
                    idx_com += 1
            elif self.center.upper() == "COM":
                for line in lines_atoms:
                    file_plumed.write("{}_c{}: COM {}".format(self.name, idx_com, line))
                    str_group += "{}_c{},".format(self.name, idx_com)
                    idx_com += 1

            str_group = str_group[:-1]
            file_plumed.write("{0}_g: GROUP ATOMS={1}\n"
                              "{0}_d: DISTANCES GROUP={0}_g MORE_THAN={{RATIONAL R_0={2} D_0={3} D_MAX={3}}} "
                              "HISTOGRAM={{{6} NBINS={5} BANDWIDTH={4} UPPER={3} LOWER={2}}}\n"
                              "PRINT ARG={0}_d.* FILE=plumed_{7}_{0}.dat\n\n"
                              "".format(self.name, str_group, self.r_0, d_max, self.bandwidth,
                                        nbins, self.kernel, simulation.name))
            file_plumed.close()

        if bash_script:

            dt, nsteps, traj_stride, traj_start, traj_end = (None, None, None, None, None)

            file_mdp = open(simulation.mdp)
            for line in file_mdp:
                if line.startswith('dt '):
                    dt = float(line.split()[2])
                elif line.startswith(("nstxout", "nstxout-compressed")):
                    traj_stride = int(line.split()[2])
                elif line.startswith('nsteps '):
                    nsteps = float(line.split()[2])
            file_mdp.close()

            traj_time = int(nsteps * dt)
            if isinstance(self.timeinterval, tuple):
                traj_start = self.timeinterval[0]
                traj_end = self.timeinterval[1]
            elif isinstance(self.timeinterval, int):
                traj_start = traj_time - self.timeinterval
                traj_end = traj_time
            else:
                print("Error: No suitable time interval.")
                exit()

            file_script = open(simulation.path_data + "/run_plumed_" + self.name + ".sh", "w")
            file_script.write('#!/bin/bash\n\n'
                              'crystal_paths="\n')
            for crystal in simulation.crystals:
                file_script.write(crystal.path + "\n")
            file_script.write('"\n\n'
                              'for crystal in $crystal_paths ; do\n'
                              'cd "$crystal" || exit \n'
                              '{0} trjconv -f {1}.xtc -o plumed_{1}.xtc -s {1}.tpr -b {2} -e {3} <<< 0\n'
                              '{4} driver --mf_xtc plumed_{1}.xtc --plumed plumed_{5}.dat --timestep {6} '
                              '--trajectory-stride {7} --mc mc.dat\n'
                              'rm plumed_{1}.xtc\n'
                              'done\n'
                              ''.format(simulation.command, simulation.name, traj_start, traj_end,
                                        self.method.project.plumed, self.name, dt, traj_stride))
            file_script.close()
        self.method.project.save()

    def check_normal_termination(self, simulation, crystals="all"):
        import numpy as np
        import os
        from PyPol.utilities import get_list

        if crystals == "all":
            list_crystals = list()
            for crystal in simulation.crystals:
                if crystal.completed:
                    list_crystals.append(crystal)
        else:
            list_crystals = get_list(crystals)

        for crystal in list_crystals:
            path_output = crystal.path + "plumed_{}_{}.dat".format(simulation.name, self.name)
            if os.path.exists(path_output):
                dn_r = np.genfromtxt(path_output, skip_header=1)[:, 2:]
                dn_r = np.average(dn_r, axis=0)

                d_max = 0.5 * np.min(np.array([crystal.box[0, 0], crystal.box[1, 1], crystal.box[2, 2]]))
                nbins = int(round((d_max - self.r_0) / self.binspace, 0))
                r = np.linspace(self.r_0, d_max, nbins)
                rho = crystal.Z / crystal.volume  # Modify for more than one molecule?

                cv = np.where(r > 0, dn_r / (4 * np.pi * rho * r ** 2 * self.binspace), 0.)
                crystal.cvs[self.name] = cv
            else:
                print("An error has occurred with Plumed. Check file {} in folder {}."
                      "".format(path_output, crystal.path))
        self.method.project.save()


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
    for cv_ranges in new_groups.values():
        for cv_range in cv_ranges:
            ranges.append(cv_range)
    ranges.sort(key=lambda x: x[0])
    gmin = grid_min
    new_groups["Others"] = list()
    for cv_range in ranges:
        if cv_range[0] > gmin and cv_range[0] - gmin > tolerance:
            new_groups["Others"].append((gmin, cv_range[0]))
            gmin = cv_range[1]
        else:
            gmin = cv_range[1]
        if cv_range == ranges[-1] and cv_range[1] < grid_max and grid_max - cv_range[1] > tolerance:
            new_groups["Others"].append((gmin, grid_max))

    if not new_groups["Others"]:
        del new_groups["Others"]

    return new_groups


def generate_atom_list(atoms, molecule, crystal, keyword="ATOMS", lines=None, index_lines=True):
    """

    :param index_lines:
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
            if index_lines:
                line = "{}{}=".format(keyword, idx_mol)
            else:
                line = "{}=".format(keyword)
            for atom in atoms:
                atom_idx = atom + mol.index * mol.natoms + 1
                line += str(atom_idx) + ","
            line = line[:-1] + "\n"
            lines.append(line)
            idx_mol += 1
    crystal.molecules = list()
    return lines


#
# Clustering analysis
#


def hellinger(y1, y2, int_type="discrete"):
    """

    :param y1:
    :param y2:
    :param int_type:
    :return:
    """
    import numpy as np

    if int_type == "discrete":
        # Normalise Distributions
        y1 /= np.sum(y1)
        y2 /= np.sum(y2)

        BC = np.sum(np.sqrt(np.multiply(y1, y2)))
        HD = round(np.sqrt(1 - BC), 5)
        return HD

    elif int_type == "simps":
        from scipy.integrate import simps
        # Normalise Distributions
        N1, N2 = (y1, y2)
        for x in y1.shape:
            N1 = simps(N1, np.linspace(0, x - 1, x))
            N2 = simps(N2, np.linspace(0, x - 1, x))
        y1 /= N1
        y2 /= N2

        BC = np.sqrt(np.multiply(y1, y2))
        for x in y1.shape:
            BC = simps(BC, np.linspace(0, x - 1, x))
        HD = round(np.sqrt(1 - BC), 5)
        return HD

    elif int_type == "trapz":
        from scipy.integrate import trapz
        # Normalise Distributions
        N1, N2 = (y1, y2)
        for x in y1.shape:
            N1 = trapz(N1, np.linspace(0, x - 1, x))
            N2 = trapz(N2, np.linspace(0, x - 1, x))
        y1 /= N1
        y2 /= N2

        BC = np.sqrt(np.multiply(y1, y2))
        for x in y1.shape:
            BC = trapz(BC, np.linspace(0, x - 1, x))
        HD = round(np.sqrt(1 - BC), 5)
        return HD

    else:
        print("Error: choose integration type among 'simps', 'trapz' or 'discrete'.")
        exit()


def decision_graph(x, y):
    class PointPicker(object):
        def __init__(self, ax, scat, clicklim=0.05):
            self.fig = ax.figure
            self.ax = ax
            self.scat = scat
            self.clicklim = clicklim
            self.sigma_cutoff = 0.1
            self.horizontal_line = ax.axhline(y=.1, color='red', alpha=0.5)
            self.text = ax.text(0, 0.5, "")
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
    scat = ax.scatter(x, y, color="C0")

    plt.title(r"Select $\sigma$-cutoff and quit to continue", fontsize=20)
    plt.xlabel(r"$\rho$", fontsize=20)
    plt.ylabel(r"$\delta$", fontsize=20)
    p = PointPicker(ax, scat)
    plt.show()
    return p.sigma_cutoff


def FSFDP(dmat, kernel="gaussian", d_c="auto", cutoff_factor=0.02, sigma_cutoff=False):
    """
    Simplified FSFDP algorithm. # TODO Instead of halo, use distance of crystals from center.
    :param sigma_cutoff:
    :param dmat:
    :param kernel:
    :param d_c:
    :param cutoff_factor:
    :return:
    """
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if d_c == "auto":
        d_c = np.sort(dmat.values.flatten())[int(dmat.values.size * cutoff_factor) + dmat.values.shape[0]]

    # Find density vector
    rho = np.zeros(dmat.values.shape[0])
    if kernel == "gaussian":
        kernel_function = lambda d_ij: np.exp(-(d_ij / d_c) * (d_ij / d_c))
    elif kernel == "cutoff":
        kernel_function = lambda d_ij: 1 if d_ij < d_c else 0
    else:
        kernel_function = lambda d_ij: np.exp(-(d_ij / d_c) * (d_ij / d_c))
        print("Kernel Function not recognized, switching to 'gaussian'")

    for i in range(dmat.values.shape[0] - 1):
        for j in range(i + 1, dmat.values.shape[0]):
            rho[i] += kernel_function(dmat.values[i][j])
            rho[j] += kernel_function(dmat.values[i][j])

    rho = pd.Series(rho, index=dmat.index)

    # Find sigma vector
    sigma = pd.Series(np.full(rho.shape, -1.), dmat.index, name="sigma")
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
        sigma_cutoff = decision_graph(rho, sigma)

    # Assign structures to cluster centers
    dataset = pd.concat((rho, sigma, nn,
                         pd.Series(np.full(rho.shape, pd.NA), dmat.index, name="cluster"),
                         pd.Series(np.full(rho.shape, pd.NA), dmat.index, name="distance")),
                        axis=1).sort_values(by="rho", ascending=False)

    for i in dataset.index:
        if dataset.loc[i, "sigma"] >= sigma_cutoff:
            dataset.at[i, "cluster"] = i
            dataset.at[i, "distance"] = 0.0

        else:
            dataset.at[i, "cluster"] = dataset.loc[dataset.loc[i]["NN"]]["cluster"]
            dataset.at[i, "distance"] = dmat.loc[i, dataset.loc[i, "cluster"]]

    return dataset


class Clustering(object):

    def __init__(self, name, method, cvs):
        self.name = name
        self.method = method
        self.method.clustering_parameters.append(self)

        self.cvs = list()
        for cv in self.method.cvs:
            if cv.name in cvs:
                self.cvs.append(cv)
        if len(self.cvs) != len(cvs):
            print("Error: Not all CVs present in this method. CVs available:")
            for cv in self.method.cvs:
                print(cv.name)
            exit()

        self.int_type = "discrete"
        self.fsfdp_kernel = "gaussian"
        self.centers = "energy"
        self.d_c = []
        self.cutoff_factor = 0.

        self.similarity_matrix = False
        self.cluster_data = {}
        self.clusters = {}

    def set_center_selection(self, center_selection):
        if center_selection.lower() in ("energy", "cluster_center"):
            self.centers = center_selection.lower()
        else:
            print("Error: Center selection method not recognized. Choose between:\n"
                  "'energy'        : select structure with the lower potential energy in the group as cluster center.\n"
                  "'cluster_center': select the geometrical center resulting from the CVs used.")
            exit()

    def set_fsfdp_kernel(self, kernel):
        if kernel.lower() in ("gaussian", "cutoff"):
            self.centers = center_selection.lower()
        else:
            print("Error: Kernel function not recognized. Choose between 'gaussian' and 'cutoff'")
            exit()

    def set_hellinger_integration_type(self, int_type):
        if int_type.lower() in ("discrete", "simps", "trapz"):
            self.int_type = int_type.lower()
        else:
            print('Error: Hellinger integration type not recognized. Choose between "discrete", "simps" or "trapz"')
            exit()

    @staticmethod
    def sort_crystal(crystal, combinations, threshold=0.8):
        for i in combinations.index[:-1]:
            for j in combinations.columns[:-2]:
                if crystal.results[j][combinations.loc[i, j]] > threshold and j == combinations.columns[-3]:
                    combinations.loc[i, "Structures"].append(crystal)
                    combinations.loc[i, "Number of structures"] += 1
                    return combinations
                elif crystal.results[j][combinations.loc[i, j]] < threshold:
                    break
        combinations.loc["Others", "Structures"].append(crystal)
        combinations.loc["Others", "Number of structures"] += 1
        return combinations

    def run(self, simulation, group_threshold=0.8, gen_sim_mat=True):
        import numpy as np
        import pandas as pd
        import itertools as its
        import copy

        if gen_sim_mat:
            import progressbar
            # Divide structures in different groups
            group_options = []
            group_names = []
            for cv in self.cvs:
                if cv.clustering_type == "classification":
                    group_options.append(list(cv.groups.keys()))
                    group_names.append(cv.name)
            if group_options:
                combinations = list(its.product(*group_options)) + [tuple([None for i in range(len(group_options[0]))])]

                index = [i for i in range(len(combinations) - 1)] + ["Others"]
                combinations = pd.concat((pd.DataFrame(combinations, columns=group_names, index=index),
                                          pd.Series([0 for i in range(len(combinations))], name="Number of structures",
                                                    dtype=int, index=index),
                                          pd.Series([[] for i in range(len(combinations))], name="Structures",
                                                    index=index)), axis=1)
                combinations.index.name = "Combination"

                for crystal in simulation.crystals:
                    combinations = self.sort_crystal(crystal, combinations, group_threshold)

            else:
                index = ["All"]
                combinations = pd.concat((pd.Series(0, name="Number of structures", dtype=int, index=index),
                                          pd.Series([], name="Structures", index=index)), axis=1)
                combinations.index.name = "Combination"
                for crystal in simulation.crystals:
                    if not crystal.melted:
                        combinations.loc["all", "Structures"].append(crystal)
                        combinations.loc["all", "Number of structures"] += 1

            # Generate Distance Matrix of each set of distributions
            distributions = [cv for cv in self.cvs if cv.type != "classification"]
            n_factors = {}
            for cv in distributions:
                combinations[cv.name] = pd.Series(copy.deepcopy(combinations["Distance Matrix"].to_dict()),
                                                  index=combinations.index)
                n_factors[cv.name] = 0.

                for index, row in combinations.iterrows():
                    if row["Structures"]:
                        crystals = row["Structures"]

                        print("CV: {} Group: {}".format(cv.name, index))
                        bar = progressbar.ProgressBar(maxval=len(crystals)*(len(crystals)-1)).start()
                        bc = 1

                        for i in range(len(crystals) - 1):
                            for j in range(i + 1, len(crystals)):
                                hd = hellinger(crystals[i].results[cv.name], crystals[j].results[cv.name])
                                combinations.loc[index, cv.name][i, j] = combinations.loc[index, cv.name][j, i] = hd
                                if hd > n_factors[cv.name]:
                                    n_factors[cv.name] = hd
                                bar.update(bc)
                                bc += 1

            # Normalize distances
            normalization = []
            for cv in distributions:
                normalization.append(1. / n_factors[cv.name])
                for index, row in combinations.iterrows():
                    if row["Structures"]:
                        row[cv.name] /= n_factors[cv.name]

            # Generate Distance Matrix
            normalization = np.linalg.norm(np.array(normalization))
            for index, row in combinations.iterrows():
                if row["Structures"]:
                    for i in range(row["Number of structures"] - 1):
                        for j in range(i + 1, row["Number of structures"]):
                            dist_ij = np.linalg.norm([k[i, j] for k in row.loc[distributions]]) / normalization
                            row["Distance Matrix"][i, j] = row["Distance Matrix"][j, i] = dist_ij
                            self.d_c.append(dist_ij)

            for index, row in combinations.iterrows():
                if row["Structures"]:
                    idx = [i.name for i in row["Structures"]]
                    row["Distance Matrix"] = pd.DataFrame(row["Distance Matrix"], index=idx, columns=idx)

            self.similarity_matrix = combinations
            list_structures = [[i.name for i in row["Structures"]] for index, row in combinations.iterrows()]
            file_output = pd.concat((combinations.loc[:, :"Number of structures"],
                                     pd.Series(list_structures, name="IDs", index=group_df.index)), axis=1)
            pd.set_option('display.expand_frame_repr', False)
            with open(simulation.path_output + str(self.name) + "_combinations.dat", 'w') as fo:
                fo.write(file_output.__str__())

            self.d_c = np.sort(np.array(self.d_c))[int(float(len(self.d_c)) * self.cutoff_factor)]

        # Remove structures that are not cluster centers
        changes_string = "\n"
        for index, row in self.similarity_matrix.iterrows():
            if row["Structures"]:
                self.cluster_data[index] = FSFDP(row["Distance Matrix"], d_c=self.d_c)

                with open(simulation.path_output + str(self.name) + "_FSFDP_" + str(index) + ".dat", 'w') as fo:
                    fo.write(self.cluster_data[index].__str__())

                self.clusters[index] = {k: clustering.index[clustering["cluster"] == k].tolist()
                                        for k in list(clustering["cluster"].unique())}

                if energy:
                    import copy
                    new_clusters = copy.deepcopy(clusters[index])
                    energies = {k.name: k.Potential for k in row["Structures"]}
                    for center in clusters[index].keys():
                        changes = [center, None]
                        for crystal in clusters[index][center]:
                            if energies[crystal] < energies[center]:
                                changes[1] = crystal
                        if changes[1]:
                            new_clusters[changes[1]] = new_clusters.pop(changes[0])
                            changes_string += "{:>25} ---> {:25}\n".format(changes[0], changes[1])
                    clusters[index] = new_clusters
        
                for crystal in row["Structures"]:
                    if crystal.name not in clusters[index].keys():
                        crystal.melted = True

        self.clusters = {k: v for g in self.clusters.keys() for k, v in self.clusters[g].items()}
        self.clusters = pd.concat((
            pd.Series(data=[len(self.clusters[x]) for x in self.clusters.keys()], index=self.clusters.keys(),
                      name="Number of Structures"),
            pd.Series(data=[", ".join(str(y) for y in self.clusters[x]) for x in self.clusters.keys()],
                      index=self.clusters.keys(), name="Structures")),
            axis=1).sort_values(by="Number of Structures", ascending=False)

        with open(simulation.path_output + str(self.name) + "_Clusters.dat", 'w') as fo:
            fo.write(self.clusters.__str__())

        # TODO List distances with respect to each structure.
        #      Also, in the FSFDP algorithm, instead of halo calculation, check distance with the cluster center
