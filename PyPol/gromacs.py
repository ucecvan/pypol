class Method(object):

    def __init__(self, name):
        """
        Create a new method that uses the Gromacs MD package.
        :param name: name of the new method
        """
        self.package = "Gromacs"
        self.command = None
        self.mdrun_options = "-nt 1"
        self.atomtype = None

        self.pypol_directory = None
        self.project = None
        self.path_data = None
        self.path_output = None
        self.path_input = None
        self.name = name
        self.initial_crystals = list()

        self.molecules = list()
        self.topology = ""
        self.topology_lammps = False
        self.nmolecules = 0

        self.energy_minimisation = list()
        self.molecular_dynamics = list()
        self.metadynamics = list()

        self.cvs = list()
        self.clustering_parameters = list()

    def get_cv(self, cv_name):
        """
        Find an existing CV by its name.
        :param cv_name:
        :return:
        """
        if self.cvs:
            for existing_cv in self.cvs:
                if existing_cv.name == cv_name:
                    return existing_cv
        print("No CV found with name {}".format(cv_name))

    def get_clustering_parameters(self, clustering_parameter_name):
        """
        Find an existing clustering parameters by its name.
        :param clustering_parameter_name:
        :return:
        """
        if self.clustering_parameters:
            for existing_clustering_parameter in self.clustering_parameters:
                if existing_clustering_parameter.name == clustering_parameter_name:
                    return existing_clustering_parameter
        print("No CV found with name {}".format(clustering_parameter_name))

    def simulation(self, simulation_name):
        """
        Find an existing simulation by its name.
        :param simulation_name:
        :return:
        """
        if self.energy_minimisation:
            for existing_simulation in self.energy_minimisation:
                if existing_simulation.name == simulation_name:
                    return existing_simulation
        if self.molecular_dynamics:
            for existing_simulation in self.molecular_dynamics:
                if existing_simulation.name == simulation_name:
                    return existing_simulation
        if self.metadynamics:
            for existing_simulation in self.metadynamics:
                if existing_simulation.name == simulation_name:
                    return existing_simulation
        print("No method found with name {}".format(simulation_name))

    def import_molecule(self, path_itp, path_crd, name="", potential_energy=0.0):
        """
        Define the molecular forcefield. The coordinate file used to generate the force field is necessary to
        identify atom properties, index order and bonds.
        :param path_itp:
        :param path_crd:
        :param name:
        :param potential_energy:
        :return:
        """
        import os
        from PyPol.Defaults.defaults import equivalent_atom_types
        from PyPol.crystals import Molecule, Atom

        if not name:
            mol_number = str(self.nmolecules + 1)
            name = "M00"[:3 - len(mol_number)] + mol_number

        if not self.path_input:
            print("Error: add this method to a project before importing files")
            exit()
        elif not os.path.exists(path_itp):
            print("Error: no file found at: " + path_itp)
            exit()
        elif not os.path.exists(path_crd):
            print("Error: no file found at: " + path_crd)
            exit()
        else:
            from shutil import copyfile
            copyfile(path_itp, self.path_input + os.path.basename(path_itp))
            path_itp = self.path_input + os.path.basename(path_itp)

        molecule = Molecule(name)
        molecule.index = len(self.molecules)
        molecule.forcefield = path_itp

        working_directory = os.path.dirname(path_crd) + "/"
        file_name = os.path.basename(path_crd)
        os.chdir(working_directory)

        if not file_name.endswith(".mol2"):
            print("File format different from .mol2. Using openbabel to convert it")
            file_name = os.path.splitext(file_name)[0]
            file_format = os.path.splitext(path_crd)[-1]
            if file_format.startswith("."):
                file_format = file_format[1:]
            from openbabel import openbabel
            ob_conversion = openbabel.OBConversion()
            ob_conversion.SetInAndOutFormats(file_format, "mol2")
            mol = openbabel.OBMol()
            ob_conversion.ReadFile(mol, working_directory + file_name + "." + file_format)
            ob_conversion.WriteFile(mol, working_directory + "PyPol_Temporary_" + file_name + ".mol2")
            file_name = "PyPol_Temporary_" + file_name + ".mol2"

            path_file_ac = working_directory + file_name[:-4] + ".ac"
            os.system(self.atomtype + " -i " + file_name + " -f mol2 -p gaff -o " + path_file_ac)
            # os.remove(working_directory + file_name)

        else:
            path_file_ac = working_directory + "PyPol_Temporary_" + file_name[:-4] + ".ac"
            os.system(self.atomtype + " -i " + file_name + " -f mol2 -p gaff -o " + path_file_ac)

        file_ac = open(path_file_ac)
        for line in file_ac:
            if line.startswith(("ATOM", "HETATM")):
                atom_index = int(line[6:11]) - 1
                atom_label = line[13:17].strip()
                atom_type = line.split()[-1]
                atom_coordinates = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                if atom_type in equivalent_atom_types:
                    atom_type = equivalent_atom_types[atom_type]
                molecule.atoms.append(Atom.loadfromff(atom_index, atom_type, atom_type, atom_label, None,
                                                      atom_coordinates))
            elif line.startswith("BOND"):
                a1 = int(line.split()[2]) - 1
                a2 = int(line.split()[3]) - 1
                for atom in molecule.atoms:
                    if atom.index == a1:
                        atom.bonds.append(a2)
                    elif atom.index == a2:
                        atom.bonds.append(a1)
        file_ac.close()
        os.remove(path_file_ac)
        os.remove(working_directory + "ATOMTYPE.INF")

        file_itp = open(path_itp)
        write_atoms = False
        for line in file_itp:
            if not line.strip() or line.strip().startswith(";"):
                continue
            elif '[ atoms ]' in line:
                write_atoms = True
            elif line.strip().startswith('['):
                write_atoms = False
            elif write_atoms:
                atom_index = int(line.strip().split()[0]) - 1
                atom_charge = float(line.strip().split()[6])
                atom_mass = float(line.strip().split()[7])
                for atom in molecule.atoms:
                    if atom.index == atom_index:
                        atom.charge = atom_charge
                        atom.mass = atom_mass
        file_itp.close()
        molecule.potential_energy = potential_energy
        molecule.natoms = len(molecule.atoms)
        self.molecules.append(molecule)
        self.project.save()

    @staticmethod
    def _sort_atom_types(molecularproperties):
        """
        Identify the least recurring atom type in the molecule.
        :param molecularproperties: molecule obj
        :return:
        """
        atom_types = {}
        for atom in molecularproperties.atoms:
            if atom.type not in atom_types.keys():
                atom_types[atom.type] = {'number': 1, 'connections': len(atom.bonds), 'atoms': [atom.index]}
            else:
                atom_types[atom.type]['number'] += 1
                atom_types[atom.type]['atoms'].append(atom.index)

        starting = sorted(atom_types.items(), key=lambda t: (t[1]['number'], t[1]['connections'], t[0]))[0][1]['atoms']
        return starting, atom_types

    @staticmethod
    def _merge_atom(mol_atom, ref_atom):
        """
        Merge coordinates of the atom with the properties from the forcefield.
        :param mol_atom:
        :param ref_atom:
        :return: atom obj
        """
        from PyPol.crystals import Atom
        new_atom = Atom.loadfromcrd(ref_atom.index, ref_atom.label, ref_atom.ff_type, ref_atom.type,
                                    [mol_atom.coordinates[0], mol_atom.coordinates[1], mol_atom.coordinates[2]],
                                    mol_atom.element, ref_atom.bonds)
        new_atom.previous_index = mol_atom.index
        new_atom.charge = ref_atom.charge
        new_atom.mass = ref_atom.mass
        return new_atom

    def _recursive_index_search(self, molecule, reference, start_1=0, start_2=0, output=None):
        """
        Sort atom index by checking the connection between each pair of bonded atoms and their atom types. (Algorithm)
        :param molecule:
        :param reference:
        :param start_1:
        :param start_2:
        :param output:
        :return:
        """
        if output is None:
            from PyPol.crystals import Molecule
            output = Molecule.load(molecule.index, molecule.residue)

        connections_1 = [molecule.atoms[x1] for x1 in molecule.atoms[start_1].bonds if
                         x1 not in [mol_atom.previous_index for mol_atom in output.atoms]]
        connections_2 = [reference.atoms[x2] for x2 in reference.atoms[start_2].bonds if
                         x2 not in [ref_atom.index for ref_atom in output.atoms]]

        if len(connections_1) == 0 and len(connections_2) == 0:
            return True, output

        match = False
        for mol_atom in connections_1:
            for ref_atom in connections_2:
                if mol_atom.type == ref_atom.type and len(connections_1) >= len(connections_2):
                    new_atom = self._merge_atom(mol_atom, ref_atom)
                    output.atoms.append(new_atom)
                    match, output = self._recursive_index_search(molecule, reference, mol_atom.index, ref_atom.index,
                                                                 output)
                    if match:
                        connections_2 = [reference.atoms[x2] for x2 in reference.atoms[start_2].bonds if
                                         x2 not in [ref_atom.index for ref_atom in output.atoms]]
                        break
                    else:
                        output.atoms.remove(new_atom)

        return match, output

    def _reassign_atom_index(self, molecule, reference):
        """
        Sort atom index by checking the connection between each pair of bonded atoms and their atom types.
        :param molecule:
        :param reference:
        :return:
        """
        from PyPol.crystals import Molecule
        molecule_sorted, t1 = self._sort_atom_types(molecule)
        reference_sorted, t2 = self._sort_atom_types(reference)

        i = 0
        new_mol = None
        while i < len(reference_sorted):
            mol_atom = molecule.atoms[molecule_sorted[0]]
            ref_atom = reference.atoms[reference_sorted[i]]
            output = Molecule.load(molecule.index, reference.residue)
            new_atom = self._merge_atom(mol_atom, ref_atom)
            output.atoms.append(new_atom)
            condition, new_mol = self._recursive_index_search(molecule, reference,
                                                              mol_atom.index, ref_atom.index, output)
            if not condition:
                i += 1
            else:
                new_mol.atoms.sort(key=lambda k: k.index)
                new_mol.natoms = len(new_mol.atoms)
                break

        if new_mol.natoms != molecule.natoms:
            print("An error occurred during the index assignation.")

        return new_mol

    def _orthogonalize(self, crystal, target_lengths=(60., 60.)):
        """
        Find the most orthogonal, non-primitive cell starting from the CSP-generated cell.
        Cell vector length are limited by the target length parameters defined in the generate_input module.
        :param crystal:
        :param target_lengths:
        :return:
        """
        import numpy as np
        from PyPol.utilities import best_b, best_c, box2cell, translate_molecule

        box = crystal.box
        max_1 = int((target_lengths[0] / 10.) / box[1, 1])
        max_2 = int((target_lengths[1] / 10.) / box[2, 2])
        new_b, replica_b = best_b(box, max_1)
        new_c, replica_c = best_c(box, max_2)
        new_box = np.stack((box[:, 0], new_b, new_c), axis=1)

        if np.array_equal(np.round(new_box, 5), np.round(crystal.box, 5)):
            return crystal

        new_crystal = self._supercell_generator(crystal, replica=(1, replica_b, replica_c))

        new_crystal.box = new_box
        new_crystal.cell_parameters = box2cell(new_box)
        for molecule in new_crystal.load_molecules():
            if molecule.centroid is None:  # Error: should be already defined!
                molecule.calculate_centroid()
            translate_molecule(molecule, new_crystal.box)
            # molecule = translate_molecule(molecule, new_crystal.box)
        return new_crystal

    @staticmethod
    def generate_masscharge(crystal):
        """
        Generates the mass-charge file used by plumed.
        :param crystal:
        :return:
        """
        import os
        os.chdir(crystal.path)
        path_mc = crystal.path + "mc.dat"
        file_mc = open(path_mc, "w")
        file_mc.write("#! FIELDS index mass charge\n")
        for molecule in crystal.load_molecules():
            for atom in molecule.atoms:
                file_mc.write("{:5}{:19.3f}{:19.3f}\n".format(atom.index + molecule.natoms * molecule.index,
                                                              atom.mass, atom.charge))
        file_mc.close()

    @staticmethod
    def _supercell_generator(crystal, box=False, replica=(1, 1, 1)):
        """
        Replicate the cell in each direction.
        :param crystal:
        :param box:
        :param replica:
        :return:
        """
        import numpy as np
        import copy
        from PyPol.utilities import cell2box

        crystal.molecules = crystal.load_molecules()

        if box:
            replica_a = int(round(box[0] / (crystal.cell_parameters[0] * 10), 0))
            replica_b = int(round(box[1] / (crystal.cell_parameters[1] * 10), 0))
            replica_c = int(round(box[2] / (crystal.cell_parameters[2] * 10), 0))
        else:
            replica_a, replica_b, replica_c = replica

        molecule_index = crystal.Z
        new_molecules_list = list()
        for a in range(replica_a):
            for b in range(replica_b):
                for c in range(replica_c):
                    if a == 0 and b == 0 and c == 0:
                        continue
                    for molecule in crystal.molecules:

                        new_molecule = copy.deepcopy(molecule)
                        new_molecule.index = molecule_index
                        molecule_index += 1

                        for atom in new_molecule.atoms:
                            atom.coordinates = np.sum([a * crystal.box[:, 0], atom.coordinates], axis=0)
                            atom.coordinates = np.sum([b * crystal.box[:, 1], atom.coordinates], axis=0)
                            atom.coordinates = np.sum([c * crystal.box[:, 2], atom.coordinates], axis=0)
                        new_molecule.calculate_centroid()
                        new_molecules_list.append(new_molecule)
        crystal.molecules += new_molecules_list
        crystal.cell_parameters = np.array(
            [crystal.cell_parameters[0] * replica_a, crystal.cell_parameters[1] * replica_b,
             crystal.cell_parameters[2] * replica_c, crystal.cell_parameters[3],
             crystal.cell_parameters[4], crystal.cell_parameters[5]])
        crystal.box = cell2box(crystal.cell_parameters)
        crystal.Z = len(crystal.molecules)
        crystal.save(False)
        return crystal

    def write_output(self, path_output):
        """
        Write main features of the currect method to the project output file.
        :param path_output:
        :return:
        """
        file_output = open(path_output, "a")
        file_output.write("=" * 100 + "\n")
        file_output.write("Method Name: {}\n"
                          "MD package: {}\t({})\n"
                          "Number of Molecules: {} # Only one molecule is accepted for the moment\n\n"
                          "".format(self.name, self.package, self.command, len(self.molecules)))
        for molecule in self.molecules:
            file_output.write("Molecule: {}\n"
                              "Molecule.itp file: {}\n"
                              "Atoms:\n"
                              "  {:8} {:8} {:8} {:8}\n"
                              "".format(molecule.residue, molecule.forcefield, "Index", "Label", "Type", "Bonds"))
            for atom in molecule.atoms:
                file_output.write("  {:<8} {:<8} {:<8} {:<8}\n".format(atom.index, atom.label, atom.type,
                                                                       " ".join(str(bond) for bond in atom.bonds)))

        file_output.write("\nCollective Variables:\n")
        file_output.close()
        for cv in self.cvs:
            cv.write_output(path_output)
        file_output = open(path_output, "a")

        file_output.write("\nSimulations:\n{:<20} ".format("IDs"))
        for simulation in self.energy_minimisation:
            if simulation.completed and not simulation.hide:
                file_output.write("{:10} ".format(simulation.name))
        for simulation in self.molecular_dynamics:
            if simulation.completed and not simulation.hide:
                file_output.write("{:10} ".format(simulation.name))
        for crystal in self.initial_crystals:
            file_output.write("\n{:20} ".format(crystal.name))
            for simulation in self.energy_minimisation:
                if simulation.completed and not simulation.hide:
                    for scrystal in simulation.crystals:
                        if scrystal.name == crystal.name:
                            file_output.write("{:10.2f} "
                                              "".format(scrystal.Potential - simulation.global_minima.Potential))
                            break

            for simulation in self.molecular_dynamics:
                if simulation.completed and not simulation.hide:
                    for scrystal in simulation.crystals:
                        if scrystal.name == crystal.name and not scrystal.melted:
                            file_output.write("{:10.2f} "
                                              "".format(scrystal.Potential - simulation.global_minima.Potential))
                            break
                        elif scrystal.name == crystal.name and scrystal.melted:
                            file_output.write("{:10} ".format(str(scrystal.melted)))
                            break

        file_output.write("\n" + "=" * 100 + "\n")
        file_output.close()

    def generate_input(self, box=(4., 4., 4.), orthogonalize=False):
        """
        Generate the coordinate and the topology files to be used for energy minimization simulations.
        Error: not suitable for more than 1 molecule!
        :param box:
        :param orthogonalize:
        :return:
        """
        from PyPol.utilities import create
        # from PyPol.crystals import Crystal
        import numpy as np

        print("Generating inputs for {}".format(self.name))
        print("-" * 100)
        new_crystal_list = list()
        crystal_index = 0
        for crystal in self.initial_crystals:
            print(crystal.name)
            new_molecules = list()
            print("Index check...", end="")
            for molecule in crystal.load_molecules():
                new_molecule = self._reassign_atom_index(molecule, self.molecules[0])
                new_molecules.append(new_molecule)

            crystal.molecules = new_molecules
            crystal.nmoleculestypes = np.full((len(self.molecules)), 0)
            for molecule_i in self.molecules:
                for molecule_j in crystal.molecules:
                    if molecule_i.residue == molecule_j.residue:
                        crystal.nmoleculestypes[molecule_i.index] += 1
            crystal.Z = len(crystal.molecules)

            crystal.index = crystal_index
            crystal_index += 1

            crystal.path = self.path_data + crystal.name + "/"
            create(crystal.path, arg_type="dir", backup=True)
            crystal.save(False)

            crystal.save_pdb(crystal.path + "pc.pdb")
            crystal.save_gro(crystal.path + "pc.gro")
            print("done", end="\n")

            if orthogonalize:
                print("Othogonalize...", end="")
                crystal = self._orthogonalize(crystal, (box[1], box[2]))
                print("done", end="\n")
            print("Supercell...", end="")
            crystal = self._supercell_generator(crystal, box)
            crystal.save_pdb(crystal.path + "sc.pdb")
            crystal.save_gro(crystal.path + "sc.gro")
            print("done", end="\n")
            self.generate_masscharge(crystal)
            new_crystal_list.append(crystal)

            print("Import topology...", end="")
            from shutil import copyfile
            import os
            for molecule in self.molecules:
                copyfile(molecule.forcefield, crystal.path + os.path.basename(molecule.forcefield))
            if not self.topology:
                self.add_topology(self.pypol_directory + "Defaults/topol.top")
            copyfile(self.topology, crystal.path + os.path.basename(self.topology))
            file_top = open(crystal.path + os.path.basename(self.topology), "a")
            for molecule in self.molecules:
                file_top.write('#include "{}"\n'.format(os.path.basename(molecule.forcefield)))
            file_top.write("\n[ system ]\n"
                           "Crystal{}\n"
                           "\n[ molecules ]\n"
                           "; Compound    nmols\n".format(crystal.index))
            for molecule in self.molecules:
                file_top.write('  {:3}         {}\n'.format(molecule.residue, crystal.Z))
            file_top.close()
            print("done", end="\n")
            print("-" * 100)
            crystal.save()
        self.initial_crystals = new_crystal_list
        self.project.save()

    def add_topology(self, path_top):
        """
        Error: not suitable for more than 1 molecule!
        Add the topology file to the project. Only the [ defaults ] section should be included.
        :param path_top:
        :return:
        """
        from shutil import copyfile
        import os
        copyfile(path_top, self.path_input + os.path.basename(path_top))
        self.topology = self.path_input + os.path.basename(path_top)
        self.project.save()

    def add_simulation(self, simulation, overwrite=False):
        """
        Add a simulation object to the method used.
        :param simulation:
        :param overwrite:
        :return:
        """
        from PyPol.crystals import Crystal
        from shutil import copyfile
        import os

        if simulation.type == "Energy Minimisation" or simulation.type == "Cell Relaxation":
            if not self.energy_minimisation:
                simulation.index = 0
                simulation.previous_name = "sc"
                for crystal in self.initial_crystals:
                    simulation_crystal = Crystal.copy_properties(crystal)
                    simulation_crystal.CVs = dict()
                    simulation.crystals.append(simulation_crystal)

            else:
                for previous_simulation in self.energy_minimisation:
                    if previous_simulation.name == simulation.name and not overwrite:
                        print("Error: Simulation with name {} already present.".format(simulation.name))
                        exit()
                    elif previous_simulation.name == simulation.name and overwrite:
                        self.energy_minimisation.remove(previous_simulation)
                        print("Simulation with name {} deleted.".format(simulation.name))
                simulation.previous_name = self.energy_minimisation[-1].name
                simulation.index = len(self.energy_minimisation)

                for crystal in self.energy_minimisation[-1].crystals:
                    if crystal.completed and not crystal.melted:
                        simulation_crystal = Crystal.copy_properties(crystal)
                        simulation_crystal.CVs = dict()
                        simulation.crystals.append(simulation_crystal)

            simulation.command = self.command
            simulation.mdrun_options = self.mdrun_options
            simulation.path_data = self.path_data
            simulation.path_output = self.path_output
            simulation.path_input = self.path_input
            simulation.project = self.project
            simulation.Method = self

            if simulation.type == "Energy Minimisation":
                copyfile(simulation.mdp, self.path_input + simulation.name + ".mdp")
                simulation.mdp = self.path_input + simulation.name + ".mdp"

            elif simulation.type == "Cell Relaxation":
                if simulation.path_lmp_in:
                    copyfile(simulation.path_lmp_in, self.path_input + "input.in")
                    simulation.path_lmp_in = self.path_input + "input.in"
                    copyfile(path_lmp_ff, self.path_input + simulation.Method.molecules[0].name + ".lmp")  # Iter here
                    simulation.path_lmp_ff = self.path_input + simulation.Method.molecules[0].name + ".lmp"
                simulation.lammps = self.project.lammps
                simulation.intermol = self.project.intermol

            self.energy_minimisation.append(simulation)

        elif simulation.type == "Molecular Dynamics":

            if not self.molecular_dynamics:
                simulation.index = 0
                if not self.energy_minimisation:
                    simulation.previous_name = "sc"
                    for crystal in self.initial_crystals:
                        simulation_crystal = Crystal.copy_properties(crystal)
                        simulation_crystal.CVs = dict()
                        simulation.crystals.append(simulation_crystal)
                else:
                    simulation.previous_name = self.energy_minimisation[-1].name
                    for crystal in self.energy_minimisation[-1].crystals:
                        if crystal.completed and not crystal.melted:
                            simulation_crystal = Crystal.copy_properties(crystal)
                            simulation.crystals.append(simulation_crystal)
            else:
                for previous_simulation in self.molecular_dynamics:
                    if previous_simulation.name == simulation.name and not overwrite:
                        print("Error: Simulation with name {} already present.".format(simulation.name))
                        exit()
                    elif previous_simulation.name == simulation.name and overwrite:
                        self.molecular_dynamics.remove(previous_simulation)
                        print("Simulation with name {} deleted.".format(simulation.name))
                simulation.previous_name = self.molecular_dynamics[-1].name
                simulation.index = len(self.molecular_dynamics)

                for crystal in self.molecular_dynamics[-1].crystals:
                    if crystal.completed and not crystal.melted:
                        simulation_crystal = Crystal.copy_properties(crystal)
                        simulation_crystal.CVs = dict()
                        simulation.crystals.append(simulation_crystal)

            simulation.command = self.command
            simulation.mdrun_options = self.mdrun_options
            simulation.path_data = self.path_data
            simulation.path_output = self.path_output
            simulation.path_input = self.path_input
            simulation.project = self.project
            simulation.Method = self
            if os.path.exists(simulation.mdp):
                copyfile(simulation.mdp, self.path_input + simulation.name + ".mdp")
                simulation.mdp = self.path_input + simulation.name + ".mdp"
            elif not os.path.exists(simulation.mdp):
                from shutil import copyfile
                if simulation.name == "nvt" and not simulation.mdp:
                    print("Default file {} will be used".format(self.pypol_directory + "Defaults/Gromacs/nvt.mdp"))
                    copyfile(self.pypol_directory + "Defaults/Gromacs/nvt.mdp", self.path_input + "nvt.mdp")
                    simulation.mdp = self.path_input + "nvt.mdp"
                elif simulation.name == "berendsen" and not simulation.mdp:
                    print("Default file {} will be used"
                          "".format(self.pypol_directory + "Defaults/Gromacs/berendsen.mdp"))
                    copyfile(self.pypol_directory + "Defaults/Gromacs/berendsen.mdp", self.path_input + "berendsen.mdp")
                    simulation.mdp = self.path_input + "berendsen.mdp"
                elif simulation.name == "parrinello" and not simulation.mdp:
                    print("Default file {} will be used"
                          "".format(self.pypol_directory + "Defaults/Gromacs/parrinello.mdp"))
                    copyfile(self.pypol_directory + "Defaults/Gromacs/parrinello.mdp", self.path_input +
                             "parrinello.mdp")
                    simulation.mdp = self.path_input + "parrinello.mdp"
                elif simulation.index == 0 and not simulation.mdp:
                    print("No mdp file has been specified.\n"
                          "File {} will be used".format(self.pypol_directory + "Defaults/Gromacs/nvt.mdp"))
                    simulation.name = "nvt"
                    copyfile(self.pypol_directory + "Defaults/Gromacs/nvt.mdp", self.path_input + "nvt.mdp")
                    simulation.mdp = self.path_input + "nvt.mdp"
                elif simulation.index == 1 and not simulation.mdp and simulation.previous_name == "nvt":
                    print("No mdp file has been specified.\n"
                          "File {} will be used".format(self.pypol_directory + "Defaults/Gromacs/berendsen.mdp"))
                    simulation.name = "berendsen"
                    copyfile(self.pypol_directory + "Defaults/Gromacs/berendsen.mdp", self.path_input + "berendsen.mdp")
                    simulation.mdp = self.path_input + "berendsen.mdp"
                elif simulation.index == 2 and not simulation.mdp and simulation.previous_name == "berendsen":
                    print("No mdp file has been specified.\n"
                          "File {} will be used".format(self.pypol_directory + "Defaults/Gromacs/parrinello.mdp"))
                    simulation.name = "parrinello"
                    copyfile(self.pypol_directory + "Defaults/Gromacs/parrinello.mdp", self.path_input +
                             "parrinello.mdp")
                    simulation.mdp = self.path_input + "parrinello.mdp"
                else:
                    print("Error: No mdp file has been found.\n"
                          "You can use the defaults mdp parameters by using the names "
                          "'nvt', 'berendsen' or 'parrinello'\n"
                          "You can check the relative mdp files in folder: {}"
                          "".format(self.pypol_directory + "Defaults/Gromacs/"))
                    exit()
            self.molecular_dynamics.append(simulation)
        self.project.save()


class EnergyMinimization(object):

    def __init__(self, name, path_mdp):
        """
        Perform an energy minimisation simulation.
        :param name:
        :param path_mdp:
        """
        import os
        from shutil import copyfile

        self.type = "Energy Minimisation"
        self.project = None
        self.method = None
        self.path_data = None
        self.path_output = None
        self.path_input = None
        self.index = 0
        self.previous_name = None
        self.name = name
        if not os.path.exists(path_mdp):
            print("Error: File '{}' not found".format(path_mdp))
            exit()
        self.mdp = path_mdp
        self.command = None
        self.mdrun_options = ""
        self.crystals = list()
        self.completed = False
        self.global_minima = None
        self.hide = False

    def generate_input(self, bash_script=False, crystals="all"):
        """
        Copy the Gromacs .mdp file to each crystal path
        :param bash_script:
        :param crystals:
        :return:
        """
        from PyPol.utilities import get_list
        from shutil import copyfile
        if crystals == "all":
            list_crystals = self.crystals
        elif crystals == "incomplete":
            list_crystals = list()
            for crystal in self.crystals:
                if not crystal.completed:
                    list_crystals.append(crystal)
        else:
            list_crystals = get_list(crystals)

        for crystal in list_crystals:
            copyfile(self.mdp, crystal.path + self.name + ".mdp")

        if bash_script:
            file_script = open(self.path_data + "/run_" + self.name + ".sh", "w")
            file_script.write('#!/bin/bash\n\n'
                              'crystal_paths="\n')
            for crystal in self.crystals:
                file_script.write(crystal.path + "\n")
            file_script.write('"\n\n'
                              'for crystal in $crystal_paths ; do\n'
                              'cd "$crystal" || exit \n'
                              '{0} grompp -f {1}.mdp -c {2}.gro -o {1}.tpr -p topol.top -maxwarn 1 \n'
                              '{0} mdrun {3} -deffnm {1} \n'
                              'done \n'
                              ''.format(self.command, self.name, self.previous_name, self.mdrun_options))
            file_script.close()

        self.project.save()

    def check_normal_termination(self, crystals="all"):
        """
        Verify if the simulation ended correctly and upload new crystal properties.
        :param crystals:
        :return:
        """
        import os
        from PyPol.utilities import get_list, box2cell
        import numpy as np

        if crystals == "all":
            list_crystals = self.crystals
        elif crystals == "incomplete":
            list_crystals = list()
            for crystal in self.crystals:
                if not crystal.completed:
                    list_crystals.append(crystal)
        else:
            list_crystals = get_list(crystals)

        for crystal in list_crystals:
            path_output = crystal.path + self.name + ".log"
            if os.path.exists(path_output):
                file_output = open(path_output)
                lines = file_output.readlines()
                if "Finished mdrun" in lines[-2] or "Finished mdrun" in lines[-1]:
                    for i in range(-2, -15, -1):
                        line = lines[i]
                        if line.lstrip().startswith("Potential Energy  ="):
                            # Modify for more than one molecule
                            lattice_energy = float(line.split()[-1]) / crystal.Z - \
                                             self.method.molecules[0].potential_energy
                            crystal.Potential = lattice_energy
                            crystal.completed = True
                            break
                else:
                    print("An error has occurred with Gromacs. Check simulation {} in folder {}."
                          "".format(self.name, crystal.path))
                file_output.close()
            else:
                print("An error has occurred with Gromacs. Check simulation {} in folder {}."
                      "".format(self.name, crystal.path))

        new_rank = dict()
        incomplete_simulations = False
        for crystal in self.crystals:
            if crystal.completed:
                new_rank[crystal.name] = crystal.Potential
                file_gro = open(crystal.path + self.name + ".gro", "r")
                new_box = file_gro.readlines()[-1].split()
                file_gro.close()
                if len(new_box) == 3:
                    new_box = [float(ii) for ii in new_box] + [0., 0., 0., 0., 0., 0.]
                idx_gromacs = [0, 5, 7, 3, 1, 8, 4, 6, 2]
                crystal.box = np.array([float(new_box[ii]) for ii in idx_gromacs]).reshape((3, 3))
                crystal.cell_parameters = box2cell(crystal.box)
                crystal.volume = np.linalg.det(crystal.box)
            else:
                incomplete_simulations = True
                break

        if not incomplete_simulations:
            rank = 1
            for crystal_name in sorted(new_rank, key=lambda c: new_rank[c]):
                for crystal in self.crystals:
                    if crystal.name == crystal_name:
                        crystal.rank = rank
                        if rank == 1:
                            self.global_minima = crystal
                        rank += 1
            self.completed = True

        self.project.save()


class CellRelaxation(object):
    """
    Error: not suitable for more than 1 molecule + not possible to define user input and forcefield.
    Divide bonded from non-bonded parameters and add read_data at the end with the LJ coeff.
    """

    def __init__(self, name, path_lmp_in=None, path_lmp_ff=None):  # Transform path_lmp_ff in iterable obj for all mol
        """
        Convert Gromacs input files to the LAMMPS ones and perform a cell relaxation.
        :param name:
        :param path_lmp_in:
        :param path_lmp_ff:
        """
        import os
        from shutil import copyfile
        self.type = "Cell Relaxation"
        self.project = None
        self.method = None
        self.path_data = None
        self.path_output = None
        self.path_input = None
        self.index = 0
        self.previous_name = None
        self.name = name
        if path_lmp_in:
            if not os.path.exists(path_lmp_in) and not os.path.exists(path_lmp_ff):
                print("Error: File '{}' or '{}' not found".format(path_lmp_in, path_lmp_ff))
                exit()
        self.path_lmp_in = path_lmp_in
        self.path_lmp_ff = path_lmp_ff
        self.command = None
        self.intermol = None
        self.lammps = None
        self.crystals = list()
        self.completed = False
        self.global_minima = None
        self.hide = True

    def convert_topology(self, path_gmx, molecule):
        """
        Convert the gromacs topology file to the LAMMPS ones with InterMol.
        :param path_gmx:
        :param molecule:
        :return:
        """
        import os
        import numpy as np
        from shutil import copyfile
        os.chdir(path_gmx)
        os.system("{0} --gro_in {1}.gro {1}.top --lammps".format(self.intermol, molecule.residue))
        path_lmp = path_gmx + molecule.residue + "_converted.lmp"

        # Check combination rule
        comb_rule = 2
        file_top = open(path_gmx + molecule.residue + ".top")
        read_com_rule = False
        for line in file_top:
            if not line.strip() or line.strip().startswith(";"):
                continue
            elif '[ defaults ]' in line:
                read_com_rule = True
            elif read_com_rule:
                comb_rule = int(line.split()[1])
                break
        file_top.close()

        # Read sigma/C6 and epsilon/C12 from itp file
        name2index = {}
        atomtype_dict = {}
        for path_itp in os.listdir(path_gmx):
            if path_itp.endswith(".itp"):
                file_itp = open(path_itp)
                read_atomtype = False
                for line in file_itp:
                    if not line.strip() or line.strip().startswith(";"):
                        continue
                    elif '[ atomtypes ]' in line:
                        read_atomtype = True
                    elif line.strip().startswith('['):
                        read_atomtype = False
                    elif read_atomtype:
                        at_1 = line.split()[0]
                        name2index[at_1] = len(name2index)
                        sigma = float(line.split()[5])
                        epsilon = float(line.split()[6])
                        if comb_rule == 1:
                            # Convert C6 and C12 to sigma and epsilon
                            new_sigma = np.power(epsilon / sigma, 1. / 6.) * 10.
                            new_epsilon = np.power(sigma, 2) / (4. * epsilon) * 0.239006
                            atomtype_dict[name2index[at_1]] = [new_epsilon, new_sigma]
                        else:
                            new_sigma = sigma * 10.
                            new_epsilon = epsilon * 0.239006
                            atomtype_dict[name2index[at_1]] = [new_epsilon, new_sigma]
                file_itp.close()

        # Check if nonbond_params are present
        nonbond_param = False
        for path_itp in os.listdir(path_gmx):
            if path_itp.endswith(".itp"):
                file_itp = open(path_itp)
                read_atomtype = False
                for line in file_itp:
                    if not line.strip() or line.strip().startswith(";"):
                        continue
                    elif '[ nonbond_params ]' in line:
                        read_atomtype = True
                        nonbond_param = {}
                    elif line.strip().startswith('['):
                        read_atomtype = False
                    elif read_atomtype:
                        at_1 = name2index[line.split()[0]]
                        at_2 = name2index[line.split()[1]]
                        sigma = float(line.split()[3])
                        epsilon = float(line.split()[4])
                        if comb_rule == 1:
                            # Convert C6 and C12 to sigma and epsilon
                            new_sigma = np.power(epsilon / sigma, 1. / 6.) * 10.
                            new_epsilon = np.power(sigma, 2) / (4. * epsilon) * 0.239006
                            nonbond_param[(at_1, at_2)] = [new_epsilon, new_sigma]
                        else:
                            new_sigma = sigma * 10.
                            new_epsilon = epsilon * 0.239006
                            nonbond_param[(at_1, at_2)] = [new_epsilon, new_sigma]
                file_itp.close()

        # Add pairs info.
        pair_ij_list = list()
        for i in atomtype_dict.keys():
            for j in [lj for lj in atomtype_dict.keys() if lj >= i]:
                if nonbond_param:
                    if (i, j) in nonbond_param.keys():
                        pair_ij_list.append([i, j] + nonbond_param[(i, j)])
                else:
                    epsilon_ij = np.sqrt(atomtype_dict[i][0] * atomtype_dict[j][0])
                    sigma_ij = None
                    if comb_rule == 1 or comb_rule == 3:
                        sigma_ij = np.sqrt(atomtype_dict[i][1] * atomtype_dict[j][1])
                    elif comb_rule == 2:
                        sigma_ij = (atomtype_dict[i][1] + atomtype_dict[j][1]) / 2.
                    else:
                        print("Unknown combination rule: {}".format(comb_rule))
                        exit()
                    pair_ij_list.append([i, j, epsilon_ij, sigma_ij])
        copyfile(path_lmp, path_gmx + molecule.residue + ".lmp")
        path_lmp = path_gmx + molecule.residue + ".lmp"
        file_lmp = open(path_lmp, "a")
        file_lmp.write("\nPairIJ Coeffs\n\n")
        for pair in pair_ij_list:
            file_lmp.write("{} {}    {:.12f} {:.12f}\n".format(pair[0] + 1, pair[1] + 1, pair[2], pair[3]))
        file_lmp.close()

        # return forcefield path to be used in concomitant with input file
        return path_lmp

    def check_lmp_input(self):
        """
        Check if conversion is done correctly.
        Error: Add energy check.
        :return:
        """
        import os
        file_lmp_in = open(self.path_lmp_in)
        file_lmp_in_new = open(self.method.path_input + "lmp_input.in", "w")
        write_read_data = True
        for line in file_lmp_in:
            if line.startswith("read_data"):
                if write_read_data:
                    for molecule in self.method.molecules:
                        file_lmp_in_new.write("read_data {}\n".format(os.path.basename(molecule.lmp_forcefield)))
                write_read_data = False
                continue
            file_lmp_in_new.write(line)
        file_lmp_in_new.close()
        file_lmp_in.close()
        if write_read_data:
            print("Error: please include the read_data keyword in the input file to let PyPol know where to substitute "
                  "the lines. \nCheck https://lammps.sandia.gov/doc/read_data.html to identify the correct position")
            exit()
        return self.method.path_input + "lmp_input.in"

    def generate_lammps_topology(self, crystal):
        """
        Check InterMol output and modify it according to crystal properties.
        :param crystal:
        :return:
        """
        import os
        import numpy as np
        from shutil import copyfile

        working_directory = crystal.path + "lammps/"
        os.mkdir(working_directory)
        os.chdir(working_directory)
        copyfile(self.path_lmp_in, working_directory + "input.in")
        path_gro = crystal.path + self.previous_name + ".gro"
        for moleculetype in self.method.molecules:
            molecules_in_crystal = int(crystal.Z / np.sum(crystal.nmoleculestypes) *
                                       crystal.nmoleculestypes[moleculetype.index])
            atoms_in_crystal = moleculetype.natoms * molecules_in_crystal

            # Import coordinates
            coordinates = np.full((atoms_in_crystal, 3), np.nan)
            i = 0
            file_gro = open(path_gro)
            next(file_gro)
            next(file_gro)
            for line in file_gro:
                if line[5:11].strip() == moleculetype.residue:
                    coordinates[i, :] = np.array([float(line[20:28]), float(line[28:36]), float(line[36:44])])
                    i += 1
            file_gro.close()
            coordinates = coordinates * 10

            # Save new lmp file
            file_lmp_ff = open(moleculetype.lmp_forcefield)
            file_lmp_ff_new = open(working_directory + os.path.basename(moleculetype.lmp_forcefield), "w")
            # write_at, write_vel, write_bon, write_ang, write_dih, write_imp = False, False, False, False, False, False
            atoms, velocities, bonds, angles, dihs, imps = [], [], [], [], [], []
            number_of_atoms, number_of_bonds, number_of_angles, number_of_dihedrals, number_of_impropers = 0, 0, 0, 0, 0
            for line in file_lmp_ff:
                # Change header of LAMMPS
                if line.rstrip().endswith("atoms"):
                    number_of_atoms = int(line.split()[0]) * molecules_in_crystal
                    file_lmp_ff_new.write("{} atoms\n".format(number_of_atoms))
                elif line.rstrip().endswith("bonds"):
                    number_of_bonds = int(line.split()[0]) * molecules_in_crystal
                    file_lmp_ff_new.write("{} bonds\n".format(number_of_bonds))
                elif line.rstrip().endswith("angles"):
                    number_of_angles = int(line.split()[0]) * molecules_in_crystal
                    file_lmp_ff_new.write("{} angles\n".format(number_of_angles))
                elif line.rstrip().endswith("dihedrals"):
                    number_of_dihedrals = int(line.split()[0]) * molecules_in_crystal
                    file_lmp_ff_new.write("{} dihedrals\n".format(number_of_dihedrals))
                elif line.rstrip().endswith("impropers"):
                    number_of_impropers = int(line.split()[0]) * molecules_in_crystal
                    file_lmp_ff_new.write("{} impropers\n".format(number_of_impropers))
                elif line.rstrip().endswith("xhi"):
                    file_lmp_ff_new.write("{:12.8f} {:12.8f} xlo xhi\n".format(0., crystal.box[0, 0] * 10))
                elif line.rstrip().endswith("yhi"):
                    file_lmp_ff_new.write("{:12.8f} {:12.8f} ylo yhi\n".format(0., crystal.box[1, 1] * 10))
                elif line.rstrip().endswith("zhi"):
                    file_lmp_ff_new.write("{:12.8f} {:12.8f} zlo zhi\n"
                                          "{:12.8f} {:12.8f} {:12.8f} xy xz yz\n"
                                          "".format(0., crystal.box[2, 2] * 10, crystal.box[0, 1] * 10,
                                                    crystal.box[0, 2] * 10, crystal.box[1, 2] * 10))
                elif line.rstrip().endswith("xy xz yz"):
                    continue

                # Change body of LAMMPS
                elif "Atoms" in line:
                    file_lmp_ff_new.write("Atoms\n\n")
                    # line = next(file_lmp_ff)
                    next(file_lmp_ff)
                    for atom in range(moleculetype.natoms):
                        line = next(file_lmp_ff)
                        atoms.append(line.split()[2:4])
                    for atom in range(number_of_atoms):
                        atomtype_idx = atom - int(atom / moleculetype.natoms) * moleculetype.natoms
                        file_lmp_ff_new.write("{:>6} {:>6} {:>6} {:>12.8f} {:>12.7f} {:>12.7f} {:>12.7f}\n"
                                              "".format(atom + 1, int(atom / moleculetype.natoms) + 1,
                                                        int(atoms[atomtype_idx][0]), float(atoms[atomtype_idx][1]),
                                                        coordinates[atom][0], coordinates[atom][1],
                                                        coordinates[atom][2]))
                    file_lmp_ff_new.write("\n")

                elif "Velocities" in line:
                    file_lmp_ff_new.write("Velocities\n\n")
                    # line = next(file_lmp_ff)
                    next(file_lmp_ff)
                    for vel in range(moleculetype.natoms):
                        line = next(file_lmp_ff)
                        velocities.append(line.split()[1:4])
                    for vel in range(number_of_atoms):
                        vel_idx = vel - int(vel / moleculetype.natoms) * moleculetype.natoms
                        file_lmp_ff_new.write("{:>6} {:>12.7f} {:>12.7f} {:>12.7f}\n"
                                              "".format(vel + 1, float(velocities[vel_idx][0]),
                                                        float(velocities[vel_idx][1]), float(velocities[vel_idx][2])))
                    file_lmp_ff_new.write("\n")

                elif "Bonds" in line:
                    file_lmp_ff_new.write("Bonds\n\n")
                    # line = next(file_lmp_ff)
                    next(file_lmp_ff)
                    bonds_in_molecule = int(number_of_bonds / molecules_in_crystal)
                    for bond in range(bonds_in_molecule):
                        line = next(file_lmp_ff)
                        bonds.append(line.split())
                    for bond in range(number_of_bonds):
                        molecule_idx = int(bond / bonds_in_molecule)
                        bondtype_idx = bond - molecule_idx * bonds_in_molecule
                        file_lmp_ff_new.write("{:>6}     {} {} {}\n"
                                              "".format(bond + 1, bonds[bondtype_idx][1],
                                                        int(bonds[bondtype_idx][
                                                                2]) + moleculetype.natoms * molecule_idx,
                                                        int(bonds[bondtype_idx][
                                                                3]) + moleculetype.natoms * molecule_idx))
                    file_lmp_ff_new.write("\n")

                elif "Angles" in line:
                    file_lmp_ff_new.write("Angles\n\n")
                    # line = next(file_lmp_ff)
                    next(file_lmp_ff)
                    ang_in_molecule = int(number_of_angles / molecules_in_crystal)
                    for angle in range(ang_in_molecule):
                        line = next(file_lmp_ff)
                        angles.append(line.split())
                    for angle in range(number_of_angles):
                        molecule_idx = int(angle / ang_in_molecule)
                        angletype_idx = angle - molecule_idx * ang_in_molecule
                        file_lmp_ff_new.write("{:>6}     {} {} {} {}\n"
                                              "".format(angle + 1, angles[angletype_idx][1],
                                                        int(angles[angletype_idx][
                                                                2]) + moleculetype.natoms * molecule_idx,
                                                        int(angles[angletype_idx][
                                                                3]) + moleculetype.natoms * molecule_idx,
                                                        int(angles[angletype_idx][
                                                                4]) + moleculetype.natoms * molecule_idx))
                    file_lmp_ff_new.write("\n")

                elif "Dihedrals" in line:
                    file_lmp_ff_new.write("Dihedrals\n\n")
                    # line = next(file_lmp_ff)
                    next(file_lmp_ff)
                    dihs_in_molecule = int(number_of_dihedrals / molecules_in_crystal)
                    for dih in range(dihs_in_molecule):
                        line = next(file_lmp_ff)
                        dihs.append(line.split())
                    for dih in range(number_of_dihedrals):
                        molecule_idx = int(dih / dihs_in_molecule)
                        dihtype_idx = dih - molecule_idx * dihs_in_molecule
                        file_lmp_ff_new.write("{:>6}     {} {} {} {} {}\n"
                                              "".format(dih + 1, dihs[dihtype_idx][1],
                                                        int(dihs[dihtype_idx][2]) + moleculetype.natoms * molecule_idx,
                                                        int(dihs[dihtype_idx][3]) + moleculetype.natoms * molecule_idx,
                                                        int(dihs[dihtype_idx][4]) + moleculetype.natoms * molecule_idx,
                                                        int(dihs[dihtype_idx][5]) + moleculetype.natoms * molecule_idx))
                    file_lmp_ff_new.write("\n")

                elif "Impropers" in line:
                    file_lmp_ff_new.write("Impropers\n\n")
                    # line = next(file_lmp_ff)
                    next(file_lmp_ff)
                    imps_in_molecule = int(number_of_impropers / molecules_in_crystal)
                    for imp in range(imps_in_molecule):
                        line = next(file_lmp_ff)
                        imps.append(line.split())
                    for imp in range(number_of_impropers):
                        molecule_idx = int(imp / imps_in_molecule)
                        imptype_idx = imp - molecule_idx * imps_in_molecule
                        file_lmp_ff_new.write("{:>6}     {} {} {} {} {}\n"
                                              "".format(imp + 1, imps[imptype_idx][1],
                                                        int(imps[imptype_idx][2]) + moleculetype.natoms * molecule_idx,
                                                        int(imps[imptype_idx][3]) + moleculetype.natoms * molecule_idx,
                                                        int(imps[imptype_idx][4]) + moleculetype.natoms * molecule_idx,
                                                        int(imps[imptype_idx][5]) + moleculetype.natoms * molecule_idx))
                    file_lmp_ff_new.write("\n")

                else:
                    file_lmp_ff_new.write(line)
            file_lmp_ff_new.close()
            file_lmp_ff.close()

    def generate_input(self, bash_script=False, crystals="all"):
        """
        Generate LAMMPS inputs.
        :param bash_script:
        :param crystals:
        :return:
        """
        from PyPol.utilities import get_list
        from shutil import copyfile
        import os

        if self.path_lmp_ff is None:
            from PyPol.utilities import create
            import os
            for molecule in self.method.molecules:
                path_gmx = self.path_input + "GMX2LMP_" + molecule.residue + "/"
                create(path_gmx, arg_type="dir")
                molecule.save_gro(path_gmx + molecule.residue + ".gro")
                copyfile(molecule.forcefield, path_gmx + os.path.basename(molecule.forcefield))

                copyfile(self.method.topology, path_gmx + molecule.residue + ".top")
                file_top = open(path_gmx + molecule.residue + ".top", "a")
                # for molecule in self.method.molecules:
                file_top.write('#include "{}"\n'.format(os.path.basename(molecule.forcefield)))
                file_top.write("\n[ system ]\n"
                               "Isolated molecule\n"
                               "\n[ molecules ]\n"
                               "; Compound    nmols\n")
                # for molecule in self.method.molecules:
                file_top.write('  {:3}         {}\n'.format(molecule.residue, "1"))
                file_top.close()
                molecule.lmp_forcefield = self.convert_topology(path_gmx, molecule)
        else:
            if isinstance(self.path_lmp_ff, str):
                if len(self.method.molecules) == 1:
                    if os.path.exists(self.path_lmp_ff):
                        self.method.molecules[0].lmp_forcefield = self.path_lmp_ff
                    else:
                        print("Error: file '{}' does not exist".format(self.path_lmp_ff))
                        exit()
                else:
                    print("Error: Incorrect number of lammps datafile: should be {}, found 1"
                          "\nPlease write a molecular forcefield for each molecule in Method.molecules"
                          "".format(len(self.method.molecules)))
                    exit()
            elif hasattr(self.path_lmp_ff, "__iter__"):
                if len(self.path_lmp_ff) == len(self.method.molecules):
                    for idx in range(len(self.method.molecules)):
                        if os.path.exists(self.path_lmp_ff[idx]):
                            self.method.molecules[idx].lmp_forcefield = self.path_lmp_ff[idx]
                        else:
                            print("Error: file '{}' does not exist".format(self.path_lmp_ff))
                            exit()
                else:
                    print("Error: Incorrect number of lammps datafile: should be {}, found {}}"
                          "\nPlease write a molecular forcefield for each molecule in Method.molecules"
                          "".format(len(self.method.molecules), len(self.path_lmp_ff)))
                    exit()
            else:
                print("Error: No molecular forcefield found in {}".format(self.path_lmp_ff))
                exit()

        if self.path_lmp_in is None:
            self.path_lmp_in = self.project.pypol_directory + "/Defaults/lmp.in"
        else:
            print("Warning: The input file must contain the keyword 'read_data' that will be later modified to "
                  "include the generated input files")

        self.path_lmp_in = self.check_lmp_input()

        if crystals == "all":
            list_crystals = self.crystals
        elif crystals == "incomplete":
            list_crystals = list()
            for crystal in self.crystals:
                if crystal.completed:
                    list_crystals.append(crystal)
        else:
            list_crystals = get_list(crystals)

        for crystal in list_crystals:
            print(crystal.name)
            self.generate_lammps_topology(crystal)

        if bash_script:
            file_script = open(self.path_data + "/run_" + self.name + ".sh", "w")
            file_script.write('#!/bin/bash\n\n'
                              'crystal_paths="\n')
            for crystal in self.crystals:
                file_script.write(crystal.path + "lammps/\n")
            file_script.write('"\n\n'
                              'for crystal in $crystal_paths ; do\n'
                              'cd "$crystal" || exit \n'
                              '{0} < {1} \n'
                              'done \n'
                              ''.format(self.lammps, "input.in"))
            file_script.close()
        self.project.save()

    def check_normal_termination(self, crystals="all"):
        """
        Verify if the simulation ended correctly and upload new crystal properties.
        Convert files back to the Gromacs file format.
        :param crystals:
        :return:
        """
        import os
        from PyPol.utilities import get_list, box2cell
        import numpy as np

        if crystals == "all":
            list_crystals = self.crystals
        elif crystals == "incomplete":
            list_crystals = list()
            for crystal in self.crystals:
                if not crystal.completed:
                    list_crystals.append(crystal)
        else:
            list_crystals = get_list(crystals)
        print(list_crystals)
        for crystal in list_crystals:
            os.chdir(crystal.path + "lammps/")
            print(crystal.name)
            path_coord = crystal.path + "lammps/coordinates.xtc"
            path_output = crystal.path + "lammps/log.lammps"
            if os.path.exists(path_output) and os.path.exists(path_coord):
                file_output = open(path_output)
                for line in file_output:
                    if "Energy initial, next-to-last, final =" in line:
                        line = next(file_output)
                        ref_pot = self.method.molecules[0].potential_energy
                        crystal.Potential = float(line.split()[-1]) * 4.184 / crystal.Z - ref_pot
                        crystal.completed = True
                        break

                os.system("{} trjconv -f coordinates.xtc -s ../{}.tpr -pbc mol -o {}.gro <<< 2"
                          "".format(self.command, self.previous_name, self.name))

                os.system("tail -{0} {1}.gro > ../{1}.gro"
                          "".format(int(crystal.Z * self.method.molecules[0].natoms + 3), self.name))
            else:
                print("An error has occurred with LAMMPS. Check simulation {} in folder {}."
                      "".format(self.name, crystal.path + "lammps/"))

        new_rank = dict()
        incomplete_simulations = False
        for crystal in self.crystals:
            if crystal.completed:
                new_rank[crystal.name] = crystal.Potential
                file_gro = open(crystal.path + self.name + ".gro", "r")
                new_box = file_gro.readlines()[-1].split()
                file_gro.close()
                if len(new_box) == 3:
                    new_box = [float(ii) for ii in new_box] + [0., 0., 0., 0., 0., 0.]
                idx_gromacs = [0, 5, 7, 3, 1, 8, 4, 6, 2]
                crystal.box = np.array([float(new_box[ii]) for ii in idx_gromacs]).reshape((3, 3))
                crystal.cell_parameters = box2cell(crystal.box)
                crystal.volume = np.linalg.det(crystal.box)
            else:
                incomplete_simulations = True
                break

        if not incomplete_simulations:
            rank = 1
            for crystal_name in sorted(new_rank, key=lambda c: new_rank[c]):
                for crystal in self.crystals:
                    if crystal.name == crystal_name:
                        crystal.rank = rank
                        if rank == 1:
                            self.global_minima = crystal
                        rank += 1
            self.completed = True
        self.project.save()


class MolecularDynamics(object):

    def __init__(self, name=None, path_mdp=""):
        """
        Generates input for MD simulations with Gromacs.
        :param name:
        :param path_mdp:
        """
        import os
        from shutil import copyfile
        self.type = "Molecular Dynamics"
        self.project = None
        self.method = None
        self.path_data = None
        self.path_output = None
        self.path_input = None
        self.index = 0
        self.previous_name = None
        self.name = name
        if path_mdp:
            if not os.path.exists(path_mdp):
                #     copyfile(path_mdp, self.path_input + name + ".mdp")
                #     self.mdp = self.path_input + name + ".mdp"
                # else:
                print("Error: File '{}' not found".format(path_mdp))
                exit()
        else:
            self.mdp = path_mdp
        self.command = None
        self.mdrun_options = ""
        self.crystals = list()
        self.completed = False
        self.global_minima = None
        self.hide = False

    def generate_input(self, bash_script=False, crystals="incomplete"):
        """

        :param bash_script:
        :param crystals:
        :return:
        """
        from PyPol.utilities import get_list
        from shutil import copyfile
        if crystals == "all":
            list_crystals = self.crystals
        elif crystals == "incomplete":
            list_crystals = list()
            for crystal in self.crystals:
                if not crystal.completed:
                    list_crystals.append(crystal)
        else:
            list_crystals = get_list(crystals)

        for crystal in list_crystals:
            copyfile(self.mdp, crystal.path + self.name + ".mdp")

        if bash_script:
            file_script = open(self.path_data + "/run_" + self.name + ".sh", "w")
            file_script.write('#!/bin/bash\n\n'
                              'crystal_paths="\n')
            for crystal in self.crystals:
                file_script.write(crystal.path + "\n")
            if self.index > 0:
                file_script.write('"\n\n'
                                  'for crystal in $crystal_paths ; do\n'
                                  'cd "$crystal" || exit \n'
                                  '{0} grompp -f {1}.mdp -c {2}.gro -t {2}.cpt -o {1}.tpr -p topol.top -maxwarn 1 \n'
                                  '{0} mdrun {3} -deffnm {1} \n'
                                  'done \n'
                                  ''.format(self.command, self.name, self.previous_name, self.mdrun_options))
            else:
                file_script.write('"\n\n'
                                  'for crystal in $crystal_paths ; do\n'
                                  'cd "$crystal" || exit \n'
                                  '{0} grompp -f {1}.mdp -c {2}.gro -o {1}.tpr -p topol.top -maxwarn 1 \n'
                                  '{0} mdrun {3} -deffnm {1} \n'
                                  'done \n'
                                  ''.format(self.command, self.name, self.previous_name, self.mdrun_options))
            file_script.close()
        self.project.save()

    def check_normal_termination(self, crystals="all"):
        """
        Verify if the simulation ended correctly and upload new crystal properties.
        :param crystals:
        :return:
        """
        import os
        from PyPol.utilities import get_list, box2cell
        import numpy as np

        if crystals == "all":
            list_crystals = self.crystals
        elif crystals == "incomplete":
            list_crystals = list()
            for crystal in self.crystals:
                if not crystal.completed:
                    list_crystals.append(crystal)
        else:
            list_crystals = get_list(crystals)

        for crystal in list_crystals:
            path_output = crystal.path + self.name + ".log"
            if os.path.exists(path_output):
                file_output = open(path_output)
                lines = file_output.readlines()
                if any("Finished mdrun" in string for string in lines[-30:]):
                    os.chdir(crystal.path)
                    os.system('{} energy -f {}.edr <<< "Potential" > PyPol_Temporary_Potential.txt'
                              ''.format(self.command, self.name))
                    file_pot = open(crystal.path + 'PyPol_Temporary_Potential.txt')
                    for line in file_pot:
                        if line.startswith("Potential"):
                            lattice_energy = float(line.split()[1]) / crystal.Z - \
                                             self.method.molecules[0].potential_energy
                            crystal.Potential = lattice_energy
                            crystal.completed = True
                            break
                    file_pot.close()
                    os.remove(crystal.path + 'PyPol_Temporary_Potential.txt')
                else:
                    print("An error has occurred with Gromacs. Check simulation {} in folder {}."
                          "".format(self.name, crystal.path))
                file_output.close()
            else:
                print("An error has occurred with Gromacs. Check simulation {} in folder {}."
                      "".format(self.name, crystal.path))

        new_rank = dict()
        incomplete_simulations = False
        for crystal in self.crystals:
            if crystal.completed:
                new_rank[crystal.name] = crystal.Potential
                file_gro = open(crystal.path + self.name + ".gro", "r")
                new_box = file_gro.readlines()[-1].split()
                file_gro.close()
                if len(new_box) == 3:
                    new_box = [float(ii) for ii in new_box] + [0., 0., 0., 0., 0., 0.]
                idx_gromacs = [0, 5, 7, 3, 1, 8, 4, 6, 2]
                crystal.box = np.array([float(new_box[ii]) for ii in idx_gromacs]).reshape((3, 3))
                crystal.cell_parameters = box2cell(crystal.box)
                crystal.volume = np.linalg.det(crystal.box)
            else:
                incomplete_simulations = True
                break

        if not incomplete_simulations:
            rank = 1
            for crystal_name in sorted(new_rank, key=lambda c: new_rank[c]):
                for crystal in self.crystals:
                    if crystal.name == crystal_name:
                        crystal.rank = rank
                        if rank == 1:
                            self.global_minima = crystal
                        rank += 1
            self.completed = True

        self.project.save()


class Metadynamics(object):

    def __init__(self, name=None, path_mdp=""):
        """
        Generates input for MD simulations with Gromacs.
        :param name:
        :param path_mdp:
        """
        import os
        from shutil import copyfile
        self.type = "Metadynamics"
        self.project = None
        self.method = None
        self.path_data = None
        self.path_output = None
        self.path_input = None
        self.index = 0
        self.previous_name = None
        self.name = name
        if path_mdp:
            if os.path.exists(path_mdp):
                copyfile(path_mdp, self.path_input + name + ".mdp")
                self.mdp = self.path_input + name + ".mdp"
            else:
                print("Error: File '{}' not found".format(path_mdp))
                exit()
        else:
            self.mdp = path_mdp
        self.command = None
        self.mdrun_options = "-plumed plumed_{}.dat".format(self.name)
        self.crystals = list()
        self.completed = False
        self.global_minima = None
        self.hide = False

        self.meta_cvp = list()
        self.meta_replicas = 1

    def generate_input(self, bash_script=False, crystals="incomplete"):
        """

        :param bash_script:
        :param crystals:
        :return:
        """
        from PyPol.utilities import get_list
        from shutil import copyfile
        if crystals == "all":
            list_crystals = self.crystals
        elif crystals == "incomplete":
            list_crystals = list()
            for crystal in self.crystals:
                if not crystal.completed:
                    list_crystals.append(crystal)
        else:
            list_crystals = get_list(crystals)

        for crystal in list_crystals:
            copyfile(self.mdp, crystal.path + self.name + ".mdp")

        if bash_script:
            file_script = open(self.path_data + "/run_" + self.name + ".sh", "w")
            file_script.write('#!/bin/bash\n\n'
                              'crystal_paths="\n')
            for crystal in self.crystals:
                file_script.write(crystal.path + "\n")
            if self.index > 0:
                file_script.write('"\n\n'
                                  'for crystal in $crystal_paths ; do\n'
                                  'cd "$crystal" || exit \n'
                                  '{0} grompp -f {1}.mdp -c {2}.gro -t {2}.cpt -o {1}.tpr -p topol.top -maxwarn 1 \n'
                                  '{0} mdrun {3} -deffnm {1} \n'
                                  'done \n'
                                  ''.format(self.command, self.name, self.previous_name, self.mdrun_options))
            else:
                file_script.write('"\n\n'
                                  'for crystal in $crystal_paths ; do\n'
                                  'cd "$crystal" || exit \n'
                                  '{0} grompp -f {1}.mdp -c {2}.gro -o {1}.tpr -p topol.top -maxwarn 1 \n'
                                  '{0} mdrun {3} -deffnm {1} \n'
                                  'done \n'
                                  ''.format(self.command, self.name, self.previous_name, self.mdrun_options))
            file_script.close()
        self.project.save()

    def check_normal_termination(self, crystals="all"):
        """
        Verify if the simulation ended correctly and upload new crystal properties.
        :param crystals:
        :return:
        """
        import os
        from PyPol.utilities import get_list, box2cell
        import numpy as np

        if crystals == "all":
            list_crystals = self.crystals
        elif crystals == "incomplete":
            list_crystals = list()
            for crystal in self.crystals:
                if not crystal.completed:
                    list_crystals.append(crystal)
        else:
            list_crystals = get_list(crystals)

        for crystal in list_crystals:
            path_output = crystal.path + self.name + ".log"
            if os.path.exists(path_output):
                file_output = open(path_output)
                lines = file_output.readlines()
                if any("Finished mdrun" in string for string in lines[-30:]):
                    os.chdir(crystal.path)
                    os.system('{} energy -f {}.edr <<< "Potential" > PyPol_Temporary_Potential.txt'
                              ''.format(self.command, self.name))
                    file_pot = open(crystal.path + 'PyPol_Temporary_Potential.txt')
                    for line in file_pot:
                        if line.startswith("Potential"):
                            lattice_energy = float(line.split()[1]) / crystal.Z - \
                                             self.method.molecules[0].potential_energy
                            crystal.Potential = lattice_energy
                            crystal.completed = True
                            break
                    file_pot.close()
                    os.remove(crystal.path + 'PyPol_Temporary_Potential.txt')
                else:
                    print("An error has occurred with Gromacs. Check simulation {} in folder {}."
                          "".format(self.name, crystal.path))
                file_output.close()
            else:
                print("An error has occurred with Gromacs. Check simulation {} in folder {}."
                      "".format(self.name, crystal.path))

        new_rank = dict()
        incomplete_simulations = False
        for crystal in self.crystals:
            if crystal.completed:
                new_rank[crystal.name] = crystal.Potential
                file_gro = open(crystal.path + self.name + ".gro", "r")
                new_box = file_gro.readlines()[-1].split()
                file_gro.close()
                if len(new_box) == 3:
                    new_box = [float(ii) for ii in new_box] + [0., 0., 0., 0., 0., 0.]
                idx_gromacs = [0, 5, 7, 3, 1, 8, 4, 6, 2]
                crystal.box = np.array([float(new_box[ii]) for ii in idx_gromacs]).reshape((3, 3))
                crystal.cell_parameters = box2cell(crystal.box)
                crystal.volume = np.linalg.det(crystal.box)
            else:
                incomplete_simulations = True
                break

        if not incomplete_simulations:
            rank = 1
            for crystal_name in sorted(new_rank, key=lambda c: new_rank[c]):
                for crystal in self.crystals:
                    if crystal.name == crystal_name:
                        crystal.rank = rank
                        if rank == 1:
                            self.global_minima = crystal
                        rank += 1
            self.completed = True

        self.project.save()
