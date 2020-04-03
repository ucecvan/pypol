class Crystal(object):

    def __init__(self, name):
        """

        :param name:
        """
        self.name = name
        self.index = None
        self.path = None

        self.box = None
        self.cell_parameters = None
        self.volume = None

        self.molecules = list()
        self.Z = 0
        self.nmoleculestypes = list()
        self.melted = False
        self.completed = False
        self.Potential = None
        self.rank = 0
        self.cvs = dict()

    def save(self, remove_molecules=True):
        """

        :param remove_molecules:
        :return:
        """
        import pickle
        import os
        if os.path.exists("{}/.initial_crystal.pkl".format(self.path)):
            os.rename("{}/.initial_crystal.pkl".format(self.path), "{}/.initial_crystal.bck.pkl".format(self.path))
        with open("{}/.initial_crystal.pkl".format(self.path), "wb") as file_pickle:
            pickle.dump(self, file_pickle)
        if remove_molecules:
            self.molecules = list()

    def load(self, use_backup=False):
        """

        :param use_backup:
        :return:
        """
        import pickle
        import os
        file_pickle = "{}/.initial_crystal.pkl".format(self.path)
        if use_backup:
            file_pickle = "{}/.initial_crystal.bck.pkl".format(self.path)
        if os.path.exists(file_pickle):
            crystal = pickle.load(open(file_pickle, "rb"))
            return crystal
        else:
            print("Error: No crystal found in {}/.initial_crystal.bck.pkl.".format(self.path))

    def load_molecules(self, use_backup=False):
        """

        :param use_backup:
        :return:
        """
        import pickle
        import os
        file_pickle = "{}/.initial_crystal.pkl".format(self.path)
        if use_backup:
            file_pickle = "{}/.initial_crystal.bck.pkl".format(self.path)
        if os.path.exists(file_pickle):
            crystal = pickle.load(open(file_pickle, "rb"))
            return crystal.molecules
        else:
            print("Error: No crystal found in {}.initial_crystal.bck.pkl.".format(self.path))

    @staticmethod
    def copy_properties(crystal):
        """

        :param crystal:
        :return:
        """
        import copy

        new_crystal = Crystal(crystal.name)
        new_crystal.index = crystal.index
        new_crystal.Z = crystal.Z
        new_crystal.nmoleculestypes = crystal.nmoleculestypes
        new_crystal.path = crystal.path
        new_crystal.cell_parameters = copy.deepcopy(crystal.cell_parameters)
        new_crystal.box = copy.deepcopy(crystal.box)
        new_crystal.cvs = copy.deepcopy(crystal.CVs)
        return new_crystal

    @staticmethod
    def _recursive_group_check(atom_i, molecule):
        """

        :param atom_i:
        :param molecule:
        :return:
        """
        for j in atom_i.bonds:
            atom_j = molecule.atoms[j]
            if not atom_j.group:
                atom_j.group = atom_i.group
                Crystal._recursive_group_check(atom_j, molecule)

    @staticmethod
    def loadfrompdb(name, path_pdb, include_atomtype=False):
        """

        :param name:
        :param path_pdb:
        :param include_atomtype:
        :return:
        """
        import numpy as np
        from PyPol.Defaults.defaults import equivalent_atom_types
        from PyPol.utilities import cell2box, point_in_box
        import os

        new_crystal = Crystal(name)
        # Open pdb file
        bonds_imported = False
        file_pdb = open(path_pdb)
        for line in file_pdb:
            # Import Crystal Properties
            if line.startswith("CRYST1"):
                new_crystal.cell_parameters = np.array(
                    [float(line[6:15]) / 10., float(line[15:24]) / 10., float(line[24:33]) / 10.,
                     float(line[33:40]), float(line[40:47]), float(line[47:54])])
                new_crystal.box = cell2box(new_crystal.cell_parameters)

                new_crystal.volume = np.linalg.det(new_crystal.box)

            # Import Molecular and Atom Properties
            elif line.startswith("ATOM") or line.startswith("HETATM"):
                atom_index = int(line[6:11]) - 1
                atom_label = line[12:16]
                molecule_name = line[17:20]
                molecule_index = int(line[22:26]) - 1
                atom_x, atom_y, atom_z = (float(line[30:38]) / 10., float(line[38:46]) / 10., float(line[46:54]) / 10.)
                atom_element = line[76:78]

                if not new_crystal.molecules:
                    new_crystal.molecules.append(Molecule.load(molecule_index, molecule_name))
                elif new_crystal.molecules[-1].index < molecule_index:
                    new_crystal.molecules.append(Molecule.load(molecule_index, molecule_name))

                for molecule in new_crystal.molecules:
                    if molecule_index == molecule.index:
                        molecule.atoms.append(Atom.loadfromcrd(atom_index, atom_label, None, None,
                                                               [atom_x, atom_y, atom_z], atom_element, None))

            elif line.startswith("CONECT"):
                bonds_imported = True
                atom_index = int(line[6:11]) - 1
                bonds = [int(bond) - 1 for bond in line.split()[2:]]
                for molecule in new_crystal.molecules:
                    molecule.natoms = len(molecule.atoms)
                    for atom in molecule.atoms:
                        if atom_index == atom.index:
                            atom.index = atom.index - (molecule.natoms * molecule.index)
                            atom.bonds = [bond - (molecule.natoms * molecule.index) for bond in bonds]
                            break
        file_pdb.close()

        # Import atomtypes (and eventually bonds) from .ac file
        if not include_atomtype and not bonds_imported:
            print("Something is wrong with the PDB file: No bonds found.\n"
                  "Try to generate a structure.ac file with the ambertool 'atomtype' or 'antechamber' and rerun "
                  "with the parameter 'include_atomtype=True'.")
            return

        if include_atomtype:
            path_ac = path_pdb[:-3] + "ac"
            list_atomtypes = list()
            file_ac = open(path_ac)
            for line in file_ac:
                if line.startswith("ATOM"):
                    atom_type = line.split()[-1]
                    if atom_type in equivalent_atom_types:
                        atom_type = equivalent_atom_types[atom_type]
                    list_atomtypes.append(atom_type)
                if not bonds_imported:
                    if line.startswith("BOND"):
                        a1 = int(line[9:14]) - 1
                        a2 = int(line[14:19]) - 1
                        for molecule in new_crystal.molecules:
                            for atom in molecule.atoms:
                                if atom.index == a1:
                                    if atom.bonds == [None] or atom.bonds is None:
                                        atom.bonds = list()
                                    atom.bonds.append(a2)
                                elif atom.index == a2:
                                    if atom.bonds == [None] or atom.bonds is None:
                                        atom.bonds = list()
                                    atom.bonds.append(a1)
            file_ac.close()

            atomtype_index = 0
            for molecule in new_crystal.molecules:
                molecule.natoms = len(molecule.atoms)
                for atom in molecule.atoms:
                    atom.type = list_atomtypes[atomtype_index]
                    if not bonds_imported:
                        atom.index = atom.index - (molecule.natoms * molecule.index)
                        # atom.bonds = sorted([bond - (molecule.natoms * molecule.index) for bond in atom.bonds])
                        atom.bonds = [bond - (molecule.natoms * molecule.index) for bond in atom.bonds]
                    atomtype_index += 1

        # Check if molecule contains more than one component.
        gn = 1
        for molecule in new_crystal.molecules:
            for atom in molecule.atoms:
                if not atom.group:
                    atom.group = gn
                    gn += 1
                    new_crystal._recursive_group_check(atom, molecule)

        if gn - 2 != new_crystal.molecules[-1].index:
            new_molecule_list = list()
            for molecule in new_crystal.molecules:
                for atom in molecule.atoms:
                    molecule_name = molecule.residue
                    molecule_index = atom.group - 1
                    if not new_molecule_list or new_molecule_list[-1].index < molecule_index:
                        new_molecule_list.append(Molecule.load(molecule_index, molecule_name))
                    for new_molecule in new_molecule_list:
                        if molecule_index == new_molecule.index:
                            new_molecule.atoms.append(atom)
                            new_molecule.natoms = len(new_molecule.atoms)
            new_crystal.molecules = new_molecule_list
            for molecule in new_crystal.molecules:
                for atom in molecule.atoms:
                    n = int(atom.index / molecule.natoms)
                    atom.index = atom.index - (molecule.natoms * n)
                    atom.bonds = [bond - (molecule.natoms * n) for bond in atom.bonds]

        # Calculate geometrical centre of each molecule and remove replicas
        new_molecule_index = 0
        for molecule in new_crystal.molecules:
            molecule.calculate_centroid()
            if not point_in_box(molecule.centroid, new_crystal.box):
                # print("Molecule '{}' removed".format(molecule.index))
                new_crystal.molecules.remove(molecule)
            else:
                # print("Molecule '{}' inside the box".format(molecule.index))
                molecule.index = new_molecule_index
                new_molecule_index += 1
        new_crystal.Z = len(new_crystal.molecules)
        new_crystal.path = os.path.dirname(path_pdb)
        return new_crystal

    def save_pdb(self, path_pdb):
        """

        :param path_pdb:
        :return:
        """
        import datetime

        today = datetime.datetime.now()
        file_pdb = open(path_pdb, "w")
        file_pdb.write("{:6}    {:40}{:8}\n".format("HEADER", "Crystal " + str(self.index), today.strftime("%d-%m-%y")))
        file_pdb.write("{:6}{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f}{:>11}{:>4}\n"
                       "".format("CRYST1", self.cell_parameters[0] * 10, self.cell_parameters[1] * 10,
                                 self.cell_parameters[2] * 10, self.cell_parameters[3], self.cell_parameters[4],
                                 self.cell_parameters[5], "P1", self.Z))
        tot_atoms = 0
        for molecule in self.load_molecules():
            tot_atoms += molecule.natoms
            for atom in molecule.atoms:
                atom_index = atom.index + 1 + (molecule.index * molecule.natoms)
                file_pdb.write("{:6}{:>5} {:>4}{:1}{:3} {:1}{:>4}{:1}   {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6}"
                               "          {:>2}{:2}\n"
                               "".format("HETATM", atom_index, atom.label, " ", molecule.residue, " ",
                                         molecule.index + 1,
                                         " ", atom.coordinates[0] * 10, atom.coordinates[1] * 10,
                                         atom.coordinates[2] * 10,
                                         1.00, " ", atom.element, " "))
        tot_bonds = 0
        for molecule in self.load_molecules():
            for atom in molecule.atoms:
                if atom.bonds:
                    tot_bonds += 1
                    n = molecule.index * molecule.natoms + 1
                    atom_index = atom.index + n
                    file_pdb.write("{:6}{:>5}".format("CONECT", atom_index))
                    for bond in atom.bonds:
                        bond += n
                        file_pdb.write("{:5}".format(bond))
                    file_pdb.write("\n")
        file_pdb.write("{:6}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}\nEND"
                       "".format("MASTER", 0, 0, 0, 0, 0, 0, 0, 0, tot_atoms, 0, tot_bonds, 0))
        file_pdb.close()

    def save_gro(self, path_gro):
        """

        :param path_gro:
        :return:
        """
        import datetime

        today = datetime.datetime.now()
        file_gro = open(path_gro, "w")
        file_gro.write("{:6}    {:40}{:8}\n".format("HEADER", "Crystal " + str(self.index), today.strftime("%d-%m-%y")))

        tot_atoms = 0
        for molecule in self.load_molecules():
            tot_atoms += molecule.natoms

        file_gro.write("{:>5}\n".format(tot_atoms))
        for molecule in self.load_molecules():
            for atom in molecule.atoms:
                atom_index = atom.index + 1 + (molecule.index * molecule.natoms)
                file_gro.write("{:>5}{:5}{:>5}{:>5}{:8.3f}{:8.3f}{:8.3f}\n"
                               "".format(molecule.index + 1, molecule.residue, atom.label, atom_index,
                                         atom.coordinates[0], atom.coordinates[1], atom.coordinates[2]))
        file_gro.write("{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}\n"
                       "".format(self.box[0, 0], self.box[1, 1], self.box[2, 2],
                                 self.box[1, 0], self.box[2, 0], self.box[0, 1],
                                 self.box[2, 1], self.box[0, 2], self.box[1, 2]))
        file_gro.close()


class Molecule(object):

    def __init__(self, name):
        """

        :param name:
        """
        self.index = None
        self.residue = name
        self.atoms = list()
        self.natoms = 0
        self.centroid = None
        self.forcefield = None
        self.potential_energy = 0.0

    @staticmethod
    def load(index, name):
        """

        :param index:
        :param name:
        :return:
        """
        new_molecule = Molecule(name)
        new_molecule.index = index
        return new_molecule

    def calculate_centroid(self):
        """

        :return:
        """
        import numpy as np
        atoms_coordinates = self.atoms[0].coordinates
        for atom in self.atoms[1:]:
            atoms_coordinates = np.vstack((atoms_coordinates, atom.coordinates))
        self.centroid = np.mean(atoms_coordinates, axis=0)

    def save_gro(self, path_gro, append=False, header=""):
        """

        :param path_gro:
        :param append:
        :param header:
        :return:
        """
        import datetime
        today = datetime.datetime.now()
        if append:
            file_gro = open(path_gro, "a")
        else:
            file_gro = open(path_gro, "w")
        if header:
            file_gro.write(header + "\n")
        else:
            file_gro.write("{:6}    {:40}{:8}\n".format("HEADER", "Molecule " + str(self.index),
                                                        today.strftime("%d-%m-%y")))

        file_gro.write("{:>5}\n".format(len(self.atoms)))

        for atom in self.atoms:
            atom_index = atom.index + 1 + (self.index * len(self.atoms))
            file_gro.write("{:>5}{:5}{:>5}{:>5}{:8.3f}{:8.3f}{:8.3f}\n"
                           "".format(self.index + 1, self.residue, atom.label, atom_index,
                                     atom.coordinates[0], atom.coordinates[1], atom.coordinates[2]))
        file_gro.write("{:>10.5f}{:>10.5f}{:>10.5f}\n".format(1.0, 1.0, 1.0))
        file_gro.close()


class Atom(object):

    def __init__(self, name):
        """

        :param name:
        """
        self.index = None
        self.label = name
        self.ff_type = None
        self.type = None
        self.coordinates = None
        self.element = None
        self.bonds = None
        self.charge = None
        self.mass = None
        self.group = False

    @staticmethod
    def loadfromcrd(index, name, ff_type, gaff_type, coordinates, element, bonds):
        """

        :param index:
        :param name:
        :param ff_type:
        :param gaff_type:
        :param coordinates:
        :param element:
        :param bonds:
        :return:
        """
        new_atom = Atom(name)
        new_atom.index = index
        new_atom.ff_type = ff_type
        new_atom.type = gaff_type
        new_atom.coordinates = coordinates
        new_atom.element = element
        new_atom.bonds = bonds
        return new_atom

    @staticmethod
    def loadfromff(index, gaff_type, ff_type, name, bonds=None, coordinates=None, charge=None, mass=None):
        """

        :param index:
        :param gaff_type:
        :param ff_type:
        :param name:
        :param bonds:
        :param coordinates:
        :param charge:
        :param mass:
        :return:
        """
        if bonds is None:
            bonds = list()
        new_atom = Atom(name)
        new_atom.index = index
        new_atom.type = gaff_type
        new_atom.ff_type = ff_type
        new_atom.coordinates = coordinates
        new_atom.bonds = bonds
        new_atom.charge = charge
        new_atom.mass = mass
        return new_atom
