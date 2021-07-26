import numpy as np
from PyPol.utilities import cell2box, translate_molecule  #, point_in_box
import os


class Crystal(object):
    """
    This object stores the relevant information about the crystal, including the molecules and atoms in it.

    Attributes:\n
    - name: ID of the crystal as saved in the project\n
    - label: Alternative name to identify special structures in crystals set. \n
    - index: Crystal index in the set\n
    - path: Folder with the simulations\n
    - box: 3x3 Matrix of the three box vectors a, b, and c with components ay = az = bz = 0\n
    - cell_parameters: Lattice parameters, a, b, c, alpha, beta, gamma\n
    - volume: Volume of the simulation box\n
    - Z: Number of molecules in the cell\n
    - nmoltypes: Types of different molecules in the crystal (For now only one is allowed)\n
    - rank: Rank of the structure in the method crystal set.\n
    - energy: Pot. energy divided by the number of atoms and rescaled by the energy of an isolated molecule.\n
    - state: State of the crystal:\n
        - incomplete: the simulation is not finished
        - complete: the simulation is finished and ready for analysis
        - melted: The simulation is completed but the structure is melted
        - clusterID: Name of the cluster the structure belongs.\n
    - cvs: Collective Variables calculated for this structure.

    Methods:\n
    - help(): print attributes and methods available
    """

    def __init__(self, name):
        """
        This object stores the relevant information about the crystal, including the molecules and atoms in it.
        :param name: ID of the crystal as saved in the project
        """
        self._name = name
        self._label = name
        self._index = None
        self._path = None

        self._box = None
        self._cell_parameters = None
        self._volume = None
        self._density = None
        self._Z = 0
        self._nmoleculestypes = list()

        self._state = "incomplete"
        self._energy = None
        self._rank = 0

        self._cvs = dict()
        self._attributes = dict()

    def __str__(self):
        return """
Crystal {0._index}
Label:            {0._label}
Folder:           {0._path}
Cell Parameters:  {0.cell_parameters}
Volume:           {0._volume}
Z:                {0._Z}""".format(self)

    @staticmethod
    def help():
        return """
This object stores the relevant information about the crystal, including the molecules and atoms in it.

Attributes:
    - name: ID of the crystal as saved in the project 
    - label: Alternative name to identify special structures in crystals set. 
    - index: Crystal index in the set
    - path: Folder with the simulations
    - box: 3x3 Matrix of the three box vectors a, b, and c with components ay = az = bz = 0
    - cell_parameters: Lattice parameters, a, b, c, alpha, beta, gamma
    - volume: Volume of the simulation box
    - Z: Number of molecules in the cell
    - nmoltypes: Types of different molecules in the crystal (For now only one is allowed)
    - rank: Rank of the structure in the method crystal set.
    - energy: Potential energy of the crystal divided by the number of atoms and rescaled by the energy of an 
    isolated molecule.
    - state: State of the crystal:
        - incomplete: the simulation is not finished
        - complete: the simulation is finished and ready for analysis
        - melted: The simulation is completed but the structure is melted
        - clusterID: Name of the cluster the structure belongs.
    - cvs: Collective Variables calculated for this structure.
    
Methods:
    - help(): print attributes and methods available"""

    @property
    def name(self):
        return self._name

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, new_label: str):
        self._label = new_label

    @property
    def index(self):
        return self._index

    @property
    def path(self):
        return self._path

    @property
    def box(self):
        return self._box

    @property
    def cell_parameters(self):
        return "a: {0._cell_parameters[0]} b: {0._cell_parameters[1]} c: {0._cell_parameters[2]} " \
               "alpha: {0._cell_parameters[3]} beta: {0._cell_parameters[4]} gamma: {0._cell_parameters[5]}" \
               "".format(self)

    @property
    def volume(self):
        return self._volume

    # noinspection PyPep8Naming
    @property
    def Z(self):
        return self._Z

    @property
    def nmoltypes(self):
        return self._nmoleculestypes

    @property
    def state(self):
        if self._state == "incomplete":
            return "The simulation for crystal {} is not completed yet".format(self._label)
        elif self._state == "complete":
            return "The simulation for crystal {} is completed and it is ready for other steps and/or analysis" \
                   "".format(self._label)
        elif self._state == "melted":
            return "Crystal {} is melted".format(self._label)
        else:
            return "Crystal {} belongs to cluster {}".format(self._label, self._state)

    @property
    def energy(self):
        return "{:.3f} kJ/mol".format(self._energy)

    @property
    def energy_long(self):
        return "{} kJ/mol".format(self._energy)

    @property
    def rank(self):
        return "Crystal {0._label} is ranked {0._rank}".format(self)

    @property
    def cvs(self):
        txt = "CollectiveVariables\n"
        for cv in self._cvs.keys():
            if not isinstance(self._cvs[cv], dict):
                txt += "{:<24}:\n{}\n\n".format(cv, self._cvs[cv])
            else:
                txt += "{:<24}:\n".format(cv)
                for group in self._cvs[cv].keys():
                    txt += " - {}: {}".format(group, self._cvs[cv][group])
                txt += "\n"
        return txt

    @property
    def molecules(self):
        return self._load_coordinates()

    @property
    def density(self):
        if not hasattr(self, "_density") or not self._density: # TODO Remove hasattr for new projects
            self._density = self._calculate_density()
        return self._density

    def _calculate_density(self):
        # TODO not suitable for 2 or more molecules
        mw = 0
        for atom in self.molecules[0].atoms:
            mw += atom._mass
        return self._Z * 1.6605 * mw / self._volume

    @property
    def attributes(self):
        txt = ""
        if self._attributes:
            for k in self._attributes.keys():
                txt += f"{k} = {self._attributes[k]}\n"
        return txt

    def set_attribute(self, att, val):
        """
        Create a custom attribute for the Crystal.
        :param att: Attribute label
        :param val: Attribute value
        :return:
        """
        self._attributes[att] = val

    def get_attribute(self, att):
        """
        Retrieve an existing attribute from a Molecule object
        :param att: Attribute label
        :return:
        """
        return self._attributes[att]

    def update_molecules(self, molecules):
        """
        Save a list of molecules in the current state.
        :param molecules:
        :return:
        """
        self._save_coordinates(molecules)

    def _save_coordinates(self, molecules):
        """
        Save molecules in the crystal folder. This is done to limit the memory use.
        :param molecules: list of Molecule objects
        :return:
        """
        import pickle
        import os
        if os.path.exists("{}/.initial_crystal.pkl".format(self._path)):
            os.rename("{}/.initial_crystal.pkl".format(self._path), "{}/.initial_crystal.bck.pkl".format(self._path))
        with open("{}/.initial_crystal.pkl".format(self._path), "wb") as file_pickle:
            pickle.dump(molecules, file_pickle)

    def _load_coordinates(self, use_backup=False):
        """
        Load the Molecule objects stored in the crystal folder.
        :param use_backup:
        :return:
        """
        import pickle
        import os
        file_pickle = "{}/.initial_crystal.pkl".format(self._path)
        if use_backup:
            file_pickle = "{}/.initial_crystal.bck.pkl".format(self._path)
        if os.path.exists(file_pickle):
            molecules = pickle.load(open(file_pickle, "rb"))
            return molecules
        else:
            print("Error: No molecules found in {}.initial_crystal.bck.pkl.".format(self._path))
            exit()

    @staticmethod
    def _copy_properties(crystal):
        """
        Create a copy of the input crystal
        :param crystal: Crystal object to copy
        :return:
        """
        import copy

        new_crystal = Crystal(crystal._name)
        new_crystal._index = crystal._index
        new_crystal._Z = crystal._Z
        new_crystal._nmoleculestypes = crystal._nmoleculestypes
        new_crystal._path = crystal._path
        new_crystal._cell_parameters = copy.deepcopy(crystal._cell_parameters)
        new_crystal._box = copy.deepcopy(crystal._box)
        new_crystal._state = "incomplete"
        new_crystal._cvs = dict()
        new_crystal._attributes = crystal._attributes
        return new_crystal

    @staticmethod
    def _arrange_atoms_in_molecules(molecules: list):
        """
        Check if the atoms in a Molecule object belongs to a single molecule. This is done to prevent errors from
        openbabel or the CSD Python API when assigning residues index. The check is performed by converting molecules to
        graphs and looking at their edges with the Breadth First Search algorithm.
        :param molecules: List of Molecule objects
        :return:
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import breadth_first_order
        new_molecules = list()
        molidx = 0
        for molecule in molecules:
            graph = csr_matrix(molecule.contact_matrix)
            removed = []
            for atom in range(len(molecule._atoms)):
                if atom in removed:
                    continue

                bfs = breadth_first_order(graph, atom, False, False)
                removed = removed + list(bfs)

                new_molecule = Molecule(molecule._residue)
                new_molecule._index = molidx
                molidx += 1
                new_molecule._atoms = [molecule._atoms[i] for i in range(len(molecule._atoms)) if i in bfs]
                new_molecule._natoms = len(new_molecule._atoms)
                for natom in new_molecule._atoms:
                    natom._index = natom._index - (new_molecule._natoms * new_molecule._index)
                    natom._bonds = [bond - (new_molecule._natoms * new_molecule._index) for bond in natom._bonds]

                new_molecule._calculate_centroid()
                new_molecule._forcefield = molecule._forcefield
                new_molecule._potential_energy = molecule._potential_energy
                new_molecule._generate_contact_matrix()

                new_molecules.append(new_molecule)
        return new_molecules

    @staticmethod
    def _loadfrompdb(name, path_pdb, include_atomtype=False):
        """
        Load Crystal from a PDB file. If include_atomtype is set to True, it also uses the 'atomtype' program from
        AmberTools to identify the atom types contained in the pdb file. If the pdb file contain the 'CONECT' keyword,
        bonds are taken from it. Alternatively, they can be generated with the 'atomtype' program.
        :param name: ID of the crystal, usually the basename of the PDB file
        :param path_pdb: Path of the PDB file
        :param include_atomtype: Include the identification of the atom types
        :return: Crystal Object
        """

        new_crystal = Crystal(name)
        molecules = list()
        # Open pdb file
        bonds_imported = False
        file_pdb = open(path_pdb)
        for line in file_pdb:
            # Import Crystal Properties
            if line.startswith("CRYST1"):
                new_crystal._cell_parameters = np.array(
                    [float(line[6:15]) / 10., float(line[15:24]) / 10., float(line[24:33]) / 10.,
                     float(line[33:40]), float(line[40:47]), float(line[47:54])])
                new_crystal._box = cell2box(new_crystal._cell_parameters)

                new_crystal._volume = np.linalg.det(new_crystal._box)

            # Import Molecular and Atom Properties
            elif line.startswith("ATOM") or line.startswith("HETATM"):
                atom_index = int(line[6:11]) - 1
                atom_label = line[12:16]
                molecule_name = line[17:20]
                molecule_index = int(line[22:26]) - 1
                atom_x, atom_y, atom_z = (float(line[30:38]) / 10., float(line[38:46]) / 10., float(line[46:54]) / 10.)
                atom_element = line[76:78]

                if not molecules:
                    molecules.append(Molecule(molecule_name, molecule_index))
                elif molecules[-1]._index < molecule_index:
                    molecules.append(Molecule(molecule_name, molecule_index))

                for molecule in molecules:
                    if molecule_index == molecule._index:
                        molecule._atoms.append(Atom(index=atom_index, label=atom_label, ff_type=None, atomtype=None,
                                                    coordinates=[atom_x, atom_y, atom_z], element=atom_element,
                                                    bonds=None))

            elif line.startswith("CONECT"):
                bonds_imported = True
                atom_index = int(line[6:11]) - 1
                bonds = [int(bond) - 1 for bond in line.split()[2:]]
                for molecule in molecules:
                    molecule._natoms = len(molecule._atoms)
                    for atom in molecule._atoms:
                        if atom_index == atom._index:
                            atom._bonds = bonds
                            break
        file_pdb.close()

        # Import atomtypes (and eventually bonds) from .ac file
        if not include_atomtype and not bonds_imported:
            print("Something is wrong with the PDB file: No bonds found.\n"
                  "Try to generate a structure.ac file with the ambertool 'atomtype' or 'antechamber' and rerun "
                  "with the parameter 'include_atomtype=True'.")
            return

        # Atom types that can be switched by antechamber, especially from experimental data. They are considered
        # equivalent only during the index assignation in the generate_input module but not during the simulation.
        # (This is because the atomtypes are used only for the reindexing and not to generate a force field)
        equivalent_atom_types = {
            'cq': 'cp',
            'cd': 'c2',  # Not true but equivalent for the index assignation. It should be 'cd': 'cc'. However,
            'cc': 'c2',  # antechamber tend to switch them when the ring is not perfectly planar.
            'cf': 'ce',
            'ch': 'cg',
            'nd': 'nc',
            'nf': 'ne',
            'pd': 'pc',
            'pf': 'pe',
            # 'hs': 'ha', # remove!
            # 's6': 'Si'
        }

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
                        for molecule in molecules:
                            for atom in molecule._atoms:
                                if atom._index == a1:
                                    if atom._bonds == [None] or atom._bonds is None:
                                        atom._bonds = list()
                                    atom._bonds.append(a2)
                                elif atom._index == a2:
                                    if atom._bonds == [None] or atom._bonds is None:
                                        atom._bonds = list()
                                    atom._bonds.append(a1)
            file_ac.close()

            atomtype_index = 0
            for molecule in molecules:
                molecule._natoms = len(molecule._atoms)
                for atom in molecule._atoms:
                    atom._type = list_atomtypes[atomtype_index]
                    if not bonds_imported:
                        atom._index = atom._index - (molecule._natoms * molecule._index)
                        # atom.bonds = sorted([bond - (molecule.natoms * molecule.index) for bond in atom.bonds])
                        atom._bonds = [bond - (molecule._natoms * molecule._index) for bond in atom._bonds]
                    atomtype_index += 1

        # Check if molecule contains more than one component.
        molecules = Crystal._arrange_atoms_in_molecules(molecules)

        # Calculate geometrical centre of each molecule and remove replicas
        def check_replica(molecule, list_molecules):
            for refmol in list_molecules:
                if np.linalg.norm(molecule.centroid - refmol.centroid) < 0.1:
                    return False
            return True

        new_molecule_index = 0
        new_molecules = []
        for molecule in molecules:
            molecule = translate_molecule(molecule, new_crystal._box)
            if check_replica(molecule, new_molecules):
                molecule._index = new_molecule_index
                new_molecules.append(molecule)
                new_molecule_index += 1
        new_crystal._Z = len(new_molecules)
        new_crystal._path = os.path.dirname(path_pdb) + "/"
        new_crystal._save_coordinates(new_molecules)
        return new_crystal

    def _save_pdb(self, path_pdb):
        """
        Save a PDB file of the crystal.
        :param path_pdb: Output path of the file
        """
        import datetime

        today = datetime.datetime.now()
        file_pdb = open(path_pdb, "w")
        file_pdb.write("{:6}    {:40}{:8}\n".format("HEADER", "Crystal " + str(self._index),
                                                    today.strftime("%d-%m-%y")))
        file_pdb.write("{:6}{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f}{:>11}{:>4}\n"
                       "".format("CRYST1", self._cell_parameters[0] * 10, self._cell_parameters[1] * 10,
                                 self._cell_parameters[2] * 10, self._cell_parameters[3], self._cell_parameters[4],
                                 self._cell_parameters[5], "P1", self._Z))
        tot_atoms = 0
        for molecule in self._load_coordinates():
            tot_atoms += molecule._natoms
            for atom in molecule._atoms:
                atom_index = atom._index + 1 + (molecule._index * molecule._natoms)
                file_pdb.write("{:6}{:>5} {:>4}{:1}{:3} {:1}{:>4}{:1}   {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6}"
                               "          {:>2}{:2}\n"
                               "".format("HETATM", atom_index, atom._label, " ", molecule._residue, " ",
                                         molecule._index + 1,
                                         " ", atom._coordinates[0] * 10, atom._coordinates[1] * 10,
                                         atom._coordinates[2] * 10,
                                         1.00, " ", atom._element, " "))
        tot_bonds = 0
        for molecule in self._load_coordinates():
            for atom in molecule._atoms:
                if atom._bonds:
                    tot_bonds += 1
                    n = molecule._index * molecule._natoms + 1
                    atom_index = atom._index + n
                    file_pdb.write("{:6}{:>5}".format("CONECT", atom_index))
                    for bond in atom._bonds:
                        bond += n
                        file_pdb.write("{:5}".format(bond))
                    file_pdb.write("\n")
        file_pdb.write("{:6}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}\nEND"
                       "".format("MASTER", 0, 0, 0, 0, 0, 0, 0, 0, tot_atoms, 0, tot_bonds, 0))
        file_pdb.close()

    def _save_gro(self, path_gro):
        """
        Save a Gromacs GRO file of the crystal
        :param path_gro: Output path of the GRO file
        """
        import datetime

        today = datetime.datetime.now()
        file_gro = open(path_gro, "w")
        file_gro.write("{:6}    {:40}{:8}\n".format("HEADER", "Crystal " + str(self._index),
                                                    today.strftime("%d-%m-%y")))

        tot_atoms = 0
        for molecule in self._load_coordinates():
            tot_atoms += molecule._natoms

        file_gro.write("{:>5}\n".format(tot_atoms))
        for molecule in self._load_coordinates():
            for atom in molecule._atoms:
                atom_index = atom._index + 1 + (molecule._index * molecule._natoms)
                file_gro.write("{:>5}{:5}{:>5}{:>5}{:8.3f}{:8.3f}{:8.3f}\n"
                               "".format(molecule._index + 1, molecule._residue, atom._label, atom_index,
                                         atom._coordinates[0], atom._coordinates[1], atom._coordinates[2]))
        file_gro.write("{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}{:>10.5f}\n"
                       "".format(self._box[0, 0], self._box[1, 1], self._box[2, 2],
                                 self._box[1, 0], self._box[2, 0], self._box[0, 1],
                                 self._box[2, 1], self._box[0, 2], self._box[1, 2]))
        file_gro.close()


class Molecule(object):
    """
    This class stores relevant details about a molecule and the atoms in it.
    Parameters can be derived from a coordinate or a topology file depending on its purpose.

    Attributes:\n
    - residue: The molecule label as specified in the forcefield\n
    - index: Index of the molecule inside the crystal\n
    - atoms: List of Atoms objects\n
    - natoms: Number of atoms in the molecule\n
    - centroid: The geometrical center of the molecule\n
    - contact_matrix: A NxN matrix (with N=natoms) with 1 elements if atoms are bonded and 0 if not.\n

    Methods:\n
    - help(): print attributes and methods available
    """

    def __init__(self, name, index=None):
        """
        This class stores relevant details about a molecule and the atoms in it.
        :param name: The molecule label as specified in the forcefield
        """
        self._index = index
        self._residue = name
        self._atoms = list()
        self._natoms = 0
        self._centroid = None
        self._forcefield = None
        self._potential_energy = 0.0
        self._contact_matrix = None
        self._attributes = {}

    def __str__(self):
        txt = "Molecule {0.index}: ResidueName = {0.residue}, NumberOfAtoms = {0.natoms}".format(self)
        if self._attributes:
            for k in self._attributes.keys():
                txt += f"{k} = {self._attributes[k]}, "

        return txt[:-2]

    @staticmethod
    def help():
        return """
This class stores relevant details about a molecule and the atoms in it.
Parameters can be derived from a coordinate or a topology file depending on its purpose.

Attributes:
- residue: The molecule label as specified in the forcefield
- index: Index of the molecule inside the crystal
- atoms: List of Atoms objects
- natoms: Number of atoms in the molecule
- centroid: The geometrical center of the molecule
- contact_matrix: A NxN matrix (with N=natoms) with 1 elements if atoms are bonded and 0 if not.

Example: 
To access molecule info:
for molecule in crystal.molecules:
    print(molecule)
    print("Atoms:")
    print("Index Label Element Type Mass Charge Bonds")
    for atom in molecule.atoms:
        print("{0._index} {0._label} {0._element} {0._type} {0._mass} {0._charge} {0._bonds} ".format(atom))"""

    @property
    def residue(self):
        return self._residue

    @property
    def index(self):
        return self._index

    @property
    def atoms(self):
        return self._atoms

    @property
    def natoms(self):
        if not self._natoms:
            self._natoms = len(self._atoms)
        return self._natoms

    @property
    def centroid(self):
        if self._centroid is None:
            self._calculate_centroid()
            return self._centroid
        else:
            return self._centroid

    @property
    def contact_matrix(self):
        if self._contact_matrix is None:
            self._generate_contact_matrix()
            return self._contact_matrix
        else:
            return self._contact_matrix

    @property
    def attributes(self):
        txt = ""
        if self._attributes:
            for k in self._attributes.keys():
                txt += f"{k} = {self._attributes[k]}\n"
        return txt

    def set_attribute(self, att, val):
        """
        Create a custom attribute for the molecule.
        :param att: Attribute label
        :param val: Attribute value
        :return:
        """
        self._attributes[att] = val

    def get_attribute(self, att):
        """
        Retrieve an existing attribute from a Molecule object
        :param att: Attribute label
        :return:
        """
        return self._attributes[att]

    def _calculate_centroid(self):
        """
        Calculate the geometrical center of the molecule.
        """
        atoms_coordinates = self._atoms[0]._coordinates
        for atom in self._atoms[1:]:
            atoms_coordinates = np.vstack((atoms_coordinates, atom._coordinates))
        self._centroid = np.mean(atoms_coordinates, axis=0)

    def _generate_contact_matrix(self):
        """
        Generate a NxN matrix (with N=number of atoms) with 1 elements if atoms are bonded and 0 if not.
        """
        cmat = np.full((len(self._atoms), len(self._atoms)), 0)
        for ai in range(len(self._atoms)):
            atom = self._atoms[ai]
            for bond in atom._bonds:
                aj = bond - min([i._index for i in self._atoms])
                cmat[ai, aj] = 1
        self._contact_matrix = cmat

    def _save_gro(self, path_gro, append=False, header=""):
        """
        Save a Gromcas GRO file of the molecule
        :param path_gro: output path
        :param append: Append new text to the file
        :param header: Add string at the beginning of the file or end of the previous file if append is True
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
            file_gro.write("{:6}    {:40}{:8}\n".format("HEADER", "Molecule " + str(self._index),
                                                        today.strftime("%d-%m-%y")))

        file_gro.write("{:>5}\n".format(len(self._atoms)))

        for atom in self._atoms:
            atom_index = atom._index + 1 + (self._index * len(self._atoms))
            file_gro.write("{:>5}{:5}{:>5}{:>5}{:8.3f}{:8.3f}{:8.3f}\n"
                           "".format(self._index + 1, self._residue, atom._label, atom_index,
                                     atom._coordinates[0], atom._coordinates[1], atom._coordinates[2]))
        file_gro.write("{:>10.5f}{:>10.5f}{:>10.5f}\n".format(1.0, 1.0, 1.0))
        file_gro.close()


class Atom(object):
    """
    The Atom Class which stores relevant info about each atom of the molecule.

    Attributes:\n
    - label: Atom label as in the original coordinate file or in the forcefield if index reassignation is performed.\n
    - index: Index of the atom inside the molecule.\n
    - ff_type: Atom type as written in the forcefield.\n
    - type: Atom type identified by the AmberTools program 'atomtype'.\n
    - coordinates: Coordinates of the atom.\n
    - element: Element of the atom.\n
    - bonds: Index of atoms bonded to it.\n
    - charge: Charge of the atom\n
    - mass: Mass of the atom.

    Methods:\n
    - help(): print attributes and methods available
    """

    def __init__(self, label, index=None, ff_type=None, atomtype=None, coordinates=None, element=None, bonds=None,
                 charge=None, mass=None):
        """
        The Atom Class which stores relevant info about each atom of the molecule.
        Parameters:
        :param label: Atom label as  written in the original coordinate file. If atom index reassignation is performed,
        it has the label used by the forcefield.
        :param index: Index of the atom inside the molecule.
        :param ff_type: Atom type as written in the forcefield.
        :param atomtype: Atom type identified by the AmberTools program 'atomtype'.
        :param coordinates: Coordinates of the atom.
        :param element: Element of the atom.
        :param bonds: Index of atoms bonded to it.
        :param charge: Charge of the atom
        :param mass: Mass of the atom.
        """

        self._index = index
        self._label = label
        self._ff_type = ff_type
        self._type = atomtype
        self._coordinates = coordinates
        self._element = element
        self._bonds = bonds
        self._charge = charge
        self._mass = mass

    def __str__(self):
        return "Atom {0._index}: Label = {0._label}, Element = {0._element}, Type = {0._type}, Mass = {0._mass}, " \
               "Charge = {0._charge}, Bonds = {0._bonds} ".format(self)

    @staticmethod
    def help():
        return """
The Atom Class which stores relevant info about each atom of the molecule.

Attributes:
- label: Atom label as  written in the original coordinate file. If atom index reassignation is performed,
         it has the label used by the forcefield.
- index: Index of the atom inside the molecule.
- ff_type: Atom type as written in the forcefield.
- type: Atom type identified by the AmberTools program 'atomtype'.
- coordinates: Coordinates of the atom.
- element: Element of the atom.
- bonds: Index of atoms bonded to it.
- charge: Charge of the atom\n
- mass: Mass of the atom.

Example:
To access atom info:
for atom in molecule.atoms:
    print(atom.index, atom.label)"""

    @property
    def index(self):
        return self._index

    @property
    def label(self):
        return self._label

    @property
    def type(self):
        return self._type

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def element(self):
        return self._element

    @property
    def bonds(self):
        return self._bonds

    @property
    def charge(self):
        return self._charge

    @property
    def mass(self):
        return self._mass
