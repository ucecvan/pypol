import copy
import os
import subprocess as sbp
from shutil import copyfile
from typing import Union
import progressbar
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from PyPol.crystals import Crystal, Molecule, Atom
from PyPol.utilities import create, box2cell, cell2box, get_list_crystals


class _GroDef(object):
    """
    Default Properties to be used in the Gromacs method and simulation classes.

    Attributes:\n
    - name: Name used to specify the object and print outputs
    - gromacs: Gromacs command line
    - mdrun_options: Options to be added to Gromacs mdrun command. For example '-v', '-v -nt 1', '-plumed plumed.dat'.
    - atomtype: 'atomtype' command line
    - pypol_directory: PyPol directory with defaults inputs
    - path_data: data folder in which simulations are performed
    - path_output: Output folder in which results are written
    - path_input: Input folder to store inputs
    - intermol: Path of the 'convert.py' InterMol program
    - lammps: LAMMPS command line
    """

    def __init__(self, name: str, gromacs: str, mdrun_options: str, atomtype: str, pypol_directory: str,
                 path_data: str, path_output: str, path_input: str, intermol: str, lammps: str):
        self._name = name
        self._gromacs = gromacs
        self._mdrun_options = mdrun_options
        self._atomtype = atomtype

        self._pypol_directory = pypol_directory
        self._path_data = path_data
        self._path_output = path_output
        self._path_input = path_input

        self._intermol = intermol
        self._lammps = lammps

    # Properties
    @property
    def name(self):
        return self._name

    @property
    def gromacs(self):
        return self._gromacs

    @property
    def mdrun_options(self):
        return self._mdrun_options

    @mdrun_options.setter
    def mdrun_options(self, opt: str):
        self._mdrun_options = opt

    @property
    def atomtype(self):
        return self._atomtype

    @property
    def path_data(self):
        return self._path_data

    @property
    def path_output(self):
        return self._path_output

    @property
    def path_input(self):
        return self._path_input

    @property
    def intermol(self):
        return self._intermol

    @property
    def lammps(self):
        return self._lammps


class Method(_GroDef):
    # TODO Create module update_crystal_list.
    # TODO Create anchor option to allow non-Sequential simulations list
    # TODO include more than one molecule
    """
    The Method object defines the forcefield and the simulations to be used in the analysis.
    Gromacs is used for MD simulations.

    Attributes:\n
    - name: Name used to specify the object and print outputs
    - package: The package used for MD.
    - topology: Path to the Gromacs topology file .top.
    - nmolecules: Number of molecules (or .itp files) in the topology
    - crystals: list of Crystal objects contained in the method. This refers to the crystals prior to any simulation.
    - gromacs: Gromacs command line
    - mdrun_options: Options to be added to Gromacs mdrun command. For example '-v', '-v -nt 1', '-plumed plumed.dat'.
    - atomtype: 'atomtype' command line
    - intermol: Path of the 'convert.py' InterMol program
    - lammps: LAMMPS command line
    - pypol_directory: PyPol directory with defaults inputs
    - path_data: data folder in which simulations are performed
    - path_output: Output folder in which results are written
    - path_input: Input folder to store inputs

    Methods:\n
    - get_cv(name): return the CV parameters object with the specified name
    - get_clustering_parameters(name): return the clustering parameters object with the specified name
    - get_simulation(name): return the simulation object with the specified name
    - import_molecule(path_itp, path_crd, name="", potential_energy=0.0): Import forcefield parameters from .itp file
    - generate_input(self, box=(4., 4., 4.), orthogonalize=False): Generate the coordinate and the topology files
    """

    def __init__(self, name: str, gromacs: str, mdrun_options: str, atomtype: str, pypol_directory: str,
                 path_data: str, path_output: str, path_input: str, intermol: str, lammps: str, initial_crystals,
                 plumed: str, htt_plumed: str):
        """
        The Method object defines the forcefield and the simulations to be used in the analysis.
        Gromacs is used for MD simulations.\n
        :param name: Name used to specify the object and print outputs
        :param gromacs: Gromacs command line
        :param mdrun_options: Options to be added to Gromacs mdrun command.
        :param atomtype: 'atomtype' command line
        :param pypol_directory: PyPol directory with defaults inputs
        :param path_data: data folder in which simulations are performed
        :param path_output:  Output folder in which results are written
        :param path_input:  Input folder to store inputs
        :param intermol:  Path of the 'convert.py' InterMol program
        :param lammps: LAMMPS command line
        :param initial_crystals: list of Crystal objects
        """
        super().__init__(name, gromacs, mdrun_options, atomtype, pypol_directory, path_data, path_output,
                         path_input, intermol, lammps)

        self._package = "Gromacs"

        self._initial_crystals = initial_crystals

        self._molecules = list()
        self._topology = ""
        self._nmolecules = 0

        self._simulations = list()

        self._cvp = list()
        self._plumed = plumed
        self._htt_plumed = htt_plumed
        self._clustering_parameters = list()

    # Read-Only Properties
    @property
    def package(self):
        return self._package

    @property
    def topology(self):
        return self._topology

    @property
    def molecules(self):
        return self._molecules

    @property
    def nmolecules(self):
        return self._nmolecules

    @property
    def crystals(self):
        return self._initial_crystals

    # Private methods
    def __str__(self):
        txt = """
Method Name: {}    
MD package: {}\t({})
Number of Molecules: {} # Only one molecule is accepted for the moment

Molecules:""".format(self._name, self._package, self._gromacs, len(self._molecules))
        for molecule in self._molecules:
            txt += """
Molecule: {0._residue}
Molecule .itp file: {0._forcefield}
Atoms:
{1:8} {2:8} {3:8}{4:>6}  {5:>8}   {6:8}""".format(molecule, "Index", "Label", "Type", "Charge", "Mass", "Bonds")
            for atom in molecule._atoms:
                txt += """
{0._index:<8} {0._label:<8} {0._type:<8}{0._charge:>6.3f}  {0._mass:>8.3f}   {1:<8}""".format(
                    atom, " ".join(str(bond) for bond in atom._bonds))
            txt += "\n"
        return txt

    @staticmethod
    def _merge_atom(mol_atom: Atom, ref_atom: Atom):
        """
        Merge coordinates of the atom with the properties from the forcefield.\n
        :param mol_atom: Atom object
        :param ref_atom: Atom object from forcefield
        :return: Atom object
        """
        new_atom = Atom(ref_atom._label, index=ref_atom._index, ff_type=ref_atom._ff_type, atomtype=ref_atom._type,
                        coordinates=[mol_atom._coordinates[0], mol_atom._coordinates[1], mol_atom._coordinates[2]],
                        element=mol_atom._element, bonds=ref_atom._bonds, charge=ref_atom._charge, mass=ref_atom._mass)
        new_atom._previous_index = mol_atom._index
        return new_atom

    def _graph_v2f_index_search(self, molecule, reference):
        # TODO Not suitable for more than one molecule.
        from networkx.algorithms import isomorphism

        reference._generate_contact_matrix()
        molecule._generate_contact_matrix()

        nodes1, nodes2 = [], []
        for atom in range(len(reference.atoms)):
            nodes1.append((reference.atoms[atom]._index, {"type": reference.atoms[atom]._type}))
            nodes2.append((molecule.atoms[atom]._index, {"type": molecule.atoms[atom]._type}))

        def _create_graph(adjacency_matrix, nodes):
            rows, cols = np.where(adjacency_matrix == 1)
            edges = zip(rows.tolist(), cols.tolist())
            gr = nx.Graph()
            gr.add_nodes_from(nodes)
            gr.add_edges_from(edges)
            return gr

        graph1 = _create_graph(reference.contact_matrix, nodes1)
        graph2 = _create_graph(molecule.contact_matrix, nodes2)

        new_molecule = Molecule(reference._residue, molecule._index)
        # noinspection PyPep8Naming
        GM = isomorphism.GraphMatcher(graph2, graph1, node_match=lambda a, b: a["type"] == b["type"])
        if GM.is_isomorphic():
            atom_map = GM.mapping
            for i_r, i_m in atom_map.items():
                new_atom = self._merge_atom(molecule._atoms[i_r], reference._atoms[i_m])
                new_molecule._atoms.append(new_atom)
            new_molecule._atoms.sort(key=lambda a: a._index)
            new_molecule._natoms = len(new_molecule._atoms)
            if hasattr(molecule, "_attributes") and molecule._attributes:
                new_molecule._attributes = molecule._attributes
            return new_molecule
        else:
            print("An error occurred during the index assignation:\n{}\n{}".format(new_molecule, molecule))

            print("-" * 50)
            for atom in reference.atoms:
                print(atom._type)
            print("-" * 50)
            for atom in molecule.atoms:
                print(atom._type)
            exit()

    def _orthogonalize(self, crystal: Crystal, target_lengths=(60., 60.)):
        """
        Find the most orthogonal, non-primitive cell starting from the CSP-generated cell.
        Cell vector length are limited by the target length parameters defined in the generate_input module.\n
        :param crystal: Target Crystal
        :param target_lengths: Maximum length when searching for the orthogonal cell
        :return:
        """
        from PyPol.utilities import best_b, best_c, translate_molecule

        box = crystal._box
        max_1 = int((target_lengths[0] / 10.) / box[1, 1])
        max_2 = int((target_lengths[1] / 10.) / box[2, 2])
        new_b, replica_b = best_b(box, max_1)
        new_c, replica_c = best_c(box, max_2)
        new_box = np.stack((box[:, 0], new_b, new_c), axis=1)

        if np.array_equal(np.round(new_box, 5), np.round(crystal._box, 5)):
            return crystal

        new_crystal = self._supercell_generator(crystal, replica=(1, replica_b, replica_c))

        new_crystal._box = new_box
        new_crystal._cell_parameters = box2cell(new_box)
        new_molecules = list()
        for molecule in new_crystal._load_coordinates():
            new_molecules.append(translate_molecule(molecule, new_crystal._box))
        new_crystal._save_coordinates(new_molecules)
        return new_crystal

    @staticmethod
    def _generate_masscharge(crystal: Crystal):
        """
        Generates the mass-charge file used by plumed.\n
        :param crystal: Crystal object
        :return:
        """
        os.chdir(crystal._path)
        path_mc = crystal._path + "mc.dat"
        file_mc = open(path_mc, "w")
        file_mc.write("#! FIELDS index mass charge\n")
        for molecule in crystal._load_coordinates():
            for atom in molecule._atoms:
                file_mc.write("{:5}{:19.3f}{:19.3f}\n".format(atom._index + molecule._natoms * molecule._index,
                                                              atom._mass, atom._charge))
        file_mc.close()

    @staticmethod
    def _supercell_generator(crystal: Crystal, box=(0, 0, 0), replica=(1, 1, 1)):
        """
        Replicate the cell in each direction.\n
        :param crystal: Target Crystal object
        :param box: Tuple, target length in the three direction. The number of replicas depends on the box parameters.
        :param replica: Number of replicas in each direction
        :return: Crystal object
        """
        molecules = crystal.molecules

        if box != (0, 0, 0):
            replica_a = int(round(box[0] / (crystal._cell_parameters[0] * 10), 0))
            if replica_a == 0:
                replica_a = 1
            replica_b = int(round(box[1] / (crystal._cell_parameters[1] * 10), 0))
            if replica_b == 0:
                replica_b = 1
            replica_c = int(round(box[2] / (crystal._cell_parameters[2] * 10), 0))
            if replica_c == 0:
                replica_c = 1
        else:
            replica_a, replica_b, replica_c = replica

        molecule_index = crystal._Z
        new_molecules_list = list()
        for a in range(replica_a):
            for b in range(replica_b):
                for c in range(replica_c):
                    if a == 0 and b == 0 and c == 0:
                        continue
                    for molecule in molecules:

                        new_molecule = copy.deepcopy(molecule)
                        new_molecule._index = molecule_index
                        molecule_index += 1

                        for atom in new_molecule._atoms:
                            atom._coordinates = np.sum([a * crystal._box[:, 0], atom._coordinates], axis=0)
                            atom._coordinates = np.sum([b * crystal._box[:, 1], atom._coordinates], axis=0)
                            atom._coordinates = np.sum([c * crystal._box[:, 2], atom._coordinates], axis=0)
                        new_molecule._calculate_centroid()
                        new_molecules_list.append(new_molecule)
        molecules += new_molecules_list
        crystal._cell_parameters = np.array(
            [crystal._cell_parameters[0] * replica_a, crystal._cell_parameters[1] * replica_b,
             crystal._cell_parameters[2] * replica_c, crystal._cell_parameters[3],
             crystal._cell_parameters[4], crystal._cell_parameters[5]])
        crystal._box = cell2box(crystal._cell_parameters)
        crystal._Z = len(molecules)
        crystal._save_coordinates(molecules)
        return crystal

    def _reindex_simulations_after_del(self):
        idx = 0
        if self._simulations:
            for existing_simulation in self._simulations:
                existing_simulation._index = idx
                idx += 1

    def _write_output(self, path_output: str):
        """
        Write main features of the current method to the project output file.
        :param path_output: Output path
        :return:
        """
        # Print Method Details
        file_output = open(path_output, "a")
        file_output.write(self.__str__())

        # Print CV available
        if self._cvp:
            file_output.write("\n   Collective Variables:\n")
            file_output.close()
            for cv in self._cvp:
                cv._write_output(path_output)

        # Print Relative Potential Energy
        file_output = open(path_output, "a")
        file_output.write("\nSimulations:\n{:<20} ".format("IDs"))
        for simulation in self._simulations:
            if simulation._completed and not simulation._hide:
                file_output.write("{:>20} ".format(simulation._name))
        for crystal in self._initial_crystals:
            file_output.write("\n{:20} ".format(crystal._name))
            for simulation in self._simulations:
                if simulation._completed and not simulation._hide:
                    for scrystal in simulation._crystals:
                        # Completed Simulations
                        if scrystal._name == crystal._name and scrystal._state == "complete":
                            file_output.write("{:20.2f} "
                                              "".format(scrystal._energy - simulation._global_minima._energy))
                            break
                        # Melted
                        elif scrystal._name == crystal._name and scrystal._state == "melted":
                            file_output.write("{:>20} ".format(str(scrystal._state)))
                            break
                        # Cluster centers
                        elif scrystal._name == crystal._name and scrystal._state == scrystal._name:
                            file_output.write("{:10.2f} Center "
                                              "".format(scrystal._energy - simulation._global_minima._energy))
                            break
                        # Cluster structures
                        elif scrystal._name == crystal._name and scrystal._state != scrystal._name:
                            file_output.write("{:10.2f} {}".format(scrystal._energy - simulation._global_minima._energy,
                                                                   scrystal._state))
                            break

        file_output.write("\n" + "=" * 100 + "\n")
        file_output.close()

    # Methods
    @staticmethod
    def help():
        print("""
The Method object defines the forcefield and the simulations to be used in the analysis. 
Gromacs is used for MD simulations while LAMMPS is used for the cell relaxation.

Attributes:\n
    - name: Name used to specify the object and print outputs
    - package: The package used for MD. 
    - topology: Path to the Gromacs topology file .top.
    - nmolecules: Number of molecules (or .itp files) in the topology 
    - crystals: list of Crystal objects contained in the method. This refers to the crystals prior to any simulation.
    - gromacs: Gromacs command line
    - mdrun_options: Options to be added to Gromacs mdrun command. For example '-v', '-v -nt 1', '-plumed plumed.dat'.
    - atomtype: 'atomtype' command line
    - intermol: Path of the 'convert.py' InterMol program
    - lammps: LAMMPS command line
    - pypol_directory: PyPol directory with defaults inputs
    - path_data: data folder in which simulations are performed
    - path_output: Output folder in which results are written
    - path_input: Input folder to store inputs

Methods:\n
    - help(): print attributes and methods available
    - new_topology(path_top): Save the [ defaults ] section of a .top file and imports other than .itp files.
    - new_molecule(path_itp, path_crd, potential_energy=0.0): Import forcefield parameters from .itp file.
                Parameters:
                    - path_itp:         Path of the .itp file
                    - path_crd:         Path of the coordinate file used to generate the forcefield. The conformation of
                                        the isolated molecule is not relevant but the atom order MUST be the same one of
                                        the forcefield.
                    - potential_energy: Potential energy of an isolated molecule. This is used to calculate the lattice 
                                        energy.
    - get_molecule(name): return the molecule object with the specified name.
    - generate_input(self, box=(40., 40., 40.), orthogonalize=True): Generate the initial coordinate and the topology 
                files. Parameters: 
                    - box: Supercells are generated so that their box vectors are close in module to the ones specified.
                           Distance values are in Angstrom.
                    - orthogonalize: If True, cells parameters are modified to have a nearly orthogonal cell using the 
                                     box parameter as limits in the search. 
    - new_simulation(name, simtype, path_mdp=None, path_lmp_in=None, path_lmp_ff=None, overwrite=False): Creates a new 
                simulation object of the specified type:
                    - "em":   Energy minimization using Gromacs. If no mdp file is specified, the default one is used.
                    - "cr":   Cell relaxation using LAMMPS. If no input or forcefiled file are specified, a new topology 
                              is obtained converting the Gromacs one with InterMol.
                    - "md":   Molecular Dynamics using Gromacs. If no mdp file is specified the default ones are used.
                              Check the PyPol/data/Defaults/Gromacs folder to see or modify them.
                    - "wtmd": Well-Tempered Metadynamics simulations 
                If no path_mdp, path_lmp_in, path_lmp_ff are given, default input files will be used.
                Use the <simulation>.help() method to obtain details on how to use it.
    - get_simulation(name): return the simulation object with the specified name.
    - del_simulation(name): Remove simulation from project and the related folders. 
    - new_cv(name, cv_type): Generates a new cv of the specified name and type (see the Examples section for more info):
                Available types:
                    - "tor":     Torsional angle.
                    - "mo":      Intermolecular torsional angle.  
                    - "rdf":     Radial Distribution Function.  
                    - "density": Density
                    - "energy":  Potential Energy
                Use the <cv>.help() method to obtain details on how to use it.
    - combine_cvs(name, cvs): Torsions ("tor") and MolecularOrientation ("mo") objects are combined in ND distributions.
    - ggfd(name, cv): Sort crystals in groups according to their similarity in the distribution used or to predefined
                group boundaries.
    - get_cv(name): return the CV parameters object with the specified name.
    - del_cv(name): delete the CV parameters object with the specified name.
    - new_clustering_parameters(name, cvs): Creates a new clustering parameters object. CVs are divided in group and 
                distribution types. Initially, crystals are sorted according to the group they belong. A distance matrix
                is generated and a clustering analysis using the FSFDP algorithm is then performed in each group.
                Use the <clustering>.help() method to obtain details on how to use it.
    - get_clustering_parameters(name): return the clustering parameters object with the specified name.
    - del_clustering_parameters(name): delete the clustering parameters object with the specified name.

Examples: 
- Import Forcefield and generate structures:                                                                            
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.new_method('GAFF')                             # Creates a new method
path_top = '/home/Work/Forcefield/topol.top'                  # Topology file to be used in simulations
path_itp = '/home/Work/Forcefield/MOL.itp'                    # Molecular forcefield file
path_crd = '/home/Work/Forcefield/molecule.mol2'              # Isolated molecule file. 
gaff.new_topology(path_top)                                   # Copy relevant part of the topology to the project folder
gaff.new_molecule(path_itp, path_crd, "MOL", -100.0000)       # Copy molecular forcefield of molecule "MOL"
gaff.generate_input(box=(50., 50., 50.), orthogonalize=True)  # Generates the input files for the simulation
project.save()                                                # Save project to be used later   

- Create a new method and print its manual:                                                                            
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.new_method('GAFF')                             # Creates a new method
gaff.help()                                                   # Print new method manual
project.save()                                                # Save project to be used later       

- Find an existing method, import topology parameters and generate initial structures:                                                                            
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
path_top = '/home/Work/InputFiles/topol.top'                  # Topology file to be used in simulations
path_itp = '/home/Work/InputFiles/MOL.itp'                    # Molecular forcefield file
path_crd = '/home/Work/InputFiles/molecule.mol2'              # Isolated molecule file. 
gaff.new_topology(path_top)                                   # Copy relevant part of the topology to the project folder
gaff.new_molecule(path_itp, path_crd, "MOL", -100.00)         # Copy molecular forcefield of molecule "MOL"
gaff.generate_input(box=(50., 50., 50.), orthogonalize=True)  # Generates the input files for the simulation
project.save()                                                # Save project to be used later  

- Create a new EnergyMinimization object using default mdp files and print its manual:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
em = gaff.new_simulation("em", simtype="em")                  # Create a new simulation using defaults em.mdp file
em.help()                                                     # Print new simulation manual
project.save()                                                # Save project to be used later

- Alternatively, create a new EnergyMinimization object using a specified mdp file:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
path_mdp = '/home/Work/InputFiles/em.mdp'                     # Gromacs mdp file
em = gaff.new_simulation("em", "em", path_mdp)                # Create a new simulation
project.save()                                                # Save project to be used later

- Create a new CellRelaxation object, using InterMol to generate LAMMPS topology:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
cr = gaff.new_simulation("cr", "cr")                          # Create a new simulation
project.save()                                                # Save project to be used later

- Create a new MolecularDynamics objects:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
path_mdp = '/home/Work/InputFiles/nvt.mdp'                    # Gromacs mdp file
nvt = gaff.new_simulation("nvt", "md", path_mdp)              # Create a new simulation
project.save()                                                # Save project to be used later

- Create a new Collective Variable object, modify some of his attributes and generate the plumed inputs: 
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
molecule = gaff.get_molecule("MOL")                           # Use molecular forcefield info for the CV 
rdf = gaff.new_cv("rdf-com", "rdf")                           # Create the RDF object for the radial distribution func
rdf.center = "geometrical"                                    # Use the geometrical center instead of center of mass
rdf.set_atoms("all", molecule)                                # Use all atoms to calculate the molecule geometric center
nvt = gaff.get_simulation("nvt")                              # Retrieve a completed simulation
rdf.generate_input(nvt)                                       # Generate plumed driver input for the selected simulation
project.save()                                                # Save project to be used later

- Retrieve an existing CV and check if plumed drive analysis is completed for all crystals:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
rdf = gaff.get_cv("rdf-com")                                  # Retrieve the RDF object
nvt = gaff.get_simulation("nvt")                              # Retrieve a completed simulation
rdf.get_results(nvt)                                          # Check and import resulting distributions
project.save()                                                # Save project to be used later

- Create two Torsions CV and combine them in 2D distributions:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
molecule = gaff.get_molecule(0)                               # Use molecular forcefield info for the CV
tor1 = gaff.new_cv("tor1", "tor")                             # Create a new Torsions object
tor1.set_atoms((7, 12, 3, 6), molecule)                       # Define the atoms for the torsional angle
tor2 = gaff.new_cv("tor2", "tor")                             # Create a new Torsions object
tor2.set_atoms((4, 8, 16, 5), molecule)                       # Define the atoms for the torsional angle
tor = gaff.combine_cv("2d-tor", (tor1, tor2))                 # Combine two CV
nvt = gaff.get_simulation("nvt")                              # Retrieve a completed simulation          
tor.generate_input(nvt)                                       # Generate plumed driver input for the selected simulation
project.save()                                                # Save project to be used later

- Retrieve an existing CV and create groups of crystals from it:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
npt = gaff.get_simulation("npt")                              # Retrieve a completed simulation
tor = gaff.get_cv("tor")                                      # Retrieve the CV Object
tor.get_results(npt)                                          # Import plumed output and check normal termination
conf = gaff.ggfd("conf", tor)                                 # Create the GGFD object
conf.grouping_method = "similarity"                           # Use the similarity grouping method
conf.integration_type = "simps"                               # Use the simps method to calculate the hellonger distance
conf.run(tor)                                                 # Generates groups
project.save() 


""")

    def new_molecule(self, path_itp: str, path_crd: str, potential_energy=0.0):
        """
        Define the molecular forcefield. The coordinate file used to generate the force field is necessary to
        identify atom properties, index order and bonds. If it is not a .mol2 file, it is converted to it with
        openbabel. \n
        :param path_itp: Path to the .itp file containing the molecular forcefield.
        :param path_crd: Path of the coordinate file used to generate the forcefield.
        :param potential_energy: Potential energy of an isolated molecule used to calculate the Lattice energy of
        Crystals
        """
        # Atom types that can be switched by antechamber, especially from experimental data. They are considered
        # equivalent only during the index assignation in the generate_input module but not during the simulation.
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

        name = ""
        if not os.path.exists(path_itp):
            print("Error: no file found at: " + path_itp)
            exit()
        elif not os.path.exists(path_crd):
            print("Error: no file found at: " + path_crd)
            exit()
        else:
            file_itp = open(path_itp, "r")
            for line in file_itp:
                if "[ moleculetype ]" in line:
                    line = next(file_itp)
                    while line.lstrip().startswith(";"):
                        line = next(file_itp)
                    name = line.split()[0]
            file_itp.close()
            if path_itp != self._path_input + os.path.basename(path_itp):
                copyfile(path_itp, self._path_input + os.path.basename(path_itp))
            path_itp = self._path_input + os.path.basename(path_itp)

        molecule = Molecule(name, len(self._molecules))
        molecule._forcefield = path_itp

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
            copyfile(working_directory + file_name, self._path_input + f"molecule_{name}.mol2")
            os.system(self._atomtype + " -i " + file_name + " -f mol2 -p gaff -o " + path_file_ac)

        else:
            if path_crd != f"{self._path_input}molecule_{name}.mol2":
                copyfile(path_crd, self._path_input + f"molecule_{name}.mol2")
            path_file_ac = working_directory + "PyPol_Temporary_" + file_name[:-4] + ".ac"
            os.system(self._atomtype + " -i " + file_name + " -f mol2 -p gaff -o " + path_file_ac)

        file_ac = open(path_file_ac)
        for line in file_ac:
            if line.startswith(("ATOM", "HETATM")):
                atom_index = int(line[6:11]) - 1
                atom_label = line[13:17].strip()
                atom_type = line.split()[-1]
                atom_coordinates = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                if atom_type in equivalent_atom_types:
                    atom_type = equivalent_atom_types[atom_type]
                atom_element = ''.join([i for i in atom_label if not i.isdigit()])  # TODO Assignation based on label?
                molecule.atoms.append(Atom(atom_label, index=atom_index, ff_type=atom_type, atomtype=atom_type,
                                           bonds=[], coordinates=atom_coordinates, element=atom_element))
            elif line.startswith("BOND"):
                a1 = int(line.split()[2]) - 1
                a2 = int(line.split()[3]) - 1
                for atom in molecule.atoms:
                    if atom._index == a1:
                        atom._bonds.append(a2)
                    elif atom._index == a2:
                        atom._bonds.append(a1)
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
                    if atom._index == atom_index:
                        atom._charge = atom_charge
                        atom._mass = atom_mass
        file_itp.close()
        molecule._potential_energy = potential_energy
        molecule._natoms = len(molecule.atoms)
        self._molecules.append(molecule)

    def get_molecule(self, mol_name):
        """
        Retrieve the Molecule object imported from the molecular forcefield (.itp file)
        :param mol_name: label of the molecule
        :return:
        """
        for molecule in self._molecules:
            if mol_name == molecule.residue:
                return molecule
        print("Molecule name not found, choose between: ")
        for molecule in self._molecules:
            print(molecule.residue)

    def del_molecule(self, mol_name):
        for molecule in self._molecules:
            if mol_name == molecule.residue:
                self._molecules.remove(molecule)
        print("Molecule name not found, choose between: ")
        for molecule in self._molecules:
            print(molecule.residue)

    def new_topology(self, path_top: str):
        """
        TODO Not suitable for more than 1 molecule!
        Add the topology file to the project. Only the [ defaults ] section is included. [ system ] and [ molecules ]
        section will be added to the topology file of each crystal.
        :param path_top:
        :return:
        """
        if os.path.exists(path_top):
            file_top = open(path_top, "r")
            file_top_new = open(self._path_input + os.path.basename(path_top), "w")
            file_top_new.write("; Topology from file {}, commenting everything except Default section"
                               " and imports other than .itp files\n".format(path_top))
            for line in file_top:
                line = line.lstrip()
                if line.startswith(";") or not line:
                    continue
                elif line.startswith("[ defaults ]"):
                    file_top_new.write(line)
                    line = next(file_top).lstrip()
                    while line.startswith(";"):
                        file_top_new.write(line)
                        line = next(file_top).lstrip()
                    file_top_new.write(line)
                elif line.startswith("#include") and not line.rstrip().endswith(('.itp"', ".itp'", ".itp")):
                    file_top_new.write(line)
                else:
                    file_top_new.write("; " + line)
            file_top.close()
            file_top_new.close()
            self._topology = self._path_input + os.path.basename(path_top)

    def del_topology(self):
        self._topology = ""

    def generate_input(self, box=(4., 4., 4.), orthogonalize=True):
        """
        TODO Not suitable for more than 1 molecule!
        Generate the coordinate and the topology files to be used for energy minimization simulations.

        :param box: Target length in the three direction. The number of replicas depends on the Crystal box parameters.
        :param orthogonalize: Find the most orthogonal primitive cell
        :return:
        """

        print("Generating inputs for {}".format(self._name))
        print("-" * 100)
        new_crystal_list = list()
        crystal_index = 0
        for crystal in self._initial_crystals:
            new_crystal_list.append(self._add_crystal(crystal, crystal_index, box=box, orthogonalize=orthogonalize))
            crystal_index += 1
        self._initial_crystals = new_crystal_list

    def update_crystal_list(self, new_list_crystals, box=(4., 4., 4.), orthogonalize=True):
        """
        Add structures to existing method. If a structure is already present in the initial set of structures, it will
        not be added. If simulations objects are available, all their state will be set to "incomplete" and inputs
        for the new structures generated.
        :param new_list_crystals:
        :param box: Target length in the three direction. The number of replicas depends on the Crystal box parameters.
        :param orthogonalize: Find the most orthogonal cell
        :return:
        """
        existing_crystals = [c._name for c in self._initial_crystals]
        list_crystals = [c for c in new_list_crystals if c._name not in existing_crystals]
        crystal_index = len(self._initial_crystals)
        if not list_crystals:
            print("Error: no new structures to add.")
            exit()
        if self._simulations:
            print("Simulations state will be set to 'incomplete' and inputs for the new structures generated")
        for crystal in list_crystals:
            new_crystal = self._add_crystal(crystal, crystal_index, box=box, orthogonalize=orthogonalize)
            self._initial_crystals.append(new_crystal)
            for simulation in self._simulations:
                simulation._crystals.append(Crystal._copy_properties(new_crystal))
        list_crystals_names = [c.name for c in list_crystals]
        for simulation in self._simulations:
            simulation._completed = False
            if simulation._type == "Cell Relaxation" and simulation._sim_index != 0:
                print("Inputs for 'Cell Relaxation' cannot be generated in advance. \n"
                      "Once previous simulation are terminated, use the CellRelaxation.genereate_input module with the "
                      "following crystals list:\ncrystals = [{}".format(list_crystals_names[0]), end="")
                for i in list_crystals_names[1:]:
                    print(", '" + i + "'", end="")
                print("]")
                continue
            simulation.generate_input(bash_script=True, crystals=list_crystals_names)

    def _add_crystal(self, crystal, crystal_index, box=(4., 4., 4.), orthogonalize=True):
        print(crystal._name)
        new_molecules = list()
        print("Index check...", end="")
        for molecule in crystal._load_coordinates():
            new_molecule = self._graph_v2f_index_search(molecule, self._molecules[0])
            new_molecules.append(new_molecule)

        crystal._nmoleculestypes = np.full((len(self._molecules)), 0)
        for molecule_i in self._molecules:
            for molecule_j in new_molecules:
                if molecule_i._residue == molecule_j.residue:
                    crystal._nmoleculestypes[molecule_i._index] += 1
        crystal._Z = len(new_molecules)

        crystal._index = crystal_index

        crystal._path = self._path_data + crystal._name + "/"
        create(crystal._path, arg_type="dir", backup=True)
        crystal._save_coordinates(new_molecules)

        crystal._save_pdb(crystal._path + "pc.pdb")
        crystal._save_gro(crystal._path + "pc.gro")
        if orthogonalize:
            print("done\nOrthogonalize...", end="")
            crystal = self._orthogonalize(crystal, (box[1], box[2]))
        print("done\nSupercell...", end="")
        crystal = self._supercell_generator(crystal, box)
        crystal._save_pdb(crystal._path + "sc.pdb")
        crystal._save_gro(crystal._path + "sc.gro")
        self._generate_masscharge(crystal)

        print("done\nImport topology...", end="")
        for molecule in self._molecules:
            copyfile(molecule._forcefield, crystal._path + os.path.basename(molecule._forcefield))
        if not self._topology:
            self.new_topology(os.path.dirname(self._pypol_directory[:-1]) + "/data/Defaults/Gromacs/topol.top")
        copyfile(self._topology, crystal._path + os.path.basename(self._topology))
        file_top = open(crystal._path + os.path.basename(self._topology), "a")
        for molecule in self._molecules:
            file_top.write('#include "{}"\n'.format(os.path.basename(molecule._forcefield)))
        file_top.write("\n[ system ]\n"
                       "Crystal{}\n"
                       "\n[ molecules ]\n"
                       "; Compound    nmols\n".format(crystal._index))
        for molecule in self._molecules:
            file_top.write('  {:3}         {}\n'.format(molecule._residue, crystal._Z))
        file_top.close()
        print("done", end="\n")
        print("-" * 100)
        return crystal

    def new_simulation(self, name: str, simtype: str, path_mdp=None, path_lmp_in=None, path_lmp_ff=None,
                       crystals="all", catt=None):
        """
        Creates a new simulation object of the specified type:
            - "em":   Energy minimization using Gromacs. Use name="em" or name="relax" without specifying the path_mdp
                      variable to use the defaul mdp file.
            - "cr":   Cell relaxation using LAMMPS. If no input or forcefiled file are specified, a new topology
                      is obtained converting the Gromacs one with InterMol.
            - "md":   Molecular Dynamics using Gromacs. Use names "nvt", "md", "mdvvnvt", "mdvvberendsen",
                      "mdvvparrinello", "mdvvmd" without specifying path_mdp to use the default ones.
                      Check the PyPol/data/Defaults/Gromacs folder to see or modify them.
            - "wtmd": Well-Tempered Metadynamics simulations
        If no path_mdp, path_lmp_in, path_lmp_ff are given, default input files will be used.
        Use the <simulation>.help() method to obtain details on how to use it.
        :param name: Label of the new simulation object.
        :param simtype: Specify which simulation object to use.
        :param path_mdp: Path to the gromacs mdp file to be used in the
        :param path_lmp_in: Path to the LAMMPS input file
        :param path_lmp_ff: Path to the LAMMPS topology file
        :param crystals: It can be either "all", use all non-melted Crystal objects from the previous simulation or
                         "centers", use only cluster centers from the previous simulation. Alternatively, you can select
                          a specific subset of crystals by listing crystal names.
        :param catt: (dict) Specify The custom attributes the crystal must have in to be added to the next simulation.
                     It must be in the form of a python dict, menaning catt={"AttributeLabel": "AttributeValue"}
        :return: Simulation object (EnergyMinimization, CellRelaxation, MolecularDynamics, Metadynamics)
        """
        if simtype.lower() in ("energy minimisation", "em", "cell relaxation", "cr"):
            if simtype.lower() in ("energy minimisation", "em"):

                if (name == "em" or name == "relax") and not path_mdp:
                    path_gromacs_data = os.path.dirname(self._pypol_directory[:-1]) + "/data/Defaults/Gromacs/"
                    print("Default file {} will be used".format(path_gromacs_data + "em.mdp"))
                    path_mdp = path_gromacs_data + "em.mdp"

                simulation = EnergyMinimization(name=name,
                                                gromacs=self._gromacs,
                                                mdrun_options=self._mdrun_options,
                                                atomtype=self._atomtype,
                                                pypol_directory=self._pypol_directory,
                                                path_data=self._path_data,
                                                path_output=self._path_output + name + "/",
                                                path_input=self._path_input,
                                                intermol=self._intermol,
                                                lammps=self._lammps,
                                                crystals=list(),
                                                path_mdp=path_mdp,
                                                molecules=self._molecules,
                                                index=-1,
                                                previous_sim="",
                                                hide=False)
            else:
                # TODO Check if CellRelaxation needs an Energy minimization step before
                # if (not self._simulations and path_lmp_in is None) or (not self._simulations and path_lmp_ff is None):
                #     print("Error: You must specify LAMMPS inputs")
                simulation = CellRelaxation(name=name,
                                            gromacs=self._gromacs,
                                            mdrun_options=self._mdrun_options,
                                            atomtype=self._atomtype,
                                            pypol_directory=self._pypol_directory,
                                            path_data=self._path_data,
                                            path_output=self._path_output + name + "/",
                                            path_input=self._path_input,
                                            intermol=self._intermol,
                                            lammps=self._lammps,
                                            crystals=list(),
                                            path_mdp="",
                                            molecules=self._molecules,
                                            index=-1,
                                            previous_sim="",
                                            hide=False,
                                            topology=self._topology,
                                            path_lmp_in=path_lmp_in,
                                            path_lmp_ff=path_lmp_ff)

            if not self._simulations:
                simulation._index = 0
                simulation._previous_sim = "sc"
                for crystal in self._initial_crystals:
                    simulation_crystal = Crystal._copy_properties(crystal)
                    simulation._crystals.append(simulation_crystal)

            else:
                for previous_simulation in self._simulations:
                    if previous_simulation._name == simulation._name:
                        print("Error: Simulation with name {} already present.".format(simulation._name))
                        exit()
                simulation._previous_sim = self._simulations[-1]._name
                simulation._sim_index = len(self._simulations)
                list_crystals = get_list_crystals(self._simulations[-1]._crystals, crystals, catt)
                for crystal in list_crystals:
                    simulation._crystals.append(Crystal._copy_properties(crystal))

            if simulation._type == "Energy Minimisation":
                if simulation._path_mdp != self._path_input + simulation._name + ".mdp":
                    copyfile(simulation._path_mdp, self._path_input + simulation._name + ".mdp")
                simulation._path_mdp = self._path_input + simulation._name + ".mdp"

            elif simulation._type == "Cell Relaxation":
                if simulation._path_lmp_in:
                    copyfile(simulation._path_lmp_in, self._path_input + "input.in")
                    simulation._path_lmp_in = self._path_input + "input.in"
                    copyfile(path_lmp_ff,
                             self._path_input + self._molecules[0]._name + ".lmp")  # Iter here
                    simulation._path_lmp_ff = self._path_input + self._molecules[0]._name + ".lmp"
                simulation._lammps = self._lammps
                simulation._intermol = self._intermol

            os.mkdir(simulation._path_output)

            self._simulations.append(simulation)
            return simulation

        elif simtype.lower() in ("molecular dynamics", "md"):
            if path_mdp is None:
                path_gromacs_data = os.path.dirname(self._pypol_directory[:-1]) + "/data/Defaults/Gromacs/"
                if name == "nvt" and not path_mdp:
                    print("Default file {} will be used".format(path_gromacs_data + "nvt.mdp"))
                    copyfile(path_gromacs_data + "nvt.mdp", self._path_input + "nvt.mdp")
                    path_mdp = self._path_input + "nvt.mdp"
                elif name == "md" and not path_mdp:
                    print("Default file {} will be used".format(path_gromacs_data + "md.mdp"))
                    copyfile(path_gromacs_data + "md.mdp", self._path_input + "md.mdp")
                    path_mdp = self._path_input + "md.mdp"
                elif name == "mdvvnvt" and not path_mdp:
                    print("Default file {} will be used".format(path_gromacs_data + "mdvvnvt.mdp"))
                    copyfile(path_gromacs_data + "mdvvnvt.mdp", self._path_input + "mdvvnvt.mdp")
                    path_mdp = self._path_input + "mdvvnvt.mdp"
                elif name == "mdvvberendsen" and not path_mdp:
                    print("Default file {} will be used".format(path_gromacs_data + "mdvvberendsen.mdp"))
                    copyfile(path_gromacs_data + "mdvvberendsen.mdp", self._path_input + "mdvvberendsen.mdp")
                    path_mdp = self._path_input + "mdvvberendsen.mdp"
                elif name == "mdvvparrinello" and not path_mdp:
                    print("Default file {} will be used".format(path_gromacs_data + "mdvvparrinello.mdp"))
                    copyfile(path_gromacs_data + "mdvvparrinello.mdp", self._path_input + "mdvvparrinello.mdp")
                    path_mdp = self._path_input + "mdvvparrinello.mdp"
                elif name == "mdvvmd" and not path_mdp:
                    print("Default file {} will be used".format(path_gromacs_data + "mdvvmd.mdp"))
                    copyfile(path_gromacs_data + "mdvvmd.mdp", self._path_input + "mdvvmd.mdp")
                    path_mdp = self._path_input + "mdvvmd.mdp"
                else:
                    print("Error: No mdp file has been found.\n"
                          "You can use the defaults mdp parameters by using the names "
                          "'nvt', 'md', 'mdvvnvt', 'mdvvberendsen', 'mdvvparrinello', 'mdvvmd'\n"
                          "You can check the relative mdp files in folder: {}"
                          "".format(path_gromacs_data))
                    exit()
            else:
                if os.path.exists(path_mdp):
                    copyfile(path_mdp, self._path_input + name + ".mdp")
                    path_mdp = self._path_input + name + ".mdp"
                else:
                    print("Error: No mdp file has been found.\n"
                          "You can use the defaults mdp parameters by using the names "
                          "'nvt', 'md', 'mdvvnvt', 'mdvvberendsen', 'mdvvparrinello', 'mdvvmd'\n"
                          "You can check the relative mdp files in folder: {}"
                          "".format(os.path.dirname(self._pypol_directory) + "/data/Defaults/Gromacs/"))
                    exit()
            simulation = MolecularDynamics(name=name,
                                           gromacs=self._gromacs,
                                           mdrun_options=self._mdrun_options,
                                           atomtype=self._atomtype,
                                           pypol_directory=self._pypol_directory,
                                           path_data=self._path_data,
                                           path_output=self._path_output + name + "/",
                                           path_input=self._path_input,
                                           intermol=self._intermol,
                                           lammps=self._lammps,
                                           crystals=list(),
                                           path_mdp=path_mdp,
                                           molecules=self._molecules,
                                           index=-1,
                                           previous_sim="",
                                           hide=False)

            if not self._simulations:
                simulation._sim_index = 0
                simulation._previous_sim = "sc"
                for crystal in self._initial_crystals:
                    simulation_crystal = Crystal._copy_properties(crystal)
                    simulation_crystal.cvs = dict()
                    simulation._crystals.append(simulation_crystal)
            else:
                for previous_simulation in self._simulations:
                    if previous_simulation._name == simulation._name:
                        print("Error: Simulation with name {} already present.".format(simulation._name))
                        exit()
                simulation._previous_sim = self._simulations[-1]._name
                simulation._sim_index = len(self._simulations)

                list_crystals = get_list_crystals(self._simulations[-1]._crystals, crystals, catt)
                for crystal in list_crystals:
                    simulation._crystals.append(Crystal._copy_properties(crystal))

            simulation._gromacs = self._gromacs
            simulation._mdrun_options = self._mdrun_options
            simulation._path_data = self._path_data
            simulation._path_output = self._path_output + name + "/"
            simulation._path_input = self._path_input

            os.mkdir(simulation._path_output)

            self._simulations.append(simulation)
            return simulation

        elif simtype.lower() in ("metadynamics", "wtmd"):
            if path_mdp is None:
                path_gromacs_data = os.path.dirname(self._pypol_directory[:-1]) + "/data/Defaults/Gromacs/"
                print("Default file {} will be used".format(path_gromacs_data + "wtmd.mdp"))
                copyfile(path_gromacs_data + "wtmd.mdp", self._path_input + name + ".mdp")
                path_mdp = self._path_input + name + ".mdp"
            else:
                if os.path.exists(path_mdp):
                    copyfile(path_mdp, self._path_input + name + ".mdp")
                    path_mdp = self._path_input + name + ".mdp"
                else:
                    print("Error: No mdp file has been found.\n"
                          "If no mdp is specified, the default mdp parameters for WTMD are used."
                          "You can check and modify the wtmd.mdp file in folder: {}"
                          "".format(os.path.dirname(self._pypol_directory) + "/data/Defaults/Gromacs/"))
                    exit()

            simulation = Metadynamics(name=name,
                                      gromacs=self._gromacs,
                                      mdrun_options=self._mdrun_options,
                                      atomtype=self._atomtype,
                                      pypol_directory=self._pypol_directory,
                                      path_data=self._path_data,
                                      path_output=self._path_output + name + "/",
                                      path_input=self._path_input,
                                      intermol=self._intermol,
                                      lammps=self._lammps,
                                      crystals=list(),
                                      path_mdp=path_mdp,
                                      molecules=self._molecules,
                                      index=-1,
                                      previous_sim="",
                                      hide=False)
            if not self._simulations:
                print("Error: Equilibration is needed before performing Metadynamics simulations.")
                exit()

            for previous_simulation in self._simulations:
                if previous_simulation._name == simulation._name:
                    print("Error: Simulation with name {} already present.".format(simulation._name))
                    exit()
            simulation._previous_sim = self._simulations[-1]._name
            simulation._sim_index = len(self._simulations)

            list_crystals = get_list_crystals(self._simulations[-1]._crystals, crystals, catt)
            for crystal in list_crystals:
                new_crystal = Crystal._copy_properties(crystal)
                new_crystal._box = crystal._box
                new_crystal._cell_parameters = crystal._cell_parameters
                new_crystal._volume = crystal._volume
                new_crystal._energy = crystal._energy
                simulation._crystals.append(new_crystal)

            default_cvs_1 = False
            default_cvs_2 = False
            default_cvs = None
            while default_cvs not in ("1", "2", "0"):
                default_cvs = input("Default Collective Variables:\n"
                                    "1) Potential Energy and Density\n"
                                    "2) Simulation Box Angles\n"
                                    "0) Custom\n")
                if default_cvs == "1":
                    default_cvs_1 = True
                    break
                elif default_cvs == "2":
                    default_cvs_2 = True
                    break

            if default_cvs_1:
                def gen_rho():
                    rho_min = list_crystals[0].density
                    rho_max = list_crystals[0].density
                    for ecrystal in list_crystals:
                        rho_cry = ecrystal.density
                        if rho_max < rho_cry:
                            rho_max = rho_cry
                        if rho_min > rho_cry:
                            rho_min = rho_cry
                    rho_min = round(rho_min, 0) - 300.
                    rho_max = round(rho_max, 0) + 300.
                    rho_bin = int(rho_max - rho_min)
                    orho = self.new_cv("density", "density")
                    orho._grid_min = rho_min
                    orho._grid_max = rho_max
                    orho._grid_bins = rho_bin
                    orho.use_walls = True
                    return orho

                def gen_energy():
                    energy_min = list_crystals[0]._energy
                    energy_max = list_crystals[0]._energy
                    for ecrystal in list_crystals:
                        energy_cry = ecrystal._energy
                        if energy_max < energy_cry:
                            energy_max = energy_cry
                        if energy_min > energy_cry:
                            energy_min = energy_cry
                    energy_min = round(energy_min, 0) - 100.
                    energy_max = round(energy_min, 0) + 500.
                    energy_bin = int(energy_max - energy_min) * 10
                    oenergy = self.new_cv("energy", "energy")
                    oenergy._grid_min = energy_min
                    oenergy._grid_max = energy_max
                    oenergy._grid_bins = energy_bin
                    return oenergy

                list_cv = [ecv._name for ecv in self._cvp]
                if "density" in list_cv:
                    dcv = input("CV called 'density' already present in the CVs set.\n"
                                "Do you want to use it (if not, it will be overwritten with a new one)? [y/n] ")
                    if dcv == "n":
                        self.del_cv("density")
                        rho = gen_rho()
                    else:
                        rho = self.get_cv("density")
                else:
                    rho = gen_rho()

                if "energy" in list_cv:
                    dcv = input("CV called 'energy' already present in the CVs set.\n"
                                "Do you want to use it (if not, it will be overwritten with a new one)? [y/n] ")
                    if dcv == "n":
                        self.del_cv("energy")
                        energy = gen_energy()
                    else:
                        energy = self.get_cv("energy")
                else:
                    energy = gen_energy()

                if "ASB" in list_cv:
                    dcv = input("CV called 'ASB' already present in the CVs set.\n"
                                "Do you want to use it (if not, it will be overwritten with a new one)? [y/n] ")
                    if dcv == "n":
                        self.del_cv("ASB")
                        asb = self.new_cv("ASB", "asb")
                    else:
                        asb = self.get_cv("ASB")
                else:
                    asb = self.new_cv("ASB", "asb")

                simulation.set_cvs(asb, rho, energy)

            elif default_cvs_2:
                def gen_box_angle():
                    box = list_crystals[0]._box
                    bp_max = np.max(np.absolute([box[i, j] for i in range(3) for j in range(3) if i != j]))
                    for ecrystal in list_crystals[1:]:
                        box = ecrystal._box
                        bp = np.max(np.absolute([box[i, j] for i in range(3) for j in range(3) if i != j]))
                        if bp_max < bp:
                            bp_max = bp
                    if bp_max < 4.:
                        bp_max = 4.
                    bp_min = -round(bp_max, 0) - 2.
                    bp_max = round(bp_max, 0) + 2.
                    bp_bins = int((bp_max - bp_min) / 0.05)
                    bx = self.new_cv("bx", "box")
                    bx._parameter = "bx"
                    bx._grid_min = bp_min
                    bx._grid_max = bp_max
                    bx._grid_bins = bp_bins
                    bx.use_walls = True
                    cx = self.new_cv("cx", "box")
                    cx._parameter = "cx"
                    cx._grid_min = bp_min
                    cx._grid_max = bp_max
                    cx._grid_bins = bp_bins
                    cx.use_walls = True
                    cy = self.new_cv("cy", "box")
                    cy._parameter = "cy"
                    cy._grid_min = bp_min
                    cy._grid_max = bp_max
                    cy._grid_bins = bp_bins
                    cy.use_walls = True
                    return bx, cx, cy

                list_cv = [ecv._name for ecv in self._cvp]
                if "bx" in list_cv or "cx" in list_cv or "cy" in list_cv:
                    dcv = input("CVs called 'bx', 'cx' or 'cy' already present in the CVs set.\n"
                                "Do you want to use it (if not, it will be overwritten with a new one)? [y/n] ")
                    if dcv == "n":
                        self.del_cv("bx")
                        self.del_cv("cx")
                        self.del_cv("cy")
                        bx, cx, cy = gen_box_angle()
                    else:
                        bx = self.get_cv("bx")
                        cx = self.get_cv("cx")
                        cy = self.get_cv("cy")
                else:
                    bx, cx, cy = gen_box_angle()

                if "ASB" in list_cv:
                    dcv = input("CV called 'ASB' already present in the CVs set.\n"
                                "Do you want to use it (if not, it will be overwritten with a new one)? [y/n] ")
                    if dcv == "n":
                        self.del_cv("ASB")
                        asb = self.new_cv("ASB", "asb")
                    else:
                        asb = self.get_cv("ASB")
                else:
                    asb = self.new_cv("ASB", "asb")
                simulation.set_cvs(asb, bx, cx, cy)
            else:
                print("Include custom CV using the command simulation.set_cvs(cv1, cv2, cv3, ...). ")

            simulation._gromacs = self._gromacs
            simulation._mdrun_options = self._mdrun_options
            simulation._path_data = self._path_data
            simulation._path_output = self._path_output + name + "/"
            simulation._path_input = self._path_input

            os.mkdir(simulation._path_output)

            self._simulations.append(simulation)
            return simulation

        else:
            print("""
Simulation Type '{}' not recognized. Choose between:
- "em":   Energy minimization using Gromacs. If no mdp file is specified, the default one is used.
- "cr":   Cell relaxation using LAMMPS. If no input or forcefiled file are specified, a new topology 
          is obtained converting the Gromacs one with InterMol.
- "md":   Molecular Dynamics using Gromacs. If no mdp file is specified the default ones are used.
          Check the PyPol/data/Defaults/Gromacs folder to see or modify them.
- "wtmd": Well-Tempered Metadynamics simulations""".format(simtype))
            exit()

    def get_simulation(self, simulation_name: str):
        """
        Find an existing simulation by its name.\n
        :param simulation_name: Name assigned to the simulation
        :return:
        """
        if self._simulations:
            for existing_simulation in self._simulations:
                if existing_simulation._name == simulation_name:
                    return existing_simulation
        print("No method found with name {}".format(simulation_name))

    def del_simulation(self, simulation_name: str):
        """
        Delete an existing simulation by its name.\n
        :param simulation_name: Name assigned to the simulation
        :return:
        """
        gromacs_file_formats = ("gro", "cpt", "g96", "pdb", "tpr", "tpa", "tpb", "tng", "trj", "trr",
                                "xtc", "ene", "edr", "log", "out", "edi", "edo", "mdp")
        if self._simulations:
            for existing_simulation in self._simulations:
                if existing_simulation._name == simulation_name and simulation_name == self._simulations[-1]._name:
                    rm = input("Simulation {} to be deleted from project. "
                               "Do you want to remove all associated files? [y/n]".format(existing_simulation._name))
                    if rm == "y":
                        for crystal in existing_simulation._crystals:
                            for ext in gromacs_file_formats:
                                if os.path.exists(crystal._path + existing_simulation._name + "." + ext):
                                    os.remove(crystal._path + existing_simulation._name + "." + ext)
                    self._simulations.remove(existing_simulation)
                    return
                elif existing_simulation._name == simulation_name:
                    rm_sim = input("Simulation {} is not the last one in the project. "
                                   "Deleting it could cause problems in the following simulations."
                                   "Are you sure? [y/n] ".format(existing_simulation._name))
                    if rm_sim == "y":
                        rm = input("Simulation {} to be deleted from project. "
                                   "Do you want to remove all associated files? [y/n]"
                                   "".format(existing_simulation._name))
                        if rm == "y":
                            for crystal in existing_simulation._crystals:
                                for ext in gromacs_file_formats:
                                    if os.path.exists(crystal._path + existing_simulation._name + "." + ext):
                                        os.remove(crystal._path + existing_simulation._name + "." + ext)
                        self._simulations.remove(existing_simulation)
                        self._reindex_simulations_after_del()
                    return
        print("No method found with name {}".format(simulation_name))

    def new_cv(self, name, cv_type):
        """
        Add a new Distribution Object or Collective Variable Object to the CV's list.
        Available Distribution types:
                    - "tor":        Torsional angle.
                    - "mo":         Intermolecular torsional angle.
                    - "planes":     Intermolecular torsional angle between planes (useful for planar molecules).
                    - "rdf":        Radial Distribution Function.
                    - "rdf-mo": 2D distribution of the RDF and molecular orientation.
                    - "rdf-planes": 2D distribution of the RDF and molecular orientation.
        Available Collective Variables:
                    - "density": Density of the crystal.
                    - "energy":  Potential Energy of the crystal.
        Use the <cv>.help() method to obtain details on how to use it.
        You can also add the AvoidScrewedBox object to the collective variables in order to avoid too tilted boxes.
        This is done by typing:
        asb= <method_name>.new_cv("asb", cv_type="asb")
        asb.generate_input(<simulation_name>)

        :param name: Object label
        :param cv_type: Specify which CV object to use
        :return:
        """

        for cv in self._cvp:
            if cv._name == name:
                print("Error: CV with label {} already present in this method. Remove it or change CV label"
                      "".format(name))
                exit()
        from inspect import getmembers, isclass
        cv_type = cv_type.lower()

        # Import distributions
        from PyPol import fingerprints
        for _, fingerprint in getmembers(fingerprints, isclass):
            if hasattr(fingerprint, "_short_type") and fingerprint._short_type == cv_type:
                if fingerprint._plumed_version == "hack-the-tree":
                    cv = fingerprint(name, self._htt_plumed)
                elif fingerprint._plumed_version == "master":
                    cv = fingerprint(name, self._plumed)
                else:
                    cv = fingerprint(name)
                self._cvp.append(cv)
                return cv

        from PyPol import metad
        for _, colvar in getmembers(metad, isclass):
            if hasattr(colvar, "_short_type") and colvar._short_type == cv_type:
                cv = colvar(name)
                self._cvp.append(cv)
                return cv

        from PyPol import walls
        for _, wall in getmembers(walls, isclass):
            if hasattr(wall, "_short_type") and wall._short_type == cv_type:
                cv = wall(name)
                self._cvp.append(cv)
                return cv

        print("Collective Variable Type '{}' not available.".format(cv_type))
        exit()

    def combine_cvs(self, name, cvs: Union[tuple, list]):
        """
        Multiple 1-D Torsions ("tor") or MolecularOrientation ("mo") objects are combined in ND distributions.
        :param name: Object label
        :param cvs: Tuple or list of 1-D Distributions (all from the same class) to combine in ND distribution.
        :return: Combine object
        """
        for cv in self._cvp:
            if cv._name == name:
                print("Error: CV with label {} already present in this method. Remove it or change CV label"
                      "".format(name))
                exit()
        from PyPol.fingerprints import Combine
        if all(cv.type == cvs[0].type for cv in cvs) and cvs[0].type in ("Torsional Angle", "Molecular Orientation"):
            cv = Combine(name, cvs=cvs)
            self._cvp.append(cv)
            return cv

    def ggfd(self, name, cv):
        """
        Generate Groups from Distributions. Sort crystals in groups according to their similarity in the distribution
        used or to predefined group boundaries. Use the GGFD.help() module for more detailed information.
        :param name: object label
        :param cv: Distribution to be used to define the different groups
        :return: GGFD object
        """
        from PyPol.groups import GGFD
        for existing_cv in self._cvp:
            if existing_cv._name == name:
                print("Error: CV with label {} already present in this method. Remove it or change CV label"
                      "".format(name))
                exit()
        cv = GGFD(name, cv)
        self._cvp.append(cv)
        return cv

    def ggfa(self, name, attribute):
        """
        Generate Groups from Attributes. Sort crystals in groups according to their attributes.
        Use the GGFA.help() module for more detailed information.
        :param name: object label
        :param attribute: dict with the attribute to use for classification
        :return: GGFA object
        """
        from PyPol.groups import GGFA
        for existing_cv in self._cvp:
            if existing_cv._name == name:
                print("Error: CV with label {} already present in this method. Remove it or change CV label"
                      "".format(name))
                exit()
        cv = GGFA(name, attribute)
        self._cvp.append(cv)
        return cv

    def get_cv(self, cv_name: str):
        """
        Find an existing CV by its name.
        :param cv_name: Name assigned to the collective variable
        :return: Distribution or CollectiveVariable object
        """
        if self._cvp:
            for existing_cv in self._cvp:
                if existing_cv._name == cv_name:
                    return existing_cv
        print("No CV found with name {}".format(cv_name))

    def del_cv(self, cv_name: str):
        """
        Delete an existing Distribution or CollectiveVariable object.
        :param cv_name: object label
        :return:
        """
        if self._cvp:
            for existing_cv in self._cvp:
                if existing_cv._name == cv_name:
                    self._cvp.remove(existing_cv)
                    return
            print("No CV found with name {}".format(cv_name))

    def new_clustering_parameters(self, name: str, cvs: Union[list, tuple]):
        """
        Creates a new clustering parameters object. CVs are divided in group and distribution types.
        Initially, crystals are sorted according to the group they belong. A distance matrix is generated and a
        clustering analysis using the FSFDP algorithm is then performed in each group.
        Use the <clustering>.help() method to obtain details on how to use it.
        :param name: object label
        :param cvs: list or tuple of Distribution/Group object to be used for the clustering
        :return:
        """
        from PyPol.cluster import Clustering
        cvp = list()

        if type(cvs) not in [list, tuple]:
            cvs = [cvs]

        if isinstance(cvs[0], str):
            for cv in self._cvp:
                if cv._name in cvs:
                    cvp.append(cv)
            if len(cvp) != len(cvs):
                print("Error: Not all CVs present in this method. CVs available:")
                for cv in self._cvp:
                    print(cv._name)
        else:
            for cv in self._cvp:
                if cv in cvs:
                    cvp.append(cv)
            if len(cvp) != len(cvs):
                print("Error: Not all CVs present in this method. CVs available:")
                for cv in self._cvp:
                    print(cv._name)

        clustering_method = Clustering(name, tuple(cvp))
        self._clustering_parameters.append(clustering_method)
        return clustering_method

    def get_clustering_parameters(self, clustering_parameter_name: str):
        """
        Find an existing clustering parameters by its name.\n
        :param clustering_parameter_name: Name assigned to the clustering method
        :return:
        """
        if self._clustering_parameters:
            for existing_clustering_parameter in self._clustering_parameters:
                if existing_clustering_parameter._name == clustering_parameter_name:
                    return existing_clustering_parameter
        print("No CV found with name {}".format(clustering_parameter_name))

    def del_clustering_parameters(self, clustering_method: str):
        """
        Delete an existing clustering parameters by.\n
        :param clustering_method: Name assigned to the clustering method
        :return:
        """
        if self._clustering_parameters:
            for existing_cm in self._clustering_parameters:
                if existing_cm._name == clustering_method:
                    self._clustering_parameters.remove(existing_cm)
                    return
            print("No CV found with name {}".format(clustering_method))


class _GroSim(_GroDef):
    """
    Default Properties to be used in the Gromacs simulation classes.

    Attributes:\n
    - name: Name used to specify the object and print outputs
    - gromacs: Gromacs command line
    - mdrun_options: Options to be added to Gromacs mdrun command. For example '-v', '-v -nt 1', '-plumed plumed.dat'.
    - atomtype: 'atomtype' command line
    - pypol_directory: PyPol directory with defaults inputs
    - path_data: data folder in which simulations are performed
    - path_output: Output folder in which results are written
    - path_input: Input folder to store inputs
    - intermol: Path of the 'convert.py' InterMol program
    - lammps: LAMMPS command line
    - type: Simulation type, could be "Energy Minimization", "Cell Relaxation", "Molecular Dynamics", "MetaD"
    - crystals: List of crystals on which simulation will be performed
    - sim_index: index of the simulation, define the position of the simulation in the workflow
    - mdp: Molecular Dynamics Parameters file path
    - completed: True if all crystall are not in the "incomplete" state
    - global_minima: Global minima of the set, available if the simulation is completed
    - hide: show or not the the relative potential energy file in the output file
    """

    def __init__(self, name, gromacs, mdrun_options, atomtype, pypol_directory, path_data, path_output,
                 path_input, intermol, lammps, simtype: str, crystals: list, path_mdp: str, molecules: list,
                 index: int, previous_sim: str, hide=False):
        """
        Create a new Simulation Objects that uses the Gromacs MD package.
        :param name: name of the new Simulation
        """
        super().__init__(name, gromacs, mdrun_options, atomtype, pypol_directory, path_data, path_output,
                         path_input, intermol, lammps)
        self._molecules = molecules

        self._type = simtype
        self._sim_index = index
        self._previous_sim = previous_sim
        if not os.path.exists(path_mdp) and simtype != "Cell Relaxation":
            print("Error: File '{}' not found".format(path_mdp))
            exit()
        self._path_mdp = path_mdp
        self._mdp = {}
        self._crystals = crystals

        self._completed = False
        self._global_minima = None
        self._hide = hide

        # Clustering Parameters
        self._clusters = {}
        self._cluster_data = {}

    @property
    def molecules(self):
        return self._molecules

    @property
    def type(self):
        return self._type

    @property
    def sim_index(self):
        return self._sim_index

    @property
    def path_mdp(self):
        return self._path_mdp

    @property
    def mdp(self):
        if not self._mdp:
            self._mdp = self._import_mdp(self._path_mdp)
        txt = ""
        for k in self._mdp.keys():
            txt += "{} = {}\n".format(k, self._mdp[k])
        return txt

    @property
    def crystals(self):
        return self._crystals

    @property
    def completed(self):
        return self._completed

    @property
    def global_minima(self):
        if self.completed:
            return self._global_minima
        else:
            print("Error: run '<simulation>.get_results()' to identify a global minima")

    @property
    def hide(self):
        return self._hide

    @hide.setter
    def hide(self, hide: bool):
        if hide:
            print("Crystal Relative Potential Energy will be shown in the output file")
        else:
            print("Crystal Relative Potential Energy will not be shown in the output file")
        self._hide = hide

    @staticmethod
    def _import_mdp(path_mdp):
        mdp = {}
        file_mdp = open(path_mdp, "r")
        for line in file_mdp:
            if not line or line.startswith(";"):
                continue
            if ";" in line:
                line = line.split(sep=";")[0]

            if "=" in line:
                line = line.split(sep="=")
                mdp[line[0].strip()] = line[1].strip()
        file_mdp.close()
        return mdp

    def _get_results(self, crystal):
        path_output = crystal._path + self.name + ".log"
        if os.path.exists(path_output):
            file_output = open(path_output)
            lines = file_output.readlines()
            if any("Finished mdrun" in string for string in lines[-30:]):
                file_output.close()
                if not self._mdp:
                    self._mdp = self._import_mdp(self._path_mdp)
                ltf = float(self._mdp["dt"]) * float(self._mdp["nsteps"])
                if ltf < 0:
                    return True
                else:
                    file_edr = sbp.getoutput("{} check -e {}".format(self._gromacs, crystal._path + self.name + ".edr"))
                    for line in file_edr.split(sep="\n"):
                        if "Last energy frame read" in line:
                            if float(line.split()[-1]) == ltf:
                                return True
                            else:
                                return False
            else:
                file_output.close()
                return False
        else:
            return False

    def plot_landscape(self, path, cluster_centers=False, save_data=True, crystals="all", catt=None):
        """
        Plot the crystal energy landscape resulting from a completed simulation.
        :param path: Output path of the plot image
        :param cluster_centers: Use only cluster centers and scale their mark size by the size of the cluster
        :param save_data: Save a file with the densities and energies of all crystal used
        :param crystals:  You can select a specific subset of crystals by listing crystal names.
        :param catt: Use crystal attributes to select the crystal list
        :return:
        """
        import pandas as pd
        print("=" * 50)
        print("Generating crystal energy landscape:")
        if not self.completed:
            print("Error: Import results before plotting energy landscape.")
            exit()
        list_crystals = get_list_crystals(self._crystals, crystals, catt)

        s = {}
        labels = {}
        for crystal in list_crystals:
            if crystal._name not in labels:
                if crystal._name != crystal._label:
                    print("Include label {} for crystal {}".format(crystal._label, crystal._name))
                    labels[crystal._name] = crystal._label
                else:
                    labels[crystal._name] = False

            if cluster_centers:
                if crystal._state == "complete":
                    print("Error: Perform Cluster analysis before plotting energy landscape of cluster centers")
                    exit()

                if labels[crystal._name]:
                    if crystal._state not in labels and crystal._state != crystal._name:
                        print("Changing label of crystal {} to {}".format(crystal._state, crystal._label))
                        labels[crystal._state] = crystal._label
                        labels[crystal._name] = False
                    elif not labels[crystal._state] and crystal._state != crystal._name:
                        print("Changing label of crystal {} to {}".format(crystal._state, crystal._label))
                        labels[crystal._state] = crystal._label
                        labels[crystal._name] = False

                if crystal._state not in s:
                    s[crystal._state] = 1
                else:
                    s[crystal._state] += 1
            else:
                s[crystal._name] = 1

        list_crystals = sorted(list_crystals, key=lambda x: x.label)

        data = pd.DataFrame(np.full((len(s.keys()), 2), pd.NA), index=list(s.keys()), columns=["Density", "Energy"])
        c = 1
        for crystal in list_crystals:
            if cluster_centers and crystal._name != crystal._state:
                continue
            data.at[crystal._name, "Density"] = crystal.density
            data.at[crystal._name, "Energy"] = crystal._energy - self._global_minima._energy
            if labels[crystal._name]:
                plt.scatter(crystal.density, crystal._energy - self._global_minima._energy,
                            s=s[crystal._name] * 50, c="C" + str(c), alpha=0.8, edgecolors=None,
                            label=labels[crystal._name])
                c += 1
            else:
                plt.scatter(crystal.density, crystal._energy - self._global_minima._energy,
                            s=s[crystal._name] * 50, c="C0", alpha=0.2, edgecolors=None, label='_no_legend_')

        plt.legend(scatterpoints=1)
        plt.ylabel(r"$\Delta$E / kJ mol$^{-1}$")
        plt.xlabel(r"$\rho$ / Kg m$^{-3}$")
        plt.savefig(path, dpi=300)
        plt.close("all")
        if save_data:
            data.to_csv(path + "_data")
        print("Plot '{}' saved".format(path))
        print("=" * 50)


class EnergyMinimization(_GroSim):
    """
    Perform Energy Minimization simulations using Gromacs.

    Attributes:\n
    - name: Name used to specify the object and print outputs
    - gromacs: Gromacs command line
    - mdrun_options: Options to be added to Gromacs mdrun command. For example '-v', '-v -nt 1', '-plumed plumed.dat'.
    - atomtype: 'atomtype' command line
    - pypol_directory: PyPol directory with defaults inputs
    - path_data: data folder in which simulations are performed
    - path_output: Output folder in which results are written
    - path_input: Input folder to store inputs
    - intermol: Path of the 'convert.py' InterMol program
    - lammps: LAMMPS command line
    - type: "Energy Minimization"
    - crystals: List of crystals on which simulation will be performed
    - sim_index: index of the simulation, define the position of the simulation in the workflow
    - mdp: Molecular Dynamics Parameters file path
    - completed: True if all crystall are not in the "incomplete" state
    - global_minima: Global minima of the set, available if the simulation is completed
    - hide: show or not the the relative potential energy file in the output file

    Methods:\n
    - help(): Print attributes and methods
    - generate_input(bash_script=False, crystals="all"): copy the .mdp file in each crystal folder.
    - get_results(crystals="all"): check if simulations ended or a rerun is necessary.
    """

    def __init__(self, name, gromacs, mdrun_options, atomtype, pypol_directory, path_data, path_output,
                 path_input, intermol, lammps, crystals, path_mdp, molecules, index, previous_sim, hide):
        """
        Perform Energy Minimization simulations using Gromacs.
        :param name: Name used to specify the object and print outputs
        :param gromacs: Gromacs command line
        :param mdrun_options: Options to be added to Gromacs mdrun command.
        :param atomtype: 'atomtype' command line
        :param pypol_directory: PyPol directory with defaults inputs
        :param path_data: data folder in which simulations are performed
        :param path_output: Output folder in which results are written
        :param path_input: Input folder to store inputs
        :param intermol: Path of the 'convert.py' InterMol program
        :param lammps: LAMMPS command line
        :param crystals: List of crystals on which simulation will be performed
        :param hide: show or not the the relative potential energy file in the output file
        """

        super().__init__(name=name,
                         gromacs=gromacs,
                         mdrun_options=mdrun_options,
                         atomtype=atomtype,
                         pypol_directory=pypol_directory,
                         path_data=path_data,
                         path_output=path_output,
                         path_input=path_input,
                         intermol=intermol,
                         lammps=lammps,
                         simtype="Energy Minimisation",
                         crystals=crystals,
                         path_mdp=path_mdp,
                         molecules=molecules,
                         index=index,
                         previous_sim=previous_sim,
                         hide=hide)

    @staticmethod
    def help():
        return """
Perform Energy Minimization simulations using Gromacs.

Attributes:
- name: Name used to specify the object and print outputs
- gromacs: Gromacs command line
- mdrun_options: Options to be added to Gromacs mdrun command. For example '-v', '-v -nt 1', '-plumed plumed.dat'.
- atomtype: 'atomtype' command line
- pypol_directory: PyPol directory with defaults inputs
- path_data: data folder in which simulations are performed
- path_output: Output folder in which results are written
- path_input: Input folder to store inputs
- intermol: Path of the 'convert.py' InterMol program
- lammps: LAMMPS command line
- type: "Energy Minimization"
- crystals: List of crystals on which simulation will be performed
- sim_index: index of the simulation, define the position of the simulation in the workflow
- path_mdp: Molecular Dynamics Parameters file path
- mdp: Molecular Dynamics Parameters
- completed: True if all crystals are not in the "incomplete" state
- global_minima: Global minima of the set, available if the simulation is completed
- hide: show or not the the relative potential energy file in the output file

Methods:
- help(): Print attributes and methods
- generate_input(bash_script=False, crystals="all"): copy the .mdp file in each crystal folder.
- get_results(crystals="all"): check if simulations ended or a rerun is necessary.

Examples: 
- Generate inputs for all the crystals, including a bash script to run all simulations:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
em = gaff.get_simulation("em")                                # Retrieve an existing simulation
em.generate_input(bash_script=True)                           # Copy the MDP file in each crystal directory
project.save()                                                # Save project to be used later

- Check the normal termination of each simulation and get the energy landscape:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
em = gaff.get_simulation("em")                                # Retrieve an existing simulation
em.get_results()                                              # Check normal termination and import potential energy
for crystal in em.crystals:                                   # Print the lattice energy of each crystal
    print(crystal.name, crystal.energy)
project.save()                                                # Save project to be used later
"""

    def generate_input(self, bash_script=False, crystals="all"):
        """
        Copy the Gromacs .mdp file to each crystal path.
        :param bash_script: If bash_script=True, a bash script is generated to run all simulations
        :param crystals: You can select a specific subset of crystals by listing crystal names in the crystal parameter
        :return:
        """
        list_crystals = get_list_crystals(self._crystals, crystals)

        for crystal in list_crystals:
            copyfile(self._path_mdp, crystal._path + self._name + ".mdp")

        if bash_script:
            file_script = open(self._path_data + "/run_" + self._name + ".sh", "w")
            file_script.write('#!/bin/bash\n\n'
                              'crystal_paths="\n')
            for crystal in list_crystals:
                file_script.write(crystal._path + "\n")
            file_script.write('"\n\n'
                              'for crystal in $crystal_paths ; do\n'
                              'cd "$crystal" || exit \n'
                              '{0} grompp -f {1}.mdp -c {2}.gro -o {1}.tpr -p topol.top -maxwarn 1 \n'
                              '{0} mdrun {3} -deffnm {1} \n'
                              'done \n'
                              ''.format(self._gromacs, self._name, self._previous_sim, self._mdrun_options))
            file_script.close()

    def get_results(self, crystals="all"):
        """
        Verify if the simulation ended correctly and upload new crystal properties.
        :param crystals: You can select a specific subset of crystals by listing crystal names in the crystal parameter.
                         Alternatively, you can use:
                         - "all": Select all non-melted structures
                         - "incomplete": Select crystals whose simulation normal ending has not been detected before.
        :return:
        """
        list_crystals = get_list_crystals(self._crystals, crystals)

        print("Checking '{}' simulations and loading results:".format(self._name))
        bar = progressbar.ProgressBar(maxval=len(list_crystals)).start()
        nbar = 1
        for crystal in list_crystals:
            path_output = crystal._path + self._name + ".log"
            if os.path.exists(path_output):
                file_output = open(path_output)
                lines = file_output.readlines()
                if "Finished mdrun" in lines[-2] or "Finished mdrun" in lines[-1]:
                    for i in range(-2, -15, -1):
                        line = lines[i]
                        if line.lstrip().startswith("Potential Energy  ="):
                            # Modify for more than one molecule
                            lattice_energy = float(line.split()[-1]) / crystal._Z - \
                                             self._molecules[0]._potential_energy
                            crystal._energy = lattice_energy
                            crystal._state = "complete"
                            if os.path.exists(crystal._path + self._name + ".trr"):
                                os.chdir(crystal._path)
                                os.system("{0._gromacs} trjconv -f {0.name}.trr -o {0.name}.xtc -s {0.name}.tpr "
                                          "<<< 0 &> /dev/null".format(self))
                                os.remove("{0.name}.trr".format(self))
                            break
                else:
                    print("An error has occurred with Gromacs. Check simulation {} in folder {}."
                          "".format(self._name, crystal._path))
                file_output.close()
                bar.update(nbar)
                nbar += 1
            else:
                print("An error has occurred with Gromacs. Check simulation {} in folder {}."
                      "".format(self._name, crystal._path))
        bar.finish()

        new_rank = dict()
        incomplete_simulations = False
        for crystal in self._crystals:
            if crystal._state != "incomplete":
                new_rank[crystal._name] = crystal._energy
                file_gro = open(crystal._path + self._name + ".gro", "r")
                new_box = file_gro.readlines()[-1].split()
                file_gro.close()
                if len(new_box) == 3:
                    new_box = [float(ii) for ii in new_box] + [0., 0., 0., 0., 0., 0.]
                idx_gromacs = [0, 5, 7, 3, 1, 8, 4, 6, 2]
                crystal._box = np.array([float(new_box[ii]) for ii in idx_gromacs]).reshape((3, 3))
                crystal._cell_parameters = box2cell(crystal._box)
                crystal._volume = np.linalg.det(crystal._box)
            else:
                incomplete_simulations = True
                break

        if not incomplete_simulations:
            rank = 1
            for crystal_name in sorted(new_rank, key=lambda c: new_rank[c]):
                for crystal in self._crystals:
                    if crystal._name == crystal_name:
                        crystal._rank = rank
                        if rank == 1:
                            self._global_minima = crystal
                        rank += 1
            self._completed = True


class CellRelaxation(_GroSim):
    """
    Perform Energy Minimization and Cell Relaxation simulations with LAMMPS. This is used to optimize the simulation
    box parameters, feature not available in Gromacs. If no input is given, it convert Gromacs input files to the
    LAMMPS ones with InterMol.

    Attributes:\n
    - name: Name used to specify the object and print outputs
    - gromacs: Gromacs command line
    - mdrun_options: Options to be added to Gromacs mdrun command. For example '-v', '-v -nt 1', '-plumed plumed.dat'.
    - atomtype: 'atomtype' command line
    - pypol_directory: PyPol directory with defaults inputs
    - path_data: data folder in which simulations are performed
    - path_output: Output folder in which results are written
    - path_input: Input folder to store inputs
    - intermol: Path of the 'convert.py' InterMol program
    - lammps: LAMMPS command line
    - type: "Energy Minimization"
    - crystals: List of crystals on which simulation will be performed
    - sim_index: index of the simulation, define the position of the simulation in the workflow
    - mdp: Molecular Dynamics Parameters file path
    - completed: True if all crystall are not in the "incomplete" state
    - global_minima: Global minima of the set, available if the simulation is completed
    - hide: show or not the the relative potential energy file in the output file
    - path_lmp_in: Input file for LAMMPS
    - path_lmp_ff: LAMMPS Topology file for molecule

    Methods:\n
    - generate_input(bash_script=False, crystals="all"): copy the .mdp file in each crystal folder.
    - get_results(crystals="all"): check if simulations ended or a rerun is necessary.

    TODO not suitable for more than 1 molecule + not possible to define user input and forcefield.
         Transform path_lmp_ff in iterable obj for all mol
         Divide bonded from non-bonded parameters and add read_data at the end with the LJ coeff.
    """

    def __init__(self, name, gromacs, mdrun_options, atomtype, pypol_directory, path_data, path_output,
                 path_input, intermol, lammps, crystals, path_mdp, molecules, index,
                 previous_sim, hide, topology, path_lmp_in=None, path_lmp_ff=None):
        """
        Perform Energy Minimization and Cell Relaxation simulations with LAMMPS. This is used to optimize the simulation
        box parameters, feature not available in Gromacs. If no input is given, it convert Gromacs input files to the
        LAMMPS ones with InterMol.

        Parameters:\n
        :param topology: Path to the .top Gromacs file
        :param name: Name used to specify the object and print outputs
        :param gromacs: Gromacs command line
        :param mdrun_options: Options to be added to Gromacs mdrun command. For example '-v', '-v -nt 1'
        :param atomtype: 'atomtype' command line
        :param pypol_directory: PyPol directory with defaults inputs
        :param path_data: data folder in which simulations are performed
        :param path_output: Output folder in which results are written
        :param path_input: Input folder to store inputs
        :param intermol: Path of the 'convert.py' InterMol program
        :param lammps: LAMMPS command line
        :param crystals: List of crystals on which simulation will be performed
        :param hide: show or not the the relative potential energy file in the output file
        :param path_lmp_in: Input file for LAMMPS
        :param path_lmp_ff:LAMMPS Topology file for molecule
        """
        super().__init__(name, gromacs, mdrun_options, atomtype, pypol_directory, path_data, path_output,
                         path_input, intermol, lammps, "Cell Relaxation", crystals, path_mdp, molecules, index,
                         previous_sim, hide)

        if path_lmp_in:
            if not os.path.exists(path_lmp_in) and not os.path.exists(path_lmp_ff):
                print("Error: File '{}' or '{}' not found".format(path_lmp_in, path_lmp_ff))
                exit()
        self._path_lmp_in = path_lmp_in
        self._path_lmp_ff = path_lmp_ff
        self._topology = topology

    @property
    def path_lmp_in(self):
        return self._path_lmp_in

    @path_lmp_in.setter
    def path_lmp_in(self, path):
        if os.path.exists(path):
            self._path_lmp_in = path
        else:
            print("Error: File '{}' not found".format(path))
            exit()

    @property
    def path_lmp_ff(self):
        return self._path_lmp_ff

    @path_lmp_ff.setter
    def path_lmp_ff(self, path):
        if os.path.exists(path):
            self._path_lmp_ff = path
        else:
            print("Error: File '{}' not found".format(path))
            exit()

    @staticmethod
    def help():
        return """
Perform Energy Minimization and Cell Relaxation simulations with LAMMPS. This is used to optimize the simulation 
box parameters, feature not available in Gromacs. If no input is given, it convert Gromacs input files to the 
LAMMPS ones with InterMol.

Attributes:
- name: Name used to specify the object and print outputs
- gromacs: Gromacs command line
- mdrun_options: Options to be added to Gromacs mdrun command. For example '-v', '-v -nt 1', '-plumed plumed.dat'.
- atomtype: 'atomtype' command line
- pypol_directory: PyPol directory with defaults inputs
- path_data: data folder in which simulations are performed
- path_output: Output folder in which results are written
- path_input: Input folder to store inputs
- intermol: Path of the 'convert.py' InterMol program
- lammps: LAMMPS command line
- type: "Energy Minimization"
- crystals: List of crystals on which simulation will be performed
- sim_index: index of the simulation, define the position of the simulation in the workflow
- mdp: Molecular Dynamics Parameters file path
- completed: True if all crystal are not in the "incomplete" state
- global_minima: Global minima of the set, available if the simulation is completed
- hide: show or not the the relative potential energy file in the output file
- path_lmp_in: Input file for LAMMPS
- path_lmp_ff: LAMMPS Topology file for molecule

Methods:
- generate_input(bash_script=False, crystals="all"): 
- get_results(crystals="all"): check if simulations ended or a rerun is necessary.

Examples: 
- Generate inputs for all the crystals, including a bash script to run all simulations:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
cr = gaff.get_simulation("cr")                                # Retrieve an existing simulation
cr.generate_input(bash_script=True)                           # Copy the MDP file in each crystal directory
project.save()                                                # Save project to be used later

- Check the normal termination of each simulation and get the energy landscape:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
cr = gaff.get_simulation("cr")                                # Retrieve an existing simulation
cr.get_results()                                              # Check normal termination and import potential energy
project.save()                                                # Save project to be used later"""

    def _convert_topology(self, path_gmx, molecule):
        """
        Convert the gromacs topology file to the LAMMPS ones with InterMol.
        :param path_gmx: Topology Gromacs Folder
        :param molecule: Molecule Object with force field parameters
        :return:
        """
        os.chdir(path_gmx)
        os.system("python {0} --gro_in {1}.gro {1}.top --lammps".format(self._intermol, molecule._residue))
        path_lmp = path_gmx + molecule._residue + "_converted.lmp"

        # Check combination rule
        comb_rule = 2
        file_top = open(path_gmx + molecule._residue + ".top")
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
        copyfile(path_lmp, path_gmx + molecule._residue + ".lmp")
        path_lmp = path_gmx + molecule._residue + ".lmp"
        file_lmp = open(path_lmp, "a")
        file_lmp.write("\nPairIJ Coeffs\n\n")
        for pair in pair_ij_list:
            file_lmp.write("{} {}    {:.12f} {:.12f}\n".format(pair[0] + 1, pair[1] + 1, pair[2], pair[3]))
        file_lmp.close()

        # return forcefield path to be used in concomitant with input file
        return path_lmp

    def _check_lmp_input(self):
        """
        Check if conversion is done correctly.
        :return:
        """
        if os.path.exists(self._path_input + "lmp_input.in") and self._path_lmp_in == self._path_input + "lmp_input.in":
            os.rename(self._path_input + "lmp_input.in", self._path_input + "bck.lmp_input.in")
            self._path_lmp_in = self._path_input + "bck.lmp_input.in"
        file_lmp_in = open(self._path_lmp_in)
        file_lmp_in_new = open(self._path_input + "lmp_input.in", "w")
        write_read_data = True
        for line in file_lmp_in:
            if line.startswith("read_data"):
                if write_read_data:
                    for molecule in self._molecules:
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
        return self._path_input + "lmp_input.in"

    def _generate_lammps_topology(self, crystal):
        """
        Check InterMol output and modify it according to crystal properties.
        :param crystal: Crystal Object
        :return:
        """

        working_directory = crystal._path + "lammps/"
        os.mkdir(working_directory)
        os.chdir(working_directory)
        copyfile(self._path_lmp_in, working_directory + "input.in")
        path_gro = crystal._path + self._previous_sim + ".gro"
        for moleculetype in self._molecules:
            molecules_in_crystal = int(crystal._Z / np.sum(crystal._nmoleculestypes) *
                                       crystal._nmoleculestypes[moleculetype._index])
            atoms_in_crystal = moleculetype._natoms * molecules_in_crystal

            # Import coordinates
            coordinates = np.full((atoms_in_crystal, 3), np.nan)
            i = 0
            file_gro = open(path_gro)
            next(file_gro)
            next(file_gro)
            for line in file_gro:
                if line[5:11].strip() == moleculetype._residue:
                    coordinates[i, :] = np.array([float(line[20:28]), float(line[28:36]), float(line[36:44])])
                    i += 1
            file_gro.close()
            coordinates = coordinates * 10

            # Save new lmp file
            file_lmp_ff = open(moleculetype.lmp_forcefield)
            file_lmp_ff_new = open(working_directory + os.path.basename(moleculetype.lmp_forcefield), "w")

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
                    file_lmp_ff_new.write("{:12.8f} {:12.8f} xlo xhi\n".format(0., crystal._box[0, 0] * 10))
                elif line.rstrip().endswith("yhi"):
                    file_lmp_ff_new.write("{:12.8f} {:12.8f} ylo yhi\n".format(0., crystal._box[1, 1] * 10))
                elif line.rstrip().endswith("zhi"):
                    file_lmp_ff_new.write("{:12.8f} {:12.8f} zlo zhi\n"
                                          "{:12.8f} {:12.8f} {:12.8f} xy xz yz\n"
                                          "".format(0., crystal._box[2, 2] * 10, crystal._box[0, 1] * 10,
                                                    crystal._box[0, 2] * 10, crystal._box[1, 2] * 10))
                elif line.rstrip().endswith("xy xz yz"):
                    continue

                # Change body of LAMMPS
                elif "Atoms" in line:
                    file_lmp_ff_new.write("Atoms\n\n")
                    next(file_lmp_ff)
                    for atom in range(moleculetype._natoms):
                        line = next(file_lmp_ff)
                        atoms.append(line.split()[2:4])
                    for atom in range(number_of_atoms):
                        atomtype_idx = atom - int(atom / moleculetype._natoms) * moleculetype._natoms
                        file_lmp_ff_new.write("{:>6} {:>6} {:>6} {:>12.8f} {:>12.7f} {:>12.7f} {:>12.7f}\n"
                                              "".format(atom + 1, int(atom / moleculetype._natoms) + 1,
                                                        int(atoms[atomtype_idx][0]), float(atoms[atomtype_idx][1]),
                                                        coordinates[atom][0], coordinates[atom][1],
                                                        coordinates[atom][2]))
                    file_lmp_ff_new.write("\n")

                elif "Velocities" in line:
                    file_lmp_ff_new.write("Velocities\n\n")
                    next(file_lmp_ff)
                    for vel in range(moleculetype._natoms):
                        line = next(file_lmp_ff)
                        velocities.append(line.split()[1:4])
                    for vel in range(number_of_atoms):
                        vel_idx = vel - int(vel / moleculetype._natoms) * moleculetype._natoms
                        file_lmp_ff_new.write("{:>6} {:>12.7f} {:>12.7f} {:>12.7f}\n"
                                              "".format(vel + 1, float(velocities[vel_idx][0]),
                                                        float(velocities[vel_idx][1]), float(velocities[vel_idx][2])))
                    file_lmp_ff_new.write("\n")

                elif "Bonds" in line:
                    file_lmp_ff_new.write("Bonds\n\n")
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
                                                                2]) + moleculetype._natoms * molecule_idx,
                                                        int(bonds[bondtype_idx][
                                                                3]) + moleculetype._natoms * molecule_idx))
                    file_lmp_ff_new.write("\n")

                elif "Angles" in line:
                    file_lmp_ff_new.write("Angles\n\n")
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
                                                                2]) + moleculetype._natoms * molecule_idx,
                                                        int(angles[angletype_idx][
                                                                3]) + moleculetype._natoms * molecule_idx,
                                                        int(angles[angletype_idx][
                                                                4]) + moleculetype._natoms * molecule_idx))
                    file_lmp_ff_new.write("\n")

                elif "Dihedrals" in line:
                    file_lmp_ff_new.write("Dihedrals\n\n")
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
                                                        int(dihs[dihtype_idx][2]) + moleculetype._natoms * molecule_idx,
                                                        int(dihs[dihtype_idx][3]) + moleculetype._natoms * molecule_idx,
                                                        int(dihs[dihtype_idx][4]) + moleculetype._natoms * molecule_idx,
                                                        int(dihs[dihtype_idx][
                                                                5]) + moleculetype._natoms * molecule_idx))
                    file_lmp_ff_new.write("\n")

                elif "Impropers" in line:
                    file_lmp_ff_new.write("Impropers\n\n")
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
                                                        int(imps[imptype_idx][2]) + moleculetype._natoms * molecule_idx,
                                                        int(imps[imptype_idx][3]) + moleculetype._natoms * molecule_idx,
                                                        int(imps[imptype_idx][4]) + moleculetype._natoms * molecule_idx,
                                                        int(imps[imptype_idx][
                                                                5]) + moleculetype._natoms * molecule_idx))
                    file_lmp_ff_new.write("\n")

                else:
                    file_lmp_ff_new.write(line)
            file_lmp_ff_new.close()
            file_lmp_ff.close()

    def generate_input(self, bash_script=False, crystals="all"):
        """
        Generate LAMMPS inputs. If no topology is given, a LAMMPS topology is generated from the gromacs one
        using Intermol. The latter is applied to a single molecule and then replicated for each molecule of the crystal.


        :param bash_script: If bash_script=True, a bash script is generated to run all simulations
        :param crystals: You can select a specific subset of crystals by listing crystal names in the crystal parameter
        :return:
        """
        if self._path_lmp_ff is None:
            for molecule in self._molecules:
                path_gmx = self._path_input + "GMX2LMP_" + molecule._residue + "/"
                create(path_gmx, arg_type="dir")
                molecule._save_gro(path_gmx + molecule._residue + ".gro")
                copyfile(molecule._forcefield, path_gmx + os.path.basename(molecule._forcefield))

                copyfile(self._topology, path_gmx + molecule._residue + ".top")
                file_top = open(path_gmx + molecule._residue + ".top", "a")
                # for molecule in self.method.molecules:
                file_top.write('#include "{}"\n'.format(os.path.basename(molecule._forcefield)))
                file_top.write("\n[ system ]\n"
                               "Isolated molecule\n"
                               "\n[ molecules ]\n"
                               "; Compound    nmols\n")
                # for molecule in self.method.molecules:
                file_top.write('  {:3}         {}\n'.format(molecule._residue, "1"))
                file_top.close()
                molecule.lmp_forcefield = self._convert_topology(path_gmx, molecule)
        else:
            if isinstance(self._path_lmp_ff, str):
                if len(self._molecules) == 1:
                    if os.path.exists(self._path_lmp_ff):
                        self._molecules[0].lmp_forcefield = self._path_lmp_ff
                    else:
                        print("Error: file '{}' does not exist".format(self._path_lmp_ff))
                        exit()
                else:
                    print("Error: Incorrect number of lammps datafile: should be {}, found 1"
                          "\nPlease write a molecular forcefield for each molecule in Method.molecules"
                          "".format(len(self._molecules)))
                    exit()
            elif hasattr(self._path_lmp_ff, "__iter__"):
                if len(self._path_lmp_ff) == len(self._molecules):
                    for idx in range(len(self._molecules)):
                        if os.path.exists(self._path_lmp_ff[idx]):
                            self._molecules[idx].lmp_forcefield = self._path_lmp_ff[idx]
                        else:
                            print("Error: file '{}' does not exist".format(self._path_lmp_ff))
                            exit()
                else:
                    print("Error: Incorrect number of lammps datafile: should be {}, found {}}"
                          "\nPlease write a molecular forcefield for each molecule in Method.molecules"
                          "".format(len(self._molecules), len(self._path_lmp_ff)))
                    exit()
            else:
                print("Error: No molecular forcefield found in {}".format(self._path_lmp_ff))
                exit()

        if self._path_lmp_in is None:
            self._path_lmp_in = os.path.dirname(self._pypol_directory[:-1]) + "/data/Defaults/LAMMPS/lmp.in"
        else:
            print("Warning: The input file must contain the keyword 'read_data' that will be later modified to "
                  "include the generated input files")

        self._path_lmp_in = self._check_lmp_input()

        list_crystals = get_list_crystals(self._crystals, crystals)

        for crystal in list_crystals:
            print(crystal._name)
            self._generate_lammps_topology(crystal)

        if bash_script:
            file_script = open(self._path_data + "/run_" + self._name + ".sh", "w")
            file_script.write('#!/bin/bash\n\n'
                              'crystal_paths="\n')
            for crystal in list_crystals:
                file_script.write(crystal._path + "lammps/\n")
            file_script.write('"\n\n'
                              'for crystal in $crystal_paths ; do\n'
                              'cd "$crystal" || exit \n'
                              '{0} < {1} \n'
                              'done \n'
                              ''.format(self._lammps, "input.in"))
            file_script.close()

    def get_results(self, crystals="all"):
        """
        Verify if the simulation ended correctly and upload new crystal properties.
        Convert files back to the Gromacs file format.
        :param crystals: You can select a specific subset of crystals by listing crystal names in the crystal parameter.
                         Alternatively, you can use:
                         - "all": Select all non-melted structures
                         - "incomplete": Select crystals whose simulation normal ending has not been detected before.
        :return:
        """
        list_crystals = get_list_crystals(self._crystals, crystals)
        print("Checking '{}' simulations and loading results:".format(self._name))
        bar = progressbar.ProgressBar(maxval=len(list_crystals)).start()
        nbar = 1

        for crystal in list_crystals:
            os.chdir(crystal._path + "lammps/")
            path_coord = crystal._path + "lammps/coordinates.xtc"
            path_output = crystal._path + "lammps/log.lammps"
            if os.path.exists(path_output) and os.path.exists(path_coord):
                file_output = open(path_output)
                for line in file_output:
                    if "Energy initial, next-to-last, final =" in line:
                        line = next(file_output)
                        ref_pot = self._molecules[0]._potential_energy
                        crystal._energy = float(line.split()[-1]) * 4.184 / crystal._Z - ref_pot
                        crystal._state = "complete"
                        break

                os.system("{} trjconv -f coordinates.xtc -s ../{}.tpr -pbc mol -o {}.gro <<< 0 &> PYPOL_TRJCONV_TMP "
                          "".format(self._gromacs, self._previous_sim, self._name))

                os.system("tail -{0} {1}.gro > ../{1}.gro"
                          "".format(int(crystal._Z * self._molecules[0]._natoms + 3), self._name))

                bar.update(nbar)
                nbar += 1
            else:
                print("An error has occurred with LAMMPS. Check simulation {} in folder {}."
                      "".format(self._name, crystal._path + "lammps/"))
        bar.finish()
        new_rank = dict()
        incomplete_simulations = False
        for crystal in self._crystals:
            if crystal._state != "incomplete":
                new_rank[crystal._name] = crystal._energy
                file_gro = open(crystal._path + self._name + ".gro", "r")
                new_box = file_gro.readlines()[-1].split()
                file_gro.close()
                if len(new_box) == 3:
                    new_box = [float(ii) for ii in new_box] + [0., 0., 0., 0., 0., 0.]
                idx_gromacs = [0, 5, 7, 3, 1, 8, 4, 6, 2]
                crystal._box = np.array([float(new_box[ii]) for ii in idx_gromacs]).reshape((3, 3))
                crystal._cell_parameters = box2cell(crystal._box)
                crystal._volume = np.linalg.det(crystal._box)
            else:
                incomplete_simulations = True
                break

        if not incomplete_simulations:
            rank = 1
            for crystal_name in sorted(new_rank, key=lambda c: new_rank[c]):
                for crystal in self._crystals:
                    if crystal._name == crystal_name:
                        crystal._rank = rank
                        if rank == 1:
                            self._global_minima = crystal
                        rank += 1
            self._completed = True


class MolecularDynamics(_GroSim):
    """
    Perform MD simulations using Gromacs.

    Attributes:\n
    - name: Name used to specify the object and print outputs
    - gromacs: Gromacs command line
    - mdrun_options: Options to be added to Gromacs mdrun command. For example '-v', '-v -nt 1', '-plumed plumed.dat'.
    - atomtype: 'atomtype' command line
    - pypol_directory: PyPol directory with defaults inputs
    - path_data: data folder in which simulations are performed
    - path_output: Output folder in which results are written
    - path_input: Input folder to store inputs
    - intermol: Path of the 'convert.py' InterMol program
    - lammps: LAMMPS command line
    - type: "Energy Minimization"
    - crystals: List of crystals on which simulation will be performed
    - sim_index: index of the simulation, define the position of the simulation in the workflow
    - mdp: Molecular Dynamics Parameters file path
    - completed: True if all crystall are not in the "incomplete" state
    - global_minima: Global minima of the set, available if the simulation is completed
    - hide: show or not the the relative potential energy file in the output file

    Methods:\n
    - help(): Print attributes and methods
    - generate_input(bash_script=False, crystals="all"): copy the .mdp file in each crystal folder.
    - get_results(crystals="all"): check if simulations ended or a rerun is necessary.
    """

    def __init__(self, name, gromacs, mdrun_options, atomtype, pypol_directory, path_data, path_output,
                 path_input, intermol, lammps, crystals, path_mdp, molecules, index, previous_sim, hide):
        """
        Perform Energy Minimization simulations using Gromacs.
        :param name: Name used to specify the object and print outputs
        :param gromacs: Gromacs command line
        :param mdrun_options: Options to be added to Gromacs mdrun command.
        :param atomtype: 'atomtype' command line
        :param pypol_directory: PyPol directory with defaults inputs
        :param path_data: data folder in which simulations are performed
        :param path_output: Output folder in which results are written
        :param path_input: Input folder to store inputs
        :param intermol: Path of the 'convert.py' InterMol program
        :param lammps: LAMMPS command line
        :param crystals: List of crystals on which simulation will be performed
        :param hide: show or not the the relative potential energy file in the output file
        """

        super().__init__(name, gromacs, mdrun_options, atomtype, pypol_directory, path_data, path_output,
                         path_input, intermol, lammps, "Molecular Dynamics", crystals, path_mdp, molecules, index,
                         previous_sim, hide)

    @staticmethod
    def help():
        return """
Perform Energy Minimization simulations using Gromacs.

Attributes:
- name: Name used to specify the object and print outputs
- gromacs: Gromacs command line
- mdrun_options: Options to be added to Gromacs mdrun command. For example '-v', '-v -nt 1', '-plumed plumed.dat'.
- atomtype: 'atomtype' command line
- pypol_directory: PyPol directory with defaults inputs
- path_data: data folder in which simulations are performed
- path_output: Output folder in which results are written
- path_input: Input folder to store inputs
- intermol: Path of the 'convert.py' InterMol program
- lammps: LAMMPS command line
- type: "Energy Minimization"
- crystals: List of crystals on which simulation will be performed
- sim_index: index of the simulation, define the position of the simulation in the workflow
- mdp: Molecular Dynamics Parameters file path
- completed: True if all crystall are not in the "incomplete" state
- global_minima: Global minima of the set, available if the simulation is completed
- hide: show or not the the relative potential energy file in the output file

Methods:
- help(): Print attributes and methods
- generate_input(bash_script=False, crystals="all"): copy the .mdp file in each crystal folder.
- get_results(crystals="all"): check if simulations ended or a rerun is necessary.

Examples: 
- Generate inputs for all the crystals, including a bash script to run all simulations:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
nvt = gaff.get_simulation("nvt")                              # Retrieve an existing simulation
nvt.generate_input(bash_script=True)                          # Copy the MDP file in each crystal directory
project.save()                                                # Save project to be used later

- Check the normal termination of each simulation and get the energy landscape:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Retrieve an existing method
nvt = gaff.get_simulation("nvt")                              # Retrieve an existing simulation
nvt.get_results()                                             # Check normal termination and import potential energy
project.save()                                                # Save project to be used later"""

    def generate_input(self, bash_script=False, crystals="all", catt=None):
        """
        Copy the Gromacs .mdp file to each crystal path.
        :param bash_script: If bash_script=True, a bash script is generated to run all simulations
        :param crystals: You can select a specific subset of crystals by listing crystal names in the crystal parameter
        :param catt: Use crystal attributes to select the crystal list
        """

        list_crystals = get_list_crystals(self._crystals, crystals, catt)

        for crystal in list_crystals:
            copyfile(self.path_mdp, crystal._path + self.name + ".mdp")

        if bash_script:
            file_script = open(self.path_data + "/run_" + self.name + ".sh", "w")
            file_script.write('#!/bin/bash\n\n'
                              'crystal_paths="\n')
            for crystal in list_crystals:
                file_script.write(crystal._path + "\n")

            if os.path.exists(self.crystals[0]._path + self._previous_sim + "cpt"):
                file_script.write('"\n\n'
                                  'for crystal in $crystal_paths ; do\n'
                                  'cd "$crystal" || exit \n'
                                  '{0} grompp -f {1}.mdp -c {2}.gro -t {2}.cpt -o {1}.tpr -p topol.top -maxwarn 1 \n'
                                  '{0} mdrun {3} -deffnm {1} \n'
                                  'done \n'
                                  ''.format(self.gromacs, self.name, self._previous_sim, self.mdrun_options))
            else:
                file_script.write('"\n\n'
                                  'for crystal in $crystal_paths ; do\n'
                                  'cd "$crystal" || exit \n'
                                  '{0} grompp -f {1}.mdp -c {2}.gro -o {1}.tpr -p topol.top -maxwarn 1 \n'
                                  '{0} mdrun {3} -deffnm {1} \n'
                                  'done \n'
                                  ''.format(self.gromacs, self.name, self._previous_sim, self.mdrun_options))
            file_script.close()

    def get_results(self, crystals="all", timeinterval=200):
        """
        Verify if the simulation ended correctly and upload new crystal properties.
        :param crystals: You can select a specific subset of crystals by listing crystal names in the crystal parameter.
                         Alternatively, you can use:
                         - "all": Select all non-melted structures
                         - "incomplete": Select crystals whose simulation normal ending has not been detected before.
        :param timeinterval: int, Time interval in ps to use for calculating different properties averages.
                             For example, a timeinterval of 500 means that the last 500 ps of the simulation are used
                             to calculate the average potential energy of the system.
        """

        list_crystals = get_list_crystals(self._crystals, crystals, _include_melted=True)
        if not self._mdp:
            self._mdp = self._import_mdp(self._path_mdp)
        traj_start = int(float(self._mdp["dt"]) * float(self._mdp["nsteps"])) - timeinterval
        print("Checking '{}' simulations and loading results:".format(self._name))
        bar = progressbar.ProgressBar(maxval=len(list_crystals)).start()
        nbar = 1
        for crystal in list_crystals:
            if super()._get_results(crystal):
                os.chdir(crystal._path)
                os.system('{} energy -f {}.edr -b {} <<< "Potential" &> PyPol_Temporary_Potential.txt'
                          ''.format(self.gromacs, self.name, traj_start))
                file_pot = open(crystal._path + 'PyPol_Temporary_Potential.txt')
                for line in file_pot:
                    if line.startswith("Potential"):
                        lattice_energy = float(line.split()[1]) / crystal._Z - self._molecules[0]._potential_energy
                        crystal._energy = lattice_energy
                        crystal._state = "complete"
                        break
                file_pot.close()
                os.remove(crystal._path + 'PyPol_Temporary_Potential.txt')

                bar.update(nbar)
                nbar += 1
            else:
                print("An error has occurred with Gromacs. Check simulation {} in folder {}."
                      "".format(self.name, crystal._path))
        bar.finish()
        new_rank = dict()
        incomplete_simulations = False
        for crystal in self.crystals:
            if crystal._state == "complete":
                new_rank[crystal._name] = crystal._energy
                os.chdir(crystal._path)
                if "pcoupl" in self._mdp and self._mdp["pcoupl"].lower() != "no":
                    os.system('{} energy -f {}.edr -b {} <<< "Box" &> PyPol_Temporary_Box.txt'
                              ''.format(self.gromacs, self.name, traj_start))
                    file_coord = open(crystal._path + 'PyPol_Temporary_Box.txt')
                    new_box = np.zeros((3, 3))
                    for line in file_coord:
                        if line.startswith("Box-XX"):
                            new_box[0, 0] = float(line.split()[1])
                        elif line.startswith("Box-YY"):
                            new_box[1, 1] = float(line.split()[1])
                        elif line.startswith("Box-ZZ"):
                            new_box[2, 2] = float(line.split()[1])
                        elif line.startswith("Box-YX"):
                            new_box[0, 1] = float(line.split()[1])
                        elif line.startswith("Box-ZX"):
                            new_box[0, 2] = float(line.split()[1])
                        elif line.startswith("Box-ZY"):
                            new_box[1, 2] = float(line.split()[1])
                    file_coord.close()
                    os.remove(crystal._path + 'PyPol_Temporary_Box.txt')
                    crystal._box = new_box
                else:
                    file_gro = open(crystal._path + self._name + ".gro", "r")
                    new_box = file_gro.readlines()[-1].split()
                    file_gro.close()
                    if len(new_box) == 3:
                        new_box = [float(ii) for ii in new_box] + [0., 0., 0., 0., 0., 0.]
                    idx_gromacs = [0, 5, 7, 3, 1, 8, 4, 6, 2]
                    crystal._box = np.array([float(new_box[ii]) for ii in idx_gromacs]).reshape((3, 3))
                crystal._cell_parameters = box2cell(crystal._box)
                crystal._volume = np.linalg.det(crystal._box)
                crystal._density = crystal._calculate_density()
            else:
                print("The {} simulation of crystal {} is not completed".format(self._name, crystal._name))
                incomplete_simulations = True

        if not incomplete_simulations:
            rank = 1
            for crystal_name in sorted(new_rank, key=lambda c: new_rank[c]):
                for crystal in self.crystals:
                    if crystal._name == crystal_name:
                        crystal._rank = rank
                        if rank == 1:
                            self._global_minima = crystal
                        rank += 1
            self._completed = "complete"


class Metadynamics(MolecularDynamics):
    _type = "WTMD"

    def __init__(self, name, gromacs, mdrun_options, atomtype, pypol_directory, path_data, path_output,
                 path_input, intermol, lammps, crystals, path_mdp, molecules, index, previous_sim, hide=True,
                 replicas=1, biasfactor=200, pace=1000, height=2.0, temp=300, stride=100):
        """
        Perform Energy Minimization simulations using Gromacs.
        :param name: Name used to specify the object and print outputs
        :param gromacs: Gromacs command line
        :param mdrun_options: Options to be added to Gromacs mdrun command.
        :param atomtype: 'atomtype' command line
        :param pypol_directory: PyPol directory with defaults inputs
        :param path_data: data folder in which simulations are performed
        :param path_output: Output folder in which results are written
        :param path_input: Input folder to store inputs
        :param intermol: Path of the 'convert.py' InterMol program
        :param lammps: LAMMPS command line
        :param crystals: List of crystals on which simulation will be performed
        :param hide: show or not the the relative potential energy file in the output file
        """

        super().__init__(name, gromacs, mdrun_options, atomtype, pypol_directory, path_data, path_output,
                         path_input, intermol, lammps, crystals, path_mdp, molecules, index,
                         previous_sim, hide)

        # MetaD Parameters
        self._biasfactor = biasfactor
        self._pace = pace
        self._height = height
        self._temp = temp
        self._stride = stride
        #self._replicas = replicas

        # Committor: EnergyCutOff
        self._energy_cutoff = 2.5
        self._energy_cutoff_stride = 100000

        # Committor: DRMSD
        self._drmsd = False
        self._drmsd_stride = 100000
        self._drmsd_toll = 0.2
        self._drmsd_lower = 0.1
        self._drmsd_upper = 0.8

        # CVS
        self._cvp = list()
        self.restart = True

        # Analysis
        self._clustering_method = None
        self._intervals = None
        self._analysis_clusters = {}
        self._analysis_clusters_data = {}
        self._analysis_tree = None

    @property
    def analysis_intervals(self):
        if self._intervals:
            return self._intervals
        else:
            print("Error: Energy steps are generated at when using the module 'generate_analysis_inputs'")

    @property
    def clustering_method(self):
        return self._clustering_method

    @clustering_method.setter
    def clustering_method(self, cm):
        from PyPol.cluster import Clustering
        if type(cm) is Clustering:
            self._clustering_method = cm

    @property
    def type(self):
        return self._type

    @property
    def replicas(self):
        return self._replicas

    @replicas.setter
    def replicas(self, value: int):
        self._replicas = value

    @property
    def pace(self):
        return self._pace

    @pace.setter
    def pace(self, value: int):
        self._pace = value

    @property
    def biasfactor(self):
        return self._biasfactor

    @biasfactor.setter
    def biasfactor(self, value: Union[int, float]):
        self._biasfactor = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value: Union[int, float]):
        self._height = value

    @property
    def temp(self):
        return self._temp

    @temp.setter
    def temp(self, value: Union[int, float]):
        self._temp = value

    @property
    def energy_cutoff(self):
        return self._energy_cutoff

    @energy_cutoff.setter
    def energy_cutoff(self, value: Union[int, float]):
        self._energy_cutoff = value

    @property
    def drmsd(self):
        return self._drmsd

    @drmsd.setter
    def drmsd(self, value: bool):
        self._drmsd = value

    @property
    def drmsd_toll(self):
        return self._drmsd_toll

    @drmsd_toll.setter
    def drmsd_toll(self, value: Union[int, float]):
        self._drmsd_toll = value

    @property
    def collective_variables(self):
        return self._cvp

    @collective_variables.setter
    def collective_variables(self, cvs):

        def put_energy_at_the_end(colvars):
            for cv in colvars:
                if cv._type == "Potential Energy":
                    colvars.append(colvars.pop(colvars.index(cv)))
            return colvars

        def put_asb_at_the_beginning(colvars):
            for cv in colvars:
                if cv._short_type == "asb":
                    colvars.insert(0, colvars.pop(colvars.index(cv)))
            return colvars

        if isinstance(cvs, list):
            self._cvp = put_energy_at_the_end(cvs)
            self._cvp = put_asb_at_the_beginning(cvs)
        elif isinstance(cvs, tuple):
            cvs = list(cvs)
            self._cvp = put_energy_at_the_end(cvs)
            self._cvp = put_asb_at_the_beginning(cvs)
        else:
            self._cvp = [cvs]

    def set_cvs(self, *cvs):
        self.collective_variables = list(cvs)

    def generate_input(self, bash_script=True, crystals="all", catt=None):
        """
        Copy the Gromacs .mdp file to each crystal path and produce plumed file for Metadynamics.
        :param catt: Use crystal attributes to select the crystal list.
        :param bash_script: If bash_script=True, a bash script is generated to run all simulations.
        :param crystals: You can select a specific subset of crystals by listing crystal names in the crystal parameter.
                         Alternatively, you can use:
                         - "all": Select all non-melted structures
                         - "incomplete": Select crystals whose simulation normal ending has not been detected before.
        """
        # TODO Only 1 replica available
        from PyPol.walls import AvoidScrewedBox
        from PyPol.metad import _MetaCV, Density, PotentialEnergy, Box
        list_crystals = get_list_crystals(self._crystals, crystals, catt)
        imp = self._molecules[0]._potential_energy
        mw = 0
        for atom in self._molecules[0].atoms:
            mw += atom._mass

        arg, sigma, grid_min, grid_max, grid_bin, walls = [], [], [], [], [], []
        for cv in self._cvp:
            if issubclass(type(cv), _MetaCV):
                arg.append(cv._name)
                sigma.append(str(cv._sigma))
                grid_min.append(str(cv._grid_min))
                grid_max.append(str(cv._grid_max))
                grid_bin.append(str(cv._grid_bins))

            if type(cv) is AvoidScrewedBox:
                walls.append(cv)
            elif type(cv) in [Density, Box]:
                if cv.use_walls:
                    walls.append(cv.lwall)
                    walls.append(cv.uwall)

        arg = ",".join(arg)
        sigma = ",".join(sigma)
        grid_min = ",".join(grid_min)
        grid_max = ",".join(grid_max)
        grid_bin = ",".join(grid_bin)
        arg_output = arg + f",{self._name}.bias,{self._name}.rbias,{self._name}.rct,rct_mol"
        if walls:
            for wall in walls:
                arg_output += f",{wall._name}.bias"

        if not isinstance(self._energy_cutoff, (int, float)):
            nmols_max = 0
            for crystal in list_crystals:
                if crystal._Z > nmols_max:
                    nmols_max = crystal._Z
            self._energy_cutoff = round(1700. / nmols_max, 2)

        print("Creating Plumed files for '{}' simulation:".format(self._name))
        bar = progressbar.ProgressBar(maxval=len(list_crystals)).start()
        nbar = 1
        for crystal in list_crystals:
            file_plumed = open(crystal._path + f"plumed_{self._name}.dat", "w")
            if self.restart:
                file_plumed.write("RESTART\n\n")

            print_cell = True
            for cv in self._cvp:
                if type(cv) is AvoidScrewedBox:
                    file_plumed.write(cv._metad(False))
                    print_cell = False
                elif type(cv) is Box:
                    file_plumed.write(cv._metad(False, print_cell))
                    print_cell = False
                elif type(cv) is Density:
                    cf = mw * crystal._Z * 1.66054
                    file_plumed.write(cv._metad(cf, False))
                elif type(cv) is PotentialEnergy:
                    file_plumed.write(cv._metad(crystal._Z, imp, walls, False))

            file_plumed.write(f"""
# Metadynamics Parameters
METAD ...
LABEL={self._name}
ARG={arg}
PACE={self._pace}
HEIGHT={self._height}
SIGMA={sigma}
TEMP={self._temp}
BIASFACTOR={self._biasfactor}
GRID_MIN={grid_min}
GRID_MAX={grid_max}
GRID_BIN={grid_bin}
CALC_RCT
FILE=HILLS
... METAD

rct_mol: MATHEVAL ARG={self._name}.rct FUNC=x/{int(crystal._Z)}.0 PERIODIC=NO

PRINT STRIDE={self._stride} ARG={arg_output} FILE=plumed_{self._name}_COLVAR

# Stop simulation after energy cutoff is reached
COMMITTOR ...
ARG=rct_mol
STRIDE={self._energy_cutoff_stride}
BASIN_LL1={self._energy_cutoff}
BASIN_UL1={self._energy_cutoff + 10.}
... COMMITTOR\n""")

            if self._drmsd:
                os.chdir(crystal._path)
                os.system("{0._gromacs} trjconv -f {0._previous_sim}.gro -o plumed_{0.name}_TEMPORARY.pdb "
                          "-s {0._previous_sim}.tpr "
                          "<<< 0 &> /dev/null".format(self))
                file_pdb = open(crystal._path + f"plumed_{self._name}_TEMPORARY.pdb", "r")
                file_pdb_out = open(crystal._path + f"plumed_{self._name}.pdb", "w")
                molnum = 1
                for line in file_pdb:
                    if line.startswith(("HETATM", "ATOM")):
                        line = line[:54] + "  1.00  1.00" + line[66:]
                        if int(line[22:26]) != molnum:
                            molnum = int(line[22:26])
                            line = "TER\n" + line
                        file_pdb_out.write(line)
                    else:
                        file_pdb_out.write(line)
                file_pdb_out.close()
                file_pdb.close()
                os.remove(crystal._path + f"plumed_{self._name}_TEMPORARY.pdb")

                file_plumed.write(f"""
# Stop simulation when a phase transition occurred
DRMSD ...
REFERENCE=plumed_{self._name}.pdb 
LOWER_CUTOFF={self._drmsd_lower} 
UPPER_CUTOFF={self._drmsd_upper}
TYPE=INTER-DRMSD
LABEL=drmsd
... DRMSD
PRINT FILE=plumed_{self._name}_DRMSD ARG=drmsd STRIDE={self._stride * 10}
COMMITTOR ...
  ARG=drmsd
  STRIDE={self._drmsd_stride}
  BASIN_LL1={self._drmsd_toll}
  BASIN_UL1={self._drmsd_toll + 10.}
... COMMITTOR\n""")
            file_plumed.close()
            bar.update(nbar)
            nbar += 1
        bar.finish()

        self._mdrun_options = f" -plumed plumed_{self._name}.dat"

        super(Metadynamics, self).generate_input(bash_script, crystals, catt)

    def _check_committor(self, crystal, timeinterval):
        os.chdir(crystal._path)

        def split_traj(traj_file, time):
            time = int(time)
            file_name, file_ext = os.path.splitext(traj_file)
            copyfile(crystal._path + traj_file, crystal._path + "TMP_PYPOL" + file_ext)
            os.system("{0._gromacs} trjconv -f TMP_PYPOL{1} -o {2} -e {3} -s {0._name}.tpr "
                      "<<< 0 &> /dev/null".format(self, file_ext, traj_file, time))
            os.system("{0._gromacs} trjconv -f TMP_PYPOL{1} -o additional_{2} -b {3} -s {0._name}.tpr "
                      "<<< 0 &> /dev/null".format(self, file_ext, traj_file, time))
            os.system("{0._gromacs} trjconv -f TMP_PYPOL{1} -o plumed_{2}.xtc -b {4} -e {3} -s {0._name}.tpr "
                      "<<< 0 &> /dev/null".format(self, file_ext, file_name, time, time - timeinterval))
            if os.path.exists(crystal._path + file_name + ".gro"):
                os.rename(crystal._path + file_name + ".gro", crystal._path + file_name + "_old.gro")
            # TODO Change time -10, time - 100 with the correct timestep in the trajectory (dt * nstxout[-compressed])
            os.system("{0._gromacs} trjconv -f TMP_PYPOL{1} -o {2}.gro -b {3} -dump {4} -s {0._name}.tpr "
                      "<<< 0 &> /dev/null".format(self, file_ext, file_name, time - 100, time - 10))

            os.remove("TMP_PYPOL" + file_ext)

        def split_hills(hills_file, time):
            os.rename(hills_file, "PYPOL_bck." + hills_file)
            file_hills = open("PYPOL_bck." + hills_file, "r")
            file_hills_out = open(hills_file, "w")
            for line in file_hills:
                if line.strip().startswith("#"):
                    file_hills_out.write(line)
                elif float(line.strip().split()[0]) <= time:
                    file_hills_out.write(line)
            file_hills.close()
            file_hills_out.close()

        if self._drmsd:
            committor_drmsd = np.genfromtxt(crystal._path + f"plumed_{self._name}_DRMSD", comments="#")
            if np.max(committor_drmsd[:, 1]) >= self._drmsd_toll:
                traj_end = committor_drmsd[np.argmax(committor_drmsd[:, 1] > self._drmsd_toll), 0]
                print(f"DRMSD cutoff ({self._drmsd}) reached at time {traj_end} ps.")
                split_traj(self._name + ".xtc", traj_end)
                split_hills("HILLS", traj_end)
                return True

        # noinspection PyTypeChecker
        committor_rct = np.genfromtxt(crystal._path + f"plumed_{self._name}_COLVAR", names=True, comments="#! FIELDS ")
        if np.max(committor_rct["rct_mol"]) >= self._energy_cutoff:
            traj_end = int(committor_rct["time"][np.argmax(committor_rct["rct_mol"] > self._energy_cutoff)])
            print(f"Energy cutoff ({self._energy_cutoff}) reached at time {traj_end} ps.")
            split_traj(self._name + ".xtc", traj_end)
            split_hills("HILLS", traj_end)
            return True
        return False

    def get_results(self, crystals="all", timeinterval=50):
        """
        Verify if the simulation ended correctly and upload new crystal properties.
        :param crystals: You can select a specific subset of crystals by listing crystal names in the crystal parameter.
                         Alternatively, you can use:
                         - "all": Select all non-melted structures
                         - "incomplete": Select crystals whose simulation normal ending has not been detected before.
        :param timeinterval: int, Time interval in ps to use for calculating different properties averages.
                             For example, a timeinterval of 500 means that the last 500 ps of the simulation are used
                             to calculate the average potential energy of the system.
        """
        list_crystals = get_list_crystals(self._crystals, crystals, _include_melted=True)
        if not self._mdp:
            self._mdp = self._import_mdp(self._path_mdp)
        print("Checking '{}' simulations and loading results:\n".format(self._name))
        for crystal in list_crystals:
            print(crystal._name)
            print("Check Normal Termination...", end="")
            if super()._get_results(crystal):
                print("done")
                if self._check_committor(crystal, timeinterval):
                    crystal._state = "complete"
                    print("-" * 50)
                else:
                    print("An error has occurred with Gromacs. Check simulation {} in folder {}."
                          "".format(self.name, crystal._path))
            else:
                print("An error has occurred with Gromacs. Check simulation {} in folder {}."
                      "".format(self.name, crystal._path))

        self._completed = "complete"
        for crystal in self.crystals:
            if not crystal._state == "complete":
                self._completed = False

    @staticmethod
    def _set_intervals(start, end, every):
        intervals = []
        if every < 0.1:
            print("Error: Minimum interval is 0.1 kJ mol-1")
        while start <= end:
            intervals.append(round(start, 1))
            start += every
        if end % every != 0:
            intervals.append(end)
        return intervals

    def generate_analysis_input(self, clustering_method=None, crystals="all", catt=None,
                                start=0.5, end=None, interval=0.5, timeinterval=50):
        from PyPol.fingerprints import _OwnDistributions, _Property
        from PyPol.groups import _GG
        from PyPol.cluster import Clustering

        if end is None:
            end = self._energy_cutoff

        if clustering_method is None:
            clustering_method = self._clustering_method

        if type(clustering_method) is not Clustering:
            print("Error: no suitable clustering method used")

        # TODO Include trr files. Check self._mdp for metadynamics
        # if "nstxout-compressed" in self._mdp.keys():
        #     file_ext = "xtc"
        # else:
        #     file_ext = "trr"
        file_ext = "xtc"

        self._intervals = self._set_intervals(start, end, interval)

        list_crystals = get_list_crystals(self._crystals, crystals, catt, _include_melted=False)
        for crystal in list_crystals:
            print(crystal._name)
            os.chdir(crystal._path)
            times = {k: [] for k in self._intervals}
            j = 0
            # noinspection PyTypeChecker
            file_plumed = np.genfromtxt(crystal._path + f"plumed_{self._name}_COLVAR", names=True,
                                        comments="#! FIELDS ")
            for i in range(file_plumed.shape[0]):
                if file_plumed["rct_mol"][i] > self._intervals[j]:
                    times[self._intervals[j]] = (file_plumed["time"][i] - timeinterval, file_plumed["time"][i])
                    j += 1
                    if j == len(self._intervals):
                        break

            if not os.path.exists(f"{self._name}_analysis"):
                os.mkdir(f"{self._name}_analysis")
            for i in self._intervals:
                if not times[i]:
                    break
                if not os.path.exists(f"{self._name}_analysis/" + str(i)):
                    os.mkdir(f"{self._name}_analysis/" + str(i))
                from shutil import copyfile
                copyfile(crystal._path + "mc.dat",
                         crystal._path + f"{self._name}_analysis/" + str(i) + "/mc.dat")
                copyfile(crystal._path + f"{self._name}.tpr",
                         crystal._path + f"{self._name}_analysis/" + str(i) + f"/{self._name}.tpr")
                os.system("{0._gromacs} trjconv -f {0._name}.{1} -o {0._name}_analysis/{2}/{0._name}.xtc -b {3} -e {4} "
                          "-s {0._name}.tpr <<< 0 &> /dev/null"
                          "".format(self, file_ext, str(i), times[i][0], times[i][1]))
                for cv in clustering_method._cvp:
                    if issubclass(type(cv), _OwnDistributions) or issubclass(type(cv), _GG) or issubclass(type(cv), _Property):
                        continue
                    cv.check_attributes()
                    wd = crystal.path + f"/{self._name}_analysis/{str(i)}/"
                    cv.generate_input(crystal,
                                      input_name=wd + f"plumed_{cv._name}.dat",
                                      output_name=wd + f"plumed_{self._name}_{cv._name}.dat")

        file_script = open(self._path_data + "/run_plumed_analysis_" + self._name + ".sh", "w")
        file_script.write('#!/bin/bash\n\n'
                          'crystal_paths="\n')
        for crystal in list_crystals:
            for i in self._intervals:
                path_sim = crystal._path + f"/{self._name}_analysis/{str(i)}/"
                if os.path.exists(path_sim):
                    file_script.write(path_sim + "\n")

        file_script.write('"\n\nfor crystal in $crystal_paths ; do\ncd "$crystal" || exit\n')
        for cv in clustering_method._cvp:
            if issubclass(type(cv), _OwnDistributions) or issubclass(type(cv), _GG):
                continue
            file_script.write('{0} driver --mf_xtc {1}.xtc --plumed plumed_{2}.dat  --mc mc.dat\n'
                              ''.format(cv._plumed, self._name, cv._name))
        file_script.write("done\n")
        file_script.close()

    def get_analysis_results(self, clustering_method=None, crystals="all", catt=None, plot=True):
        import networkx as nwx
        from PyPol.fingerprints import _OwnDistributions
        from PyPol.groups import _GG

        if not os.path.exists(self._path_output + "/analysis"):
            os.mkdir(self._path_output + "/analysis")

        if not self._intervals:
            print("Error: Run 'generate_analysis_inputs' before this module.")

        if clustering_method is None:
            clustering_method = self._clustering_method

        list_crystals = get_list_crystals(self._crystals, crystals, catt, _include_melted=False)

        i_prev = 0

        self._analysis_tree = nwx.Graph()
        for crystal in list_crystals:
            self._analysis_tree.add_node(f"i{round(i_prev, 3)}_{crystal._name}",
                                         structures=1, energy=i_prev, stable=False)

        for i in self._intervals:
            c_name = f"i{round(i, 3)}_"
            c_prev = f"i{round(i_prev, 3)}_"
            suffix = "_" + self._name + "_" + str(i)

            # Remove unfinished structures
            new_list_crystals = []
            for crystal in list_crystals:
                if not os.path.exists(f"{crystal._path}{self._name}_analysis/{i}"):
                    print(i, crystal._name, "melted")
                    self._analysis_tree.add_node(c_name + crystal._name,
                                                 structures=self._analysis_tree.nodes[c_prev + crystal._name][
                                                     "structures"],
                                                 energy=i, stable=False)
                    self._analysis_tree.add_edge(c_name + crystal._name, c_prev + crystal._name)
                else:
                    new_list_crystals.append(crystal)
            list_crystals = new_list_crystals

            if not new_list_crystals:
                continue

            # Import and generate Fingerprints
            for crystal in list_crystals:
                print(i, crystal._name)
                os.chdir(crystal._path)
                for cv in clustering_method._cvp:
                    if issubclass(type(cv), _GG):
                        continue
                    elif issubclass(type(cv), _OwnDistributions):
                        crystal._cvs[cv._name + suffix] = cv.gen_from_traj(
                            crystal=crystal,
                            simulation=self,
                            input_traj=f"{crystal._path}{self._name}_analysis/{i}/{self._name}.xtc",
                            output_label=self._name,
                            plot=plot)
                    else:
                        crystal._cvs[cv._name + suffix] = cv.get_from_file(
                            crystal=crystal,
                            input_file=f"{crystal._path}{self._name}_analysis/{i}/plumed_{self._name}_{cv._name}.dat",
                            output_label=self._name,
                            plot=plot)

            # Generate groups
            for cv in clustering_method._cvp:
                if not issubclass(type(cv), _GG):
                    continue
                cv.run(simulation=self, crystals=list_crystals, catt=catt, suffix=suffix)

            # Cluster
            clustering_method.run(simulation=self,
                                  crystals=list_crystals,
                                  catt=catt,
                                  suffix=suffix,
                                  path_output=self._path_output + "/analysis/" + str(i),
                                  _check=True)  # TODO TEST, change to False after test--->simulation must be completed

            from copy import deepcopy
            self._analysis_clusters[i] = deepcopy(self._clusters)
            self._analysis_clusters_data[i] = deepcopy(self._cluster_data)

            # Update tree
            for crystal in list_crystals:
                if c_name + crystal._state not in self._analysis_tree:
                    self._analysis_tree.add_node(c_name + crystal._state,
                                                 structures=self._analysis_tree.nodes[
                                                     c_prev + crystal._state]["structures"],
                                                 energy=i, stable=True)

                if crystal._state == crystal._name:
                    self._analysis_tree.add_edge(c_name + crystal._name, c_prev + crystal._name)
                else:
                    self._analysis_tree.add_edge(c_name + crystal._state, c_prev + crystal._name)
                    self._analysis_tree.nodes[c_name + crystal._state]["structures"] += 1

            # Update list crystals
            new_list_crystals = []
            for crystal in list_crystals:
                if crystal._name == crystal._state:
                    new_list_crystals.append(crystal)

            list_crystals = new_list_crystals
            i_prev = i

    def _plot_tree(self, tree=None, output_file=None):
        # TODO Move to method including previous step. Define style based on number of molecules.
        #      Possible styles: Sankey, Circles, Histogram.
        #      Force Sankey for large number of structures.
        if tree is None:
            tree = self._analysis_tree
        if output_file is None:
            output_file = self._path_output + f"/analysis/tree_{self._name}.png"
        pos = {}
        labels = {}
        labels_pos = {}
        nodes = [node for node in tree.nodes.keys() if node.startswith(f"i{round(self._intervals[-1], 3)}_")]
        nodes = sorted(nodes, key=lambda n: tree.nodes[n]["structures"], reverse=True)
        for node in nodes:
            labels[node] = node.replace(f"i{round(self._intervals[-1], 3)}_", "")
        for layer in reversed(self._intervals[:-1]):
            tmp_nodes = [node for node in tree.nodes.keys() if node.startswith(f"i{round(layer, 3)}_")]
            for node in tmp_nodes:
                if not tree.nodes[node]["stable"]:
                    nodes.append(node)
                    labels[node] = node.replace(f"i{round(layer, 3)}_", "")

        start = 0.
        for node in nodes:
            bfs = list(nx.bfs_edges(tree, node))
            pos[node] = np.array([start + tree.nodes[node]["structures"] / 2, tree.nodes[node]["energy"]])
            labels_pos[node] = np.array([start + tree.nodes[node]["structures"] / 2, tree.nodes[node]["energy"] + 0.25])
            spl = {}
            for n1, n2 in bfs:
                if tree.nodes[n2]["energy"] not in spl:
                    spl[tree.nodes[n2]["energy"]] = 0.
                pos[n2] = np.array([start + spl[tree.nodes[n2]["energy"]] + tree.nodes[n2]["structures"] / 2.,
                                    tree.nodes[n2]["energy"]])

                spl[tree.nodes[n2]["energy"]] += tree.nodes[n2]["structures"]
            start += tree.nodes[node]["structures"] + 1

        fig, ax = plt.subplots(figsize=(len(self._crystals), len(self._intervals)))
        nx.draw(tree, pos=pos, node_size=10, ax=ax)
        for node in nodes:
            x = labels_pos[node][0]
            y = labels_pos[node][1]
            ax.text(x, y, labels[node], rotation="vertical", fontsize=8)
        ax.yaxis.grid(True)
        ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
        ax.set_ylim(0.25 + self._intervals[-1], -0.25)

        plt.savefig(output_file, dpi=300)
