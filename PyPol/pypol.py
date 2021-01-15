import datetime
import os
import pickle


class Project(object):
    """
    The project class that stores and manage all information and methods.

    Attributes:\n
    - name: Name of the project
    - working_directory: path to the project folder
    - methods: list of methods used in the project
    - initial_crystals: list of structures imported in the project
    - path_progress: general output file of the project
    - path_input: Folder that contains initial structures and methods input files
    - path_output: Folder that contains results and outputs not present in the progress file
    - path_data: Folder that contains all simulations data files

    Methods:\n
    - help(): print attributes and methods available with examples of how to use them
    - new_project(overwrite=False): creates a new project in the working_directory path
    - save(): Save current project
    - change_working_directory(path, reset_program_paths=False): Change project working directory path
    - add_structures(path_structures, gen_unit_cell=False): Add crystal structures to the project
    - new_method(name, package="gromacs"): Create a new Method object
    - get_method(method_name): Get a Method object stored in the project
    - del_method(method_name): Delete a stored Method object and all the dat, Input and Output folders used for it
    """

    def __init__(self, path_working_directory, name="project"):
        """
        Define a new project name and location.
        :param path_working_directory:
        :param name:
        """
        from PyPol import version
        from PyPol import check_package_paths
        package_paths = check_package_paths()

        if path_working_directory.rstrip().endswith("/"):
            path_working_directory = path_working_directory.rstrip()[:-1]

        self._working_directory = path_working_directory
        self._name = name
        self._path_progress = self._working_directory + "/" + self._name + "_output.dat"
        self._path_pickle = self._working_directory + "/.pypol.pkl"
        self._path_input = self._working_directory + "/Input/"
        self._path_input_structures = self._working_directory + "/Input/Initial_Structures/"
        self._path_output = self._working_directory + "/Output/"
        self._path_data = self._working_directory + "/data/"

        self._initial_crystals = list()
        self._methods = list()

        self._pypol_directory = package_paths["path"]
        self._version = version
        self._atomtype = package_paths["atomtype"]
        self._gromacs = package_paths["gromacs"]
        self._lammps = package_paths["lammps"]
        self._intermol = package_paths["intermol"]
        self._plumed = package_paths["plumed"]
        self._htt_plumed = package_paths["htt_plumed"]

    # Read-Only Properties
    @property
    def initial_crystals(self):
        txt = "IDs:\n"
        if self._initial_crystals:
            for crystal in self._initial_crystals:
                txt += crystal._name + "\n"
        return txt

    @property
    def crystals(self):
        return self._initial_crystals

    @property
    def methods(self):
        txt = ""
        if self._methods:
            for method in self._methods:
                txt += method.__str__() + "\n"
        return txt

    @property
    def working_directory(self):
        return self._working_directory

    @working_directory.setter
    def working_directory(self, new_path: str):
        self.change_working_directory(new_path, reset_program_paths=True)

    @property
    def name(self):
        return self._name

    @property
    def path_progress(self):
        return self._path_progress

    @property
    def path_data(self):
        return self._path_data

    @property
    def path_input(self):
        return self._path_input

    @property
    def path_output(self):
        return self._path_output

    @property
    def atomtype_path(self):
        return self._atomtype

    @atomtype_path.setter
    def atomtype_path(self, new_path: str):
        if os.path.exists(new_path):
            self._atomtype = new_path
            for method in self._methods:
                if hasattr(method, "_atomtype"):
                    method._atomtype = new_path
                for simulation in method._simulations:
                    if hasattr(simulation, "_atomtype"):
                        simulation._atomtype = new_path

    @property
    def gromacs_path(self):
        return self._gromacs

    @gromacs_path.setter
    def gromacs_path(self, new_path: str):
        if os.path.exists(new_path):
            self._gromacs = new_path
            for method in self._methods:
                if hasattr(method, "_gromacs"):
                    method._gromacs = new_path
                for simulation in method._simulations:
                    if hasattr(simulation, "_gromacs"):
                        simulation._gromacs = new_path

    @property
    def lammps_path(self):
        return self._lammps

    @lammps_path.setter
    def lammps_path(self, new_path: str):
        if os.path.exists(new_path):
            self._lammps = new_path
            for method in self._methods:
                if hasattr(method, "_lammps"):
                    method._lammps = new_path
                for simulation in method._simulations:
                    if hasattr(simulation, "_lammps"):
                        simulation._lammps = new_path

    @property
    def intermol_path(self):
        return self._intermol

    @intermol_path.setter
    def intermol_path(self, new_path: str):
        if os.path.exists(new_path):
            self._intermol = new_path
            for method in self._methods:
                if hasattr(method, "_intermol"):
                    method._intermol = new_path
                for simulation in method._simulations:
                    if hasattr(simulation, "_intermol"):
                        simulation._intermol = new_path

    @property
    def plumed_path(self):
        return self._plumed

    @plumed_path.setter
    def plumed_path(self, new_path: str):
        if os.path.exists(new_path):
            self._plumed = new_path
            for method in self._methods:
                if hasattr(method, "_plumed"):
                    method._plumed = new_path
                for cv in method._cvp:
                    if hasattr(cv, "_plumed") and cv._type in ("Radial Distribution Function", "Density", "Energy"):
                        cv._plumed = new_path

    @property
    def htt_plumed_path(self):
        return self._htt_plumed

    @htt_plumed_path.setter
    def htt_plumed_path(self, new_path: str):
        if os.path.exists(new_path):
            self._htt_plumed = new_path
            for method in self._methods:
                if hasattr(method, "_htt_plumed"):
                    method._htt_plumed = new_path
                for cv in method._cvp:
                    if hasattr(cv, "_plumed") and cv._type.startswith(("Torsional Angle", "Molecular Orientation")):
                        cv._plumed = new_path

    @property
    def pypol_path(self):
        return self._pypol_directory

    @pypol_path.setter
    def pypol_path(self, new_path: str):
        new_path = os.path.realpath(new_path)
        if os.path.exists(new_path) and os.path.exists(new_path + "/pypol.py"):
            self._pypol_directory = new_path
            for method in self._methods:
                if hasattr(method, "_pypol_directory"):
                    method._pypol_directory = new_path
                for simulation in method._simulations:
                    if hasattr(simulation, "_pypol_directory"):
                        simulation._pypol_directory = new_path

    @staticmethod
    def help():
        print("""
The project class that stores and manage all information and methods.

Attributes:
- name: Name of the project
- working_directory: path to the project folder
- methods: list of methods used in the project
- initial_crystals: list of structures imported in the project
- path_progress: general output file of the project
- path_input: Folder that contains initial structures and methods input files
- path_output: Folder that contains results and outputs not present in the progress file
- path_data: Folder that contains all simulations data files

Methods:
- help(): print attributes and methods available with examples of how to use them.
- new_project(overwrite=False): creates a new project in the working_directory path. 
                If overwrite=True, the previous folder is deleted.
- save(): Save current project.
- change_working_directory(path, reset_program_paths=True): Change project working directory path. 
                If reset_program_paths=True, also programs paths are updated. 
- add_structures(path_structures, gen_unit_cell=False): Add crystal structures to the project in the Input folder.
                For available file formats, please refer to the Open Babel documentation. 
                If gen_unit_cell=True, it converts asymmetric cells to unit cells with the CSD Python API.
- new_method(name, package="gromacs"): Create a new Method object using the Gromacs MD package.
- get_method(method_name): Get a Method object stored in the project
- del_method(method_name): Delete a stored Method object and all the data, Input and Output folders used for it.

Examples:
- Create a new project and print the help() function:
from PyPol import pypol as pp
project = pp.new_project(r'/home/Work/Project/')              # Creates a new project in folder /home/Work/Project/
project.help()                                                # Print available attributes, methods and examples
project.save()                                                # Save project to be used later

- Load an existing project and print information
from PyPol import pypol as pp
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder
print(project.name)                                           # Name of the project
print(project.working_directory)                              # Path to the project folder
project.save()                                                # Save project to be used later

- Change the working directory. Copy/Move the project in the new path and then run:
from PyPol import pypol as pp                                                   
project = pp.load_project(r'/home/Work/New_Project_Path/')    # Load project from the specified folder
project.working_directory = r'/home/Work/New_Project_Path/'   # Change the working directory. This might take a while
project.save()                                                # Save project to be used later
                                                              
- Add new crystal structures to the project:                  
from PyPol import pypol as pp
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
project.add_structures(r'/home/Work/Structures/')             # Add all crystal structures in the given folder
project.add_structures(r'/home/Work/Structures/str1.pdb')     # Add the specified crystal structure
print(project.initial_crystals)                               # List of structures imported in the project
project.save()                                                # Save project to be used later      

- Create a new method and print its manual:                                                                            
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.new_method('GAFF')                             # Creates a new method
gaff.help()                                                   # Print new method manual
project.save()                                                # Save project to be used later                           
        
- Retrieve an existing method:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
gaff = project.get_method('GAFF')                             # Get an existing method
print(gaff.package)                                           # Print method MD package
project.save()                                                # Save project to be used later

- Delete an existing method:
from PyPol import pypol as pp                                                                                           
project = pp.load_project(r'/home/Work/Project/')             # Load project from the specified folder                  
project.del_method('GAFF')                                    # Delete an existing method
project.save()                                                # Save project to be used later""")

    def __str__(self):
        return """PyPol {0._version} {1}
Project Name: {0._name}\n
Working Directory: {0._working_directory}
Number of Structures: {2}
Number of Methods: {3}
        """.format(self, datetime.datetime.now().strftime("%c"), len(self._initial_crystals), len(self._methods))

    def _write_output(self):
        """
        Write main features of the currect project to the project output file.
        :return:
        """

        file_output = open(self._path_progress, "w")
        file_output.write(self.__str__())
        file_output.close()

        for method in self._methods:
            method._write_output(self._path_progress)

    def save(self):
        """
        Save project to project folder.
        :return:
        """
        print("Saving Project...", end="")
        import pickle
        import os
        if os.path.exists(self._working_directory + "/.pypol.pkl"):
            os.rename(self._working_directory + "/.pypol.pkl", self._working_directory + "/.pypol.bck.pkl")
        with open(self._path_pickle, "wb") as file_pickle:
            pickle.dump(self, file_pickle)
        self._write_output()
        print("done")

    def change_working_directory(self, path: str, reset_program_paths: bool = False):
        """
        Change the working directory and all the paths in the project.
        :param path: New Project Path
        :param reset_program_paths: If True, check packages paths
        :return:
        """
        print("-" * 50)
        print("Changing project directory from:\n'{}'\nto:\n'{}'".format(self._working_directory, path))
        self._working_directory = path
        self._path_progress = self._working_directory + "/" + self._name + "_output.dat"
        self._path_pickle = self._working_directory + "/.pypol.pkl"
        self._path_input = self._working_directory + "/Input/"
        self._path_input_structures = self._working_directory + "/Input/Initial_Structures/"
        self._path_output = self._working_directory + "/Output/"
        self._path_data = self._working_directory + "/data/"

        for crystal in self._initial_crystals:
            crystal._path = self._path_input_structures + crystal._name

        for method in self._methods:
            method._path_data = self._path_data + method._name + "/"
            method._path_input = self._path_input + method._name + "/"
            method._path_output = self._path_output + method._name + "/"
            method._topology = method._path_input + os.path.basename(method._topology)

            for molecule in method.molecules:
                molecule._forcefield = method._path_input + os.path.basename(molecule._forcefield)
            for crystal in method._initial_crystals:
                crystal._path = method._path_data + crystal._name + "/"

            for simulation in method._simulations:
                simulation._path_data = method._path_data
                simulation._path_output = method._path_output
                simulation._path_input = method._path_input
                simulation._project = method._project
                simulation._path_mdp = simulation._path_input + simulation._name + ".mdp"
                if hasattr(simulation, "_topology"):
                    simulation._topology = simulation._path_input + os.path.basename(simulation._topology)
                for crystal in simulation._crystals:
                    crystal._path = method._path_data + crystal._name + "/"

        if reset_program_paths:
            from PyPol import check_package_paths
            package_paths = check_package_paths()

            from PyPol import version
            self.pypol_path = package_paths["path"]
            self._version = version
            self.atomtype_path = package_paths["atomtype"]
            self.gromacs_path = package_paths["gromacs"]
            self.lammps_path = package_paths["lammps"]
            self.intermol_path = package_paths["intermol"]
            self.plumed_path = package_paths["plumed"]
            self.htt_plumed_path = package_paths["htt_plumed"]
        print("-" * 50)

    def add_structures(self, path_structures: str):
        """
        Add a new structure (if path_structures is a file) or a set of structures (if path_structures is a folder)
        in the project_folder/Input/Sets/Set_name directory. Structures are converted to the pdb files using openbabel.
        The GAFF atomtype of each atom is detected using the atomtype program from AmberTools. This is required for the
        reindexing of atoms once a molecular forcefield is uploaded but has no impact in the forcefield used for
        simulations. Finally, for each structure, a Crystal object is created and stored in the Project object.
        A list of stored structures can be printed by typing "print(<project_name>.initial_crystals)".

        :param path_structures: Path to a structure file or a folder containg the structures.
        """
        import os
        from openbabel import openbabel
        from PyPol.utilities import create, get_identifier
        from PyPol.crystals import Crystal
        import progressbar

        if not path_structures.endswith("/") and os.path.isdir(path_structures):
            path_structures += "/"

        available_file_formats_ob = ("abinit", "acesout", "acr", "adfband", "adfdftb", "adfout", "alc", "aoforce",
                                     "arc", "axsf", "bgf", "box", "bs", "c09out", "c3d1", "c3d2", "caccrt", "can",
                                     "car", "castep", "ccc", "cdjson", "cdx", "cdxml", "cif", "ck", "cml", "cmlr",
                                     "cof", "CONFIG", "CONTCAR", "CONTFF", "crk2d", "crk3d", "ct", "cub", "cube",
                                     "dallog", "dalmol", "dat", "dmol", "dx", "ent", "exyz", "fa", "fasta", "fch",
                                     "fchk", "fck", "feat", "fhiaims", "fract", "fs", "fsa", "g03", "g09", "g16",
                                     "g92", "g94", "g98", "gal", "gam", "gamess", "gamin", "gamout", "got", "gpr",
                                     "gro", "gukin", "gukout", "gzmat", "hin", "HISTORY", "inchi", "inp", "ins", "jin",
                                     "jout", "log", "lpmd", "mcdl", "mcif", "MDFF", "mdl", "ml2", "mmcif", "mmd",
                                     "mmod", "mol", "mol2", "mold", "molden", "molf", "moo", "mop", "mopcrt", "mopin",
                                     "mopout", "mpc", "mpo", "mpqc", "mrv", "msi", "nwo", "orca", "out", "outmol",
                                     "output", "pc", "pcjson", "pcm", "pdb", "pdbqt", "png", "pos", "POSCAR", "POSFF",
                                     "pqr", "pqs", "prep", "pwscf", "qcout", "res", "rsmi", "rxn", "sd", "sdf",
                                     "siesta", "smi", "smiles", "smy", "sy2", "t41", "tdd", "text", "therm", "tmol",
                                     "txt", "txyz", "unixyz", "VASP", "vmol", "xml", "xsf", "xtc", "xyz", "yob")

        items = list()
        if os.path.isdir(path_structures):
            items = [f for f in os.listdir(path_structures) if os.path.isfile(path_structures + f)]
        elif os.path.isfile(path_structures):
            items = [os.path.basename(path_structures)]
            path_structures = os.path.dirname(path_structures) + "/"
            print(items, path_structures)
        else:
            print("No such file or directory")

        print("Importing structures from folder {}".format(path_structures))
        bar = progressbar.ProgressBar(maxval=len(items)).start()
        nbar = 1
        for item in items:
            id_name, extension = get_identifier(path_structures + item)
            path_id = self._path_input_structures + id_name
            path_structure_pdb = path_id + "/pc.pdb"
            path_structure_mol2 = path_id + "/pc.mol2"
            path_structure_ac = path_id + "/pc.ac"

            if extension in available_file_formats_ob:
                # Create folder in Input
                create(path_id, arg_type='dir', backup=True)
                os.chdir(path_id)
                os.system("cp {}{} {}".format(path_structures, item, path_id))
                # Change file format to pdb
                ob_conversion = openbabel.OBConversion()
                ob_conversion.SetInAndOutFormats(extension, "pdb")
                mol = openbabel.OBMol()
                ob_conversion.ReadFile(mol, path_structures + item)
                ob_conversion.WriteFile(mol, path_structure_pdb)

            else:
                print("Ignore structure '{}': unknown file format".format(item))
                continue

            # Convert structure to mol2 to identify atomtype
            ob_conversion = openbabel.OBConversion()
            ob_conversion.SetInAndOutFormats("pdb", "mol2")
            mol = openbabel.OBMol()
            ob_conversion.ReadFile(mol, path_structure_pdb)
            ob_conversion.WriteFile(mol, path_structure_mol2)
            os.system(self._atomtype + " -i " + path_structure_mol2 + " -f mol2 -p gaff -o " + path_structure_ac)

            new_crystal = Crystal._loadfrompdb(id_name, path_structure_pdb, include_atomtype=True)
            new_crystal._index = len(self._initial_crystals)
            new_crystal._save_pdb(path_structure_pdb)
            self._initial_crystals.append(new_crystal)
            bar.update(nbar)
            nbar += 1
            print(new_crystal._path)
        bar.finish()
        print("=" * 100)

    def new_method(self, name: str, package="gromacs", _import=False):
        """
        Add a new method to the project. Type <method_name>.help() for more information about how to use it.
        :param name: str, label of the method, it will be used to retrieve the method later on
        :param package: MD package used to perform simulations
        :param _import:
        :return: Method object
        """
        import copy
        if package.lower() == "gromacs":
            from PyPol.gromacs import Method
            from PyPol.utilities import create
            if self._methods:
                for existing_method in self._methods:
                    if existing_method._name == name:
                        print("Error: Method name already used")
                        return

            path_new_data_directory = self._path_data + name + "/"
            path_new_input_directory = self._path_input + name + "/"
            path_new_output_directory = self._path_output + name + "/"
            if not _import:
                create(path_new_data_directory, arg_type="dir")
                create(path_new_input_directory, arg_type="dir")
                create(path_new_output_directory, arg_type="dir")

            method = Method(name=name, gromacs=self._gromacs, mdrun_options="", atomtype=self._atomtype,
                            pypol_directory=self._pypol_directory, path_data=path_new_data_directory,
                            path_output=path_new_output_directory, path_input=path_new_input_directory,
                            intermol=self._intermol, lammps=self._lammps,
                            initial_crystals=copy.deepcopy(self._initial_crystals), plumed=self._plumed,
                            htt_plumed=self._htt_plumed)
            self._methods.append(method)
            return method

    def get_method(self, method_name):
        """
        Find an existing method by its name.
        :param method_name: Label given to the Method object
        :return: Method object
        """
        if self._methods:
            for existing_method in self._methods:
                if existing_method._name == method_name:
                    return existing_method
        print("No method found with name {}".format(method_name))

    def del_method(self, method_name):
        """
        Delete an existing method.
        :param method_name: Label given to the Method object
        """
        import shutil

        for method in self._methods:
            if method._name == method_name:
                delete = input("All files in folders:\n{}\n{}\n{}\nwill be deleted. Continue [y/n]? "
                               "".format(method._path_output, method._path_input, method._path_data))
                if delete == "y":
                    shutil.rmtree(method._path_output)
                    shutil.rmtree(method._path_input)
                    shutil.rmtree(method._path_data)
                    self._methods.remove(method)
                return
        print("No method found with name {}".format(method_name))


def load_project(project_folder: str, use_backup=False):
    """
    Load an existing project. The Project object is saved in the project directory every time the command Project.save()
    is used.
    :param project_folder: project folder specified in the new_project function
    :param use_backup: Use the Project object for the previous save
    :return: Project object
    """
    project_folder = os.path.realpath(project_folder)
    file_pickle = project_folder + "/.pypol.pkl"
    if use_backup:
        file_pickle = project_folder + "/.pypol.bck.pkl"
    if os.path.exists(file_pickle):
        project = pickle.load(open(file_pickle, "rb"))
        print("PyPol {}\nProject Name: {}\n".format(project._version, project._name))
        if os.path.realpath(project._working_directory) != project_folder:
            project.working_directory = project_folder
        return project
    else:
        print("No PyPol project found in '{}'. Use the 'Project.new_project' module to create a new project."
              "".format(project_folder))


def new_project(path_working_directory: str, name="project", overwrite=False):
    """
    Create a new Project object and generate all the fundamental directories in the project folder.
    :param name: Name of the new project.
    :param path_working_directory: Path to the new project directory. A new directory will be created.
    :param overwrite: if True, delete the previous folder.
    :return: Project object
    """

    from PyPol.utilities import create
    path_working_directory = os.path.realpath(path_working_directory)
    nproject = Project(path_working_directory, name)

    if not os.path.exists(nproject._working_directory):
        print("PyPol {}\nNew Project: {}".format(nproject._version, nproject._name))
        print("=" * 100)
        create(nproject._working_directory, arg_type='dir', backup=False)
    else:
        if overwrite:
            print("PyPol {}\nNew Project: {}".format(nproject._version, nproject._name))
            print("=" * 100)
            create(nproject._working_directory, arg_type='dir', backup=False)
        else:
            print("Error: Folder already exists.\n "
                  "You can change directory name or overwrite with 'overwrite=True' "
                  "(everything inside will be deleted)")
            exit()
    create(nproject._path_input, arg_type='dir')
    create(nproject._path_output, arg_type='dir')
    create(nproject._path_data, arg_type='dir')

    return nproject
