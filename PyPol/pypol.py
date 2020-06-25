class Project(object):

    def __init__(self, path_working_directory, name="project"):
        """
        Define a new project name and location.
        :param path_working_directory:
        :param name:
        """
        from PyPol.Defaults.defaults import package_paths, pypol_info
        if path_working_directory.rstrip().endswith("/"):
            path_working_directory = path_working_directory.rstrip()[:-1]

        self.working_directory = path_working_directory
        self.name = name
        self.path_progress = self.working_directory + "/" + self.name + "_output.dat"
        self.path_pickle = self.working_directory + "/.pypol.pkl"
        self.path_input = self.working_directory + "/Input/"
        self.path_input_structures = self.working_directory + "/Input/Initial_Structures/"
        self.path_output = self.working_directory + "/Output/"
        self.path_data = self.working_directory + "/data/"

        self.initial_crystals = list()
        self.methods = list()

        self.pypol_directory = pypol_info["path"]
        self.version = pypol_info["version"]
        self.run_csd_python_api = package_paths["run_csd_python_api"]
        self.atomtype = package_paths["atomtype"]
        self.gromacs = package_paths["gromacs"]
        self.lammps = package_paths["lammps"]
        self.intermol = package_paths["intermol"]
        self.plumed = package_paths["plumed"]
        self.htt_plumed = package_paths["htt_plumed"]

    def new_project(self, overwrite=False):
        """
        Generate all the fundamental directories in the project folder.
        :param overwrite: if True, delete the previous folder
        :return:
        """
        import os
        from PyPol.utilities import create

        print("PyPol {}\nNew Project: {}".format(self.version, self.name))
        print("=" * 100)
        if not os.path.exists(self.working_directory):
            create(self.working_directory, arg_type='dir', backup=False)
        else:
            if overwrite:
                create(self.working_directory, arg_type='dir', backup=False)
            else:
                print("Error: Folder already exists. "
                      "You can change directory name or overwrite with overwrite=True")
                exit()
        create(self.path_input, arg_type='dir')
        create(self.path_output, arg_type='dir')
        create(self.path_data, arg_type='dir')

        self.save()

    def save(self):
        """
        Save project to project folder.
        :return:
        """
        import pickle
        import os
        # print("Saving updates...", end="")
        if os.path.exists(self.working_directory + "/.pypol.pkl"):
            os.rename(self.working_directory + "/.pypol.pkl", self.working_directory + "/.pypol.bck.pkl")
        with open(self.path_pickle, "wb") as file_pickle:
            pickle.dump(self, file_pickle)
        self.write_output()
        # print("done")

    def write_output(self):
        """
        Write main features of the currect project to the project output file.
        :return:
        """
        import datetime
        file_output = open(self.path_progress, "w")
        today = datetime.datetime.now()
        file_output.write("PyPol {} {}\n\n"
                          "Project Name: {}\n"
                          "Working Directory: {}\n\n"
                          "Number of Structures: {}\n"
                          "Number of Methods: {}\n\n"
                          "".format(self.version, today.strftime("%c"), self.name, self.working_directory,
                                    len(self.initial_crystals), len(self.methods)))
        file_output.close()

        for method in self.methods:
            method.write_output(self.path_progress)

    def change_working_directory(self, path, reset_program_paths=False):
        """
        Change the working directory and all the paths in the project.
        :param path:
        :param reset_program_paths:
        :return:
        """
        print("Changing project directory from:\n'{}'\nto:\n'{}'".format(self.working_directory, path))
        self.working_directory = path
        self.path_progress = self.working_directory + "/" + self.name + "_output.dat"
        self.path_pickle = self.working_directory + "/.pypol.pkl"
        self.path_input = self.working_directory + "/Input/"
        self.path_input_structures = self.working_directory + "/Input/Initial_Structures/"
        self.path_output = self.working_directory + "/Output/"
        self.path_data = self.working_directory + "/data/"
        for crystal in self.initial_crystals:
            crystal.path = self.path_input_structures + crystal.name

        for method in self.methods:
            method.path_data = self.path_data + method.name + "/"
            method.path_input = self.path_input + method.name + "/"
            method.path_output = self.path_output + method.name + "/"
            method.project = self
            for crystal in method.initial_crystals:
                crystal.path = method.path_data + crystal.name + "/"
                crystal.molecules = crystal.load_molecules()
                crystal.save()

            for simulation in method.energy_minimisation + method.molecular_dynamics:  # + method.metadynamics !!!!
                simulation.path_data = method.path_data
                simulation.path_output = method.path_output
                simulation.path_input = method.path_input
                simulation.project = method.project
                simulation.method = method
                simulation.mdp = simulation.path_input + simulation.name + ".mdp"
                for crystal in simulation.crystals:
                    crystal.path = method.path_data + crystal.name + "/"
                    # if simulation.type == "Cell Relaxation":
                    #     crystal.path = method.path_data + crystal.name + "/lammps/"
                    # else:
                    #     crystal.path = method.path_data + crystal.name + "/"

        if reset_program_paths:
            from PyPol.Defaults.defaults import package_paths, pypol_info

            self.pypol_directory = pypol_info["path"]
            self.version = pypol_info["version"]
            self.run_csd_python_api = package_paths["run_csd_python_api"]
            self.atomtype = package_paths["atomtype"]
            self.gromacs = package_paths["gromacs"]
            self.lammps = package_paths["lammps"]
            self.intermol = package_paths["intermol"]
            self.plumed = package_paths["plumed"]
            self.htt_plumed = package_paths["htt_plumed"]
            for method in self.methods:
                if method.package == "Gromacs":
                    method.pypol_directory = pypol_info["path"]
                    method.command = self.gromacs
                if method.package == "LAMMPS":
                    method.pypol_directory = pypol_info["path"]
                    method.command = self.lammps
                for simulation in method.energy_minimisation + method.molecular_dynamics:
                    simulation.command = method.command
                    if method.package != "LAMMPS" and simulation.type == "Cell Relaxation":
                        simulation.lammps = self.lammps
                        simulation.intermol = self.intermol
        self.save()

    def _file2pdb(self, path_id):
        """
        Convert a structure to the .pdb file format, normalize labels and pack it.
        Error: Many steps should be made since the conversion program of the CSD python API fails in the direct
        conversion of many file format to the .pdb one.
        :param path_id:
        :return:
        """
        import os
        print("Importing structure '{}'".format(os.path.basename(path_id)))
        path_id_dir = os.path.dirname(path_id)
        file_python = path_id_dir + "/converter_csd.py"

        file_default = open(self.pypol_directory + "Defaults/converter_csd.py")
        file_converter = open(file_python, "w")
        for line in file_default:
            if "PATH_TO_FILE" in line:
                file_converter.write('path_id = r"{}"\n'.format(path_id))
            else:
                file_converter.write(line)
        file_converter.close()
        file_default.close()

        os.system(self.run_csd_python_api + " < " + file_python)

    def add_structures(self, path_structures, asymmetric_unit=True):
        """
        Error: Use the CSD Python API only if the asymmetric unit is given (use openbabel otherwise).
        Add a new set of structures in the project_folder/Input/Sets/Set_name directory.
        :param asymmetric_unit:
        :param path_structures:
        :return:
        """
        import os
        from openbabel import openbabel
        from PyPol.utilities import create, get_identifier
        from PyPol.crystals import Crystal

        if not path_structures.endswith("/"):
            path_structures += "/"

        available_file_formats_csd = ("aser", "cif", "csdsql", "csdsqlx", "identifiers", "mariadb", "mol", "mol2",
                                      "res", "sdf", "sqlite", "sqlmol2", "pdb")  # change for openbabel
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
            items = list(os.path.basename(path_structures))
            path_structures = os.path.dirname(path_structures) + "/"
        else:
            print("No such file or directory")

        for item in items:

            id_name, extension = get_identifier(path_structures + item)
            path_id = self.path_input_structures + id_name
            path_structure_pdb = path_id + "/pc.pdb"
            path_structure_mol2 = path_id + "/pc.mol2"
            path_structure_ac = path_id + "/pc.ac"
            path_structure = path_id + "/" + item

            if asymmetric_unit:
                if extension in available_file_formats_csd:
                    # Create folder in Input
                    create(path_id, arg_type='dir', backup=True)
                    os.chdir(path_id)
                    os.system("cp {}{} {}".format(path_structures, item, path_id))
                    # Change fileformat to pdb
                    self._file2pdb(path_structure)
                else:
                    print("Ignore structure '{}': unknown file format".format(item))
                    continue

            elif extension in available_file_formats_ob:
                # Create folder in Input
                create(path_id, arg_type='dir', backup=True)
                os.chdir(path_id)
                os.system("cp {}{} {}".format(path_structures, item, path_id))
                # Change file format to pdb
                ob_conversion = openbabel.OBConversion()
                ob_conversion.SetInAndOutFormats(extension, "pdb")
                mol = openbabel.OBMol()
                ob_conversion.ReadFile(mol, path_structures+item)
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
            os.system(self.atomtype + " -i " + path_structure_mol2 + " -f mol2 -p gaff -o " + path_structure_ac)

            new_crystal = Crystal.loadfrompdb(id_name, path_structure_pdb, include_atomtype=True)
            new_crystal.index = len(self.initial_crystals)
            self.initial_crystals.append(new_crystal)
            new_crystal.save()

        self.save()
        print("=" * 100)

    def method(self, method_name):
        """
        Find an existing method by its name.
        :param method_name:
        :return:
        """
        if self.methods:
            for existing_method in self.methods:
                if existing_method.name == method_name:
                    return existing_method
        print("No method found with name {}".format(method_name))

    def delete_method(self, method_name):
        """
        Delete an existing method.
        :param method_name:
        :return:
        """
        import shutil
        for method in self.methods:
            if method.name == method_name:
                shutil.rmtree(method.path_output)
                shutil.rmtree(method.path_input)
                shutil.rmtree(method.path_data)
                self.methods.remove(method)
                return
        print("No method found with name {}".format(method_name))

    def add_method(self, method):
        """
        Add a method to the project.
        :param method:
        :return:
        """
        import copy
        from PyPol.utilities import create
        if self.methods:
            for existing_method in self.methods:
                if existing_method.name == method.name:
                    print("Method name already used")
                    return

        method.initial_crystals = copy.deepcopy(self.initial_crystals)

        path_new_data_directory = self.path_data + method.name + "/"
        method.path_data = path_new_data_directory
        create(path_new_data_directory, arg_type="dir")

        path_new_input_directory = self.path_input + method.name + "/"
        method.path_input = path_new_input_directory
        create(path_new_input_directory, arg_type="dir")

        path_new_output_directory = self.path_output + method.name + "/"
        method.path_output = path_new_output_directory
        create(path_new_output_directory, arg_type="dir")

        method.atomtype = self.atomtype
        method.pypol_directory = self.pypol_directory
        method.command = self.gromacs
        self.methods.append(method)
        method.project = self
        self.save()


def load_project(project_folder, use_backup=False):
    """
    Load an existing project.
    :param project_folder:
    :param use_backup:
    :return:
    """
    import pickle
    import os
    file_pickle = project_folder + "/.pypol.pkl"
    if use_backup:
        file_pickle = project_folder + "/.pypol.bck.pkl"
    if os.path.exists(file_pickle):
        project = pickle.load(open(file_pickle, "rb"))
        print("PyPol {}\nProject Name: {}\n".format(project.version, project.name))
        return project
    else:
        print("No PyPol project found in '{}'. Use the 'Project.new_project' module to create a new project."
              "".format(project_folder))
