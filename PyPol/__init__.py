import pickle
import os
import shutil

# PyPol Version
version = "20.10"


def set_default_package_path(package, path):

    path_packages = os.path.dirname(os.path.realpath(__file__)) + "/packages.pkl"
    if os.path.exists(path_packages):
        with open(path_packages, "rb") as packages:
            package_paths = pickle.load(packages)

    if package in package_paths.keys()[2:]:
        if os.path.exists(path):
            package_paths[package] = path
            with open(os.path.dirname(os.path.realpath(__file__)) + "/packages.pkl", "wb") as file_pickle:
                pickle.dump(package_paths, file_pickle)
        else:
            print("Program '{}' does not exist.")
    else:
        print("No package with name '{}' needed. Packages available in PyPol:")
        for k in package_paths.keys()[2:]:
            print(k)


def _which(commands, name):
    cmd_exe = None
    for cmd in commands:
        if shutil.which(cmd):
            cmd_exe = shutil.which(cmd)
            break

    if cmd_exe:
        cmd_inp = input("Command '{}' found. Confirm [press enter] or write path for {}: ".format(cmd_exe, name))
        if cmd_inp.strip() in ("", "yes", "y", "Yes", "YES"):
            return cmd_exe
        else:
            if os.path.exists(cmd_inp):
                return cmd_inp
    else:
        cmd_inp = input("Command {} not found. Enter path manually or exit: ".format(name))
        if os.path.exists(cmd_inp):
            return cmd_inp
    return False


def check_package_paths():
    path_packages = os.path.dirname(os.path.realpath(__file__)) + "/packages.pkl"
    if os.path.exists(path_packages):
        with open(path_packages, "rb") as packages:
            package_paths = pickle.load(packages)
    else:
        package_paths = {
            "path": os.path.dirname(os.path.realpath(__file__)) + "/",
            "data": os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data/",
            "atomtype": "",
            "gromacs": "",
            "intermol": "",
            "lammps": "",
            "plumed": "",
            "htt_plumed": ""
        }
        with open(os.path.dirname(os.path.realpath(__file__)) + "/packages.pkl", "wb") as file_pickle:
            pickle.dump(package_paths, file_pickle)
        print("saved")

    # Pypol
    if not os.path.exists(package_paths["path"]):
        package_paths["path"] = os.path.dirname(os.path.realpath(__file__)) + "/"
    if not os.path.exists(package_paths["data"]):
        package_paths["data"] = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data/"

    # Gromacs >= 2020
    if not os.path.exists(package_paths["gromacs"]):
        print("Check Gromacs path")
        cmd_gmx = _which(("gmx", "gmx_mpi", "gmx_d", "gmx_mpi_d"), "Gromacs(>2020)")
        if cmd_gmx:
            gmx_version = os.popen('{} --version | grep "GROMACS version" -m 1'.format(cmd_gmx)).read().split()[2][:4]
            if not int(gmx_version) >= 2020:
                print("Error: Gromacs version lower than 2020. "
                      "Update Gromacs at http://manual.gromacs.org/documentation/")
                exit()
        else:
            print("Error: Gromacs package (>2020) not found."
                  "Download Gromacs at http://manual.gromacs.org/documentation/")
            exit()
        package_paths["gromacs"] = cmd_gmx

    # LAMMPS
    if not os.path.exists(package_paths["lammps"]):
        print("Check LAMMPS path")
        cmd_lmp = _which(("lmp", "lmp_mpi", "lmp_serial", "lmp_mac", "lmp_mac_mpi"), "LAMMPS")
        if cmd_lmp:
            package_paths["lammps"] = cmd_lmp
        else:
            print("Error: LAMMPS package not found."
                  "Download LAMMPS at https://lammps.sandia.gov/")
            exit()

    # Atomtype (AmberTools)
    if not os.path.exists(package_paths["atomtype"]):
        print("Check AmberTools folder")
        cmd_atomtype = _which(("atomtype"), "atomtype")
        if cmd_atomtype:
            package_paths["atomtype"] = cmd_atomtype
        else:
            print("Error: 'atomtype' program not found."
                  "Download AmberTools at https://ambermd.org/AmberTools.php")
            exit()

    # Plumed
    if not os.path.exists(package_paths["plumed"]):
        print("Check plumed2.6 path")
        cmd_plumed = _which(("plumed", "plumed2.6", "plumed2.7"), "Plumed")
        if cmd_plumed:
            package_paths["plumed"] = cmd_plumed
        else:
            print("Error: Plumed2 not found."
                  "Download Plumed2 at https://github.com/plumed/plumed2")
            exit()

    if not os.path.exists(package_paths["htt_plumed"]):
        print("Check hack-the-tree branch path")
        cmd_htt_plumed = _which(("htt_plumed"), "Hack-The-Tree")
        if cmd_htt_plumed:
            package_paths["htt_plumed"] = cmd_htt_plumed
        else:
            print("Error: Plumed2 not found."
                  "Download Plumed2 at https://github.com/plumed/plumed2")
            exit()

    # InterMol
    if not os.path.exists(package_paths["intermol"]):
        cmd_intermol = input("Enter path of the InterMol module convert.py: ")
        if cmd_intermol:
            package_paths["intermol"] = cmd_intermol
        else:
            print("Error: InterMol not found."
                  "Download InterMol at https://github.com/shirtsgroup/InterMol")
            exit()

    with open(os.path.dirname(os.path.realpath(__file__)) + "/packages.pkl", "wb") as file_pickle:
        pickle.dump(package_paths, file_pickle)

    return package_paths
