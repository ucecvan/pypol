import os


def gaff(path_coord, path_output, res_name="UNK", generate_charges='bcc'):
    """
    Starting from a mol2 file (other file formats available) containing ONE molecule it generates a GAFF forcefield
    and perform non-periodical energy minimization of the isolated molecule.
    :param path_coord: path of the file containing the coordinates of the atoms
    :param path_output: Directory output path in which the topology will be saved
    :param res_name: Residue name, max 3 letters
    :param generate_charges: Charge generation method used by antechamber
    :return:
    """
    from PyPol import check_package_paths
    package_paths = check_package_paths()

    path_ambertools = os.path.dirname(package_paths["atomtype"])
    path_acpype = input("Enter path for acpype.py program: ")  # https://pypi.org/project/acpype/
    path_tleap = path_ambertools + "/tleap"
    path_parmchk = path_ambertools + "/parmchk2"
    path_antechamber = path_ambertools + "/antechamber"
    path_gromacs = package_paths["gromacs"]

    atomtype = 'gaff'

    # Antechamber extensions and charge methods availability
    available_extensions = ["ac", "mol2", "pdb", "mpdb", "prepi", "prepc", "gzmat", "gcrt", "mopint",
                            "mopcrt", "gout", "mopout", "alc", "csd", "mdl", "hin", "rst"]

    available_charge_methods = ["resp", "cm2", "mul", "rc", "bcc", "esp", "gas", "wc"]

    # Errors
    if not os.path.exists(path_coord):
        print("Error: File '{}' does not exits!".format(path_coord))
        exit()

    input_file_type = None
    for ext in available_extensions:
        if path_coord.endswith(ext):
            input_file_type = ext
            break

    if not input_file_type:
        print("Error: File '{}' does not have one of the required extension!      \n"
              "List of the File Formats:                                          \n"
              "     file format type  abbre. index | file format type abbre. index\n"
              "     --------------------------------------------------------------\n"
              "     Antechamber        ac       1  | Sybyl Mol2         mol2    2 \n"
              "     PDB                pdb      3  | Modified PDB       mpdb    4 \n"
              "     AMBER PREP (int)   prepi    5  | AMBER PREP (car)   prepc   6 \n"
              "     Gaussian Z-Matrix  gzmat    7  | Gaussian Cartesian gcrt    8 \n"
              "     Mopac Internal     mopint   9  | Mopac Cartesian    mopcrt 10 \n"
              "     Gaussian Output    gout    11  | Mopac Output       mopout 12 \n"
              "     Alchemy            alc     13  | CSD                csd    14 \n"
              "     MDL                mdl     15  | Hyper              hin    16 \n"
              "     AMBER Restart      rst     17 ".format(os.path.basename(path_coord)))
        exit()

    if generate_charges not in available_charge_methods:
        print("Error: '{}' method unknown!                                          \n"
              "List of the Charge Methods:                                          \n"
              "     charge method    abbre.  index | charge method      abbre. index\n"
              "     ----------------------------------------------------------------\n"
              "     RESP             resp     1    |  AM1-BCC            bcc     2  \n"
              "     CM2              cm2      3    |  ESP (Kollman)      esp     4  \n"
              "     Mulliken         mul      5    |  Gasteiger          gas     6  \n"
              "     Read in Charge   rc       7    |  Write out charge   wc      8  \n"
              "".format(os.path.basename(generate_charges)))

    if not res_name:
        print("Error: Residue name parameter missing!")
        exit()

    if len(res_name) > 3:
        print("Error: Residue name's length must be 3 or lower!")
        exit()

    # Define new variables and change working directory
    path_wd = path_output
    os.chdir(path_wd)
    res_name = res_name.upper()

    # Generate MOL2 file with charges
    print("Generate MOL2 molecule file")
    os.system(
        path_antechamber + " -i {0} -fi {4} -o {1}/{2}.mol2 -fo mol2 -c {3} -rn {2} -pf y -at {5} -nc 0".format(
            path_coord,
            path_wd,
            res_name,
            generate_charges,
            input_file_type,
            atomtype))

    # Create a compound library to generate topologies
    file_lib = open(path_wd + "/lib.leap", "w")
    file_lib.write("{0}=loadmol2 {0}.mol2\n"
                   "saveoff {0} {0}.lib\n"
                   "savepdb {0} {0}_leap.pdb\n"
                   "quit".format(res_name))
    file_lib.close()
    os.system(path_tleap + " -f lib.leap")
    os.system(path_parmchk + " -i {0}.mol2 -f mol2 -o {0}.frcmod".format(res_name))

    # Check output (atom types such as SO have no parameters yet)
    skip_structure = False
    if os.path.exists(path_wd + "/{}.frcmod".format(res_name)):
        file_frcmod = open(path_wd + "/{}.frcmod".format(res_name), "r")
        for line in file_frcmod:
            if "ATTN, need revision" in line:
                skip_structure = True
                break
        file_frcmod.close()
    else:
        skip_structure = True

    if skip_structure:
        print("Error: Some of the atoms involved have not been parametrized yet.")
        return

    # Write topology and coordinates in Amber format
    file_lib = open(path_wd + "/parm.leap", "w")
    file_lib.write("source leaprc.gaff\n"
                   "loadamberparams {0}.frcmod\n"
                   "loadoff {0}.lib\n"
                   "a=loadpdb {0}_leap.pdb\n"
                   "saveamberparm a {0}.prmtop {0}.inpcrd\n"
                   "quit".format(res_name))
    file_lib.close()
    os.system(path_tleap + " -f parm.leap")

    # Check topology formation
    if not os.path.exists(path_wd + "/{0}.inpcrd".format(res_name)):
        print("Error: Something went wrong during the topology generation")
        return

    # Convert topology and coordinates to Gromacs format
    os.system("python " + path_acpype + " -p {0}.prmtop -x {0}.inpcrd -a 'gaff'".format(res_name))

    # Generate .itp and .top files from acpype output
    file_acpype = open(path_wd + "/{}_GMX.top".format(res_name), "r")
    file_top = open(path_wd + "/topol.top", "w")
    file_top.write('; File .top created by acpype\n\n')
    file_itp = open(path_wd + "/{}.itp".format(res_name), "w")
    file_itp.write("; File .top created by acpype\n\n")
    write_to_file = False
    for line in file_acpype:

        if line.rstrip().startswith(("[ defaults ]", "[ system ]", "[ molecules ]")):
            write_to_file = False
            if line.rstrip().startswith("[ defaults ]"):
                file_top.write(line)
                file_top.write(next(file_acpype))
                file_top.write(next(file_acpype))
                file_top.write('#include "{0}.itp\n\n'.format(res_name))
            elif line.rstrip().startswith("[ system ]"):
                file_top.write(line)
                file_top.write(next(file_acpype))
                file_top.write("\n")
            elif line.rstrip().startswith("[ molecules ]"):
                file_top.write(line)
                file_top.write(next(file_acpype))
                file_top.write(next(file_acpype))
                file_top.write("\n")

        elif line.rstrip().startswith(("[ moleculetype ]", "[ atomtypes ]", "[ atoms ]", "[ bonds ]", "[ pairs ]", "[ angles ]", "[ dihedrals ]")):
            write_to_file = True

        if write_to_file:
            file_itp.write(line)
    file_itp.close()
    file_acpype.close()

    os.rename(path_wd + "/{}_GMX.gro".format(res_name), path_wd + "/{}.gro".format(res_name))

    # Create directory for intermediate files
    path_tmp = path_wd + "/files"
    os.mkdir(path_tmp)
    for file_name in os.listdir(path_wd):
        if not file_name.endswith((".itp", ".gro", ".top", ".py")):
            if file_name != "files":
                os.rename(path_wd + "/" + file_name, path_tmp + "/" + file_name)

    os.rename(path_wd + "/{}_GMX.top".format(res_name), path_wd + "/files/{}_GMX.top".format(res_name))

    # Create .mdp file for energy minimization
    file_em = open(path_wd + "/em.mdp", "w")
    file_em.write("integrator      = steep   \n"
                  "nsteps          = 100000  \n"
                  "pbc             = no      \n"
                  "emtol           = 1e-12   \n"
                  "emstep          = 0.01    \n"
                  "cutoff-scheme   = Verlet  \n"
                  "comm-mode       = Angular \n")
    file_em.close()

    os.system(path_gromacs + " grompp -f em.mdp -c {0}.gro -p topol.top -o em".format(res_name))
    os.system(path_gromacs + " mdrun -v -deffnm em")


def gen_unit_cells(path_structures, path_output):
    """
    TODO: Delete, CSD Python API updated to python3
    Add a new set of structures in the project_folder/Input/Sets/Set_name directory.
    :param path_structures:
    :param path_output:
    :return:
    """
    from PyPol.utilities import get_identifier

    run_csd_python_api = input("Enter csd_python_api interpreter path: ")

    if not path_structures.endswith("/"):
        path_structures += "/"

    available_file_formats_csd = ("aser", "cif", "csdsql", "csdsqlx", "identifiers", "mariadb", "mol", "mol2",
                                  "res", "sdf", "sqlite", "sqlmol2", "pdb")

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
        if extension not in available_file_formats_csd:
            print("Ignore structure '{}': unknown file format".format(item))
            continue

        path_id = path_output + id_name + ".pdb"

        print("Importing structure '{}'".format(os.path.basename(path_id)))
        path_converter = path_output + "/converter_csd.py"
        file_default = """
# Input file for the unit cell generation. The CSD Python API uses Python2.7 and different input is necessary.
# Different files are created due to inaccuracies in the I/O CSD modules.
import ccdc.io as io
import os

path_id = "PATH_TO_FILE"
identifier_basename = os.path.splitext(os.path.basename(path_id))[0]
path_id_dir = os.path.dirname(path_id) + "/"
file_out_pdb1 = path_id_dir + identifier_basename + "_1.pdb"
file_out_pdb2 = path_id_dir + identifier_basename + "_2.pdb"
file_out_pdb3 = path_id_dir + identifier_basename + "_3.pdb"
file_out_pdb = path_id_dir + "pc.pdb"

with io.CrystalReader(path_id) as cry_reader:
    file2pdb_polymorph = cry_reader[0]
    file2pdb_packed = file2pdb_polymorph.packing(box_dimensions=((0, 0, 0), (1, 1, 1)),
                                                 inclusion="CentroidIncluded")

with io.CrystalWriter(file_out_pdb1) as cry_writer:
    cry_writer.write(file2pdb_polymorph)

with io.CrystalWriter(file_out_pdb2) as cry_writer:
    cry_writer.write(file2pdb_packed)

with io.CrystalReader(file_out_pdb1) as cry_reader:
    out_pdb_1 = cry_reader[0]
    a_axis = out_pdb_1.cell_lengths[0]
    b_axis = out_pdb_1.cell_lengths[1]
    c_axis = out_pdb_1.cell_lengths[2]
    a_ang = out_pdb_1.cell_angles[0]
    b_ang = out_pdb_1.cell_angles[1]
    c_ang = out_pdb_1.cell_angles[2]
spacegroup_symbol = "P1"
new_line_pdb = "CRYST1{:>9.4f}{:>9.4f}{:>9.4f}{:>7.2f}{:>7.2f}{:>7.2f} {:>9}" \
               "".format(a_axis, b_axis, c_axis, a_ang, b_ang, c_ang, spacegroup_symbol)
out_pdb_1_file = open(file_out_pdb1, "r")
out_pdb_1_list = list()
for line in out_pdb_1_file:
    if "SCALE" in line:
        out_pdb_1_list.append(line.rstrip())
out_pdb_1_file.close()

out_pdb_2_file = open(file_out_pdb2, "r")
new_file = open(file_out_pdb3, "w")
a = 0
for line in out_pdb_2_file:
    if "CRYST" in line:
        line = new_line_pdb + "\n"
    if "SCALE" in line:
        line = out_pdb_1_list[a] + "\n"
        a += 1
    new_file.write(line)
out_pdb_1_file.close()
new_file.close()

with io.MoleculeReader(file_out_pdb3) as file2pdb_mol_reader:
    file2pdb_polymorph = file2pdb_mol_reader[0]
    # file2pdb_polymorph.normalise_labels()

with io.MoleculeWriter(file_out_pdb) as mol_writer:
    mol_writer.write(file2pdb_polymorph)

os.chdir(path_id_dir)
os.system("rm {} {} {}".format(file_out_pdb1, file_out_pdb2, file_out_pdb3))"""

        file_converter = open(path_converter, "w")
        file_converter.write(file_default.replace("PATH_TO_FILE", path_id))
        file_converter.close()

        os.system(run_csd_python_api + " < " + path_converter)
