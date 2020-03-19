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
os.system("rm {} {} {}".format(file_out_pdb1, file_out_pdb2, file_out_pdb3))
