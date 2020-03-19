# General
def create(path, arg_type, backup=True):
    """
    Error: Rewrite in a more compact way. Use os.remove instead of os.system(rm ...)
    Generate a new directory or a new file.
    :param path: Path of the directory/file to generate
    :param arg_type: Is it a file or directory?
    :param backup: If the directory/file already exists, create a backup directory/file
    :return:
    """
    import os
    if backup:
        if arg_type == 'dir':
            path = os.path.relpath(path)
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                print("Directory '{}' already exists!".format(path))
                i = 0
                while True:
                    if os.path.exists(path + ".bkp." + str(i)):
                        i += 1
                        continue
                    else:
                        os.system("mv {0} {0}.bkp.{1}".format(path, i))
                        os.makedirs(path)
                        break
        elif arg_type == 'file':
            if not os.path.exists(path):
                os.mknod(path)
            else:
                print("File '{}' already exists!".format(path))
                i = 0
                while True:
                    if os.path.exists(path + ".bkp." + str(i)):
                        i += 1
                        continue
                    else:
                        os.system("mv {0} {0}.bkp.{1}".format(path, str(i)))
                        os.mknod(path)
                        break
        else:
            print("Only 'dir' and 'file' are available as arg_type")
    else:
        if arg_type == 'dir':
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                os.system("rm -r " + path)
                os.makedirs(path)
        elif arg_type == 'file':
            if not os.path.exists(path):
                os.mknod(path)
            else:
                os.system("rm " + path)
                os.makedirs(path)
        else:
            print("Only 'dir' and 'file' are available as arg_type")


def get_identifier(target_path):
    """
    Get the basename of a specific structure
    :param target_path: complete path of the structure
    :return: identifier name
    """
    import os
    target_identifier = ""
    available_file_formats = (".aser", ".cif", ".csdsql", ".csdsqlx", ".identifiers", ".mariadb", ".mol", ".mol2",
                              ".res", ".sdf", ".sqlite", ".sqlmol2", ".pdb")
    basename = os.path.basename(target_path)
    for suffix in available_file_formats:
        if basename.endswith(suffix):
            target_identifier = basename[:-len(suffix)]
    return target_identifier


def get_list(elements):
    """

    :param elements:
    :return:
    """
    if isinstance(elements, list):
        return elements
    if isinstance(elements, set) or isinstance(elements, tuple):
        return list(elements)
    else:
        return [elements]


# Cell parameters - Box matrix interconversion
def cell2box(cell):
    """

    :param cell:
    :return:
    """
    import numpy as np
    box = np.full((3, 3), 0.)
    box[0, 0] = cell[0]
    box[0, 1] = cell[1] * np.cos(np.radians(cell[5]))
    box[0, 2] = cell[2] * np.cos(np.radians(cell[4]))
    box[1, 1] = np.sqrt(cell[1] ** 2 - box[0, 1] ** 2)
    box[1, 2] = (cell[1] * cell[2] * np.cos(np.radians(cell[3])) - box[0, 1] * box[0, 2]) / box[1, 1]
    box[2, 2] = np.sqrt(cell[2] ** 2 - box[0, 2] ** 2 - box[1, 2] ** 2)
    return box


def box2cell(box):
    """

    :param box:
    :return:
    """
    import numpy as np
    cell = [None, None, None, None, None, None]
    cell[0] = box[0, 0]
    cell[1] = np.sqrt(box[1, 1] ** 2 + box[0, 1] ** 2)
    cell[2] = np.sqrt(box[2, 2] ** 2 + box[1, 2] ** 2 + box[0, 2] ** 2)
    cell[3] = np.rad2deg(np.arccos(box[0, 1] * box[0, 2] + box[1, 1] * box[1, 2] / (cell[1] * cell[2])))
    cell[4] = np.rad2deg(np.arccos(box[0, 2] / cell[2]))
    cell[5] = np.rad2deg(np.arccos(box[0, 1] / cell[1]))
    return cell


# Simulation box variations
def best_c(box, max_replica, toll=0.08):
    """

    :param box:
    :param max_replica:
    :param toll:
    :return:
    """
    import numpy as np
    new_c = box[:, 2]
    distance_min = np.linalg.norm(new_c[:2])
    replica_c = 1
    for i in sorted([i for i in range(-max_replica, max_replica) if i != 0], key=abs):
        for j in sorted([j for j in range(-max_replica, max_replica) if i != 0], key=abs):
            for k in range(1, max_replica):
                vz = i * box[:, 0] + j * box[:, 1] + k * box[:, 2]
                if np.linalg.norm(vz[:2]) < distance_min:
                    distance_min = np.linalg.norm(vz[:2])
                    new_c = vz
                    replica_c = k
                if np.absolute(vz[0] / box[0, 0]) <= toll and np.absolute(vz[1] / box[1, 1]) <= toll:
                    return new_c, replica_c
    return new_c, replica_c


def best_b(box, max_replica, toll=0.08):
    import numpy as np
    new_b = box[:, 1]
    distance_min = np.absolute(new_b[0])
    replica_b = 1
    for i in sorted([i for i in range(-max_replica, max_replica) if i != 0], key=abs):
        for j in range(1, max_replica):
            vy = i * box[:, 0] + j * box[:, 1]
            if np.absolute(vy[0]) < distance_min:
                distance_min = vy[0]
                new_b = vy
                replica_b = j
            if np.absolute(vy[0] / box[0, 0]) <= toll:
                return new_b, replica_b
    return new_b, replica_b


def translate_molecule(molecule, box):
    import numpy as np

    point = molecule.centroid
    a = box[:, 0]
    b = box[:, 1]
    c = box[:, 2]
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    norm_c = np.linalg.norm(c)

    def cell2planes(vect):
        ba = vect[:, 0]
        bb = vect[:, 1]
        bc = vect[:, 2]
        bplane1 = np.cross(ba, bb)
        bplane2 = np.cross(ba, bc)
        bplane3 = np.cross(bb, bc)
        return bplane1, bplane2, bplane3

    def point_projection_in_plane(pl1, pl2, pl3, pnt):
        xp, yp, zp = pnt
        a1, b1, c1 = pl1
        a2, b2, c2 = pl2
        a3, b3, c3 = pl3
        nxp = -(b1 * yp + c1 * zp) / a1
        nyp = -(a2 * xp + c2 * zp) / b2
        nzp = -(b3 * yp + a3 * xp) / c3
        pip1 = np.array([nxp, yp, zp])
        pip2 = np.array([xp, nyp, zp])
        pip3 = np.array([xp, yp, nzp])
        return pip1, pip2, pip3

    plane1, plane2, plane3 = cell2planes(box)
    int1, int2, int3 = point_projection_in_plane(plane3, plane2, plane1, point)

    diff1 = point[0] - int1[0]
    diff2 = point[1] - int2[1]
    diff3 = point[2] - int3[2]

    def translate_atoms(target, vector):
        for atom in target.atoms:
            atom.coordinates += vector
        target.calculate_centroid()
        return target

    if diff1 > norm_a:
        n = int(-diff1 / norm_a) * a
        molecule = translate_atoms(molecule, n)
    elif diff1 <= 0:
        n = (int(-diff1 / norm_a) + 1) * a
        molecule = translate_atoms(molecule, n)

    if diff2 > norm_b:
        n = int(-diff2 / norm_b) * b
        molecule = translate_atoms(molecule, n)
    elif diff2 <= 0:
        n = (int(-diff2 / norm_b) + 1) * b
        molecule = translate_atoms(molecule, n)

    if diff3 > norm_c:
        n = int(-diff3 / norm_c) * c
        molecule = translate_atoms(molecule, n)
    elif diff3 <= 0:
        n = (int(-diff3 / norm_c) + 1) * c
        molecule = translate_atoms(molecule, n)

    return molecule


def point_in_box(point, cell):
    import numpy as np

    a = cell[:, 0]
    b = cell[:, 1]
    c = cell[:, 2]
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    norm_c = np.linalg.norm(c)

    def cell2planes(vect):
        ba = vect[:, 0]
        bb = vect[:, 1]
        bc = vect[:, 2]
        bplane1 = np.cross(ba, bb)
        bplane2 = np.cross(ba, bc)
        bplane3 = np.cross(bb, bc)
        return bplane1, bplane2, bplane3

    def point_projection_in_plane(pl1, pl2, pl3, pnt):
        xp, yp, zp = pnt
        a1, b1, c1 = pl1
        a2, b2, c2 = pl2
        a3, b3, c3 = pl3
        nxp = -(b1 * yp + c1 * zp) / a1
        nyp = -(a2 * xp + c2 * zp) / b2
        nzp = -(b3 * yp + a3 * xp) / c3
        pip1 = np.array([nxp, yp, zp])
        pip2 = np.array([xp, nyp, zp])
        pip3 = np.array([xp, yp, nzp])
        return pip1, pip2, pip3

    plane1, plane2, plane3 = cell2planes(cell)
    int1, int2, int3 = point_projection_in_plane(plane3, plane2, plane1, point)

    diff1 = point[0] - int1[0]
    diff2 = point[1] - int2[1]
    diff3 = point[2] - int3[2]

    if diff1 > norm_a or diff1 <= 0 or diff2 > norm_b or diff2 <= 0 or diff3 > norm_c or diff3 <= 0:
        return False
    else:
        return True
