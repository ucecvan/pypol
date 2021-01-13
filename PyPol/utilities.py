# General
def create(path, arg_type, backup=True):
    """
    Generate a new directory or a new file.
    TODO Rewrite in a more compact way. Use os.remove instead of os.system(rm ...)
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

    basename = os.path.basename(target_path)
    target_identifier, extension = os.path.splitext(basename)
    extension = extension[1:]
    return target_identifier, extension


def get_list(elements):
    """
    Return a list from set, tuple or single item
    :param elements: list, tuple, set, single obj
    :return:
    """
    if isinstance(elements, list):
        return elements
    if isinstance(elements, set) or isinstance(elements, tuple):
        return list(elements)
    else:
        return [elements]


def check_attributes(list_crystals: list, attributes: dict):
    """
    Given a list of crystals, check if attribute is present in each of them and discard those that do not have it.
    :param list_crystals: list of Crystal objects
    :param attributes: dict of attributes
    :return: List of Crystals that have the specified attributes
    """
    new_list = []
    for crystal in list_crystals:
        if attributes.items() <= crystal._attributes.items():
            new_list.append(crystal)
    return new_list


def get_list_crystals(scrystals, crystals, attributes=None, _include_melted=False):
    """
    Select a subgroup of crystal from a list. This can be done by specifying the state of the crystal, a list of crystal
    objects or identifier and by including crystal attributes
    :param scrystals: Simulation Crystals, list of crystals from a specific simulation object.
    :param crystals: Crystal state to select. It can be "all", "incomplete", "centers"
    :param attributes: Crystal attributes to match for selecting the output crystal list
    :param _include_melted: Together with crystals="all" include also melted structures in the output crystal list.
    :return: list of Crystal objects
    """
    if attributes is None:
        attributes = {}
    from PyPol.crystals import Crystal

    list_crystals = list()
    if crystals == "incomplete":
        for sc in scrystals:
            if sc._state == "incomplete":
                list_crystals.append(sc)
    elif crystals == "all" and _include_melted:
        for sc in scrystals:
            list_crystals.append(sc)
    elif crystals == "all":
        for sc in scrystals:
            if sc._state != "melted":
                list_crystals.append(sc)
    elif crystals == "centers":
        for sc in scrystals:
            if sc._state != "melted" and sc._name == sc._state:
                list_crystals.append(sc)
    else:
        crystals = get_list(crystals)
        if isinstance(crystals[0], str):
            for sc in scrystals:
                if sc._name in crystals:
                    list_crystals.append(sc)
        elif isinstance(crystals[0], Crystal):
            for sc in scrystals:
                if sc in crystals:
                    list_crystals.append(sc)
        else:
            print("Something went wrong in crystal selection, please check that simulation is completed")
            exit()
    if attributes:
        list_crystals = check_attributes(list_crystals, attributes)
    return list_crystals


# Cell parameters - Box matrix interconversion
def cell2box(cell):
    """
    Convert cell parameters to 3x3 box matrix.
    :param cell: Iterable obj with 6 cell parameters, [a, b, c, alpha, beta, gamma]
    :return: box matrix
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
    Convert box matrix to cell parameters.
    :param box: 3x3 box matrix
    :return: list with the 6 cell parameters [a, b, c, alpha, beta, gamma]
    """
    import numpy as np
    cell = [None, None, None, None, None, None]
    cell[0] = box[0, 0]
    cell[1] = np.sqrt(box[1, 1] ** 2 + box[0, 1] ** 2)
    cell[2] = np.sqrt(box[2, 2] ** 2 + box[1, 2] ** 2 + box[0, 2] ** 2)
    cell[3] = np.rad2deg(np.arccos((box[0, 1] * box[0, 2] + box[1, 1] * box[1, 2]) / (cell[1] * cell[2])))
    cell[4] = np.rad2deg(np.arccos(box[0, 2] / cell[2]))
    cell[5] = np.rad2deg(np.arccos(box[0, 1] / cell[1]))
    return cell


# Simulation box variations
# best_b and best_c select the non-primitive cell that minimize off-diagonal elements of the box matrix.
def best_c(box, max_replica, toll=0.08):
    """
    Return c vector of a non-primitive cell with minimum angle with respect to z axis.
    :param box:
    :param max_replica:
    :param toll:
    :return:
    """
    import numpy as np
    new_c = box[:, 2]
    distance_min = np.linalg.norm(new_c[:2])
    replica_c = 1
    for i in sorted([i for i in range(-max_replica, max_replica+1) if i != 0], key=abs):
        for j in sorted([j for j in range(-max_replica, max_replica+1) if i != 0], key=abs):
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
    """
    Return b vector of a non-primitive cell with minimum angle with respect to y axis.
    :param box:
    :param max_replica:
    :param toll:
    :return:
    """
    import numpy as np
    new_b = box[:, 1]
    distance_min = np.absolute(new_b[0])
    replica_b = 1
    for i in sorted([i for i in range(-max_replica, max_replica + 1) if i != 0], key=abs):
        for j in range(1, max_replica + 1):
            vy = i * box[:, 0] + j * box[:, 1]
            if np.absolute(vy[0]) < distance_min:
                distance_min = vy[0]
                new_b = vy
                replica_b = j
            if np.absolute(vy[0] / box[0, 0]) <= toll:
                return new_b, replica_b
    return new_b, replica_b


def translate_molecule(molecule, box):
    """
    Translate the molecule center of mass (COM) inside the simulation box.
    :param molecule: molecule center of mass
    :param box: box matrix
    :return: Molecule object
    """
    import numpy as np

    def translate_atoms(target, vector):
        for atom in target._atoms:
            atom._coordinates += vector
        target._calculate_centroid()
        return target
    a = np.round(np.dot(molecule.centroid, np.linalg.inv(box.T)), 3)
    for i in range(3):
        if a[i] < 0.0:
            molecule = translate_atoms(molecule, (int(a[i])-1) * -box[:, i])
        if a[i] == 0.0:
            molecule = translate_atoms(molecule, box[:, i])
        elif a[i] > 1.0:
            molecule = translate_atoms(molecule, int(a[i]) * -box[:, i])
    return molecule


def point_in_box(point, box):
    """
    Check if point coordinates are inside the box.
    :param point: point coordinates
    :param box: 3x3 Matrix
    :return: bool
    """
    import numpy as np
    a = np.dot(point, np.linalg.inv(box.T))
    if (a >= 0.).all() and (a <= 1.).all():
        return True
    else:
        return False
