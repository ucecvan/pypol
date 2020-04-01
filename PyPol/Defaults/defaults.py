# PyPol
import datetime

last_update = 1585731216.638416  # import datetime; print(datetime.datetime.now().timestamp())
pypol_info = {
    "path": r"/home/nicholas/Programs/PyPol/PyPol/",
    "version": datetime.datetime.fromtimestamp(last_update).strftime("%Y.%m.%d")
}

# Default command lines
package_paths = {
    "run_csd_python_api": r'/nas/shared/programs/CCDC/Python_API_2018/run_csd_python_api',
    "atomtype":           r'/home/nicholas/anaconda3/bin/atomtype',
    "gromacs":            r"/usr/local/gromacs/bin/gmx_d",
    "intermol":           r"python3 /home/nicholas/Programs/InterMol/Intermol-2019/intermol/convert.py",
    "lammps":             r"/home/nicholas/.local/bin//lmp",
    "plumed":             r"/home/nicholas/.local/bin/plumed2.6",
    "htt_plumed":         r"/home/nicholas/.local/bin/htt_plumed",
}

# Atom types that can be switched by antechamber, especially from experimental data. They are considered equivalent
# only during the index assignation in the Method.generate_input module but not during the simulation.
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
}
