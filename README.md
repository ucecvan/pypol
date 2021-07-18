Pypol 2021.04
===============================================================================

This repository includes a set of tools developed to simulate and quickly analyse the behaviour of a large set of structures at finite-temperature and pressure.
In particular, it has been developed to reduce the overprediction problem of computationl Crystal Structure Prediction (CSP) methods of molecular crystals.
The program organizes the different structures in separate folders and automatically prepares the input files for the Gromacs MD package and the Plumed library.
A set of tools have also been implemented to analyse the resulting trajectories and identify crystal structures that melt or transform into a different form.

The method is described in the paper:

__Systematic Finite-Temperature Reduction of Crystal Energy Landscapes__,
Nicholas F. Francia, Louise S. Price, Jonas Nyman, Sarah L. Price, and Matteo Salvalaglio
Crystal Growth & Design 2020 20 (10), 6847-6862
DOI: 10.1021/acs.cgd.0c00918

A more in-depth description of the Python modules available and guidelines to PyPol installation are available in doc/Pypol_Manual.pdf.
