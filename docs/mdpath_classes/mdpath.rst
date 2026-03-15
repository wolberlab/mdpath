MDPath --- MD signal transduction calculation and visualization --- :mod:`mdapth.mdpath`
=====================================================================================================

MDPath is a Python package for calculating signal transduction paths in molecular dynamics (MD) simulations. 
The package uses mutual information to identify connections between residue movements.
Using a graph shortest paths with the highest mutual information are calculated.
Paths are then clustered based on the overlap between them to identify a continuous network throught the analysed protein.
The package also includes functionalitys for the visualization of results.

Release under the MIT License. See LICENSE for details.

This is the main script of MDPath. It is used to run MDPath from the command line.
MDPath can be called from the comadline using 'mdapth' after instalation
Use the -h flag to see the available options.

Functions
----------

.. autofunction:: mdpath.mdpath.main
