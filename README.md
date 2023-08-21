# dwarp: learning-based image registration

This repository contains code for:
 - **POLAFFINI** [1]: a segmentation-based polyaffine initialization for improved non-linear image Registration. 
 - **dwarp**: tools for deep learning non-linear image registration to a template.
Code is in Python, deep-learning stuffs are based on Tensorflow library, image io and processing is done using SimpleITK, deep-learning registration uses Voxelmorph [2] core.

# How to use POLAFFINI
## Simple tutorial between 2 subjects
scripts/polaffini_example.py is a basic tutorial for POLAFFNI initialization between 2 subjects whose images can be found in example_data. 

# References
We have several VoxelMorph tutorials:
  - [1] **POLAFFINI** [[IPMI 2023 paper]](https://link.springer.com/content/pdf/10.1007/978-3-031-34048-2_47.pdf?pdf=inline%20link).

# External repositories
  - [2] **Voxelmorph** [[github]](https://github.com/voxelmorph/voxelmorph) - a general purpose library for learning-based tools for alignment/registration, and more generally modelling with deformations. In addition to the code, a list of their papers is available there. Especially, if using **dwarp**, please cite Voxelmorph's TMI 2019 and MICCAI 2018 articles.
