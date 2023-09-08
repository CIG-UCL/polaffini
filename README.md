# dwarp

This repository contains code for:
 - **POLAFFINI** [1]: a segmentation-based polyaffine initialization for improved non-linear image Registration. 
 - **dwarp**: tools for deep learning non-linear image registration to a template, and more...

Most of the code is in Python, deep-learning stuffs are based on Tensorflow library, image IO and processing is done using SimpleITK, deep-learning registration uses Voxelmorph [2] core.

# Simple POLAFFINI tutorial between 2 subjects
scripts/polaffini_example.py is a basic tutorial for POLAFFINI initialization between 2 subjects whose images can be found in example_data. 

# Registering a dataset to the MNI template
The default MNI templates in **dwarp** are [ICBM 2009c Nonlinear Symmetric](https://www.mcgill.ca/bic/icbm152-152-nonlinear-atlases-version-2009) versions with voxel sizes 1 and 2 mm isotropic.
The non-linear registration model has been trained on skull-stripped T1-weighted images.

This tutorial requires an MR dataset containing homologous data of 2 types:
 - T1-weighted images, skull-stripped.
 - Segmentations, DKT protocol. Can been obtained using FreeSurfer, FastSurfer, SynthSeg...
   
## POLAFFINI

## Non-linear registration


# References
  - [1] **POLAFFINI** [[IPMI 2023 paper]](https://link.springer.com/content/pdf/10.1007/978-3-031-34048-2_47.pdf?pdf=inline%20link).

# External repositories
  - [2] **Voxelmorph** [[github]](https://github.com/voxelmorph/voxelmorph) - a general purpose library for learning-based tools for alignment/registration, and more generally modelling with deformations. In addition to the code, a list of their papers is available there. Especially, if using **dwarp**, please cite Voxelmorph's TMI 2019 and MICCAI 2018 articles.
