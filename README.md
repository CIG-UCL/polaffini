# Polaffini

This repository contains code for:
 - **POLAFFINI** [1]: a segmentation-based polyaffine initialization for improved non-linear image registration. 

Most of the code is in Python, deep-learning stuffs are based on Tensorflow library, image IO and processing is done using SimpleITK (conversion through nibabel for mgh/mgz files), deep-learning registration uses Voxelmorph [2] core.

# Installation

Installation can be done classically through git clone and installing the dependancies in `requirement.txt`. If you only need POLAFFFINI and do not install deep-learning stuff, you can use it as an independant module and use the requirement file in `requirements`.

Alternatively, one can use a pip command to do everything in one go:
`pip install git+https://github.com/CIG-UCL/polaffini.git`


# A. POLAFFINI
<p align="center">
<img src="imgs/diagram_polaffini.svg" width="85%">
</p>

POLAFFINI is an efficient initialization to improve non-linear registration compared to the usual intensity-based affine pre-alignment (e.g. with FSL FLIRT).\
POLAFFINI uses fine-grain segmentations to estimate a polyaffine transformation which anatomically grounded, fast to compute, and has more dofs than its affine counterpart.

Fine-grained segmentations can be obtained using traditional tools like:
 - `recon-all` from FreeSurfer suite [[website]](https://surfer.nmr.mgh.harvard.edu/)
   
or very quickly using pre-trained deep-learning models like:

 - FastSurfer [[github]](https://github.com/Deep-MI/FastSurfer)[[paper]](https://doi.org/10.1016/j.neuroimage.2020.117012)
 - SynthSeg [`mri_synthseg` in Freesurfer][[paper]](https://doi.org/10.1016/j.media.2023.102789) which is contrast agnostic.

 
## A.1. Small POLAFFINI tutorial
A good way to understand how it works is to go through the following small tutorial: `dwarp_public/scripts/polaffini_example.py`.\
This script uses the data available `dwarp_public/exmaple_data`. Extract and tweak bits to fit your needs.

## A.2. POLAFFINI between 2 subjects

The following script covers most usage, it performs POLAFFINI registration between two subjects.\
It uses the moving and target segmentations to estimate the polyaffine transformation, then applies the transformation to the moving image.
```bash
python <path-to-dwarp_public>/scripts/polaffini_pair.py -m <path-to-moving-image>\
                                                        -ms <path-to-moving-segmentation>\
                                                        -rs <path-to-target-segmentation>\
                                                        -oi <path-to-output-moved-image>
```

## A.3. POLAFFINI of a dataset onto a template

The script `/scripts/polaffini_set2template.py` allows to perform POLAFFINI on a set of subjects as well as various data preparation such as intensity normalization, one-hot encoding of segmentations... It can be typically used to prepare the data to be fed to a deep-learning model during its training.\
See Section B.2.a. for an example.



# Included ressources
  - MNI template: The default MNI template used here is the [ICBM 2009c Nonlinear Symmetric](https://www.mcgill.ca/bic/icbm152-152-nonlinear-atlases-version-2009) version. One can find it, together with its associated DKT segmentation, in `dwarp_public/ref/` with voxel sizes 1 and 2 mm isotropic.

# References
  - [1] **POLAFFINI** [[IPMI 2023 paper]](https://link.springer.com/content/pdf/10.1007/978-3-031-34048-2_47.pdf?pdf=inline%20link).
