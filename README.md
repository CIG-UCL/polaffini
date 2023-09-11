# dwarp

This repository contains code for:
 - **POLAFFINI** [1]: a segmentation-based polyaffine initialization for improved non-linear image Registration. 
 - **dwarp**: tools for deep learning non-linear image registration to a template, and more...

Most of the code is in Python, deep-learning stuffs are based on Tensorflow library, image IO and processing is done using SimpleITK, deep-learning registration uses Voxelmorph [2] core.

# Simple POLAFFINI tutorial between 2 subjects
scripts/polaffini_example.py is a basic tutorial for POLAFFINI initialization between 2 subjects whose images can be found in example_data. 

# Registering a dataset to the MNI template
The default MNI templates in **dwarp** are [ICBM 2009c Nonlinear Symmetric](https://www.mcgill.ca/bic/icbm152-152-nonlinear-atlases-version-2009) versions with voxel sizes 1 and 2 mm isotropic.
The non-linear registration model has been trained on skull-stripped T1-weighted images with the MNI 2 mm as target.

This tutorial requires an MR dataset containing homologous data of 2 types:
 - T1-weighted images, skull-stripped.
 - Segmentations, DKT protocol. Can been obtained using FreeSurfer, FastSurfer, SynthSeg...
   
## Training a new registration model from scratch
This is typically designed to prepare the data for a subsequent registration model training.
### 1. POLAFFINI and data preparation ###

 - Perform POLAFFINI.
 - Prepare the data for training: resizing, intensity normalization, one-hot encoding for segmentations...
    
```bash
# training data
python <path-to-dwarp_public>/scripts/init_polaffini.py -m "<path-to-directory-with-images-for-training>/*"\
                                                        -ms "<path-to-directory-with-segmentations-for-training>/*"\
                                                        -r mni2\
                                                        -o <path-to-output-directory>/train\
                                                        -kpad 5 -os 1 -downf 2 -omit_labs 2 41
# validation data
python <path-to-dwarp_public>/scripts/init_polaffini.py -m "<path-to-directory-with-images-for-validation>/*"\
                                                        -ms "<path-to-directory-with-segmentations-for-validation>/*"\
                                                        -r mni2\
                                                        -o <path-to-output-directory>/val\
                                                        -kpad 5 -os 1 -downf 2 -omit_labs 2 41
```
Use `-h` to display help.\
`-r mni2` indicates that the target template is the MNI with voxel size 2 mm isotropic. You can instead provide the path to a template of your choice (in this case you also need to provide the associated segmentation using `-rs`).\
`-kpad 5` ensures that the output image dimensions are a multiple of 2<sup>5</sup> since we'll train a U-net model with 5 levels of econding / decoding. Adapt this to your model architecture.\
`-os 1` toggles the output of the moved segmentations (in one-hot encoding) so that they can be leveraged during the training of the model.
`-omit_labs 2 41` will omit those labels for POLAFFINI as they are too big (whole left and right white matter) so taking their centroids is a bit meaningless.\
The output directories will be organized as follow:\
&ensp; ├ img - folder containing moved images\
&ensp; ├ seg (if `-os 1`) - folder containing moved segmentations (in one-hot encoding by default)\
&ensp; └ transfo (if `-ot 1`) - folder containing transformations (an affine transformation and a polyaffine one in SVF form)


### 2. Model training ###
```
python <path-to-dwarp_public>/scripts/train.py -o <path-to-output-directory>/model.h5\
                                               -e 1000\
                                               -t <path-to-output-directory>/train\
                                               -v <path-to-output-directory>/val\
                                               -s 1 -l nlcc -ls dice
```
Use `-h` to display help.\
`-s 1` indicates that segmentations are leveraged during the training.\
`-l nlcc` indicates the normalized squared local correlation coefficient is used as image loss.\
`-ls dice` indicates that Dice score is used as segmentation loss.

## POLAFFINI + Non-linear registration for a subject (typical use)
This is 


# References
  - [1] **POLAFFINI** [[IPMI 2023 paper]](https://link.springer.com/content/pdf/10.1007/978-3-031-34048-2_47.pdf?pdf=inline%20link).

# External repositories
  - [2] **Voxelmorph** [[github]](https://github.com/voxelmorph/voxelmorph) - a general purpose library for learning-based tools for alignment/registration, and more generally modelling with deformations. In addition to the code, a list of their papers is available there. Especially, if using **dwarp**, please cite Voxelmorph's TMI 2019 and MICCAI 2018 articles.
