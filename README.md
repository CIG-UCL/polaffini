# dwarp & POLAFFINI

This repository contains code for:
 - **POLAFFINI** [1]: a segmentation-based polyaffine initialization for improved non-linear image Registration. 
 - **dwarp**: tools for deep learning non-linear image registration to a template, and more...

Most of the code is in Python, deep-learning stuffs are based on Tensorflow library, image IO and processing is done using SimpleITK, deep-learning registration uses Voxelmorph [2] core.

# Simple POLAFFINI tutorial between 2 subjects
The `dwarp_public/scripts/polaffini_example.py` script is a basic tutorial for POLAFFINI between 2 subjects. Data for this tutorial can be found in `dwarp_public/exmaple_data`.

# Registering a dataset to the MNI template

This tutorial requires an MR dataset containing homologous data of 2 types:
 - T1-weighted images, skull-stripped.
 - Segmentations, DKT protocol. Can been obtained using FreeSurfer, FastSurfer, SynthSeg...

## Using a pre-trained model
```bash
python ~/tests/ainostics/dwarp/register.py -M <path-to-dwarp-public-directory>/diffeo2mni.h5\
                                           -m <path-to-moving-image>\
                                           -ms <path-to-moving-segmentation>\
                                           -oi <path-to-moved-image>
                                           -os <path-to-moved-segmentation>
                                           -g mni1\
                                           -polaffini 1 -omit_labs 2 41 -downf 2
```
                                           
Use `-h` to display help.\
`diffeo2mni.h5` is a pre-trained model depicted in section Resources. You can instead provide the path to another model trained as depicted in section Training a new registration model from scratch.
`-g mni1` indicates that the geometry (orientation + image dimensions + voxel size) image used for resampling is the MNI template with voxel size 1 mm isotropic.\
`-os 1` toggles the output of the moved segmentations (in one-hot encoding) so that they can be leveraged during the training of the model.
`-polaffini 1` indicates the POLAFFINI is performed.
`-omit_labs 2 41` will omit those labels for POLAFFINI as they are too big (whole left and right white matter) so taking their centroids is a bit meaningless.\

   
## Training a new registration model from scratch

### 1. POLAFFINI and data preparation
The `dwarp_public/script/init_polaffini.py` script is designed to carry out 2 tasks:
 - Perform POLAFFINI.
 - Prepare the data for training: resizing, intensity normalization, one-hot encoding for segmentations...
    
```bash
# training data
python <path-to-dwarp_public>/scripts/init_polaffini.py -m "<path-to-training-images-directory>/*"\
                                                        -ms "<path-to-training-segmentations-directory>/*"\
                                                        -r mni2\
                                                        -o <path-to-output-directory>/train\
                                                        -kpad 5 -os 1 -downf 2 -omit_labs 2 41
# validation data
python <path-to-dwarp_public>/scripts/init_polaffini.py -m "<path-to-validation-images-directory>/*"\
                                                        -ms "<path-to-validation-segmentations-directory>/*"\
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

# Ressources
  - MNI template: The default MNI template used here is the [ICBM 2009c Nonlinear Symmetric](https://www.mcgill.ca/bic/icbm152-152-nonlinear-atlases-version-2009) version. One can find it, together with its associated DKT segmentation, in `dwarp_public/ref/` with voxel sizes 1 and 2 mm isotropic.
  - Pre-trained model: `diffeo2mni.h5` is a pre-trained model for non-linear registration to the MNI template. The training procedure is the one depicted in section Training a new registration model from scratch. It has been trained on skull-strpped T1-weighted images from 100 subjects (20 UKBiobank, 20 IXI, 60 ADNI (20 HC, 20 MCI, 20 AD)), with 25 subjects for validation (5 UKBiobank, 5 IXI, 15 ADNI (5 HC, 5 MCI, 5 AD)). The MNI version is the one depicted above with voxel size 2 mm istropic. The image loss was the normalized squared local correlation coefficient (nlcc), regularization loss had a weight of 1, segmentation was leveraged during training using a Dice loss with weight 0.01.
    
# References
  - [1] **POLAFFINI** [[IPMI 2023 paper]](https://link.springer.com/content/pdf/10.1007/978-3-031-34048-2_47.pdf?pdf=inline%20link).

# External repositories
  - [2] **Voxelmorph** [[github]](https://github.com/voxelmorph/voxelmorph) - a general purpose library for learning-based tools for alignment/registration, and more generally modelling with deformations. In addition to the code, a list of their papers is available there. Especially, if using **dwarp**, please cite Voxelmorph's TMI 2019 and MICCAI 2018 articles.
