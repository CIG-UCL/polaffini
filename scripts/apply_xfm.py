#!/usr/bin/env python3
import os
import pathlib
import SimpleITK as sitk
import polaffini.utils as utils


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mov-img', type=pathlib.Path, required=True, help="Path to image to transform")
    parser.add_argument('-r', '--ref-img', type=pathlib.Path, required=True, help="Path to reference image")
    parser.add_argument('-o', '--out-img', type=pathlib.Path, required=True, help="Path to output moved image")
    parser.add_argument('-t', '--transforms', nargs='+', type=pathlib.Path, required=True, help="List of transform paths to apply. Will be applied in order from left to right")
    parser.add_argument('--is_log_euclid', action='store_true', help='Flag to denote that tranform image files are alread log-euclidean')
    parser.add_argument('-alpha', '--alpha', type=float, required=False, default=1, help='Position of the overall transformation on the diffeomorphic path from identity to the transfo from moving to reference (e.g. use 0.5 for half-way registration). Default: 1.')

    args = parser.parse_args()

    transform_files = args.transforms
    transform_extensions = [''.join(x.suffixes) for x in transform_files]
    print(transform_extensions)

    transforms = []
    for xfm, ext in zip(transform_files, transform_extensions):
        if ext == ".nii.gz":
            transforms.append(sitk.DisplacementFieldTransform(utils.imageIO(xfm).read()))
        else:
            transforms.append(sitk.ReadTransform(xfm))

    print(transforms)

    transform = sitk.CompositeTransform(transforms)

    mov_img = utils.imageIO(args.mov_img).read()
    ref_img = utils.imageIO(args.ref_img).read()

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)

    mov_img = resampler.Execute(mov_img)

    utils.imageIO(args.out_img).write(mov_img)
