import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # remove this for more tensorflow logs
import sys
maindir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(maindir)
import numpy as np
import SimpleITK as sitk
import tensorflow as tf      
tf.config.experimental.set_visible_devices([], 'GPU')    
import polaffini.utils as utils
import dwarp
import polaffini.polaffini as polaffini
import argparse
import pathlib
from typing import Optional

import faulthandler
with open('/scratch0/NOT_BACKED_UP/alegouhy/tmp/fault.log', 'w') as f:
    faulthandler.enable(file=f)
    
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

parser = argparse.ArgumentParser(description="")
# model
parser.add_argument('-M', '--model', type=pathlib.Path, required=True, help="Path to the model (.h5).")
parser.add_argument('-np', '--nb-passes', type=int, required=False, default=1, help="Number of passes. Default: 1.")
# input files
parser.add_argument('-m', '--mov-img', type=pathlib.Path, required=True, help='Path to the moving image. May be a single file or a directory containing multiple images.')
parser.add_argument('-g', '--geom', type=str, required=False, default='mni1', help="Path to geometry image for resampling, can be 'mni1' or 'mni2'. Default: same as ref for training.")
parser.add_argument('-ms', '--mov-seg', type=pathlib.Path, required=False, default=None, help='Path to the moving segmentation (only required for POLAFFINI initialization). Must be a directory if -m is also a directory, where segmentations are saved with the same filename as their corresponding moving image')
parser.add_argument('-rs', '--ref-seg', type=str, required=False, default=None, help="Path to the reference template segmentation, can be 'mni1' or 'mni2' (only required for POLAFFINI initialization).")
# output files
parser.add_argument('-oi', '--out-img', type=pathlib.Path, required=False, default=None, help='Path to the output moved image. If a directory is specified, moved images will be saved there with the same name as the moving image')
parser.add_argument('-os', '--out-seg', type=pathlib.Path, required=False, default=None, help='Path to the output moved segmentation. If a directory is specified, moved segmentations will be saved there with the same name as the moving image')
parser.add_argument('-ot', '--out-transfo', type=pathlib.Path, required=False, default=None, help='Path to the output full transformation (all combined). If a directory is specified, transformations will be saved there with the same name as the moving image')
parser.add_argument('-of', '--out-svf', type=pathlib.Path, required=False, default=None, help='Path to the output transformation in SVF form (from the DL model only). If a directory is specified, transformations will be saved there with the same name as the moving image')
parser.add_argument('-of0', '--out-svf0', type=pathlib.Path, required=False, default=None, help='Path to the output transformation in SVF form (from the DL model only). If a directory is specified, transformations will be saved there with the same name as the moving image')
# polaffini parameters
parser.add_argument('-polaffini', '--polaffini', type=int, required=False, default=1, help='Perform POLAFFINI (1:yes, 0:no). Default: 1.')
parser.add_argument('-noinit', '--noinit', type=int, required=False, default=0, help='Just read images without any kind of initialization. Default: 0.')
parser.add_argument('-transfo', '--transfos_type', type=str, required=False, default='affine', help="Type of the local tranformations ('affine' or 'rigid'). Default: 'affine'.")
parser.add_argument('-sigma', '--sigma', type=float, required=False, default=15, help='Standard deviation (in mm) for the Gaussian kernel. The higher the sigma, the smoother the output transformation. Use inf for affine transformation. Default: 15.')
parser.add_argument('-wbg', '--weight_bg', type=float, required=False, default=1e-5, help='Weight of the global background transformation for stability. Default: 1e-5.')
parser.add_argument('-downf', '--down_factor', type=float, required=False, default=4, help='Downsampling factor of the transformation. Default: 4.')
parser.add_argument('-dist', '--dist', type=str, required=False, default='center', help="Distance used for the weight maps. 'center': distance to neighborhood center, or 'maurer': distance to label. Default: 'center'.")
parser.add_argument('-omit_labs','--omit_labs', type=int, nargs='+', required=False, default=[], help='List of labels to omit. Default: []. Example: 2 41. 0 (background) is always omitted.')
parser.add_argument('-bg_transfo','--bg_transfo', type=int, required=False, default=1, help='Compute an affine background transformation(1:yes, 0:no). Default: 1.')

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
args.polaffini = bool(args.polaffini)
args.noinit = bool(args.noinit)
if args.noinit:
    args.polaffini = False
args.bg_transfo = bool(args.bg_transfo)

mov_img_path: pathlib.Path = args.mov_img
mov_seg_path: pathlib.Path = args.mov_seg

out_img_path: pathlib.Path = args.out_img
out_seg_path: pathlib.Path = args.out_seg
out_transfo_path: pathlib.Path = args.out_transfo
out_svf_path: pathlib.Path = args.out_svf
out_svf0_path: pathlib.Path = args.out_svf0

if mov_img_path.is_dir():
    if (mov_seg_path is not None and not mov_seg_path.is_dir()) \
        or (out_img_path is not None and not out_img_path.is_dir()) \
        or (out_seg_path is not None and not out_seg_path.is_dir()) \
        or (out_transfo_path is not None and not out_transfo_path.is_dir()) \
        or (out_svf_path is not None and not out_svf_path.is_dir()) \
        or (out_svf0_path is not None and not out_svf0_path.is_dir()):
        sys.exit("\nWhen specifying a directory as input, an existing directory must also be passed for the moving segmentations and all output paths.")

if mov_seg_path is None and (args.polaffini or out_seg_path is not None):
    sys.exit("\nNeed a moving segmentation.")
if args.ref_seg is None and (args.polaffini):
    sys.exit("\nNeed a reference segmentation.")

print('\nLoading model...')

if args.ref_seg == "mni2":
    args.ref_seg = os.path.join(maindir, 'refs', 'mni_dkt_2mm.nii.gz')
elif args.ref_seg == "mni1":
    args.ref_seg = os.path.join(maindir, 'refs', 'mni_dkt.nii.gz')

ref_seg: Optional[sitk.Image] = None
if args.polaffini:
    ref_seg = utils.imageIO(args.ref_seg).read()

model = dwarp.networks.diffeo2atlas.load(args.model)

inshape = model.output_shape[0][1:-1]
ndims = len(inshape)
matO = np.asmatrix(model.config.params['orientation'])
origin, spacing, direction = utils.decomp_matOrientation(matO)

if mov_img_path.is_file():
    mov_img_names = [mov_img_path.name]
    mov_img_path = mov_img_path.parent
    if args.polaffini:
        mov_seg_names = [mov_seg_path.name]
        mov_seg_path = mov_seg_path.parent
else:
    mov_img_names = sorted([f.name for f in mov_img_path.glob('*') if f.is_file]) 
    if args.polaffini:
        mov_seg_names = sorted([f.name for f in mov_seg_path.glob('*') if f.is_file])
        if len(mov_seg_names) != len(mov_img_names):
            sys.exit("\nNot the same number of images and segmentations.")
            
geom: Optional[sitk.Image] = None
if out_img_path is not None or out_seg_path is not None or out_transfo_path is not None:
    if args.geom == "mni2":
        geom_file = os.path.join(maindir, 'refs', 'mni_brain_2mm.nii.gz')
    elif args.geom == "mni1":
        geom_file = os.path.join(maindir, 'refs', 'mni_brain.nii.gz')
    else:
        geom_file = args.geom
    geom = utils.imageIO(geom_file).read()


for i in range(len(mov_img_names)):

    print(f"\nRegistration {i+1}/{len(mov_img_names)}")
    print(f" - img: {mov_img_names[i]}")
    if args.polaffini:
        print(f" - seg: {mov_seg_names[i]}")

    moving = utils.imageIO(mov_img_path.joinpath(mov_img_names[i])).read()

    if args.polaffini:
        moving_seg = utils.imageIO(mov_seg_path.joinpath(mov_seg_names[i])).read()

        print('Initializing through POLAFFINI...')
        init_aff, polyAff_svf = polaffini.estimateTransfo(mov_seg=moving_seg,
                                                          ref_seg=ref_seg,
                                                          sigma=args.sigma,
                                                          weight_bg=args.weight_bg,
                                                          transfos_type=args.transfos_type,
                                                          down_factor=args.down_factor,
                                                          dist=args.dist,
                                                          omit_labs=args.omit_labs,
                                                          bg_transfo=args.bg_transfo)
        transfo = polaffini.get_full_transfo(init_aff, polyAff_svf)

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputOrigin(origin)
        resampler.SetOutputSpacing(spacing)
        resampler.SetOutputDirection(direction)
        resampler.SetTransform(transfo)
        resampler.SetSize(inshape)
        mov = resampler.Execute(moving)
        mov = utils.normalize_intensities(mov)

        transfo_full = sitk.CompositeTransform(transfo)
    else:
        if not args.noinit:
            mov = utils.resample_image(moving, inshape, matO, sitk.sitkLinear)
            mov = utils.normalize_intensities(mov)
        else:
            mov = moving

        transfo_full = sitk.CompositeTransform(ndims)

    mov = sitk.GetArrayFromImage(mov)[np.newaxis,..., np.newaxis]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputDirection(direction)
    resampler.SetSize(inshape)
    resampler.SetInterpolator(sitk.sitkLinear)

    for _ in range(args.nb_passes):
        print('Registering through model...')
        _, field, svf = model.register(mov)

        print('Composing transformations...')
        field = utils.get_real_field(field, matO)
        field = sitk.GetImageFromArray(field[0, ...], isVector=True)
        field.SetDirection(direction)
        field.SetSpacing(spacing)
        field.SetOrigin(origin)
        field = sitk.DisplacementFieldTransform(field)

        transfo_full.AddTransform(field)

    if geom is not None:
        print('Resampling...')
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(geom.GetSize())
        resampler.SetOutputOrigin(geom.GetOrigin())
        resampler.SetOutputDirection(geom.GetDirection())
        resampler.SetOutputSpacing(geom.GetSpacing())
        resampler.SetTransform(transfo_full)
        if out_img_path is not None:
            moved = resampler.Execute(moving)
        if out_seg_path is not None:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            moved_seg = resampler.Execute(moving_seg)

    if out_img_path is not None or out_seg_path is not None or out_transfo_path is not None or out_svf_path is not None:
        print('Writing output files...')
        if out_img_path is not None:
            out_path = out_img_path if out_img_path.suffix != "" else out_img_path.joinpath(mov_img_names[i])
            utils.imageIO(out_path).write(moved)
        if out_seg_path is not None:
            out_path = out_seg_path if out_seg_path.suffix != "" else out_seg_path.joinpath(mov_seg_names[i])
            utils.imageIO(out_path).write(moved_seg)
        if out_transfo_path is not None:
            mov_img_name_noext = mov_img_names[i].split(".")[0]
            out_path = out_transfo_path if out_transfo_path.suffix != "" else out_transfo_path.joinpath(f"{mov_img_name_noext}.nii.gz")
            tr2disp = sitk.TransformToDisplacementFieldFilter()
            tr2disp.SetReferenceImage(geom)
            utils.imageIO(out_path).write(tr2disp.Execute(transfo_full))
        if out_svf_path is not None:
            mov_img_name_noext = mov_img_names[i].split(".")[0]
            out_path = out_svf_path if out_svf_path.suffix != "" else out_svf_path.joinpath(f"{mov_img_name_noext}.nii.gz")
            svf = utils.get_real_field(svf, matO)
            svf = sitk.GetImageFromArray(svf[0, ...], isVector=True)
            svf.SetDirection(direction)
            svf.SetSpacing(spacing)
            svf.SetOrigin(origin)
            utils.imageIO(out_path).write(svf)

print('\nboom\n')
