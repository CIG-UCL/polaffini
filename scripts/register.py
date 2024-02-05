import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # remove this for more tensorflow logs
import sys
maindir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(maindir)
import numpy as np
import SimpleITK as sitk
import tensorflow as tf      
tf.config.experimental.set_visible_devices([], 'GPU')    
import utils
import dwarp
import polaffini.polaffini as polaffini
import argparse

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

parser = argparse.ArgumentParser(description="")
# model
parser.add_argument('-M', '--model', type=str, required=True, help="Path to the model (.h5).")
# input files
parser.add_argument('-m', '--mov-img', type=str, required=True, help='Path to the moving image.')
parser.add_argument('-g', '--geom', type=str, required=False, default='mni1', help="Path to geometry image for resampling, can be 'mni1' or 'mni2'. Default: same as ref for training.")
parser.add_argument('-ms', '--mov-seg', type=str, required=False, default=None, help='Path to the moving segmentation (only required for POLAFFINI initialization).')
parser.add_argument('-rs', '--ref-seg', type=str, required=False, default=None, help="Path to the reference template segmentation, can be 'mni1' or 'mni2' (only required for POLAFFINI initialization).")
# output files
parser.add_argument('-oi', '--out-img', type=str, required=False, default=None, help='Path to the output moved image.')
parser.add_argument('-os', '--out-seg', type=str, required=False, default=None, help='Path to the output moved segmentation.')
parser.add_argument('-ot', '--out-transfo', type=str, required=False, default=None, help='Path to the output transformation.')
parser.add_argument('-of', '--out-svf', type=str, required=False, default=None, help='Path to the output transformation in SVF form.')
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
    
#%% Images and model loading

if args.mov_seg is None and (args.polaffini or args.out_seg is not None):  
    sys.exit("\nNeed a moving segmentation.")
if args.ref_seg is None and (args.polaffini):  
    sys.exit("\nNeed a reference segmentation.")
    
print('\nLoading model and images...')

if args.ref_seg == "mni2":
    args.ref_seg = os.path.join(maindir, 'refs', 'mni_dkt_2mm.nii.gz')
elif args.ref_seg == "mni1":
    args.ref_seg = os.path.join(maindir, 'refs', 'mni_dkt.nii.gz')

model = dwarp.networks.diffeo2atlas.load(args.model)

inshape = model.output_shape[0][1:-1]
ndims = len(inshape)
matO = np.asmatrix(model.config.params['orientation'])
origin, spacing, direction = utils.decomp_matOrientation(matO)

moving = utils.imageIO(args.mov_img).read()

if args.polaffini or args.out_seg is not None:  
    moving_seg = utils.imageIO(args.mov_seg).read()
    
#%% POLAFFINI initialization

if args.polaffini:
    print('Initializing through POLAFFINI...')
    ref_seg = utils.imageIO(args.ref_seg).read()
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
    
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mov_seg = resampler.Execute(moving_seg)

    transfo_full = sitk.CompositeTransform(transfo)
else:
    if not args.noinit:
        mov = utils.resample_image(moving, inshape, matO, sitk.sitkLinear) 
        mov = utils.normalize_intensities(mov)
    else:
        mov = moving

    transfo_full = sitk.CompositeTransform(ndims)
   
mov = sitk.GetArrayFromImage(mov)[np.newaxis,..., np.newaxis]

#%% Regsitration through model

print('Registering through model...')
moved, field, svf = model.register(mov)

#%% Transformations composition and resampling

print('Composing transformations...')

field = utils.get_real_field(field, matO)
field = sitk.GetImageFromArray(field[0, ...], isVector=True)
field.SetDirection(direction)
field.SetSpacing(spacing)
field.SetOrigin(origin)
field = sitk.DisplacementFieldTransform(field)

transfo_full.AddTransform(field)

if args.out_img is not None or args.out_seg is not None or args.out_transfo is not None:
    print('Resampling...')
    resampler = sitk.ResampleImageFilter()
    if args.geom == "mni2":
        geom_file = os.path.join(maindir, 'refs', 'mni_brain_2mm.nii.gz')
    elif args.geom == "mni1":
        geom_file = os.path.join(maindir, 'refs', 'mni_brain.nii.gz')
    else:
        geom_file = args.geom
    geom = utils.imageIO(geom_file).read()
    resampler.SetSize(geom.GetSize())
    resampler.SetOutputOrigin(geom.GetOrigin())
    resampler.SetOutputDirection(geom.GetDirection())
    resampler.SetOutputSpacing(geom.GetSpacing())
    resampler.SetTransform(transfo_full)
    if args.out_img is not None:
        moved = resampler.Execute(moving)
    if args.out_seg is not None:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        moved_seg = resampler.Execute(moving_seg)    

if args.out_img is not None or args.out_seg is not None or args.out_transfo is not None or args.out_svf is not None:
    print('Writing output files...')
    if args.out_img is not None:
        utils.imageIO(args.out_img).write(moved) 
    if args.out_seg is not None:
        utils.imageIO(args.out_seg).write(moved_seg) 
    if args.out_transfo is not None:
        tr2disp = sitk.TransformToDisplacementFieldFilter()
        tr2disp.SetReferenceImage(geom)
        utils.imageIO(tr2disp.Execute(args.out_transfo)).write(transfo_full)
    if args.out_svf is not None:
        svf = utils.get_real_field(svf, matO)
        svf = sitk.GetImageFromArray(svf[0, ...], isVector=True)
        svf.SetDirection(direction)
        svf.SetSpacing(spacing)
        svf.SetOrigin(origin)
        utils.imageIO(args.out_svf).write(svf)
        
print('boom\n')    





