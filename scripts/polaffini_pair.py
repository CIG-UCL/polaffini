import os
import sys
maindir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(maindir)
import SimpleITK as sitk      
import polaffini.polaffini as polaffini
import argparse

parser = argparse.ArgumentParser(description="POLAFFINI segmentation-based initialization for non-linear registration.")

# inputs
parser.add_argument('-ms', '--mov-seg', type=str, required=True, help='Path to the moving segmentations (can use *, should have same alphabetical order as the images).')
parser.add_argument('-rs', '--ref-seg', type=str, required=True, help="Path to the reference segmentation, can be 'mni1' or 'mni2'")
parser.add_argument('-m', '--mov-img', type=str, required=False, default=None, help='Path to the moving images (can use *).')
# outputs
parser.add_argument('-oi', '--out-img', type=str, required=False, default=None, help='Path to output image.')
parser.add_argument('-os', '--out-seg', type=str, required=False, default=None, help='Path to output moved segmentation.')
parser.add_argument('-ot', '--out-transfo', type=str, required=False, default=None, help='Path to output full transformations in diffeo form.')
parser.add_argument('-ota', '--out-aff-transfo', type=str, required=False, default=None, help='Path to output affine part of transformation (.txt)')
parser.add_argument('-otp', '--out-poly-transfo', type=str, required=False, default=None, help='Path to output polyaffine part of the transformation in SVF form.')
# polaffini parameters
parser.add_argument('-transfo', '--transfos-type', type=str, required=False, default='affine', help="Type of the local tranformations ('affine' or 'rigid'). Default: 'affine'.")
parser.add_argument('-sigma', '--sigma', type=float, required=False, default=15, help='Standard deviation (in mm) for the Gaussian kernel. The higher the sigma, the smoother the output transformation. Use inf for affine transformation. Default: 15.')
parser.add_argument('-wbg', '--weight-bg', type=float, required=False, default=1e-5, help='Weight of the global background transformation for stability. Default: 1e-5.')
parser.add_argument('-downf', '--down-factor', type=float, required=False, default=4, help='Downsampling factor of the transformation. Default: 4.')
parser.add_argument('-dist', '--dist', type=str, required=False, default='center', help="Distance used for the weight maps. 'center': distance to neighborhood center, or 'maurer': distance to label. Default: 'center'.")
parser.add_argument('-omit_labs','--omit-labs', type=int, nargs='+', required=False, default=[], help='List of labels to omit. Default: []. 0 (background) is always omitted.')
parser.add_argument('-bg_transfo','--bg-transfo', type=int, required=False, default=1, help='Compute an affine background transformation. (1:yes, 0:no). Default: 1.')

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

#%% Main

if args.ref_seg == "mni2":
    args.ref_seg = os.path.join(maindir, 'refs', 'mni_dkt_2mm.nii.gz')
elif args.ref_seg == "mni1":
    args.ref_seg = os.path.join(maindir, 'refs', 'mni_dkt.nii.gz')


mov_seg = sitk.ReadImage(args.mov_seg)
ref_seg = sitk.ReadImage(args.ref_seg)

init_aff, polyAff_svf = polaffini.estimateTransfo(mov_seg=mov_seg, 
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
resampler.SetReferenceImage(ref_seg)   
resampler.SetTransform(transfo)   

if args.out_img is not None:
    if args.mov_img is None:
        sys.exit('Need a moving image.')
    mov_img = sitk.ReadImage(args.mov_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    mov_img = resampler.Execute(mov_img)
    sitk.WriteImage(mov_img, args.out_img)

if args.out_seg is not None:
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mov_seg = resampler.Execute(mov_seg)
    sitk.WriteImage(mov_seg, args.out_seg)
    
if args.out_aff_transfo is not None:      
    sitk.WriteTransform(init_aff, args.out_aff_transfo)

if args.out_poly_transfo is not None:  
    sitk.WriteImage(polyAff_svf, args.out_poly_transfo)
        
if args.out_transfo is not None:
    tr2disp = sitk.TransformToDisplacementFieldFilter()
    tr2disp.SetReferenceImage(polyAff_svf)
    sitk.WriteImage(tr2disp.Execute(transfo), args.out_transfo)

