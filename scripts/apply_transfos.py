import os
import sys
maindir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(maindir)
import SimpleITK as sitk
from scipy.linalg import logm, expm
import polaffini.utils as utils
import argparse
import pathlib


parser = argparse.ArgumentParser(description="")
# input files
parser.add_argument('-m', '--mov-img', type=pathlib.Path, required=True, help='Path to the moving image..')
parser.add_argument('-g', '--geom', type=str, required=True, help="Path to geometry image for resampling, can be 'mni1' or 'mni2'.")
parser.add_argument('-t', '--transfos', nargs='+', type=pathlib.Path, required=False, default=[None], help="List of paths of transformations to be applied composed such that T1 T2 ... -> T1(T2(...)). Default: None (identity).")
parser.add_argument('-log', '--is-log', nargs='+', type=int, required=False, default=[0], help="List of 0 and 1 to indicate which transfo is in SVF form and needs to be integrated. Default: 0 (none needs to be integrated).")
parser.add_argument('-inv', '--invert', type=int, required=False, default=0, help="Switch to invert (1) or not (0) the list of tranformations. For fields, it only works if in SVF form (log==1). Default: 0.")
# output files
parser.add_argument('-oi', '--out-img', type=pathlib.Path, required=False, default=None, help='Path to the output moved image.')
parser.add_argument('-otf', '--out-transfo-full', type=pathlib.Path, required=False, default=None, help='Path to the output full transformation (all combined).')
# resampling parameters
parser.add_argument('-i', '--interp', type=str, required=False, default='linear', help="Interpolator for resampling. Can be 'linear', 'nearest' or 'spline'. Default: 'linear'.")

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
args.is_log = [bool(args.is_log[i]) for i in range(len(args.is_log))]
args.invert = bool(args.invert)

if len(args.is_log) == 1:
    args.is_log *= len(args.transfos)
if args.interp == 'linear':
    interp = sitk.sitkLinear
elif args.interp == 'nearest':
    interp = sitk.sitkNearestNeighbor
elif args.interp == 'spline':
    interp = sitk.sitkBSpline


mov_img = utils.imageIO(args.mov_img).read()
ndims = mov_img.GetDimension()
if args.geom == "mni2":
    geom_file = os.path.join(maindir, 'refs', 'mni_brain_2mm.nii.gz')
elif args.geom == "mni1":
    geom_file = os.path.join(maindir, 'refs', 'mni_brain.nii.gz')
else:
    geom_file = args.geom
geom = utils.imageIO(geom_file).read()
origin = geom.GetOrigin()
spacing = geom.GetSpacing()
direction = geom.GetDirection()
size = geom.GetSize()


print('Composing transformations...')

transfo_list = []
for t in range(len(args.transfos)):
    print(' - transfo ',t+1)
    if args.transfos[t].suffix == '.txt':        
        transfo = sitk.ReadTransform(args.transfos[t])      
        if args.is_log[t]:
            aff_mat = utils.aff_tr2mat(transfo)
            aff_mat = expm(aff_mat)
            transfo = utils.mat2tr(aff_mat)
        if args.invert:
            transfo = transfo.GetInverse()
    else:     
        transfo = sitk.ReadImage(args.transfos[t])
        if args.is_log[t]:
            if args.invert:
                transfo = sitk.Compose([-sitk.VectorIndexSelectionCast(transfo, i) for i in range(ndims)])
            transfo = utils.integrate_svf(transfo, out_tr=False)
        transfo = sitk.DisplacementFieldTransform(transfo)
    transfo_list += [transfo]

if args.invert:
    transfo_list = transfo_list[::-1]

transfo_full = sitk.CompositeTransform(ndims)
for transfo in transfo_list:
    transfo_full.AddTransform(transfo) 

print('Resampling...')
resampler = sitk.ResampleImageFilter()
resampler.SetOutputOrigin(origin)
resampler.SetOutputSpacing(spacing)
resampler.SetOutputDirection(direction)
resampler.SetSize(size)
resampler.SetInterpolator(interp)
resampler.SetTransform(transfo_full)
  
moved_img = resampler.Execute(mov_img)


print('Writing output files...')
if args.out_img is not None:
     utils.imageIO(args.out_img).write(moved_img)
 
if args.out_transfo_full is not None:
    tr2disp = sitk.TransformToDisplacementFieldFilter()
    tr2disp.SetReferenceImage(geom)
    utils.imageIO(args.out_transfo_full).write(tr2disp.Execute(transfo_full))
        
    
print('\nboom\n')
