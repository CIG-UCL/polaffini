import os
import sys
maindir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(maindir)

import glob
import numpy as np
import SimpleITK as sitk    
import argparse  
import polaffini.utils as utils
import polaffini.polaffini as polaffini

parser = argparse.ArgumentParser(description="POLAFFINI segmentation-based initialization for non-linear registration to template.")

# inputs
parser.add_argument('-m', '--mov-img', type=str, required=True, help='Path to the moving images (can use *).')
parser.add_argument('-ms', '--mov-seg', type=str, required=True, help='Path to the moving segmentations (can use *, should have same alphabetical order as the images).')
parser.add_argument('-ma', '--mov-aux', type=str, required=False, default=None, help='Path to the moving auxiliary images (can use *, should have same alphabetical order as the images).')
parser.add_argument('-r', '--ref-img', type=str, required=False, default=None, help="Path to the reference template image, can be 'mni1' or 'mni2'")
parser.add_argument('-rs', '--ref-seg', type=str, required=False, default=None, help="Path to the reference template segmentation, can be 'mni1' or 'mni2'")
parser.add_argument('-ra', '--ref-aux', type=str, required=False, default=None, help='Path to the reference template auxiliary image.')
# outputs
parser.add_argument('-o', '--out-dir', type=str, required=True, help='Path to output directory.')
parser.add_argument('-ot', '--out-transfo', type=int, required=False, default=0, help='Also output transformations (1:yes, 0:no). Default: 0.')
parser.add_argument('-os', '--out-seg', type=int, required=False, default=0, help='Also output moved segmentations (1:yes, 0:no). Default: 0.')
parser.add_argument('-oa', '--out-aux', type=int, required=False, default=0, help='Also output moved auxiliary images (1:yes, 0:no). Default: 0.')
parser.add_argument('-ohot', '--one-hot', type=int, required=False, default=0, help='Perform one-hot encoding on moved output segmentations (1:yes, 0:no). Default: 1.')
parser.add_argument('-mask', '--mask', type=int, required=False, default=1, help='Perform masking using all labels except 24 (1:yes, 0:no). Default: 1.')
parser.add_argument('-kpad', '--k-padding', type=int, required=False, default=5, help='Pad an image such that image size along each dimension  is a multiple of 2^k (k must be greater than the number of contracting levels). Default: 5.')
parser.add_argument('-ext', '--ext', type=str, required=False, default='.nii.gz', help="Extension of output images. Default: '.nii.gz'.")
# polaffini parameters
parser.add_argument('-transfo', '--transfos-type', type=str, required=False, default='affine', help="Type of the local tranformations ('affine', 'rigid', 'translation' or 'volrot' (rigid and volume)). Default: 'affine'.")
parser.add_argument('-transfo_bg', '--bg-transfo-type', type=str, required=False, default='affine', help="Type of the background tranformation ('affine', 'rigid', 'translation' or 'volrot' (rigid and volume)). Default: 'affine'.")
parser.add_argument('-sigma', '--sigma', type=float, required=False, default=15, help='Standard deviation (in mm) for the Gaussian kernel. The higher the sigma, the smoother the output transformation. Use inf for affine transformation. Default: 15.')
parser.add_argument('-wbg', '--weight-bg', type=float, required=False, default=1e-5, help='Weight of the global background transformation for stability. Default: 1e-5.')
parser.add_argument('-downf', '--down-factor', type=float, required=False, default=4, help='Downsampling factor of the transformation. Default: 4.')
parser.add_argument('-dist', '--dist', type=str, required=False, default='center', help="Distance used for the weight maps. 'center': distance to neighborhood center, or 'maurer': distance to label. Default: 'center'.")
parser.add_argument('-omit_labs','--omit-labs', type=int, nargs='+', required=False, default=[], help='List of labels to omit. Default: []. 0 (background) is always omitted.')
parser.add_argument('-bg_transfo','--bg-transfo', type=int, required=False, default=1, help='Compute an affine background transformation. (1:yes, 0:no). Default: 1.')
# other       
parser.add_argument('-p', '--proc', type=int, required=False, default=1, help='Number of processes for parallel computing. Default: 1')

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
args.out_transfo = bool(args.out_transfo)
args.out_seg = bool(args.out_seg)
args.mask = bool(args.mask)
args.one_hot = bool(args.one_hot)
args.bg_transfo = bool(args.bg_transfo)
if args.ext[0] != '.':
    args.ext = '.' + args.ext
                
if args.proc > 1:
  from joblib import Parallel, delayed

#%% 

def init_polaffini_set(mov_files, 
                       mov_seg_files,
                       ref_seg,
                       labs,
                       resampler,
                       out_dir,
                       proc,
                       mov_aux_files=None,
                       out_transfo=False, 
                       out_seg=False,
                       one_hot=True,
                       ext='.nii.gz',
                       sigma=30, weight_bg=1e-5, transfos_type='affine', down_factor=8, dist='center', omit_labs=[], bg_transfo=True):

    for i in range(len(mov_files)):
        if args.proc <= 1:
            print(i+1, '/', len(mov_files), end="\r", flush=True)
            
        mov_seg = utils.imageIO(mov_seg_files[i]).read()
        
        init_aff, polyAff_svf = polaffini.estimateTransfo(mov_seg=mov_seg, 
                                                          ref_seg=ref_seg,
                                                          sigma=sigma,
                                                          weight_bg=weight_bg,
                                                          transfos_type=transfos_type,
                                                          bg_transfo_type = args.bg_transfo_type,
                                                          down_factor=down_factor,
                                                          dist=dist,
                                                          omit_labs=omit_labs,
                                                          bg_transfo=bg_transfo)
        transfo = polaffini.get_full_transfo(init_aff, polyAff_svf)
        
        # apply transfos, rescale intensities, one hot encoding, write images
        resampler.SetTransform(transfo)   
        resampler.SetReferenceImage(ref_seg)
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetInterpolator(sitk.sitkLinear)
        mov_img = sitk.Cast(utils.imageIO(mov_files[i]).read(), sitk.sitkFloat32)
        if args.mask:
            mask = sitk.BinaryThreshold(mov_seg, 1, 23) + sitk.BinaryThreshold(mov_seg, 25, 1e9)
            mask = sitk.BinaryMorphologicalClosing(mask, [6]*3)
            mov_img = sitk.Mask(mov_img, mask)
        mov_img = resampler.Execute(mov_img)
        mov_img = utils.normalize_intensities(mov_img)
            
        outfile, _ = utils.imageIO(os.path.split(mov_files[i])[-1])._splitext()
        utils.imageIO(os.path.join(out_dir, 'img', outfile + ext)).write(mov_img)
        if mov_aux_files is not None:
            mov_aux = sitk.Cast(utils.imageIO(mov_aux_files[i]).read(), sitk.sitkFloat32)
            mov_aux = resampler.Execute(mov_aux)
            mov_aux = utils.normalize_intensities(mov_aux, wmax=1)
            outfile, _ = utils.imageIO(os.path.split(mov_aux_files[i])[-1])._splitext()            
            utils.imageIO(os.path.join(out_dir,'auxi', outfile + ext)).write(mov_aux)
        if out_seg:
            resampler.SetOutputPixelType(sitk.sitkInt16)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            mov_seg = resampler.Execute(mov_seg)
            if one_hot:
                mov_seg = utils.one_hot_enc(mov_seg, labs)
            outfile, _ = utils.imageIO(os.path.split(mov_seg_files[i])[-1])._splitext()    
            utils.imageIO(os.path.join(out_dir,'seg', outfile + ext)).write(mov_seg)
        if out_transfo:
            filename = os.path.splitext(os.path.splitext(mov_files[i])[0])[0]           
            sitk.WriteTransform(init_aff, os.path.join(out_dir,'transfo',os.path.split(filename)[-1]+'.txt'))
            if sigma < np.inf:
                utils.imageIO(os.path.join(out_dir,'transfo',os.path.split(filename)[-1]+'.nii.gz')).write(polyAff_svf)


#%% Main

if args.ref_seg == "mni2" or args.ref_img == "mni2":
    args.ref_seg = os.path.join(maindir, 'refs', 'mni_dkt_2mm.nii.gz')
    args.ref_img = os.path.join(maindir, 'refs', 'mni_t1_2mm.nii.gz')
elif args.ref_seg == "mni1" or args.ref_img == "mni1":
    args.ref_seg = os.path.join(maindir, 'refs', 'mni_dkt.nii.gz')
    args.ref_img = os.path.join(maindir, 'refs', 'mni_t1.nii.gz')  
    
os.makedirs(os.path.join(args.out_dir), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, 'img'), exist_ok=True)
if args.out_seg:
    os.makedirs(os.path.join(args.out_dir, 'seg'), exist_ok=True)
if args.mov_aux:
    os.makedirs(os.path.join(args.out_dir, 'auxi'), exist_ok=True)
if args.out_transfo:
    os.makedirs(os.path.join(args.out_dir, 'transfo'), exist_ok=True)
   
mov_files = sorted(glob.glob(args.mov_img))
mov_seg_files = sorted(glob.glob(args.mov_seg))
if args.mov_aux is not None:
    mov_aux_files = sorted(glob.glob(args.mov_aux))
else:
    mov_aux_files = None
    
ref = sitk.Cast(utils.imageIO(args.ref_img).read(), sitk.sitkFloat32)
ref = utils.normalize_intensities(ref)
ref = utils.pad_image(ref, k=args.k_padding)

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(ref)   

resampler.SetInterpolator(sitk.sitkNearestNeighbor)  
resampler.SetOutputPixelType(sitk.sitkFloat32)
ref_seg = utils.imageIO(args.ref_seg).read()
ref_seg = resampler.Execute(ref_seg)
if args.mask:
    mask = sitk.BinaryThreshold(ref_seg, 1, 23) + sitk.BinaryThreshold(ref_seg, 25, 1e9)
    mask = sitk.BinaryMorphologicalClosing(mask, [6]*3)
    ref = sitk.Mask(ref, mask)
labs = np.unique(sitk.GetArrayFromImage(ref_seg))
labs = np.delete(labs, labs==0)
   
if args.ref_aux is not None:
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetInterpolator(sitk.sitkLinear)
    ref_aux = utils.imageIO(args.ref_aux).read()
    ref_aux = resampler.Execute(ref_aux)
    ref_aux = utils.normalize_intensities(ref_aux,wmax=1)
      
inshape = ref.GetSize()
ndims = ref.GetDimension()

print('\nPOLAFFINI initialization on set (n='+str(len(mov_files))+'):')
if args.proc <= 1:
    init_polaffini_set(mov_files=mov_files, 
                       mov_seg_files=mov_seg_files,
                       mov_aux_files=mov_aux_files,
                       ref_seg=ref_seg,
                       labs=labs,
                       resampler=resampler,
                       out_dir=args.out_dir,
                       out_seg=args.out_seg,
                       one_hot=args.one_hot,
                       ext=args.ext,
                       proc=args.proc,
                       sigma=args.sigma,
                       weight_bg=args.weight_bg,
                       transfos_type=args.transfos_type,
                       bg_transfo_type = args.bg_transfo_type,
                       out_transfo=args.out_transfo,
                       down_factor=args.down_factor,
                       dist=args.dist,
                       omit_labs=args.omit_labs,
                       bg_transfo=args.bg_transfo)

else:
     if mov_aux_files is None:
         mov_aux_files = [None]*len(mov_files)
     Parallel(n_jobs=args.proc)(delayed(init_polaffini_set)(mov_files=[mov_files[i]], 
                                                            mov_seg_files=[mov_seg_files[i]], 
                                                            mov_aux_files=[mov_aux_files[i]],
                                                            ref_seg=ref_seg, 
                                                            labs=labs,
                                                            resampler=resampler,
                                                            out_dir=args.out_dir, 
                                                            out_seg=args.out_seg,
                                                            one_hot=args.one_hot,
                                                            ext=args.ext,
                                                            proc=args.proc,
                                                            sigma=args.sigma, 
                                                            weight_bg=args.weight_bg,
                                                            transfos_type=args.transfos_type,
                                                            bg_transfo_type = args.bg_transfo_type,
                                                            out_transfo=args.out_transfo,
                                                            down_factor=args.down_factor,
                                                            dist=args.dist,
                                                            omit_labs=args.omit_labs,
                                                            bg_transfo=args.bg_transfo) for i in range(len(mov_files)))

utils.imageIO(os.path.join(args.out_dir, 'ref_img' + args.ext)).write(ref)
if args.ref_aux:
    utils.imageIO(os.path.join(args.out_dir, 'ref_aux' + args.ext)).write(ref_aux)
if args.out_seg:  
    if args.one_hot:
        ref_seg = utils.one_hot_enc(ref_seg, labs)
    utils.imageIO(os.path.join(args.out_dir, 'ref_seg' + args.ext)).write(ref_seg)
