import os
import sys
maindir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(maindir)
import SimpleITK as sitk   
import argparse   
import polaffini.utils as utils
import polaffini.polaffini as polaffini

def polaffini_pair():
    parser = argparse.ArgumentParser(description="POLAFFINI segmentation-based initialization for non-linear registration.")

    # inputs
    parser.add_argument('-ms', '--mov-seg', type=str, required=True, help='Path to the moving segmentation.')
    parser.add_argument('-rs', '--ref-seg', type=str, required=True, help="Path to the reference segmentation, can be 'mni1' or 'mni2'")
    parser.add_argument('-m', '--mov-img', type=str, required=False, default=None, help='Path to the moving image.')
    parser.add_argument('-ma', '--mov-aux', type=str, required=False, default=None, help='Path to the moving auxiliary image.')
    parser.add_argument('-g', '--geom', type=str, required=False, default=None, help='Path to geometry image for resampling.')
    parser.add_argument('-r', '--ref-img', type=str, required=False, default=None, help='Path to the reference image (for kissing).')
    parser.add_argument('-ra', '--ref-aux', type=str, required=False, default=None, help='Path to the reference auxiliary image (for kissing).')
    # outputs
    parser.add_argument('-oi', '--out-img', type=str, required=False, default=None, help='Path to output image.')
    parser.add_argument('-os', '--out-seg', type=str, required=False, default=None, help='Path to output moved segmentation.')
    parser.add_argument('-oa', '--out-aux', type=str, required=False, default=None, help='Path to output moved auxiliary image.')
    parser.add_argument('-ot', '--out-transfo', type=str, required=False, default=None, help='Path to output full transformations in diffeo form.')
    parser.add_argument('-ota', '--out-aff-transfo', type=str, required=False, default=None, help='Path to output affine part of transformation (.txt)')
    parser.add_argument('-otp', '--out-poly-transfo', type=str, required=False, default=None, help='Path to output polyaffine part of the transformation in SVF form.')
    parser.add_argument('-k', '--kissing', type=int, required=False, default=0, help='Kissing mapping: meets at location alpha on the diffeomorphic path.')
    # polaffini parameters
    parser.add_argument('-transfo', '--transfos-type', type=str, required=False, default='affine', help="Type of the local tranformations ('affine', 'rigid', 'translation' or 'volrot' (rigid and volume)). Default: 'affine'.")
    parser.add_argument('-transfo_bg', '--bg-transfo-type', type=str, required=False, default='affine', help="Type of the background tranformation ('affine', 'rigid', 'translation' or 'volrot' (rigid and volume)). Default: 'affine'.")
    parser.add_argument('-sigma', '--sigma', type=float, required=False, default=None, help="Standard deviation (in mm) for the Gaussian kernel. The higher the sigma, the smoother the output transformation. Use inf for affine transformation. Default: 'silverman'.")
    parser.add_argument('-alpha', '--alpha', type=float, required=False, default=1, help='Position of the overall transformation on the diffeomorphic path from identity to the transfo from moving to reference (e.g. use 0.5 for half-way registration). Default: 1.')
    parser.add_argument('-wbg', '--weight-bg', type=float, required=False, default=1e-5, help='Weight of the global background transformation for stability. Default: 1e-5.')
    parser.add_argument('-downf', '--down-factor', type=float, required=False, default=4, help='Downsampling factor of the transformation. Default: 4.')
    parser.add_argument('-dist', '--dist', type=str, required=False, default='center', help="Distance used for the weight maps. 'center': distance to neighborhood center, or 'maurer': distance to label. Default: 'center'.")
    parser.add_argument('-omit_labs','--omit-labs', type=int, nargs='+', required=False, default=[], help='List of labels to omit. Default: []. 0 (background) is always omitted.')
    parser.add_argument('-bg_transfo','--bg-transfo', type=int, required=False, default=1, help='Compute an affine background transformation. (1:yes, 0:no). Default: 1.')
    parser.add_argument('-volw','--vol_weights', type=int, required=False, default=0, help='Weight by region volumes when estimating the background transformation. (1:yes, 0:no). Default: 1.')
    # other
    parser.add_argument('-do_bch','--do-bch', type=int, required=False, default=0, help='Use the BCH formula to compute the overall field. (1:yes, 0:no). Default: 0.')
    
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    
    args.bg_transfo = bool(args.bg_transfo)
    args.vol_weights = bool(args.vol_weights)
    args.do_bch = bool(args.do_bch)

    #%% Main

    if args.ref_seg == "mni2":
        args.ref_seg = os.path.join(maindir, 'refs', 'mni_dkt_2mm.nii.gz')
    elif args.ref_seg == "mni1":
        args.ref_seg = os.path.join(maindir, 'refs', 'mni_dkt.nii.gz')
        
    mov_seg = utils.imageIO(args.mov_seg).read()
    ref_seg = utils.imageIO(args.ref_seg).read()
    if args.geom is not None:
        geom = utils.imageIO(args.geom).read()
    
    if args.sigma is None:
        args.sigma = 'silverman'
        
    if args.do_bch:
        init_aff, polyAff_svf, polyAff_svf_jac = polaffini.estimateTransfo(mov_seg=mov_seg,
                                                                           ref_seg=ref_seg,
                                                                           sigma=args.sigma,
                                                                           alpha=1,
                                                                           weight_bg=args.weight_bg,
                                                                           transfos_type=args.transfos_type,
                                                                           bg_transfo_type = args.bg_transfo_type,
                                                                           down_factor=args.down_factor,
                                                                           dist=args.dist,
                                                                           omit_labs=args.omit_labs,
                                                                           bg_transfo=args.bg_transfo,
                                                                           out_jac=True,
                                                                           vol_weights=args.vol_weights)
        full_svf = polaffini.get_full_svf(init_aff, polyAff_svf, polyAff_svf_jac)
        transfo = polaffini.integrate_svf_lowMem(full_svf, alpha=args.alpha)
        if args.kissing:
            transfo_rev = polaffini.integrate_svf_lowMem(full_svf, alpha=args.alpha-1)
            
    else:
        init_aff, polyAff_svf = polaffini.estimateTransfo(mov_seg=mov_seg,
                                                          ref_seg=ref_seg,
                                                          sigma=args.sigma,
                                                          alpha=args.alpha,
                                                          weight_bg=args.weight_bg,
                                                          transfos_type=args.transfos_type,
                                                          bg_transfo_type = args.bg_transfo_type,
                                                          down_factor=args.down_factor,
                                                          dist=args.dist,
                                                          omit_labs=args.omit_labs,
                                                          bg_transfo=args.bg_transfo,
                                                          vol_weights=args.vol_weights)
        transfo = polaffini.get_full_transfo(init_aff, polyAff_svf)
    
        if args.kissing:
             init_aff_rev, polyAff_svf_rev = polaffini.estimateTransfo(mov_seg=ref_seg,
                                                                       ref_seg=mov_seg,
                                                                       sigma=args.sigma,
                                                                       alpha=1-args.alpha,
                                                                       weight_bg=args.weight_bg,
                                                                       transfos_type=args.transfos_type,
                                                                       bg_transfo_type = args.bg_transfo_type,
                                                                       down_factor=args.down_factor,
                                                                       dist=args.dist,
                                                                       omit_labs=args.omit_labs,
                                                                       bg_transfo=args.bg_transfo,
                                                                       vol_weights=args.vol_weights)
             transfo_rev = polaffini.get_full_transfo(init_aff_rev, polyAff_svf_rev)

     
    resampler = sitk.ResampleImageFilter()
    if args.geom is None:
        geom = ref_seg
    resampler.SetReferenceImage(geom)
    resampler.SetTransform(transfo)
     
    if args.out_img is not None:
        if args.mov_img is None:
            sys.exit('Need a moving image.')
        mov_img = utils.imageIO(args.mov_img).read()
        resampler.SetInterpolator(sitk.sitkLinear)
        mov_img = resampler.Execute(mov_img)
        utils.imageIO(args.out_img).write(mov_img)
     
    if args.out_aux is not None:
        if args.mov_aux is None:
            sys.exit('Need an auxiliary moving image.')
        mov_aux = utils.imageIO(args.mov_aux).read()
        resampler.SetInterpolator(sitk.sitkLinear)
        mov_aux = resampler.Execute(mov_aux)
        utils.imageIO(args.out_aux).write(mov_aux)
        
    if args.out_seg is not None:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        mov_seg = resampler.Execute(mov_seg)
        utils.imageIO(args.out_seg).write(mov_seg)
    
    if args.out_aff_transfo is not None:
        sitk.WriteTransform(init_aff, args.out_aff_transfo)
    
    if args.out_poly_transfo is not None:
        if polyAff_svf is None:
            sys.exit('No polyaffine transfo. Maybe because sigma=inf (only an affine is estimated in that case).')
        utils.imageIO(args.out_poly_transfo).write(polyAff_svf)
    
    if args.out_transfo is not None:
        tr2disp = sitk.TransformToDisplacementFieldFilter()
        tr2disp.SetReferenceImage(ref_seg)
        utils.imageIO(args.out_transfo).write(tr2disp.Execute(transfo))
   
    if args.kissing:
        if args.geom is None:
            geom = mov_seg
        resampler.SetReferenceImage(geom)
        resampler.SetTransform(transfo_rev)
    
        if args.out_img is not None:
            if args.ref_img is None:
                sys.exit('Need a reference image.')
            ref_img = utils.imageIO(args.ref_img).read()
            resampler.SetInterpolator(sitk.sitkLinear)
            ref_img = resampler.Execute(ref_img)
            filename, ext = utils.imageIO(args.out_img)._splitext()
            utils.imageIO(filename + '_rev' + ext).write(ref_img)
    
        if args.out_aux is not None:
            if args.ref_aux is None:
                sys.exit('Need an auxiliary reference image.')
            ref_aux = utils.imageIO(args.ref_aux).read()
            resampler.SetInterpolator(sitk.sitkLinear)
            ref_aux = resampler.Execute(ref_aux)
            filename, ext = utils.imageIO(args.out_aux)._splitext()
            utils.imageIO(filename + '_rev' + ext).write(ref_aux)
        if args.out_seg is not None:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            ref_seg = resampler.Execute(ref_seg)
            filename, ext = utils.imageIO(args.out_seg)._splitext()
            utils.imageIO(filename + '_rev' + ext).write(ref_seg)  


if __name__ == "__main__":
    polaffini_pair()
