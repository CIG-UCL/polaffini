import numpy as np
import SimpleITK as sitk        
import utils
import polaffini.polaffini as polaffini

def mov2atlas_res(mov_files, 
                  ref_file,
                  mov_seg_files=None,
                  ref_seg_file=None,
                  weight_file=None, 
                  kpad = 5,
                  batch_size=1):
    """
    Generator for moving images and a single reference, optionnally with weights.
    Moving images can be of different sizes and voxels sizes as well.
    Moving images are properly resampled in the geometry of the reference image.
    Center of the moving FOV are enforced to match the center of the reference FOV. 
    One-hot encoding is performed on segmentations.
    
    Parameters
    ----------
    mov_files : list.
        Path to moving images.
    ref_file : string.
        Path to single target image. 
    mov_seg-files : list, optional.
        Path to moving segmentations.
    ref_seg_file : string, optional.
        Path to target segmentation.
    weight_file : string, optional.
        Path to weight map associated to target atlas. The default is None.
    batch_size : int, optional.
        Batch size. The default is 1.

    Yields
    ------
    inputs : numpy ND-array or list of numpy ND-array.
        Batch of moving images properly resampled in the atlas geometry.
        Batch of moving segmentations properly resampled in the atlas geometry (optional).
    groundTruths : numpy ND-array or list of numpy ND-array.
        Batch of repeated reference image. If there is weights, they are concatenated along the last axis.
        Batch of repeated reference segmentation. (optional).
    """
    
    is_weight = weight_file is not None
    is_seg = (mov_seg_files is not None) and (ref_seg_file is not None)
    
    while True:
        
        ref = sitk.ReadImage(ref_file)
        ndims = ref.GetDimension() 
        inshape = ref.GetSize()
        refC = [float(inshape[i]) / 2 for i in range(ndims)]
        refC = ref.TransformContinuousIndexToPhysicalPoint(refC)
        ref = utils.pad_image(ref, k=kpad)
        inshape = ref.GetSize()
        ref = utils.normalize_intensities(ref)        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref)
        ref = sitk.GetArrayFromImage(ref)[np.newaxis,..., np.newaxis]
        ref = ref.astype(np.float32)
        ref = np.concatenate([ref]*batch_size, axis=0)        
        if is_seg:
            ref_seg = sitk.ReadImage(ref_seg_file)
            ref_seg = utils.pad_image(ref_seg, k=kpad) 
            ref_seg = sitk.GetArrayFromImage(ref_seg)
            labs = np.unique(ref_seg)
            labs = np.delete(labs, labs==0)
            ref_seg = utils.one_hot_enc(ref_seg, labs, segtype='array')
            ref_seg = np.concatenate([ref_seg]*batch_size, axis=0)
 
        if is_weight:
            weight = sitk.ReadImage(weight_file)
            weight = utils.pad_image(weight)
            weight = sitk.GetArrayFromImage(weight)[np.newaxis,..., np.newaxis]
            weight = weight.astype(np.float32)  
            weight = np.concatenate([weight]*batch_size, axis=0)
                    
        ind_batch = np.random.choice(range(0, len(mov_files)), size=batch_size, replace=False)
        mov_imgs = [] 
        mov_segs = [] 
        for i in ind_batch:
            
            mov_img = sitk.ReadImage(mov_files[i])
            imgC = [float(mov_img.GetSize()[k] / 2) for k in range(ndims)]
            imgC = mov_img.TransformContinuousIndexToPhysicalPoint(imgC)        
            shift = np.subtract(imgC, refC)
            transfo = sitk.TranslationTransform(ndims, shift)
            resampler.SetTransform(transfo)
            
            mov_img = resampler.Execute(mov_img)
            mov_img = utils.normalize_intensities(mov_img)
            mov_img = sitk.GetArrayFromImage(mov_img)[np.newaxis,..., np.newaxis]
            mov_img = mov_img.astype(np.float32)            
            mov_imgs += [mov_img]

            if is_seg:
                mov_seg = sitk.ReadImage(mov_seg_files[i])
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                mov_seg = resampler.Execute(mov_seg)
                mov_seg = sitk.GetArrayFromImage(mov_seg)
                mov_seg = utils.one_hot_enc(mov_seg, labs, segtype='array')
                mov_segs += [mov_seg] 

        mov_imgs = np.concatenate(mov_imgs, axis=0)

        if is_seg:
            mov_segs = np.concatenate(mov_segs, axis=0)

        inputs = [mov_imgs]

        if is_weight:
            groundTruths = [np.concatenate([ref, weight], axis=-1)]
        else:
            groundTruths = [ref]

        if is_seg:
            inputs += [mov_segs]
            groundTruths += [ref_seg]
            
        field0 = np.zeros((batch_size, *inshape, ndims))
        groundTruths += [field0]
        
        yield (inputs, groundTruths)
        
        
        

def mov2atlas_initialized(mov_files, 
                          ref_file,
                          mov_seg_files=None,
                          ref_seg_file=None,
                          weight_file=None,
                          batch_size=1):
    """
    Generator for moving images and a single reference. 
    Moving images are assumed to have already undergone affine or 
    polaffini initialization (typically using apply_polaffini). 
    Therefore the moving and target images are already resampled in the same grid. 
    Segmentations are supposed to be one-hot encoded.
    
    Parameters
    ----------
    mov_files : list.
        Path to moving images.
    ref_file : string.
        Path to single target image. 
    mov_seg-files : list, optional.
        Path to moving segmentations (in one-hot encoding).
    ref_seg_file : string, optional.
        Path to target segmentation (in one-hot encoding).
    weight_file : string, optional.
        Path to weight map associated to target atlas. The default is None.
    batch_size : int, optional.
        Batch size. The default is 1.

    Yields
    ------
    inputs : numpy ND-array or list of numpy ND-array.
        Batch of moving images properly resampled in the atlas geometry.
        Batch of moving segmentations properly resampled in the atlas geometry (optional).
    groundTruths : numpy ND-array or list of numpy ND-array.
        Batch of repeated reference image. If there is weights, they are concatenated along the last axis.
        Batch of repeated reference segmentation. (optional).
    """
    
    is_weight = weight_file is not None
    is_seg = (mov_seg_files is not None) and (ref_seg_file is not None)
    
    while True:
        
        ref = sitk.ReadImage(ref_file)
        if is_seg:
            ref_seg = sitk.ReadImage(ref_seg_file)

        inshape = ref.GetSize()
        ndims = ref.GetDimension()
        ref = sitk.GetArrayFromImage(ref)[np.newaxis,..., np.newaxis]
        ref = np.concatenate([ref]*batch_size, axis=0)
        
        if is_weight:
            weight = sitk.ReadImage(weight_file)
            weight = sitk.GetArrayFromImage(weight)[np.newaxis,..., np.newaxis] 
            weight = np.concatenate([weight]*batch_size, axis=0)
                    
        ind_batch = np.random.choice(range(0, len(mov_files)), size=batch_size, replace=False)
        mov_imgs = [] 
        mov_segs = []
        for i in ind_batch:        
            mov_img = sitk.ReadImage(mov_files[i])
            mov_img = sitk.GetArrayFromImage(mov_img)[np.newaxis,..., np.newaxis]           
            mov_imgs += [mov_img]         
            if is_seg:
                mov_seg = sitk.ReadImage(mov_seg_files[i])
                mov_seg = sitk.GetArrayFromImage(mov_seg)
                mov_seg = np.transpose(mov_seg,[*range(1,ndims+1)]+[0])[np.newaxis,...]
                mov_seg = mov_seg.astype(np.float32)
                mov_segs += [mov_seg] 

        mov_imgs = np.concatenate(mov_imgs, axis=0)
        
        if is_seg:
            mov_segs = np.concatenate(mov_segs, axis=0)

            ref_seg = sitk.GetArrayFromImage(ref_seg)
            ref_seg = np.transpose(ref_seg,[*range(1,ndims+1)]+[0])[np.newaxis,...]
            ref_seg = ref_seg.astype(np.float32)
            ref_seg = np.concatenate([ref_seg]*batch_size, axis=0)
         
        inputs = [mov_imgs]

        if is_weight:
            groundTruths = [np.concatenate([ref, weight], axis=-1)]
        else:
            groundTruths = [ref]
        
        if is_seg:
            inputs += [mov_segs]
            groundTruths += [ref_seg]
            
        field0 = np.zeros((batch_size, *inshape, ndims), np.float32)
        groundTruths += [field0]
        
        yield (inputs, groundTruths)
        
        

def pair_polaffini(mov_files, 
                   mov_seg_files,
                   vox_sz=[2,2,2],
                   grid_sz=[96,128,96],
                   labels=[2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28, 31, 41, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
                           1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035,
                           2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2034, 2035],
                   aug_axes = False,
                   polaffini_sigma=15,
                   polaffini_downf=4,
                   polaffini_omit_labs=[2,23,41],      
                   batch_size=1):
    """
    Generator for moving images and a single reference. 
    Moving images are assumed to have already undergone affine or 
    polaffini initialization (typically using apply_polaffini). 
    Therefore the moving and target images are already resampled in the same grid. 
    Segmentations are supposed to be one-hot encoded.
    
    Parameters
    ----------
    mov_files : list.
        Path to moving images.
    mov_seg-files : list, optional.
        Path to moving segmentations (in one-hot encoding).
    weight_file : string, optional.
        Path to weight map associated to target atlas. The default is None.
    batch_size : int, optional.
        Batch size. The default is 1.

    Yields
    ------
    inputs : numpy ND-array or list of numpy ND-array.
        Batch of moving images properly resampled in the atlas geometry.
        Batch of moving segmentations properly resampled in the atlas geometry (optional).
    groundTruths : numpy ND-array or list of numpy ND-array.
        Batch of repeated reference image. If there is weights, they are concatenated along the last axis.
        Batch of repeated reference segmentation. (optional).
    """
    
    ndims = len(vox_sz)
    
    while True:
                    
        ind_batch_ref = np.random.choice(range(0, len(mov_files)), size=batch_size, replace=False)
        ind_batch_mov = np.random.choice(range(0, len(mov_files)), size=batch_size, replace=False)
        ref_imgs = [] 
        ref_segs = []
        mov_imgs = [] 
        mov_segs = []
        for i, j in zip(ind_batch_ref, ind_batch_mov): 
            
            ref_img = sitk.ReadImage(mov_files[i])
            ref_img = utils.change_img_res(ref_img, vox_sz)
            ref_img = utils.change_img_size(ref_img, grid_sz)
            
            ref_seg = sitk.ReadImage(mov_seg_files[i])
            ref_seg = utils.change_img_res(ref_seg, vox_sz, interp=sitk.sitkNearestNeighbor)
            ref_seg = utils.change_img_size(ref_seg, grid_sz)
            
            if aug_axes:
                perm = np.random.permutation(range(ndims)).tolist()
                flip = np.random.choice(2, size=ndims).tolist()
                ref_img = sitk.PermuteAxes(ref_img, perm)
                ref_img = sitk.Flip(ref_img, flip)
                ref_seg = sitk.PermuteAxes(ref_seg, perm)
                ref_seg = sitk.Flip(ref_seg, flip)
            else:
                ref_img = sitk.DICOMOrient(ref_img, 'RPS')
                ref_seg = sitk.DICOMOrient(ref_seg, 'RPS')
                        
            mov_img = sitk.ReadImage(mov_files[j])
            mov_seg = sitk.ReadImage(mov_seg_files[j])

            init_aff, polyAff_svf = polaffini.estimateTransfo(mov_seg=mov_seg, 
                                                              ref_seg=ref_seg,
                                                              sigma=polaffini_sigma,
                                                              down_factor=polaffini_downf,
                                                              omit_labs=polaffini_omit_labs)
            transfo = polaffini.get_full_transfo(init_aff, polyAff_svf)  
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(ref_img)
            resampler.SetTransform(transfo) 

            mov_img = resampler.Execute(mov_img)
            mov_img = utils.normalize_intensities(mov_img) 
            mov_img = sitk.GetArrayFromImage(mov_img)[..., np.newaxis]
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            mov_seg = resampler.Execute(mov_seg)
            mov_seg = sitk.GetArrayFromImage(mov_seg)
            mov_seg = utils.one_hot_enc(mov_seg, labels, segtype='array')                    
            
            ref_img =  sitk.GetArrayFromImage(ref_img)[..., np.newaxis]  
            ref_seg = sitk.GetArrayFromImage(ref_seg)
            ref_seg = utils.one_hot_enc(ref_seg, labels, segtype='array')

            ref_imgs += [ref_img.astype(np.float32)]
            ref_segs += [ref_seg]
            mov_imgs += [mov_img.astype(np.float32)]
            mov_segs += [mov_seg]

        ref_imgs = np.stack(ref_imgs, axis=0)
        ref_segs = np.stack(ref_segs, axis=0)
        mov_imgs = np.stack(mov_imgs, axis=0)
        mov_segs = np.stack(mov_segs, axis=0)
        field0 = np.zeros((batch_size, *np.flip(grid_sz), ndims), np.float32)
        
        inputs = [mov_imgs, ref_imgs, mov_segs, ref_segs]
        groundTruths = [ref_imgs, mov_imgs, field0, ref_segs, mov_segs]
        
        yield (inputs, groundTruths)
        
        
