import numpy as np
import os
import glob
import SimpleITK as sitk        
import polaffini.utils as utils
import polaffini.polaffini as polaffini
import dwarp.augmentation as augmentation

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
        Path to weight map associated to target atlas. Default: None.
    batch_size : int, optional.
        Batch size. Default: 1.

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
                          one_hot=True,
                          dtype=np.float32,
                          batch_size=1):
    """
    Generator for moving images and a single reference. 
    Moving images are assumed to have already undergone affine or 
    polaffini initialization (typically using polaffini_set2template.py). 
    Therefore the moving and target images are already resampled in the same grid. 
    
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
        Path to weight map associated to target atlas. Default: None.
    one_hot : bool, optional.
        Switch indicating if the segemntations are encoded in one-hot or not. Default: True.
    batch_size : int, optional.
        Batch size. Default: 1.

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

        ndims = ref.GetDimension()
        ref = sitk.GetArrayFromImage(ref)[np.newaxis,..., np.newaxis]
        ref = ref.astype(dtype)
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
            mov_img = mov_img.astype(dtype)
            mov_imgs += [mov_img]         
            if is_seg:
                mov_seg = sitk.ReadImage(mov_seg_files[i])
                mov_seg = sitk.GetArrayFromImage(mov_seg)
                if one_hot:
                    mov_seg = np.transpose(mov_seg,[*range(1,ndims+1)]+[0])[np.newaxis,...]
                else:
                    mov_seg = mov_seg[np.newaxis,..., np.newaxis]   
                mov_seg = mov_seg.astype(dtype)
                mov_segs += [mov_seg] 

        mov_imgs = np.concatenate(mov_imgs, axis=0)
        
        if is_seg:
            mov_segs = np.concatenate(mov_segs, axis=0)

            ref_seg = sitk.GetArrayFromImage(ref_seg)
            if one_hot:
                ref_seg = np.transpose(ref_seg,[*range(1,ndims+1)]+[0])[np.newaxis,...]
            else:
                ref_seg = ref_seg[np.newaxis,..., np.newaxis]
            ref_seg = ref_seg.astype(dtype)
            ref_seg = np.concatenate([ref_seg]*batch_size, axis=0)
         
        inputs = [mov_imgs]

        if is_weight:
            groundTruths = [np.concatenate([ref, weight], axis=-1)]
        else:
            groundTruths = [ref]
        
        if is_seg:
            inputs += [mov_segs]
            groundTruths += [ref_seg]

        field0 = np.zeros((*mov_imgs.shape[:-1], ndims), np.float32)    
        groundTruths += [field0]
        
        yield (inputs, groundTruths)
        
        

def pair_polaffini(mov_files, 
                   mov_seg_files,
                   vox_sz=[2,2,2],
                   grid_sz=[96,128,96],
                   labels='dkt',
                   aug_axes = False,
                   one_hot = False,
                   sdf = False, # signed distance field
                   polaffini_sigma=15,
                   polaffini_downf=4,
                   polaffini_omit_labs=[2,23,41],      
                   batch_size=1):
    """
    Generator for a pair of moving images. 
    Paired POLAFFINI is included.
    
    Parameters
    ----------
    mov_files : list (str).
        Path to moving images.
    mov_seg-files : list (str), optional.
        Path to moving segmentations.
    vox_sz : list (int), optional
        Voxel size to resample the images to. Default: [2,2,2].
    grid_sz : TYPE, optional
        Grid size to crop / pad the images to. Default: [96,128,96].
    labels : list (int), optional
        List of labels. Default: dkt labels.
    aug_axes : bool, optional
        Do axes augmentation through permutation. Default: False.
    one_hot : bool, optional
        Do one-hot encoding of the segmentations. Default: False.
    sdf : TYPE, optional
        Compute surface distance fields to be used instead of label images. Default: False.
    polaffini_sigma : float, optional
        Sigma smoothness parameter for polaffini. Default: 15.
    polaffini_downf : int, optional
        Down sampling factor for polaffini. Default: 4.
    polaffini_omit_labs : list (int), optional
        Labels to omit for polaffini. Default: [2,23,41].
    batch_size : int, optional
        Batch size. Default: 1.

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
    if isinstance(labels, str):
        labels = get_labels(labels)
    nlabs = len(labels)

    while True:
                       
        ref_imgs = [] 
        ref_segs = []
        mov_imgs = [] 
        mov_segs = []
        for _ in range(batch_size): 
            i, j = np.random.choice(range(len(mov_files)), size=2, replace=False)
            # print(mov_files[i])
            # print(mov_files[j])
            ref_img = sitk.ReadImage(mov_files[i])
            ref_seg = sitk.ReadImage(mov_seg_files[i])
            mask = sitk.BinaryThreshold(ref_seg, 1, 23) + sitk.BinaryThreshold(ref_seg, 25, 1e9)
            mask = sitk.BinaryMorphologicalClosing(mask, [6]*3)
            ref_img = sitk.Mask(ref_img, mask)
            
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
                
            ref_img = utils.change_img_res(ref_img, vox_sz)
            ref_img = utils.change_img_size(ref_img, grid_sz)
            ref_img = utils.normalize_intensities(ref_img)
            ref_seg = utils.change_img_res(ref_seg, vox_sz, interp=sitk.sitkNearestNeighbor)
            ref_seg = utils.change_img_size(ref_seg, grid_sz)
   
            mov_img = sitk.ReadImage(mov_files[j])
            mov_seg = sitk.ReadImage(mov_seg_files[j])
            mask = sitk.BinaryThreshold(mov_seg, 1, 23) + sitk.BinaryThreshold(mov_seg, 25, 1e9)
            mask = sitk.BinaryMorphologicalClosing(mask, [6]*3)
            mov_img = sitk.Mask(mov_img, mask)

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
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            mov_seg = resampler.Execute(mov_seg)

            ref_img = sitk.GetArrayFromImage(ref_img)[..., np.newaxis]  
            mov_img = sitk.GetArrayFromImage(mov_img)[..., np.newaxis]
            if sdf or one_hot:
                ref_seg = [ref_seg==l for l in labels]
                mov_seg = [mov_seg==l for l in labels]
                if sdf:
                    ref_seg = [sitk.SignedMaurerDistanceMap(ref_seg[l], squaredDistance=False, useImageSpacing=True) for l in range(nlabs)]
                    mov_seg = [sitk.SignedMaurerDistanceMap(mov_seg[l], squaredDistance=False, useImageSpacing=True) for l in range(nlabs)]             
                ref_seg = [sitk.GetArrayFromImage(ref_seg[l]) for l in range(nlabs)]
                mov_seg = [sitk.GetArrayFromImage(mov_seg[l]) for l in range(nlabs)]
                ref_seg = np.stack(ref_seg, axis=-1)   
                mov_seg = np.stack(mov_seg, axis=-1)               
            else:
                ref_seg = sitk.GetArrayFromImage(ref_seg)[..., np.newaxis]
                mov_seg = sitk.GetArrayFromImage(mov_seg)[..., np.newaxis]

            ref_imgs += [ref_img.astype(np.float32)]
            mov_imgs += [mov_img.astype(np.float32)]
            if sdf:
                ref_segs += [ref_seg.astype(np.float32)]
                mov_segs += [mov_seg.astype(np.float32)]
            else:
                ref_segs += [ref_seg.astype(np.uint16)]
                mov_segs += [mov_seg.astype(np.uint16)]

        ref_imgs = np.stack(ref_imgs, axis=0)
        ref_segs = np.stack(ref_segs, axis=0)
        mov_imgs = np.stack(mov_imgs, axis=0)
        mov_segs = np.stack(mov_segs, axis=0)

        field0 = np.zeros((batch_size, *np.flip(grid_sz), ndims), np.float32)
        
        inputs = [mov_imgs, ref_imgs, mov_segs, ref_segs]
        groundTruths = [ref_imgs, mov_imgs, ref_segs, mov_segs, field0]
        
        yield (inputs, groundTruths)
        


def get_labels(lut='dkt'):
    
    if lut == 'dkt':
        labels=[2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28, 31, 41, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
                1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035,
                2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2034, 2035]
        
        return labels    
