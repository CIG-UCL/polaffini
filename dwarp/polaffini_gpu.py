import os
import sys
maindir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(maindir)
import SimpleITK as sitk
import numpy as np
import scipy.spatial
import tensorflow as tf

import polaffini.utils as utils

def estimateTransfo(mov_seg, ref_seg,
                    transfos_type='affine',
                    omit_labs=[], sigma=15, weight_bg=1e-5, down_factor=4, bg_transfo=True):
    """
    Polyaffine image registration through label centroids matching. 

    Parameters
    ----------
    mov_seg : ITK image.
        Moving segmentation.
    ref_seg : ITK image.
        Reference segmentation.
    transfos_type : str, optional
        Type of the local transformations between neightborhoods of feature points.
        'affine', 'rigid', 'translation' or 'volrot' (uses volumes also). The default is 'affine'.
    omit_labs : list, optional
        List of labels to omit. The default is [] (0 (background) is always omitted). 
    sigma : float, optional
        Standard deviation of Gaussian kernel for interpolation ('inf' for linear transfo). The default is 20 mm.
    weight_bg : float, optional
        Weight of the background transformation. The default is 1e-5.
    res_sz : float, optional
        Voxel size (isotropic) in mm to be resliced to for faster computation. The default is 4.
    bg_transfo : bool, optional
        Compute an affine background transformation. The default is True.

    Returns
    -------
    aff_init : ITK affine transfo
        Background affine transformation.
    polyAff_svf : ITK image
        SVF form of the polyaffine transformation.
    """
    
    sigma = float(sigma)
    omit_labs = np.array(omit_labs)
    
    ndims = ref_seg.GetDimension()
    ref_seg_down = sitk.Shrink(ref_seg, [int(down_factor)]*ndims)    
    
    mov_spacing = mov_seg.GetSpacing()
    ref_spacing = ref_seg.GetSpacing()
    mov_matO = tf.constant(utils.get_matOrientation(mov_seg, indexing='numpy'), tf.float32)
    ref_matO = tf.constant(utils.get_matOrientation(ref_seg, indexing='numpy'), tf.float32)
    ref_matO_down = tf.constant(utils.get_matOrientation(ref_seg_down, indexing='numpy'), tf.float32)
    mov_seg = tf.constant(sitk.GetArrayFromImage(mov_seg), tf.int32)
    ref_seg = tf.constant(sitk.GetArrayFromImage(ref_seg), tf.int32)
    
    volshape = ref_seg_down.GetSize()[::-1]
    
    labs, _, _ = get_common_labels(ref_seg, mov_seg, omit_labs=omit_labs)
    labs = labs.numpy().tolist()

    get_volumes = False
    if transfos_type == 'volrot':
        get_volumes = True

    ref_pts, ref_vols = get_label_stats(ref_seg, ref_matO, ref_spacing,
                                        labs, get_volumes=get_volumes)
    mov_pts, mov_vols = get_label_stats(mov_seg, mov_matO, mov_spacing,
                                        labs, get_volumes=get_volumes)

    if transfos_type != 'translation':
        DT = scipy.spatial.Delaunay(tf.transpose(ref_pts))
        DT = tf.constant(DT.simplices)
        
    aff_bg = tf.eye(ndims+1)
    if bg_transfo:
        aff_bg = opti_linear_transfo_between_point_sets(ref_pts, mov_pts, 
                                                        ref_vol=tf.reduce_sum(ref_vols) if ref_vols is not None else None,
                                                        mov_vol=tf.reduce_sum(mov_vols) if mov_vols is not None else None,
                                                        transfos_type=transfos_type)
        aff_bg_inv = tf.linalg.inv(aff_bg)     
        mov_pts = tf.concat((mov_pts, tf.ones((1,len(labs)))), axis=0)
        mov_pts = tf.matmul(aff_bg_inv, mov_pts)[:ndims,:]

    if sigma != float('inf'):
        
        grid = tf.meshgrid(*[tf.range(volshape[d]) for d in range(ndims)], indexing='ij')
        grid = [tf.cast(grid[d], dtype=tf.float32) for d in range(ndims)]
        grid = tf.expand_dims(tf.stack(grid+[tf.ones(volshape)], axis=-1), axis=-1)
        grid = tf.matmul(ref_matO_down, grid)

        polyAff_svf = tf.zeros((*volshape, ndims))
        weight_map_sum = tf.zeros((*volshape, 1))
        
        for l, lab in enumerate(labs):    
            
            # Find local optimal local transfo
            if transfos_type == 'translation':
                ind = labs == lab
            else:
                
                rows_l = tf.where(DT == l)[:,0]
                
                ind = tf.gather(DT, rows_l, axis=0)
                ind = tf.unique(tf.reshape(ind, [-1]))[0]  
                
            ref_neighbors_pts = tf.gather(ref_pts, ind, axis=1)
            mov_neighbors_pts = tf.gather(mov_pts, ind, axis=1)
            
            loc_mat = opti_linear_transfo_between_point_sets(ref_neighbors_pts, 
                                                             mov_neighbors_pts, 
                                                             ref_vol=ref_vols[l] if ref_vols is not None else None,
                                                             mov_vol=mov_vols[l] if mov_vols is not None else None,
                                                             transfos_type=transfos_type) 
            
            loc_mat = tf.cast(loc_mat, tf.complex64)
            loc_mat = tf.linalg.logm(loc_mat)
            if tf.reduce_max(tf.math.imag(loc_mat)) > 1e-4:
                continue  # Skip control point if leading to improper local transfo.
            loc_mat = tf.math.real(loc_mat)[:ndims,:]
            
            # Compute weight map
            if transfos_type in ('volrot', 'translation'):
                point = tf.expand_dims(ref_pts[:,l], axis=1) # distance to the control point
            else:
                point = tf.reduce_mean(ref_neighbors_pts, axis=1, keepdims=True) # distance to the center of the neighborhood
            weight_map = tf.reduce_sum((point-grid[..., :ndims, :])**2, axis=-2)     
            print(weight_map.shape)
            weight_map = tf.exp(-weight_map / (2*sigma**2)) 
            print(weight_map.shape)
            # Update polyaffine with current field
            print(polyAff_svf.shape, weight_map.shape, (tf.matmul(loc_mat, grid)[..., 0]).shape, loc_mat.shape, grid.shape)
            polyAff_svf += weight_map * tf.matmul(loc_mat, grid)[..., 0]
            weight_map_sum += weight_map

        weight_map_sum += weight_bg*sigma*tf.sqrt(2*np.pi)
        polyAff_svf = polyAff_svf / weight_map_sum
        
        polyAff_svf = sitk.GetImageFromArray(polyAff_svf)
        polyAff_svf.SetOrigin(ref_seg_down.GetOrigin())
        polyAff_svf.SetSpacing(ref_seg_down.GetSpacing())
        polyAff_svf.SetDirection(ref_seg_down.GetDirection())
    
    else:
        polyAff_svf = None
        
    aff_bg_tr = sitk.AffineTransform(ndims)
    aff_bg_tr.SetMatrix(np.reshape(aff_bg[:ndims,:ndims], -1).tolist())
    aff_bg_tr.SetTranslation(aff_bg[:ndims,ndims].numpy().tolist())
    
    return aff_bg_tr, polyAff_svf
 


def get_common_labels(seg1, seg2, omit_labs=[]):
    
    omit_labs = np.append(omit_labs, 0)
    
    seg1 = tf.cast(seg1, dtype=tf.int32)
    seg2 = tf.cast(seg2, dtype=tf.int32)
    labs1, _ = tf.unique(tf.reshape(seg1,(-1)))
    labs2, _ = tf.unique(tf.reshape(seg2,(-1)))
    print(labs1)
    print(labs2)
    for l in omit_labs:
        labs1 = labs1[labs1 != l]
        labs2 = labs2[labs2 != l]
    
    inter_labs = tf.sets.intersection(labs1[None,:],labs2[None,:])
    inter_labs = tf.squeeze(tf.sparse.to_dense(inter_labs))
    
    return inter_labs, labs1, labs2


def get_label_stats(seg, matO, spacing, labs, get_centroids=True, get_volumes=False):
    
    centroids = [] if get_centroids else None
    volumes = [] if get_volumes else None
    print(labs)
    
    for lab in labs:

        lab_mask = seg == lab
        lab_coords = tf.where(lab_mask) 

        if get_centroids:
            centroid = tf.reduce_mean(tf.cast(lab_coords, tf.float32), axis=0)
            print(lab, centroid)
            centroids.append(centroid)

        if get_volumes:
            volumes.append(tf.shape(lab_coords)[0])
    
    if get_centroids:
        centroids = tf.stack(centroids, axis=1)  
        centroids = tf.concat([centroids, tf.ones((1, len(labs)), dtype=tf.float32)], axis=0)
        centroids = tf.matmul(tf.cast(matO, tf.float32), centroids)[:-1, :]
        
    if get_volumes:
        volumes = tf.cast(tf.stack(volumes), tf.float32)
        volumes *= tf.reduce_prod(spacing)

    return centroids, volumes


def opti_linear_transfo_between_point_sets(ref_pts, mov_pts,
                                           ref_vol=None, mov_vol=None, transfos_type='affine'):
    
    ndims = ref_pts.shape[0]
    ref_pts_mean = tf.reduce_mean(ref_pts, axis=1, keepdims=True)
    mov_pts_mean = tf.reduce_mean(mov_pts, axis=1, keepdims=True)
    ref_pts = ref_pts - ref_pts_mean
    mov_pts = mov_pts - mov_pts_mean

    if transfos_type == 'translation':
        linear_part = tf.eye(ndims)
        
    elif transfos_type == 'affine':
        linear_part = tf.transpose(tf.linalg.lstsq(tf.transpose(ref_pts), 
                                                   tf.transpose(mov_pts)))
        
    elif transfos_type in ('rigid','volrot'):
        # see Pennec PhD or Horn 1987 for rigid
        corr = tf.matmul(mov_pts, tf.transpose(ref_pts))
        _, u, v = tf.linalg.svd(corr)
        s = [1]*(ndims-1) + [tf.round(tf.linalg.det(u)*tf.linalg.det(v))]
        s = tf.linalg.diag(s)
        linear_part = tf.matmul(tf.matmul(u, s),tf.transpose(v))
        
        if transfos_type == 'volrot':
            linear_part *= (mov_vol / ref_vol) ** (1 / ndims)
    
    translat_part = mov_pts_mean - tf.matmul(linear_part, ref_pts_mean)
    loc_mat = tf.concat((linear_part, translat_part), axis=1)
    loc_mat = tf.concat((loc_mat, [[0]*(ndims)+[1]]), axis=0)

    return loc_mat

