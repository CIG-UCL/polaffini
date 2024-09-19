import SimpleITK as sitk
import numpy as np
import scipy.spatial
from scipy.linalg import logm, expm
import copy
from . import utils
import time

#%% 

def estimateTransfo(mov_seg, ref_seg, alpha=1,
                    transfos_type='affine', dist='center', out_jac=False,
                    omit_labs=[], sigma=15, weight_bg=1e-5, down_factor=4, bg_transfo=True):
    """
    Polyaffine image registration through label centroids matching. 

    Parameters
    ----------
    mov_seg : ITK image.
        Moving segmentation.
    ref_seg : ITK image.
        Reference segmentation.
    alpha : float, optional
        Position of the overall transformation on the diffeomorphic path 
        from identity to the transfo from moving to reference (e.g. use 0.5 for half-way registration).
        The default is 1.
    transfos_type : str, optional
        Type of the local transformations between neightborhoods of feature points.
        'affine', 'rigid', 'translation' or 'volrot' (uses volumes also). The default is 'affine'.
    dist : str, optional
        Distance used for the weight maps.
        'center': distance to neighborhood center, or 'maurer': distance to label. The default is 'center'.
    out_jac : bool, optional
        Also output the jacobian of the polyaffine SVF. The default is False.
    omit_labs : list, optional
        List of labels to omit. The default is [] (0 (background) is always omitted). 
    sigma : float, optional
        Standard deviation of Gaussian kernel for interpolation ('inf' for linear transfo). The default is 20 mm.
    weight_bg : float, optional
        Weight of the background transformation. The default is 1e-5.
    down_factor : int, optional
        Downsampling factor for faster computation. The default is 4.
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
    
    ref_seg = sitk.Cast(ref_seg, sitk.sitkInt64)
    ndims = ref_seg.GetDimension()
    mov_seg = sitk.Cast(mov_seg, sitk.sitkInt64)

    labs, _, _ = get_common_labels(ref_seg, mov_seg, omit_labs=omit_labs)

    ref_seg_down = sitk.Shrink(ref_seg, [int(down_factor)]*ndims)    
    
    get_volumes = False
    if transfos_type == 'volrot':
        get_volumes = True

    ref_pts, ref_vols = get_label_stats(ref_seg, labs, get_volumes=get_volumes)
    mov_pts, mov_vols = get_label_stats(mov_seg, labs, get_volumes=get_volumes)

    if transfos_type != 'translation':
        DT = delaunay_triangulation(ref_pts, labs)    
    
    aff_init = sitk.AffineTransform(ndims)
    if bg_transfo:
        transfo_aff = opti_linear_transfo_between_point_sets(ref_pts, mov_pts, 
                                                             ref_vol=np.sum(ref_vols) if ref_vols is not None else None,
                                                             mov_vol=np.sum(mov_vols) if mov_vols is not None else None,
                                                             transfos_type=transfos_type) 
        transfo_aff_mov = expm(alpha*logm(transfo_aff)) 
        transfo_aff_mov_inv = expm(-alpha*logm(transfo_aff)) 
        transfo_aff_ref = expm((1-alpha)*logm(transfo_aff)) 

        aff_init.SetMatrix((transfo_aff_mov[0:ndims, 0:ndims]).ravel())
        aff_init.SetTranslation(transfo_aff_mov[0:ndims, ndims])        
        mov_pts = np.transpose(np.matmul(transfo_aff_mov_inv[0:ndims, 0:ndims], np.transpose(mov_pts))
                                + np.reshape(transfo_aff_mov_inv[0:ndims, ndims], (ndims,1))) 
        ref_pts = np.transpose(np.matmul(transfo_aff_ref[0:ndims, 0:ndims], np.transpose(ref_pts))
                                + np.reshape(transfo_aff_ref[0:ndims, ndims], (ndims,1)))    

    if sigma != float('inf'):
        
        weight_map_sum = sitk.Image(ref_seg_down.GetSize(), sitk.sitkFloat64)
        weight_map_sum.CopyInformation(ref_seg_down) 
        
        polyAff_svf_jac = None
        if out_jac:
            polyAff_svf_jac = sitk.GetImageFromArray(np.zeros(ref_seg_down.GetSize()[::-1] + (ndims**2,)), isVector=True)
            polyAff_svf_jac.CopyInformation(ref_seg_down)
            
        loc_transfo = sitk.AffineTransform(ndims)
        polyAff_svf = sitk.Image(ref_seg_down.GetSize(), sitk.sitkVectorFloat64)
        polyAff_svf.CopyInformation(ref_seg_down)
        trsf2disp = sitk.TransformToDisplacementFieldFilter()
        trsf2disp.SetReferenceImage(ref_seg_down)
        
        if dist == 'center':
            id2 = sitk.AffineTransform(ndims)
            id2.SetMatrix(2*np.eye(ndims).ravel())
        elif dist == 'maurer':
            maurerdist = sitk.SignedMaurerDistanceMapImageFilter()
            maurerdist.SetSquaredDistance(True)
            maurerdist.SetUseImageSpacing(True)

        for l, lab in enumerate(labs):  

            # Find local optimal local transfo
            if transfos_type == 'translation':
                ind = labs == lab
            else:
                rows_l, _ = np.where(DT == lab)
                connected_labs = np.unique(DT[rows_l])
                ind = [i in connected_labs for i in labs]       
            
            if transfos_type == 'affine' and sum(ind) < 4:
                continue
            if transfos_type == 'rigid' and sum(ind) < 2:
                continue
            
            loc_mat = opti_linear_transfo_between_point_sets(ref_pts[ind, :], 
                                                             mov_pts[ind, :], 
                                                             ref_vol=ref_vols[l] if ref_vols is not None else None,
                                                             mov_vol=mov_vols[l] if mov_vols is not None else None,
                                                             transfos_type=transfos_type)   
            loc_mat = alpha * logm(loc_mat)
            if not np.isrealobj(loc_mat):
                continue
            
            # Compute weight map
            if dist == 'center':
                if transfos_type == 'volrot':
                    id2.SetTranslation(-ref_pts[l,:]) # distance to the control point
                else:
                    id2.SetTranslation(-np.mean(ref_pts[ind,:], axis=0)) # distance to the center of the neighborhood
                weight_map = trsf2disp.Execute(id2)    
                weight_map = sitk.VectorMagnitude(weight_map)**2
            elif dist == 'maurer':
                weight_map = ref_seg_down == lab
                weight_map = maurerdist.Execute(weight_map)
                weight_map = sitk.Cast(weight_map, sitk.sitkFloat32)            
            weight_map = sitk.Exp(-weight_map/(2*sigma**2)) 
            
            if out_jac:
                loc_mat_flat = np.ravel(loc_mat[:ndims,:ndims])
                polyAff_svf_jac += sitk.Compose([weight_map * loc_mat_flat[d] for d in range(ndims**2)])
                
            # Update polyaffine with current field
            loc_transfo.SetMatrix((loc_mat[0:ndims, 0:ndims] + np.eye(ndims)).ravel())
            loc_transfo.SetTranslation(loc_mat[0:ndims, ndims])
            loc_svf = trsf2disp.Execute(loc_transfo)
            polyAff_svf += sitk.Compose([sitk.VectorIndexSelectionCast(loc_svf,d)*weight_map for d in range(ndims)])
            weight_map_sum += weight_map
        weight_map_sum += weight_bg*sigma*np.sqrt(2*np.pi)

        polyAff_svf = sitk.Compose([sitk.VectorIndexSelectionCast(polyAff_svf, d)/weight_map_sum for d in range(ndims)])
        if out_jac:
            polyAff_svf_jac = sitk.Compose([sitk.VectorIndexSelectionCast(polyAff_svf_jac, d)/weight_map_sum for d in range(ndims**2)])
    
    else:
        polyAff_svf = None
    
    if out_jac:
        return aff_init, polyAff_svf, polyAff_svf_jac
    else:
        return aff_init, polyAff_svf


#%% Utils


def scale_aff_transfo(aff_transfo, alpha):
    
    ndims = aff_transfo.GetDimension()
    
    aff_mat = np.reshape(aff_transfo.GetParameters()[:-ndims], (ndims,ndims))
    aff_mat = np.c_[aff_mat, aff_transfo.GetParameters()[ndims**2:]]
    aff_mat = np.r_[aff_mat, np.reshape([0]*ndims+[1], (1,ndims+1))]
    
    aff_mat_scaled = expm(alpha*logm(aff_mat))
    aff_transfo_scaled = sitk.AffineTransform(ndims)
    aff_transfo_scaled.SetMatrix(np.ravel(aff_mat_scaled[:ndims,:ndims]))
    aff_transfo_scaled.SetTranslation(aff_mat_scaled[:ndims,ndims])
    
    return aff_transfo_scaled

    
def get_full_transfo(aff_init, polyAff_svf, invert=False, alpha=1):
    ndims = aff_init.GetDimension()
    if alpha != 1:
        aff_init = scale_aff_transfo(aff_init, alpha)
        polyAff_svf = sitk.Compose([alpha * sitk.VectorIndexSelectionCast(polyAff_svf, d) for d in range(ndims)])
    if polyAff_svf is None:
        if invert:
            transfo_full = aff_init.GetInverse()
        else:
            transfo_full = aff_init
    else:
        if invert:
            polyAff = utils.integrate_svf(-polyAff_svf)
            transfo_full = sitk.CompositeTransform(polyAff)
            transfo_full.AddTransform(aff_init.GetInverse())
        else:
            polyAff = utils.integrate_svf(polyAff_svf)
            transfo_full = sitk.CompositeTransform(aff_init)
            transfo_full.AddTransform(polyAff)
    return transfo_full


def write_dispField(transfo, out_file, ref_image=None):
    
    tr2disp = sitk.TransformToDisplacementFieldFilter()
    if ref_image is None:
        ndims = transfo.GetDimension()   
        size = [int(transfo.GetFixedParameters()[d]) for d in range(ndims)]
        origin = transfo.GetFixedParameters()[ndims:2*ndims]
        spacing = transfo.GetFixedParameters()[2*ndims:3*ndims]
        direction = transfo.GetFixedParameters()[3*ndims:]
        
        tr2disp.SetSize(size)
        tr2disp.SetOutputOrigin(origin)
        tr2disp.SetOutputSpacing(spacing)
        tr2disp.SetOutputDirection(direction)
        
    else:
        tr2disp.SetReferenceImage(ref_image)
    
    sitk.WriteImage(tr2disp.Execute(transfo), out_file)    


def get_common_labels(seg_img1, seg_img2, omit_labs=[]):
    
    omit_labs = np.append(omit_labs, 0)
    labs1 = np.unique(sitk.GetArrayFromImage(seg_img1))
    labs2 = np.unique(sitk.GetArrayFromImage(seg_img2))
    for l in omit_labs:
        labs1 = np.delete(labs1, labs1==l)
        labs2 = np.delete(labs2, labs2==l)

    return np.intersect1d(labs1, labs2), labs1, labs2


def get_label_stats(seg_img, labs, get_centroids=True, get_volumes=False):
    
    ndims = seg_img.GetDimension()
    seg_stat = sitk.LabelIntensityStatisticsImageFilter()
    seg_stat.Execute(seg_img, seg_img)

    centroids = np.zeros((len(labs), ndims)) if get_centroids else None
    volumes =  np.zeros((len(labs)))  if get_volumes else None
        
    for l, lab in enumerate(labs):
        if get_centroids:
            centroids[l, :] = seg_stat.GetCenterOfGravity(int(lab))
        if get_volumes:
            volumes[l] = seg_stat.GetPhysicalSize(int(lab))

    return centroids, volumes


def delaunay_triangulation(points, labs):
    
    tri = scipy.spatial.Delaunay(points)

    DTlab = np.zeros_like(tri.simplices);
    for i, lab in enumerate(labs):
        DTlab[tri.simplices==i] = lab
        
    return DTlab


def opti_linear_transfo_between_point_sets(ref_pts, mov_pts,
                                           ref_vol=None, mov_vol=None, transfos_type='affine'):

    ref_pts_mean = np.mean(ref_pts, axis=0)
    mov_pts_mean = np.mean(mov_pts, axis=0)
    ref_pts = ref_pts - ref_pts_mean
    mov_pts = mov_pts - mov_pts_mean
    ndims = ref_pts.shape[1]
    
    if transfos_type == 'translation':
        linear_part = np.eye(ndims)
        
    elif transfos_type == 'affine':
        linear_part = np.transpose(np.linalg.lstsq(ref_pts, mov_pts, rcond=None)[0])
        
    elif transfos_type in ('rigid','volrot'):
        # see Pennec PhD or Horn 1987 for rigid
        corr = np.matmul(np.transpose(mov_pts), ref_pts)
        u, d, vt = np.linalg.svd(corr)
        s = [1]*(ndims-1) + [round(np.linalg.det(u)*np.linalg.det(vt))]
        linear_part = np.matmul(np.matmul(u, np.diag(s)),vt)
        if transfos_type == 'volrot':
            linear_part *= (mov_vol / ref_vol) ** (1 / ndims)
    
    translat_part = mov_pts_mean - np.matmul(linear_part, ref_pts_mean)
    loc_mat = np.concatenate((linear_part, translat_part[:,np.newaxis]), axis=1)
    loc_mat = np.concatenate((loc_mat, [[0]*ndims+[1]]))
    
    return loc_mat


    
def get_full_svf(aff_init, polyAff_svf, polyAff_svf_jac=None, bch_order=2):
    
    ndims = polyAff_svf.GetDimension()
    volshape = polyAff_svf.GetSize()[::-1]
    origin = polyAff_svf.GetOrigin()
    direction = polyAff_svf.GetDirection()
    spacing = polyAff_svf.GetSpacing()
    
    aff_svf = utils.aff_to_svf()
    
    aff_mat = utils.aff_tr2mat(aff_init)
    log_aff_mat = logm(aff_mat)
    log_aff = utils.aff_mat2tr(log_aff_mat)
    aff_svf = utils.aff_tr2field(log_aff)
    aff_svf = sitk.GetArrayFromImage(aff_svf)
    aff_svf = aff_svf.reshape(volshape + (ndims,1))

    polyAff_svf = sitk.GetArrayFromImage(polyAff_svf)
    polyAff_svf = polyAff_svf.reshape(volshape + (ndims,1))

    full_svf = aff_svf  + polyAff_svf   # BCH formula order 1
    
    if bch_order > 1: 
        aff_svf_jac = np.ones(volshape + (ndims, ndims))
        aff_svf_jac *= log_aff_mat[:ndims,:ndims].reshape([1]*ndims + [ndims,ndims])
        
        polyAff_svf_jac = sitk.GetArrayFromImage(polyAff_svf_jac).reshape(aff_svf_jac.shape)
    
        lie_bracket = np.matmul(aff_svf_jac, polyAff_svf) - np.matmul(polyAff_svf_jac, aff_svf)
    
        full_svf += lie_bracket / 2      # BCH formula order 2
    
    full_svf = full_svf.reshape(volshape + (ndims,)) 
    full_svf = sitk.GetImageFromArray(full_svf, isVector=True)
    full_svf.SetOrigin(origin)
    full_svf.SetDirection(direction)
    full_svf.SetSpacing(spacing)
    
    return full_svf
    
    
    