import SimpleITK as sitk
import numpy as np
import scipy.spatial



def estimateTransfo(mov_seg,
                    ref_seg,
                    transfos_type='affine',
                    omit_labs=[2, 41], sigma=15, weight_bg=1e-5, res_sz=4, bg_transfo=True):
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

    volshape = ref_seg.shape
    ndims = len(volshape)
    
    
    labs, _, _ = get_common_labels(ref_seg, mov_seg, omit_labs=omit_labs)
    
    get_volumes = False
    if transfos_type == 'volrot':
        get_volumes = True
    ref_pts, ref_vols = get_label_stats(ref_seg, labs, get_volumes=get_volumes)
    mov_pts, mov_vols = get_label_stats(mov_seg, labs, get_volumes=get_volumes)
    
    if transfos_type != 'translation':
        DT = delaunay_triangulation(ref_pts, labs)    
    
    aff_bg = np.eye(ndims+1)
    if bg_transfo:
        aff_bg = opti_linear_transfo_between_point_sets(ref_pts, mov_pts, transfos_type)  
        aff_bg_inv = np.linalg.inv(aff_bg)      
        mov_pts = np.transpose(np.matmul(aff_bg_inv[:ndims, :ndims], np.transpose(mov_pts))
                               + np.reshape(aff_bg_inv[:ndims, ndims], (ndims,1))) 
    
    if sigma != float('inf'):
        
        coords = np.indices(volshape).reshape(ndims, -1)
        
        polyAff_svf = np.zeros((ndims, coords.shape[1]))
        weight_map_sum = np.zeros(coords.shape[1])
        
        for l, lab in enumerate(labs):    
            
            # Find local optimal local transfo
            if transfos_type == 'translation':
                ind = [i == lab for i in labs]
            else:
                rows_l, _ = np.where(DT == lab)
                connected_labs = np.unique(DT[rows_l])
                ind = [i in connected_labs for i in labs]       
                
            loc_mat = opti_linear_transfo_between_point_sets(ref_pts[ind, :], 
                                                             mov_pts[ind, :], 
                                                             ref_vol = ref_vols[l],
                                                             mov_vol = mov_vols[l],
                                                             transfos_type=transfos_type)   
            loc_mat = scipy.linalg.logm(loc_mat)[:ndims,:]           
            if not np.isrealobj(loc_mat):
                continue  # Skip control point if leading to improper local transfo.
            
            # Compute weight map
            if transfos_type == 'volrot':
                point = ref_pts[l,:] # distance to the control point
            else:
                point = np.mean(ref_pts[ind,:], axis=0) # distance to the center of the neighborhood
            weight_map = np.sum((point[:,None] - coords[:ndims,:])**2, axis=0)         
            weight_map = np.exp(-weight_map/(2*sigma**2)) 

            # Update polyaffine with current field
            polyAff_svf += weight_map * np.matmul(loc_mat, coords)
            weight_map_sum += weight_map
            
        weight_map_sum += weight_bg*sigma*np.sqrt(2*np.pi)
        polyAff_svf = polyAff_svf / weight_map_sum
        polyAff_svf = np.reshape(polyAff_svf, (*volshape, ndims))

    else:
        polyAff_svf = None
        
    return aff_bg, polyAff_svf



def integrate_svf(svf, svf_grid2world, int_steps=7):
    
    ndims = len(svf.shape)-1
    if ndims == 2:
        compose_vector_fields = vfu.compose_vector_fields_2d
    elif ndims == 3:
        compose_vector_fields = vfu.compose_vector_fields_3d
        
    svf_world2grid = np.linalg.inv(svf_grid2world)

    # scaling
    diffeo = svf / (2**int_steps)

    # squaring  
    for _ in range(int_steps):
        compo, _ = compose_vector_fields(diffeo, diffeo, None, svf_world2grid, 1, None)
        diffeo += compo
    
    return diffeo

    
def get_full_transfo(aff_init, polyAff_svf, invert=False):
    
    if polyAff_svf is None:
        if invert:
            transfo_full = aff_init.affine_inv
        else:
            transfo_full = aff_init
            
    else:
        if invert:
            polyAff = integrate_svf(-polyAff_svf)
            transfo_full = sitk.CompositeTransform(polyAff)
            transfo_full.AddTransform(aff_init.GetInverse())
        else:
            polyAff = integrate_svf_lowMem(polyAff_svf)
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
    
    omit_labs = omit_labs + [0]
    labs1 = np.unique(seg_img1)
    labs2 = np.unique(seg_img2)
    for l in omit_labs:
        labs1 = np.delete(labs1, labs1==l)
        labs2 = np.delete(labs2, labs2==l)

    return np.intersect1d(labs1, labs2), labs1, labs2


def get_label_stats(seg, labs, get_centroids=True, get_volumes=False):
    
    ndims = len(seg.shape)
    stats = []
    if get_centroids:
        centroids = np.ones((ndims+1, len(labs)))
        coord_vox = np.indices(seg.shape)
    else:
        centroids = [None]*len(labs)
        
    if get_volumes:
        volumes =  np.zeros((len(labs)))
    else:
        volumes = [None]*len(labs)
    
    # get stats in voxel coordinates
    for l, lab in enumerate(labs):
        lab_mask = seg == lab
        if get_centroids:
            centroids[:ndims, l] = np.mean(coord_vox[:, lab_mask], axis=1)
        if get_volumes:
            volumes[l] = np.sum(lab_mask)
    
    stats = [np.transpose(centroids)[:,:ndims], volumes]
            
    return stats


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
        s = [1]*(ref_pts.shape[1]-1) + [round(np.linalg.det(u)*np.linalg.det(vt))]
        linear_part = np.matmul(np.matmul(u, np.diag(s)),vt)
        
        if transfos_type == 'volrot':
            linear_part *= (mov_vol / ref_vol) ** (1 / ndims)
    
    translat_part = mov_pts_mean - np.matmul(linear_part, ref_pts_mean)
    loc_mat = np.concatenate((linear_part, translat_part[:,np.newaxis]), axis=1)
    loc_mat = np.concatenate((loc_mat, [[0]*(ref_pts.shape[1])+[1]]))
    
    return loc_mat


def get_real_grid_coord(img):
    
    ndims = img.GetDimension()
    id2 = sitk.AffineTransform(ndims)
    id2.SetMatrix(2*np.eye(ndims).ravel())
    id2.GetMatrix()
    
    trsf2disp = sitk.TransformToDisplacementFieldFilter()
    trsf2disp.SetReferenceImage(img)
    
    return sitk.GetArrayFromImage(trsf2disp.Execute(id2))
