import numpy as np
import SimpleITK as sitk

    
def pad_image(img, k=5, outSize=None, bg_val=0):
    """
    Pad an image such that image size along each dimension is a multiple of 2^k.
    """
    inSize = np.array(img.GetSize(), dtype=np.float32)
    if outSize is None:
        if k is None:
            outSize = np.power(2, np.ceil(np.log(inSize)/np.log(2)))
        else:
            outSize = np.ceil(inSize / 2**k) * 2**k
            
    lowerPad = np.round((outSize - inSize) / 2)
    upperPad = outSize - inSize - lowerPad
    
    padder = sitk.ConstantPadImageFilter()
    padder.SetConstant(bg_val)
    padder.SetPadLowerBound(lowerPad.astype(int).tolist())
    padder.SetPadUpperBound(upperPad.astype(int).tolist())
    
    paddedImg = padder.Execute(img)
    
    return paddedImg


def one_hot_enc(seg, labs, segtype='itkimg', dtype=np.int8):
    """
    segtype can be itkimg or array
    """
    if segtype == 'itkimg':
        ndims = seg.GetDimension()
        origin = seg.GetOrigin() + (0,)
        direction = seg.GetDirection()
        direction = np.eye(ndims+1)
        direction[0:ndims,0:ndims] = np.reshape(seg.GetDirection(),[ndims,ndims])
        direction = np.ravel(direction)
        spacing = seg.GetSpacing() + (1,)
        seg = sitk.GetArrayFromImage(seg)
    seg = [seg==lab for lab in labs]
    seg = np.stack(seg, axis=-1)
    seg = seg.astype(dtype)
    if segtype == 'itkimg':    
        seg = np.transpose(seg, [ndims] + [*range(ndims)])
        seg = sitk.GetImageFromArray(seg, isVector=False)
        seg.SetOrigin(origin)
        seg.SetDirection(direction)
        seg.SetSpacing(spacing)
    
    return seg


def get_matOrientation(img, decomp=False):
    
    ndims = img.GetDimension() 
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    direction = img.GetDirection()
    
    if decomp:
        return (origin, spacing, direction)
    
    else:
        matO = np.matmul(np.diag(spacing), np.reshape(direction,(ndims, ndims)))
        matO = np.concatenate((matO, np.reshape(origin, (ndims,1))), axis=1)
        matO = np.concatenate((matO, np.reshape([0]*ndims+[1], (1,ndims+1))), axis=0)
        return matO
    
    
def decomp_matOrientation(matO):
    """
    Decompose the orientation matrix into origin, scaling and direction.
    """
    
    ndims = matO.shape[1]-1
    mat = matO[0:ndims, 0:ndims]   
    spacing = np.linalg.norm(mat, axis=1)
    direction = np.squeeze(np.asarray(np.matmul(np.diag(1/spacing), mat)))
    origin = np.squeeze(np.asarray(matO[0:ndims, ndims]))
    
    return (origin, spacing, direction.ravel())


def resample_image(img, size, matO, interp):
    """
    Resample an image in a new grid defines by its size and orientation using a given interpolation method.
    """
    
    origin, spacing, direction = decomp_matOrientation(matO)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size)
    resampler.SetOutputOrigin(origin.tolist())
    resampler.SetOutputDirection(direction.tolist())
    resampler.SetOutputSpacing(spacing.tolist())   
    resampler.SetInterpolator(interp)
    
    return resampler.Execute(img)


def grid_img(volshape, omitdim=[2], spacing=5):
    g = np.zeros(volshape)
    
    for i in range(0,volshape[0], spacing):
        if 0 not in omitdim:
            g[i,:,:] = 1
    for j in range(0,volshape[1], spacing):
        if 1 not in omitdim:
            g[:,j,:] = 1
    for k in range(0,volshape[2], spacing):
        if 2 not in omitdim:
            g[:,:,k] = 1 
    return g
    

def normalize_intensities(img, wmin=0, wmax=None, omin=0, omax=1, dtype=sitk.sitkFloat32):
    """
    Normalize intensities of an image between 0 and 1.
    """
    listed = True
    if not isinstance(img, (list, tuple)):
        img = [img]
        listed = False
    
    intensityFilter = sitk.IntensityWindowingImageFilter() 
    intensityFilter.SetOutputMaximum(omax)
    intensityFilter.SetOutputMinimum(omin)
    intensityFilter.SetWindowMinimum(wmin)
        
    for i in range(len(img)):
        if dtype is not None:
            img[i] = sitk.Cast(img[i], dtype)
    
    if wmax is None:
        minmaxFilter = sitk.MinimumMaximumImageFilter()
        wmax = -np.inf
        for i in range(len(img)):
            minmaxFilter.Execute(img[i])
            wmax = np.max((wmax, minmaxFilter.GetMaximum()))
    intensityFilter.SetWindowMaximum(wmax)
    
    for i in range(len(img)):
        img[i] = intensityFilter.Execute(img[i])
        
    if not listed:
        img = img[0]
    
    return img