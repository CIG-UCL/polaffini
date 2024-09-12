import numpy as np
import SimpleITK as sitk
import os
import pathlib
import tempfile
import nibabel as nib
import copy
from scipy.linalg import logm, expm

class imageIO:
    # SimpleITK is preferred to nibabel but the latter is used for freesurfer formats.
    
    def __init__(self, filename, convertvia='nii', tmpdir=None):
        self.convertvia = convertvia
        if convertvia is not None:
            if convertvia[0] != '.':
                self.convertvia = '.' + convertvia
        self.tmpdir = tmpdir
        self.filename = pathlib.Path(filename).expanduser().as_posix()    
        _, self.ext = self._splitext()
        self.is_mgx = self.ext in ('.mgh', '.mgz')
        
    def read(self):
        if self.is_mgx:
            tmpfile = self._get_tmpfile()
            self._convert(self.filename, tmpfile)
            img = sitk.ReadImage(tmpfile)
            os.remove(tmpfile)
        else:
            img = sitk.ReadImage(self.filename)
        return img
    
    def write(self, img):
        if self.is_mgx is not None:
            tmpfile = self._get_tmpfile()
            sitk.WriteImage(img, tmpfile)
            self._convert(tmpfile, self.filename)
            os.remove(tmpfile)
        else:
            sitk.Write(img, self.filename)
     
    def _get_tmpfile(self):
        if self.tmpdir is None:
            tmpdir, _ = os.path.split(self.filename)
        else:
            tmpdir = pathlib.Path(self.tmpdir).expanduser().as_posix()
        tmpfile = tempfile.NamedTemporaryFile(dir=tmpdir, prefix='tmp_')
        tmpfile = tmpfile.name + self.convertvia
        return tmpfile
        
    def _convert(self, filename1, filename2):
        img = nib.load(filename1)
        nib.save(img, filename2)

    def _splitext(self):     
        filename, ext = os.path.splitext(self.filename)
        if ext == '.gz':
            filename, ext = os.path.splitext(filename)
            ext = ext + '.gz'       
        return filename, ext
    

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


def get_matOrientation(img, indexing='itk'):
    # CAREFUL: 
    # It's from itk indices to physical space by default.
    # For numpy indices to physical space, use indexing='numpy'.

    ndims = img.GetDimension() 
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    direction = img.GetDirection()
    
    perm = np.eye(ndims+1)
    if indexing == 'numpy':
        perm[:ndims,:ndims] = np.eye(ndims)[::-1]
    
    matO = np.matmul(np.reshape(direction,(ndims, ndims)), np.diag(spacing))
    matO = np.concatenate((matO, np.reshape(origin, (ndims,1))), axis=1)
    matO = np.concatenate((matO, np.reshape([0]*ndims+[1], (1,ndims+1))), axis=0)
    matO = np.matmul(matO, perm)

    return matO
  

    
def decomp_matOrientation(matO):
    """
    Decompose the orientation matrix into origin, scaling and direction.
    """
    
    ndims = matO.shape[1]-1
    mat = matO[0:ndims, 0:ndims]   
    spacing = np.linalg.norm(mat, axis=0)
    direction = np.squeeze(np.asarray(np.matmul(mat, np.diag(1/spacing))))
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


def change_img_res(img, vox_sz=[2,2,2], interp=sitk.sitkLinear):
    """
    Change the resolution while keeping the position and all.
    """
    
    ndims = img.GetDimension()
    direction = list(img.GetDirection()) 
    spacing = list(img.GetSpacing())
    origin = list(img.GetOrigin())
    size = list(img.GetSize())

    size_new = [int(size[d] * spacing[d] / vox_sz[d]) for d in range(ndims)]
    true_vox_sz = [size[d] * spacing[d] / size_new[d] for d in range(ndims)]
    origin_new = [origin[d] + (true_vox_sz[d] - spacing[d]) / 2 for d in range(ndims)]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size_new)
    resampler.SetOutputOrigin(origin_new)
    resampler.SetOutputSpacing(true_vox_sz)
    resampler.SetOutputDirection(direction)
    resampler.SetInterpolator(interp)
    
    return resampler.Execute(img)
    

def change_img_size(img, grid_sz=[96,128,96]):
    
    size = img.GetSize()
    
    center = np.flip(np.floor(np.mean(np.array(np.where(sitk.GetArrayFromImage(img))), axis=1)))
    half_sz = np.floor(np.array(grid_sz) / 2)
    bound_inf = (center - half_sz).astype(np.int16)
    bound_sup = (center + grid_sz - half_sz - size).astype(np.int16)
    
    pad_bound_inf = (np.abs(bound_inf) * (bound_inf < 0)).tolist()
    pad_bound_sup = (bound_sup * (bound_sup > 0)).tolist()
    crop_bound_inf = (bound_inf * (bound_inf > 0)).tolist()
    crop_bound_sup = (np.abs(bound_sup) * (bound_sup < 0)).tolist()
    
    img = sitk.ConstantPad(img, pad_bound_inf, pad_bound_sup)
    img = sitk.Crop(img, crop_bound_inf, crop_bound_sup)
    
    return img


def get_real_field(field, matO, nobatch=False):
    """
    Compute the real coordinates affine transformation based on
        - A deformation field (ND-array) in voxelic coordinates. (nb,nx,ny,nz,3)
        - An orientation matrix.
    """
    
    extdims = len(field.shape) - 2
    ndims = extdims
    if nobatch:
        ndims += 1
        
    field = np.expand_dims(field, -1)
    linearPartO = np.expand_dims(matO[:-1,:-1], list(range(extdims+1))) 
    perm = np.expand_dims(np.eye(ndims)[::-1], list(range(extdims+1))) # permut dimensions from numpy to itk
    
    field_real = np.matmul(linearPartO, np.matmul(perm,field))     
    
    return field_real[..., 0]


def aff_tr2mat(aff_tr):
    
    ndims = aff_tr.GetDimension()
    aff_mat = np.reshape(aff_tr.GetParameters()[:-ndims], (ndims,ndims))
    aff_mat = np.c_[aff_mat, aff_tr.GetParameters()[ndims**2:]]
    aff_mat = np.r_[aff_mat, np.reshape([0]*ndims+[1], (1,ndims+1))]
    
    return aff_mat
    

def aff_mat2tr(aff_mat):
    
    ndims = aff_mat.shape[1]-1
    aff_tr = sitk.AffineTransform(ndims)
    aff_tr.SetMatrix(np.ravel(aff_mat[:ndims,:ndims] + np.eye(ndims)))
    aff_tr.SetTranslation(aff_mat[:ndims,ndims])
    
    return aff_tr

    
def aff_tr2field(aff_tr, 
                 geom_img=None, 
                 size=None, origin=None, direction=None, spacing=None):
    
    if geom_img is not None:        
        size = geom_img.GetSize()
        origin = geom_img.GetOrigin()
        direction = geom_img.GetDirection()
        spacing = geom_img.GetSpacing()
            
    trsf2disp = sitk.TransformToDisplacementFieldFilter()
    trsf2disp.SetSize(size)
    trsf2disp.SetOutputOrigin(origin)
    trsf2disp.SetOutputDirection(direction)
    trsf2disp.SetOutputSpacing(spacing)
    
    return trsf2disp.Execute(aff_tr)
    

def integrate_svf(svf, int_steps=7, out_tr=True, alpha=1):
    
    ndims = svf.GetDimension()
    
    if alpha != 1:
        svf = sitk.Compose([sitk.VectorIndexSelectionCast(svf, d) * alpha for d in range(ndims)])
        
    # scaling
    svf = sitk.Compose([sitk.VectorIndexSelectionCast(svf, d)/(2**int_steps) for d in range(ndims)])

    # squaring
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetUseNearestNeighborExtrapolator(True)
    resampler.SetReferenceImage(svf)
    for _ in range(int_steps): 
        svf0 = copy.deepcopy(svf)
        transfo = sitk.DisplacementFieldTransform(svf)    
        resampler.SetTransform(transfo)
        svf = svf0 + resampler.Execute(svf0)
    
    if out_tr:
        return sitk.DisplacementFieldTransform(svf)
    else:
        return svf


def jacobian(disp):   # TODO: add identity option !!
    
    ndims = disp.GetDimension()
    size = disp.GetSize()
    volshape = size[::-1]
    origin = disp.GetOrigin()
    direction = disp.GetDirection()
    spacing = disp.GetSpacing()
    
    disp = [sitk.VectorIndexSelectionCast(disp, i) for i in range(ndims)]
    
    grad_filter = sitk.GradientImageFilter()
    grad_filter.UseImageDirectionOn()
    grad_filter.UseImageSpacingOn()
    grad = [grad_filter.Execute(d) for d in disp]

    jacob = np.zeros((*volshape, ndims, ndims))  
    for i in range(ndims):
        grad_i = [sitk.VectorIndexSelectionCast(grad[i], j) for j in range(ndims)]
        for j in range(ndims):
            jacob[..., i, j] = sitk.GetArrayFromImage(grad_i[j])
    jacob = np.reshape(jacob, (*volshape, ndims**2))
    
    jacob = sitk.GetImageFromArray(jacob)
    jacob.SetOrigin(origin)
    jacob.SetDirection(direction)
    jacob.SetSpacing(spacing)    
        
    return jacob


def bch(svf1_img, svf2_img, order=2):
    """
    svf1 and sv2 are assumed to have same size, orientation, spacing, origin.
    """
    
    ndims = svf1_img.GetDimension()
    size = svf1_img.GetSize()
    volshape = size[::-1]
    
    compo_svf_img = svf1_img  + svf2_img   
    
    if order > 1: 
        svf1 = sitk.GetArrayFromImage(svf1_img)
        svf1 = np.reshape(svf1, (*volshape, ndims, 1))
        svf1_jac = jacobian(svf1_img)
        svf1_jac = sitk.GetArrayFromImage(svf1_jac)
        svf1_jac = np.reshape(svf1_jac, (*volshape, ndims, ndims))
        
        svf2 = sitk.GetArrayFromImage(svf2_img)
        svf2 = np.reshape(svf2, (*volshape, ndims, 1))
        svf2_jac = jacobian(svf2_img)
        svf2_jac = sitk.GetArrayFromImage(svf2_jac)
        svf2_jac = np.reshape(svf2_jac, (*volshape, ndims, ndims))
        
        lie_bracket = np.matmul(svf1_jac, svf2) - np.matmul(svf2_jac, svf1)
        lie_bracket = np.reshape(lie_bracket, (*volshape, ndims))
        lie_bracket /= 2
        lie_bracket_img = sitk.GetImageFromArray(lie_bracket, isVector=True)
        lie_bracket_img.CopyInformation(svf1_img)
    
        compo_svf_img += lie_bracket_img
        
    return compo_svf_img