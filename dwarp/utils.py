import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pandas as pd

def develop(x, dec='|_'):
    
    if isinstance(x, (list, tuple)):
        print(dec, type(x), len(x))
        dec = '|    ' + dec
        for i in range(len(x)):
            develop(x[i], dec)
    elif isinstance(x, np.ndarray) or tf.is_tensor(x):
        print(dec, type(x), x.dtype, x.shape)   
    else: 
        print(dec, type(x))   


def print_inputGT(sample):
    
    print('Generator:')
    inputGT = ['Inputs', 'Ground truths']
    for j in range(len(sample)):
        if j <= 2:
            print('  - ' + inputGT[j] + ':')
        else:
            print('Other:')
        for i in range(len(sample[j])):
            if sample[j][i] is None :
                print('    ' + 'None')
            else:
                print('    ' + str(sample[j][i].shape) + ' - ', sample[j][i].dtype)


    
def shift_to_transfo(loc_shift, indexing='ij'):
    
    if isinstance(loc_shift.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = loc_shift.shape[1:-1].as_list()
    else:
        volshape = loc_shift.shape[1:-1]
    ndims = len(volshape)
    
    ij = [range(volshape[i]) for i in range(ndims)]
    mesh = tf.meshgrid(*ij, indexing=indexing)
    mesh = tf.cast(tf.expand_dims(tf.stack(mesh, axis=-1), 0), 'float32')
    
    return mesh + loc_shift


def plot_losses(loss_file, is_val=False):

    tab_loss = pd.read_csv(loss_file, sep=',')
    nb_losses = len(tab_loss.columns) - 1
    if is_val:
        nb_losses = int(nb_losses / 2)
    
    f, axs = plt.subplots(1,nb_losses); f.dpi = 200
    plt.rcParams['font.size'] = '3'
    plt.rcParams["xtick.major.size"] = 2
    plt.rcParams["ytick.major.size"] = 2
    
    for l in range(nb_losses):
        axs[l].plot(tab_loss.epoch, tab_loss.loc[:,tab_loss.columns[l+1]], linewidth=0.5) 
        if is_val:
            axs[l].plot(tab_loss.epoch, tab_loss.loc[:,tab_loss.columns[nb_losses+l+1]], linewidth=0.5) 
        axs[l].set_title(tab_loss.columns[l+1], fontsize=6)
    
    plt.show()



def jacobian(transfo, outDet=False, dire=None, is_shift=False):
    # takes a tensor of shape [batch_size, sx, sy, (sz,) ndims] as input.
    
    if isinstance(transfo.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = transfo.shape[1:-1].as_list()
    else:
        volshape = transfo.shape[1:-1]
    ndims = len(volshape)
    ndirs = transfo.shape[-1]
    if dire is not None and ndirs != 1:
        raise Exception('the last dim should be of size 1 for a unidirectional field, but got: %s' % ndims)
    
    jacob = []
    for d in range(ndims):
        grad = tf.gather(transfo, range(2, volshape[d]), axis=d+1)-tf.gather(transfo, range(volshape[d]-2), axis=d+1)
        grad_left = tf.gather(transfo, [1], axis=d+1)-tf.gather(transfo, [0], axis=d+1)
        grad_right = tf.gather(transfo, [volshape[d]-1], axis=d+1)-tf.gather(transfo, [volshape[d]-2], axis=d+1)
        grad = tf.concat((grad_left, grad/2, grad_right), axis=d+1)  
        grad = tf.expand_dims(grad, axis=-1)
        jacob += [grad]
    
    jacob = tf.concat(jacob, axis=-1) 
    
    if is_shift:
        if dire is None:
            jacob += tf.eye(ndims, ndims, transfo.shape[:-1])
        else:
            identity = [tf.ones(transfo.shape[:-1]) if d==dire else tf.zeros(transfo.shape[:-1]) for d in range(ndims)]
            jacob += tf.expand_dims(tf.stack(identity, axis=-1), axis=-2)

    if outDet:
        # detjac = tf.linalg.det(jacob)
        if ndims == 2:
            if ndirs == 2:
                detjac =  jacob[:,:,:,0,0] * jacob[:,:,:,1,1]\
                        - jacob[:,:,:,1,0] * jacob[:,:,:,0,1] 
            elif ndirs == 1:
                detjac = jacob[:,:,:,0,dire]
        elif ndims == 3:
            if ndirs == 3:
                detjac =  jacob[:,:,:,:,0,0] * jacob[:,:,:,:,1,1] * jacob[:,:,:,:,2,2]\
                        + jacob[:,:,:,:,0,1] * jacob[:,:,:,:,1,2] * jacob[:,:,:,:,2,0]\
                        + jacob[:,:,:,:,0,2] * jacob[:,:,:,:,1,0] * jacob[:,:,:,:,2,1]\
                        - jacob[:,:,:,:,2,0] * jacob[:,:,:,:,1,1] * jacob[:,:,:,:,0,2]\
                        - jacob[:,:,:,:,1,0] * jacob[:,:,:,:,0,1] * jacob[:,:,:,:,2,2]\
                        - jacob[:,:,:,:,0,0] * jacob[:,:,:,:,2,1] * jacob[:,:,:,:,1,2]
            elif ndirs == 1:
                detjac = jacob[:,:,:,:,0,dire]
        else:
            raise Exception('Only dimension 2 or 3 supported, but got: %s' % ndims)
            
        return jacob, detjac
    else: 
        return jacob 


class get_real_transfo_aff:
    """
    Compute the real coordinates affine transformation based on
        - A voxelic transformation.
        - An initial real transformation.
        - An orientation matrix.
    This transformation can be used in softwares that properly handle orientation 
    (like ITK-based one, NOT like fsl.)
    """

    def __init__(self, ndims, **kwargs):
        self.ndims = ndims
        if ndims != 3 and ndims != 2:
            raise NotImplementedError('2D or 3D only')

        super().__init__(**kwargs)
        

    def __call__(self, init, matO, matAff):
        """
        Parameters
            init: initialization translation, (dim 0 = batch dim).
            matO: orientation matrix, (no batch dim).
            matAff: estimated voxelic affine transformation, (dim 0 = batch dim).

        """
        
        matInit = tf.map_fn(self._single_trans2mat, init, dtype=tf.float32)
        if matAff.shape[1] == self.ndims:
            matAff = tf.map_fn(self._homogen_ext, matAff, dtype=tf.float32)
        matO = self._matO_perm(matO)
        
        matReal = tf.matmul(matO, matAff)
        matReal = tf.matmul(matReal, tf.linalg.inv(matO))
        matReal = tf.matmul(matInit, matReal)
        
        return matReal

    def _single_trans2mat(self, vector):
        
        vector = tf.concat((vector, [1]), axis=0)
        mat = tf.eye(self.ndims)
        mat = tf.concat((mat, tf.zeros((1, self.ndims))), axis=0)
        mat = tf.concat((mat, tf.expand_dims(vector, axis=1)), axis=1)  
            
        return mat

    def _homogen_ext(self, mat):
        
        extensionh = tf.zeros((1, self.ndims))
        extensionh = tf.concat((extensionh, [[1]]), axis=1)
        mat = tf.concat((mat, extensionh), axis=0)
        
        return mat
    
    def _matO_perm(self, matO):

        trans = tf.zeros((self.ndims,1))
        perm = tf.eye(self.ndims)[::-1]
        perm = tf.concat((perm, trans), axis=1)
        perm = self._homogen_ext(perm)
        
        if matO.shape[1] == self.ndims:
            matO = self._homogen_ext(matO)

        matO = tf.matmul(matO, perm)
        
        return matO
   
    
def get_real_field(field, matO, nobatch=False):
    """
    Compute the real coordinates affine transformation based on
        - A deformation field in voxelic coordinates. (nb,nx,ny,nz,3)
        - An orientation matrix.
    """
    
    extdims = len(field.shape) - 2
    ndims = extdims
    if nobatch:
        ndims += 1
        
    field = np.expand_dims(field, -1)
    linearPartO = np.expand_dims(matO[:-1,:-1], list(range(extdims+1))) 
    perm = np.expand_dims(np.eye(ndims)[::-1], list(range(extdims+1))) # permut dimensions from voxelmorph to itk
    
    field_real = tf.matmul(linearPartO, tf.matmul(perm,field))     
    
    return field_real[..., 0]
    
    
def pad_image(img, k=5, outSize=None, bg_val=0):
    """
    Pad an image such that image size along each dimension becomes of form 2^k.
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
   