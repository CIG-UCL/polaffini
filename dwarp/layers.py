import tensorflow as tf
from tensorflow.keras.layers import Layer
from . import utils
    

class expLinearTransfo(Layer):
    """
    """
           
    def __init__(self, ndims, **kwargs):
           
        self.ndims = ndims
        super().__init__(**kwargs)

    def call(self, vector):

        return tf.map_fn(self._expMatrix, vector, dtype=tf.float32)

    def _expMatrix(self, vector):
        
        matAff = tf.reshape(vector, (self.ndims, self.ndims+1))
        matAff = tf.concat((matAff, tf.zeros((1,(self.ndims+1)))), axis=0)    
        matAff = tf.cast(matAff, tf.float64)
        matAff = tf.linalg.expm(matAff)   
        matAff = tf.cast(matAff, tf.float32)   
        matAff = tf.slice(matAff,[0, 0],[self.ndims, self.ndims+1])

        return matAff
      

class AffCoeffToMatrix(Layer):
    """
    Build the affine matrix from estimated affine coefficients.
    Rotation parameters are assumed in Lie algebra, exponentiation ensure proper transformation.
    
    A = [RUDU' T]
    where R: rotation, U: direction of scaling, D: scaling, T: translation.
    """

    def __init__(self, ndims, transfo_type='affine', dire=None, **kwargs):
        self.ndims = ndims
        if ndims != 3 and ndims != 2:
            raise NotImplementedError('2D or 3D only')
        self.transfo_type = transfo_type
        self.dire = dire
        super().__init__(**kwargs)
        

    def call(self, aff_params):
        """
        transfo_type == 'affine'     -> aff_params = [trans, rot, scalDir, scal], shape=(3,3,3,3) in 3D.  
                        'rigid'      -> aff_params = [trans, rot],                shape=(3,3) in 3D.
                        'dir_affine' -> aff_params = [trans, lin, dir],           shape=(1,3,1) in 3D.
        """
        
        if self.transfo_type == 'affine':
            trans = aff_params[0]
            rotat = tf.map_fn(self._single_rotMat, aff_params[1], dtype=tf.float32)
            scalDir = tf.map_fn(self._single_rotMat, aff_params[2], dtype=tf.float32)
            scal = self.expm(tf.linalg.diag(aff_params[3]))          
            lin = tf.matmul(scal, scalDir, transpose_b=True)
            lin = tf.matmul(scalDir, lin)
            lin = tf.matmul(rotat, lin)
            
        elif self.transfo_type == 'rigid': 
            trans = aff_params[0]
            lin = tf.map_fn(self._single_rotMat, aff_params[1], dtype=tf.float32)
            lin = self.expm(lin) 
            
        elif self.transfo_type == 'dir_affine':
            trans = [aff_params[0] if d == self.dire else tf.zeros_like(aff_params[0]) for d in range(self.ndims)]
            trans = tf.concat(trans, axis=1)
            lin = tf.map_fn(self._single_linDirMat, aff_params[1], dtype=tf.float32)
            lin = self.expm(lin)
                 
        mat = tf.concat((lin, tf.expand_dims(trans,axis=2)), axis=2)
        
        return mat      

    def _single_rotMat(self, vector):
        
        if self.ndims == 3:
            rotMat = tf.convert_to_tensor([[ 0        ,-vector[2], vector[1]],
                                           [ vector[2], 0        ,-vector[0]],
                                           [-vector[1], vector[0], 0       ]]) 
        elif self.ndims == 2:
            rotMat = tf.convert_to_tensor([[ 0        ,-vector[0]],
                                           [ vector[0], 0        ]])     

        return rotMat
    
    def _single_linDirMat(self, vector):

        lin = []
        for d in range(self.ndims):
            if d == self.dire:
                lin.append(tf.expand_dims(vector, axis=0))
            else:
                lin.append(tf.zeros((1, self.ndims)))
        lin = tf.concat(lin, axis=0)

       # lin = tf.convert_to_tensor([[ 0       ,0        ,0        ],
       #                             [vector[0],vector[1],vector[2]],
       #                             [0        , 0       ,0        ]]) 
                   
        return lin     
     
    def expm(self, mat):
        
        dtype = mat.dtype
        mat = tf.cast(mat, dtype=tf.float64)
        mat = tf.linalg.expm(mat)
        mat = tf.cast(mat, dtype=dtype) 
        
        return mat


class QuadCoeffToMatrix(Layer):
    """
    Build the quadradic matrix from estimated quadratic coefficients.

    """

    def __init__(self, ndims, **kwargs):
        self.ndims = ndims
        if ndims != 3 and ndims != 2:
            raise NotImplementedError('2D or 3D only')
        super().__init__(**kwargs)
        

    def call(self, quad_params):
        """
        """
        quad = tf.map_fn(self._single_quadMat, quad_params, dtype=tf.float32)
        
        return quad   

    def _single_quadMat(self, vector):
        
        if self.ndims == 3:
            mat = tf.convert_to_tensor([[vector[0], vector[1], vector[2]],
                                        [vector[1], vector[3], vector[4]],
                                        [vector[2], vector[4], vector[5]]]) 
        elif self.ndims == 2:
            mat = tf.convert_to_tensor([[vector[0], vector[1]],
                                        [vector[1], vector[2]]])     

        return mat
    
 
class QuadUnidirToDenseShift(Layer):
    """
    Converts an affine transform to a dense shift transform.
    """

    def __init__(self, shape, dire, shift_center=True, **kwargs):
        """
        Parameters:
            shape: Target shape of dense shift.
        """
        self.dire = dire
        self.shape = shape
        self.ndims = len(shape)
        self.shift_center = shift_center
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dire': self.dire,
            'shape': self.shape,
            'ndims': self.ndims,
            'shift_center': self.shift_center,
        })
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.shape, self.ndims)


    def call(self, matrix):
        """
        Parameters:
            matrix: symmetric matrix of shape [B, N, N].
        """
        single = lambda mat: utils.quadratic_unidir_to_dense_shift(mat, self.dire, self.shape, shift_center=self.shift_center)
        return tf.map_fn(single, matrix)
    
    
class get_real_transfo(Layer):
    """
    Compute the real coordinates affine transformation based on
        - The voxelic one estimated by the model.
        - The initial transformation from the generator.
        - The orientation from the header.
    This transformation can be used in softwares that properly handle orientation 
    (like ITK-based one, NOT like fsl.)
    """

    def __init__(self, ndims, **kwargs):
        self.ndims = ndims
        if ndims != 3 and ndims != 2:
            raise NotImplementedError('2D or 3D only')

        super().__init__(**kwargs)
        

    def call(self, transfos):
        """
        Parameters
            tuple size 3 for init trans, orientation, voxelic affine.

        """
        
        matInit = tf.map_fn(self._single_vec2mat, transfos[0], dtype=tf.float32)
        matO = transfos[1]
        matAff = tf.map_fn(self._single_matAff, transfos[2], dtype=tf.float32)

        matReal = tf.matmul(matO, matAff)
        matReal = tf.matmul(matReal, tf.linalg.inv(matO))
        matReal = tf.matmul(matInit, matReal)
        
        return matReal

    def _single_vec2mat(self, vector):
        
        vector = tf.concat((vector, [1]), axis=0)
        mat = tf.eye(self.ndims)
        mat = tf.concat((mat, tf.zeros((1, self.ndims))), axis=0)
        mat = tf.concat((mat, tf.expand_dims(vector, axis=1)), axis=1)  
            
        return mat

    def _single_matAff(self, mat):
        
        extensionh = tf.zeros((1, self.ndims))
        extensionh = tf.concat((extensionh, [[1]]), axis=1)
        mat = tf.concat((mat, extensionh), axis=0)
        
        return mat
    
    def _single_matO(self, mat):

        extensionv = tf.concat((tf.zeros((self.ndims)), [1]), axis=0)
        perm = tf.image.flip_left_right(tf.expand_dims(tf.eye(self.ndims),axis=0))[0]
        perm = tf.concat((perm, tf.zeros((1, self.ndims))), axis=0)
        perm = tf.concat((perm, tf.expand_dims(extensionv, axis=1)), axis=1)  

        extensionh = tf.zeros((1, self.ndims))
        extensionh = tf.concat((extensionh, [[1]]), axis=1)
        mat = tf.concat((mat, extensionh), axis=0)

        mat = tf.matmul(mat, perm)
        
        return mat
    
    
class JacobianMultiplyIntensities(Layer):

    def __init__(self, indexing='ij', outDet=True, is_shift=True, dire=None, **kwargs):
        self.indexing = indexing
        self.outDet = outDet
        self.is_shift = is_shift
        self.dire = dire
        super(self.__class__, self).__init__(**kwargs)
  
    def build(self, input_shape):
        self.inshape = input_shape

    def call(self, inputs):
        """
        input : [moved_image, loc_shift]
        output : Moved image with intensities multiplied by Jacobian determinant of the transformation.
        """
        _, jacTransfo = utils.jacobian(inputs[1], dire=self.dire, outDet=self.outDet, is_shift=self.is_shift)
        jacTransfo = tf.math.abs(jacTransfo)        
        
        return tf.expand_dims(jacTransfo,-1) * inputs[0]