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

    def __init__(self, ndims, transfo_type='affine', **kwargs):
        self.ndims = ndims
        if ndims != 3 and ndims != 2:
            raise NotImplementedError('2D or 3D only')
        self.transfo_type = transfo_type
        super().__init__(**kwargs)
        

    def call(self, aff_params):
        """
        Parameters
            tuple size 4 for trans, rot, scalDir, scal.
        """
        
        trans = aff_params[0]
        rotat = tf.map_fn(self._single_rotMat, aff_params[1], dtype=tf.float32)
        if self.transfo_type == 'affine':
            scalDir = tf.map_fn(self._single_rotMat, aff_params[2], dtype=tf.float32)
            scal = tf.linalg.expm(tf.linalg.diag(aff_params[3]))          
            mat = tf.matmul(scal, scalDir, transpose_b=True)
            mat = tf.matmul(scalDir, mat)
            mat = tf.matmul(rotat, mat)
            
        elif self.transfo_type == 'rigid':
            mat = rotat
        
        mat = tf.concat((mat, tf.expand_dims(trans,axis=2)), axis=2)
        
        return mat


    def _single_rotMat(self, vector):

        if self.ndims == 3:
            # extract components of input vector

            rotMat = tf.convert_to_tensor([[ 0        ,-vector[2], vector[1]],
                                           [ vector[2], 0        ,-vector[0]],
                                           [-vector[1], vector[0], 0       ]])
            
        elif self.ndims == 2:
            
            rotMat = tf.convert_to_tensor([[ 0        ,-vector[0]],
                                           [ vector[0], 0        ]])     

        rotMat = tf.linalg.expm(rotMat)  
            
        return rotMat


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

    def __init__(self, indexing='ij', **kwargs):
        self.indexing = indexing
        super(self.__class__, self).__init__(**kwargs)
  
    def build(self, input_shape):
        self.inshape = input_shape

    def call(self, inputs):
        """
        input : [loc_shift, moved_image]
        output : Moved image with intensities multiplied by Jacobian determinant of the transformation.
        """
        _, jacTransfo = utils.jacobian(inputs[1], outDet=True, is_shift=True)
        jacTransfo = tf.math.abs(jacTransfo)        
        
        return tf.expand_dims(jacTransfo,-1) * inputs[0]