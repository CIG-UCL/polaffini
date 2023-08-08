import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

   
class wMSE:
    """
    Mean squared error.
    Handles weightmap (assumed to be concatenated to y_true on the last axis).
    """
    
    def __init__(self, is_weighted=False, is_stacked=False):
        self.is_weighted = is_weighted 
        self.is_stacked = is_stacked
        
    def loss(self, y_true, y_pred): 
        
        if self.is_weighted:
            if self.is_stacked:
                weights = y_true
                y_true, y_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)
            else:
                y_true, weights = tf.split(y_true, num_or_size_splits=2, axis=-1)
            loss = K.sum(weights * K.square(y_true - y_pred)) / K.sum(weights)        
        else:
            if self.is_stacked:
                y_true, y_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)
            loss = K.mean(K.square(y_true - y_pred))
        
        return loss
    
    
    
class wLCC:  # extension VoxelMorph LCC loss from dev branch (not main!!)
    """
    Local (over window) correlation coefficient loss. (Copied on Voxelmorph).
    Handles weightmap (assumed to be concatenated to y_true on the last axis).
    """

    def __init__(self, win=None, eps=1e-5, is_weighted=False, is_stacked=False):
        self.win = win
        self.eps = eps
        self.is_weighted = is_weighted
        self.is_stacked = is_stacked

    def ncc(self, Ii, Ji):
        
        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        # ndims = len(Ii.get_shape().as_list()) - 2
        ndims = len(Ii.shape) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims
        elif not isinstance(self.win, list):  # user specified a single number not a list
            self.win = [self.win] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        # compute filters
        # in_ch = Ji.get_shape().as_list()[-1]
        in_ch = Ji.shape[-1]
        sum_filt = tf.ones([*self.win, in_ch, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)

        # compute local sums via convolution
        padding = 'SAME'
        I_sum = conv_fn(Ii, sum_filt, strides, padding)
        J_sum = conv_fn(Ji, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win) * in_ch
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        # TODO: simplify this
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        cross = tf.maximum(cross, self.eps)
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        I_var = tf.maximum(I_var, self.eps)
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        J_var = tf.maximum(J_var, self.eps)

        # cc = (cross * cross) / (I_var * J_var)
        cc = (cross / I_var) * (cross / J_var)  
        
        return cc
    
    def loss(self, y_true, y_pred):
        
        if self.is_weighted:
            if self.is_stacked:
                weights = y_true
                y_true, y_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)
            else:
                y_true, weights = tf.split(y_true, num_or_size_splits=2, axis=-1)
        else:
            if self.is_stacked:
                y_true, y_pred = tf.split(y_pred, num_or_size_splits=2, axis=-1)
            weights = K.ones_like(y_pred)
            
        cc = self.ncc(y_true, y_pred)
        
        return 1 - K.sum(weights * cc) / K.sum(weights) 
    
    def loss_map(self, y_true, y_pred):
        
        if self.is_weighted:
            y_true, weights = tf.split(y_true, num_or_size_splits=2, axis=-1)

        cc = self.ncc(y_true, y_pred)
        
        return 1 - cc
            
    
class Dice: # from VoxelMorph but modified for better div_no_nan handling
    """
    N-D dice for segmentation
    """
    
    def loss(self, y_true, y_pred):
        
        ndims = len(y_pred.get_shape().as_list()) - 2
        vol_axes = list(range(1, ndims + 1))
    
        top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
        bottom = tf.reduce_sum(y_true + y_pred, vol_axes)
    
        div_no_nan = tf.math.divide_no_nan if hasattr(
            tf.math, 'divide_no_nan') else tf.div_no_nan  
        dice = tf.reduce_mean(div_no_nan(top, bottom))
        
        return -dice