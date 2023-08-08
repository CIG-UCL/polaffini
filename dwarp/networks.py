# third party imports
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.initializers as KI

import voxelmorph   
# This code uses Voxelmorph pieces directly or contains pieces inspired by Voxelmorph.
# If you use it, please cite them appropriately too. See https://github.com/voxelmorph/voxelmorph       
import neurite as ne

# local imports
from . import layers

class aff2atlas(ne.modelio.LoadableModel):
    """
    Network for affine registration to atlas.
    """
    @ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 decomp_transfo=False,
                 nb_enc_features=[16,32,32,32,32],
                 nb_dense_features=[128],
                 nb_enc_levels=None,
                 enc_feat_mult=1,
                 nb_enc_conv_per_level=1,
                 src_feats=1,   
                 orientation=None,
                 name='aff2atlas'):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_enc_features: encoder convolutional features. Can be specified via a list or as a single integer. 
            nb_unet_levels: Number of levels in unet. Only used when nb_unet_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_unet_features is an integer. Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            src_feats: Number of moving image features. Default is 1.
            input_model: Model to replace default input layer before concatenation. Default is None.
            name: Model name - also used as layer name prefix. Default is 'aff2atlas'.
        """
        
        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [2, 3], 'ndims should be one of 2, or 3. found: %d' % ndims

        moving = tf.keras.Input(shape=(*inshape, src_feats), name='%s_moving_input' % name)
        init_transfo = tf.keras.Input(shape=(ndims), name='%s_init_transfo_input' % name)
        inputs = [moving, init_transfo]
        input_model = tf.keras.Model(inputs=inputs, outputs=moving)

        # build core enc model and grab inputs
        enc_model = encoder(input_model=input_model,
                            nb_features=nb_enc_features,
                            nb_levels=nb_enc_levels,
                            feat_mult=enc_feat_mult,
                            nb_conv_per_level=nb_enc_conv_per_level)
        
        # flatten
        nbAffParams = ndims * (ndims + 1)
        last = KL.Flatten()(enc_model.output)
        
        # dense part 
        # build affine transformation
             
        if decomp_transfo:
            translat = last
            rotat = last
            scalDir = last
            scal = last
            if nb_dense_features is not None:
                for i, nf in enumerate(nb_dense_features):
                    translat = KL.Dense(nf, activation='relu', name='translat_%d' % i)(translat)
                    rotat = KL.Dense(nf, activation='relu', name='rotat_%d' % i)(rotat)
                    scalDir = KL.Dense(nf, activation='relu', name='scalDir_%d' % i)(scalDir)
                    scal = KL.Dense(nf, activation='relu', name='scal_%d' % i)(scal)
            
            tranlat_init = KI.RandomNormal(stddev=0.1)
            rotat_init = KI.RandomNormal(stddev=0.0001)  
            scal_init = KI.RandomNormal(stddev=0.001)  
            
            if ndims == 3:                
                translat = KL.Dense(3, kernel_initializer=tranlat_init, name='translat')(translat)
                rotat  = KL.Dense(3, kernel_initializer=rotat_init, name='rotat')(rotat)
                scalDir = KL.Dense(3, kernel_initializer=rotat_init, name='scalDir')(scalDir)
                scal = KL.Dense(3, kernel_initializer=scal_init, name='scal')(scal)
            elif ndims == 2:
                translat = KL.Dense(2, kernel_initializer=tranlat_init, name='translat')(translat)
                rotat  = KL.Dense(1, kernel_initializer=rotat_init, name='rotat')(rotat)
                scalDir = KL.Dense(1, kernel_initializer=rotat_init, name='scalDir')(scalDir)
                scal = KL.Dense(2, kernel_initializer=scal_init, name='scal')(scal)
            
            affMat = layers.AffCoeffToMatrix(ndims=ndims, name='build_affine_transfo')([translat, rotat, scalDir, scal])
            
        else:
            if nb_dense_features is not None:
                for nf in nb_dense_features:
                    last = KL.Dense(nf, activation='relu')(last)
                    
            last = KL.Dense(nbAffParams, 
                            kernel_initializer=KI.RandomNormal(stddev=0.001),
                            bias_initializer=KI.Zeros())(last)  
            
            affMat = layers.expLinearTransfo(ndims=ndims, name='build_affine_transfo')(last)
        

        # warp image with affine trasnformation
        moved = voxelmorph.layers.SpatialTransformer(interp_method='linear', indexing='ij', add_identity=False, shift_center=False, name='resampler')([moving, affMat])
            
        outputs = [moved]
          
        super().__init__(name=name, inputs=input_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.ndims = ndims
        self.references.moved = moved
        self.references.affMat = affMat
        self.references.init_transfo = init_transfo
        self.references.matO = orientation
        
    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, 
                              [self.references.moved, self.references.affMat])

    def register(self, src):
        """
        Predicts the transform from src to atlas.
        """
        return self.get_registration_model().predict(src)

    def apply_transform(self, src, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output[1]])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, img])
    
    
    
class diffeo2atlas(ne.modelio.LoadableModel): # Inspired by voxelmorph's VxmDense network.
    """
    Network for diffeomorphic registration to atlas.
    """

    @ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 nb_labs=None,
                 is_aux=False,  # -> is there auxiliary images (e.g. FA maps).
                 orientation=None,
                 nb_enc_features=[16, 32, 32, 32, 32],
                 nb_dec_features=[32, 32, 32, 32, 32, 16, 16],
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 src_feats=1,   
                 trg_feats=1,
                 name='diffeo2atlas'):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_enc_features: encoder convolutional features. Can be specified via a list or as a single integer. 
            nb_unet_levels: Number of levels in unet. Only used when nb_unet_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_unet_features is an integer. Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            src_feats: Number of moving image features. Default is 1.
            input_model: Model to replace default input layer before concatenation. Default is None.
            name: Model name - also used as layer name prefix. Default is 'aff2atlas'.
        """
        is_seg = False
        if nb_labs is not None:
            is_seg = True    # -> there is auxiliary segmentation
                        
        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [2, 3], 'ndims should be one of 2, or 3. found: %d' % ndims

        moving = tf.keras.Input(shape=(*inshape, src_feats), name='%s_moving_input' % name)            
        inputs = [moving]
        input_model = tf.keras.Model(inputs=inputs, outputs=moving)

        # build core unet model and grab inputs
        unet_model = voxelmorph.networks.Unet(input_model=input_model,
                                              nb_features=[nb_enc_features, nb_dec_features],
                                              nb_levels=nb_unet_levels,
                                              feat_mult=unet_feat_mult,
                                              nb_conv_per_level=nb_unet_conv_per_level)
        
        # build diffeo transfo
        Conv = getattr(KL, 'Conv%dD' % ndims)
        flow = Conv(ndims, kernel_size=3, padding='same',
                kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5), name='%s_flow' % name)(unet_model.output)

        ## SVF integration into diffeo
        if int_steps > 0:
            transfo = voxelmorph.layers.VecInt(method='ss', name='%s_flow_int' % name, int_steps=int_steps)(flow)
        else:
            transfo = flow
        
        # warp image with diffeomorphic transformation
        moved = voxelmorph.layers.SpatialTransformer(interp_method='linear', indexing='ij', name='resampler')([moving, transfo])
        dummy_layer = KL.Lambda(lambda x: x, name='img')
        moved = dummy_layer(moved)
        outputs = [moved]

        if is_seg:
            moving_seg = tf.keras.Input(shape=(*inshape, nb_labs), name='%s_moving_seg_input' % name)  #, dtype='bool')
            inputs = inputs + [moving_seg]
            # moved_seg = voxelmorph.layers.SpatialTransformer(datatype='bool', interp_method='nearest', indexing='ij', name='resampler_seg')([moving_seg, transfo])  
            moved_seg = layers.Resampler(interp_method='nearest', name='resampler_seg')([moving_seg, transfo])  
            dummy_layer = KL.Lambda(lambda x: x, name='seg')
            moved_seg = dummy_layer(moved_seg)
            outputs += [moved_seg]
        if is_aux:
            moving_aux = tf.keras.Input(shape=(*inshape, src_feats), name='%s_moving_aux_input' % name)
            inputs = inputs + [moving_aux]
            moved_aux = layers.Resampler(interp_method='linear', name='resampler_aux')([moving_aux, transfo])  
            dummy_layer = KL.Lambda(lambda x: x, name='aux')
            moved_aux = dummy_layer(moved_aux)
            outputs += [moved_aux]
            
        dummy_layer = KL.Lambda(lambda x: x, name='reg')
        flow = dummy_layer(flow)
        outputs += [flow]
          
        super().__init__(name=name, inputs=inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.ndims = ndims
        self.references.moved = moved
        self.references.transfo = transfo
        self.references.flow = flow
        
        
    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs[:1], 
                              [self.references.moved, self.references.transfo, self.references.flow])

    def register(self, src):
        """
        Predicts the transform from src to atlas.
        """
        return self.get_registration_model().predict(src)

    def apply_transform(self, src, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output[1]])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, img])

        

class sudistoc(ne.modelio.LoadableModel):
    """
    sudistoc network for suceptibility distortion correction of echo-planar images
    through constrained non-linear registration of images acquired with reversed PED.
    """

    @ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 input_model=None,              
                 kissing=True,
                 constraint='oppsym',  # 'oppsym' (opposite symmetry) or 'diffeo' (inverse symmetry)
                 ped=None,   # axis corresponding to the phase encoding direction. If None, unconstrained registration.
                 jacob_mod=True,   # jacobian intensity modulation               
                 transfo_sup=False,  # supervised with ground truth transformations
                 image_sup=False,  # supervised with ground truth image
                 unsup=True,  # unsupervised similarity between undistorted images
                 name='sudistoc_net'):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_unet_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_unet_features is an integer. Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. Default is False.
            input_model: Model to replace default input layer before concatenation. Default is None.
            name: Model name - also used as layer name prefix. Default is 'vxm_dense'.
        """
        
        # ensure compatible settings
        if unsup and not kissing:
            raise ValueError('Unsupervised setting impossible with single image (no kissing)')
        if not transfo_sup and not image_sup and not unsup:
            raise ValueError('At least one of transfo_sup, image_sup or unsup has to be True')


        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=(*inshape, src_feats), name='%s_source_input' % name)
            target = tf.keras.Input(shape=(*inshape, trg_feats), name='%s_target_input' % name)
            input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        else:
            source, target = input_model.outputs[:2]

        # build core unet model and grab inputs
        unet_model = voxelmorph.networks.Unet(input_model=input_model,
                                              nb_features=nb_unet_features,
                                              nb_levels=nb_unet_levels,
                                              feat_mult=unet_feat_mult,
                                              nb_conv_per_level=nb_unet_conv_per_level,
                                              half_res=unet_half_res)

        # transform unet output into a flow field
        Conv = getattr(KL, 'Conv%dD' % ndims)
        if ped is None:
            flow_mean = Conv(ndims, kernel_size=3, padding='same',
                    kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5), name='%s_flow' % name)(unet_model.output)
        else:
            flow_mean = Conv(1, kernel_size=3, padding='same',
                    kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5), name='%s_flow' % name)(unet_model.output)
            flow_zeros = tf.keras.backend.zeros_like(flow_mean)
            
            flow_list = []  
            for i in range(0, ndims):
                if i == ped:
                    flow_list.append(flow_mean)
                else:
                    flow_list.append(flow_zeros)
             
            flow_mean = KL.concatenate(flow_list, name='%s_concat_flow' % name)
            
        # optionally include probabilities
        if use_probs:
            # initialize the velocity variance very low, to start stable
            flow_logsigma = Conv(ndims, kernel_size=3, padding='same',
                            kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-10),
                            bias_initializer=KI.Constant(value=-10),
                            name='%s_log_sigma' % name)(unet_model.output)
            flow_params = KL.concatenate([flow_mean, flow_logsigma], name='%s_prob_concat' % name)
            flow = ne.layers.SampleNormalLogVar(name='%s_z_sample' % name)([flow_mean, flow_logsigma])
        else:
            flow_params = flow_mean
            flow = flow_mean

        if not unet_half_res:
            # optionally resize for integration
            if int_steps > 0 and int_downsize > 1:
                flow = voxelmorph.layers.RescaleTransform(1 / int_downsize, name='%s_flow_resize' % name)(flow)

        preint_flow = flow

        pos_flow = flow
        
        if kissing and constraint == 'diffeo':
            neg_flow = ne.layers.Negate(name='%s_neg_flow' % name)(flow)            

        # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
        if int_steps > 0:
            pos_flow = voxelmorph.layers.VecInt(method='ss', name='%s_flow_int' % name, int_steps=int_steps)(pos_flow)
            if kissing and constraint == 'diffeo':
                neg_flow = voxelmorph.layers.VecInt(method='ss', name='%s_neg_flow_int' % name, int_steps=int_steps)(neg_flow)
                
            # resize to final resolution
            if int_downsize > 1:
                pos_flow = voxelmorph.layers.RescaleTransform(int_downsize, name='%s_diffflow' % name)(pos_flow)
                if kissing and constraint == 'diffeo':
                    neg_flow = voxelmorph.layers.RescaleTransform(int_downsize, name='%s_neg_diffflow' % name)(neg_flow)
                    
        if constraint == 'oppsym':
            neg_flow = ne.layers.Negate(name='%s_neg_transfo' % name)(pos_flow)               


        # warp image with flow field
        y_source = voxelmorph.layers.SpatialTransformer(interp_method='linear', indexing='ij', name='%s_transformer' % name)([source, pos_flow])
        if kissing:
            y_target = voxelmorph.layers.SpatialTransformer(interp_method='linear', indexing='ij', name='%s_neg_transformer' % name)([target, neg_flow])
        if jacob_mod:
            y_source = layers.JacobianMultiplyIntensities(indexing='ij', name='%s_det_Jac_multiply_source' % name)([y_source, pos_flow])
            if kissing:
                y_target = layers.JacobianMultiplyIntensities(indexing='ij', name='%s_det_Jac_multiply_target' % name)([y_target, neg_flow])
        
        if ped is not None:
            pos_flow_dir = layers.Slice(index=ped, name='%s_slice_pos_transfo' % name)(pos_flow)
            neg_flow_dir = layers.Slice(index=ped, name='%s_slice_neg_transfo' % name)(neg_flow)
        else:
            pos_flow_dir = pos_flow
            neg_flow_dir = neg_flow
            
        outputs=[]
        
        if unsup:
            y_diff = KL.Subtract(name='unsup')([y_target, y_source])
            outputs += [y_diff]
         
        if image_sup:
            dummy_layer = KL.Lambda(lambda x: x, name='sup_image')
            y_source = dummy_layer(y_source)
            outputs += [y_source]
            if kissing: 
                dummy_layer = KL.Lambda(lambda x: x, name='sup_image_neg')
                y_target = dummy_layer(y_target)
                outputs += [y_target]
         
        if transfo_sup:
            dummy_layer = KL.Lambda(lambda x: x, name='sup_field')
            pos_flow_dir = dummy_layer(pos_flow_dir)
            outputs += [pos_flow_dir]
            if kissing:
                dummy_layer = KL.Lambda(lambda x: x, name='sup_field_neg')
                neg_flow_dir = dummy_layer(neg_flow_dir)
                outputs += [neg_flow_dir]
         
        if use_probs:
            # compute loss on flow probabilities
            dummy_layer = KL.Lambda(lambda x: x, name='smooth')
            flow_params = dummy_layer(flow_params)
            outputs += [flow_params]
        else:
            # compute smoothness loss on pre-integrated warp
            dummy_layer = KL.Lambda(lambda x: x, name='smooth')
            preint_flow = dummy_layer(preint_flow)
            outputs += [preint_flow]
          
        super().__init__(name=name, inputs=input_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.kissing = kissing if kissing else None
        self.references.unsup = unsup if unsup else None
        self.references.y_diff = y_diff if unsup else None
        self.references.y_source = y_source
        self.references.y_target = y_target if kissing else None
        self.references.pos_flow = pos_flow
        self.references.neg_flow = neg_flow if kissing else None
        self.references.pos_flow_dir = pos_flow_dir
        self.references.neg_flow_dir = neg_flow_dir
        
    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs[:2], self.references.y_source)
    
    def register(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        if self.references.kissing:
            return tf.keras.Model(self.inputs, [self.references.y_source, self.references.y_target, self.references.pos_flow_dir, self.references.neg_flow_dir])
        else:
            return tf.keras.Model(self.inputs, [self.references.y_source, self.references.pos_flow])

    def register2(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.register().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = voxelmorph.layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

        
class encoder(tf.keras.Model): # Similar code to voxelmorph's Unet but truncated to only keep the encoder part.
    """
    An encoder architecture that builds off either an input keras model or input shape. Layer features can be
    specified directly as a list of encoder features or as a single integer along with a number of encoder levels.
    """

    def __init__(self,
                 inshape=None,
                 input_model=None,
                 nb_features=[16, 32, 32, 32, 32],
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 kernel_initializer='he_normal',
                 name='enc'):
        """
        Parameters:
            inshape: Optional input tensor shape (including features). e.g. (192, 192, 192, 2).
            input_model: Optional input model that feeds directly into the enc before concatenation.
            nb_features: enc convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the enc features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in enc. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            nb_conv_per_level: Number of convolutions per enc level. Default is 1.
            name: Model name - also used as layer name prefix. Default is 'enc'.
        """

         # have the option of specifying input shape or input model
        if input_model is None:
            if inshape is None:
                raise ValueError('inshape must be supplied if input_model is None')
            enc_input = KL.Input(shape=inshape, name='%s_input' % name)
            model_inputs = [enc_input]
        else:
            if len(input_model.outputs) == 1:
                enc_input = input_model.outputs[0]
            else:
                enc_input = KL.concatenate(input_model.outputs, name='%s_input_concat' % name)
            model_inputs = input_model.inputs

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide enc nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [np.repeat(feats[:-1], nb_conv_per_level),
                           np.repeat(np.flip(feats), nb_conv_per_level)]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        ndims = len(enc_input.get_shape()) - 2
        assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        MaxPooling = getattr(KL, 'MaxPooling%dD' % ndims)
        
        nb_levels = int(len(nb_features) / nb_conv_per_level) + 1
        
        if isinstance(max_pool, int):
            max_pool = [max_pool] * nb_levels

        # configure encoder (down-sampling path)
        last = enc_input
        for level in range(nb_levels - 1):
            for conv in range(nb_conv_per_level):
                nf = nb_features[level * nb_conv_per_level + conv]
                layer_name = '%s_enc_conv_%d_%d' % (name, level, conv)
                last = voxelmorph.networks._conv_block(last, nf, name=layer_name,
                                                       kernel_initializer=kernel_initializer)

            # temporarily use maxpool since downsampling doesn't exist in keras
            last = MaxPooling(max_pool[level], name='%s_enc_pooling_%d' % (name, level))(last)

        super().__init__(inputs=model_inputs, outputs=last, name=name)
