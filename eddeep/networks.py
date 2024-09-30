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
import os
import sys
maindir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(maindir)
import dwarp.layers as layers



class pix2pix_dis(ne.modelio.LoadableModel): 
    """patchGAN discriminator"""
    
    @ne.modelio.store_config_args
    def __init__(self,
                 volshape,
                 nb_feats=[64,128,256,512],
                 dropout=None, 
                 name='patchGAN_dis'):
        """
        """
        
        x = KL.Input(shape=(*volshape, 1))
        y = KL.Input(shape=(*volshape, 1))
        inputs = KL.concatenate((x, y))
 
        ndims = len(volshape)
        
        use_bias = False 
        kw = 4
        padw = 'same'
        strides = [2]*ndims
        Conv = getattr(KL, 'Conv%dD' % ndims)
        
        last = Conv(nb_feats[0], kernel_size=kw, strides=strides, padding=padw)(inputs)
        last = KL.LeakyReLU(0.2)(last)
        for n in range(1, len(nb_feats)-1): 
            last = Conv(nb_feats[n], kernel_size=kw, strides=strides, padding=padw, use_bias=use_bias)(last)
            last = KL.BatchNormalization()(last)
            if dropout is not None:
                last = KL.Dropout(dropout)(last)
            last = KL.LeakyReLU(0.2)(last)
        
        last = Conv(nb_feats[-1], kernel_size=kw, strides=[1]*ndims, padding=padw, use_bias=use_bias)(last)
        last = KL.BatchNormalization()(last)
        last = KL.LeakyReLU(0.2)(last)
        
        last = Conv(1, kernel_size=kw, strides=[1]*ndims, padding=padw)(last)
        last = tf.keras.activations.sigmoid(last)
        
        super().__init__(inputs=[x, y], outputs=last, name=name)  


class pix2pix_gen(ne.modelio.LoadableModel):
    """
    """
    @ne.modelio.store_config_args
    def __init__(self,
                 volshape,
                 nb_enc_features=[16,32,64,128],
                 nb_dec_features=[128,64,32,16,1],
                 final_activation = None, 
                 name='pix2pix_gen'):
        
        img_in = tf.keras.Input(shape=(*volshape, 1), name='%s_img_input' % name)
        unet = voxelmorph.networks.Unet(inshape=[*volshape,1],
                                        nb_features=[nb_enc_features,nb_dec_features],
                                        final_activation_function=final_activation)
        img_out = unet(img_in)
        
        super().__init__(inputs=img_in, outputs=img_out, name=name)


def gan(generator, discriminator, image_shape):

    for layer in discriminator.layers:
        if not isinstance(layer, KL.BatchNormalization):
            layer.trainable = False

    in_src = KL.Input(shape=(*image_shape, 1))
    gen_out = generator(in_src)
    dis_out = discriminator([in_src, gen_out])
    model = tf.keras.Model(in_src, [dis_out, gen_out])
    
    return model


class eddy_reg(ne.modelio.LoadableModel): 
    
    @ne.modelio.store_config_args
    def __init__(self,
                 volshape,
                 ped,
                 nb_enc_features=[16, 32, 32, 32, 32],
                 nb_dec_features=[32, 32, 32, 32, 32, 16, 16],  # only used if transfo is 'deformable'
                 transfo='linear',                              # 'linear', 'quadratic' or 'deformable'
                 jacob_mod = True,
                 nb_dense_features=[64],
                 name='eddy_reg'):
        
        ndims = len(volshape)
        Conv = getattr(KL, 'Conv%dD' % ndims)
        
        trans_init = KI.RandomNormal(stddev=1e-2)
        lin_init = KI.RandomNormal(stddev=1e-3)  
        quad_init = KI.RandomNormal(stddev=1e-5) 
            
        b0 = KL.Input(shape=(*volshape, 1))
        dw = KL.Input(shape=(*volshape, 1)) 
        
        if transfo == 'deformable':
            newshape = [int(np.ceil(volshape[d] / 2**len(nb_enc_features)) * 2**len(nb_enc_features)) for d in range(ndims)]
            lowerPad = [0] + [int(np.round((newshape[d] - volshape[d]) / 2)) for d in range(ndims)] + [0]
            upperPad = [0] + [int(newshape[d] - volshape[d] - lowerPad[d+1]) for d in range(ndims)] + [0]
            pads = tf.stack((lowerPad, upperPad), axis=1)
            input_model = tf.keras.Model(inputs=[b0, dw], outputs=[tf.pad(b0, pads), tf.pad(dw, pads)])
        else:
            input_model = tf.keras.Model(inputs=[b0, dw], outputs=[b0, dw])
        
        if transfo in ('linear', 'quadratic'):
            
            eddy_model = encoder(input_model=input_model,
                                  nb_features=nb_enc_features,
                                  name='eddy_model')
            
            transfo_params = KL.Flatten()(eddy_model.output)    
                        
            trans_eddy = transfo_params
            lin_eddy = transfo_params
            if transfo == 'quadratic':
                quad_eddy = transfo_params
            trans_rig = transfo_params
            lin_rig = transfo_params
            
            for j, nf in enumerate(nb_dense_features):
                trans_eddy = KL.Dense(nf, activation='relu', name='trans_eddy_%d' % j)(trans_eddy)
                lin_eddy = KL.Dense(nf, activation='relu', name='lin_eddy_%d' % j)(lin_eddy)
                if transfo == 'quadratic':
                    quad_eddy = KL.Dense(nf, activation='relu', name='quad_eddy_%d' % j)(quad_eddy)
                trans_rig = KL.Dense(nf, activation='relu', name='trans_rig_%d' % j)(trans_rig)
                lin_rig = KL.Dense(nf, activation='relu', name='lin_rig_%d' % j)(lin_rig) 
        
            trans_eddy = KL.Dense(1, kernel_initializer=trans_init, name='trans_eddy')(trans_eddy)
            lin_eddy  = KL.Dense(ndims, kernel_initializer=lin_init, name='lin_eddy')(lin_eddy)
            
            transfo_eddy = layers.AffCoeffToMatrix(ndims=ndims,dire=ped,transfo_type='dir_affine', name='build_eddy_affine_transfo')([trans_eddy,lin_eddy])
            transfo_eddy = voxelmorph.layers.AffineToDenseShift(shape=volshape, shift_center=False)(transfo_eddy)
            if transfo == 'quadratic':
                quad_eddy = KL.Dense(ndims*(ndims+1)/2, kernel_initializer=quad_init, name='quad_eddy')(quad_eddy)
                transfo_quad_eddy = layers.QuadCoeffToMatrix(ndims=ndims, name='build_quad_transfo')(quad_eddy)
                transfo_quad_eddy = layers.QuadUnidirToDenseShift(shape=volshape, dire=ped, shift_center=False)(transfo_quad_eddy)
                transfo_eddy = KL.add((transfo_eddy, transfo_quad_eddy))
                
        elif transfo == 'deformable':
            # eddy part
            eddy_model = voxelmorph.networks.Unet(input_model=input_model,
                                                  nb_features=[nb_enc_features, nb_dec_features],
                                                  name='eddy_model')
            transfo_eddy = layers.unpad(eddy_model.output, pads)
            transfo_eddy = Conv(1, kernel_size=3, padding='same', kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5), name='%s_field' % name)(transfo_eddy)
            transfo_eddy_list = []  
            for i in range(0, ndims):
                if i == ped:
                    transfo_eddy_list.append(transfo_eddy)
                else:
                    transfo_eddy_list.append(tf.keras.backend.zeros_like(transfo_eddy))
            transfo_eddy = KL.concatenate(transfo_eddy_list)

            # rigid part
            bottleneckLayer_name = 'eddy_model_dec_conv_%d_0_activation' % (len(nb_enc_features)-1)
            rigid_params = KL.Flatten()(eddy_model.get_layer(bottleneckLayer_name).output)     
            trans_rig = rigid_params
            lin_rig = rigid_params
            for j, nf in enumerate(nb_dense_features):
                trans_rig = KL.Dense(nf, activation='relu', name='trans_rig_%d' % j)(trans_rig)
                lin_rig = KL.Dense(nf, activation='relu', name='lin_rig_%d' % j)(lin_rig) 
                
        trans_rig = KL.Dense(ndims, kernel_initializer=trans_init, name='trans_rig')(trans_rig)
        lin_rig  = KL.Dense(ndims, kernel_initializer=lin_init, name='lin_rig')(lin_rig)    
        transfo_rig = layers.AffCoeffToMatrix(ndims=ndims,transfo_type='rigid', name='build_rigid_transfo')([trans_rig,lin_rig])

        full_transfo = voxelmorph.layers.ComposeTransform(shift_center=False, name='compose_transfos')([transfo_eddy, transfo_rig])

        dw_mov = voxelmorph.layers.SpatialTransformer(interp_method='linear', indexing='ij', name='full_transformer')([dw, full_transfo])
        if jacob_mod:        
            dw_mov = layers.JacobianMultiplyIntensities(indexing='ij', outDet=True, is_shift=True, name='jac_modul_dw')([dw_mov, full_transfo])
        
        outputs = [tf.concat((b0, dw_mov), axis=-1)]
        if transfo == 'deformable':
            outputs += [transfo_eddy]
        
        super().__init__(inputs=[b0, dw], outputs=outputs, name=name) 

        # cache pointers to layers and tensors for future reference
        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.full_transfo = full_transfo
        self.references.dw_mov = dw_mov
        self.references.jacob_mod = jacob_mod
        
        
    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, [self.references.dw_mov,self.references.full_transfo])
    
    def apply_corr(self, in_b0, in_dw, dw, get_transfo=False):

        warp_model = self.get_registration_model()
        dw_input = tf.keras.Input(shape=dw.shape[1:])
        dw_corr = voxelmorph.layers.SpatialTransformer(interp_method='linear', indexing='ij')([dw_input, warp_model.output[1]])
        if self.references.jacob_mod:
            dw_corr = layers.JacobianMultiplyIntensities(indexing='ij')([dw_corr, warp_model.output[1]])
        outputs = [dw_corr, warp_model.output[0]]
        if get_transfo:
            outputs += [warp_model.output[1]]
        return tf.keras.Model(warp_model.inputs + [dw_input], outputs).predict([in_b0, in_dw, dw], verbose=0)
    

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
   