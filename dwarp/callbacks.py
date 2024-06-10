import os
import tensorflow as tf
import voxelmorph
import matplotlib.pyplot as plt
import numpy as np
from . import utils, layers, losses


class plotImgReg(tf.keras.callbacks.Callback):
   
    def __init__(self, ref, mov, img_prefix, dim=0, modeltype='diffeo_pair'):
        self.dim = dim
        self.ref = ref
        self.mov = mov
        center = np.mean(np.where(ref[0,...,0]>0), axis=1)
        self.sl = int(center[dim])
        self.img_prefix = img_prefix
        self.modeltype = modeltype
        
    def on_epoch_begin(self, epoch, logs=None):
        if self.modeltype == 'diffeo_pair':
            moved, _, _, _, _ = self.model.register(self.mov, self.ref)
        elif self.modeltype == 'diffeo2template':
            moved, _, _ = self.model.register(self.mov)
        moved = moved[0, ..., 0]
        moved = moved.take(self.sl, axis=self.dim)[...,np.newaxis]
        moved= np.uint8(moved * [0,127,255])
        
        ref = self.ref[0, ..., 0]
        ref = ref.take(self.sl, axis=self.dim)[...,np.newaxis]
        ref = np.uint8(ref * [255,127,0])

        if self.dim in (1, 2):
            ref = np.flipud(ref)
            moved = np.flipud(moved)
        img = np.concatenate((ref, moved, ref+moved), axis=1)

        plt.imsave(self.img_prefix + '_' + str(epoch) + '.jpg', img)


class plotImgReg2(tf.keras.callbacks.Callback):
   
    def __init__(self, x, sl_sag, sl_axi, is_aux=False, is_weighted=False):
        self.is_aux = is_aux
        self.is_weighted = is_weighted
        self.sl_sag = sl_sag
        self.sl_axi = sl_axi
        self.mov = x[0][0]
        self.ref = x[1][0]
        self.inputs = x[0]
        if is_aux:
            self.mov_aux = x[0][1]
            self.ref_aux = x[1][1]     
        self.inshape = x[1][0].shape[1:-1]
        self.gi_sag = np.expand_dims(utils.grid_img(self.inshape, omitdim=[2], spacing=5), [0,-1])
        self.gi_axi = np.expand_dims(utils.grid_img(self.inshape, omitdim=[0], spacing=5), [0,-1])

    def on_epoch_begin(self, epoch, logs=None):
        
        moved, moved_aux, svf = self.model.predict(self.inputs)
        
        transfo = voxelmorph.utils.integrate_vec(tf.constant(svf[0]), nb_steps=7)
        transfo = tf.expand_dims(transfo, axis=0)
        # _, jac = utils.jacobian(transfo, outDet=True)
        jac = losses.Grad().loss_map(np.repeat(0, 1),tf.constant(svf))
        moved_gi_sag = layers.Resampler(fill_value=0,interp_method='linear')([self.gi_sag, transfo])
        moved_gi_axi = layers.Resampler(fill_value=0,interp_method='linear')([self.gi_axi, transfo])
        
        lcc = losses.wLCC(win=5,is_weighted=self.is_weighted).loss_map(self.ref,moved) 
        mse = tf.abs(self.ref - moved)
        
        f, axs = plt.subplots(1+self.is_aux, 6); f.dpi = 200
        plt.subplots_adjust(wspace=0.01,hspace=-0.63)
        axs[0][0].imshow(np.fliplr(self.mov[0,:,:,self.sl_sag,0]), vmin=0, vmax=1, origin="lower")
        axs[0][0].set_title('moving', fontsize=7)
        axs[0][0].axis('off')
        axs[0][1].imshow(np.fliplr(self.ref[0,:,:,self.sl_sag,0]), vmin=0, vmax=1, origin="lower")
        axs[0][1].set_title('target', fontsize=7)
        axs[0][1].axis('off')
        axs[0][2].imshow(np.fliplr(moved[0,:,:,self.sl_sag,0]), vmin=0, vmax=1, origin="lower")
        axs[0][2].set_title('moved', fontsize=7)
        axs[0][2].axis('off')
        axs[0][5].imshow(np.fliplr(moved_gi_sag[0,:,:,self.sl_sag,0]), origin="lower")
        axs[0][5].set_title('transfo', fontsize=7)
        axs[0][5].axis('off')
        axs[0][3].imshow(np.fliplr(mse[0,:,:,self.sl_sag,0]), vmin=0, vmax=1, origin="lower")
        axs[0][3].set_title('mse', fontsize=7)
        axs[0][3].axis('off') 
        axs[0][4].imshow(np.fliplr(lcc[0,:,:,self.sl_sag]), vmin=0.8, vmax=1, origin="lower")
        axs[0][4].set_title('lcc', fontsize=7)
        axs[0][4].axis('off')
        if self.is_aux:
            lcc_aux = losses.wLCC(win=5, is_weighted=self.is_weighted).loss_map(self.ref_aux,moved_aux) 
            mse_aux = tf.abs(self.ref_aux - moved_aux)
            axs[1][0].imshow(np.fliplr(self.mov_aux[0,:,:,self.sl_sag,0]), vmin=0, vmax=1, origin="lower")
            axs[1][0].axis('off')
            axs[1][1].imshow(np.fliplr(self.ref_aux[0,:,:,self.sl_sag,0]), vmin=0, vmax=1, origin="lower")
            axs[1][1].axis('off')
            axs[1][2].imshow(np.fliplr(moved_aux[0,:,:,self.sl_sag,0]), vmin=0, vmax=1, origin="lower")
            axs[1][2].axis('off')
            axs[1][3].imshow(np.fliplr(mse_aux[0,:,:,self.sl_sag,0]), vmin=0, vmax=1, origin="lower")
            axs[1][3].axis('off') 
            axs[1][4].imshow(np.fliplr(lcc_aux[0,:,:,self.sl_sag]), vmin=0.5, vmax=1, origin="lower")
            axs[1][4].axis('off')
            axs[1][5].imshow(np.fliplr(jac[0,:,:,self.sl_sag]), vmin=0, vmax=2, origin="lower")
            axs[1][5].axis('off')
        plt.suptitle('epoch: ' + str(epoch), ha='center', y=0.75, fontsize=8)
        plt.show()

        f, axs = plt.subplots(1+self.is_aux, 6); f.dpi = 200
        plt.subplots_adjust(wspace=0.01,hspace=-0.63)
        axs[0][0].imshow(np.fliplr(self.mov[0,self.sl_axi,:,:,0]), vmin=0, vmax=1, origin="lower")
        axs[0][0].set_title('moving', fontsize=7)
        axs[0][0].axis('off')
        axs[0][1].imshow(np.fliplr(self.ref[0,self.sl_axi,:,:,0]), vmin=0, vmax=1, origin="lower")
        axs[0][1].set_title('target', fontsize=7)
        axs[0][1].axis('off')
        axs[0][2].imshow(np.fliplr(moved[0,self.sl_axi,:,:,0]), vmin=0, vmax=1, origin="lower")
        axs[0][2].set_title('moved', fontsize=7)
        axs[0][2].axis('off')
        axs[0][5].imshow(np.fliplr(moved_gi_axi[0,self.sl_axi,:,:,0]), origin="lower")
        axs[0][5].set_title('transfo', fontsize=7)
        axs[0][5].axis('off')
        axs[0][3].imshow(np.fliplr(mse[0,self.sl_axi,:,:,0]), vmin=0, vmax=1, origin="lower")
        axs[0][3].set_title('mse', fontsize=7)
        axs[0][3].axis('off') 
        axs[0][4].imshow(np.fliplr(lcc[0,self.sl_axi,:,:,0]), vmin=0, vmax=1, origin="lower")
        axs[0][4].set_title('lcc', fontsize=7)
        axs[0][4].axis('off')
        if self.is_aux:
            axs[1][0].imshow(np.fliplr(self.mov_aux[0,self.sl_axi,:,:,0]), vmin=0, vmax=1, origin="lower")
            axs[1][0].axis('off')
            axs[1][1].imshow(np.fliplr(self.ref_aux[0,self.sl_axi,:,:,0]), vmin=0, vmax=1, origin="lower")
            axs[1][1].axis('off')
            axs[1][2].imshow(np.fliplr(moved_aux[0,self.sl_axi,:,:,0]), vmin=0, vmax=1, origin="lower")
            axs[1][2].axis('off')
            axs[1][3].imshow(np.fliplr(mse_aux[0,self.sl_axi,:,:,0]), vmin=0, vmax=1, origin="lower")
            axs[1][3].axis('off') 
            axs[1][4].imshow(np.fliplr(lcc_aux[0,self.sl_axi,:,:]), vmin=0, vmax=1, origin="lower")
            axs[1][4].axis('off')
            axs[1][5].imshow(np.fliplr(jac[0,self.sl_axi,:,:]), vmin=0, vmax=2, origin="lower")
            axs[1][5].axis('off')
        plt.suptitle('epoch: ' + str(epoch), ha='center', y=0.78, fontsize=8)
        plt.show()
 
