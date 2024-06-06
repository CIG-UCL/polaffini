import os
import sys
maindir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(maindir)
import glob
import numpy as np
import tensorflow as tf
import voxelmorph           
import dwarp
import SimpleITK as sitk
import argparse
import generators
import utils

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

parser = argparse.ArgumentParser(description="Training script for dwarp diffeomorphic registration to template. TetraReg segmentation-based initialization")

# training and validation data
parser.add_argument('-it', '--img_train', type=str, required=True, help='Path to the training images.')
parser.add_argument('-st', '--train_data', type=str, required=True, help='Path to the training segmentations (must match the training images).')
parser.add_argument('-iv', '--img_val', type=str, required=False, default=None, help='Path to the validation images.')
parser.add_argument('-sv', '--val_data', type=str, required=False, default=None, help='Path to the validation segmentations (must match the validation images).')
# model and its hyper-paramaters
parser.add_argument('-o', '--model', type=str, required=True, help="Path to the output model (.h5).")
parser.add_argument('-vsz', '--vox_size', type=str, required=False, default=[2,2,2], help="Voxel size to resample the images to. Default: [2,2,2].")
parser.add_argument('-gsz', '--grid_size', type=str, required=False, default=[96,128,96], help="Grid size to crop / pad the images to (must be of form 2^k where k > n_encoder_levels). Default: [96,128,96].")
parser.add_argument('-lr', '--learning-rate', type=float, required=False, default=1e-4, help="Learning rate. Default: 1e-4.")
parser.add_argument('-enc', '--enc-nf', type=list, required=False, default=[16, 32, 32, 32, 32], help="Number of encoder features. Default: [16, 32, 32, 32, 32].")
parser.add_argument('-dec', '--dec-nf', type=list, required=False, default=[32, 32, 32, 32, 32, 16, 16], help="Number of decoder features. Default: [32, 32, 32, 32, 32, 16, 16].")
parser.add_argument('-e', '--epochs', type=int, required=False, default=1000, help="Number of epochs. Default: 1000.")
parser.add_argument('-b', '--batch-size', type=int, required=False, default=1, help='Batch size. Default: 1.')
# model losses
parser.add_argument('-l', '--loss', type=str, required=False, default='nlcc', help="Intensity-based similarity loss: 'nlcc' (normalized local squared correlation coefficient) or 'mse' (mean square error). Default: nlcc.")
parser.add_argument('-ls', '--loss-seg', type=str, required=False, default='dice', help="Segmentation-based overlap loss: 'dice' or other to be added. Default: 'dice'")
parser.add_argument('-lw', '--loss-win', type=int, required=False, default=5, help="Window diameter (in voxels) for local losses (nlcc). Default: 5")
parser.add_argument('-ws', '--weight-seg-loss', type=float, required=False, default=0.1, help="Weight for the segmentation loss. Default: 0.1.")
parser.add_argument('-wr', '--weight-reg-loss', type=float, required=False, default=1, help="Weight for the regularization loss. Default: 1.")
# polaffini parameters
parser.add_argument('-sigma', '--sigma', type=float, required=False, default=15, help='Standard deviation (in mm) for the Gaussian kernel. The higher the sigma, the smoother the output transformation. Use inf for affine transformation. Default: 15.')
parser.add_argument('-downf', '--down_factor', type=float, required=False, default=4, help='Downsampling factor of the transformation. Default: 4.')
parser.add_argument('-omit_labs','--omit_labs', type=int, nargs='+', required=False, default=[], help='List of labels to omit. Default: []. Example: 2 41. 0 (background) is always omitted.')
# other
parser.add_argument('-r', '--resume', type=int, required=False, default=0, help='Resume a traning that stopped for some reason (1: yes, 0: no). Default: 0.')


args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
args.use_seg = bool(args.use_seg)
args.resume = bool(args.resume)

#%% Generators to access training (and validation) data.

if args.labels == 'dkt':
    labels = [2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28, 31, 41, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
              1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035,
              2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2034, 2035]

# training images
mov_files = sorted(glob.glob(os.path.join(args.img_train, '*')))  
mov_seg_files = sorted(glob.glob(os.path.join(args.seg_train, '*')))  

gen_train = generators.pair_polaffini(mov_files, 
                                      mov_seg_files=mov_seg_files,
                                      vox_sz=args.vox_size,
                                      grid_sz=args.grid_size,
                                      labels=args.labels,
                                      polaffini_sigma=args.sigma,
                                      polaffini_downf=args.downf,
                                      polaffini_omit_labs=args.omit_labs,      
                                      batch_size=args.batch_size)
    
n_train = len(mov_files)

# validation images
if args.val_data is None:  
    gen_val = None
else:
    mov_files_val = sorted(glob.glob(os.path.join(args.img_train, '*')))  
    mov_seg_files_val = sorted(glob.glob(os.path.join(args.seg_train, '*')))  
    
    gen_val = generators.pair_polaffini(mov_files_val, 
                                        mov_seg_files=mov_seg_files,
                                        vox_sz=args.vox_size,
                                        grid_sz=args.grid_size,
                                        labels=args.labels,
                                        polaffini_sigma=args.sigma,
                                        polaffini_downf=args.downf,
                                        polaffini_omit_labs=args.omit_labs,      
                                        batch_size=args.batch_size)
    n_val = len(mov_files_val)

sample_train = next(gen_train)
dwarp.utils.print_inputGT(sample_train)
inshape = sample_train[0][0].shape[1:-1]
nfeats = sample_train[0][0].shape[-1]
nb_labs = None
if args.use_seg:
    nb_labs = sample_train[0][1].shape[-1]
      
#%% Prepare and build he model

if args.loss == 'nlcc':
    losses = [dwarp.losses.wLCC(win=args.loss_win).loss]
elif args.loss == 'mse':
    losses = [dwarp.losses.wMSE().loss]
else:
    sys.exit("Error: only 'mse' and 'nlcc' intensity-based losses accepted.")
loss_weights = [1]

if args.use_seg:
    losses += [dwarp.losses.Dice().loss]
    loss_weights += [args.weight_seg_loss]
    
losses += [voxelmorph.losses.Grad('l2', loss_mult=1).loss]
loss_weights += [args.weight_reg_loss]

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

# # TensorFlow handling
# device, nb_devices = vxm.tf.utils.setup_device(arg.gpu)
# assert np.mod(arg.batch_size, nb_devices) == 0, \
#     f'batch size {arg.batch_size} not a multiple of the number of GPUs {nb_devices}'
# assert tf.__version__.startswith('2'), f'TensorFlow version {tf.__version__} is not 2 or later'

# with tf.device(device):
if args.resume:
    # load existing model
    model = dwarp.networks.diffeo2atlas.load(args.model)
    with open(args.model[:-3] + '_losses.csv', 'r') as loss_file:
        for initial_epoch, _ in enumerate(loss_file):
            pass
    print('resuming training at epoch: ' + str(initial_epoch))
else:
    # build the model
    model = dwarp.networks.diffeo2atlas(inshape=inshape,
                                        orientation=matO,
                                        nb_enc_features=args.enc_nf,
                                        nb_dec_features=args.dec_nf,
                                        src_feats=nfeats,
                                        nb_labs = nb_labs,
                                        int_steps=7)
    initial_epoch = 0
  
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

#%% Train the model

steps_per_epoch = n_train // args.batch_size
if args.val_data is None:   
    val_steps = None
    monitor='loss'
else:
    val_steps = n_val // args.batch_size
    monitor='val_loss'

os.makedirs(os.path.dirname(args.model), exist_ok=True)

model.save(args.model.format(epoch=initial_epoch))

save_callback = tf.keras.callbacks.ModelCheckpoint(args.model, monitor=monitor, mode='min', save_best_only=True)
csv_logger = tf.keras.callbacks.CSVLogger(args.model[:-3] + '_losses.csv', append=True, separator=',')

hist = model.fit(gen_train,
                 validation_data=gen_val,
                 validation_steps=val_steps,
                 initial_epoch=initial_epoch,
                 epochs=args.epochs,
                 steps_per_epoch=steps_per_epoch,
                 callbacks=[save_callback, csv_logger],
                 verbose=1)

utils.plot_losses(args.model[:-3] + '_losses.csv', is_val=args.val_data is not None)



    