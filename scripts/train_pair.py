import os
import sys
maindir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(maindir)
import glob
import tensorflow as tf
import voxelmorph           
import dwarp
import argparse
import generators
import utils
import numpy
import random

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

parser = argparse.ArgumentParser(description="Training script for dwarp diffeomorphic registration to template. TetraReg segmentation-based initialization")

# training and validation data
parser.add_argument('-it', '--img_train', type=str, required=True, help='Path to the training images.')
parser.add_argument('-st', '--seg_train', type=str, required=True, help='Path to the training segmentations (must match the training images).')
parser.add_argument('-iv', '--img_val', type=str, required=False, default=None, help='Path to the validation images.')
parser.add_argument('-sv', '--seg_val', type=str, required=False, default=None, help='Path to the validation segmentations (must match the validation images).')
parser.add_argument('-lab', '--labels', type=str, required=False, default='dkt', help="Path to the file containing the label numbers (1 line csv file) or 'dkt' for DKT labels.")
# model and its hyper-paramaters
parser.add_argument('-o', '--model', type=str, required=True, help="Path to the output model (.h5).")
parser.add_argument('-vsz', '--vox_size', type=str, required=False, default=[2,2,2], help="Voxel size to resample the images to. Default: [2,2,2].")
parser.add_argument('-gsz', '--grid_size', type=str, required=False, default=[96,128,96], help="Grid size to crop / pad the images to (must be a multiple of 2^k where k is the number of encoder levels). Default: [96,128,96].")
parser.add_argument('-lr', '--learning-rate', type=float, required=False, default=1e-4, help="Learning rate. Default: 1e-4.")
parser.add_argument('-enc', '--enc-nf', type=list, required=False, default=[16, 32, 32, 32, 32], help="Number of encoder features. Default: [16, 32, 32, 32, 32].")
parser.add_argument('-dec', '--dec-nf', type=list, required=False, default=[32, 32, 32, 32, 32, 16, 16], help="Number of decoder features. Default: [32, 32, 32, 32, 32, 16, 16].")
parser.add_argument('-e', '--epochs', type=int, required=False, default=1000, help="Number of epochs. Default: 1000.")
parser.add_argument('-b', '--batch-size', type=int, required=False, default=1, help='Batch size. Default: 1.')
# model losses
parser.add_argument('-l', '--loss', type=str, required=False, default='nlcc', help="Intensity-based similarity loss: 'nlcc' (normalized local squared correlation coefficient) or 'mse' (mean square error). Default: nlcc.")
parser.add_argument('-ls', '--loss-seg', type=str, required=False, default='dice', help="Segmentation-based overlap loss: 'dice' or 'sdf' (signed distance field). Default: 'dice'")
parser.add_argument('-lw', '--loss-win', type=int, required=False, default=5, help="Window diameter (in voxels) for local losses (nlcc). Default: 5")
parser.add_argument('-wi', '--weight-img-loss', type=float, required=False, default=1, help="Weight for the image loss. Default: 1.")
parser.add_argument('-ws', '--weight-seg-loss', type=float, required=False, default=0.1, help="Weight for the segmentation loss. Default: 0.1.")
parser.add_argument('-wr', '--weight-reg-loss', type=float, required=False, default=1, help="Weight for the regularization loss. Default: 1.")
# polaffini parameters
parser.add_argument('-sigma', '--sigma', type=float, required=False, default=15, help='Standard deviation (in mm) for the Gaussian kernel. The higher the sigma, the smoother the output transformation. Use inf for affine transformation. Default: 15.')
parser.add_argument('-downf', '--down_factor', type=float, required=False, default=4, help='Downsampling factor of the transformation. Default: 4.')
parser.add_argument('-omit_labs','--omit_labs', type=int, nargs='+', required=False, default=[], help='List of labels to omit. Default: []. Example: 2 41. 0 (background) is always omitted.')
# other
parser.add_argument('-r', '--resume', type=int, required=False, default=0, help='Resume a traning that stopped for some reason (1: yes, 0: no). Default: 0.')
parser.add_argument('-seed', '--seed', type=int, required=False, default=None, help='Seed for random. Default: None.')

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
args.resume = bool(args.resume)

if args.seed is not None:
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    tf.random.set_seed(args.seed)

with open(args.model[:-3] + '_args.txt', 'w') as file:
    for arg in vars(args):
        file.write("{}: {}\n".format(arg, getattr(args, arg)))

#%% Generators to access training (and validation) data.

if args.labels == 'dkt':
    labels = [2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 26, 28, 31, 41, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
              1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035,
              2002, 2003, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2034, 2035]
else:
    with open(args.labels, 'r') as file:
        labels = file.readline().strip().split(',')
    labels = [int(l) for l in labels]

is_val = args.img_val is not None and args.seg_val is not None
is_sdf = args.loss_seg == 'sdf'

# training data
mov_files = sorted(glob.glob(os.path.join(args.img_train, '*')))  
mov_seg_files = sorted(glob.glob(os.path.join(args.seg_train, '*')))  

gen_train = generators.pair_polaffini(mov_files, 
                                      mov_seg_files=mov_seg_files,
                                      vox_sz=args.vox_size,
                                      grid_sz=args.grid_size,
                                      labels=args.labels,
                                      sdf=is_sdf,
                                      polaffini_sigma=args.sigma,
                                      polaffini_downf=args.down_factor,
                                      polaffini_omit_labs=args.omit_labs,      
                                      batch_size=args.batch_size)
    
n_train = len(mov_files)

# validation data
if not is_val:  
    gen_val = None
    sample = next(gen_train)
else:
    mov_files_val = sorted(glob.glob(os.path.join(args.img_train, '*')))  
    mov_seg_files_val = sorted(glob.glob(os.path.join(args.seg_train, '*')))  
    
    gen_val = generators.pair_polaffini(mov_files_val, 
                                        mov_seg_files=mov_seg_files,
                                        vox_sz=args.vox_size,
                                        grid_sz=args.grid_size,
                                        labels=args.labels,
                                        sdf=is_sdf,
                                        polaffini_sigma=args.sigma,
                                        polaffini_downf=args.down_factor,
                                        polaffini_omit_labs=args.omit_labs,      
                                        batch_size=args.batch_size)
    n_val = len(mov_files_val)
    sample = next(gen_val)


dwarp.utils.print_inputGT(sample)
inshape = sample[0][0].shape[1:-1]

      
#%% Prepare and build the model

if args.loss == 'nlcc':
    # loss_img = dwarp.losses.wLCC(win=args.loss_win).loss
    loss_img = voxelmorph.losses.NCC(win=args.loss_win).loss   
elif args.loss == 'mse':
    loss_img = dwarp.losses.wMSE().loss

if args.loss_seg == 'dice':
    loss_seg = dwarp.losses.Dice(is_onehot=False,
                                 labels=labels).loss
elif is_sdf:
    loss_seg = dwarp.losses.wMSE().loss
    
loss_smo = voxelmorph.losses.Grad('l2', loss_mult=1).loss

losses = [loss_img]*2 + [loss_seg]*2 + [loss_smo]
loss_weights = [args.weight_img_loss]*2 + [args.weight_seg_loss]*2 + [args.weight_reg_loss]
    
optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)


if args.resume:
    # load existing model
    model = dwarp.networks.diffeo_pair_seg.load(args.model)
    with open(args.model[:-3] + '_losses.csv', 'r') as loss_file:
        for initial_epoch, _ in enumerate(loss_file):
            pass
    print('resuming training at epoch: ' + str(initial_epoch))
else:
    # build the model
    model = dwarp.networks.diffeo_pair_seg(inshape=args.grid_size,
                                           nb_labs=sample[0][0].shape[-1],
                                           nb_enc_features=args.enc_nf,
                                           nb_dec_features=args.dec_nf,
                                           int_steps=7,
                                           name='diffeo_pair_seg')
    initial_epoch = 0
  
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)
tf.keras.utils.plot_model(model, to_file=args.model[:-3] + '_plot.png', show_shapes=True, show_layer_names=True)

#%% Train the model

steps_per_epoch = n_train // args.batch_size
if not is_val:   
    val_steps = None
    monitor='loss'
else:
    val_steps = n_val // args.batch_size
    monitor='val_loss'

os.makedirs(os.path.dirname(args.model), exist_ok=True)

model.save(args.model.format(epoch=initial_epoch))

save_callback = tf.keras.callbacks.ModelCheckpoint(args.model, monitor=monitor, mode='min', save_best_only=True)
csv_logger = tf.keras.callbacks.CSVLogger(args.model[:-3] + '_losses.csv', append=True, separator=',')
imgdir = os.path.join(os.path.dirname(args.model), 'imgs')
os.makedirs(imgdir, exist_ok=True)
plot_reg = dwarp.callbacks.plotImgReg(sample[0][1], sample[0][0], os.path.join(imgdir, 'img'), modeltype='diffeo_pair')

hist = model.fit(gen_train,
                 validation_data=gen_val,
                 validation_steps=val_steps,
                 initial_epoch=initial_epoch,
                 epochs=args.epochs,
                 steps_per_epoch=steps_per_epoch,
                 callbacks=[save_callback, csv_logger, plot_reg],
                 verbose=1)

utils.plot_losses(args.model[:-3] + '_losses.csv', is_val=is_val)



    
