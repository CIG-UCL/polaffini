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
import random

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12000)])          
        
parser = argparse.ArgumentParser(description="Training script for dwarp diffeomorphic registration to template. TetraReg segmentation-based initialization")

# training and validation data
parser.add_argument('-t', '--train_data', type=str, required=True, help='Path to the training data initialized using pretrain script (should be the same as the -o from pretrain script).')
parser.add_argument('-v', '--val_data', type=str, required=False, default=None, help='Path to the validation data initialized using pretrain script (should be the same as the -o from pretrain script).')
parser.add_argument('-s', '--use-seg', type=int, required=False, default=0, help='Use segmentations at training (1: yes, 0: no). Default: 0.')
parser.add_argument('-ohot', '--ohot', type=int, required=False, default=1, help='Segmentations are one-hot encoded (1: yes, 0: no). Default: 1.')
# model and its hyper-paramaters
parser.add_argument('-o', '--model', type=str, required=True, help="Path to the output model (.h5).")
parser.add_argument('-lr', '--learning-rate', type=float, required=False, default=1e-4, help="Learning rate. Default: 1e-4.")
parser.add_argument('-enc', '--enc-nf', type=int, nargs='+', required=False, default=[16, 32, 32, 32, 32], help="Number of encoder features. Default: 16 32 32 32 32.")
parser.add_argument('-dec', '--dec-nf', type=int, nargs='+', required=False, default=[32, 32, 32, 32, 32, 16, 16], help="Number of decoder features. Default: 32 32 32 32 32 16 16.")
parser.add_argument('-e', '--epochs', type=int, required=True, help="Number of epochs.")
parser.add_argument('-b', '--batch-size', type=int, required=False, default=1, help='Batch size. Default: 1.')
# model losses
parser.add_argument('-l', '--loss', type=str, required=False, default='nlcc', help="Intensity-based similarity loss: 'nlcc' (normalized local squared correlation coefficient) or 'mse' (mean square error). Default: nlcc.")
parser.add_argument('-ls', '--loss-seg', type=str, required=False, default='dice', help="Segmentation-based overlap loss: 'dice' or other to be added. Default: 'dice'")
parser.add_argument('-lw', '--loss-win', type=int, required=False, default=5, help="Window diameter (in voxels) for local losses (nlcc). Default: 5")
parser.add_argument('-wi', '--weight-img-loss', type=float, required=False, default=1, help="Weight for the image loss. Default: 1.")
parser.add_argument('-ws', '--weight-seg-loss', type=float, required=False, default=0.01, help="Weight for the segmentation loss. Default: 0.01.")
parser.add_argument('-wr', '--weight-reg-loss', type=float, required=False, default=1, help="Weight for the regularization loss. Default: 1.")
# other
parser.add_argument('-r', '--resume', type=int, required=False, default=0, help='Resume a traning that stopped for some reason (1: yes, 0: no). Default: 0.')
parser.add_argument('-seed', '--seed', type=int, required=False, default=None, help='Seed for random. Default: None.')

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
args.use_seg = bool(args.use_seg)
args.ohot = bool(args.ohot)
args.resume = bool(args.resume)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

with open(args.model[:-3] + '_args.txt', 'w') as file:
    for arg in vars(args):
        file.write("{}: {}\n".format(arg, getattr(args, arg)))
        
#%% Generators to access training (and validation) data.

is_val = args.val_data is not None

# ref images
ref_file = os.path.join(args.train_data, 'ref_img.nii.gz') 
ref_seg_file = None
if args.use_seg:
    ref_seg_file = os.path.join(args.train_data, 'ref_seg.nii.gz') 

# training images
mov_files = sorted(glob.glob(os.path.join(args.train_data, 'img/*')))  
mov_seg_files = None
if args.use_seg:
    mov_seg_files = sorted(glob.glob(os.path.join(args.train_data, 'seg/*')))
gen_train = generators.mov2atlas_initialized(mov_files = mov_files, 
                                             ref_file = ref_file,
                                             mov_seg_files = mov_seg_files,
                                             ref_seg_file = ref_seg_file,
                                             one_hot=args.ohot,
                                             batch_size = args.batch_size)
n_train = len(mov_files)

# validation images
if not is_val:  
    gen_val = None
    sample = next(gen_train)
else:
    mov_files_val = sorted(glob.glob(os.path.join(args.val_data, 'img/*')))  
    mov_seg_files_val = None
    if args.use_seg:
        mov_seg_files_val = sorted(glob.glob(os.path.join(args.val_data, 'seg/*')))
    gen_val = generators.mov2atlas_initialized(mov_files = mov_files_val, 
                                               ref_file = ref_file,
                                               mov_seg_files = mov_seg_files_val,
                                               ref_seg_file = ref_seg_file,
                                               one_hot=args.ohot,
                                               batch_size = args.batch_size)
    n_val = len(mov_files_val)
    sample = next(gen_val)
    
    
ref = sitk.ReadImage(ref_file)
matO = utils.get_matOrientation(ref)

dwarp.utils.print_inputGT(sample)
inshape = sample[0][0].shape[1:-1]
nfeats = sample[0][0].shape[-1]
nb_labs = None
if args.use_seg:
    nb_labs = sample[0][1].shape[-1]
    if args.ohot:
        labels = None
    else:
        labels = np.unique(sample[1][1]).tolist()
      
#%% Prepare and build the model

if args.loss == 'nlcc':
    # losses = [dwarp.losses.wLCC(win=args.loss_win).loss]
    losses = [voxelmorph.losses.NCC(win=args.loss_win).loss]
elif args.loss == 'mse':
    losses = [dwarp.losses.wMSE().loss]
else:
    sys.exit("Error: only 'mse' and 'nlcc' intensity-based losses accepted.")
loss_weights = [args.weight_img_loss]

if args.use_seg:
    losses += [dwarp.losses.Dice(is_onehot=args.ohot, labels=labels).loss]
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
tf.keras.utils.plot_model(model, to_file=args.model[:-3] + '_plot.png', show_shapes=True, show_layer_names=True)

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
imgdir = os.path.join(os.path.dirname(args.model), 'imgs')
os.makedirs(imgdir, exist_ok=True)
# plot_reg = dwarp.callbacks.plotImgReg(sample[1][0], sample[0][0], os.path.join(imgdir, 'img'), modeltype='diffeo2template')

hist = model.fit(gen_train,
                 validation_data=gen_val,
                 validation_steps=val_steps,
                 initial_epoch=initial_epoch,
                 epochs=args.epochs,
                 steps_per_epoch=steps_per_epoch,
                 callbacks=[save_callback, csv_logger], # plot_reg],
                 verbose=2)

dwarp.utils.plot_losses(args.model[:-3] + '_losses.csv', is_val=args.val_data is not None)



