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

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

parser = argparse.ArgumentParser(description="Training script for dwarp diffeomorphic registration to template. TetraReg segmentation-based initialization")

# training and validation data
parser.add_argument('-t', '--train_data', type=str, required=True, help='Path to the training data initialized using pretrain script (should be the same as the -o from pretrain script).')
parser.add_argument('-v', '--val_data', type=str, required=False, default=None, help='Path to the validation data initialized using pretrain script (should be the same as the -o from pretrain script).')
parser.add_argument('-s', '--use-seg', type=int, required=False, default=0, help='Use segmentations at training (1: yes, 0: no). Default: 0.')
# model and its hyper-paramaters
parser.add_argument('-o', '--model', type=str, required=True, help="Path to the output model (.h5).")
parser.add_argument('-lr', '--learning-rate', type=float, required=False, default=1e-4, help="Learning rate. Default: 1e-4.")
parser.add_argument('-enc', '--enc-nf', type=list, required=False, default=[16, 32, 32, 32, 32], help="Number of encoder features. Default: [16, 32, 32, 32, 32].")
parser.add_argument('-dec', '--dec-nf', type=list, required=False, default=[32, 32, 32, 32, 32, 16, 16], help="Number of decoder features. Default: [32, 32, 32, 32, 32, 16, 16].")
parser.add_argument('-e', '--epochs', type=int, required=True, help="Number of epochs.")
parser.add_argument('-b', '--batch-size', type=int, required=False, default=1, help='Batch size. Default: 1.')
# model losses
parser.add_argument('-l', '--loss', type=str, required=False, default='nlcc', help="Intensity-based similarity loss: 'nlcc' (normalized local squared correlation coefficient) or 'mse' (mean square error). Default: nlcc.")
parser.add_argument('-ls', '--loss-seg', type=str, required=False, default='dice', help="Segmentation-based overlap loss: 'dice' or other to be added. Default: 'dice'")
parser.add_argument('-lw', '--loss-win', type=int, required=False, default=5, help="Window diameter (in voxels) for local losses (nlcc). Default: 5")
parser.add_argument('-ws', '--weight-seg-loss', type=float, required=False, default=0.01, help="Weight for the segmentation loss. Default: 0.01.")
parser.add_argument('-wr', '--weight-reg-loss', type=float, required=False, default=1, help="Weight for the regularization loss. Default: 1.")
# other
parser.add_argument('-r', '--resume', type=int, required=False, default=0, help='Resume a traning that stopped for some reason (1: yes, 0: no). Default: 0.')


args = parser.parse_args()
args.use_seg = bool(args.use_seg)
args.resume = bool(args.resume)

#%% Generators to access training (and validation) data.

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
gen_train = dwarp.generators.mov2atlas_initialized(mov_files = mov_files, 
                                                   ref_file = ref_file,
                                                   mov_seg_files = mov_seg_files,
                                                   ref_seg_file = ref_seg_file,
                                                   batch_size = args.batch_size)
n_train = len(mov_files)

# validation images
if args.val_data is None:  
    gen_val = None
else:
    mov_files_val = sorted(glob.glob(os.path.join(args.val_data, 'img/*')))  
    mov_seg_files_val = None
    if args.use_seg:
        mov_seg_files_val = sorted(glob.glob(os.path.join(args.val_data, 'seg/*')))
    gen_val = dwarp.generators.mov2atlas_initialized(mov_files = mov_files_val, 
                                                     ref_file = ref_file,
                                                     mov_seg_files = mov_seg_files_val,
                                                     ref_seg_file = ref_seg_file,
                                                     batch_size = args.batch_size)
    n_val = len(mov_files_val)

ref = sitk.ReadImage(ref_file)
matO = dwarp.utils.get_matOrientation(ref)

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

# tensorflow device handling
device, nb_devices = voxelmorph.tf.utils.setup_device(1)
assert np.mod(args.batch_size, nb_devices) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_devices)

with tf.device(device):
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
    
    dwarp.utils.plot_losses(args.model[:-3] + '_losses.csv', is_val=args.val_data is not None)



    