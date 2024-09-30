import os
import sys
maindir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(maindir)
import glob
import tensorflow as tf
import voxelmorph           
import dwarp
import argparse
import dwarp.generators as generators
import SimpleITK as sitk
import numpy as np
import random
import time
import pandas

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

parser = argparse.ArgumentParser(description="Training script for dwarp diffeomorphic registration to template. TetraReg segmentation-based initialization")

# training and validation data
parser.add_argument('-it', '--img_train', type=str, required=True, help='Path to the training images.')
parser.add_argument('-st', '--seg_train', type=str, required=True, help='Path to the training segmentations (must match the training images).')
parser.add_argument('-iv', '--img_val', type=str, required=False, default=None, help='Path to the validation images.')
parser.add_argument('-sv', '--seg_val', type=str, required=False, default=None, help='Path to the validation segmentations (must match the validation images).')
parser.add_argument('-lab', '--labels', type=str, required=False, default=None, help="Path to the file containing the label numbers (1 line csv file) or 'dkt' for DKT labels. Default: unique labels of the first segmentation.")
parser.add_argument('-omit-slabs', '--omit_slabs', type=int, nargs='+', required=False, default=[], help='List of labels to omit for the segmentation overlap loss. Default: []. Example: 2 24 41. 0 (background) is always omitted.')
# model and its hyper-paramaters
parser.add_argument('-o', '--model', type=str, required=True, help="Path to the output model (.h5).")
parser.add_argument('-vsz', '--vox_size', type=int, nargs='+', required=False, default=[2,2,2], help="Voxel size to resample the images to. Default: 2 2 2.")
parser.add_argument('-gsz', '--grid_size', type=int, nargs='+', required=False, default=[96,128,96], help="Grid size to crop / pad the images to (must be a multiple of 2^k where k is the number of encoder levels). Default: 96 128 96.")
parser.add_argument('-lr', '--learning-rate', type=float, required=False, default=1e-4, help="Learning rate. Default: 1e-4.")
parser.add_argument('-enc', '--enc-nf', type=int, nargs='+', required=False, default=[16, 32, 32, 32, 32], help="Number of encoder features. Default: 16 32 32 32 32.")
parser.add_argument('-dec', '--dec-nf', type=int, nargs='+', required=False, default=[32, 32, 32, 32, 32, 16, 16], help="Number of decoder features. Default: 32 32 32 32 32 16 16.")
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
parser.add_argument('-sigma', '--sigma', type=float, required=False, default=15, help='Standard deviation (in mm) for the Gaussian kernel for POLAFFINI. The higher the sigma, the smoother the output transformation. Use inf for affine transformation. Default: 15.')
parser.add_argument('-downf', '--down_factor', type=float, required=False, default=4, help='Downsampling factor of the transformation for POLAFFINI. Default: 4.')
parser.add_argument('-omit-labs','--omit_labs', type=int, nargs='+', required=False, default=[], help='List of labels to omit for POLAFFINI. Default: []. Example: 2 24 41. 0 (background) is always omitted.')
parser.add_argument('-p-gpu', '--polaffini_gpu', type=int, required=False, default=0, help='POLAFFINI with GPU implementation (1: yes, 0: no). Default: 0.')
# other
parser.add_argument('-r', '--resume', type=int, required=False, default=0, help='Resume a traning that stopped for some reason (1: yes, 0: no). Default: 0.')
parser.add_argument('-seed', '--seed', type=int, required=False, default=None, help='Seed for random. Default: None.')

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
args.resume = bool(args.resume)
args.polaffini_gpu = bool(args.polaffini_gpu)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

os.makedirs(os.path.dirname(args.model), exist_ok=True)

with open(args.model[:-3] + '_args.txt', 'w') as file:
    for arg in vars(args):
        file.write("{}: {}\n".format(arg, getattr(args, arg)))

#%% Generators to access training (and validation) data.

is_val = args.img_val is not None and args.seg_val is not None
is_sdf = args.loss_seg == 'sdf'

# training data
mov_files = sorted(glob.glob(os.path.join(args.img_train, '*')))  
mov_seg_files = sorted(glob.glob(os.path.join(args.seg_train, '*')))  

if args.labels is None:
    labels = np.unique(sitk.GetArrayFromImage(sitk.ReadImage(mov_seg_files[0])))
elif args.labels == 'dkt':
    labels = np.array(generators.get_labels('dkt'))
else:
    with open(args.labels, 'r') as file:
        labels = file.readline().strip().split(',')
    labels = np.array([int(l) for l in labels])
for l in args.omit_slabs + [0]:
    labels = np.delete(labels, labels==l)    

gen_train = generators.pair_polaffini(mov_files=mov_files, 
                                      mov_seg_files=mov_seg_files,
                                      vox_sz=args.vox_size,
                                      grid_sz=args.grid_size,
                                      labels=labels,
                                      sdf=is_sdf,
                                      polaffini_sigma=args.sigma,                                                              
                                      polaffini_downf=args.down_factor,
                                      polaffini_omit_labs=args.omit_labs,  
                                      polaffini_usegpu=args.polaffini_gpu,
                                      batch_size=args.batch_size)
    
n_train = len(mov_files)

# validation data
if not is_val:  
    gen_val = None
    sample = next(gen_train)
else:
    mov_files_val = sorted(glob.glob(os.path.join(args.img_val, '*')))  
    mov_seg_files_val = sorted(glob.glob(os.path.join(args.seg_val, '*')))  
    
    gen_val = generators.pair_polaffini(mov_files=mov_files_val, 
                                        mov_seg_files=mov_seg_files_val,
                                        vox_sz=args.vox_size,
                                        grid_sz=args.grid_size,
                                        labels=labels,
                                        sdf=is_sdf,
                                        polaffini_sigma=args.sigma,
                                        polaffini_downf=args.down_factor,
                                        polaffini_omit_labs=args.omit_labs,  
                                        polaffini_usegpu=args.polaffini_gpu,
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
loss_file = args.model[:-3] + '_losses.csv'

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)


if args.resume:
    # load existing model
    model = dwarp.networks.diffeo_pair_seg.load(args.model)
    tab_loss = pandas.read_csv(loss_file, sep=',')
    if is_val:
        best_loss = np.min(tab_loss.val_loss)
    else:
        best_loss = np.min(tab_loss.loss)
    initial_epoch = tab_loss.epoch.iloc[-1]
    print('resuming training at epoch: ' + str(initial_epoch))
    
else:
    # build the model
    model = dwarp.networks.diffeo_pair_seg(inshape=args.grid_size,
                                           vox_sz=args.vox_size,
                                           nb_labs=sample[0][0].shape[-1],
                                           nb_enc_features=args.enc_nf,
                                           nb_dec_features=args.dec_nf,
                                           int_steps=7,
                                           name='diffeo_pair_seg')
    initial_epoch = 0
    best_loss = np.Inf
    try: 
        os.remove(loss_file)
    except OSError:
        pass
    f = open(loss_file,'w')
    if is_val:
        f.write('epoch,loss,img_pos,img_neg,seg_pos,seg_neg,reg,val_loss,val_img_pos,val_img_neg,val_seg_pos,val_seg_neg,val_reg,\n')
    else:
        f.write('epoch,loss,img_pos,img_neg,seg_pos,seg_neg,reg\n') 
                    
                    
model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)
tf.keras.utils.plot_model(model, to_file=args.model[:-3] + '_plot.png', show_shapes=True, show_layer_names=True)

#%% Train the model


n_train_steps = n_train // args.batch_size
if is_val:
    n_val_steps = n_val // args.batch_size
    

for epoch in range(initial_epoch, args.epochs):   
    t = time.time()
    
    print('epoch: %d/%d' % (epoch+1, args.epochs), end=' |')
    v_loss_epoch = []    
    
    for _ in range(n_train_steps):
            
        inputs, targets = next(gen_train)
        
        v_loss = model.train_on_batch(inputs, targets)
        v_loss_epoch += [v_loss] 
    
        print('-', end='')
        
    v_loss_epoch = np.mean(v_loss_epoch, axis=0)
    
    if is_val:
        v_loss_epoch_val = [] 
        for _ in range(n_val_steps):

            inputs, targets = next(gen_val)          
           
            v_loss_val = model.test_on_batch(inputs, targets)
            v_loss_epoch_val += [v_loss_val] 
                
            print('.', end='')
            
        v_loss_epoch_val = np.mean(v_loss_epoch_val, axis=0)
 
        
    print('| ' + str(np.round(time.time()-t,3)) + ' s')
    f = open(loss_file,'a')
    
    print('train - all: %.3e, img_pos: %.3e, img_neg: %.3e, seg_pos: %.3e, seg_neg: %.3e, reg: %.3e' % tuple(v_loss_epoch))
    if is_val:
        print('val   - all: %.3e, img_pos: %.3e, img_neg: %.3e, seg_pos: %.3e, seg_neg: %.3e, reg: %.3e' % tuple(v_loss_epoch_val))
        f.write(str(epoch+1) + ',' + str(v_loss_epoch[0]) + ',' + str(v_loss_epoch[1]) + ',' + str(v_loss_epoch[2]) + ',' + str(v_loss_epoch[3]) + ',' + str(v_loss_epoch[4]) + ',' + str(v_loss_epoch[5])
                             + ',' + str(v_loss_epoch_val[0]) + ',' + str(v_loss_epoch_val[1]) + ',' + str(v_loss_epoch_val[2]) + ',' + str(v_loss_epoch_val[3]) + ',' + str(v_loss_epoch_val[4]) + ',' + str(v_loss_epoch_val[5]) + '\n')
    else:
        f.write(str(epoch+1) + ',' + str(v_loss_epoch[0]) + ',' + str(v_loss_epoch[1]) + ',' + str(v_loss_epoch[2]) + ',' + str(v_loss_epoch[3]) + ',' + str(v_loss_epoch[4]) + ',' + str(v_loss_epoch[5]) + '\n')
    f.close()
    
    if is_val:
        v_loss_epoch = v_loss_epoch_val[0]
    else:
        v_loss_epoch = v_loss_epoch[0]
    if v_loss_epoch < best_loss:
        best_loss = v_loss_epoch
        model.save(args.model[:-3] + '_best.h5')
    model.save(args.model[:-3] + '_last.h5')


dwarp.utils.plot_losses(args.model[:-3] + '_losses.csv', is_val=is_val)



    
