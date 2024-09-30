import os
import sys
maindir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(maindir)
import glob
import time
import numpy as np
import tensorflow as tf    
import voxelmorph   
import dwarp
import argparse
import random
import matplotlib.pyplot as plt
import pandas
import eddeep

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))       
        
parser = argparse.ArgumentParser(description="Training script for the image translation part of Eddeep.")

# training and validation data, pre-trained translator
parser.add_argument('-t', '--train_data', type=str, required=True, help='Path to the raw training data following nested subfolder structure below.')
parser.add_argument('-v', '--val_data', type=str, required=False, default=None, help='Path to the raw validation data following nested subfolder structure below.')
# ├── sub1
# │   ├── ped1
# │   │   ├── bval1
# │   │   │   ├── sub1_ped1_bval1_dir1.nii.gz
# │   │   │   ├── sub1_ped1_bval1_dir1.nii.gz
# │   │   │   ├── ...
# │   │   ├── bval2
# │   │   │   ├── ...
# │   │   ├── ...
# │   │   ├── sub1_ped1_bvaltarget_meandir.nii.gz 
# │   │   ├── ...
# │   └── ped2
# │       ├── ...
# ├── sub2
# │   ├── ...
# ├── ...
parser.add_argument('-tr', '--trans', type=str, required=True, help="Path to the pre-trained image translation model.")
parser.add_argument('-k', '--kpad', type=int, required=False, default=5, help='k to pad the input so that its shape is of form 2**k. Has to be >= number encoding steps for deformable transfo, and equal to the one used for the translator.')
# distortion constraints
parser.add_argument('-p', '--ped', type=int, required=True, help='Axis number of the phase encoding directions (starting at 0). Usually 1 for AP/PA.')
parser.add_argument('-trsf', '--transfo', type=str, required=False, default='quadratic', help="Type of geometric transformation for the distortion ('linear', 'quadratic' or 'deformable'). Default: 'quadratic'.")
# model and its hyper-paramaters
parser.add_argument('-o', '--model', type=str, required=True, help="Path prefix to the output model (without extension).")
parser.add_argument('-lr', '--learning-rate', type=float, required=False, default=1e-4, help="Learning rate. Default: 1e-4.")
parser.add_argument('-enc', '--enc-nf', type=int, nargs='+', required=False, default=[16,32,40,48,56], help="Number of encoder features for the generator. Default: 16 32 40 48 56.")
parser.add_argument('-dec', '--dec-nf', type=int, nargs='+', required=False, default=[56,48,40,32,24,16,16], help="Number of decoder features for the generator (only for deformable transfo). Default: 56 48 40 32 24 16 16.")
parser.add_argument('-den', '--dense-nf', type=int, nargs='+', required=False, default=[64], help="Number of encoder features for the generator (only for linear or quadratic transfo). Default: 32 64 128 256.")
parser.add_argument('-e', '--epochs', type=int, required=True, help="Number of epochs.")
parser.add_argument('-b', '--batch-size', type=int, required=False, default=1, help='Batch size. Default: 1.')
# augmentation
parser.add_argument('-as', '--aug_spat_prob', type=float, required=False, default=0, help='Probability of performing spatial augmentation. Default: 0.')
parser.add_argument('-ai', '--aug_int_prob', type=float, required=False, default=0, help='Probability of performing intensity augmentation. Default: 0.')
# other
parser.add_argument('-r', '--resume', type=int, required=False, default=0, help='Resume a traning that stopped for some reason (1: yes, 0: no). Default: 0.')
parser.add_argument('-seed', '--seed', type=int, required=False, default=None, help='Seed for random. Default: None.')

args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
args.resume = bool(args.resume)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

os.makedirs(os.path.join(args.model + '_imgs'), exist_ok=True)

with open(args.model + '_args.txt', 'w') as file:
    for arg in vars(args):
        file.write("{}: {}\n".format(arg, getattr(args, arg)))
      
#%% Generators to access training (and validation) data.

is_val = args.val_data is not None

# training images
sub_dirs = sorted(glob.glob(os.path.join(args.train_data, '*', '')))
random.shuffle(sub_dirs)

gen_train = eddeep.generators.eddeep_fromDWI(subdirs=sub_dirs,
                                            k=args.kpad,
                                            spat_aug_prob = args.aug_spat_prob,
                                            aug_dire = args.ped,
                                            batch_size=args.batch_size)

n_train = len(sub_dirs)

# validation images
if not is_val:  
    gen_val = None
    sample = next(gen_train)
else:
    sub_dirs_val = sorted(glob.glob(os.path.join(args.train_data, '*', '')))
    gen_val = eddeep.generators.eddeep_fromDWI(subdirs=sub_dirs_val,                                                          
                                              k=len(args.enc_nf),
                                              spat_aug_prob = 0,
                                              batch_size=args.batch_size)
    n_val = len(sub_dirs_val)
    sample = next(gen_val)
    

dwarp.utils.develop(sample)
inshape = sample[0].shape[1:-1]
sl_sag=int(sample[0].shape[3]*0.45)
sl_axi=int(sample[0].shape[1]*0.45)

#%% Prepare and build the model

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.learning_rate)
loss_weights = [1]
losses = [dwarp.losses.wMSE(is_stacked=True).loss]
if args.transfo == 'deformable':
    losses += [voxelmorph.losses.Grad('l2').loss]
    loss_weights += [0.05]

model_path = args.model + '_best.h5'
model_last_path = args.model + '_last.h5'
loss_file= args.model + '_losses.csv'

translator = eddep.networks.pix2pix_gen.load(args.trans)
translator.trainable = False 

if args.resume:
    # load existing model
    registrator = eddeep.networks.eddy_reg.load(args.model)
    tab_loss = pandas.read_csv(loss_file, sep=',')
    if is_val:
        best_loss = np.min(tab_loss.val_loss)
    else:
        best_loss = np.min(tab_loss.loss)
    initial_epoch = tab_loss.epoch.iloc[-1]
    print('resuming training at epoch: ' + str(initial_epoch))

else:
    # build the model
    registrator = eddeep.networks.eddy_reg(volshape=inshape,
                                          ped=args.ped,
                                          nb_enc_features=args.enc_nf,
                                          nb_dec_features=args.dec_nf,
                                          transfo=args.transfo,               
                                          jacob_mod=True,
                                          nb_dense_features=args.dense_nf)
    initial_epoch = 0
    try: 
        os.remove(loss_file)
    except OSError:
        pass
    f = open(loss_file,'w')
    if args.transfo == 'deformable':
        if is_val:
            f.write('epoch,loss,img_loss,reg_loss,val_loss,val_img_loss,val_reg_loss\n') 
        else:
            f.write('epoch,loss,img_loss,reg_loss\n') 
    else:
        if is_val:
            f.write('epoch,loss,val_loss\n') 
        else:
            f.write('epoch,loss\n') 
    f.close()
    best_loss = np.Inf
        
registrator.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

tf.keras.utils.plot_model(registrator, to_file=args.model + '_plot.png', show_shapes=True, show_layer_names=True)

#%% Train the model

n_train_steps = n_train // args.batch_size
if is_val:
    n_val_steps = n_val // args.batch_size
    monitor='val_loss'
else:
    monitor='loss'

ytrans = [translator(sample[0]), translator(sample[1])]

for epoch in range(initial_epoch, args.epochs):   
    t = time.time()
    
    print('epoch: %d/%d' % (epoch+1, args.epochs), end=' |')
    v_loss_epoch = []    
    
    for _ in range(n_train_steps):
            
        b0, dw = next(gen_train)
        b0 = translator(b0)
        dw = translator(dw)
        
        if args.transfo == 'deformable':
            v_loss = registrator.train_on_batch([b0, dw], 2*[np.zeros((args.batch_size,1))])
        else:
            v_loss = registrator.train_on_batch([b0, dw], np.zeros((args.batch_size,1)))
        v_loss_epoch += [v_loss] 
    
        print('-', end='')
    v_loss_epoch = np.mean(v_loss_epoch, axis=0)
    
    if is_val:
        v_loss_epoch_val = [] 
        for _ in range(n_val_steps):

            b0, dw = next(gen_val)          
            b0 = translator(b0)
            dw = translator(dw)
           
            if args.transfo == 'deformable':
                v_loss_val = registrator.test_on_batch([b0, dw], 2*[np.zeros((args.batch_size,1))])
            else:
                v_loss_val = registrator.test_on_batch([b0, dw], np.zeros((args.batch_size,1)))
            v_loss_epoch_val += [v_loss_val] 
                
            print('.', end='')
            
        v_loss_epoch_val = np.mean(v_loss_epoch_val, axis=0)
 
        
    print('| ' + str(np.round(time.time()-t,3)) + ' s')
    f = open(loss_file,'a')
    if args.transfo == 'deformable':
        if is_val:
            print('train - all: %.3e, img %.3e, reg: %.3e | val - all: %.3e, img %.3e, reg: %.3e ' % (tuple(v_loss_epoch)+tuple(v_loss_epoch_val)))
            f.write(str(epoch+1) + ',' + str(v_loss_epoch[0]) + ',' + str(v_loss_epoch[1]) + ',' + str(v_loss_epoch[2]) + ',' + str(v_loss_epoch_val[0]) + ',' + str(v_loss_epoch_val[1]) + ',' + str(v_loss_epoch_val[2]) + '\n')
        else:
            print('train - all: %.3e, img %.3e, reg: %.3e' % tuple(v_loss_epoch))
            f.write(str(epoch+1) + ',' + str(v_loss_epoch[0]) + ',' + str(v_loss_epoch[1]) + ',' + str(v_loss_epoch[2]) + '\n')
    else:
        if is_val:
            print('train: %.3e  |  val: %.3e' % (v_loss_epoch, v_loss_epoch_val))
            f.write(str(epoch+1) + ',' + str(v_loss_epoch) + ',' + str(v_loss_epoch_val) + '\n')
        else:
            print('train: %.3e' % v_loss_epoch)
            f.write(str(epoch+1) + ',' + str(v_loss_epoch) + '\n')
        
    f.close()
    
    if is_val:
        v_loss_epoch = v_loss_epoch_val
    if args.transfo == 'deformable':
        v_loss_epoch = v_loss_epoch[0]
    if v_loss_epoch < best_loss:
        best_loss = v_loss_epoch
        registrator.save(args.model + '_best.h5')
    registrator.save(args.model + '_last.h5')
    
    yhat = registrator.predict(ytrans, verbose=0)
    if args.transfo == 'deformable':
        yhat = yhat[0]

    f, axs = plt.subplots(2,3); f.dpi = 200
    plt.subplots_adjust(wspace=0.01,hspace=-0.1)

    axs[0][0].imshow(np.fliplr(yhat[0,:,:,sl_sag,0]), vmin=0, vmax=0.5, origin="lower")
    axs[0][1].imshow(np.fliplr(yhat[0,:,:,sl_sag,1]), vmin=0, vmax=0.5, origin="lower")
    axs[0][2].imshow(np.fliplr((yhat[0,:,:,sl_sag,1]-yhat[0,:,:,sl_sag,0]))**2, vmin=0, vmax=0.005, origin="lower")
    axs[1][0].imshow(np.fliplr(yhat[0,sl_axi,:,:,0]), vmin=0, vmax=0.5, origin="lower")
    axs[1][1].imshow(np.fliplr(yhat[0,sl_axi,:,:,1]), vmin=0, vmax=0.5, origin="lower")
    axs[1][2].imshow(np.fliplr((yhat[0,sl_axi,:,:,1]-yhat[0,sl_axi,:,:,0]))**2, vmin=0, vmax=0.005, origin="lower")
    axs[0][0].axis('off'); axs[0][1].axis('off');  axs[1][0].axis('off'); axs[1][1].axis('off'); axs[0][2].axis('off'); axs[1][2].axis('off'); 
    
    plt.suptitle('epoch: ' + str(epoch+1), ha='center', y=0.89, fontsize=8)

    plt.savefig(os.path.join(args.model + '_imgs','img_' + str(epoch) + '.png'), bbox_inches='tight')        
    plt.close()

dwarp.utils.plot_losses(loss_file, is_val=is_val)



