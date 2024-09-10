import os
import sys
maindir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(maindir)
import glob
import time
import numpy as np
import tensorflow as tf       
import dwarp
import argparse
import random
import matplotlib.pyplot as plt
import pandas
import eddeep

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))       
        
parser = argparse.ArgumentParser(description="Training script for the image translation part of Eddeep.")

# training and validation data
parser.add_argument('-t', '--train_data', type=str, required=True, help='Path to the eddy-corrected training data following nested subfolder structure below.')
parser.add_argument('-v', '--val_data', type=str, required=False, default=None, help='Path to the eddy-corrected validation data following nested subfolder structure below.')
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
parser.add_argument('-B', '--target_bval', type=int, required=True, help='Target b-value for translation (must be among the existing b-values in the data).')
parser.add_argument('-k', '--kpad', type=int, required=False, default=5, help='k to pad the input so that its shape is of form 2**k. Has to be >= number encoding steps.')
# model and its hyper-paramaters
parser.add_argument('-o', '--model', type=str, required=True, help="Path prefix to the output model (without extension).")
parser.add_argument('-lr', '--learning-rate', type=float, required=False, default=1e-4, help="Learning rate. Default: 1e-4.")
parser.add_argument('-denc', '--dis-enc-nf', type=int, nargs='+', required=False, default=[16,32,64,128], help="Number of encoder features for the discriminator. Default: 16 32 64 128.")
parser.add_argument('-genc', '--gen-enc-nf', type=int, nargs='+', required=False, default=[32,64,128,256], help="Number of encoder features for the generator. Default: 32 64 128 256.")
parser.add_argument('-gdec', '--gen-dec-nf', type=int, nargs='+', required=False, default=[256,128,64,32], help="Number of decoder features for the generator. Default: 256 128 64 32.")
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

os.makedirs(os.path.join(args.model) + '_imgs', exist_ok=True)

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
                                            target_bval = args.target_bval,
                                            get_dwmean = True,
                                            spat_aug_prob = args.aug_spat_prob,
                                            int_pair_aug_prob = args.aug_int_prob,
                                            batch_size=args.batch_size)

n_train = len(sub_dirs)

# validation images
if not is_val:  
    gen_val = None
    sample = next(gen_train)
else:
    sub_dirs_val = sorted(glob.glob(os.path.join(args.train_data, '*', '')))
    gen_val = eddeep.generators.eddeep_fromDWI(subdirs=sub_dirs_val,                                                          
                                              k=len(args.gen_enc_nf),
                                              target_bval = args.target_bval,
                                              get_dwmean = True,
                                              spat_aug_prob = 0,
                                              int_pair_aug_prob = 0,
                                              batch_size=args.batch_size)
    n_val = len(sub_dirs_val)
    sample = next(gen_val)
    

dwarp.utils.develop(sample)
inshape = sample[0].shape[1:-1]
sl_sag=int(sample[0].shape[3]*0.45)
sl_axi=int(sample[0].shape[1]*0.45)

#%% Prepare and build the model

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.learning_rate)

dis_path = args.model + '_dis_best.h5'
gen_path = args.model + '_gen_best.h5'
dis_last_path = args.model + '_dis_last.h5'
gen_last_path = args.model + '_gen_last.h5'
loss_file= args.model + '_losses.csv'

if args.resume:
    # load existing model
    discriminator = eddeep.networks.pix2pix_dis.load(dis_last_path)
    generator = eddeep.networks.pix2pix_gen.load(gen_last_path)
    tab_loss = pandas.read_csv(loss_file, sep=',')
    if is_val:
        best_img_loss = np.min(tab_loss.val_img0 + tab_loss.val_img) / 2
    else:
        best_img_loss = np.min(tab_loss.img0 + tab_loss.img) / 2
    initial_epoch = tab_loss.epoch.iloc[-1]
    print('resuming training at epoch: ' + str(initial_epoch))

else:
    # build the models
    discriminator = eddeep.networks.pix2pix_dis(inshape,
                                               nb_feats=args.dis_enc_nf)   
    generator = eddeep.networks.pix2pix_gen(volshape=inshape,
                                           nb_enc_features=args.gen_enc_nf,
                                           nb_dec_features=args.gen_dec_nf + [1],
                                           final_activation=None,
                                           name='gen')
    initial_epoch = 0
    try:
        os.remove(loss_file)
    except OSError:
        pass
    f = open(loss_file,'w')
    if is_val:
        f.write('epoch,dis0_real,dis0_fake,gen0,img0,dis_real,dis_fake,gen,img,val_dis0_real,val_dis0_fake,val_gen0,val_img0,val_dis_real,val_dis_fake,val_gen,val_img\n') 
    else:
        f.write('epoch,dis0_real,dis0_fake,gen0,img0,dis_real,dis_fake,gen,img\n')
    f.close()
    best_img_loss = np.Inf
        
discriminator.compile(loss='mse', optimizer=optimizer)
gan = eddeep.networks.gan(generator, discriminator, inshape)
gan.compile(loss=['mse','mse'], loss_weights=[1,100], optimizer=optimizer)

tf.keras.utils.plot_model(discriminator, to_file=args.model + '_dis_plot.png', show_shapes=True, show_layer_names=True)
tf.keras.utils.plot_model(generator, to_file=args.model + '_gen_plot.png', show_shapes=True, show_layer_names=True)
tf.keras.utils.plot_model(gan, to_file=args.model + '_gan_plot.png', show_shapes=True, show_layer_names=True)

#%% Train the model

n_train_steps = n_train // args.batch_size
if is_val:
    n_val_steps = n_val // args.batch_size
    monitor='val_loss'
else:
    monitor='loss'

patch_shape = discriminator.layers[-1].output_shape[1:-1]
real = tf.ones((args.batch_size, *patch_shape, 1))
fake = tf.zeros((args.batch_size, *patch_shape, 1))

for epoch in range(initial_epoch, args.epochs):
    t = time.time()

    print('epoch: %d/%d' % (epoch+1, args.epochs), end=' |')
    dis0_real_loss_epoch = []; val_dis0_real_loss_epoch = []
    dis0_fake_loss_epoch = []; val_dis0_fake_loss_epoch = []
    gen0_loss_epoch = [];      val_gen0_loss_epoch = []
    img0_loss_epoch = [];      val_img0_loss_epoch = []
    dis_real_loss_epoch = [];  val_dis_real_loss_epoch = []
    dis_fake_loss_epoch = [];  val_dis_fake_loss_epoch = []
    gen_loss_epoch = [];       val_gen_loss_epoch = []
    img_loss_epoch = [];       val_img_loss_epoch = []
    
    for _ in range(n_train_steps):
        x0, x, y = next(gen_train)
            
        yhat0 = generator.predict(x0, verbose=0)
        dis0_real_loss = discriminator.train_on_batch([x0,y], real) 
        dis0_fake_loss = discriminator.train_on_batch([x0,yhat0], fake)
        gen0_loss, _, img0_loss = gan.train_on_batch(x0, [real, y])     
        dis0_real_loss_epoch += [dis0_real_loss] 
        dis0_fake_loss_epoch += [dis0_fake_loss] 
        gen0_loss_epoch += [gen0_loss]
        img0_loss_epoch += [img0_loss]
        
        yhat = generator.predict(x, verbose=0)
        dis_real_loss = discriminator.train_on_batch([x,y], real) 
        dis_fake_loss = discriminator.train_on_batch([x,yhat], fake)
        gen_loss, _, img_loss = gan.train_on_batch(x, [real, y])     
        dis_real_loss_epoch += [dis_real_loss] 
        dis_fake_loss_epoch += [dis_fake_loss] 
        gen_loss_epoch += [gen_loss]
        img_loss_epoch += [img_loss]
        
        print('-'*args.batch_size, end='')
        
    if is_val:
        for _ in range(n_val_steps):
            x0, x, y = next(gen_val)
                
            yhat0 = generator.predict(x0, verbose=0)
            dis0_real_loss = discriminator.test_on_batch([x0,y], real) 
            dis0_fake_loss = discriminator.test_on_batch([x0,yhat0], fake)
            gen0_loss, _, img0_loss = gan.test_on_batch(x0, [real, y])     
            val_dis0_real_loss_epoch += [dis0_real_loss] 
            val_dis0_fake_loss_epoch += [dis0_fake_loss] 
            val_gen0_loss_epoch += [gen0_loss]
            val_img0_loss_epoch += [img0_loss]
            
            yhat = generator.predict(x, verbose=0)
            dis_real_loss = discriminator.test_on_batch([x,y], real) 
            dis_fake_loss = discriminator.test_on_batch([x,yhat], fake)
            gen_loss, _, img_loss = gan.test_on_batch(x, [real, y])     
            val_dis_real_loss_epoch += [dis_real_loss] 
            val_dis_fake_loss_epoch += [dis_fake_loss] 
            val_gen_loss_epoch += [gen_loss]
            val_img_loss_epoch += [img_loss]
            
            print('.'*args.batch_size, end='')
                    
    dis0_real_loss_epoch = np.mean(dis0_real_loss_epoch);
    dis0_fake_loss_epoch = np.mean(dis0_fake_loss_epoch);
    gen0_loss_epoch = np.mean(gen0_loss_epoch);
    img0_loss_epoch = np.mean(img0_loss_epoch);
    dis_real_loss_epoch = np.mean(dis_real_loss_epoch);
    dis_fake_loss_epoch = np.mean(dis_fake_loss_epoch);
    gen_loss_epoch = np.mean(gen_loss_epoch);
    img_loss_epoch = np.mean(img_loss_epoch);
    if is_val:
        val_dis0_real_loss_epoch = np.mean(val_dis0_real_loss_epoch)
        val_dis0_fake_loss_epoch = np.mean(val_dis0_fake_loss_epoch)
        val_gen0_loss_epoch = np.mean(val_gen0_loss_epoch)
        val_img0_loss_epoch = np.mean(val_img0_loss_epoch)
        val_dis_real_loss_epoch = np.mean(val_dis_real_loss_epoch)
        val_dis_fake_loss_epoch = np.mean(val_dis_fake_loss_epoch) 
        val_gen_loss_epoch = np.mean(val_gen_loss_epoch)
        val_img_loss_epoch = np.mean(val_img_loss_epoch)
        
        
    print('| ' + str(np.round(time.time()-t,3)) + ' s')
    print('  train (b=0, dw) | dis_real (%.3f, %.3f), dis_fake: (%.3f, %.3f), gen: (%.3f, %.3f), img: (%.3e, %.3e)' % (dis0_real_loss_epoch, dis_real_loss_epoch, dis0_fake_loss_epoch, dis_fake_loss_epoch, gen0_loss_epoch, gen_loss_epoch, img0_loss_epoch, img_loss_epoch))
    if is_val:
        print('  val   (b=0, dw) | dis_real (%.3f, %.3f), dis_fake: (%.3f, %.3f), gen: (%.3f, %.3f), img: (%.3e, %.3e)' % (val_dis0_real_loss_epoch, val_dis_real_loss_epoch, val_dis0_fake_loss_epoch, val_dis_fake_loss_epoch, val_gen0_loss_epoch, val_gen_loss_epoch, val_img0_loss_epoch, val_img_loss_epoch))
    f = open(loss_file,'a')
    if is_val:
        f.write(str(epoch+1) + ',' + str(dis0_real_loss_epoch)+ ',' + str(dis0_fake_loss_epoch) + ',' + str(gen0_loss_epoch) + ',' + str(img0_loss_epoch) + ',' + str(dis_real_loss_epoch)+ ',' + str(dis_fake_loss_epoch) + ',' + str(gen_loss_epoch) + ',' + str(img_loss_epoch) + ',' + str(val_dis0_real_loss_epoch)+ ',' + str(val_dis0_fake_loss_epoch) + ',' + str(val_gen0_loss_epoch) + ',' + str(val_img0_loss_epoch) + ',' + str(val_dis_real_loss_epoch)+ ',' + str(val_dis_fake_loss_epoch) + ',' + str(val_gen_loss_epoch) + ',' + str(val_img_loss_epoch) + '\n')
    else:
        f.write(str(epoch+1) + ',' + str(dis0_real_loss_epoch)+ ',' + str(dis0_fake_loss_epoch) + ',' + str(gen0_loss_epoch) + ',' + str(img0_loss_epoch) + ',' + str(dis_real_loss_epoch)+ ',' + str(dis_fake_loss_epoch) + ',' + str(gen_loss_epoch) + ',' + str(img_loss_epoch) + '\n')
    f.close()
    
    if is_val:
        val_img_loss_meanEpoch = (val_img0_loss_epoch + val_img_loss_epoch) / 2
    else:
        val_img_loss_meanEpoch = (img0_loss_epoch + img_loss_epoch) / 2
    if val_img_loss_meanEpoch < best_img_loss:
        best_img_loss = val_img_loss_meanEpoch
        generator.save(gen_path)
        discriminator.save(dis_path)
    generator.save(gen_last_path)
    discriminator.save(dis_last_path)

    y0_fake = generator.predict(sample[0], verbose=0)
    y_fake = generator.predict(sample[1], verbose=0)
    
    f, axs = plt.subplots(2,5); f.dpi = 200
    plt.subplots_adjust(wspace=0.01,hspace=-0.58)
    
    axs[0][0].imshow(np.fliplr(sample[0][0,:,:,sl_sag,0]), vmin=0, vmax=0.5, origin="lower")
    axs[0][0].axis('off'); axs[0][0].set_title('b=0', fontsize=7)
    axs[0][1].imshow(np.fliplr(y0_fake[0,:,:,sl_sag,0]), vmin=0, vmax=0.5, origin="lower")
    axs[0][1].axis('off'); axs[0][1].set_title('fake from b=0', fontsize=7)
    axs[0][2].imshow(np.fliplr(sample[2][0,:,:,sl_sag,0]), vmin=0, vmax=0.5, origin="lower")
    axs[0][2].axis('off'); axs[0][2].set_title('target', fontsize=7)
    axs[0][3].imshow(np.fliplr(y_fake[0,:,:,sl_sag,0]), vmin=0, vmax=0.5, origin="lower")
    axs[0][3].axis('off'); axs[0][3].set_title('fake from dw', fontsize=7)
    axs[0][4].imshow(np.fliplr(sample[1][0,:,:,sl_sag,0]), vmin=0, vmax=0.5, origin="lower")
    axs[0][4].axis('off'); axs[0][4].set_title('dw', fontsize=7)

    axs[1][0].imshow(np.fliplr(sample[0][0,sl_axi,:,:,0]), vmin=0, vmax=0.5, origin="lower")
    axs[1][0].axis('off'); 
    axs[1][1].imshow(np.fliplr(y0_fake[0,sl_axi,:,:,0]), vmin=0, vmax=0.5, origin="lower")
    axs[1][1].axis('off'); 
    axs[1][2].imshow(np.fliplr(sample[2][0,sl_axi,:,:,0]), vmin=0, vmax=0.5, origin="lower")
    axs[1][2].axis('off'); 
    axs[1][3].imshow(np.fliplr(y_fake[0,sl_axi,:,:,0]), vmin=0, vmax=0.5, origin="lower")
    axs[1][3].axis('off'); 
    axs[1][4].imshow(np.fliplr(sample[1][0,sl_axi,:,:,0]), vmin=0, vmax=0.5, origin="lower")
    axs[1][4].axis('off'); 
    
    plt.suptitle('epoch: ' + str(epoch+1), ha='center', y=0.8, fontsize=8)

    plt.savefig(os.path.join(args.model + '_imgs','img_' + str(epoch) + '.png'), bbox_inches='tight')
    plt.close()

dwarp.utils.plot_losses(loss_file, is_val=is_val)



