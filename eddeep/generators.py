import numpy as np
import os
import glob
import pickle
import SimpleITK as sitk        
import polaffini.utils as utils
import dwarp.augmentation as augmentation
from dmipy.core import acquisition_scheme


def eddeep_fromDWI(subdirs,
                   k = 4,
                   ped = None,
                   bval = None,
                   dw_file = None,
                   b0_file = None,
                   target_bval = 'same',
                   sl_axi = None,
                   get_dwmean = False,
                   spat_aug_prob=0,
                   int_pair_aug_prob=0,
                   aug_dire = None,
                   batch_size=1):
    # sub
    #   |_ PED
    #        |_ bval
    
    while True:
        b0s = []
        dws = []
        dws_mean = []
        
        for b in range(batch_size):
            # random subject
            i = np.random.choice(range(0, len(subdirs)))
            sub_i = subdirs[i]
            
            # random PED
            if ped is None:
                list_ped = sorted(next(os.walk(sub_i))[1])
                ind_ped = np.random.choice(range(0, len(list_ped)))
                ped_i = list_ped[ind_ped]
            else:
                ped_i = ped
            
            # random bval
            if bval is None:
                list_bval = sorted(next(os.walk(os.path.join(sub_i, ped_i)))[1])
                ind_bval = np.random.choice(range(1, len(list_bval)))
                bval_i = list_bval[ind_bval]
            else: 
                bval_i = 'b' + str(bval)
            if target_bval == 'same':
                target_bval = bval
            # random DW and b=0 image  
            if b0_file is None:
                list_b0 = sorted(glob.glob(os.path.join(sub_i, ped_i, 'b0', '*.nii*')))
                list_b0 = [os.path.split(list_b0[j])[-1] for j in range(len(list_b0))]
                ind_b0 = np.random.choice(range(0, len(list_b0)))   
                b0_file_i = list_b0[ind_b0]
            else:
                b0_file_i = b0_file
            
            if dw_file is None:
                list_dw = sorted(glob.glob(os.path.join(sub_i, ped_i, bval_i, '*.nii*')))
                list_dw = [os.path.split(list_dw[j])[-1] for j in range(len(list_dw))]
                ind_dw = np.random.choice(range(0, len(list_dw))) 
                dw_file_i = list_dw[ind_dw]
            else:
                dw_file_i = dw_file
                
            b0 = sitk.ReadImage(os.path.join(sub_i, ped_i, 'b0', b0_file_i))
            dw = sitk.ReadImage(os.path.join(sub_i, ped_i, bval_i, dw_file_i))

            if get_dwmean:
                dw_mean = sitk.ReadImage(glob.glob(os.path.join(sub_i, ped_i,'*_b' + str(target_bval) + '_mean.nii.gz'))[0])
            
            if np.random.rand() < spat_aug_prob:
                if aug_dire is not None:
                    aug = augmentation.spatial_aug_dir(b0, dire=aug_dire)
                else:
                    aug = augmentation.spatial_aug(b0)
                aug.set_aff_params()
                aug.set_diffeo_params()  
                aug.gen_transfo()
                if get_dwmean:
                    b0, dw, dw_mean = aug.transform([b0, dw, dw_mean])
                else:
                    b0, dw = aug.transform([b0, dw])
                    
            if np.random.rand() < int_pair_aug_prob:
                stat_filter = sitk.StatisticsImageFilter()
                
                stat_filter.Execute(b0)
                b0_mu = stat_filter.GetMean()
                b0_std = stat_filter.GetSigma()
                b0 = (b0-b0_mu) / b0_std
                
                stat_filter.Execute(dw)
                dw_mu = stat_filter.GetMean()
                dw_std = stat_filter.GetSigma()
                dw = (dw-dw_mu) / dw_std
                
                alpha = float(np.exp(np.random.normal(0,0.5)))
                b0 = alpha*b0 + (1-alpha)*dw
                beta = float(np.exp(np.random.normal(0,0.5)))
                dw = beta*dw + (1-beta)*b0
                
            b0 = utils.normalize_intensities(b0, dtype=sitk.sitkFloat32)
            b0 = utils.pad_image(b0, k=k)
            b0 = sitk.GetArrayFromImage(b0)[np.newaxis,..., np.newaxis]
                        
            dw = utils.normalize_intensities(dw, dtype=sitk.sitkFloat32)
            dw = utils.pad_image(dw, k=k)
            dw = sitk.GetArrayFromImage(dw)[np.newaxis,..., np.newaxis]
            
            if get_dwmean:
                dw_mean = utils.normalize_intensities(dw_mean, dtype=sitk.sitkFloat32)
                dw_mean = utils.pad_image(dw_mean, k=k)
                dw_mean = sitk.GetArrayFromImage(dw_mean)[np.newaxis,..., np.newaxis]
            
            if sl_axi is not None:
                b0 = b0[:,sl_axi,:,:,:]
                dw = dw[:,sl_axi,:,:,:]
                dw_mean = dw_mean[:,sl_axi,:,:,:]
                
            b0s += [b0]
            dws += [dw]
            if get_dwmean:
                dws_mean += [dw_mean]
        
        if get_dwmean:
            yield [np.concatenate(b0s,axis=0), np.concatenate(dws,axis=0), np.concatenate(dws_mean,axis=0)]
        else:
            yield [np.concatenate(b0s,axis=0), np.concatenate(dws,axis=0)]
        

def eddeep_fromModel(subdirs,
                     target_bval,
                     k = 4,
                     ped = None,
                     geom_img = None,
                     bval_range = [0,5000],
                     dw_file = None,
                     b0_file = None,
                     sl_axi = None,
                     get_dwmean = False,
                     spat_aug_prob=0,
                     int_aug_prob=0,
                     aug_dire = None,
                     batch_size=1):
    # sub
    #   |_ PED
    #        |_ model_fit.pkl
    #        |_ dw_mean.nii.gz
    
    if geom_img is not None:
        geom_img = sitk.ReadImage(geom_img)
        
    while True:
        b0s = []
        dws = []
        dws_mean = []
        
        for b in range(batch_size):
            do_int_aug = np.random.rand() < int_aug_prob
            do_spat_aug = np.random.rand() < spat_aug_prob
            
            # random subject
            i = np.random.choice(range(0, len(subdirs)))
            sub_i = subdirs[i]
            
            # random PED
            if ped is None:
                list_ped = sorted(next(os.walk(sub_i))[1])
                ind_ped = np.random.choice(range(0, len(list_ped)))
                ped_i = list_ped[ind_ped]
            else:
                ped_i = ped
            
            # load fitted model
            list_fitted = sorted(glob.glob(os.path.join(sub_i, ped_i, '*.pkl')))
            with open(list_fitted[0], 'rb') as pickle_file:
                fitted_model = pickle.load(pickle_file)
            
            # random bval
            bval = (bval_range[1]-bval_range[0])*np.random.rand() + bval_range[0]
            bval = np.reshape(bval, 1)
            
            # random bvec
            theta = 2*np.pi*np.random.rand()
            phi = np.arccos(2*np.random.rand()-1)
            bvec = [np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)               ]
            bvec = np.reshape(bvec, (1,3))
            
            scheme_dw = acquisition_scheme.acquisition_scheme_from_bvalues(bval * 1e6, bvec) 

            b0 = fitted_model.S0
            b0 = sitk.GetImageFromArray(b0)
            if do_int_aug:     
                if geom_img is not None:
                    b0.CopyInformation(geom_img)
                aug_int = augmentation.intensity_aug(n_cpts=20)
                b0 = aug_int.transform(b0)          
            fitted_model.S0 = sitk.GetArrayFromImage(b0)
            
            dw = fitted_model.predict(scheme_dw)[...,0]
            dw = sitk.GetImageFromArray(dw)
            if geom_img is not None:
                dw.CopyInformation(geom_img)
                
            if get_dwmean:
                dw_mean = sitk.ReadImage(glob.glob(os.path.join(sub_i, ped_i,'*_b' + str(target_bval) + '_mean.nii.gz'))[0])
                if geom_img is not None:
                    dw_mean.CopyInformation(geom_img)
                else:
                    dw_mean = sitk.GetImageFromArray(sitk.GetArrayFromImage(dw_mean))
                    
            if do_spat_aug:
                if aug_dire is not None:
                    aug_spat = augmentation.spatial_aug_dir(b0, dire=aug_dire)
                else:
                    aug_spat = augmentation.spatial_aug(b0)
                aug_spat.set_aff_params()
                aug_spat.set_diffeo_params()  
                aug_spat.gen_transfo()
                if get_dwmean:
                    b0, dw, dw_mean = aug_spat.transform([b0, dw, dw_mean])
                else:
                    b0, dw = aug_spat.transform([b0, dw])            
            
            if do_int_aug:     
                aug_int = augmentation.intensity_aug(n_cpts=0, distrib_noise='rician')
                b0 = aug_int.transform(b0)
                dw = aug_int.transform(dw)
                
                
            b0 = utils.normalize_intensities(b0, dtype=sitk.sitkFloat32)
            if k is not None: b0 = utils.pad_image(b0, k=k)  
            b0 = sitk.GetArrayFromImage(b0)[np.newaxis,..., np.newaxis]
                        
            dw = utils.normalize_intensities(dw, dtype=sitk.sitkFloat32)
            if k is not None: dw = utils.pad_image(dw, k=k)
            dw = sitk.GetArrayFromImage(dw)[np.newaxis,..., np.newaxis]
            
            if get_dwmean:
                dw_mean = utils.normalize_intensities(dw_mean, dtype=sitk.sitkFloat32)
                if k is not None: dw_mean = utils.pad_image(dw_mean, k=k)
                dw_mean = sitk.GetArrayFromImage(dw_mean)[np.newaxis,..., np.newaxis]
            
            if sl_axi is not None:
                b0 = b0[:,sl_axi,:,:,:]
                dw = dw[:,sl_axi,:,:,:]
                dw_mean = dw_mean[:,sl_axi,:,:,:]
                
            b0s += [b0]
            dws += [dw]
            if get_dwmean:
                dws_mean += [dw_mean]
        
        if get_dwmean:
            yield [np.concatenate(b0s,axis=0), np.concatenate(dws,axis=0), np.concatenate(dws_mean,axis=0)]
        else:
            yield [np.concatenate(b0s,axis=0), np.concatenate(dws,axis=0)]
        

