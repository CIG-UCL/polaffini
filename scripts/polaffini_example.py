dwarp_public_path = ''   # put here the path to dwarp_public (where you cloned the repo)

import sys
if dwarp_public_path == '':
    sys.exit('modify the path to where you clone the repo in line 1')
sys.path.append(dwarp_public_path) 
import os
import polaffini.utils as utils
import polaffini.polaffini as polaffini
import SimpleITK as sitk
import time
import matplotlib.pyplot as plt

datadir = os.path.join(dwarp_public_path, 'example_data')

#%% POLAFFINI matching using segmentations

# Loading the segmentations
moving_seg_file = os.path.join(datadir, 'moving_dkt.nii.gz')
target_seg_file = os.path.join(datadir, 'target_dkt.nii.gz')
moving_seg = utils.imageIO(moving_seg_file).read()
target_seg = utils.imageIO(target_seg_file).read()

# Peforming POLAFFINI estimation
t = time.time()
init_aff, polyAff_svf = polaffini.estimateTransfo(moving_seg,
                                                  target_seg,
                                                  omit_labs=[24,2,41])
print("POLAFFINI estimation done in " + str(round(time.time()-t,3)) + " seconds.")

# Composition to the full transformation (moving to target)
polaff = polaffini.get_full_transfo(init_aff, polyAff_svf)

# Composition to the full inverse transformation (target to moving)
polaff_inv = polaffini.get_full_transfo(init_aff, polyAff_svf, invert=True)


#%% Applying the estimated transformation to T1w images

f, axs = plt.subplots(1,2); f.dpi = 200
axs[0].axis('off'); axs[1].axis('off');
sl_sag = 85
axs[0].set_title('before'); axs[1].set_title('after')
plt.suptitle('target - moving', y=0.87)

moving_img_file = os.path.join(datadir, 'moving_t1.nii.gz')
target_img_file = os.path.join(datadir, 'target_t1.nii.gz')
moving_img = utils.imageIO(moving_img_file).read()
target_img = utils.imageIO(target_img_file).read()

axs[0].imshow(sitk.GetArrayFromImage(target_img)[:,:,sl_sag]
            - sitk.GetArrayFromImage(sitk.Resample(moving_img, target_img))[:,:,sl_sag],
              origin='lower', cmap=plt.cm.twilight, vmin=-2500,vmax=2500)


resampler = sitk.ResampleImageFilter()
resampler.SetInterpolator(sitk.sitkLinear)

# moving to target
resampler.SetReferenceImage(target_img)  
resampler.SetTransform(polaff)  
moved_img = resampler.Execute(moving_img)

moved_img_file = os.path.join(datadir, 'moving2target_t1.nii.gz')
utils.imageIO(moved_img_file).write(moved_img)

disp_file = os.path.join(datadir, 'moving2target_disp.nii.gz')
# polaffini.write_dispField(polaff, disp_file)

axs[1].imshow(sitk.GetArrayFromImage(target_img)[:,:,sl_sag]
            - sitk.GetArrayFromImage(moved_img)[:,:,sl_sag],
              origin='lower', cmap=plt.cm.twilight, vmin=-2500,vmax=2500)

# target to moving
resampler.SetReferenceImage(moving_img) 
resampler.SetTransform(polaff_inv)  
moved_img = resampler.Execute(target_img)

moved_img_file = os.path.join(datadir, 'target2moving_t1.nii.gz')
utils.imageIO(moved_img_file).write(moved_img)

plt.show()
