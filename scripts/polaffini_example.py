dwarp_public_path = ''   # put here the path to dwarp_public (where you cloned the repo)

import sys
if dwarp_public_path == '':
    sys.exit('modify the path to dwarp_public in line 1')
import os
os.chdir(dwarp_public_path)

import utils
import polaffini.polaffini as polaffini
import SimpleITK as sitk
import time
import matplotlib.pyplot as plt

#%% POLAFFINI matching using segmentations

# Loading the segmentations
moving_seg = utils.imageIO('example_data' + os.sep + 'moving_dkt.nii.gz').read()
target_seg = utils.imageIO('example_data' + os.sep + 'target_dkt.nii.gz').read()

# Peforming POLAFFINI estimation
t = time.time()
init_aff, polyAff_svf = polaffini.estimateTransfo(moving_seg, 
                                                  target_seg, 
                                                  omit_labs=[2,41])
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

moving_img = utils.imageIO('example_data' + os.sep + 'moving_t1.nii.gz').read()
target_img = utils.imageIO('example_data' + os.sep + 'target_t1.nii.gz').read()
axs[0].imshow(sitk.GetArrayFromImage(target_img)[:,:,sl_sag]
            - sitk.GetArrayFromImage(sitk.Resample(moving_img, target_img))[:,:,sl_sag],
              origin='lower', cmap=plt.cm.twilight, vmin=-2500,vmax=2500)


resampler = sitk.ResampleImageFilter()
resampler.SetInterpolator(sitk.sitkLinear)

# moving to target
resampler.SetReferenceImage(target_img)  
resampler.SetTransform(polaff)  
moved_img = resampler.Execute(moving_img)
utils.imageIO('example_data' + os.sep + 'moving2target_t1.nii.gz').write(moved_img)
polaffini.write_dispField(polaff, 'example_data' + os.sep + 'moving2target_disp.nii.gz')
axs[1].imshow(sitk.GetArrayFromImage(target_img)[:,:,sl_sag]
            - sitk.GetArrayFromImage(moved_img)[:,:,sl_sag],
              origin='lower', cmap=plt.cm.twilight, vmin=-2500,vmax=2500)

# target to moving
resampler.SetReferenceImage(moving_img) 
resampler.SetTransform(polaff_inv)  
moved_img = resampler.Execute(target_img)
utils.imageIO('example_data' + os.sep + 'target2moving_t1.nii.gz').write(moved_img)

plt.show()