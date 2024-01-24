dwarp_public_path = '/Users/alegouhy/dev/dwarp_public'   # put here the path to dwarp_public (where you cloned the repo)

import sys
if dwarp_public_path == '':
    sys.exit('modify the path to dwrp_public in line 1')
sys.path.append(dwarp_public_path)

import os
import polaffini.polaffini as polaffini
import SimpleITK as sitk
import time

example_data_path = dwarp_public_path + os.sep + 'example_data' + os.sep

#%% POLAFFINI matching using segmentations

moving_seg = sitk.ReadImage(example_data_path + 'moving_dkt.nii.gz')
target_seg = sitk.ReadImage(example_data_path + 'target_dkt.nii.gz')

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

resampler = sitk.ResampleImageFilter()
resampler.SetInterpolator(sitk.sitkLinear)
moving_img = sitk.ReadImage(example_data_path + 'moving_t1.nii.gz')
target_img = sitk.ReadImage(example_data_path + 'target_t1.nii.gz')

# moving to target
resampler.SetReferenceImage(target_img)  
resampler.SetTransform(polaff)  
moved_img = resampler.Execute(moving_img)
sitk.WriteImage(moved_img, example_data_path + 'moving2target_t1.nii.gz')
polaffini.write_dispField(polaff, example_data_path + 'moving2target_disp.nii.gz')

# target to moving
resampler.SetReferenceImage(moving_img) 
resampler.SetTransform(polaff_inv)  
moved_img = resampler.Execute(target_img)
sitk.WriteImage(moved_img,example_data_path + 'target2moving_t1.nii.gz')



