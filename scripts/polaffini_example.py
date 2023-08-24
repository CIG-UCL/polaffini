import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
import polaffini.polaffini as polaffini
import SimpleITK as sitk
import time

#%% POLAFFINI matching using segmentations

moving_seg = sitk.ReadImage('example_data/moving_dkt.nii.gz')
target_seg = sitk.ReadImage('example_data/target_dkt.nii.gz')


t = time.time()
init_aff, polyAff_svf = polaffini.estimateTransfo(moving_seg, target_seg, omit_labs=[2,41])
print("POLAFFINI estimation done in " + str(round(time.time()-t,3)) + " seconds.")

# Composition to the full transformation (moving to target)
polaff = polaffini.get_full_transfo(init_aff, polyAff_svf)

# Composition to the full inverse transformation (target to moving)
polaff_inv = polaffini.get_full_transfo(init_aff, polyAff_svf, invert=True)


#%% Applying the estimated transformation to T1w images

resampler = sitk.ResampleImageFilter()
resampler.SetInterpolator(sitk.sitkLinear)
moving_img = sitk.ReadImage('example_data/moving_t1.nii.gz')
target_img = sitk.ReadImage('example_data/target_t1.nii.gz')

# moving to target
resampler.SetReferenceImage(target_img)  
resampler.SetTransform(polaff)  
moved_img = resampler.Execute(moving_img)
sitk.WriteImage(moved_img, 'example_data/moving2target_t1.nii.gz')
polaffini.write_dispField(polaff, 'example_data/moving2target_disp.nii.gz')

# target to moving
resampler.SetReferenceImage(moving_img) 
resampler.SetTransform(polaff_inv)  
moved_img = resampler.Execute(target_img)
sitk.WriteImage(moved_img, 'example_data/target2moving_t1.nii.gz')



