import numpy as np
import matplotlib.pyplot as plt
 
from dipy.io.image import load_nifti
from dipy.align.imaffine import AffineMap
from dipy.align.imwarp import DiffeomorphicMap
 
#%%
a, b, c = load_nifti('/scratch1/NOT_BACKED_UP/alegouhy/data/diverse_t1/img_test/IXI069-Guys-0769-T1.nii.gz', return_voxsize=True)

moving_img_file = '/scratch0/NOT_BACKED_UP/alegouhy/dev/dwarp_public/example_data/target_t1.nii.gz'
moving_img, grid2world, moving_zooms =  load_nifti(moving_img_file, return_voxsize=True)
 
grid2world = np.array([[1, 0, 0, 10],
                       [0, 1, 0, 20],
                       [0, 0, 1, 30],
                       [0, 0, 0, 1 ]], dtype=np.float64)

world2grid = np.linalg.inv(grid2world)
volshape = moving_img.shape
 
#%% apply affine transfo
 
aff_mat = np.array([[ 1.1 , 0.01, 0.01,   5],
                    [-0.03, 0.9 , 0.01, -15],
                    [-0.02, 0.01, 0.8 ,  -5],
                    [0    , 0   , 0   ,   1]])
aff_map = AffineMap(aff_mat)
moved_img = aff_map.transform(moving_img,
                              image_grid2world=grid2world,
                              sampling_grid_shape=volshape,
                              sampling_grid2world=grid2world)
 
 
f, axs = plt.subplots(1,3); f.dpi = 200
axs[0].imshow(moving_img[128,:,:]); 
axs[0].set_title('moving')
axs[1].imshow(moved_img[128,:,:]);
axs[1].set_title('moved')
axs[2].imshow(moving_img[128,:,:]-moved_img[128,:,:]); 
axs[2].set_title('diff')
for ax in np.ravel(axs):
    ax.axis('off')
plt.show()
 
#%% apply diffeo transfo
 
diffeo_map = DiffeomorphicMap(3, volshape)
disp = np.zeros(volshape + (3,), dtype=np.float32)
disp += 15
diffeo_map.forward = disp
# moved_img = diffeo_map.transform(moving_img, 
#                                  image_world2grid=world2grid,
#                                  out_shape=volshape, 
#                                  out_grid2world=grid2world)
# grid2world = np.eye(4)
# world2grid = np.eye(4)
moved_img = diffeo_map._warp_forward(moving_img, 
                                     image_world2grid=world2grid,
                                     out_shape=volshape, 
                                     out_grid2world=grid2world)
f, axs = plt.subplots(1,3); f.dpi = 200
axs[0].imshow(moving_img[128,:,:]); 
axs[0].set_title('moving')
axs[1].imshow(moved_img[128,:,:]);
axs[1].set_title('moved')
axs[2].imshow(moving_img[128,:,:]-moved_img[128,:,:]); 
axs[2].set_title('diff')
for ax in np.ravel(axs):
    ax.axis('off')
plt.show()