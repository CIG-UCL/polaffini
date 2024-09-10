import os
import sys
maindir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.append(maindir)
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import scipy
import matplotlib.pyplot as plt
import polaffini.utils as utils
import time



class intensity_aug:
    
    def __init__(self, n_cpts=20, fix0=True,
                 std_defo_add=None, std_defo_mult=None, std_defo_smo=None,
                 distrib_noise=None, max_snr=50):
        
        self.n_cpts = n_cpts
        self.std_defo_add = std_defo_add
        self.std_defo_mult = std_defo_mult
        self.std_defo_smo = std_defo_smo
        self.fix0 = fix0
        self.distrib_noise = distrib_noise
        self.max_snr = max_snr
        
    def transform(self, x_img):
        
        x = sitk.GetArrayFromImage(x_img)
        
        self.compute_defo_parameters(x)
        self.compute_defo(x)
        self.compute_noise(x)
                
        defo_add_img = sitk.GetImageFromArray(self.defo_add)
        defo_add_img.CopyInformation(x_img)
        defo_add_img = sitk.Cast(defo_add_img, x_img.GetPixelID())
        
        defo_mult_img = sitk.GetImageFromArray(self.defo_mult)
        defo_mult_img.CopyInformation(x_img)
        defo_mult_img = sitk.Cast(defo_mult_img, x_img.GetPixelID())

        noise_img = sitk.GetImageFromArray(self.noise)
        noise_img.CopyInformation(x_img)
        noise_img = sitk.Cast(noise_img, x_img.GetPixelID())
        
        return x_img * defo_mult_img + defo_add_img + noise_img
    
    
    def compute_defo_parameters(self, x):
        
        if self.n_cpts > 0:
            self.x_shape = x.shape
            self.bounds = [np.amin(x), np.amax(x)]
            
            if self.std_defo_add is None:
                self.std_defo_add = np.std(np.ravel(x))
                
            if self.std_defo_mult is None:
                self.std_defo_mult = 0.25
                
            if self.std_defo_smo is None:
                self.std_defo_smo = (self.bounds[1]-self.bounds[0]) / self.n_cpts 
    
            self.cpts = np.squeeze(scipy.stats.qmc.Sobol(1).random(self.n_cpts))
            self.cpts = (self.bounds[1]-self.bounds[0]) * self.cpts + self.bounds[0]
            
            self.loc_defo_add = self.std_defo_add*np.random.randn(self.n_cpts) 
            self.loc_defo_mult = np.exp(np.log(1+self.std_defo_mult)*np.random.randn(self.n_cpts))

        
    def compute_defo(self, x):  
        
        if self.n_cpts > 0:
            defo_add = np.zeros(self.x_shape)
            defo_mult = np.zeros(self.x_shape)
            weight_sum = np.zeros_like(x) + 1e-15
            for c in range(self.n_cpts):
                weight = np.exp(-(x - self.cpts[c])**2 / (2*self.std_defo_smo**2))
                defo_add += weight * self.loc_defo_add[c]
                defo_mult += weight * self.loc_defo_mult[c]
                weight_sum += weight
            defo_add /= weight_sum
            defo_mult /= weight_sum
    
            if self.fix0:
                loc_defo_add = -defo_add[x == 0][0]
                loc_defo_mult = 1/defo_mult[x == 0][0] - 1
                weight = np.exp(-x**2 / (2*self.std_defo_smo**2))
                defo_add += weight * loc_defo_add
                defo_mult *= 1 + weight * loc_defo_mult 
                
            self.defo_add = defo_add
            self.defo_mult = defo_mult
            
        else:
            self.defo_add = np.zeros_like(x)
            self.defo_mult = np.ones_like(x)

    
    def compute_noise(self, x):
        
        snr = self.max_snr * np.random.rand()
        sample_signal = np.quantile(x.ravel(), 0.8)
        sigma = sample_signal / snr
        
        if self.distrib_noise in ('rician', 'gaussian'):
            noise = np.random.normal(0, sigma, x.shape)
        
            if self.distrib_noise == 'rician':
                noise_imag = np.random.normal(0, sigma, x.shape)
                x_real_noisy = x + noise
                x_imag_noisy = noise_imag
                noise = np.sqrt(x_real_noisy**2 + x_imag_noisy**2) - x
        
        else:
            noise = np.zeros_like(x)
        
        self.noise = noise
        
        
    
class spatial_aug:
    
    def __init__(self, img_geom, dire=None):
        self.img_geom = img_geom
        self.ndims = img_geom.GetDimension()
        self.spacing = img_geom.GetSpacing()
        self.size = img_geom.GetSize()
        center_vox = [int(float(self.size[k]) / 2) for k in range(self.ndims)]
        self.center = img_geom.TransformIndexToPhysicalPoint(center_vox)
        self.dire = dire
        self.aff_params = None
        self.diffeo_params = None
        self.polyaff_params = None
        self.transfo = None
        self.volshape = np.flip(self.size)
        
    def transform(self, img_list):
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.img_geom)
        resampler.SetTransform(self.transfo)  
        img_aug_list = []
        for img in img_list:
            img_aug_list += [resampler.Execute(img)]
        
        return img_aug_list
    
    
    def gen_transfo(self):
        
        transfo = sitk.CompositeTransform(self.ndims)
        if self.aff_params is not None:
            affine_transfo = self.aff_transfo(self.aff_params)
            transfo.AddTransform(affine_transfo)
        if self.diffeo_params is not None:
            diffeo_transfo = self.diffeo_transfo(self.diffeo_params)
            transfo.AddTransform(diffeo_transfo)
        if self.polyaff_params is not None:
            if self.polyaff_params['gpu']:
                polyaff_transfo = self.polyaff_transfo_gpu(self.polyaff_params)
            else:
                polyaff_transfo = self.polyaff_transfo(self.polyaff_params)
            transfo.AddTransform(polyaff_transfo)
        self.transfo = transfo
    
    
    def get_transfo(self):
        return self.transfo
    
    def set_transfo(self, transfo):
        self.transfo = transfo
    
        
    def set_aff_params(self, trans_bounds=5, rot_bounds=np.pi/8, scalDir_bounds=np.pi/12, scal_bounds=1.3):
        self.aff_params = {'trans_bounds': trans_bounds,
                           'rot_bounds': rot_bounds,
                           'scalDir_bounds': scalDir_bounds,
                           'scal_bounds': scal_bounds}

    def set_polyaff_params(self, nb_points=1000, sigma=15, trans_bounds=5, rot_bounds=np.pi/2, scalDir_bounds=np.pi/4, scal_bounds=5, int_steps=6, gpu=False, disp=False):
        self.polyaff_params = {'nb_points': nb_points,
                               'sigma': sigma,
                               'trans_bounds': trans_bounds,
                               'rot_bounds': rot_bounds,
                               'scalDir_bounds': scalDir_bounds,
                               'scal_bounds': scal_bounds,
                               'int_steps': int_steps,
                               'gpu': gpu,
                               'disp': False}
        
    def set_diffeo_params(self, shrink_factor=8, smooth_factor=6, svf_std_max=8, int_steps=6):
        self.diffeo_params = {'shrink_factor': shrink_factor,
                              'smooth_factor': smooth_factor,
                              'svf_std_max': svf_std_max,
                              'int_steps': int_steps}
        
    def aff_transfo(self, aff_params, center=None):
        
        trans_bounds = aff_params['trans_bounds']
        rot_bounds = aff_params['rot_bounds']
        scalDir_bounds = aff_params['scalDir_bounds']
        scal_bounds = aff_params['scal_bounds']
        if center is None:
            center = np.array(self.center)
            
        affine_transfo = sitk.AffineTransform(self.ndims)

        if self.ndims == 2:
            rot_angle = 2*rot_bounds*(np.random.rand()-0.5)
            rot = [[ 0        , rot_angle],
                   [-rot_angle, 0        ]]        

            scalDir_angle = 2*scalDir_bounds*(np.random.rand()-0.5)
            scalDir = [[ 0            , scalDir_angle],
                       [-scalDir_angle, 0            ]]
            
        elif self.ndims == 3:
            rot_angles = 2*rot_bounds*(np.random.rand(3)-0.5)
            rot = [[0, -rot_angles[2], rot_angles[1]],
                   [rot_angles[2], 0, -rot_angles[0]],
                   [-rot_angles[1], rot_angles[0], 0]]
        
            scalDir_angles = 2*scalDir_bounds*(np.random.rand(3)-0.5)
            scalDir = [[0, -scalDir_angles[2], scalDir_angles[1]],
                       [scalDir_angles[2], 0, -scalDir_angles[0]],
                       [-scalDir_angles[1], scalDir_angles[0], 0]]
        
        rot = scipy.linalg.expm(rot)
        scalDir = scipy.linalg.expm(scalDir)
        
        scal_factors = np.exp(2*np.log(scal_bounds)*(np.random.rand(self.ndims)-0.5))
        scal = np.diag(scal_factors)
        
        affine = np.matmul(rot,np.matmul(scalDir,np.matmul(scal,np.transpose(scalDir))))
        trans = 2*trans_bounds*(np.random.rand(self.ndims)-0.5)
        
        affine_transfo.SetMatrix(np.ravel(affine)) 
        affine_transfo.SetTranslation(np.matmul(affine, -center) + trans + center)
        
        return affine_transfo
    

    def aff_transfo_gpu(self, aff_params, centers=None, n=1):
        
        trans_bounds = aff_params['trans_bounds']
        rot_bounds = aff_params['rot_bounds']
        scalDir_bounds = aff_params['scalDir_bounds']
        scal_bounds = aff_params['scal_bounds']
        if centers is None:
            centers = tf.tile(np.reshape(self.center, (1,self.ndims)), [n,1])
        else:
            centers = tf.transpose(centers)
        centers =  tf.expand_dims(centers, axis=-1)
        
        if self.ndims == 2:
            rot_angles = 2*rot_bounds*(tf.random.uniform([n])-0.5)
            rot = tf.stack([tf.stack([ tf.zeros(n), rot_angles ], axis=1),
                            tf.stack([-rot_angles , tf.zeros(n)], axis=1)], axis=2)   

            scalDir_angles = 2*scalDir_bounds*(tf.random.uniform([n])-0.5)
            scalDir = tf.stack([tf.stack([ tf.zeros(n)   , scalDir_angles], axis=1),
                                tf.stack([-scalDir_angles, tf.zeros(n)   ], axis=1)], axis=2)   
            
        elif self.ndims == 3:

            rot_angles = 2*rot_bounds*(tf.random.uniform([n,3])-0.5) 
            rot = tf.stack([tf.stack([ tf.zeros(n)      ,  rot_angles[...,2], -rot_angles[...,1]], axis=1),
                            tf.stack([-rot_angles[...,2],  tf.zeros(n)      ,  rot_angles[...,0]], axis=1),
                            tf.stack([ rot_angles[...,1], -rot_angles[...,0],  tf.zeros(n)      ], axis=1)], axis=2)
        
            scalDir_angles = 2*scalDir_bounds*(tf.random.uniform([n,3])-0.5) 
            scalDir = tf.stack([tf.stack([ tf.zeros(n)          ,  scalDir_angles[...,2], -scalDir_angles[...,1]], axis=1),
                                tf.stack([-scalDir_angles[...,2],  tf.zeros(n)          ,  scalDir_angles[...,0]], axis=1),
                                tf.stack([ scalDir_angles[...,1], -scalDir_angles[...,0],  tf.zeros(n)          ], axis=1)], axis=2)
        
        rot = tf.linalg.expm(rot)
        scalDir = tf.linalg.expm(scalDir)
        
        scal_factors = np.exp(2*np.log(scal_bounds)*(tf.random.uniform((n,self.ndims))-0.5))
        scal = tf.linalg.diag(scal_factors)
        
        affine = tf.matmul(rot,tf.matmul(scalDir,tf.matmul(scal,tf.transpose(scalDir, [0,2,1]))))
        trans = 2*trans_bounds*(tf.random.uniform([n,self.ndims,1])-0.5)
        trans = tf.matmul(affine, -centers) + trans + centers
        
        affine = tf.concat((affine, trans), axis=2)
        
        affine = tf.concat((affine, 
                            tf.concat((tf.zeros((n,1,3)), tf.ones((n,1,1))), axis=2)), axis=1)
        
        return affine
    
    
    def polyaff_transfo(self, polyaff_params):
        
        nb_points = polyaff_params['nb_points']
        sigma = polyaff_params['sigma']
        int_steps = polyaff_params['int_steps']
        
        theta = (sigma**2 + 4*(sigma/2)**2)**(1/2)/2 - sigma/2
        k = (sigma*(sigma + (sigma**2 + 4*(sigma/2)**2)**(1/2)))/(2*(sigma/2)**2) + 1

        id2 = sitk.AffineTransform(self.ndims)
        id2.SetMatrix(2*np.eye(self.ndims).ravel())
        trsf2disp = sitk.TransformToDisplacementFieldFilter()
        trsf2disp.SetReferenceImage(self.img_geom)
        polyaff_svf = sitk.Image(self.size, sitk.sitkVectorFloat64)
        polyaff_svf.CopyInformation(self.img_geom)
        weight_map_sum = sitk.Image(self.size, sitk.sitkFloat64)
        weight_map_sum.CopyInformation(self.img_geom) 
        loc_transfo = sitk.AffineTransform(self.ndims)
        
        upper_vox = np.array(self.size)
        lower_vox = np.array([0]*self.ndims)
        center_vox = (upper_vox + lower_vox) / 2 
        upper_vox = 1.1 * (upper_vox - center_vox) + center_vox
        lower_vox = 1.1 * (lower_vox - center_vox) + center_vox
        
        r = scipy.stats.qmc.Sobol(self.ndims).random(nb_points) 
        
        for i in range(nb_points):
            # print(i)
            
            # Get (pseudo) random control point 
            point_vox = r[i,:]*upper_vox + (1-r[i,:])*lower_vox 
            point = np.array(self.img_geom.TransformContinuousIndexToPhysicalPoint(point_vox))
            
            # Get random affine transfo
            affine_transfo = self.aff_transfo(polyaff_params, center=point)  
            loc_mat = np.concatenate((np.reshape(affine_transfo.GetMatrix(), [self.ndims]*2),
                                      np.reshape(affine_transfo.GetTranslation(), [self.ndims,1])), axis=1)
            loc_mat = np.concatenate((loc_mat, [[0]*self.ndims + [1]]), axis=0)
            loc_mat = scipy.linalg.logm(loc_mat)
            if not np.isrealobj(loc_mat):
                continue
            
            # Compute weight map
            sigma_i = np.random.gamma(k, theta)
            id2.SetTranslation(-point) 
            weight_map = trsf2disp.Execute(id2)    
            weight_map = sitk.VectorMagnitude(weight_map)**2          
            weight_map = sitk.Exp(-weight_map/(2*sigma_i**2)) 
            
            # Update polyaffine with current field
            loc_transfo.SetMatrix((loc_mat[0:self.ndims, 0:self.ndims] + np.eye(self.ndims)).ravel())
            loc_transfo.SetTranslation(loc_mat[0:self.ndims, self.ndims])
            loc_svf = trsf2disp.Execute(loc_transfo)
            polyaff_svf += sitk.Compose([sitk.VectorIndexSelectionCast(loc_svf,d)*weight_map for d in range(self.ndims)])
            weight_map_sum += weight_map

        polyaff_svf = sitk.Compose([sitk.VectorIndexSelectionCast(polyaff_svf, d)/weight_map_sum for d in range(self.ndims)])
        polyaff = utils.integrate_svf(polyaff_svf, int_steps=int_steps)
        
        ####
        weight_map_sum = sitk.GetArrayFromImage(weight_map_sum)
        img = sitk.GetArrayFromImage(self.img_geom)
        if len(img.shape) == 3:
            plt.subplot(1,3,1)
            plt.imshow(img[45,:,:])
            plt.axis('off')
            plt.imshow(weight_map_sum[45,:,:], alpha=0.7)
            plt.subplot(1,3,2)
            plt.imshow(img[:,45,:])
            plt.axis('off')
            plt.imshow(weight_map_sum[:,45,:], alpha=0.7)
            plt.subplot(1,3,3)
            plt.imshow(img[:,:,35])
            plt.axis('off')
            plt.imshow(weight_map_sum[:,:,35], alpha=0.7)
        else:
            plt.imshow(weight_map_sum)
        plt.show()
        #####
           
        return polyaff
    
 
    def polyaff_transfo_gpu(self, polyaff_params):
        t = time.time()
        nb_points = polyaff_params['nb_points']
        sigma = polyaff_params['sigma']
        int_steps = polyaff_params['int_steps']
        disp =  polyaff_params['disp']
        
        theta = (sigma**2 + 4*(sigma/2)**2)**(1/2)/2 - sigma/2
        k = (sigma*(sigma + (sigma**2 + 4*(sigma/2)**2)**(1/2)))/(2*(sigma/2)**2) + 1

        polyaff_svf = tf.zeros([self.ndims, np.prod(self.volshape)])
        weight_map_sum = tf.zeros(np.prod(self.volshape))
 
        matO = tf.cast(utils.get_matOrientation(self.img_geom), tf.float32)
        grid = tf.meshgrid(*[tf.range(sz) for sz in self.volshape], indexing='ij')
        grid = tf.cast(grid, dtype=tf.float32) 
        grid = tf.stack([tf.reshape(grid[d], -1) for d in range(self.ndims)], axis=0)       
        grid = tf.concat((grid, tf.ones((1,tf.reduce_prod(self.volshape)), dtype=tf.float32)), axis=0)
        grid = tf.matmul(matO, grid)
  
        upper_vox = tf.constant(self.volshape, dtype=tf.float32)
        lower_vox = tf.zeros(self.ndims, dtype=tf.float32)
        center_vox = (upper_vox + lower_vox) / 2 
        upper_vox = tf.reshape(1.1 * (upper_vox - center_vox) + center_vox, (self.ndims,1))
        lower_vox = tf.reshape(1.1 * (lower_vox - center_vox) + center_vox, (self.ndims,1))
        
        # Get (pseudo) random control point  
        r = scipy.stats.qmc.Sobol(self.ndims).random(nb_points).T
        r = tf.constant(r, dtype=tf.float32)       
        points = r*upper_vox + (1-r)*lower_vox
        points = tf.concat((points, tf.ones((1, nb_points))), axis=0)
        points = tf.matmul(matO, points)[:self.ndims,:] 

        # Get random affine transfo
        
        aff_mats = self.aff_transfo_gpu(polyaff_params, centers=points, n=nb_points)     
        aff_mats = tf.cast(aff_mats, dtype=tf.complex64)
        aff_mats = tf.math.real(tf.linalg.logm(aff_mats))

        for i in range(nb_points):
            loc_mat = aff_mats[i, ...]
            point = tf.reshape(points[:,i], (self.ndims,1))
            
            # Compute weight map
            sigma_i = np.random.gamma(k, theta)  # sigma_i = tf.random.gamma([1], k, theta)
            weight_map = tf.reduce_sum((grid[:self.ndims,:] - point)**2, axis=0)
            weight_map = tf.exp(-weight_map/(2*sigma_i**2))            
            
            # Update polyaffine with current field
            polyaff_svf += tf.matmul(loc_mat, grid)[:self.ndims,:] * weight_map        
            weight_map_sum += weight_map

        weight_map_sum += 1e-5*sigma*tf.sqrt(2*np.pi)
        polyaff_svf /= weight_map_sum
        
        weight_map_sum = tf.reshape(weight_map_sum, self.volshape)
        polyaff_svf = np.stack([np.reshape(polyaff_svf[d, ...], self.volshape)
                                for d in range(self.ndims)], axis=-1)
        
        polyaff = sitk.GetImageFromArray(polyaff_svf)
        polyaff.CopyInformation(self.img_geom)
        polyaff = utils.integrate_svf(polyaff, int_steps=int_steps)

        if disp:
            img = sitk.GetArrayFromImage(self.img_geom)
            if len(img.shape) == 3:
                plt.subplot(1,3,1)
                plt.imshow(img[45,:,:])
                plt.axis('off')
                plt.imshow(weight_map_sum[45,:,:], alpha=0.7)
                plt.subplot(1,3,2)
                plt.imshow(img[:,45,:])
                plt.axis('off')
                plt.imshow(weight_map_sum[:,45,:], alpha=0.7)
                plt.subplot(1,3,3)
                plt.imshow(img[:,:,35])
                plt.axis('off')
                plt.imshow(weight_map_sum[:,:,35], alpha=0.7)
            else:
                plt.imshow(weight_map_sum)
            plt.show()

 
        return polyaff
    
    
    def diffeo_transfo(self, diffeo_params):

        shrink_factor = diffeo_params['shrink_factor']
        smooth_factor = diffeo_params['smooth_factor']
        svf_std_max = diffeo_params['svf_std_max']
        int_steps = diffeo_params['int_steps']
        
        shrink_img = sitk.Shrink(self.img_geom, [shrink_factor]*self.ndims)
        shrinked_shape = list(shrink_img.GetSize()[::-1])            
        svf_std = 2*svf_std_max*(np.random.rand(1)-0.5)
        svf = svf_std * np.random.normal(size = shrinked_shape + [self.ndims])
        svf_img = sitk.GetImageFromArray(svf, isVector=True)
        svf_img.CopyInformation(shrink_img)
        
        shrink_img = sitk.Shrink(self.img_geom, [2]*self.ndims)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(shrink_img)
        resampler.SetUseNearestNeighborExtrapolator(True)
        svf_img = resampler.Execute(svf_img)
        
        gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian.SetSigma(smooth_factor)
        svf_img = gaussian.Execute(svf_img)
        svf_img = sitk.Cast(svf_img, sitk.sitkVectorFloat64)
        
        transfo = utils.integrate_svf(svf_img, int_steps=int_steps)
        
        return sitk.DisplacementFieldTransform(transfo)
        


class spatial_aug_dir:
    """
    Keep lines parallel to dire that way after transfo.
    """
    def __init__(self, img_geom, dire=1):
        img_geom.SetDirection([1,0,0,0,1,0,0,0,1])
        img_geom.SetOrigin([0,0,0])
        img_geom.SetSpacing([2,2,2])
        self.img_geom = img_geom
        self.ndims = img_geom.GetDimension()
        self.spacing = img_geom.GetSpacing()
        self.size = img_geom.GetSize()
        center_vox = [int(float(self.size[k]) / 2) for k in range(self.ndims)]
        self.center = img_geom.TransformIndexToPhysicalPoint(center_vox)
        self.dire = dire
        self.aff_params = None
        self.diffeo_params = None
        
    def transform(self, img_list):
        affine_transfo = self.aff_transfo()
        diffeo_transfo = self.diffeo_transfo()
        
        transfo = sitk.CompositeTransform(affine_transfo)
        if diffeo_transfo is not None:
            transfo.AddTransform(diffeo_transfo)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.img_geom)
        resampler.SetTransform(transfo)
        img_aug_list = []
        for img in img_list:
            img.SetDirection([1,0,0,0,1,0,0,0,1])
            img.SetOrigin([0,0,0])
            img.SetSpacing([2,2,2])
            img_aug_list += [resampler.Execute(img)]
        
        return img_aug_list
    

    def set_aff_params(self, trans_bounds=5, rot_bounds=np.pi/8, shear_bounds=0.3, scal_bounds=1.3):
        self.aff_params = {'trans_bounds': trans_bounds,
                           'rot_bounds': rot_bounds,
                           'shear_bounds': shear_bounds,
                           'scal_bounds': scal_bounds}
        
    def set_diffeo_params(self, shrink_factor=8, smooth_factor=6, svf_std_max=8, int_steps=4):
        self.diffeo_params = {'shrink_factor': shrink_factor,
                              'smooth_factor': smooth_factor,
                              'svf_std_max': svf_std_max,
                              'int_steps': int_steps}
        
    def aff_transfo(self):
        
        affine_transfo = sitk.AffineTransform(self.ndims) 
        affine_transfo.SetCenter(self.center) 
        
        if self.aff_params is not None:
            trans_bounds = self.aff_params['trans_bounds']
            rot_bounds = self.aff_params['rot_bounds']
            shear_bounds = self.aff_params['shear_bounds']
            scal_bounds = self.aff_params['scal_bounds']
            
            trans = 2*trans_bounds*(np.random.rand(3)-0.5)
            affine_transfo.SetTranslation(trans)
            
            rot_angles = np.zeros(3)
            rot_angles[self.dire] = 2*rot_bounds*(np.random.rand()-0.5)
            rot = [[0, -rot_angles[2], rot_angles[1]],
                   [rot_angles[2], 0, -rot_angles[0]],
                   [-rot_angles[1], rot_angles[0], 0]]
            rot = scipy.linalg.expm(rot)
            
            shear = np.eye(3)
            shear_factors = 2*shear_bounds*(np.random.rand(2)-0.5)
            if self.dire == 0:
                shear[0,1] = shear_factors[0]
                shear[0,2] = shear_factors[1]
            elif self.dire == 1:
                shear[1,0] = shear_factors[0]
                shear[1,2] = shear_factors[1]
            elif self.dire == 2:
                shear[2,0] = shear_factors[0]
                shear[2,1] = shear_factors[1]
            
            scal_factors = np.exp(2*np.log(scal_bounds)*(np.random.rand(3)-0.5))
            scal = np.diag(scal_factors)
 
            affine = np.matmul(rot,np.matmul(scal,shear))
            affine_transfo.SetMatrix(np.ravel(affine)) 

        return affine_transfo


    def diffeo_transfo(self):
        
        if self.diffeo_params is not None:
            shrink_factor = self.diffeo_params['shrink_factor']
            smooth_factor = self.diffeo_params['smooth_factor']
            svf_std_max = self.diffeo_params['svf_std_max']
            int_steps = self.diffeo_params['int_steps']
            
            shrink_img = sitk.Shrink(self.img_geom, [shrink_factor]*self.ndims)
            shrinked_shape = list(shrink_img.GetSize()[::-1])            
            svf_std = 2*svf_std_max*(np.random.rand(1)-0.5)
            svf = np.zeros(shrinked_shape + [self.ndims])
            svf[..., self.dire] = svf_std * np.random.normal(size = shrinked_shape)
            svf_img = sitk.GetImageFromArray(svf, isVector=True)
            svf_img.CopyInformation(shrink_img)
            
            shrink_img = sitk.Shrink(self.img_geom, [2]*self.ndims)
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(shrink_img)
            resampler.SetUseNearestNeighborExtrapolator(True)
            svf_img = resampler.Execute(svf_img)
            
            gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
            gaussian.SetSigma(smooth_factor)
            svf_img = gaussian.Execute(svf_img)
            svf_img = sitk.Cast(svf_img, sitk.sitkVectorFloat64)
            
            transfo = utils.integrate_svf(svf_img, int_steps=int_steps)

            return transfo
        
        else:
            return None