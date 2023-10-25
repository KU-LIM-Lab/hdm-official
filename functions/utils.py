import os

import torch
from torch import nn, einsum
import tqdm
import numpy as np
from einops import rearrange

from tqdm.asyncio import trange, tqdm

import matplotlib.pyplot as plt
import torchvision.utils as tvu
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from scipy.spatial.distance import cdist

def kernel(x, y, gain=1.0, lens=1,  metric='seuclidean',device='cuda'):

    # x = x.view(-1, 1, 1, 1)
    # y = y.view(1, 1,-1, 1)
    x=x.cpu()
    y=y.cpu()
    x = x.view(-1, 1)
    y = y.view(-1,1)
  
    x_ = x.view(-1,1)
    y_ = y.view(-1,1)


    squared_diff_x = rearrange(cdist(x,y, metric=metric), ' (i j) (k l) -> i j k l', j =1, l=1)
    squared_diff_y =  rearrange(cdist(x_,y_, metric=metric), ' (i j) (k l) -> i j k l', i =1, k=1)
    K = torch.from_numpy((squared_diff_x + squared_diff_y) / (2 * lens))
    K = gain * torch.exp(-K).to(torch.float32)

    return K.to(device).to(torch.float32)


class hilbert_noise:
    def __init__(self,config,device='cuda'):
        self.grid = grid = config.diffusion.grid
        self.device= device
        self.metric =metric =config.diffusion.metric 
        self.initial_point = config.diffusion.initial_point
        self.end_point = config.diffusion.end_point

        self.x =  torch.linspace(config.diffusion.initial_point, config.diffusion.end_point, grid).to(self.device)
        self.y = self.x
        
        self.lens = lens= config.diffusion.lens
        self.gain = gain = config.diffusion.gain
        
        K = kernel(self.x,self.y,lens=lens,gain=gain, metric=metric, device=device)  # (2grid,2grid)
        K = rearrange(K, 'i j k l -> (i j) (k l)')
        eig_val, eig_vec = torch.linalg.eigh(K+1e-6*torch.eye(K.shape[0], K.shape[0]).to(self.device))
        self.lens =lens
        self.eig_val =  eig_val.to(self.device) 
        self.eig_vec = eig_vec.to(torch.float32).to(self.device) 
        print('eig_val', eig_val.min(), eig_val.max())
        self.D = torch.diag(self.eig_val).to(torch.float32).to(self.device) 
        self.M = torch.matmul(self.eig_vec, self.D).to(self.device) 
        self.M = rearrange(self.M, '(i j) (k l) -> i j k l ', j=grid, k=grid ).to(self.device)#(grid**2, grid**2)

        self.gain=gain
 
    def sample(self,size):
        size = list(size) # batch*ch*grid*grid
        x_0 = torch.randn(size).to(self.device)  

        output = einsum('j k x y, b c x y -> b c j k',  self.M,x_0) # R^d -> H_lambda transformation 

        return output #(batch, ch, grid,grid) 

    def free_sample(self,resolution_grid): #input (batch,channel, resolution, resolution)
            y =  torch.linspace(self.initial_point, self.end_point, resolution_grid).to(self.device).unsqueeze(1).to(self.device) # grid x 1

            phi = rearrange(self.eig_vec, '(i j) (k l) -> i j k l ', j=self.grid, k=self.grid ).to(self.device)
            K =kernel(self.x,y,lens=self.lens,gain=self.gain).to(self.device)


            N = einsum('i j x y, i j k l -> x y k l',phi, K )


            return N
    
def data_store(train_loader,path):
    transform = transforms.Compose([transforms.Resize((28, 28)),
                                      transforms.ToTensor()
                                      ])
    j=0
    for x,y in tqdm(train_loader):
        x = x.to('cuda')
        n = len(x)
        for i in range(n):
            sam = x[i]
            j = j + 1
            tvu.save_image( sam, os.path.join(path, f"{j}.png"))




"""Taken from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
Some modifications have been made to work with newer versions of Pytorch"""

import numpy as np
import torch
import torch.nn as nn


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    #Vc = torch.fft.rfft(v, 1)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))
    
    k = - torch.arange(N, dtype=x.dtype,
                       device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V

def dct_shift(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    #Vc = torch.fft.rfft(v, 1)
    Vc = torch.view_as_real(torch.fft.fftshift(torch.fft.fft(v, dim=1), dim=1))
    
    k = - torch.arange(N, dtype=x.dtype,
                       device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype,
                     device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    #v = torch.fft.irfft(V, 1)
    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)

def idct_shift(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype,
                     device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    #v = torch.fft.irfft(V, 1)
    v = torch.fft.irfft(torch.fft.fftshift(torch.view_as_complex(V), dim=1), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)

def dct_2d_shift(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct_shift(x, norm=norm)
    X2 = dct_shift(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)

def idct_2d_shift(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct_shift(X, norm=norm)
    x2 = idct_shift(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_3d(dct_3d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


class LinearDCT(nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""

    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False  # don't learn this!


def apply_linear_2d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 2D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 2 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)


def apply_linear_3d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 3D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    X3 = linear_layer(X2.transpose(-1, -3))
    return X3.transpose(-1, -3).transpose(-1, -2)


if __name__ == '__main__':
    x = torch.Tensor(1000, 4096)
    x.normal_(0, 1)
    linear_dct = LinearDCT(4096, 'dct')
    error = torch.abs(dct(x) - linear_dct(x))
    assert error.max() < 1e-3, (error, error.max())
    linear_idct = LinearDCT(4096, 'idct')
    error = torch.abs(idct(x) - linear_idct(x))
    assert error.max() < 1e-3, (error, error.max())
    
class DCTBlur(nn.Module):

    def __init__(self, image_size, device):
        super(DCTBlur, self).__init__()
        self.device = device

    def forward(self, x,t):
        image_size = x.shape[-1]
        freqs = np.pi*torch.linspace(0, image_size-1,
                                     image_size).to(self.device)/image_size
        self.frequencies_squared = freqs[:, None]**2 + freqs[None, :]**2
        self.frequencies_squared = self.frequencies_squared.to('cuda')
        t = 0.5*(20/0.5)**t
        t = t**2/2
        t=t[:,None,None,None].to('cuda')
        dct_coefs = dct_2d(x, norm='ortho').to('cuda')
        dct_coefs = dct_coefs * torch.exp(- self.frequencies_squared * t)
        # dct_coefs = idct_2d(dct_coefs, norm="ortho")
        return dct_coefs
    

def initial_samples(train_dataset,batch_size,sde):
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0)
    x,y =next(iter(dataloader))
    t=torch.ones(x.shape[0])*1
    t=t.to('cuda')
    x_t_mean = D(x,t) #times*grid 
    print('ih')
    x_t =idct_2d(x_t_mean.to('cuda'), norm='ortho')
    

    return x_t
    



import torch.linalg

from torch.utils import data
from torchvision import transforms, utils
from torchvision import datasets

import numpy as np
from tqdm import tqdm
from einops import rearrange

import torchgeometry as tgm
import glob
import os
from torch import linalg as LA


from scipy.ndimage import zoom as scizoom
from PIL import Image as PILImage
from kornia.color.gray import rgb_to_grayscale
import cv2
import torch.nn.functional as F

class ForwardProcessBase:
    
    def forward(self, x, i):
        pass

    @torch.no_grad()
    def reset_parameters(self, batch_size=32):
        pass



class Snow(ForwardProcessBase):
    
    def __init__(self,
                 image_size=(32,32),
                 snow_level=1,
                 num_timesteps=50,
                 snow_base_path=None,
                 random_snow=False,
                 single_snow=False,
                 batch_size=32,
                 load_snow_base=False,
                 fix_brightness=False):
        
        self.num_timesteps = num_timesteps
        self.random_snow = random_snow
        self.snow_level = snow_level
        self.image_size = image_size
        self.single_snow = single_snow
        self.batch_size = batch_size
        self.generate_snow_layer()
        self.fix_brightness = fix_brightness
    
    @torch.no_grad()
    def reset_parameters(self, batch_size=-1):
        if batch_size != -1:
            self.batch_size = batch_size
        if self.random_snow:
            self.generate_snow_layer()



    @torch.no_grad()
    def generate_snow_layer(self):
        if not self.random_snow:
            rstate = np.random.get_state()
            np.random.seed(123321)
        # c[0]/c[1]: mean/std of Gaussian for snowy pixels
        # c[2]: zoom factor
        # c[3]: threshold for snowy pixels
        # c[4]/c[5]: radius/sigma for motion blur
        # c[6]: brightness coefficient
        if self.snow_level == 1:
            c = (0.1, 0.3, 3, 0.5, 5, 4, 0.8)
            snow_thres_start = 0.7
            snow_thres_end = 0.3
            mb_sigma_start = 0.5
            mb_sigma_end = 5.0
            br_coef_start = 0.95
            br_coef_end = 0.7
        elif self.snow_level == 2:
            c = (0.55, 0.3, 2.5, 0.85, 11, 12, 0.55) 
            snow_thres_start = 1.15
            snow_thres_end = 0.7
            mb_sigma_start = 0.05
            mb_sigma_end = 12
            br_coef_start = 0.95
            br_coef_end = 0.55
        elif self.snow_level == 3:
            c = (0.55, 0.3, 2.5, 0.7, 11, 16, 0.4) 
            snow_thres_start = 1.15
            snow_thres_end = 0.7
            mb_sigma_start = 0.05
            mb_sigma_end = 16
            br_coef_start = 0.95
            br_coef_end = 0.4
        elif self.snow_level == 4:
            c = (0.55, 0.3, 2.5, 0.55, 11, 20, 0.3) 
            snow_thres_start = 1.15
            snow_thres_end = 0.55
            mb_sigma_start = 0.05
            mb_sigma_end = 20
            br_coef_start = 0.95
            br_coef_end = 0.3



        self.snow_thres_list = torch.linspace(snow_thres_start, snow_thres_end, self.num_timesteps).tolist()

        self.mb_sigma_list = torch.linspace(mb_sigma_start, mb_sigma_end, self.num_timesteps).tolist()

        self.br_coef_list = torch.linspace(br_coef_start, br_coef_end, self.num_timesteps).tolist()


        self.snow = []
        self.snow_rot = []
        
        if self.single_snow:
            sb_list = []
            for _ in range(self.batch_size):
                cs = np.random.normal(size=self.image_size, loc=c[0], scale=c[1])
                cs = cs[..., np.newaxis]
                cs = scizoom(cs, c[2])
                sb_list.append(cs)
            snow_layer_base = np.concatenate(sb_list, axis=2)
        else:
            snow_layer_base = np.random.normal(size=self.image_size, loc=c[0], scale=c[1])
            snow_layer_base = snow_layer_base[..., np.newaxis]
            snow_layer_base = scizoom(snow_layer_base, c[2])
        
        vertical_snow = False
        if np.random.uniform() > 0.5:
            vertical_snow = True

        for i in range(self.num_timesteps):

            snow_layer = torch.Tensor(snow_layer_base).clone()
            snow_layer[snow_layer < self.snow_thres_list[i]] = 0
            snow_layer = torch.clip(snow_layer, 0, 1)
            snow_layer = snow_layer.permute((2, 0, 1)).unsqueeze(1)
            # Apply motion blur
            kernel_param = tgm.image.get_gaussian_kernel(c[4], self.mb_sigma_list[i])
            motion_kernel = torch.zeros((c[4], c[4]))
            motion_kernel[int(c[4] / 2)] = kernel_param

            horizontal_kernel = motion_kernel[None, None, :]
            horizontal_kernel = horizontal_kernel.repeat(3, 1, 1, 1)
            vertical_kernel = torch.rot90(motion_kernel, k=1, dims=[0,1])
            vertical_kernel = vertical_kernel[None, None, :]
            vertical_kernel = vertical_kernel.repeat(3, 1, 1, 1)

            vsnow = F.conv2d(snow_layer, vertical_kernel, padding='same', groups=1)
            hsnow = F.conv2d(snow_layer, horizontal_kernel, padding='same', groups=1)
            if self.single_snow:
                vidx = torch.randperm(snow_layer.shape[0])
                vidx = vidx[:int(snow_layer.shape[0]/2)]
                snow_layer = hsnow
                snow_layer[vidx] = vsnow[vidx]
            elif vertical_snow:
                snow_layer = vsnow
            else:
                snow_layer = hsnow
            self.snow.append(snow_layer)
            self.snow_rot.append(torch.rot90(snow_layer, k=2, dims=[2,3]))
        
        if not self.random_snow:
            np.random.set_state(rstate)

    @torch.no_grad()
    def total_forward(self, x_in):
        return self.forward(None, self.num_timesteps-1, og=x_in)
    
    @torch.no_grad()
    def forward(self, x, i, og=None):
        og_r = (og + 1.) / 2.
        og_gray = rgb_to_grayscale(og_r) * 1.5 + 0.5
        og_gray = torch.maximum(og_r, og_gray)
        br_coef = self.br_coef_list[i]
        scaled_og = br_coef * og_r + (1 - br_coef) * og_gray
        if self.fix_brightness:
            snowy_img = torch.clip(og_r + self.snow[i].cuda() + self.snow_rot[i].cuda(), 0.0, 1.0)
        else:
            snowy_img = torch.clip(scaled_og + self.snow[i].cuda() + self.snow_rot[i].cuda(), 0.0, 1.0)
        return (snowy_img * 2.) - 1.
    


import math

import torch
import torch.nn as nn

from kornia.color.rgb import linear_rgb_to_rgb, rgb_to_linear_rgb
from kornia.color.xyz import rgb_to_xyz, xyz_to_rgb

def rgb2hsv(image_old: torch.Tensor, eps: float = 1e-8, rescale=True) -> torch.Tensor:
    r"""Convert an image from RGB to HSV.

    .. image:: _static/img/rgb_to_hsv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
        eps: scalar to enforce numarical stability.

    Returns:
        HSV version of the image with shape of :math:`(*, 3, H, W)`.
        The H channel values are in the range 0..2pi. S and V are in the range 0..1.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # 2x3x4x5
    """
    if rescale: 
        image = (image_old + 1) * 0.5
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    max_rgb, argmax_rgb = image.max(-3)
    min_rgb, argmin_rgb = image.min(-3)
    deltac = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + eps)

    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

    h1 = (bc - gc)
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h = (h / 6.0) % 1.0
    h = 2. * math.pi * h  # we return 0/2pi output

    return torch.stack((h, s, v), dim=-3)



def hsv2rgb(image: torch.Tensor, rescale=True) -> torch.Tensor:
    r"""Convert an image from HSV to RGB.

    The H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.

    Args:
        image: HSV Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hsv_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    h: torch.Tensor = image[..., 0, :, :] / (2 * math.pi)
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    one: torch.Tensor = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)

    hi = hi.long()
    indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)
    
    if rescale:
        out = 2.0 * out - 1

    return out


"""
The RGB to Lab color transformations were translated from scikit image's rgb2lab and lab2rgb

https://github.com/scikit-image/scikit-image/blob/a48bf6774718c64dade4548153ae16065b595ca9/skimage/color/colorconv.py

"""

def rgb2lab(image_old: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to Lab.

    .. image:: _static/img/rgb_to_lab.png

    The image data is assumed to be in the range of :math:`[0, 1]`. Lab
    color is computed using the D65 illuminant and Observer 2.

    Args:
        image: RGB Image to be converted to Lab with shape :math:`(*, 3, H, W)`.

    Returns:
        Lab version of the image with shape :math:`(*, 3, H, W)`.
        The L channel values are in the range 0..100. a and b are in the range -127..127.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_lab(input)  # 2x3x4x5
    """
    image = (image_old + 1) * 0.5
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)

    xyz_im: torch.Tensor = rgb_to_xyz(lin_rgb)

    # normalize for D65 white point
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    threshold = 0.008856
    power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int = torch.where(xyz_normalized > threshold, power, scale)

    x: torch.Tensor = xyz_int[..., 0, :, :]
    y: torch.Tensor = xyz_int[..., 1, :, :]
    z: torch.Tensor = xyz_int[..., 2, :, :]

    L: torch.Tensor = (116.0 * y) - 16.0
    a: torch.Tensor = 500.0 * (x - y)
    _b: torch.Tensor = 200.0 * (y - z)

    out: torch.Tensor = torch.stack([L, a, _b], dim=-3)

    return out


def lab2rgb(image: torch.Tensor, clip: bool = True) -> torch.Tensor:
    r"""Convert a Lab image to RGB.

    Args:
        image: Lab image to be converted to RGB with shape :math:`(*, 3, H, W)`.
        clip: Whether to apply clipping to insure output RGB values in range :math:`[0, 1]`.

    Returns:
        Lab version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = lab_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    L: torch.Tensor = image[..., 0, :, :]
    a: torch.Tensor = image[..., 1, :, :]
    _b: torch.Tensor = image[..., 2, :, :]

    fy = (L + 16.0) / 116.0
    fx = (a / 500.0) + fy
    fz = fy - (_b / 200.0)

    # if color data out of range: Z < 0
    fz = fz.clamp(min=0.0)

    fxyz = torch.stack([fx, fy, fz], dim=-3)

    # Convert from Lab to XYZ
    power = torch.pow(fxyz, 3.0)
    scale = (fxyz - 4.0 / 29.0) / 7.787
    xyz = torch.where(fxyz > 0.2068966, power, scale)

    # For D65 white point
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz.device, dtype=xyz.dtype)[..., :, None, None]
    xyz_im = xyz * xyz_ref_white

    rgbs_im: torch.Tensor = xyz_to_rgb(xyz_im)

    # https://github.com/richzhang/colorization-pytorch/blob/66a1cb2e5258f7c8f374f582acc8b1ef99c13c27/util/util.py#L107
    #     rgbs_im = torch.where(rgbs_im < 0, torch.zeros_like(rgbs_im), rgbs_im)

    # Convert from RGB Linear to sRGB
    rgb_im = linear_rgb_to_rgb(rgbs_im)

    # Clip to 0,1 https://www.w3.org/Graphics/Color/srgb
    if clip:
        rgb_im = torch.clamp(rgb_im, min=0.0, max=1.0)
    
    rgb_im = 2.0 * rgb_im - 1

    return rgb_im


class RgbToLab(nn.Module):
    r"""Convert an image from RGB to Lab.

    The image data is assumed to be in the range of :math:`[0, 1]`. Lab
    color is computed using the D65 illuminant and Observer 2.

    Returns:
        Lab version of the image.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> lab = RgbToLab()
        >>> output = lab(input)  # 2x3x4x5

    Reference:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

        [2] https://www.easyrgb.com/en/math.php

        [3] https://github.com/torch/image/blob/dc061b98fb7e946e00034a5fc73e883a299edc7f/generic/image.c#L1467
    """

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_lab(image)




class LabToRgb(nn.Module):
    r"""Convert an image from Lab to RGB.

    Returns:
        RGB version of the image. Range may not be in :math:`[0, 1]`.

    Shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> rgb = LabToRgb()
        >>> output = rgb(input)  # 2x3x4x5

    References:
        [1] https://docs.opencv.org/4.0.1/de/d25/imgproc_color_conversions.html

        [2] https://www.easyrgb.com/en/math.php

        [3] https://github.com/torch/image/blob/dc061b98fb7e946e00034a5fc73e883a299edc7f/generic/image.c#L1518
    """

    def forward(self, image: torch.Tensor, clip: bool = True) -> torch.Tensor:
        return lab_to_rgb(image, clip)
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
