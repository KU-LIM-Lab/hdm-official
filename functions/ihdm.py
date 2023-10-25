import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from datasets import data_scaler
from .utils import dct_2d, idct_2d
from .sde import VPSDE

class IHDM(VPSDE):
    def __init__(self, schedule='cosine', k=28, index=1, sig=1):
        super.__init__(self, schedule, k, index, sig)

    def create_forward_process_from_sigmas(self, config, sigmas, device):
        forward_process_module = DCTBlur(sigmas, config.data.image_size, device)
        return forward_process_module
    
    def get_blur_schedule(self, sigma_min, sigma_max, K):
        blur_schedule = np.exp(np.linspace(np.log(sigma_min, sigma_max), K))
        return np.array([0] + list(blur_schedule))
    
    def get_initial_samples(self, config, train_dataset, batch_size):
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        x, _ = next(iter(dataloader))
        x = data_scaler(x)
        x = x.to('cuda')

        blur_schedule = self.get_blur_schedule(config.ihdm.blur_sigma_min, config.ihdm.blur_sigma_max, config.ihdm.K)
        
        forward_process_module = self.create_forward_process_from_sigmas(config, blur_schedule, device=torch.device('cuda'))
        x_t_mean = forward_process_module(x, K * torch.ones(x.shape[0], dtype=torch.long).to('cuda'))
        x_t_mean = x_t_mean 
        noise = torch.randn_like(x_t_mean)

        x_t = (x_t_mean * config.ihdm.scale_factor) + (noise * 0.01)
        return x_t.float()
    
class DCTBlur(nn.Module):

    def __init__(self, blur_sigmas, image_size, device):
        super(DCTBlur, self).__init__()
        self.blur_sigmas = torch.tensor(blur_sigmas).to(device)
        freqs = np.pi*torch.linspace(0, image_size-1,
                                     image_size).to(device)/image_size
        self.frequencies_squared = freqs[:, None]**2 + freqs[None, :]**2

    def forward(self, x, fwd_steps):
        if len(x.shape) == 4:
            sigmas = self.blur_sigmas[fwd_steps][:, None, None, None]
        elif len(x.shape) == 3:
            sigmas = self.blur_sigmas[fwd_steps][:, None, None]
        t = sigmas**2/2
        dct_coefs = dct_2d(x, norm='ortho')
        dct_coefs = dct_coefs * torch.exp(- self.frequencies_squared * t)
        return idct_2d(dct_coefs, norm='ortho')