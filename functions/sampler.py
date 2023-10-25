import torch
import torch.nn.functional as F
import numpy as np
import tqdm

from functions.utils import dct_2d_shift, idct_2d_shift
import copy

class Sampler:
    def __init__(self, args, config, model, device='cuda'):
        self.steps = args.nfe
        self.device = device
        self.threshold = config.sampling.clamp_threshold
        self.method = args.sample_type
        self.model = model

        self.eps = 1e-5

    def sample(self, sde, x_init, masked_data=None, mask=None, multiplier=4):
        timesteps = torch.linspace(sde.T, self.eps, self.steps + 1)
        self.sde = sde
        with torch.no_grad():
            x = copy.deepcopy(x_init).to(self.device)

            for i in tqdm.tqdm(range(self.steps)):
                vec_s = torch.ones((x.shape[0],), device=self.device) * timesteps[i]
                vec_t = torch.ones((x.shape[0],), device=self.device) * timesteps[i + 1]

                if self.method == 'sde':
                    x = self.sde_score_update(x, vec_s, vec_t)

                    if self.threshold:
                        x = self.norm_grad(x)

                elif self.method == 'sde_imputation':
                    x = self.sde_score_update_imputation(masked_data, mask, x, vec_s, vec_t)

                    if self.threshold:
                        x = self.norm_grad(x)

                elif self.method == 'sde_super_resolution':
                    x = self.sde_score_update_super_resolution(masked_data, x, vec_s, vec_t, multiplier)

                    if self.threshold:
                        x = self.norm_grad(x)

        return x

    def create_circular_mask(self, h, w, center=None, radius=None):
        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return torch.tensor(mask, device=self.device)

    def score_model(self, x, t):
        return self.model(x, t)

    def sde_score_update(self, x, s, t):
        score_s = self.sde.inverse_marginal_std(self.score_model(x, s), s)
        e = torch.randn(x.shape, device=self.device)

        x_t = -self.sde.beta(s)[:, None, None, None] * self.sde.laplacian(x, t, s) / 2 + x
        x_t -= self.sde.beta(s)[:, None, None ,None] * score_s * (t-s)[:, None, None, None]

        noise = torch.sqrt(self.sde.beta(s))[:, None, None, None] * self.sde.marginal_laplacian(e, t, s)

        x_t = x_t + noise
        return x_t

    def sde_score_update_super_resolution(self, x, s, t, masked_data, multiplier=4):

        def create_circular_mask(self, h, w, center=None, radius=None):
            if center is None: # use the middle of the image
                center = (int(w/2), int(h/2))
            if radius is None: # use the smallest distance between the center and image walls
                radius = min(center[0], center[1], w-center[0], h-center[1])

            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

            mask = dist_from_center <= radius
            return torch.tensor(mask, device=self.device)

        def forward_process(self, data, t):

            e = torch.randn(data.shape, device=self.device)
            noise = self.sde.marginal_std(e, t)
            d_mean = self.sde.diffusion_coeff(data, t)

            d_t = d_mean + noise
            return d_t

        score_s = self.sde.inverse_marginal_std(self.score_model(x, s) + self.eps, s)
        e = torch.randn(x.shape, device=self.device)

        x_t = -self.sde.beta(s)[:, None, None, None] * self.sde.laplacian(x, t, s) / 2 + x
        x_t -= self.sde.beta(s)[:, None, None, None] * score_s * (t-s)[:, None, None, None]
        noise = torch.sqrt(self.sde.beta(s))[:, None, None, None] \
                * self.sde.marginal_laplacian(e, t ,s)

        x_t = x_t + noise

        masked_data = forward_process(masked_data, t)
        x = dct_2d_shift(x_t, norm='ortho')
        y = dct_2d_shift(masked_data, norm='ortho')

        mask = create_circular_mask(x_t.shape[2], x_t.shape[3], center=None, radius=multiplier)

        a1 = x * (~mask)
        a2 = y * mask

        x = idct_2d_shift(a1 + a2, norm='ortho')
        return x

    def sde_score_update_imputation(self, x, s, t, masked_data, mask):

        def impainted_noise(data, x_t, mask, t):
            e = torch.randn(data.shape, device=self.device)
            noise = self.sde.marginal_std(e, t)
            d_mean = self.sde.diffusion_coeff(data, t)

            d_t = d_mean + noise

            masked_data = (d_t * mask) + (x_t * ~ mask)
            return masked_data

        score_s = self.sde.inverse_marginal_std(self.score_model(x, s) + self.eps, s)
        e = torch.randn(x.shape, device=self.device)

        x_t = -self.sde.beta(s)[:, None, None, None] * self.sde.laplacian(x, t, s) / 2 + x
        x_t -= self.sde.beta(s)[:, None, None, None] * score_s * (t-s)[:, None, None, None]
        noise = torch.sqrt(self.sde.beta(s))[:, None, None, None] \
                * self.sde.marginal_laplacian(e, t, s)

        x_t = x_t + noise

        x = impainted_noise(masked_data, x_t, mask, t)
        return x

    def forward_process(self, data, t):

        e = torch.randn(data.shape, device=self.device)
        noise = self.sde.marginal_std(e, t)
        d_mean = self.sde.diffusion_coeff(data, t)

        d_t = d_mean + noise
        return d_t

    def norm_grad(self, x):
        size = x.shape
        l = len(x)

        x = x.reshape((l, -1))
        indices = x.norm(dim=1) > self.threshold
        x[indices] = x[indices] / x[indices].norm(dim=1)[:, None] * self.threshold

        x = x.reshape(size)
        return x

def sampler(x, model, sde, device, W, eps, dataset, steps=1000, sampler_input=None):
    def sde_score_update(x, s, t):
        """
        input: x_s, s, t
        output: x_t
        """
        models = model(x, s)
        score_s = models * torch.pow(sde.marginal_std(s), -(2.0 - 1))[:, None].to(device)

        beta_step = sde.beta(s) * (s - t)
        x_coeff = 1 + beta_step / 2.0

        noise_coeff = torch.pow(beta_step, 1 / 2.0)
        if sampler_input == None:
            e = W.sample(x.shape)
        else:
            e = W.free_sample(free_input=sampler_input)

        score_coeff = beta_step
        x_t = x_coeff[:, None].to(device) * x + score_coeff[:, None].to(device) * score_s + noise_coeff[:, None].to(device) * e.to(device)

        return x_t

    timesteps = torch.linspace(sde.T, eps, steps + 1).to(device)

    with torch.no_grad():
        for i in tqdm.tqdm(range(steps)):
            vec_s = torch.ones((x.shape[0],)).to(device) * timesteps[i]
            vec_t = torch.ones((x.shape[0],)).to(device) * timesteps[i + 1]

            x = sde_score_update(x, vec_s, vec_t)

            size = x.shape
            l = x.shape[0]
            x = x.reshape((l, -1))
            indices = x.norm(dim=1) > 10
            if dataset == 'Gridwatch': # Gridwatch - 23.05.11
                x[indices] = x[indices] / x[indices].norm(dim=1)[:, None] * 17
            else:
                x[indices] = x[indices] / x[indices].norm(dim=1)[:, None] * 10
            x = x.reshape(size)

    return x
