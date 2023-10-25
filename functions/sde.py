import numpy as np
import torch
import math

from .utils import dct_2d, idct_2d

class VPSDE:
    def __init__(self, schedule='cosine', k=28, index=1, sig=1):
        self.k = k
        self.beta_0 = 0.01
        self.beta_1 = 20
        self.sig = sig
        self.cosine_s = 0.008
        self.schedule = schedule
        self.cosine_beta_max = 999.
        self.index = index

        self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. \
                            * (1. + self.cosine_s) / math.pi - self.cosine_s

        freqs = np.pi * torch.linspace(0, k - 1, k) / k
        self.frequencies_squared = freqs[None, None, :, None] ** 2 + freqs[None, None, None, :] ** 2
        self.frequencies_squared = self.frequencies_squared.to('cuda')

        if schedule == 'cosine':
            self.T = 0.9946
        else:
            self.T = 1.

        self.sigma_min = 0.01
        self.sigma_max = 20
        self.eps = 1e-5
        self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))

    def beta(self, t):
        if self.schedule == 'linear':
            beta = (self.beta_1 - self.beta_0) * t + self.beta_0
        elif self.schedule == 'cosine':
            beta = math.pi / 2 * 2 / (self.cosine_s + 1) * torch.tan(
                (t + self.cosine_s) / (1 + self.cosine_s) * math.pi / 2)
        else:
            beta = 2 * np.log(self.sigma_max / self.sigma_min) * (t * 0 + 1)

        return beta

    def marginal_log_mean_coeff(self, t):
        if self.schedule == 'linear':
            log_alpha_t = - 1 / (2 * 2) * (t ** 2) * (self.beta_1 - self.beta_0) - 1 / 2 * t * self.beta_0

        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(
                torch.clamp(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.), -1, 1))
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0

        else:
            log_alpha_t = -torch.exp(np.log(self.sigma_min) + t * np.log(self.sigma_max / self.sigma_min))

        return log_alpha_t

    def Lambda(self, time):
        t = - 2 * self.marginal_log_mean_coeff(time)
        a = 1 - torch.exp(-t[:, None, None, None] * (self.sig + self.frequencies_squared))
        b = self.frequencies_squared + self.sig
        c = a / b
        return c

    def square_Lambda(self, t):
        return torch.sqrt(torch.abs(self.Lambda(t)))

    def diffusion_coeff(self, x, time):
        t = - self.marginal_log_mean_coeff(time).to(x.device)

        dct_coefs = dct_2d(x, norm='ortho')
        dct_coefs = dct_coefs * torch.exp(
            - (self.sig + self.frequencies_squared).to(x.device) * t[:, None, None, None])

        out = dct_coefs
        return idct_2d(out, norm='ortho').to(x.device)

    def marginal_std(self, x, time):
        dct_coefs = dct_2d(x, norm='ortho')
        time = time.to(x.device)
        c = self.square_Lambda(time)
        c *= torch.pow((self.sig + self.frequencies_squared), -(self.index) / 2).to(x.device)
        dct_coefs = dct_coefs.to(x.device) * c.to(x.device)

        out = idct_2d(dct_coefs, norm='ortho')
        return out

    def inverse_marginal_std(self, x, t):
        t = t.to(x.device)
        dct_coefs = dct_2d(x, norm='ortho')
        c = torch.pow(self.Lambda(t) + 1e-5, -1)
        dct_coefs = dct_coefs.to(x.device) * c.to(x.device)

        out = idct_2d(dct_coefs, norm='ortho')
        return out

    def laplacian(self, x, t, s):
        t = (t - s)
        k = x.shape[-1]
        freqs = np.pi * torch.linspace(0, k - 1, k) / k
        frequencies_squared = freqs[:, None] ** 2 + freqs[None, :] ** 2
        dct_coefs = dct_2d(x, norm='ortho')
        dct_coefs = dct_coefs.to(x.device) * (
                    (self.sig + frequencies_squared.to(x.device)) * t[:, None, None, None].to(x.device)).to(x.device)

        out = idct_2d(dct_coefs, norm='ortho')
        return out

    def marginal_laplacian(self, x, t, s):
        t = (t - s)
        k = x.shape[-1]
        freqs = np.pi * torch.linspace(0, k - 1, k) / k
        frequencies_squared = freqs[:, None] ** 2 + freqs[None, :] ** 2
        frequencies_squared = frequencies_squared.to(x.device)
        dct_coefs = dct_2d(x, norm='ortho') * torch.sqrt(torch.abs(t))[:, None, None, None]
        dct_coefs = dct_coefs.to(x.device) * torch.pow(torch.abs((self.sig + frequencies_squared.to(x.device))),
                                                       -(self.index) / 2).to(x.device)
        out = idct_2d(dct_coefs, norm='ortho')
        return out

    def initial_samples(self, train_dataset, batch_size):

        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        x, _ = next(iter(dataloader))
        t = torch.ones(x.shape[0], device='cuda') * self.T

        x_t_mean = self.diffusion_coeff(x, t).to('cuda')  # times*grid
        n = torch.randn(x.shape, device='cuda')
        noise = self.marginal_std(n, t).to('cuda')

        x_t = noise + x_t_mean
        return x_t


class VPSDE1D(VPSDE):

    def __init__(self, schedule='cosine'):
        super().__init__(schedule=schedule)

    def diffusion_coeff(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.pow(1. - torch.exp(self.marginal_log_mean_coeff(t) * 2), 1 / 2)

    def inverse_a(self, a):
        return 2 / np.pi * (1 + self.cosine_s) * torch.acos(a) - self.cosine_s
