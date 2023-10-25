import itertools
import numpy as np

import tensorly as tl
from tensorly.plugins import use_opt_einsum
tl.set_backend('pytorch')
use_opt_einsum('optimal')

from tltorch.factorized_tensors.core import FactorizedTensor
einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

import torch.nn as nn
import torch

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """Ported from JAX. """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init


def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')


# Position embedding
class SinPositionEmbeddingsClass(nn.Module):
    def __init__(self,dim=256,T=10000):
        super().__init__()
        self.dim = dim
        self.T = T
    @torch.no_grad()
    def forward(self,steps):
        steps = 1000*steps
        device = steps.device
        half_dim = self.dim // 2
        embeddings = np.log(self.T) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = steps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class contract_dense(nn.Module):
    def __init__(self, weight, hidden, act,temb_dim = None, separable=False):
       super().__init__()
       self.weight= weight
       self.separable =separable
       self.act =act

       self.Dense_0 = nn.Linear(temb_dim, hidden)
       self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
       nn.init.zeros_(self.Dense_0.bias)
       self.separable = separable

    def forward(self,x,temb=None):
      order = tl.ndim(x)
      # batch-size, in_channels, x, y...
      x_syms = list(einsum_symbols[:order])

      # in_channels, out_channels, x, y...
      weight_syms = list(x_syms[1:]) # no batch-size

      # batch-size, out_channels, x, y...
      if self.separable:
          out_syms = [x_syms[0]] + list(weight_syms)
      else:
        weight_syms.insert(1, einsum_symbols[order]) # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]

      eq= ''.join(x_syms) + ',' + ''.join(weight_syms) + '->' + ''.join(out_syms)

      if not torch.is_tensor(self.weight):
        weight = self.weight.to_tensor()

      if temb is not None:

        x+=self.Dense_0.to('cuda')(self.act.to('cuda')(temb))[:,:,None]

      return tl.einsum(eq, x, self.weight)

class contract_dense_separable(nn.Module):
  def __init__(self, weight,  hidden, act, temb_dim = None, separable=False):
    super().__init__()
    self.weight = weight
    self.act = act
    self.Dense_0 = nn.Linear(weight.shape[0], hidden)
    self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
    nn.init.zeros_(self.Dense_0.bias)
  def forward(self,x,temb=None):
    if self.separable == False:
        raise ValueError('This function is only for separable=True')
    if temb is not None:
        x+=self.Dense_0(self.act(temb))[:,None]
    h=x*self.weight
    return h


class contract_cp(nn.Module):
  def __init__(self, cp_weight,  hidden, act,temb_dim = None, separable=False):
      super().__init__()
      self.cp_weight= cp_weight

      self.Dense_0 = nn.Linear(temb_dim, hidden)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)
      self.separable = separable
      self.act =act

  def forward(self, x,temb=None):
    order = tl.dim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order+1]
    out_syms = list(x_syms)
    if self.separable:
        factor_syms = [einsum_symbols[1]+rank_sym] #in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1]+rank_sym,out_sym+rank_sym] #in, out
    factor_syms += [xs+rank_sym for xs in x_syms[2:]] #x, y, ...
    eq = x_syms + ',' + rank_sym + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)
    if temb is not None:
        x+=self.Dense_0(self.act(temb))[:,None]
    h= tl.einsum(eq, x, self.cp_weight.weights, *self.cp_weight.factors)
    return h


class contract_tucker_1d(nn.Module):
  def __init__(self, tucker_weight,  hidden, act,temb_dim = None, separable=False):
    super().__init__()
    self.tucker_weight =tucker_weight
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(tucker_weight.shape[1], hidden)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)
    self.separable = separable
    self.act =act

  def forward(self,x,temb=None):
    order = tl.ndim(x)
    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)

    if self.separable:
        core_syms = einsum_symbols[order+1:2*order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        factor_syms = [xs+rs for (xs, rs) in zip(x_syms[1:], core_syms)] #x, y, ...

    else:
        core_syms = einsum_symbols[order+1:2*order+1]
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1]+core_syms[0], out_sym+core_syms[1]] #out, in
        factor_syms += [xs+rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])] #x, y, ...

    eq = x_syms + ',' + core_syms + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)
    if temb is not None:
        x+=self.Dense_0(self.act(temb))[:,None]
    h= tl.einsum(eq, x, self.tucker_weight.core, *self.tucker_weight.factors)
    return h

class contract_tucker_2d(nn.Module):
  def __init__(self, tucker_weight,  hidden, act,temb_dim = None, separable=False):
    super().__init__()
    self.tucker_weight =tucker_weight
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(tucker_weight.shape[1], hidden)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)
    self.separable = separable
    self.act =act

  def forward(self,x,temb=None):
    order = tl.ndim(x)
    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)

    if self.separable:
        core_syms = einsum_symbols[order+1:2*order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        factor_syms = [xs+rs for (xs, rs) in zip(x_syms[1:], core_syms)] #x, y, ...

    else:
        core_syms = einsum_symbols[order+1:2*order+1]
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1]+core_syms[0], out_sym+core_syms[1]] #out, in
        factor_syms += [xs+rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])] #x, y, ...

    eq = x_syms + ',' + core_syms + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)
    if temb is not None:
        x=x+self.Dense_0(self.act(temb))[:,:,None,None]
    h= tl.einsum(eq, x, self.tucker_weight.core, *self.tucker_weight.factors)
    return h


class contract_tt(nn.Module):

  def __init__(self, tt_weight,  hidden, act,temb_dim = None, separable=False):
    super().__init__()
    self.tt_weight=tt_weight
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(tt_weight.shape[0], hidden)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)
      self.act =act
      self.separable = separable

  def forward(self,x,temb=None):
    order = tl.nladim(x)
    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:]) # no batch-size
    if not self.separable:
        weight_syms.insert(1, einsum_symbols[order]) # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[order+1:])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i+1]])
    eq = ''.join(x_syms) + ',' + ','.join(''.join(f) for f in tt_syms) + '->' + ''.join(out_syms)
    if temb is not None:
        x+=self.Dense_0(self.act(temb))[:,None]
    return tl.einsum(eq, x, *self.tt_weight.factors)

def get_contract_fun(implementation, weight,  hidden, act,temb_dim = None, separable=False, is_2d=False):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input (factorized)

    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == 'reconstructed':
        if separable:
            print('SEPARABLE')
            return contract_dense_separable
        else:
            return contract_dense
    elif implementation == 'factorized':
        if torch.is_tensor(weight):
            return contract_dense(weight,  hidden, act,temb_dim = temb_dim, separable=separable)
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower() == 'complexdense':
                return contract_dense(weight, hidden, act,temb_dim = temb_dim, separable=separable)
            elif weight.name.lower() == 'complextucker':
                if is_2d:
                   return contract_tucker_2d(weight,  hidden, act,temb_dim = temb_dim, separable=separable)
                else:
                    return contract_tucker_1d(weight,  hidden, act,temb_dim = temb_dim, separable=separable)
            elif weight.name.lower() == 'complextt':
                return contract_tt(weight,  hidden, act,temb_dim = temb_dim, separable=separable)
            elif weight.name.lower() == 'complexcp':
                return contract_cp(weight,  hidden, act,temb_dim = temb_dim, separable=separable)
            else:
                raise ValueError(f'Got unexpected factorized weight type {weight.name}')
        else:
            raise ValueError(f'Got unexpected weight type of class {weight.__class__.__name__}')
    else:
        raise ValueError(f'Got {implementation=}, expected "reconstructed" or "factorized"')


class FactorizedSpectralConv(nn.Module):
    """Generic N-Dimensional Fourier Neural Operator

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels
    out_channels : int, optional
        Number of output channels
    kept_modes : int tuple
        total number of modes to keep in Fourier Layer, along each dim
    separable : bool, default is True
    scale : float or 'auto', default is 'auto'
        scale to use for the init
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    joint_factorization : bool, optional
        Whether all the Fourier Layers should be parametrized by a single tensor (vs one per layer), by default False
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
    factorization : str, {'tucker', 'cp', 'tt'}, optional
        Tensor factorization of the parameters weight to use, by default 'tucker'
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    fft_norm : str, optional
        by default 'forward'
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of the decomposition
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
    """
    def __init__(self, in_channels, out_channels, n_modes, n_layers=1, scale='auto', separable=False,
                 fft_norm='backward', bias=True, implementation='reconstructed', joint_factorization=False,
                 rank=0.5, factorization='cp', fixed_rank_modes=False, decomposition_kwargs=dict(),indices=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.order = len(n_modes)

        # We index quadrands only
        # n_modes is the total number of modes kept along each dimension
        # half_modes is half of that except in the last mode, correponding to the number of modes to keep in *each* quadrant for each dim
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        half_modes = [m//2 for m in n_modes]
        self.half_modes = half_modes

        self.rank = rank
        self.factorization = factorization
        self.n_layers = n_layers
        self.implementation = implementation

        if scale == 'auto':
            scale = (1 / (in_channels * out_channels))

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes=[0]
            else:
                fixed_rank_modes=None

        # self.mlp = None


        self.fft_norm = fft_norm

        # Make sure we are using a Complex Factorized Tensor
        if factorization is None:
            factorization = 'Dense' # No factorization
        if not factorization.lower().startswith('complex'):
            factorization = f'Complex{factorization}'

        if separable:
            if in_channels != out_channels:
                raise ValueError('To use separable Fourier Conv, in_channels must be equal to out_channels, ',
                                 f'but got {in_channels=} and {out_channels=}')
            weight_shape = (in_channels, *self.half_modes)
        else:
            weight_shape = (in_channels, out_channels, *self.half_modes)
        self.separable = separable

        if joint_factorization:
            self.weight = FactorizedTensor.new(((2**(self.order-1))*n_layers, *weight_shape),
                                                rank=self.rank, factorization=factorization,
                                                fixed_rank_modes=fixed_rank_modes,
                                                **decomposition_kwargs).cuda()
            self.weight.normal_(0, scale)
        else:
            self.weight = nn.ModuleList([
                 FactorizedTensor.new(
                    weight_shape,
                    rank=self.rank, factorization=factorization,
                    fixed_rank_modes=fixed_rank_modes,
                    **decomposition_kwargs
                    ).cuda() for _ in range((2**(self.order-1))*n_layers)]
                )
            for w in self.weight:
                w.normal_(0, scale)

        if bias:
            self.bias = nn.Parameter(scale * torch.randn(*((n_layers, self.out_channels) + (1, )*self.order)))
        else:
            self.bias = None
        self.layer=[]
        mode_indexing = [((None, m), (-m, None)) for m in self.half_modes[:-1]] + [((None, self.half_modes[-1]), )]
        for i, boundaries in enumerate(itertools.product(*mode_indexing)):
            if len(n_modes) == 1: 
                self.layer.append(get_contract_fun(implementation, self.weight[indices + i], hidden=out_channels, temb_dim=256, act=nn.SiLU(), separable=self.separable, is_2d=False))
            elif len(n_modes) == 2:
                self.layer.append(get_contract_fun(implementation, self.weight[indices + i], hidden=out_channels, temb_dim=256, act=nn.SiLU(), separable=self.separable, is_2d=True))
            else:
               raise NotImplementedError(f'length of n_modes should be either 1 or 2, but got {len(n_modes)}')
        self.layer =nn.ModuleList(self.layer)

    def forward(self, y, indices=0):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
        indices : int, default is 0
            if joint_factorization, index of the layers for n_layers > 1

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        x = y[0].to('cuda')
        temb = y[1].to('cuda')


        batchsize, channels, *mode_sizes = x.shape
        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1]//2 + 1 # Redundant last coefficient

        #Compute Fourier coeffcients
        fft_dims = list(range(-self.order, 0))
        x = torch.fft.rfftn(x.float(), norm=self.fft_norm, dim=fft_dims)

        out_fft = torch.zeros([batchsize, self.out_channels, *fft_size], device=x.device, dtype=torch.cfloat)

        # We contract all corners of the Fourier coefs
        # Except for the last mode: there, we take all coefs as redundant modes were already removed
        mode_indexing = [((None, m), (-m, None)) for m in self.half_modes[:-1]] + [((None, self.half_modes[-1]), )]

        for i, boundaries in enumerate(itertools.product(*mode_indexing)):
            # Keep all modes for first 2 modes (batch-size and channels)
            idx_tuple = [slice(None), slice(None)] + [slice(*b) for b in boundaries]

            # For 2D: [:, :, :height, :width] and [:, :, -height:, width]
            # out_fft[idx_tuple] = self._contract(x[idx_tuple], t, self.weight[indices + i], separable=self.separable)

            out_fft[idx_tuple] = self.layer[i](x[idx_tuple], temb)
        x = torch.fft.irfftn(out_fft, s=(mode_sizes), norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution

        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError('A single convolution is parametrized, directly use the main class.')

        return SubConv2d(self, indices)

    def __getitem__(self, indices):
        return self.get_conv(indices)



class SubConv2d(nn.Module):
    """Class representing one of the convolutions from the mother joint factorized convolution

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules, they all point to the same data,
    which is shared.
    """
    def __init__(self, main_conv, indices):
        super().__init__()
        self.main_conv = main_conv
        self.indices = indices

    def forward(self, x,temb=None):
        return self.main_conv.forward(x, self.indices)
