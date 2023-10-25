import torch.nn as nn
import torch.nn.functional as F
from functools import partial, partialmethod
import torch
import math
from models.fno_block import *
from models.padding import *
from models.mlp import *

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

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2
  timesteps = 1000*timesteps
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  # emb = math.log(2.) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
  # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


def get_timestep_embedding_2(timesteps, embedding_dim, max_positions=111):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int256
    half_dim = embedding_dim // 2

    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None].to(timesteps.device)* emb[None, :].to(timesteps.device)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class Lifting(nn.Module):
    def __init__(self, in_channels, out_channels, n_dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        Conv = getattr(nn, f'Conv{n_dim}d')
        self.fc = Conv(in_channels, out_channels, 1)

    def forward(self, x):
        return self.fc(x)


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, n_dim=2, non_linearity=F.gelu):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.non_linearity = non_linearity
        Conv = getattr(nn, f'Conv{n_dim}d')
        self.fc1 = Conv(in_channels, hidden_channels, 1)
        self.fc2 = Conv(hidden_channels, out_channels, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        return x


class FNO(nn.Module):
    """N-Dimensional Fourier Neural Operator

    Parameters
    ----------
    n_modes : int tuple
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the TFNO is inferred from ``len(n_modes)``
    hidden_channels : int
        width of the FNO (i.e. number of channels)
    in_channels : int, optional
        Number of input channels, by default 3
    out_channels : int, optional
        Number of output channels, by default 1
    lifting_channels : int, optional
        number of hidden channels of the lifting block of the FNO, by default 256
    projection_channels : int, optional
        number of hidden channels of the projection block of the FNO, by default 256
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    use_mlp : bool, optional
        Whether to use an MLP layer after each FNO block, by default False
    mlp : dict, optional
        Parameters of the MLP, by default None
        {'expansion': float, 'dropout': float}
    non_linearity : nn.Module, optional
        Non-Linearity module to use, by default F.gelu
    norm : F.module, optional
        Normalization layer to use, by default None
    preactivation : bool, default is False
        if True, use resnet-style preactivation
    skip : {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use, by default 'soft-gating'
    separable : bool, default is False
        if True, use a depthwise separable spectral convolution
    factorization : str or None, {'tucker', 'cp', 'tt'}
        Tensor factorization of the parameters weight to use, by default None.
        * If None, a dense tensor parametrizes the Spectral convolutions
        * Otherwise, the specified tensor factorization is used.
    joint_factorization : bool, optional
        Whether all the Fourier Layers should be parametrized by a single tensor (vs one per layer), by default False
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of the decomposition
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
    domain_padding : None or float, optional
        If not None, percentage of padding to use, by default None
    domain_padding_mode : {'symmetric', 'one-sided'}, optional
        How to perform domain padding, by default 'one-sided'
    fft_norm : str, optional
        by default 'forward'
    """
    def __init__(self, n_modes, hidden_channels,
                 act=nn.SiLU(),
                 in_channels=3,
                 out_channels=1,
                 lifting_channels=256,
                 projection_channels=256,
                 n_layers=4,
                 use_mlp=True, mlp= {'expansion': 4.0, 'dropout': 0.},
                 non_linearity=F.gelu,
                 norm='group_norm', preactivation=True,
                 skip='soft-gating',
                 separable=True,
                 factorization=None,
                 rank=1.0,
                 joint_factorization=False,
                 fixed_rank_modes=False,
                 implementation='factorized',
                 decomposition_kwargs=dict(),
                 domain_padding=None,
                 domain_padding_mode='one-sided',
                 fft_norm='ortho',
                 **kwargs):
        super().__init__()
        self.n_dim = len(n_modes)
        self.act = act
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.skip = skip,
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation


        Dense= [nn.Linear(self.lifting_channels , self.lifting_channels ).to('cuda')]
        Dense[0].weight.data = default_init()(Dense[0].weight.data.shape)
        nn.init.zeros_(Dense[0].bias)
        Dense.append(nn.Linear(self.lifting_channels , self.lifting_channels ).to('cuda'))
        Dense[1].weight.data = default_init()(Dense[1].weight.data.shape)
        nn.init.zeros_(Dense[1].bias)
        self.Dense=nn.ModuleList(Dense)
        # self.attns = nn.ModuleList([ AFNO2D( hidden_size=).to('cuda')]*self.n_layers)
        if domain_padding is not None and domain_padding > 0:
            self.domain_padding = DomainPadding(domain_padding=domain_padding, padding_mode=domain_padding_mode)
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode

        self.convs = FactorizedSpectralConv(
            self.hidden_channels, self.hidden_channels, self.n_modes,
            rank=rank,
            fft_norm=fft_norm,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            n_layers=n_layers,
        )

        self.fno_skips = nn.ModuleList([skip_connection(self.hidden_channels, self.hidden_channels, type=skip, n_dim=self.n_dim) for _ in range(n_layers)])

        if use_mlp:
            self.mlp = nn.ModuleList(
                [MLP(in_channels=self.hidden_channels, hidden_channels=int(round(self.hidden_channels*mlp['expansion'])),
                     dropout=mlp['dropout'], n_dim=self.n_dim,temb_dim=self.hidden_channels) for _ in range(n_layers)]
            )
            self.mlp_skips = nn.ModuleList([skip_connection(self.hidden_channels, self.hidden_channels, type=skip, n_dim=self.n_dim) for _ in range(n_layers)])
        else:
            self.mlp = None

        if norm is None:
            self.norm = None
        elif norm == 'instance_norm':
            self.norm = nn.ModuleList([getattr(nn, f'InstanceNorm{self.n_dim}d')(num_features=self.hidden_channels) for _ in range(n_layers)])
        elif norm == 'group_norm':
            self.norm = nn.ModuleList([nn.GroupNorm(num_groups=4, num_channels=self.hidden_channels) for _ in range(n_layers)])
        elif norm == 'layer_norm':
            self.norm = nn.ModuleList([nn.LayerNorm() for _ in range(n_layers)])
        else:
            raise ValueError(f'Got {norm=} but expected None or one of [instance_norm, group_norm, layer_norm]')

        self.lifting = Lifting(in_channels=in_channels, out_channels=self.hidden_channels, n_dim=self.n_dim)
        self.projection = Projection(in_channels=self.hidden_channels, out_channels=out_channels, hidden_channels=projection_channels,
                                     non_linearity=non_linearity, n_dim=self.n_dim)


    def forward(self, x, t):
        """TFNO's forward pass
        """
        if x.dim()==2:
            x =x.unsqueeze(1)

        x = self.lifting(x)


        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)
        m_idx=0

        temb = get_timestep_embedding(t, self.lifting_channels)
        temb = self.Dense[m_idx].to('cuda')(temb.to('cuda'))

        m_idx += 1
        temb = self.Dense[m_idx].to('cuda')(self.act.to('cuda')(temb.to('cuda')))
        m_idx += 1

        x = x+temb[:,:,None]
        for i in range(self.n_layers):

            if self.preactivation:
                x = self.non_linearity(x)


                if self.norm is not None:
                    x = self.norm[i](x)

            x_fno = self.convs[i]((x,temb))


            if not self.preactivation and self.norm is not None:
                x_fno = self.norm[i](x_fno)

            x_skip = self.fno_skips[i](x)
            x = x_fno + x_skip


            if not self.preactivation and i < (self.n_layers - 1):
                x = self.non_linearity(x)

            if self.mlp is not None:
                x_skip = self.mlp_skips[i](x)

                if self.preactivation:
                    if i < (self.n_layers - 1):
                        x = self.non_linearity(x)

                x = self.mlp[i](x) + x_skip

                if not self.preactivation:
                    if i < (self.n_layers - 1):
                        x = self.non_linearity(x)



        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        x=x.squeeze(1)
        return x


class FNO2(nn.Module):
    """N-Dimensional Fourier Neural Operator

    Parameters
    ----------
    n_modes : int tuple
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the TFNO is inferred from ``len(n_modes)``
    hidden_channels : int
        width of the FNO (i.e. number of channels)
    in_channels : int, optional
        Number of input channels, by default 3
    out_channels : int, optional
        Number of output channels, by default 1
    lifting_channels : int, optional
        number of hidden channels of the lifting block of the FNO, by default 256
    projection_channels : int, optional
        number of hidden channels of the projection block of the FNO, by default 256
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    use_mlp : bool, optional
        Whether to use an MLP layer after each FNO block, by default False
    mlp : dict, optional
        Parameters of the MLP, by default None
        {'expansion': float, 'dropout': float}
    non_linearity : nn.Module, optional
        Non-Linearity module to use, by default F.gelu
    norm : F.module, optional
        Normalization layer to use, by default None
    preactivation : bool, default is False
        if True, use resnet-style preactivation
    skip : {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use, by default 'soft-gating'
    separable : bool, default is False
        if True, use a depthwise separable spectral convolution
    factorization : str or None, {'tucker', 'cp', 'tt'}
        Tensor factorization of the parameters weight to use, by default None.
        * If None, a dense tensor parametrizes the Spectral convolutions
        * Otherwise, the specified tensor factorization is used.
    joint_factorization : bool, optional
        Whether all the Fourier Layers should be parametrized by a single tensor (vs one per layer), by default False
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of the decomposition
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
    domain_padding : None or float, optional
        If not None, percentage of padding to use, by default None
    domain_padding_mode : {'symmetric', 'one-sided'}, optional
        How to perform domain padding, by default 'one-sided'
    fft_norm : str, optional
        by default 'forward'
    """
    def __init__(self, config, device='cuda'):
        super().__init__()
        self.n_modes = n_modes = config.model.n_modes
        self.n_dim = len(n_modes)
        self.act = act=nn.GELU().to(device)
        self.fft_norm = fft_norm= config.model.fft_norm
        self.norm=norm = config.model.norm
        self.mlp_expension = config.model.mlp_expansion
        self.mlp_dropout = config.model.mlp_dropout
        self.hidden_channels = hidden_channels= config.model.hidden_channels
        self.lifting_channels =lifting_channels= config.model.lifting_channels
        self.projection_channels = projection_channels= config.model.projection_channels
        self.in_channels = in_channels =config.model.in_channels
        self.out_channels =out_channels  =config.model. out_channels
        self.n_layers = n_layers=config.model.n_layers
        self.joint_factorization = joint_factorization=config.model.joint_factorization
        self.non_linearity = non_linearity =nn.GELU().to(device)
        self.rank = rank =config.model.rank
        self.use_mlp = use_mlp = config.model.use_mlp
        self.factorization =factorization= config.model.factorization
        self.fixed_rank_modes = fixed_rank_modes= config.model.fixed_rank_modes
        self.skip = skip = config.model.skip
        self.implementation = implementation= config.model.implementation
        self.separable = separable = config.model.separable
        self.preactivation = config.model.preactivation
        domain_padding = config.model.domain_padding
        domain_padding_mode = config.model.domain_padding_mode

        Dense = [nn.Linear(self.lifting_channels, self.hidden_channels).to(device)]
        Dense[0].weight.data = default_init()(Dense[0].weight.data.shape)
        nn.init.zeros_(Dense[0].bias)
        Dense.append(nn.Linear( self.hidden_channels ,  self.hidden_channels).to(device))
        Dense[1].weight.data = default_init()(Dense[1].weight.data.shape)
        nn.init.zeros_(Dense[1].bias)
        self.Dense = nn.ModuleList(Dense)
        if domain_padding is not None and domain_padding > 0:
            self.domain_padding = DomainPadding(domain_padding=domain_padding, padding_mode=domain_padding_mode).to(device)
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode
        self.convs = FactorizedSpectralConv(
            self.hidden_channels, self.hidden_channels, self.n_modes,
            rank=rank,
            fft_norm=fft_norm,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            joint_factorization=joint_factorization,
            n_layers=n_layers, device=device
        )

        self.fno_skips = nn.ModuleList([skip_connection(self.hidden_channels, self.hidden_channels, type=skip, n_dim=self.n_dim).to(device) for _ in range(n_layers)])

        if use_mlp:
            self.mlp = nn.ModuleList(
                [MLP(in_channels=self.hidden_channels, hidden_channels=int(round(self.hidden_channels*self.mlp_expension)),
                     dropout=self.mlp_dropout, n_dim=self.n_dim,temb_dim=self.hidden_channels).to(device) for _ in range(n_layers)]
            )
            self.mlp_skips = nn.ModuleList([skip_connection(self.hidden_channels, self.hidden_channels, type=skip, n_dim=self.n_dim).to(device) for _ in range(n_layers)])
        else:
            self.mlp = None

        if norm is None:
            self.norm = None
        elif norm == 'instance_norm':
            self.norm = nn.ModuleList([getattr(nn, f'InstanceNorm{self.n_dim}d')(num_features=self.hidden_channels).to(device) for _ in range(n_layers)])
        elif norm == 'group_norm':
            self.norm = nn.ModuleList([nn.GroupNorm(num_groups=1, num_channels=self.hidden_channels).to(device) for _ in range(n_layers)])
        elif norm == 'layer_norm':
            self.norm = nn.ModuleList([nn.LayerNorm().to(device) for _ in range(n_layers)])
        else:
            raise ValueError(f'Got {norm=} but expected None or one of [instance_norm, group_norm, layer_norm]')

        self.lifting = Lifting(in_channels=in_channels, out_channels=self.hidden_channels, n_dim=self.n_dim).to(device)
        self.projection = Projection(in_channels=self.hidden_channels, out_channels=out_channels, hidden_channels=projection_channels,
                                     non_linearity=non_linearity, n_dim=self.n_dim).to(device)


    def forward(self, x,t,y=None):
        """TFNO's forward pass
        """

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        m_idx=0

        temb = get_timestep_embedding(t, self.lifting_channels)
        if y is not None:
            cemb = get_timestep_embedding_2(y, self.lifting_channels)
            temb = temb+cemb

        temb = self.Dense[m_idx].to(temb.device)(temb)

        m_idx += 1

        temb = self.Dense[m_idx].to(temb.device)(self.act.to(temb.device)(temb))
        m_idx += 1

        x = x + temb[:,:,None,None]

        for i in range(self.n_layers):

            if self.preactivation:
                x = self.non_linearity(x)

                if self.norm is not None:
                    x = self.norm[i](x)

            x_fno = self.convs[i].to(x.device)((x,temb))

            if not self.preactivation and self.norm is not None:
                x_fno = self.norm[i].to(x.device)(x_fno)

            x_skip = self.fno_skips[i].to(x.device)(x)

            x = x_fno + x_skip

            if not self.preactivation and i < (self.n_layers - 1):
                x = self.non_linearity(x)

            if self.mlp is not None:
                x_skip = self.mlp_skips[i](x)

                if self.preactivation:
                    if i < (self.n_layers - 1):
                        x = self.non_linearity(x)

                x = self.mlp[i].to(x.device)(x, temb) + x_skip

                if not self.preactivation:
                    if i < (self.n_layers - 1):
                        x = self.non_linearity(x)


        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x
    
    
def partialclass(new_name, cls, *args, **kwargs):
    """Create a new class with different default values
    Notes
    -----
    An obvious alternative would be to use functools.partial
    >>> new_class = partial(cls, **kwargs)
    The issue is twofold:
    1. the class doesn't have a name, so one would have to set it explicitly:
    >>> new_class.__name__ = new_name
    2. the new class will be a functools object and one cannot inherit from it.
    Instead, here, we define dynamically a new class, inheriting from the existing one.
    """
    __init__ = partialmethod(cls.__init__, *args, **kwargs)
    new_class = type(new_name, (cls,),  {
        '__init__': __init__,
        '__doc__': cls.__doc__,
        'forward': cls.forward,
    })
    return new_class

TFNO   = partialclass('TFNO', FNO, factorization='Tucker')
TFNO2   = partialclass('TFNO', FNO2, factorization='Tucker')
