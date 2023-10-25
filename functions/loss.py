import torch.nn.functional as F

def loss_fn(model, sde, x_0, t, e):
    x_mean = sde.diffusion_coeff(x_0, t)
    noise = sde.marginal_std(e, t)

    x_t = x_mean + noise
    score = -noise

    output = model(x_t, t)

    loss = (output - score).square().sum(dim=(1,2,3)).mean(dim=0)
    return loss

def hilbert_loss_fn(model, sde, x_0, t, e):
    x_mean = sde.diffusion_coeff(t)
    noise = sde.marginal_std(t)

    x_t = x_0 * x_mean[:, None] + e * noise.view(-1, 1)
    score = -e

    output = model(x_t, t.float())

    loss = (output - score).square().sum(dim=(1)).mean(dim=0)
    return loss
