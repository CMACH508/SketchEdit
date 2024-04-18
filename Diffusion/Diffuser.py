import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyper_params import hp


def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, cond, noise=None):
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = (
                extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        # predict_noise, rec = self.model(x_t, t, cond)
        predict_noise = self.model(x_t, t, cond)

        loss = F.mse_loss(predict_noise, noise, reduction='none')
        # return loss, rec
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T] 

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                extract(self.coeff1, t, x_t.shape) * x_t -
                extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t, cond):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        eps = self.model(x_t, t, cond)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T, cond):
        x_t = x_T
        with torch.no_grad():
            for time_step in reversed(range(self.T)):
                # print(time_step)
                t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                mean, var = self.p_mean_variance(x_t=x_t, t=t, cond=cond)
                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0
                x_t = mean + torch.sqrt(var) * noise
                assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)


class DDIM(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, ddim_step=hp.ddim_step):
        super().__init__()

        self.model = model
        self.T = T
        self.ts = torch.linspace(T, 0,
                                 (ddim_step + 1)).to(torch.long)
        self.ddim_step = ddim_step
        self.eta = 1

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.alpha_bars = alphas_bar
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T] 

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # self.sigma = (
        #         self.eta
        #         * torch.sqrt((1 - alphas_bar_prev) / (1 - alphas_bar))
        #         * torch.sqrt(1 - alphas_bar / alphas_bar_prev)
        # )
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        var = self.eta * (1 - self.ab_prev) / (1 - self.ab_cur) * (1 - self.ab_cur / self.ab_prev)

        first_term = (self.ab_prev / self.ab_cur) ** 0.5 * self.x_t

        second_term = ((1 - self.ab_prev - var) ** 0.5 -
                       (self.ab_prev * (1 - self.ab_cur) / self.ab_cur) ** 0.5) * eps
        simple_var = False
        noise = torch.randn_like(self.x_t).to(x_t.device)

        if simple_var:
            third_term = (1 - self.ab_cur / self.ab_prev) ** 0.5 * noise
        else:
            third_term = var ** 0.5 * noise

        return first_term + second_term + third_term

    def p_mean_variance(self, x_t, t, cond):
        eps = self.model(x_t, t, cond)
        xt_1 = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_1

    def forward(self, x_T, cond):
        x_t = x_T
        with torch.no_grad():
            for time_step in (range(1, self.ddim_step + 1)):
                self.x_t = x_t
                cur_t = self.ts[time_step - 1] - 1
                prev_t = self.ts[time_step] - 1
                self.ab_cur = self.alpha_bars[cur_t]
                self.ab_prev = self.alpha_bars[prev_t] if prev_t >= 0 else 1

                t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * cur_t
                xt_1 = self.p_mean_variance(x_t=x_t, t=t, cond=cond)
                x_t = torch.clip(xt_1, -1, 1)
                assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t

        return torch.clip(x_0, -1, 1)

    def get_noise_map(self, x_0):
        t = x_0.new_ones([x_0.shape[0], ], dtype=torch.long) * (self.T-1)

        noise = torch.randn_like(x_0)
        x_t = (
                extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        return x_t

