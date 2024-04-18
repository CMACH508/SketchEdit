import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from hyper_params import hp
import numpy as np
from torch.autograd import Variable


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal=False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias=False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.causal:
            mask = torch.ones(sim.shape[-2:], device=device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super(SpatialGatingUnit, self).__init__()
        self.norm = nn.LayerNorm(d_ffn // 2)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, 1)
        # self.attn = Attention(d_ffn, d_ffn//2, 64)

    def forward(self, x):
        u, v = torch.chunk(x, 2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)  # + self.attn(x)
        out = u * v
        return out


class GMLPblock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len, dpr=0.0):
        super(GMLPblock, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn)
        self.channel_proj2 = nn.Linear(d_ffn // 2, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)

        self.droppath = DropPath(dpr) if dpr > 0.0 else nn.Identity()

    def forward(self, x):
        residual = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = self.droppath(x) + residual
        return out


class stroke2points(nn.Module):
    def __init__(self, d_model, d_ffn, stroke_num):
        super(stroke2points, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_model)
        self.sgu = nn.Conv1d(stroke_num, hp.max_seq_length, 1, 1)

    def forward(self, x):
        x = F.gelu(self.channel_proj1(self.norm(x)))
        x = self.norm2(x)
        x = self.sgu(x)
        return x


class First_Stage_Encoder(nn.Module):
    def __init__(self):
        super(First_Stage_Encoder, self).__init__()
        self.dffn = 96
        self.embedding = nn.Linear(5, 96)
        self.layers = nn.ModuleList([GMLPblock(self.dffn, self.dffn*4, hp.stroke_length, dpr=hp.dropath) for i in range(8)])

        self.linear1 = nn.Sequential(nn.Linear(self.dffn, hp.d_model), nn.GELU())
        self.combine = nn.Sequential(nn.LayerNorm(hp.d_model), nn.Linear(hp.d_model, hp.d_model), nn.GELU())
        self.mu = nn.Linear(hp.d_model, hp.d_model)
        self.sigma = nn.Linear(hp.d_model, hp.d_model)

        self.de_mu = nn.Parameter(2 * torch.rand(hp.k, hp.d_model) - 1, requires_grad=False)
        self.de_sigma2 = nn.Parameter(torch.ones(size=[hp.k, hp.d_model], dtype=torch.float32), requires_grad=False)
        self.de_alpha = nn.Parameter(torch.ones(size=[hp.k, 1], dtype=torch.float32) / hp.k, requires_grad=False)

    def forward(self, strokes):
        '''
        :param strokes: (batchsize, stroke_num, points_per_stroke*5)
        :return: (batchsize, stroke_num, hp.d_model)
        '''
        out = self.embedding(strokes.view(strokes.shape[0] * hp.stroke_num, hp.stroke_length, -1))
        for layer in self.layers:
            out = layer(out)
        out = torch.sum(out, dim=1, keepdim=True)
        out = out.view(-1, hp.stroke_num, self.dffn)

        out = self.combine(self.linear1(out))
        mu = self.mu(out)
        sigma = self.sigma(out)

        z_size = mu.size()
        sigma2 = F.softplus(sigma) + 1e-10
        sigma_e = torch.sqrt(sigma2)

        if mu.get_device() != -1:  # not in cpu
            n = torch.normal(torch.zeros(z_size), torch.ones(z_size)).cuda(mu.get_device())
        else:  # in cpu
            n = torch.normal(torch.zeros(z_size), torch.ones(z_size))
        z = mu + sigma_e * n

        return z, mu, sigma_e

    def assign(self, data):
        self.de_mu.data = data[0]
        self.de_alpha.data = data[2]
        self.de_sigma2.data = data[1]

    def calculate_prob(self, x, q_mu, q_sigma2):
        """ Calculate the probabilistic density """
        # x [batch_size, Nz] q[k,Nz]
        mu = q_mu.view(1, hp.k, hp.d_model).repeat(x.shape[0], 1, 1)
        sigma2 = q_sigma2.view(1, hp.k, hp.d_model).repeat(x.shape[0], 1, 1)
        x = x.view(-1, 1, hp.d_model).repeat(1, hp.k, 1)

        log_exp_part = -0.5 * torch.sum(torch.div(torch.square(x - mu), 1e-30 + sigma2), dim=2).to(x.device)
        log_frac_part = torch.sum(torch.log(torch.sqrt(sigma2 + 1e-30)), dim=2).to(x.device)
        log_prob = log_exp_part - log_frac_part - float(hp.d_model) / 2. * torch.from_numpy(
            np.log([2. * 3.1416])).to(x.device)
        return torch.exp(log_prob)

    def calculate_posterior(self, y, q_mu, q_sigma2, q_alpha, inference=False):
        prob = self.calculate_prob(y, q_mu, q_sigma2)
        temp = torch.multiply(torch.transpose(q_alpha, 1, 0).repeat(y.shape[0], 1), prob)
        sum_temp = torch.sum(temp, dim=1, keepdim=True).repeat(1, hp.k)
        gamma = torch.clamp((torch.div(temp, 1e-300 + sum_temp)), 1e-5, 1.)
        gamma_st = gamma / (1e-10 + (torch.sum(gamma, dim=1, keepdim=True)).repeat(1, hp.k))

        if not inference:
            return gamma_st, gamma_st.argmax(dim=1)

        first = F.one_hot(torch.argmax(gamma_st, dim=1), hp.k)
        second = gamma_st - first
        return gamma_st, gamma_st.argmax(dim=1), second.argmax(dim=1)

    def get_alpha_loss(self, p_alpha, q_alpha):
        p_alpha = p_alpha.view(-1, hp.k)
        q_alpha = q_alpha.view(1, hp.k).repeat(p_alpha.shape[0], 1)
        return torch.sum(torch.mean(p_alpha * torch.log(torch.div(p_alpha, q_alpha + 1e-10) + 1e-10), dim=0))

    def get_gaussian_loss(self, p_alpha, p_mu, p_sigma2, q_mu, q_sigma2):
        p_alpha = p_alpha.view(-1, hp.k)
        p_mu = p_mu.view(-1, 1, hp.d_model).repeat(1, hp.k, 1)
        p_sigma2 = p_sigma2.view(-1, 1, hp.d_model).repeat(1, hp.k, 1)
        q_mu = q_mu.view(1, hp.k, hp.d_model).repeat(p_alpha.shape[0], 1, 1)
        q_sigma2 = q_sigma2.view(1, hp.k, hp.d_model).repeat(p_alpha.shape[0], 1, 1)

        return torch.mean(torch.sum(0.5 * p_alpha * torch.sum(
            torch.log(q_sigma2 + 1e-5) + (p_sigma2 + (p_mu - q_mu) ** 2) / (q_sigma2 + 1e-5)
            - 1.0 - torch.log(p_sigma2 + 1e-10), dim=2), dim=1))

    def em(self, y, en_sigma2, gamma, q_mu_old, q_sigma2_old, q_alpha_old):
        en_sigma2 = en_sigma2.view(-1, 1, hp.d_model).repeat(1, hp.k, 1)

        sum_gamma = torch.sum(gamma, dim=0).unsqueeze(1).repeat(1, hp.d_model)

        temp_y = y.unsqueeze(1).repeat(1, hp.k, 1)
        q_mu_new = torch.sum(temp_y * gamma.unsqueeze(2).repeat(1, 1, hp.d_model), dim=0) / (sum_gamma + 1e-10)

        q_sigma2_new = torch.sum(
            (torch.square(temp_y - q_mu_new.unsqueeze(0).repeat(en_sigma2.shape[0], 1, 1)) + en_sigma2)
            * gamma.unsqueeze(2).repeat(1, 1, hp.d_model), dim=0) \
                       / (sum_gamma + 1e-10)

        q_alpha_new = torch.mean(gamma, dim=0).unsqueeze(1)
        factor = 0.95

        q_mu = q_mu_old * factor + q_mu_new * (1 - factor)
        q_sigma2 = torch.clamp(q_sigma2_old * factor + q_sigma2_new * (1 - factor), 1e-10, 1e10)
        q_alpha = torch.clamp(q_alpha_old * factor + q_alpha_new * (1 - factor), 0., 1.)
        q_alpha_st = q_alpha / torch.sum(q_alpha)

        return q_mu, q_sigma2, q_alpha_st

    def rpcl(self, y, en_sigma2, gamma, q_mu_old, q_sigma2_old, q_alpha_old):
        en_sigma2 = en_sigma2.view(-1, 1, hp.d_model).repeat(1, hp.k, 1)
        penalize = 1e-4  # De-learning rate
        # y:[batch_size,Nz]
        # temp_y [batch_size,k,Nz]
        temp_y = y.unsqueeze(1).repeat(1, hp.k, 1)

        winner = F.one_hot(torch.argmax(gamma, dim=1), hp.k)
        rival = F.one_hot(torch.argmax(gamma - gamma * winner, dim=1), hp.k)

        gamma_rpcl = winner - penalize * rival  # -penalize_2*rival_2  # [batch_size,  k]
        sum_gamma_rpcl = torch.sum(gamma_rpcl, dim=0).unsqueeze(1).repeat(1, hp.d_model)

        q_mu_new = torch.sum(temp_y * gamma_rpcl.unsqueeze(2).repeat(1, 1, hp.d_model), dim=0) / (
                sum_gamma_rpcl + 1e-10)

        q_sigma2_new = torch.sum(
            (torch.square(temp_y - q_mu_new.unsqueeze(0).repeat(en_sigma2.shape[0], 1, 1)) + en_sigma2)
            * gamma_rpcl.unsqueeze(2).repeat(1, 1, hp.d_model), dim=0) \
                       / (sum_gamma_rpcl + 1e-10)

        q_alpha_new = torch.mean(gamma_rpcl, dim=0).unsqueeze(1)

        factor = 0.95

        q_mu = q_mu_old * factor + q_mu_new * (1 - factor)
        q_sigma2 = torch.clamp(q_sigma2_old * factor + q_sigma2_new * (1 - factor), 1e-10, 1e10)
        q_alpha = torch.clamp(q_alpha_old * factor + q_alpha_new * (1 - factor), 0., 1.)
        q_alpha_st = q_alpha / torch.sum(q_alpha)

        return q_mu, q_sigma2, q_alpha_st


class First_Stage_Decoder(nn.Module):
    def __init__(self):
        super(First_Stage_Decoder, self).__init__()
        self.pos2emb = nn.Sequential(nn.Linear(2, hp.d_model), nn.GELU())
        self.layers1 = nn.ModuleList(
            [GMLPblock(hp.d_model, hp.d_ffn, hp.stroke_num, dpr=hp.dropath) for i in range(2)])

        self.seq_decoder_branch = seq_decoder_branch()
        self.img_decoder_branch = img_decoder_branch()

    def forward(self, z, start_points):
        '''
        :param z: (bs, stroke_num, d_model)
        :param start_points:
        :return:
        '''
        #out = z + self.pos2emb(start_points)
        if self.training:
            pos_noise = (torch.rand_like(start_points)-0.5)/10 #start_points [-1, 1]  noise: -0.05 -> 0.05
            pos_noise = pos_noise.cuda(start_points.get_device())
            out = z + self.pos2emb(start_points + pos_noise)
        else:
            out = z + self.pos2emb(start_points)

        for layer in self.layers1:
            out = layer(out)

        rec_img = self.img_decoder_branch(out)
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q = self.seq_decoder_branch(out)

        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, rec_img


class seq_decoder_branch(nn.Module):
    def __init__(self):
        super(seq_decoder_branch, self).__init__()
        self.stroke2points = stroke2points(hp.d_model, hp.d_ffn, hp.stroke_num)
        self.layers2 = nn.ModuleList(
            [GMLPblock(hp.d_model, hp.d_ffn, hp.max_seq_length, dpr=hp.dropath) for i in range(12)])

        self.predicted_head = nn.Sequential(nn.LayerNorm(hp.d_model),
                                            nn.Linear(hp.d_model, 6*hp.M+3))
    def forward(self, x):
        out = self.stroke2points(x)
        for layer in self.layers2:
            out = layer(out)
        out = self.predicted_head(out)

        b,l,d = out.shape
        # out (bs,200,6*20+3)
        out = out.permute(1, 0, 2).contiguous().view(b*l, d)
        params = torch.split(out, 6, 1)

        params_mixture = torch.stack(params[:-1])
        params_pen = params[-1]
        # identify mixture params:
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy = torch.split(params_mixture, 1, 2)

        pi = F.softmax(pi.transpose(0, 1).squeeze(), dim=-1).view(l, -1, hp.M).permute(1, 0, 2)
        sigma_x = torch.exp(sigma_x.transpose(0, 1).squeeze()).view(l, -1, hp.M).permute(1, 0, 2)
        sigma_y = torch.exp(sigma_y.transpose(0, 1).squeeze()).view(l, -1, hp.M).permute(1, 0, 2)
        rho_xy = torch.tanh(rho_xy.transpose(0, 1).squeeze()).view(l, -1, hp.M).permute(1, 0, 2)
        mu_x = mu_x.transpose(0, 1).squeeze().contiguous().view(l, -1, hp.M).permute(1, 0, 2)
        mu_y = mu_y.transpose(0, 1).squeeze().contiguous().view(l, -1, hp.M).permute(1, 0, 2)
        q = F.softmax(params_pen, dim=-1).view(l, -1, 3).permute(1, 0, 2)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q


class stroke2image(nn.Module):
    def __init__(self, d_model, d_ffn, stroke_num):
        super(stroke2image, self).__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_model)
        self.sgu = nn.Conv1d(stroke_num, 4 * 4, 1, 1)

    def forward(self, x):
        '''
        :param x:
        :return: x: (bs,d_model,3*3)
        '''
        x = F.gelu(self.channel_proj1(self.norm(x)))
        x = self.norm2(x)
        x = self.sgu(x)
        return x.view(x.shape[0], self.d_model, 4, 4)


class img_decoder_branch(nn.Module):
    def __init__(self):
        super(img_decoder_branch, self).__init__()
        self.stroke2image = stroke2image(hp.d_model, hp.d_ffn, hp.stroke_num)

        self.layers2 = nn.Sequential(
            nn.Conv2d(hp.d_model, hp.d_model, 3, stride=1, padding=1),
            nn.InstanceNorm2d(hp.d_model), nn.GELU(),
            nn.ConvTranspose2d(hp.d_model, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.InstanceNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.InstanceNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.InstanceNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.InstanceNorm2d(32), nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.InstanceNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.InstanceNorm2d(16), nn.GELU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1, stride=2, output_padding=1)
            )

    def forward(self, x):
        out_0 = self.stroke2image(x)
        out = self.layers2(out_0)
        return out


class stroke_img_decoder_branch(nn.Module):
    def __init__(self):
        super(stroke_img_decoder_branch, self).__init__()
        self.Linear1 = nn.Sequential(nn.LayerNorm(hp.d_model), nn.Linear(hp.d_model, 16 * hp.d_model), nn.GELU())

        self.layers2 = nn.Sequential(
            nn.ConvTranspose2d(hp.d_model, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.InstanceNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.InstanceNorm2d(32), nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.InstanceNorm2d(16), nn.GELU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1, stride=2, output_padding=1)
            )

    def forward(self, x):
        '''
        :param x: (bs, stroke_num, hp.d_model)
        :return:
        '''
        out_0 = self.Linear1(x).view(x.shape[0] * hp.stroke_num, hp.d_model, 4, 4)
        out = self.layers2(out_0)
        return out