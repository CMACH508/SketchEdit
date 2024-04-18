import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from Networks import GMLPblock
from hyper_params import hp

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class EBlock(nn.Module):
    def __init__(self):
        super(EBlock, self).__init__()
        self.layer1 = GMLPblock(hp.ud_model, hp.ud_ffn, hp.stroke_num, hp.dropath)
        self.temb_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(128, hp.ud_model),
        )
        self.layer2 = GMLPblock(hp.ud_model, hp.ud_ffn, hp.stroke_num, hp.dropath)

    def forward(self, x, t):
        out = self.layer1(x) + self.temb_proj(t)
        out = self.layer2(out)
        return out


class DBlock(nn.Module):
    def __init__(self):
        super(DBlock, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(hp.ud_model*2, hp.ud_model), nn.GELU(),
            GMLPblock(hp.ud_model, hp.ud_ffn, hp.stroke_num, hp.dropath))
        self.temb_proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(128, hp.ud_model),
        )
        self.layer2 = GMLPblock(hp.ud_model, hp.ud_ffn, hp.stroke_num, hp.dropath)

    def forward(self, x, t):
        out = self.layer1(x) + self.temb_proj(t)
        out = self.layer2(out)
        return out

class UNet(nn.Module):
    def __init__(self, T = 1000):
        super(UNet, self).__init__()
        self.point2emb = nn.Sequential(nn.Linear(2, hp.ud_model), nn.GELU())
        self.time_embedding = TimeEmbedding(T, hp.ud_model, 128)

        self.fushion = nn.Sequential(nn.LayerNorm(hp.ud_model),
                                     nn.Linear(hp.ud_model, hp.ud_model),
                                     nn.GELU(),
                                     nn.LayerNorm(hp.ud_model),
                                     nn.Conv1d(hp.stroke_num, hp.stroke_num, kernel_size=1, stride=1),
                                     nn.GELU()
                                     )

        self.e_layer = nn.ModuleList([EBlock() for i in range(12)])
        self.d_layer = nn.ModuleList([EBlock() for i in range(12)])

        self.predict = nn.Sequential(nn.LayerNorm(hp.ud_model),
                                     nn.Linear(hp.ud_model, 2))

        self.condlayer = nn.Sequential(nn.Linear(hp.d_model, hp.ud_model), nn.GELU())
        #self.img_decoder_branch = img_decoder_branch()

    def forward(self, x, t, cond):
        '''
        :param x: (bs, hp.num_stroke, 2)
        :param t:
        :param cond: (bs, hp.num_stroke, hp.ud_model)
        :return:
        '''
        mid = []
        time = self.time_embedding(t)
        time = time.view(-1, 1, 128).repeat(1, hp.stroke_num, 1)
        out = self.point2emb(x)+self.condlayer(cond)
        out = self.fushion(out)
        mid.append(out)

        for layer in self.e_layer:
            out = layer(out, time)
            mid.append(out)

        m = mid.pop()

        for layer in self.d_layer:
            #out = layer(torch.cat([out, mid.pop()], dim=-1), time)
            out = layer(out+mid.pop(), time)

        out = self.predict(out)
        #if self.training:
            #rec_img, _ = self.img_decoder_branch(m)
            #return out, rec_img
        return out

class img_decoder_branch(nn.Module):
    def __init__(self):
        super(img_decoder_branch, self).__init__()
        self.stroke2image = stroke2image(hp.ud_model, hp.ud_ffn, hp.stroke_num)

        self.layers2 = nn.Sequential(
            nn.Conv2d(hp.ud_model, hp.ud_model, 3, stride=1, padding=1),
            nn.InstanceNorm2d(hp.ud_model), nn.GELU(),
            nn.ConvTranspose2d(hp.ud_model, 64, kernel_size=3, padding=1, stride=2, output_padding=1),
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
        return out, out_0

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


if __name__ == '__main__':
    tb = TimeEmbedding(1000,128,128)
    a = torch.zeros([5,48,128])
    print(tb(torch.tensor([[20],[0],[1],[3],[8]])).shape)
    print((tb(torch.tensor([[20],[0],[1],[3],[8]]))+a).shape)
