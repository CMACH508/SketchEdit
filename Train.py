import os
from Dataset import Dataset
import torch
import numpy as np
import torch.nn as nn
from Utils import save_checkpoint, load_checkpoint
from Networks import First_Stage_Encoder, First_Stage_Decoder
from hyper_params import hp
import torch.optim as optim
import random
from tqdm.auto import tqdm
from Utils import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable


def seed_torch(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()

print("***********- ***********- READ DATA and processing-*************")
train_set = Dataset(mode='train', draw_image=True)
dataloader_Train = DataLoader(train_set, batch_size=hp.batch_size, shuffle=True, num_workers=32)

print("***********- loading model -*************")
if (len(hp.gpus) == 0):  # cpu
    encoder = First_Stage_Encoder()
    decoder = First_Stage_Decoder()
elif (len(hp.gpus) == 1):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(hp.gpus[0])
    encoder = First_Stage_Encoder().cuda()
    decoder = First_Stage_Decoder().cuda()
else:  # multi gpus
    gpus = ','.join(str(i) for i in hp.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    encoder = First_Stage_Encoder().cuda()
    decoder = First_Stage_Decoder().cuda()
    gpus = [i for i in range(len(hp.gpus))]
    encoder = torch.nn.DataParallel(encoder, device_ids=gpus)
    decoder = torch.nn.DataParallel(decoder, device_ids=gpus)

optimizer = optim.AdamW([
    {'params': encoder.parameters()},
    {'params': decoder.parameters(), }],
    lr=hp.lr, weight_decay=0.01)

scheduler = get_cosine_schedule_with_warmup(optimizer, hp.warmup_step, int(len(dataloader_Train) * hp.epochs))


class trainer:
    def __init__(self, encoder, decoder, optimizer, scheduler):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.iter = 0

    def batch_train(self, batch_data):
        sketches = batch_data["sketch"].cuda()
        strokes = batch_data["stroke"].cuda()
        start_points = batch_data["start_points"].cuda()
        image = batch_data["images"].cuda()
        #stroke_image = batch_data["stroke_images"].cuda()
        stroke_length = batch_data['stroke_length'].cuda()

        z, mu, sigma = self.encoder(strokes)
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, rec_image = self.decoder(z, start_points)

        loss, pt_loss, kl_loss = self.myloss(sketches, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q,
                                              stroke_length, mu, sigma)
        alpha_loss, gaussianloss, data = self.RPCLLoss(z, mu, sigma)

        rec_loss = F.mse_loss(image, rec_image)
        wkl = hp.wKL/4000 * self.iter if self.iter<4000 else hp.wKL
        loss = loss + rec_loss + wkl * (alpha_loss + gaussianloss)
        return loss, pt_loss, kl_loss, alpha_loss, gaussianloss, rec_loss, data

    def train_epoch(self, loader, epoch):
        self.encoder.train()
        self.decoder.train()
        tqdm_loader = tqdm(loader)

        print("\n************Training*************")
        for batch_idx, (batch) in enumerate(tqdm_loader):
            loss, pt_loss, kl_loss, alpha_loss, gaussianloss, rec_loss, reset_data = self.batch_train(batch)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1, norm_type=2)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1, norm_type=2)
            self.optimizer.step()

            self.iter = self.iter + 1
            self.scheduler.step()
            self.encoder.module.assign(reset_data)
            tqdm_loader.set_description(
                'Training: loss:{:.4} lr:{:.4} pt_loss:{:.4} kl_loss:{:.4} alpha_loss:{:.4} gaussian_loss:{:.4}'
                ' image_loss:{:.4}'.
                format(loss, self.optimizer.param_groups[0]['lr'], pt_loss, kl_loss, alpha_loss,
                       gaussianloss,
                       rec_loss))

    def bivariate_normal_pdf(self, dx, dy, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q):
        z_x = ((dx - mu_x) / sigma_x) ** 2
        z_y = ((dy - mu_y) / sigma_y) ** 2
        z_xy = (dx - mu_x) * (dy - mu_y) / (sigma_x * sigma_y)
        z = z_x + z_y - 2 * rho_xy * z_xy
        exp = torch.exp(-z / (2 * (1 - rho_xy ** 2)))
        norm = 2 * np.pi * sigma_x * sigma_y * torch.sqrt(1 - rho_xy ** 2)
        return exp / (norm+1e-6)

    def myloss(self, o_sketch, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q,
               stroke_length, mu, sigma):
        #o_sketch[:, 1:, :1] = (o_sketch[:, 1:, :1] - o_sketch[:, :-1, :1]) * (1 - o_sketch[:, 1:, 4:5])
        #o_sketch[:, 1:, :2] = (o_sketch[:, 1:, 1:2] - o_sketch[:, :-1, 1:2]) * (1 - o_sketch[:, 1:, 4:5])

        pdf = self.bivariate_normal_pdf(o_sketch[:, :, :1], o_sketch[:, :, 1:2],
                                        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q)
        mask = torch.zeros([o_sketch.shape[0], o_sketch.shape[1]]).cuda()
        for i in range(len(o_sketch)):
            idx = stroke_length[i].sum(-1).cpu()
            mask[i, :int(idx)] = 1

        pt_loss = -torch.sum(mask * torch.log(1e-3 + torch.sum(pi * pdf, 2)))  \
                   / float(hp.max_seq_length * o_sketch.shape[0])

        kl_loss = -torch.sum((o_sketch[:, :, 2:]) * torch.log(1e-3 + q.view(o_sketch.shape[0], -1, 3))) \
                  / float(hp.max_seq_length * o_sketch.shape[0])

        return pt_loss + kl_loss, pt_loss, kl_loss

    def RPCLLoss(self, z, mu, sigmae):
        self.z = z.view(-1, hp.d_model)
        self.p_mu = mu.view(-1, hp.d_model)
        self.p_sigma2 = (sigmae.view(-1, hp.d_model)) ** 2
        self.p_alpha, self.gau_label = self.encoder.module.calculate_posterior(self.z, self.encoder.module.de_mu,
                                                                               self.encoder.module.de_sigma2,
                                                                               self.encoder.module.de_alpha)
        if self.iter > 2000:
            if (self.iter % 5) == 0:
                q_mu, q_sigma2, q_alpha = self.encoder.module.em(self.p_mu, self.p_sigma2, self.p_alpha,
                                                                 self.encoder.module.de_mu,
                                                                 self.encoder.module.de_sigma2,
                                                                 self.encoder.module.de_alpha)
            else:
                q_mu, q_sigma2, q_alpha = self.encoder.module.rpcl(self.p_mu, self.p_sigma2, self.p_alpha,
                                                                   self.encoder.module.de_mu,
                                                                   self.encoder.module.de_sigma2,
                                                                   self.encoder.module.de_alpha)
        else:

            q_mu, q_sigma2, q_alpha = self.encoder.module.em(self.p_mu, self.p_sigma2, self.p_alpha,
                                                             self.encoder.module.de_mu,
                                                             self.encoder.module.de_sigma2,
                                                             self.encoder.module.de_alpha)

        alpha_loss = self.encoder.module.get_alpha_loss(self.p_alpha, Variable(q_alpha.data, requires_grad=False))

        gaussian_loss = self.encoder.module.get_gaussian_loss(self.p_alpha, self.p_mu, self.p_sigma2,
                                                              Variable(q_mu.data, requires_grad=False),
                                                              Variable(q_sigma2.data, requires_grad=False))

        return alpha_loss, gaussian_loss, (q_mu, q_sigma2, q_alpha)

    def run(self, train_loder):
        start_epoch = 0

        for e in range(hp.epochs):
            e = e + start_epoch + 1
            print("------model:{}----Epoch: {}--------".format("stroke sketch", e))
            self.train_epoch(train_loder, e)

            save_checkpoint(self.encoder, self.optimizer, e, savepath=hp.model_save, m_name="encoder")
            save_checkpoint(self.decoder, self.optimizer, e, savepath=hp.model_save, m_name="decoder")


print('''***********- training -*************''')
params_total = sum(p.numel() for p in encoder.parameters())
print("Number of encoder parameter: %.2fM" % (params_total / 1e6))

params_total = sum(p.numel() for p in decoder.parameters())
print("Number of decoder parameter: %.2fM" % (params_total / 1e6))

Trainer = trainer(encoder, decoder, optimizer, scheduler)
Trainer.run(dataloader_Train)

