from Diffusion.Diffuser import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
import os
from typing import Dict
import torch.nn as nn
import numpy as np
import torch
from Utils import save_checkpoint, load_checkpoint
from Networks import First_Stage_Encoder, First_Stage_Decoder
import torch.optim as optim
import random
from tqdm.auto import tqdm
from Utils import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from torch.utils.data import DataLoader
from hyper_params import hp
from Dataset import Dataset


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
train_set = Dataset(mode='train', draw_image=True, noise=False)
dataloader_Train = DataLoader(train_set, batch_size=hp.ubatch_size, shuffle=True, num_workers=64)

print("***********- loading model -*************")
if (len(hp.gpus) == -1):  # cpu
    encoder = First_Stage_Encoder()
    net_model = UNet(T=1000)
    trainer = GaussianDiffusionTrainer(
        net_model, hp.beta0, hp.betaT, 1000)

else:  # multi gpus
    gpus = ','.join(str(i) for i in hp.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    encoder = First_Stage_Encoder().cuda()
    net_model = UNet(T=1000).cuda()
    trainer = GaussianDiffusionTrainer(
        net_model, hp.beta0, hp.betaT, 1000).cuda()

    gpus = [i for i in range(len(hp.gpus))]
    encoder = torch.nn.DataParallel(encoder, device_ids=gpus)
    net_model = torch.nn.DataParallel(net_model, device_ids=gpus)
    trainer = torch.nn.DataParallel(trainer, device_ids=gpus)

e_checkpoint = torch.load('./'+hp.model_save+'/encoder_epoch_'+str(hp.epochs)+'.pkl')['net_state_dict']
encoder.load_state_dict(e_checkpoint)

optimizer = optim.AdamW(net_model.parameters(),
                        lr=hp.ulr, weight_decay=0.01)

scheduler = get_cosine_schedule_with_warmup(optimizer, hp.warmup_step, int(len(dataloader_Train) * hp.uepochs))


class Trainer:
    def __init__(self, trainer, optimizer, scheduler):
        self.trainer = trainer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.iter = 0

    def batch_train(self, batch_data):
        strokes = batch_data["stroke"].cuda()
        start_points = batch_data["start_points"].cuda()
        #noise = batch_data['noise'].cuda()
        #image = batch_data['images'].cuda()

        # mask = torch.zeros([stroke_length.shape[0], hp.stroke_num, 2]).cuda()
        # for i in range(len(stroke_length)):
        # if len(np.where(stroke_length[i]==0)[0])!=0:
        # idx = hp.stroke_num
        # else:
        # idx = np.where(stroke_length[i]==0)[0][0]
        # mask[i, :int(idx), :] = 1
        with torch.no_grad():
            z, mu, sigma = encoder(strokes)
        #diffloss, rec = trainer(start_points, z, noise)
        diffloss = trainer(start_points, z)
        #rec_loss = F.mse_loss(rec, image)
        diff_loss =  diffloss.sum() / z.shape[0] / z.shape[1]
        loss = diff_loss  #+ 0.1 * rec_loss
        return loss, diff_loss#, rec_loss

    def train_epoch(self, loader, epoch):
        net_model.train()
        encoder.eval()
        tqdm_loader = tqdm(loader)

        print("\n************Training*************")
        for batch_idx, (batch) in enumerate(tqdm_loader):
            #loss, diff_loss, rec_loss = self.batch_train(batch)
            loss, diff_loss = self.batch_train(batch)
            # print(predicted.size(),labels.size())

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net_model.parameters(), max_norm=1, norm_type=2)
            self.optimizer.step()
            # self.scheduler.step()

            self.iter = self.iter + 1
            self.scheduler.step()
            #tqdm_loader.set_description('Training: loss:{:.4} diff_loss:{:.4} rec_loss:{:.4} lr:{:.4}'.
                                        #format(loss, diff_loss, rec_loss, self.optimizer.param_groups[0]['lr']))

            tqdm_loader.set_description('Training: loss:{:.4} diff_loss:{:.4} lr:{:.4}'.
                                        format(loss, diff_loss, self.optimizer.param_groups[0]['lr']))

    def run(self, train_loder):
        start_epoch = 0

        for e in range(hp.uepochs):
            e = e + start_epoch + 1
            print("------model:{}----Epoch: {}--------".format("stroke sketch", e))
            self.train_epoch(train_loder, e)

            save_checkpoint(net_model, self.optimizer, e, savepath=hp.model_save, m_name="UNet")


print('''***********- training -*************''')
params_total = sum(p.numel() for p in net_model.parameters())
print("Number of diffusion net parameter: %.2fM" % (params_total / 1e6))

diff = Trainer(trainer, optimizer, scheduler)
diff.run(dataloader_Train)