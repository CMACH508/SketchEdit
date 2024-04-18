import os
from Dataset import Dataset
import torch
import numpy as np
import torch.nn as nn
from Utils import save_checkpoint, load_checkpoint, draw_sketch
from Networks import First_Stage_Encoder, First_Stage_Decoder
from hyper_params import hp
import torch.optim as optim
import random
from tqdm.auto import tqdm
from Utils import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
from Diffusion.Diffuser import GaussianDiffusionSampler, GaussianDiffusionTrainer, DDIM
from Diffusion.Model import UNet

fre = 1

def seed_torch(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

print("***********- ***********- READ DATA and processing-*************")
test_set = Dataset(mode='test', draw_image=True)
dataloader_Test = DataLoader(test_set, batch_size=1, num_workers=8)


print("***********- loading model -*************")
if(len(hp.gpus)==0):#cpu
    encoder = First_Stage_Encoder()
    decoder = First_Stage_Decoder()
    net_model = UNet(T=1000)
    #sampler = GaussianDiffusionSampler(net_model, hp.beta0, hp.betaT, 1000)
    sampler = DDIM(net_model, hp.beta0, hp.betaT, 1000)
# elif(len(hp.gpus)==1):
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(hp.gpus[0])
#     encoder = First_Stage_Encoder().cuda()
#     decoder = First_Stage_Decoder().cuda()
else:#multi gpus
    gpus = ','.join(str(i) for i in hp.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    encoder = First_Stage_Encoder().cuda()
    decoder = First_Stage_Decoder().cuda()
    net_model = UNet(T=1000).cuda()
    #sampler = GaussianDiffusionSampler(net_model,hp.beta0, hp.betaT, 1000).cuda()
    sampler = DDIM(net_model, hp.beta0, hp.betaT, 1000).cuda()

    gpus = [i for i in range(len(hp.gpus))]
    encoder = torch.nn.DataParallel(encoder, device_ids=gpus)
    decoder = torch.nn.DataParallel(decoder, device_ids=gpus)
    net_model = torch.nn.DataParallel(net_model, device_ids=gpus)
    sampler = torch.nn.DataParallel(sampler, device_ids=gpus)

e_checkpoint = torch.load('./'+hp.model_save+'/encoder_epoch_'+str(hp.epochs)+'.pkl')['net_state_dict']
encoder.load_state_dict(e_checkpoint)
d_checkpoint = torch.load('./'+hp.model_save+'/decoder_epoch_'+str(hp.epochs)+'.pkl')['net_state_dict']
decoder.load_state_dict(d_checkpoint)
u_checkpoint = torch.load('./'+hp.model_save+'/UNet_epoch_'+str(hp.uepochs)+'.pkl')['net_state_dict']
net_model.load_state_dict(u_checkpoint)
hp.batch_size = 1

class tester:
    def __init__(self, encoder, decoder, net_model):
        self.encoder = encoder
        self.decoder = decoder
        self.net_model = net_model
        self.idx = 1
        self.cat_id = 0
        self.cats = sorted(hp.category)

    def batch_test(self, batch_data):
        sketches = batch_data["sketch"].cuda()
        strokes = batch_data["stroke"].cuda()
        start_points = batch_data["start_points"].cuda()
        labels = batch_data["labels"].cuda()
        stroke_length = batch_data['stroke_length'].cuda()

        z, mu, sigma = self.encoder(strokes)

        noisyImage = torch.randn(
            size=start_points.shape, dtype=torch.float32).to(stroke_length.device)

        d_start_points = sampler(noisyImage, z)

        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _ = self.decoder(z, d_start_points)
        return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q

    def test_epoch(self, loader):
        self.encoder.eval()
        self.decoder.eval()
        self.net_model.eval()
        sampler.eval()
        tqdm_loader = tqdm(loader)

        print("\n************test*************")
        for batch_idx, (batch) in enumerate(tqdm_loader):
            if self.idx % fre ==0:
                self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho_xy, self.q, = self.batch_test(batch)

                p = self.q.squeeze(0)
                state, x, y = self.sample_state(p)
                state = np.asarray(state).astype(np.float32).reshape(-1, 1)

                x = np.asarray(x).astype(np.float32).reshape(-1, 1)
                y = np.asarray(y).astype(np.float32).reshape(-1, 1)
                l = len(state)
                gen_sketch = np.concatenate([x[:l, :], y[:l, :], state], axis=-1)
                gen_image = draw_sketch(gen_sketch)

            if not os.path.exists('./diffusion_results/'+self.cats[self.cat_id]):
                os.mkdir('./diffusion_results/'+self.cats[self.cat_id])
            tqdm_loader.set_description(f'drawing {self.cats[self.cat_id]}, {self.idx}')
            if self.idx % fre == 0:
                path = './diffusion_results/' + self.cats[self.cat_id] + '/' + str(self.idx) + '.jpg'
                cv2.imwrite(path, gen_image * 255)

           # if not os.path.exists('./diffusion_results'+str(hp.ddim_step)+'/'+self.cats[self.cat_id]):
             #   os.mkdir('./diffusion_results'+str(hp.ddim_step)+'/'+self.cats[self.cat_id])
           # tqdm_loader.set_description(f'drawing {self.cats[self.cat_id]}, {self.idx}')
           # if self.idx % fre == 0:
                #path = './diffusion_results'+str(hp.ddim_step)+'/' + self.cats[self.cat_id] + '/' + str(self.idx) + '.jpg'
               # cv2.imwrite(path, gen_image * 255)

            if self.idx % 2500 == 0:
                self.cat_id = self.cat_id+1
            self.idx = self.idx+1


    def run(self, test_loder):
        print("start inference")
        self.test_epoch(test_loder)

    def sample_state(self, state):
        def adjust_temp(pi_pdf):
            """
            SoftMax
            """
            pi_pdf = np.log(pi_pdf) / hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        q_collect = []
        x_collect = []
        y_collect = []
        for i in range(len(state)):
            # get pen state:
            pi = self.pi.data[0, i, :].cpu().numpy()
            pi = adjust_temp(pi)
            pi_idx = np.random.choice(hp.M, p=pi)  # 抽一个数字
            # get mixture params:
            mu_x = self.mu_x.data[0, i, pi_idx]
            mu_y = self.mu_y.data[0, i, pi_idx]
            sigma_x = self.sigma_x.data[0, i, pi_idx]
            sigma_y = self.sigma_y.data[0, i, pi_idx]
            rho_xy = self.rho_xy.data[0, i, pi_idx]
            x, y = self.sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, greedy=False)  # get samples.
            x_collect.append(x)
            y_collect.append(y)

            q = state[i].data.cpu().numpy()
            q = adjust_temp(q)
            q_idx = np.random.choice(3, p=q)  # 抽一个数字
            if q_idx == 0:
                q_collect.append(1)
            elif q_idx == 1:
                q_collect.append(0)
            elif q_idx == 2:
                break
        return q_collect, x_collect, y_collect
    
    def sample_bivariate_normal(self, mu_x: torch.Tensor, mu_y: torch.Tensor,
                                sigma_x: torch.Tensor, sigma_y: torch.Tensor,
                                rho_xy: torch.Tensor, greedy=False):
        mu_x = mu_x.item()
        mu_y = mu_y.item()
        sigma_x = sigma_x.item()
        sigma_y = sigma_y.item()
        rho_xy = rho_xy.item()
        # inputs must be floats
        if greedy:
            return mu_x, mu_y
        mean = [mu_x, mu_y]
        sigma_x *= np.sqrt(hp.temperature)
        sigma_y *= np.sqrt(hp.temperature)

        cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
               [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]


Tester = tester(encoder, decoder, net_model)
Tester.run(test_loder=dataloader_Test )