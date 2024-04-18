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
dataloader_Test = DataLoader(test_set,batch_size=1, num_workers=32)


print("***********- loading model -*************")
if(len(hp.gpus)==0):#cpu
    encoder = First_Stage_Encoder()
    decoder = First_Stage_Decoder()
# elif(len(hp.gpus)==1):
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(hp.gpus[0])
#     encoder = First_Stage_Encoder().cuda()
#     decoder = First_Stage_Decoder().cuda()
else:#multi gpus
    gpus = ','.join(str(i) for i in hp.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    encoder = First_Stage_Encoder().cuda()
    decoder = First_Stage_Decoder().cuda()
    gpus = [i for i in range(len(hp.gpus))]
    encoder = torch.nn.DataParallel(encoder, device_ids=gpus)
    decoder = torch.nn.DataParallel(decoder, device_ids=gpus)

hp.batch_size = 1
class tester:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.idx = 1
        self.cat_id = 0
        self.cats = sorted(hp.category)

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



    def batch_test(self, batch_data):
        sketches = batch_data["sketch"].cuda()
        return sketches

    def test_epoch(self, loader):
        self.encoder.eval()
        self.decoder.eval()
        tqdm_loader = tqdm(loader)

        print("\n************test*************")
        for batch_idx, (batch) in enumerate(tqdm_loader):
            sketches = self.batch_test(batch)
            sketches = sketches.squeeze(0).cpu().numpy()
            sketches[:,2] = sketches[:,2]+ sketches[:,4]
            gen_image = draw_sketch(sketches[:,:3])

            if not os.path.exists('./GroundTruth/'+self.cats[self.cat_id]):
                os.mkdir('./GroundTruth/'+self.cats[self.cat_id])
            tqdm_loader.set_description(f'drawing {self.cats[self.cat_id]}, {self.idx}')
            path = './GroundTruth/'+self.cats[self.cat_id]+'/'+str(self.idx)+'.jpg'
            cv2.imwrite(path, gen_image*255)

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
            #if i !=0:
                #x_collect.append(x + x_collect[-1])
                #y_collect.append(y + y_collect[-1])
            #else:
            x_collect.append(x)
            y_collect.append(y)

            q = state[i].data.cpu().numpy()
            q = adjust_temp(q)
            q_idx = np.random.choice(3, p=q)  # 抽一个数字
            if q_idx == 0:
                q_collect.append(1)
            elif q_idx == 1:
                q_collect.append(0)
            elif q_idx==2:
                break
        return q_collect, x_collect, y_collect

Tester = tester(encoder, decoder)
Tester.run(test_loder=dataloader_Test )