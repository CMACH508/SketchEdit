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

if not os.path.exists("./stroke"):
    os.mkdir("./stroke")
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
dataloader_Test = DataLoader(test_set,batch_size=1, num_workers=24)


print("***********- loading model -*************")
if(len(hp.gpus)==0):#cpu
    encoder = First_Stage_Encoder()
    decoder = First_Stage_Decoder()
else:#multi gpus
    gpus = ','.join(str(i) for i in hp.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    encoder = First_Stage_Encoder().cuda()
    decoder = First_Stage_Decoder().cuda()
    gpus = [i for i in range(len(hp.gpus))]
    encoder = torch.nn.DataParallel(encoder, device_ids=gpus)
    decoder = torch.nn.DataParallel(decoder, device_ids=gpus)

e_checkpoint = torch.load('./'+hp.model_save+'/encoder_epoch_'+str(hp.epochs)+'.pkl')['net_state_dict']
encoder.load_state_dict(e_checkpoint)
d_checkpoint = torch.load('./'+hp.model_save+'/decoder_epoch_'+str(hp.epochs)+'.pkl')['net_state_dict']
decoder.load_state_dict(d_checkpoint)

hp.batch_size = 1
class tester:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.idx = 1
        self.cat_id = 0
        self.cats = sorted(hp.category)

    def batch_test(self, batch_data):
        sketches = batch_data["sketch"].cuda()
        strokes = batch_data["stroke"].cuda()
        start_points = batch_data["start_points"].cuda()
        labels = batch_data["labels"].cuda()
        stroke_length = batch_data['stroke_length'].cuda()
        #stroke_image = batch_data["stroke_images"].cuda()
        with torch.no_grad():
            z, mu, sigma = self.encoder(strokes)
        return z

    def test_epoch(self, loader):
        self.encoder.eval()
        self.decoder.eval()
        tqdm_loader = tqdm(loader)

        print("\n************test*************")
        for batch_idx, (batch) in enumerate(tqdm_loader):
            if self.idx % 1 == 0:
                if not os.path.exists(f'./stroke/{self.idx}'):
                    os.mkdir(f'./stroke/{self.idx}')
                mu = self.batch_test(batch)
                for i in range(hp.stroke_num):
                    emb = mu[0][i].cpu().numpy()
                    np.save(f"./stroke/{self.idx}/{i}.npy", emb)

            if self.idx % 2500 == 0:
                self.cat_id = self.cat_id+1
            self.idx = self.idx+1
            
        print("\n************record mu*************")
        for i in range(hp.k):
            emb = self.encoder.module.de_mu.data[i].cpu().numpy()
            if not os.path.exists("./stroke/mu"):
                os.mkdir("./stroke/mu")
            np.save(f"./stroke/mu/{i}.npy", emb)


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
        for i in range(len(state)):
            # get pen state:
            q = state[i].data.cpu().numpy()
            q = adjust_temp(q)
            q_idx = np.random.choice(3, p=q) 
            if q_idx == 0:
                q_collect.append(1)
            elif q_idx == 1:
                q_collect.append(0)
            elif q_idx==2:
                break
        return q_collect


Tester = tester(encoder, decoder)
Tester.run(test_loder=dataloader_Test )
