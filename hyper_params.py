import torch
import os

class HParams:
    def __init__(self):
        self.data_location = './dataset/'#location of  of origin data
        self.category = ["airplane.npz", "angel.npz", "alarm clock.npz", "apple.npz",
                         "butterfly.npz", "belt.npz", "bus.npz",
                         "cake.npz", "cat.npz", "clock.npz", "eye.npz", "fish.npz",
                         "pig.npz", "sheep.npz", "spider.npz", "The Great Wall of China.npz",
                         "umbrella.npz"]
        #self.category = ["airplane.npz"]
        self.model_save = "./model_save"
        if not os.path.exists(self.model_save):
            os.mkdir(self.model_save)
        self.gpus=[0, 1,2,3, 4]

        self.k = 40
        self.M = 20

        self.stroke_num = 25
        self.stroke_length = 96

        self.d_model = 128
        self.d_ffn = self.d_model*4

        self.ud_model = 96
        self.ud_ffn = self.ud_model*4


        self.dropath = 0.1
        self.batch_size = 200 
        self.ubatch_size = 768

        self.warmup_step = 1000
        self.epochs = 15
        self.uepochs = 40

        self.eta_min = 0.01
        self.wKL = 0.0001
        self.lr = 0.002 
        self.ulr = 5e-4

        self.beta0 = 1e-4
        self.betaT = 0.02

        self.min_lr = 0.00001
        self.temperature = 0.001

        self.ddim_step = 60

        self.max_seq_length = 180
        self.min_seq_length = 0


hp = HParams()
