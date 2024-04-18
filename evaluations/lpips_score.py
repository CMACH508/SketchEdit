import lpips
from PIL import Image
import numpy as np
import torch
import sys
sys.path.append('../')
from hyper_params import hp
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='lpips score')
parser.add_argument('--path1')
parser.add_argument('--path2')

args = parser.parse_args()

#path_1 = "../DDIMeta0CLIP/diffusion_results30/"
path_1 = args.path1
path_2 = args.path2

def img2tensor(img):
    img = (img/255.).astype('float32')
    if img.ndim ==2:
        img = np.expand_dims(np.expand_dims(img, axis = 0),axis=0)
    else:
        img = np.transpose(img, (2, 0, 1))  # C, H, W
        img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img, dtype=np.float32)
    tensor = torch.from_numpy(img)
    return tensor

loss_fn_alex = lpips.LPIPS(net='alex')

def main():
    score = 0
    for i, cat in enumerate(hp.category):
        if not os.path.exists(os.path.join(path_1, cat)):
            cat_tmp = cat.split('.')[0]
            path1 = os.path.join(path_1, cat_tmp)
        else:
            path1 = os.path.join(path_1, cat)

        if not os.path.exists(os.path.join(path_2, cat)):
            cat_tmp = cat.split('.')[0]
            path2 = os.path.join(path_2, cat_tmp)
        else:
            path2 = os.path.join(path_2, cat)

        dir_name_list1 = sorted(os.listdir(path1 + '/'))
        dir_name_list2 = sorted(os.listdir(path2 + '/'))

        for i in tqdm(range(len(dir_name_list1))):
            name1 = dir_name_list1[i]
            name2 = dir_name_list2[i]

            img1 = Image.open(path1 + '/' + name1)
            img1 = np.array(img1)
            img1 = img2tensor(img1)

            img2 = Image.open(path2+'/'+name2)
            img2 = np.array(img2)
            img2 = img2tensor(img2)

            score = loss_fn_alex(img1, img2)+score
    print(f"LPIPS SCORE: {score/(len(hp.category) * 2500)}")

main()
