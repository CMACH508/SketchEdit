from __future__ import absolute_import
from __future__ import print_function
from utils import relative
from utils import witness
import numpy as np
import sys
sys.path.append('../')
from hyper_params import hp
import os
from PIL import Image
from tqdm import tqdm
path_1 = "../diffusion_results/"
path_2 = "../GroundTruth/black/"
sample_num = 300

def rlt(X, L_0=64, gamma=None, i_max=100):
    """
      This function implements Algorithm 1 for one sample of landmarks.

    Args:
      X: np.array representing the dataset.
      L_0: number of landmarks to sample.
      gamma: float, parameter determining the maximum persistence value.
      i_max: int, upper bound on the value of beta_1 to compute.

    Returns
      An array of size (i_max, ) containing RLT(i, 1, X, L)
      for randomly sampled landmarks.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError('X should be a numpy array')
    if len(X.shape) != 2:
        raise ValueError('X should be 2d array, got shape {}'.format(X.shape))
    N = X.shape[0]
    if gamma is None:
        gamma = 1.0 / 128 * N / 5000
    I_1, alpha_max = witness(X, L_0=L_0, gamma=gamma)
    res = relative(I_1, alpha_max, i_max=i_max)
    return res


def rlts(X, L_0=64, gamma=None, i_max=100, n=1000):
    """
      This function implements Algorithm 1.

    Args:
      X: np.array representing the dataset.
      L_0: number of landmarks to sample.
      gamma: float, parameter determining the maximum persistence value.
      i_max: int, upper bound on the value of beta_1 to compute.
      n: int, number of samples
    Returns
      An array of size (n, i_max) containing RLT(i, 1, X, L)
      for n collections of randomly sampled landmarks.
    """
    rlts = np.zeros((n, i_max))
    for i in range(n):
        rlts[i, :] = rlt(X, L_0, gamma, i_max)
        if i % 100 == 0:
            print('Done {}/{}'.format(i, n))
    return rlts


def geom_score(rlts1, rlts2):
    """
      This function implements Algorithm 2.

    Args:
       rlts1 and rlts2: arrays as returned by the function "rlts".
    Returns
       Float, a number representing topological similarity of two datasets.

    """
    mrlt1 = np.mean(rlts1, axis=0)
    mrlt2 = np.mean(rlts2, axis=0)
    return np.sum((mrlt1 - mrlt2) ** 2)

def main():
    imageset_1 = []
    imageset_2 = []

    for i, cat in enumerate(tqdm(hp.category)):
        if not os.path.exists(os.path.join(path_1, cat)):
            cat_tmp = cat.split('.')[0]
            path = os.path.join(path_1, cat_tmp)
        else:
            path = os.path.join(path_1, cat)

        dir_name_list = sorted(os.listdir(path + '/'))
        for name in dir_name_list[:sample_num]:
            img = Image.open(path + '/' + name).convert('1')
            img = img.resize((32, 32))
            imgArray = np.abs(np.array(img)).astype(np.int8).reshape(32*32)
            imageset_1.append(imgArray)
    imageset_1 = np.stack(imageset_1)

    for i, cat in enumerate(tqdm(hp.category)):
        if not os.path.exists(os.path.join(path_2, cat)):
            cat_tmp = cat.split('.')[0]
            path = os.path.join(path_2, cat_tmp)
        else:
            path = os.path.join(path_2, cat)

        dir_name_list = sorted(os.listdir(path + '/'))
        for name in dir_name_list[:sample_num]:
            img = Image.open(path + '/' + name).convert('1')
            img = img.resize((32, 32))

            imgArray = np.abs(np.array(img)).astype(np.int8).reshape(32*32)
            imageset_2.append(imgArray)
    imageset_2 = np.stack(imageset_2)

    rlts_1 = rlts(imageset_1, n=len(hp.category)*sample_num)
    rlts_2 = rlts(imageset_2, n=len(hp.category) * sample_num)
    gs = geom_score(rlts_1, rlts_2)
    print(f"Geometric score: {gs}")

main()
