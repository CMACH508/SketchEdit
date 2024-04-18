import numpy as np
from hyper_params import hp
import cv2
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch
import math
from hyper_params import hp

def points3to5(sketch):
    n_sketch = np.zeros([hp.max_seq_length, 5])
    n_sketch[:len(sketch), :3] = sketch.copy()
    n_sketch[:len(sketch), 3] = 1-n_sketch[:len(sketch), 2]
    n_sketch[len(sketch):, 4] = 1
    return n_sketch

def points2stroke(sketch, points_ps_num=hp.stroke_length, stroke_num=hp.stroke_num):
    '''
    :param sketch:
    :param points_ps_num: points in per stroke hyperparameters
    :param stroke_num: strokes in per sketch   hyperparameters
    :return: start_points->(max_stroke_num, 2)
    '''
    points_per_stroke_th = hp.stroke_length
    strokes_per_sketch_th = hp.stroke_num

    stroke_idx = np.where(sketch[:,2]>0)[0]
    stroke_idx_copy = np.concatenate([[-1], stroke_idx[:-1]],axis=0)
    stroke_length = stroke_idx - stroke_idx_copy

    n_sketch = np.zeros([stroke_num, points_ps_num*5]).astype(np.float32)
    n_stroke_length = np.zeros([strokes_per_sketch_th]).astype(np.float32)
    start_points = np.zeros([strokes_per_sketch_th, 2]).astype(np.float32)

    start = 0

    # only one stroke
    if len(stroke_idx) < 2:
        if len(sketch)>points_per_stroke_th:
            start_points[0] = sketch[0, :2].copy().astype(np.float32)
            tsketch = sketch.copy().astype(np.float32)
            tsketch[:,:2] = tsketch[:,:2] - start_points[0]
            n_sketch[0,:points_per_stroke_th*5] = tsketch[:points_per_stroke_th,:].reshape(-1)
            stroke_length[0] = points_per_stroke_th
        else:
            start_points[0] = sketch[0, :2].copy().astype(np.float32)
            tsketch = sketch.copy().astype(np.float32)
            tsketch[:, :2] = tsketch[:, :2] - start_points[0]
            n_sketch[0,:len(sketch)*5] = tsketch.reshape(-1)

        n_stroke_length[:len(stroke_length)] = stroke_length
        return n_sketch, n_stroke_length, start_points

    for i, end in enumerate(stroke_idx):
        if i >= strokes_per_sketch_th:
            break
        end =end+1
        if stroke_length[i] > points_per_stroke_th:
            start_points[i] = sketch[start, :2].copy().astype(np.float32)
            tsketch = sketch.copy().astype(np.float32)
            tsketch[start:start+points_per_stroke_th, :2] = tsketch[start:start+points_per_stroke_th, :2] - start_points[i]
            n_sketch[i:i+1,:points_per_stroke_th*5] = tsketch[start:start+points_per_stroke_th,:].reshape(1,-1)
            stroke_length[i] = points_per_stroke_th
        else:
            start_points[i] = sketch[start, :2].copy().astype(np.float32)
            tsketch = sketch.copy().astype(np.float32)
            tsketch[start:end, :2] = tsketch[start:end, :2] - start_points[i]
            n_sketch[i:i+1, :stroke_length[i]*5] = tsketch[start:end,:].reshape(1,-1)
        start = end
    if len(stroke_length)> strokes_per_sketch_th:
        n_stroke_length = stroke_length[: strokes_per_sketch_th]
    else:
        n_stroke_length[:len(stroke_length)] = stroke_length
    return n_sketch, n_stroke_length, start_points


def sketch_normalize(sketch):
    '''

    :param sketch: (points, abs x, abs y, state)
    :return: normalize(sketch)
    '''
    n_sketch = sketch.copy().astype(np.float32)
    w_max = np.max(n_sketch[:,0], axis=0)
    w_min = np.min(n_sketch[:,0], axis=0)

    h_max = np.max(n_sketch[:,1], axis=0)
    h_min = np.min(n_sketch[:,1], axis=0)

    scale_w = w_max - w_min
    scale_h = h_max - h_min

    scale = scale_w if scale_w> scale_h else scale_h

    n_sketch[:,0] = (n_sketch[:,0] - w_min)/scale *2 -1
    n_sketch[:,1] = (n_sketch[:,1] - h_min)/scale *2 -1
    return n_sketch

def draw_sketch(sketch, size=256, thickness=6):
    '''

    :param sketch: (points, abs x, abs y, state) normalized
    :return: canvas
    '''
    n_sketch = sketch.copy().astype(np.float32)

    canvas = np.zeros([size+2*thickness,size+2*thickness,3])

    n_sketch[:,:2] = (n_sketch[:,:2]+1)/2*size
    n_sketch[:,:2] = n_sketch[:,:2]+thickness+1
    x1 = n_sketch[0,0]
    y1 = n_sketch[0,1]
    pri_state = n_sketch[0,2]

    for (x2,y2,curr_state) in n_sketch[1:,:]:
        if curr_state !=1 and curr_state!=0:
            break
        if pri_state == 1:
            x1 = x2
            y1 = y2
            pri_state = curr_state
            continue
        cv2.line(canvas,(int(x1), int(y1)),(int(x2), int(y2)), color=(1,1,1), thickness=thickness)

        x1 = x2
        y1 = y2
        pri_state = curr_state

    return  cv2.resize(canvas, (size, size))


def save_checkpoint(net=None, optimizer=None, epoch=None, train_losses=None, train_acc=None, val_loss=None,
                    val_acc=None, check_loss=None, savepath=None, m_name=None, GPUdevices=1):
    if GPUdevices > 1:
        net_weights = net.module.state_dict()
    else:
        net_weights = net.state_dict()
    save_json = {
        'net_state_dict': net_weights,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }

    savepath = savepath + '/{}_epoch_{}.pkl'.format(m_name, epoch)
    torch.save(save_json, savepath)
    print("checkpoint of {}th epoch saved at {}".format(epoch, savepath))



def load_checkpoint(model=None, optimizer=None, checkpoint_path=None, losses_flag=None):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['net_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    if not losses_flag:
        return model, optimizer, start_epoch
    else:
        losses = checkpoint['train_losses']
        return model, optimizer, start_epoch, losses


def off2abs(tvector, img_size=256):
    off_vector = tvector.copy()
    new_sketch = np.zeros_like(off_vector)
    new_sketch[:, 2] = off_vector[:, 2]
    a = off_vector[:, :2]
    h_s = np.max(a[:,0])-np.min(a[:,0])
    w_s = np.max(a[:,1])-np.min(a[:,1])
    if h_s>w_s:
        scale = h_s
    else:
        scale = w_s
    new_sketch[:,0] = (a[:,0]-np.min(a[:,0]))/scale*img_size
    new_sketch[:,1] = (a[:,1]-np.min(a[:,1]))/scale*img_size
    return new_sketch


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5,
        last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
