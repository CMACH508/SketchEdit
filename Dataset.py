import torch
import numpy as np
import os
import torch.utils.data as data
from hyper_params import hp
from Utils import points2stroke, sketch_normalize, points3to5, draw_sketch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import cv2

class Dataset(data.Dataset):
    def __init__(self, mode, draw_image=False, noise=False):
        self.path = hp.data_location
        self.draw_image = draw_image

        self.images = []
        self.sketches = []
        self.strokes = []
        self.stroke_length =[]
        self.labels = []
        self.statr_points = []
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.noise = noise

        for label, cat in enumerate(sorted(hp.category)):
            print(f'loading {cat}, mode {mode}.')
            npz_file = np.load(os.path.join(self.path, cat), allow_pickle=True, encoding='latin1')
            if mode == 'train':
                for file in npz_file['train']:
                    tmp_abs = file.copy()
                    tmp_abs[:, :2] = np.cumsum(file[:, :2], axis=0)
                    t_sketch = sketch_normalize(tmp_abs)
                    tmp_sketch = points3to5(t_sketch)
                    self.sketches.append(tmp_sketch)
                    self.labels.append(label/len(hp.category))

                    sketch_strokes, stroke_length, start_points = points2stroke(tmp_sketch)
                    self.statr_points.append(start_points)

                    self.strokes.append(sketch_strokes)
                    self.stroke_length.append(stroke_length)


            elif mode == 'test':
                for file in npz_file['test']:
                    tmp_abs = file.copy()
                    tmp_abs[:, :2] = np.cumsum(file[:, :2], axis=0)
                    t_sketch = sketch_normalize(tmp_abs)
                    tmp_sketch = points3to5(t_sketch)
                    self.sketches.append(tmp_sketch)
                    self.labels.append(label / len(hp.category))

                    sketch_strokes, stroke_length, start_points = points2stroke(tmp_sketch)
                    self.statr_points.append(start_points)

                    self.strokes.append(sketch_strokes)
                    self.stroke_length.append(stroke_length)


            elif mode == 'valid':
                for file in npz_file['valid']:
                    tmp_abs = file.copy()
                    tmp_abs[:, :2] = np.cumsum(file[:, :2], axis=0)
                    t_sketch = sketch_normalize(tmp_abs)
                    tmp_sketch = points3to5(t_sketch)
                    self.sketches.append(tmp_sketch)
                    self.labels.append(label / len(hp.category))

                    sketch_strokes, stroke_length, start_points = points2stroke(tmp_sketch)
                    self.statr_points.append(start_points)

                    self.strokes.append(sketch_strokes)
                    self.stroke_length.append(stroke_length)


            else:
                print('error dataset type.')

    def __len__(self):
        return len(self.sketches)

    def __getitem__(self, item):
        if self.draw_image:
            if not self.noise:
                image = self.trans(draw_sketch(self.sketches[item][:,:3], size=64, thickness=1))
            else:
                noise = torch.randn_like(torch.tensor(self.statr_points[item], dtype=torch.float32))
                for i in range(hp.stroke_num):
                    tmp = self.strokes[item][i].copy()
                    tmp = tmp.reshape(hp.stroke_length, 5)
                    tmp[:,:2] = tmp[:, :2] + noise[i].numpy()
                    if i ==0:
                        t_img = draw_sketch(tmp[:,:3], size=64, thickness=1)
                    else:
                        t_img = draw_sketch(tmp[:,:3], size=64, thickness=1)+t_img
                t_img = np.clip(t_img, 0, 1)
                image  = self.trans(t_img)

                return {"sketch": torch.tensor(self.sketches[item], dtype=torch.float32),
                        "images": image.to(torch.float32),
                        "noise": noise,
                        # "stroke_images": torch.stack(stroke_image),
                        "stroke": torch.tensor(self.strokes[item], dtype=torch.float32),
                        "start_points": torch.tensor(self.statr_points[item], dtype=torch.float32),
                        'labels': torch.tensor(self.labels[item], dtype=torch.float32),
                        "stroke_length": torch.tensor(self.stroke_length[item], dtype=torch.float32)}

            # stroke_image = []
            # for i in range(hp.stroke_num):
            #     tmp = self.strokes[item][i].copy()
            #     tmp = tmp.reshape(hp.stroke_length, 5)
            #     tmp[:,:2] = tmp[:, :2]/2
            #     t_img = draw_sketch(tmp[:,:3], size=64, thickness=1)
            #     # cv2.imshow("test", t_img)
            #     # cv2.waitKey(1000)
            #     stroke_image.append(self.trans(t_img).to(torch.float32))

            return {"sketch": torch.tensor(self.sketches[item], dtype=torch.float32),
                    "images": image.to(torch.float32),
                    #"stroke_images": torch.stack(stroke_image),
                    "stroke": torch.tensor(self.strokes[item], dtype=torch.float32),
                    "start_points": torch.tensor(self.statr_points[item], dtype=torch.float32),
                    'labels': torch.tensor(self.labels[item], dtype=torch.float32),
                    "stroke_length": torch.tensor(self.stroke_length[item], dtype=torch.float32)}
        else:
            return {"sketch": torch.tensor(self.sketches[item], dtype=torch.float32),
                    "stroke": torch.tensor(self.strokes[item], dtype=torch.float32),
                    "start_points": torch.tensor(self.statr_points[item], dtype=torch.float32),
                    'labels': torch.tensor(self.labels[item], dtype=torch.float32),
                    "stroke_length": torch.tensor(self.stroke_length[item], dtype=torch.float32)}


if __name__ == '__main__':
    ds = Dataset(mode='test', draw_image=True)
    dataloader = DataLoader(ds, batch_size=1)
    idx = 1
    for data in dataloader:
        if idx % 1 == 0:
            if not os.path.exists(f'./stroke/{idx}'):
                os.mkdir(f'./stroke/{idx}')
            for i in range(hp.stroke_num):
                sketch = data["stroke"][0][i].view(-1, 5).numpy()
                sketch[:,:2] = sketch[:,:2]/2
                sketch[:,2] = sketch[:,2]+sketch[:,-1]
                image = draw_sketch(sketch[:,:3], size=96, thickness=1)
                cv2.imwrite(f"./stroke/{idx}/{i}.jpg", image*255)
        idx =idx+1
