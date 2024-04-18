import matplotlib.pyplot as PLT
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.image as read_png
from matplotlib.artist import Artist
import os
import numpy as np
from PIL import  Image
from sklearn.manifold import TSNE
fre = 50
n_stroke = 15


idx =1
img_feat = []
for a in range(2500*5):
    if idx % fre == 0:
        for i in range(n_stroke):
            img_feat.append(np.load(f'./stroke/{idx}/{i}.npy'))
    idx = idx+1
imag_feat = np.asarray(img_feat)

tsne = TSNE(n_components=2, random_state=0, perplexity=50, learning_rate=200, init='pca')
res = tsne.fit_transform(img_feat)
tx,ty = res[:,0],res[:,1]
tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

bk_width = 2400
bk_height = 1800
max_dim = 128

back = Image.new('RGB',(bk_width,bk_height),(255,255,255))

iidx = 1
count = 0
for b in range(2500*5):
    if iidx % fre == 0:
        for i in range(n_stroke):
            img_path = f'./stroke/{iidx}/{i}.jpg'
            img = Image.open(img_path).convert("RGBA")
            # for x in range(img.width):
            #     for y in range(img.height):
            #         r, g, b, a = img.getpixel((x, y))
            #         a = int(a * 0.5)
            #         img.putpixel((x, y), (r, g, b, a))

            rs = max(1, img.width / max_dim, img.height / max_dim)
            #img = img.resize((int(img.width / rs), int(img.height / rs)), Image.ANTIALIAS)
            img = img.resize((36, 36))
            back.paste(img, (int((bk_width - max_dim) * tx[count]), int((bk_height - max_dim) * ty[count])), img)
            count = count+1

    iidx = iidx+1



back.save('res_table.png')