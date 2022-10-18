import torch
import cv2
from einops import rearrange
import numpy as np
from matplotlib import pyplot as plt

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

heatmap = torch.load('wzy/heatmap.pt')
image = torch.load('wzy/image.pt')

idx = 11
heatmap = rearrange(heatmap[idx], 'c w h -> w h c')
image = rearrange(image[idx], 'c w h -> w h c')

image = image.cpu().numpy()
heatmap = heatmap.cpu().numpy()

# 
def _normalization(img):
    _range = np.max(img) - np.min(img)
    return (img - np.min(img)) /_range

image = _normalization(image)
heatmap = _normalization(heatmap)

# 将原图转为灰度图，便于与热图叠加
gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# 叠加多个关节点的热图
feat_map = np.sum(heatmap, axis=2)
# 上采样四倍 64 --> 256
feat_map = cv2.pyrUp(feat_map)
feat_map = cv2.pyrUp(feat_map)


print('heatmap generate begin!')
plt.figure()
fig, ax = plt.subplots(5, 4, figsize=(50, 50), dpi=250)
heatmap = rearrange(heatmap,'w h c -> c w h')

# 原图、热图、灰度图、全部叠加图
ax[0, 0].imshow(image)
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])

ax[0, 1].imshow(feat_map)
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])

ax[0, 2].imshow(gray_img)
ax[0, 2].set_xticks([])
ax[0, 2].set_yticks([])

ax[0, 3].imshow(feat_map + gray_img)
ax[0, 3].set_xticks([])
ax[0, 3].set_yticks([])

for i in range(16):
    row = (i+4) // 4
    col = (i+1) % 4
    plt.axis('off')
    h = heatmap[i]
    h = cv2.pyrUp(h)
    h = cv2.pyrUp(h)
    ax[row, col].imshow(h + gray_img)
    ax[row, col].set_xticks([])
    ax[row, col].set_yticks([])

plt.savefig(f'wzy/all_in_one_{idx}.jpg')
print('heatmap generate over!')