# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/24 15:39 
# @Author : wzy 
# @File : visual.py
# ---------------------------------------
import math

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from data import data_load, transform, data_to_tensor
from separate_num import separate
import matplotlib
import seaborn as sns
from einops import rearrange
from img_patch import patchify, masking

matplotlib.use('Agg')  # 加上这句话plt.show就会报错。作用是控制绘图不显示


def vis_filter(model_weights, layer):
    """
    :param model_weights: 传入整个模型的权重
    :param layer: 选择可视化哪一层
    :return:
    """
    print('p====================卷积核可视化====================q')
    filter_num = model_weights[layer].shape[0]
    filter_channel = model_weights[layer].shape[1]
    print(model_weights[layer].shape)
    print(f'该层一共有[{filter_num}]个卷积核,每个卷积核的维度为[{filter_channel}]')
    row, column = separate(filter_num)
    for j in range(filter_channel):
        plt.figure(figsize=(5, 5))
        for i, filter in enumerate(model_weights[layer]):
            plt.subplot(row, column, i + 1)
            plt.imshow(filter[j, :, :].detach(), cmap='gray')
            plt.axis('off')
        plt.savefig(f'./imgs_out/filter_layer{layer + 1}_channel{j}.png')
        plt.show()
    print('b==================卷积核可视化结束===================d')


def vis_image(image):
    h, w = image.shape[0], image.shape[1]
    plt.subplots(figsize=(w * 0.01, h * 0.01))
    plt.imshow(image, alpha=1)
    # plt.axis('off')
    # plt.show()
    return h, w


def vis_feature(features, num_layers):
    """
    :param features: 特征图(即图像经过卷积后的样子)
    :param num_layers:所有的特征图层数
    :return:
    """
    print('p====================特征图可视化====================q')
    for num_layer in range(len(features)):
        plt.figure(figsize=(5, 5))  # 5*5有点小了，建议改大
        layer_vis = features[num_layer][0, :, :, :]
        layer_vis = layer_vis.data
        print(f'[{num_layer + 1}] feature size :{layer_vis.size()}')

        # 得到特征图通道数，也就是后面特征图的数量
        feature_num = layer_vis.shape[0]
        row, column = separate(feature_num)

        for i, filter in enumerate(layer_vis):
            plt.subplot(row, column, i + 1)
            # plt.imshow(filter, cmap='gray')  # 灰度图
            plt.imshow(filter)
            plt.axis('off')
        print(f'Saving layer feature maps : [{num_layer + 1}]/[{num_layers}] ')
        plt.savefig(f'./imgs_out/layer_{num_layer}.png')
        plt.close()
    print('b==================特征图可视化结束===================d')


def vis_attention_matrix(attention_map, index=0, cmap="YlGnBu"):
    """
    :param attention_map: 注意力得分矩阵
    :param index: map编号,便于多个注意力可视化的存储
    :param cmap: 颜色样式
    :return:
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        attention_map,
        vmin=0.0, vmax=1.0,
        cmap=cmap,
        # annot=True,  # 每个格子上显示数据
        square=True)
    plt.savefig(f'./imgs_out/attention_matrix_{index}.png')
    print(f'[attention_matrix_{index}.png] is generated')


def vis_grid_attention(img_path, attention_map, cmap='jet'):
    """
    :param img_path:图像路径
    :param attention_map:注意力图
    :param cmap: cmap是图像的颜色类型，有很多预设的颜色类型
    :return:
    """
    # draw the img
    img = data_load(img_path)
    h, w = vis_image(img)

    # draw the attention
    map = cv2.resize(attention_map, (w, h))
    normed_map = map / map.max()
    normed_map = (normed_map * 255).astype('uint8')
    plt.imshow(normed_map, alpha=0.4, interpolation='nearest', cmap=cmap)  # alpha值决定图像的透明度,0为透明,1不透明

    # 去掉图片周边白边
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # 调整图像与画布的边距(此时填充满)
    plt.margins(0, 0)

    # 保存图像,以300dpi
    img_name = img_path.split('/')[-1].split('.')[0] + "_with_attention.jpg"
    plt.savefig(f'./imgs_out/{img_name}', dpi=300)
    print(f'[{img_name}] is generated')
    # plt.show()


def vis_img_patch(patch_tensor, patch_size, mask=None):
    """
    将图片划分成patch块并显示
    :param patch_tensor: (B,NUM,patch_size^2*C) 其中NUM是patch块的数量
    :param patch_size: patch的边长(默认正方形)
    :param mask:masking函数得到的mask矩阵，代表哪些patch被mask
    :return:
    """
    if mask is not None:
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, 16 ** 2 * 3)  # (N, H*W, p*p*3)
        patch_tensor = patch_tensor * (1 - mask)
    print(patch_tensor.shape)
    patch_tensor = patch_tensor.squeeze(0)
    n, d = patch_tensor.shape
    h = w = int(math.sqrt(n))
    plt.figure(figsize=(5, 5))

    # way-1 单个patch分离明显(建议查看patch效果时使用)
    # divide the img from (NUM_patch,D)->(NUM_patch,H_patch,W_patch,C) C放在后面便于plt处理
    # img = rearrange(patch_tensor, 'n (n1 n2 c) -> n n1 n2 c', n1=patch_size, n2=patch_size)
    # for i in range(h * w):
    #     plt.subplot(h, w, i + 1)
    #     plt.imshow(img[i].data.numpy(), cmap='gray')
    #     plt.axis('off')

    # way-2 patch紧密相连(建议查看mask效果时使用)
    img = rearrange(patch_tensor, '(h w) (n1 n2 c) -> (h n1) (w n2) c', h=h, w=w, n1=patch_size, n2=patch_size)
    plt.imshow(img)
    plt.axis('off')

    if mask is None:
        plt.savefig(f'./imgs_out/img_patch.png')
    else:
        plt.savefig(f'./imgs_out/masked_patch.png')


if __name__ == '__main__':
    attention_map = np.zeros((20, 20))
    attention_map[9][9] = 1
    attention_map[10][12] = 1
    vis_grid_attention(img_path="./imgs_in/dog_1.jpg", attention_map=attention_map)

    attention_map = np.random.normal(size=(10, 10))
    vis_attention_matrix(attention_map)

    # 进行patch划分及可视化
    img = data_to_tensor(img_path="./imgs_in/dog_1.jpg")
    img = patchify(img, patch_size=16)
    vis_img_patch(img, patch_size=16)

    # 进行mask及可视化
    x, mask, ids_restore = masking(img, 0.75)
    vis_img_patch(img, patch_size=16, mask=mask)
