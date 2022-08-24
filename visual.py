# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/24 15:39 
# @Author : wzy 
# @File : visual.py
# ---------------------------------------
import cv2
from matplotlib import pyplot as plt

from separate_num import separate


def vis_filter(model_weights, layer):
    print('p====================卷积核可视化====================q')
    plt.figure(figsize=(5, 5))
    filter_num = model_weights[layer].shape[0]
    print(f'该层一共有[{filter_num}]个卷积核')
    row, column = separate(filter_num)
    for i, filter in enumerate(model_weights[layer]):
        plt.subplot(row, column, i + 1)
        plt.imshow(filter[0, :, :].detach(), cmap='gray')
        plt.axis('off')
        plt.savefig(f'./imgs_out/filter_{layer + 1}.png')
    plt.show()
    print('b==================卷积核可视化结束===================d')


def vis_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def vis_feature(features, num_layers):
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
