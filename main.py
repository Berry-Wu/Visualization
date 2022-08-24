# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/24 16:47 
# @Author : wzy 
# @File : main.py
# ---------------------------------------
from torchvision import models
from data import data_process, data_load, data_to_model
from get_parameter import model_param
from visual import vis_image, vis_filter, vis_feature

model = models.resnet50(pretrained=True)  # loading the pre-trained ImageNet weights
img_path = 'imgs_in/1.jpg'
# print(model)
if __name__ == '__main__':
    model_weights, conv_layers, num_layers = model_param(model)
    img = data_load(img_path)
    vis_image(img)
    # 可视化卷积核
    vis_filter(model_weights, 2)

    # 可视化特征图
    img = data_process(img)
    features = data_to_model(conv_layers, img)
    vis_feature(features, num_layers)
