# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/24 16:31 
# @Author : wzy 
# @File : get_parameter.py
# ---------------------------------------
from torch import nn


def model_param(model):
    """
    :param model:
    :return: model_weights:模型权重;conv_layers:存储所有的卷积层;counter:卷积层个数
    """
    model_weights = []  # save the conv layer weights
    conv_layers = []  # save the conv layers
    counter = 0  # counter: keep count of the conv layers
    model_children = list(model.children())

    # 将所有卷积层以及相应权重加入到两个空list中
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)

    print(f'Total convolutional layers: {counter}')
    # for weight, conv in zip(model_weights, conv_layers):
    #     print(f'CONV: {conv} ====> SHAPE: {weight.shape}')
    return model_weights, conv_layers, counter
