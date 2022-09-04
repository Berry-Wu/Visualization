# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/9/4 18:16 
# @Author : wzy 
# @File : model_test.py
# ---------------------------------------
import numpy as np
from matplotlib import pyplot as plt
from Grad_CAM import GradCAM, show_cam_on_image
import torchvision
from wzy.Visualization.data import data_load, data_process

img_path = '../imgs_in/1.jpg'
target_category = 281


def main():
    model = torchvision.models.resnet34(pretrained=True)
    layers = [model.layer4]

    img = data_load(img_path)
    # (遇到的坑)里面存在resize，所以如果直接将上面的img与后续的cam_out叠加就会失败
    # 需要把data_process()中的resize去掉
    img_tensor = data_process(img)

    cam = GradCAM(model=model, target_layers=layers, use_cuda=False)
    # 不能用forward，
    cam_out = cam(input_tensor=img_tensor, target_category=target_category)
    cam_out = cam_out[0, :]

    img_out = show_cam_on_image(img.astype(dtype=np.float32) / 255., cam_out, use_rgb=True)

    plt.imshow(img_out)
    plt.axis('off')
    plt.savefig(f'../imgs_out/grad-cam.png')


if __name__ == '__main__':
    main()
