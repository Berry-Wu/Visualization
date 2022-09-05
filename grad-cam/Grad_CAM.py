# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/9/4 16:00
# @Author : wzy 
# @File : Grad_CAM.py
# @Reference : https://github.com/jacobgil/pytorch-grad-cam
# @Reference : https://blog.csdn.net/qq_37541097/article/details/123089851
# ---------------------------------------
import cv2
import numpy as np


class GradCAM:
    def __init__(self, model, target_layers, use_cuda=False, reshape_transform=None):
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        self.reshape_transform = reshape_transform
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform)

    def get_cam_weights(self, grads):
        """
        :param grads: yc对特征图A反向传播计算的梯度  (B C H W)
        :return: 原文第一个公式，也就是将梯度对高和宽求平均，从而得到某个通道的重要性权重
        """
        # 本来应该是2，3维，到这里少了batchsize，还在debug--> 见本py的L149-L154
        return np.mean(grads, axis=(2, 3), keepdims=True)

    def get_cam_image(self, activations, grads):
        """
        该函数将重要性权重与通道特征图加权
        :param activations: 特征图A
        :param grads: yc对特征图A反向传播计算的梯度
        :return:
        """
        # weight就是通道的重要性权重，是一个形状为1*channels的向量
        weights = self.get_cam_weights(grads)  # weights:(1, 512, 1, 1)  activations:(1, 512, 13, 20)
        weighted_activations = weights * activations  # (1, 512, 13, 20)
        print(weighted_activations.shape)
        cam = weighted_activations.sum(axis=1)
        return cam

    def get_target_width_height(self, input_tensor):
        """
        获取特征图的高和宽，后续进行平均
        :param input_tensor: 输入特征图张量
        :return: 特征图的高和宽
        """
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def scale_cam_image(self, cam, target_size=None):
        """
        :param cam:
        :param target_size:
        :return:
        """
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)
        return result

    def compute_cam_per_layer(self, input_tensor):
        """
        :param input_tensor:
        :return: cam_per_target_layer:是一个列表，里面存储了每个通道的cam
        """
        # 得到该特征层的激活、反向传播梯度、特征图大小;目前list里只有一个元素，因为这里只取一层特征图
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []

        # 对该特征图每个通道循环进行下述处理
        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        """
        将多个特征图层的cam求和 (！！！在Grad-CAM论文中选取最后一层进行计算，这里可以将多个层计算的结果聚合在一起)
        :param cam_per_target_layer: compute_cam_per_layer()得到的结果，其实就是对每层特征图求cam，是个列表
        :return:返回整个特征图，即A的cam
        """
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        outputs = self.activations_and_grads(input_tensor)  # outputs就是经过网络后的输出
        if target_category is None:
            raise Exception("please enter the target_category(a num represents a category)")
        else:
            target_categories = [target_category] * input_tensor.size(0)

        self.model.zero_grad()
        loss = 0
        for i in range(len(target_categories)):
            loss = loss + outputs[i, target_categories[i]]
        loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


class ActivationsAndGradients:
    """
    Class for extracting activations and registering gradients from targeted intermediate layers
    调用时会返回经过模型处理后的输出结果
    """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        # 模型的获取grad和activation
        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))

            # 直接写下面这一行会报错，在WZMIAOMIAO大佬的github中看到这个
            # self.handles.append(target_layer.register_forward_hook(self.save_gradient))
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(target_layer.register_full_backward_hook(self.save_gradient))
            else:
                self.handles.append(target_layer.register_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):  # 是register那个函数默认的，需要输入这几个参数；删掉就报错
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        grad = output[0]
        # Gradients are computed in reverse order
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


def show_cam_on_image(img: np.ndarray, mask: np.ndarray, use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    This function overlays the cam mask on the image as an heatmap.By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
