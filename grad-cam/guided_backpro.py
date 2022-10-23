# --------------------------------------
# -*- encoding: utf-8 -*-
# @File   : guided_backpro.py
# @Time   : 2022/10/23 12:50:51
# @Author : wzy 
# @Reference : https://github1s.com/jacobgil/pytorch-grad-cam/blob/HEAD/pytorch_grad_cam/guided_backprop.py
# @Reference : https://blog.csdn.net/m0_46653437/article/details/113201181
# --------------------------------------
import numpy as np
import torch
from torch.autograd import Function
import torchvision
import cv2
from torchvision import transforms

class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        # 模拟relu
        output = torch.clamp(input_img, min=0.0)  # clamp函数限制元素的下限为0
        self.save_for_backward(input_img, output)
        return output  # torch.Size([1, 64, 112, 112])


    @staticmethod
    def backward(self, grad_output):
        # saved_tensor函数可以获得save_for_backward函数存储的数据
        input_img, output = self.saved_tensors  # torch.Size([1, 2048, 7, 7]) torch.Size([1, 2048, 7, 7])
        positive_mask_1 = (input_img > 0).type_as(grad_output)  # torch.Size([1, 2048, 7, 7])  输入的特征大于零
        positive_mask_2 = (grad_output > 0).type_as(grad_output)  # torch.Size([1, 2048, 7, 7])  梯度大于零
        grad_input = grad_output * positive_mask_1 * positive_mask_2  
        return grad_input


class GuidedBackpropReLUasModule(torch.nn.Module):
    def __init__(self):
        super(GuidedBackpropReLUasModule, self).__init__()

    def forward(self, input_img):
        return GuidedBackpropReLU.apply(input_img)


class GuidedBackpropReLUModel:

    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = self.model.cuda()


    def __call__(self, input_img, target_category=None):
        # 相对于某个类别的置信度得分,计算输入图片上的梯度,并返回
        replace_all_layer_type_recursive(self.model,
                                         torch.nn.ReLU,
                                         GuidedBackpropReLUasModule())
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.model(input_img)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())
        
        # print(target_category)

        loss = output[0, target_category]
        loss.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()  # (1, 3, 224, 224)
        output = output[0, :, :, :]  # (3, 224, 224)
        output = output.transpose((1, 2, 0))  # 变成(224, 224, 3)便于图片打印

        replace_all_layer_type_recursive(self.model,
                                         GuidedBackpropReLUasModule,
                                         torch.nn.ReLU())

        return output

def replace_all_layer_type_recursive(model, old_layer_type, new_layer):
    for name, layer in model._modules.items():
        if isinstance(layer, old_layer_type):
            model._modules[name] = new_layer
        replace_all_layer_type_recursive(layer, old_layer_type, new_layer)

def preprocess_image(img):
    '''将numpy的(H, W, RGB)格式多维数组转为张量后再进行指定标准化,最后再增加一个batch维度'''
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def deprocess_image(img):
    '''先作标准化处理,然后做变换y=0.1*x+0.5,限定[0,1]区间后映射到[0,255]区间'''
    """https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py"""
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

if __name__ == '__main__':
    img_path = 'wzy/Visualization/grad-cam/d_c.jpg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR格式转换为RGB格式 shape: (224, 224, 3) 即(H, W, RGB)
    img = np.float32(img) / 255  # 转为float32类型,范围[0,1]

    input_img = preprocess_image(img)  # torch.Size([1, 3, H, W])

    model = torchvision.models.resnet50(pretrained=True)
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    gb = gb_model(input_img, target_category=None)
    gb = deprocess_image(gb)  # shape: (H, W, 3)

    cv2.imwrite('wzy/Visualization/grad-cam/gb_dc.jpg', gb)  # 保存图片


