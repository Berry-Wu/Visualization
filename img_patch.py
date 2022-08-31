# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/31 16:04 
# @Author : wzy 
# @File : img_patch.py
# @reference:https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/models_mae.py
# ---------------------------------------
import math

import torch
from einops import rearrange


def patchify(img, patch_size):
    """
    将完整图片转换为多个patch图像块
    :param img:(B,C,H,W)
    :param patch_size:patch块的大小
    :return:tokens:(B,NUM,patch_size^2*C)
    """
    # 只考虑h=w且能整除patch_size的情况
    assert img.shape[2] == img.shape[3] and img.shape[2] % patch_size == 0
    tokens = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
    return tokens


def unpatchify(tokens, patch_size):
    """
    将patch块恢复为整张图像
    :param tokens:(B,NUM,patch_size^2*C)
    :param patch_size:patch块的大小
    :return:img:(B,C,H,W)
    """
    _, n, d = tokens.shape
    h = w = int(math.sqrt(n))
    img = rearrange(tokens, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, p1=patch_size, p2=patch_size)
    return img


def masking(tokens, mask_ratio):
    """
    在前面的到的patch块生成随机mask.(该部分参考MAE官方实现)
    :param tokens:(B,NUM,patch_size^2*C)
    :param mask_ratio:masked patches in the all patches
    :return:x_masked:经过mask后的输入，表示mask后剩下的，用于encoder
    :return:mask:对patch进行编码，可以视作图像的代表，mask掉的为1，没mask的为0
    :return:ids_restore:存储
    """
    b, n, d = tokens.shape
    remain_num = int(n * (1 - mask_ratio))  # 未被mask的patch数量
    noise = torch.rand(b, n, device=tokens.device)  # noise in [0, 1],为每个patch随机一个参数，用于后续的排序和mask

    # 根剧noise从小到大排列，返回对应下标
    ids_shuffle = torch.argsort(noise, dim=1)
    # 还原得到原本的noise顺序(妙)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # 根据前面得到的未被mask的patch数量，保留其对应的id
    ids_keep = ids_shuffle[:, :remain_num]
    # torch.gather:利用index来索引input特定位置的数值

    x_masked = torch.gather(tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))
    print(ids_keep.unsqueeze(-1).repeat(1, 1, d).shape)

    # 生成 mask: 0 is keep, 1 is remove
    mask = torch.ones([b, n], device=tokens.device)
    mask[:, :remain_num] = 0  # 此时得到的mask矩阵中，0全在前面，需要将这些排在前面的patch恢复到原本位置
    # 根据前面存储的原始分布ids_restore，获取真正的mask矩阵
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)
    tokens = patchify(img, 16)
    print('tokens:', tokens.shape)

    img = unpatchify(tokens, 16)
    print('img:', img.shape)

    masking(tokens, 0.75)
