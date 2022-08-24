# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/24 15:56 
# @Author : wzy 
# @File : separate_num.py
# @reference : https://blog.csdn.net/qq_36607894/article/details/103595912
# ---------------------------------------
import numpy as np


def separate(input):
    """
    该函数用于将卷积核数量变为两个相近的数相乘，以便用于subplot的row和column(更好看)
    """
    start = int(np.sqrt(input))
    factor = input / start
    while not is_integer(factor):
        start += 1
        factor = input / start
    return int(factor), start


def is_integer(number):
    if int(number) == number:
        return True
    else:
        return False


if __name__ == '__main__':
    print(separate(120))