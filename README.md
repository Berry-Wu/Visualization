# 卷积核及特征图可视化
> 参考链接：https://debuggercafe.com/visualizing-filters-and-feature-maps-in-convolutional-neural-networks-using-pytorch/
1. 导入预训练模型
2. 输入一张图片经过网络
3. 调用visual.py中的卷积核可视化及特征图可视化

**效果示例**
> 这里使用resnet50预训练模型

- 卷积核：
    
  - filter3
  ![](imgs_out/filter_3.png)
  - filter48
  ![](imgs_out/filter_48.png)
  
- 特征图：

  - layer0
  ![](imgs_out/layer_0.png)
  - layer4
  ![](imgs_out/layer_4.png)
  