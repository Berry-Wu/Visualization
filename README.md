# 卷积核及特征图可视化
> 参考链接：https://debuggercafe.com/visualizing-filters-and-feature-maps-in-convolutional-neural-networks-using-pytorch/
1. 导入预训练模型
2. 输入一张图片经过网络
3. 调用visual.py中的卷积核可视化及特征图可视化

**效果示例**
> 这里使用resnet50预训练模型

- 注意力可视化(8.27新增)
  
  > 见visual.py 中 vis_grid_attention函数
  - 原图:
  
  ![](imgs_in/dog_1.jpg)
  - 注意力可视化后:
  
  ![](imgs_out/dog_1_with_attention.jpg)
  - 注:这里的attention_map并非来自真实得到,是定义的一个二维数组
  ```python
    attention_map = np.zeros((20, 20))
    attention_map[9][9] = 1
    attention_map[10][12] = 1
  ```
- 卷积核可视化：
    
  - filter3
 
  ![](imgs_out/filter_3.png)
  - filter48
 
  ![](imgs_out/filter_48.png)
  
- 特征图可视化：

  - layer0
 
  ![](imgs_out/layer_0.png)
  - layer4

  ![](imgs_out/layer_4.png)
  
