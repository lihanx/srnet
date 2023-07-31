# Star Reduction Net

用于深空摄影后期的星点缩减及降噪


## 数据集素材

来自 [NOIRLab](https://noirlab.edu/public/) 公开的图像数据

原始数据链接如下：

- [1303a.tif](https://noirlab.edu/public/images/noao1303a/)
- [iotw2104a.tif](https://noirlab.edu/public/images/iotw2104a/)
- [noao-m20johnson.tif](https://noirlab.edu/public/images/noao-m20johnson/)
- [noao-rosette_mulligan.tif](https://noirlab.edu/public/images/noao-rosette_mulligan/)
- [noao-n7000mosblock.tif](https://noirlab.edu/public/images/noao-n7000mosblock/)
- [noao-n7380sandburg.tif](https://noirlab.edu/public/images/noao-n7380sandburg/)

## Photoshop 缩星步骤

1. 打开「原始图像」
2. 选中「原始图像」，右键选择「复制图层」（Ctrl+J），创建「图层 1」
3. 打开「滤镜」-「其他」-「最小值」
3. 设置「半径」为 `2.4` 像素，「保留」为 `圆度`，点击「确定」
4. 再次选中「原始图像」，右键选择「复制图层」，将「新复制的图层」移动到图层列表的最上方（移动只是为了方便查看，不影响功能）
5. 选中「新复制的图层」，选择「滤镜」-「风格化」-「查找边缘」
6. 选择「图像」-「调整」-「黑白」，点击「确定」
7. 选择「图像」-「调整」-「色阶」，「输入色阶」右侧的 255 改为 `180`(根据个人需求设置)，点击「确定」
8. Ctrl+A 全选内容，Ctrl+C 复制，点击「图层 1」，使用 PS 窗口右下角的工具按钮「创建蒙版」
9. 按住 Alt 并点击「图层 1 的蒙版」，Ctrl+C 粘贴刚刚复制的内容
10. 选择「图像」-「调整」-「反相」
11. 返回「图层 1」，隐藏「新复制的图层」
12. 完成缩星（可以根据需求保存图像）

*缩星过程中使用的数字参数值均为经验值，可以根据个人需求调整*


## Todo List

- [x] AugmentationDataSet 随机增强数据集
  - [x] 使用 PhotoShop 处理原始数据集
  - [ ] _normalize_input_image 实现
  - [x] _rand_color_distort 实现
- [ ] DataLoader 数据加载器测试
- [ ] SRNetTrainer 训练模型
  - [ ] 收敛过程记录到 csv 用于绘图
- [ ] SRNet/ResNetEncoder/Decoder
- [ ] SRNetTransformer 处理图片
  - [ ] 中间状态的 feature map 保存，用于论文图示
- [ ] SSIM/PSNR