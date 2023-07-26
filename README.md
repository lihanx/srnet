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


## Todo List

- [x] AugmentationDataSet 随机增强数据集
  - [ ] 使用 PhotoShop 处理原始数据集
- [ ] DataLoader 数据加载器测试
- [ ] SRNetTrainer 训练模型
  - [ ] 收敛过程记录到 csv 用于绘图
- [ ] SRNet/ResNetEncoder/Decoder
- [ ] SRNetTransformer 处理图片
  - [ ] 中间状态的 feature map 保存，用于论文图示
- [ ] SSIM/PSNR