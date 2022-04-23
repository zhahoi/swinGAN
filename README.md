# swinGAN

### 仓库简介

这个仓库使用了Swin Transformer中的模块构建基于Transformer的生成对抗网络(GANs)模型。

### Data preparation

数据集来自于Kaggle中开源的数据集[Anime Faces](https://www.kaggle.com/datasets/soumikrakshit/anime-faces)，数据集共有给 21551张动漫脸的图像，并且每张图像的大小为64×64。本实验从该数据集中随机选取10000张图像用于训练。

### Prequisites

你可以从`requirements.txt`安装依赖

```
pip install -r requirements.txt
```

### Training

```
python train.py
```

### Result

*训练300个epoch后的结果：*

[![Ls1iiq.png](https://s1.ax1x.com/2022/04/20/Ls1iiq.png)](https://imgtu.com/i/Ls1iiq)

### Acknowledgements

[Swin Transformer](https://github.com/berniwal/swin-transformer-pytorch.git)
