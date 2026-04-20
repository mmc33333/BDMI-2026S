# 课程大作业4-说明

The English version is [here](BDMI-Lab-4-Instructions.md)

## 组织方式与要求

1人完成； 

* 时间要求

校历第12周：课上验收

校历第13周：提交报告和完整代码


## 内容描述

###  数据集 

24句语音指令的语音时频谱数据集(spectrogram dataset)，语音数据集中包括：

40个人（/100人）的人声语料数据集，20个人的人声验证数据集。 

时频谱图训练&测试数据集：https://cloud.tsinghua.edu.cn/f/750d2cc680ab45fdb9dc/?dl=1


请同学们拷贝这个数据集即可。

### 任务描述

在时频谱图数据集（或原始数据集wav文件）上，设计（1）通过基于ViT预训练模型（视觉Transformer）和（2）传统用于对比的深度神经网络模型，根据时频谱图数据对语音指令进行分类，得出预测结果，并分析之。 

* 深度学习解决时频谱图分类的几种参考方案

** （1）基于预训练ViT视觉Transformer来完成作业。考虑不同的patch size和head个数，隐藏层的维度Hiden size. 见ViT和AST研究论文，和huggingface的ViT预训练模型。
 
  [1] A. Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ArXiv. https://arxiv.org/abs/2010.11929

  [2] Gong, Y., Chung, Y., & Glass, J. (2021). AST: Audio Spectrogram Transformer. ArXiv. https://arxiv.org/abs/2104.01778

  [3] Hugging Face community, https://huggingface.co

** （2）基准分类，参考分类过程1，参考链接：https://tensorflow.google.cn/tutorials/images/classification （较方便，推荐。）

    也可以参考链接：https://tensorflow.google.cn/tutorials/load_data/images  （你可以学习tf.data数据集过程！）

** （3）背景补充：（谷歌simpleAudio）项目链接：https://tensorflow.google.cn/tutorials/audio/simple_audio
（谷歌TensorFlow官方提供的一个详细从语音文件（*.wav）到时频图（spectrogram），再进行分类的案例。）（你可以学习从波形文件到时频谱图的制作过程！）


* 提示：

** {1} 用于对比的网络模型，可以参考一下audioNet项目。
项目网页链接：https://github.com/saturn-lab/audioNet 

** {2} 可能需要数据增强，进行resize标准大小等方法，来解决过拟合问题.
参考链接：https://tensorflow.google.cn/tutorials/images/data_augmentation


## 作业提交与评分

- 实验代码验收 (可成功运行的notebook文件)
- 实验报告 1 份（列出你的创新点和工作内容，以及对结果的分析）


** 评分标准

第12周课上检查，每位同学找老师验收，验收主要看代码运行和训练过程，并会记录代码运行得到的测试集accuracy指标。

验收完成则本次作业合格；根据实验报告和模型准确度可获得额外加分。

## 补充说明

audioPlot项目生成时频谱图。项目网页链接： https://github.com/saturn-lab/audioPlot
