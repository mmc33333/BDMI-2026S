### 2024诺贝尔物理学奖

John Hopfield和Geoffrey Hinton

发明了深度学习方法

## PyTorch模型参数优化

### 网络训练流程

- **前向传播（Forward Propagation）**： 在前向传播阶段，输入数据通过网络层传递，每层应用权重和激活函数，直到产生输出

- **计算损失（Calculate Loss）**： 根据网络的输出和真实标签，计算损失函数的值

- **反向传播（Backpropagation）**： 反向传播利用自动求导技术计算损失函数关于每个参数的梯度

- **参数更新（Parameter Update）**： 使用优化器根据梯度更新网络的权重和偏置

- **迭代（Iteration）**： 重复上述过程，直到模型在训练数据上的性能达到满意的水平

### 超参数的设置

- 训练轮数epochs
- 批次大小batch size
- 学习率learning rate

### 损失函数

- nn.MSELoss均方误差用于回归
- nn.NLLLoss负对数似然用于分类
- nn.CrossEntropyLoss交叉熵

### 优化器

| 优化器类型 | 特点             | 适用场景 |
| ---------- | ---------------- | -------- |
| SGD        | 基础             | 基础     |
| Adam       | 自适应学习率     | DL       |
| RMSprop    | 适应学习率       | RNN      |
| Adagrad    | 参数独立的学习率 | 稀疏数据 |

### 优化的步骤

1. **清空梯度**：optimizer.zero_grad()
2. **前向传播**：pred = model(X)
3. **计算损失**：loss = loss_fn(pred, y)
4. **反向传播**：loss.backward()
5. **更新参数**：optimizer.step()

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 超参数
learning_rate = 1e-3
epochs = 5

# 模型、损失函数、优化器
model = MyModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)


# =====================
# 代码1: train
# =====================
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for X, y in dataloader:
        # 前向传播
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播三步骤
        optimizer.zero_grad()   # 清空梯度
        loss.backward()         # 计算梯度
        optimizer.step()        # 更新参数


# =====================
# 代码2: test
# =====================
def test(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()

    print(f"Accuracy: {correct}")


# =====================
# 代码3: 训练流程（调用）
# =====================
for epoch in range(epochs):
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)

```



## CNN

属于一种稀疏神经网络

netron工具可以用来查看模型文件（https://github.com/lutzroeder/netron）

### 基本结构

- 卷积层
  - 权重共享指同一个卷积核在滑动的过程中权重参数不变，赋予了平移不变性
  - 不同卷积核的权重相互独立、不共享
  - 卷积核的个数 = 卷积后的通道数

- 激活函数层
- 池化层
  - max-pooling：取邻域最大值
  - avg-pooling：取邻域平均值
  - 不增加通道数
- 批归一化（batch normalization）：
  - 通常在ReLU前
  - 减均值除以标准差
- 随机丢弃层（dropout）
  - 减少过拟合
  - 设置keep_prob训练时随机丢弃部分神经元

### 卷积运算

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260331140320898.png" alt="image-20260331140320898" style="zoom:33%;" />

### 零填充

- 单边填充P个0
- 卷积核大小F
- 卷积核移动的步长S
- 输出的长度$L=(W-F+2P)/S+1$

### 3D卷积

- 卷积核核卷积图像的深度要一致
- 输出张量的通道数channel取决于卷积核的个数

### 稀疏连接

- 区别于全连接
- 卷积核运算可以等效为局部规则的连接方式

### 优点

- 参数个数少，深度深
- 并行计算快

### VGG实例

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260331143555367.png" alt="image-20260331143555367" style="zoom:33%;" />

用3个3×3卷积核替换一个7×7卷积核，保持感受野的同时增加网络深度和非线性表达能力

### 卷积网络发展

- 1980Fukushima 提出Neocognitron
- NiN引入1×1卷积层和全局平均池化（GAP)，增强了非线性，压缩了模型体积
- 90年代：LeCun的手写数字识别LeNet-5
  - 其中1×1卷积能减小通道数，压缩特征，替代全连接层，且由于参数共享大幅减小了参数量
  - <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260331143942217.png" alt="image-20260331143942217" style="zoom:20%;" />
- 2012Hinton的ImageNet图像分类
  - 李飞飞团队标注大量数据
  - ILSVRC图像分类挑战赛冠军为AlexNet
  - <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260331144436615.png" alt="image-20260331144436615" style="zoom:33%;" />
  - 注意深度（通道数）和卷积核个数的对应
  - 双流结构来源于GPU的现存限制
- GoogleNet引入Inception模块，在统一卷积层并行使用多种尺寸的卷积核和池化
- ResNet引入残差思想，增加了skip connection
- DenseNet的DenseBlock将当前层的输出特征与之后所有层直连，使每一层都能直接访问前面所有层的特征图，允许了特征复用，防止梯度消失



## Transformer

### 历史

- 2018年NLP的ImageNet时刻

- 来源于机器翻译

- 无监督预训练+自监督微调对自然语言处理任务的有效性

- 被称为Vanilla Transformer用来指代最经典、标准、基础的模型

### 核心

encoder-decoder（Seq2Seq）结构

- **编码器encoder**：
  - 多头自注意力层（仅填充掩码）
  - 点式前馈网络
  - 每个子层后有一个残差连接，然后层归一化
  - <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260402192640822.png" alt="image-20260402192640822" style="zoom:25%;" align='left'/>
- **解码器decoder**：
  - <u>掩码</u>多头自注意力层
  - 多头<u>交叉</u>注意力
    - Query 来自 Decoder
    - Key / Value 来自 Encoder 输出
    - 让 Decoder “关注输入内容”
  - 点式前馈网络
- 输出为每个词的概率

### 结构

1. 获取输入数据的向量表示（embedding）和位置embedding组合后，表示为向量矩阵（tokenize）
   - <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260402202705938.png" alt="image-20260402202705938" style="zoom:23%;" align='left'/>
2. 传入Encoder，得到编码信息矩阵C
3. 将C传递到Decoder，mask后依次进行预测

### 自注意力机制Self-Attention

- 来源：从机器翻译中的Seq2Seq模型提出
- 词与词之间的相关性即为attention
- 增加了单词1语义的上下文含义

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260331152118158.png" alt="image-20260331152118158" style="zoom:38%;" />

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260331152620522.png" alt="image-20260331152620522" style="zoom:38%;" />

### 点积注意力

#### QKV模型

$Attention(Q,K,V)=softmax(\dfrac{QK^T}{\sqrt{d_k}})V$

- 计算得到相关度，全局范围加权得到不同词语之间的关联（softmax将杂乱的logits变为概率）
- 乘以V即依据概率提取信息

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260331153004134.png" alt="image-20260331153004134" style="zoom:33%;" />

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260331153337454.png" alt="image-20260331153337454" style="zoom:33%;" />

### 多头注意力

- 并排多个线性层（Dense）成为多个头
- 每个头通过按比例缩放的点积注意力模块
- 多个头串联成一体（concatenate）
- 最后一层线性层
- 计算复杂度仍为$O(n^2)$

### 位置编码

- position信息要加入到embedding向量中表示词在句子中的位置，弥补自注意力机制对顺序不敏感的缺陷
- 计算公式:
  - $PE_{(pos,2i)}=sin(\dfrac{pos}{10000^{2i/d_{model}}})$
  - $PE_{(pos,2i+1)}=cos(\dfrac{pos}{10000^{2i/d_{model}}})$
  - pos为位置索引，i为维度索引

### 遮挡（masking）

- **填充遮挡**：遮挡填充的部分，确保模型不会将填充作为输入
- **前瞻遮挡（look-ahead mask）**：用于遮挡一个序列中的后续标记（future tokens），只使用前面的条目，确保自回归生成

### ViT（Vision Transfomer）

- 只有encoder没有decoder
- 将一张 `H x W x C` 的图像，分割成 `N` 个 `P x P x C` 的图像块（Patches）
- 添加上位置编码，再拉平到一维
- 序列开头添加分类标识
- <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260331162649900.png" alt="image-20260331162649900" style="zoom:33%;" />

### 分类

- 类 GPT（也称自动回归Transformer模型）
  - Generative Pre-trained Transformer

  - Decoder-only生成

  - 根据上文预测下一个词

- 类 BERT（又称自动编码Transformer模型）
  - Bidirectional Encoder Representations from Transformer

  - Encoder-only理解

  - 双向，随机遮盖一定比例的词（masked），并让模型预测他们

- 类 BART/T5（又称Seq2Seq的Transformer模型）
  - BidirectionalAuto-Regressive Transformer
  - Encoder-Decoder理解+生成
  - 重建被破坏的文本


| 特性     | BERT                                                   | GPT                                | BART                                       |
| -------- | ------------------------------------------------------ | ---------------------------------- | ------------------------------------------ |
| 全称     | Bidirectional Encoder Representations from Transformer | Generative Pre-trained Transformer | BidirectionalAuto-Regressive Transformer   |
| 架构     | Encoder-only (仅编码器)                                | Decoder-only (仅解码器)            | Encoder-Decoder (编码器-解码器)            |
| 核心能力 | 理解 (双向上下文)                                      | 生成 (单向自回归)                  | 理解 + 生成                                |
| 训练任务 | 完形填空 (MLM)                                         | 预测下一个词                       | 文本去噪与重建                             |
| 信息流   | 双向，可同时看到全文                                   | 单向，只能看到上文                 | 先双向编码，再单向生成                     |
| 优势领域 | 自然语言理解 (NLU)                                     | 自然语言生成 (NLG)                 | 序列到序列任务 (Seq2Seq)，重建被破坏的文本 |

### 优缺点

- **优点**：
  - 不对数据间的时空关系做任何假设，可扩展性高
  - 层数深，表达能力强
  - 并行计算效率高
  - 全局建模能力强，能学习长距离的依赖
- **缺点**：
  - 计算复杂度高
  - 长序列不友好

### BERT

- encoder-only
- 提供其他任务迁移学习的模型，作为特征提取器
- 本质是自监督学语料的特征表示



