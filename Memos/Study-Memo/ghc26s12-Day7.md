## NLP

### 任务

- 语法正确检查
- 预测情感
- 确定一对句子的语义等价
- 确定一对问题的语义等价

### 语言模型

- 统计语言模型，预测下一个单词
- 学习了单词序列的联合概率函数

### 语言建模

- N-gram——神经网络LM——LLM（预训练）
- 通过cross-entropy和perplexity评估语言建模能力

### word embedding

- **语言模型的神经网络表示基本思路**：神经网络去学习上下文情况下，下一个词出现的概率

- 将word转换为向量（从稀疏空间映射到稠密空间）
- word vector：乘以词汇库矩阵（独热编码）
- king - man + woman ≈ queen
  - <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260407141135991.png" alt="image-20260407141135991" style="zoom:33%;" align='left'/>

### Word2Vec

用**浅层全连接神经网络**学习单词的向量化表示

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260407141215725.png" alt="image-20260407141215725" style="zoom:33%;" />

- **连续词袋模型（CBOW）**：基于周围的上下文词来预测中间词
- **连续跳过语法模型（Skip-gram**）：基于中间词预测上下文，更常用

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260407141502418.png" alt="image-20260407141502418" style="zoom:23%;" />

### 负样本Negative Sampling

- 随机选择“错误上下文词”，让模型学习相关性，而不必一直算softmax（相似词接近，错误词原理）
- 可以用NCE（噪声对比估计）作为完整softmax的有效近似，进一步可以用负采样简化

| 方法              | 目标         | 复杂度 |
| ----------------- | ------------ | ------ |
| Softmax           | 精确概率     | 很高   |
| NCE               | 近似概率     | 中     |
| Negative Sampling | 只学向量关系 | 很低 ✅ |

#### 问题

- 语料库带来的歧视和刻板印象



## HuggingFace

- 类似GItHub的AI开源社区
- 里面的transformers是一个预训练的模板库

### 情感分析例子

```py
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")

classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)

[{'label': 'POSITIVE', 'score': 0.9598047137260437},
 {'label': 'NEGATIVE', 'score': 0.9994558095932007}]

```

### pipeline

| Pipeline                   | 中文含义            | 输入               | 输出                  |
| -------------------------- | ------------------- | ------------------ | --------------------- |
| `feature-extraction`       | 特征提取 / 向量表示 | 一段文本           | 文本向量（embedding） |
| `fill-mask`                | 填词                | 含 `[MASK]` 的句子 | 被遮盖位置最可能的词  |
| `ner`                      | 命名实体识别        | 一段文本           | 实体及类别            |
| `question-answering`       | 问答                | 问题 + 一段上下文  | 答案片段              |
| `sentiment-analysis`       | 情感分析            | 一段文本           | 情感标签 + 分数       |
| `summarization`            | 文本摘要            | 长文本             | 简短总结              |
| `text-generation`          | 文本生成            | 提示词 / 开头文本  | 续写内容              |
| `translation`              | 机器翻译            | 源语言文本         | 目标语言文本          |
| `zero-shot-classification` | 零样本分类          | 文本 + 候选标签    | 最匹配的标签          |

### 整体流程

#### tokenizer

- 需要用分词器进行预处理
- return_tensors指定返回的张量类型
- 输出通常是input_ids和attention_mask

#### AutoModel

- 自动模型选择、加载预训练模型、模型配置与初始化、模型适配与调整以及下游任务应用

#### 后处理postprocessing

- 将logits变为概率

- ```py
  import torch
  from transformers import AutoTokenizer, AutoModelForSequenceClassification
  model_name = "distilbert-base-uncased-finetuned-sst-2-english"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(model_name)
  raw_inputs = [
  "We are very happy to show you the 🤗 Transformers library.",
  "We hope you don't hate it."
  ]
  inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
  # 获取模型输出
  outputs = model(**inputs)
  # 应用Softmax来得到概率分布
  logits = outputs.logits
  probabilities = torch.softmax(logits, dim=-1)
  # 获取预测结果
  predictions = torch.argmax(probabilities, dim=1)
  scores = torch.max(probabilities, dim=1).values
  # 映射模型预测到实际标签
  labels = ['NEGATIVE', 'POSITIVE']
  for i, (prediction, score) in enumerate(zip(predictions, scores)):
      print(f"Sentence:'{raw_inputs[i]}'")
      print(f"Prediction: {labels[prediction]}, Score: {score}\n")
  
  ```

  #### 图像识别

  ```py
  from transformers import pipeline
  from PIL import Image
  import requests
  
  # 加载图像分类模型
  classifier = pipeline("image-classification")
  
  # 加载本地图片
  image = Image.open("apple.jpg")
  
  # 进行识别
  result = classifier(image)
  print(result)
  # AI 会告诉你图片里是猫、狗还是其他东西
  ```

  ## BERT

  - Bidirectional Encoder Representations from Transformers
  - 一种encoder-only的预训练模型（可供迁移学习）
  - 运行自监督学习方法，为单词学习好的特征表示，微调后作为特征提取器，作为NLP任务的词嵌入特征

  ### 结构

  <img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260407150406778.png" alt="image-20260407150406778" style="zoom:33%;" />

  ### 优势

  - 替代Word2Vec刷新NLP的精度
  - 更能捕捉语句中的双向关系
  - 使用了多任务训练目标

  ### 输入表示

  - WordPiece嵌入
    - token embedding
    - position embedding
    - segment embedding
    - 求和得到最终的embedding（避免维度爆炸）

  ### 多任务模型

  - pre-training task：
    - **掩码语言模型（MLM）**：自监督，80%的时候会直接替换为[Mask]，10%的时候将其替换为其它任意单词，10%的时候会保留原始Token
    - **下一句预测任务（NSP）**：判断句子B是否是句子A的下文（IsNext）
  - fine-tuning task：
    - 基于句子对的分类
    - 基于单个句子的分类
    - 问答任务
    - 单个句子的标签任务

  ### RoBERTa

  - A Robustly Optimized BERT Pretraining Approach
  - 数据集
    - GLUE The General Language Understanding Evaluation (GLUE) benchmark
    - RACE The ReAding Comprehension from Examinations (RACE)
    - SQuAD The Stanford Question Answering Dataset (SQuAD)

### T5

- Text-to-Text Transfer Transformer
- encoder-decoder结构

### BART

- Bidirectional and Auto-Regressive Transformers

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260407152920619.png" alt="image-20260407152920619" style="zoom:33%;" />

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260407153202242.png" alt="image-20260407153202242" style="zoom:33%;" />

- GPT是单向预测（自回归——一个一个生成，每一步输出都依赖之前已经生成的内容）
- BERT只用于双向理解

### GPT

- Generative Pretrained Transformer
- 基于transformer，只保留了mask multi-head attention
  - 非监督学习：从大量语料库中学习语言规律
  - 多任务学习：适应不同的自然语言处理任务，泛化能力强

#### InstructGPT训练过程

| 步骤   | 名称                | 输入                      | 人类参与方式                       | 核心目的                           |
| ------ | ------------------- | ------------------------- | ---------------------------------- | ---------------------------------- |
| Step 1 | SFT（监督微调）     | Prompt + 人工示范答案     | 标注员直接写出理想回答             | 让模型先学会按指令回答             |
| Step 2 | RM（奖励模型训练）  | Prompt + 多个模型候选回答 | 标注员对回答排序（从好到差）       | 学会判断哪个回答更符合人类偏好     |
| Step 3 | PPO（强化学习优化） | 新Prompt + 模型生成回答   | 人类不直接参与当前轮打分，由RM代替 | 让模型朝高奖励、高偏好方向继续优化 |

## 知识蒸馏（KD/Knowledge Distillation）

### 主流压缩技术

- 架构设计（如MobileNet、ShuffleNet）
- 剪枝（pruning）
- 量化预低秩分解
- 知识蒸馏：不改变网络层结构的前提下实现知识迁移

### Teacher-Student架构

- 用庞大、复杂的教师模型指导轻量级的学生模型训练
- 用高质量的数据对来对齐能力

### 三种蒸馏方式

#### Logit蒸馏

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260407163620734.png" alt="image-20260407163620734" style="zoom:23%;" />

- 模型最后一层未经activation或softmax的输出
- 设计学生模型的损失函数向教师模型靠齐（KL散度衡量差异）

#### 特征蒸馏

- 引入转换函数来匹配教师和学生的特征维度（MLP或1*1 conv）
- 可以匹配中间层特征

#### 相似度蒸馏

- 正负样本
  - 样本/实例级（Instance-level）： 学习不同数据样本特征之间的相对距离和角度
  - 特征/通道级（Channel-level）： 学习特征通道之间的相关性矩阵
  - 类别级（Class-level）： 学习类内和类间的相似性

### 蒸馏训练机制

#### 离线蒸馏

- 老师是预先训练好的大模型，参数冻结
- 受限于老师的预训练水平

#### 在线蒸馏

- 没有预训练好的老师
- 多个学生从头训练，合成一个“虚拟教师”

#### 自蒸馏

- 老师和学生是同一个网络，将深层知识传递给浅层，或前一epoch的知识传递给当前轮
- 自监督学习

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20260407164424502.png" alt="image-20260407164424502" style="zoom:23%;" />

### 关键的蒸馏算法

- 对抗蒸馏：引入GAN
- 注意力蒸馏：传递CAM或GradCAM等注意力图
- 多教师蒸馏：投票、引入噪声、计算信息熵自适应分配权重，融合多个老师的知识
- 跨模态蒸馏：解决数据模态缺乏大规模人工标注数据集的问题
- 自适应蒸馏：动态调整损失权重、裁剪等
- 对比蒸馏：利用对比学习，拉近正样本距离，拉开负样本距离









