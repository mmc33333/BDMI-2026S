# Mi0407 学习笔记

## NLP Word2Vector

### Natural Language Processing

统计语言模型：预测文档中下一个单词或字符

语言建模能力：cross-entropy + perplexity

**LM->LLM**: Transformer -> GPT & BERT

### 

用神经网络去学习给定上下文情况下，下一个词出现的概率

**word embedding** word embedding是一种对单词从稀疏（one hot vector）空间映射到稠密（vector）空间技术的统称

### word2vec

全连接网络

word2vec: CBOW 负采样

#### Continuous Bag-of-Words & Continuous Skip-gram

1. CBOW: 上下文由当前（中间）词之前和之后的几个词组成。
2. CSG: 用于预测同一句子中当前单词前后一定范围内的单词。

#### Negative sampling

The skipgrams function returns all positive skip-gram pairs by sliding over a given window span. 
To produce additional skip-gram pairs that would serve as negative samples for training, you need to sample random words from the vocabulary. 

### NLP based on Huggingface (practical)

Models Datasets Metrics Docs

```
pip install torch transformers transformers[sentencepiece]
```

transformers库中基本对象是pipeline函数

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("~~")
```

1. feature-extraction (get the vector representation of a text)
2. fill-mask
3. ner (named entity recognition)
4. question-answering
5. sentiment-analysis
6. summarization
7. text-generation
8. translation
9. ero-shot-classification

```python
from transformers import pipeline
from PIL import Image
import requests

classifier = pipeline("image-classification")

image = Image.open("apple.jpg")

result = classifier(image)
print(result)
```

---
## Delve into Transformer

### BERT Bidirectional Encoder Representations from Transformers

Encoder Only

Pre-Training Tasks:

1. Masked Language Model
2. Next Sentence Prediction

Fine-Tuning Tasks:

（a）基于句子对的分类任务
（b）基于单个句子的分类任务
（c）问答任务
（d）单个句子标签任务

### T5 Text-to-Text Transfer Transformer

Encoder-Decoder

### BART Bidirectional and Auto-Regressive Transformers

### GPT Generative Pretrained Transformer

Decoder Only

与BERT相比，GPT使用了一种掩码语言模型（masked language model）的方法，从而在语言理解能力上有所提高。

Unsupervised learning & multitasks learning

GPT 使用 Transformer的 Decoder 结构，并对 Transformer Decoder 进行了一些改动，原本的 Decoder 包含了两个 Multi-Head Attention 结构，GPT 只保留了 Mask Multi-Head Attention

## Distillation

主流压缩技术：

架构设计, Pruning, Quantization, KD（在不改变网络层结构的前提下实现知识迁移）

除了压缩：跨任务迁移 & 数据隐私保护（Data-free KD）