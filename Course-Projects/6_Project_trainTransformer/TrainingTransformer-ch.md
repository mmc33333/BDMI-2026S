超参数自选（seq_length, batch_size, D_model）

Task1. 设计实现一个Transformer
1.1 attention机制
1.2 multihead-attention
1.2 PoE
1.3 Mask


Task2. 外围tokenizer

Task3. 训练一个小型Transformer模型

block数，Add&norm，MoE

来生成中文新闻文本。  

训练数据：19910条中文新闻文本（课程提供）（新闻稿）

大模型生成的小故事集



Task4.评估

测试数据：500条中文新闻文本（课程提供）
评估指标：  
（1）模型参数量越小越好。  
（2）运行时间越短越好。  
（3）语法正确性（满分：10分，分数越高越好）。  
（4）内容一致性（满分：10分，分数越高越好）。  
（5）表达严谨性（满分：10分，分数越高越好）。