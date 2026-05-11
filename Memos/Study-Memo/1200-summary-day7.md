# Day7 学习总结：NLP、Word2Vec、Transformer 与大模型效率

## 1. 本周内容概览

WW07 文件夹主要围绕自然语言处理、Word2Vec、RNN、Seq2Seq、Attention、Transformer、GPT、ChatGPT、HuggingFace 和大模型推理效率展开。资料中还包含 Transformer 论文、代码 notebook、知识蒸馏和 KV Cache 相关课件。

这一周的内容从传统 NLP 表示方法逐步过渡到现代大语言模型，重点是理解文本如何被表示、模型如何建模序列关系，以及推理阶段如何提高效率。

## 2. 文本表示与 Word2Vec

早期文本表示方法包括 N-gram、词袋模型和词向量。词袋模型简单直观，但忽略词序和语义关系。Word2Vec 通过上下文学习词向量，使语义相近的词在向量空间中更接近。

课程资料中包含 CBOW 和 Skip-Gram 的代码。CBOW 通过上下文预测中心词，Skip-Gram 通过中心词预测上下文。它们都体现了“词义由上下文决定”的思想。

## 3. RNN、Seq2Seq 与 Attention

RNN 适合处理序列数据，因为它可以把前面的状态传递到后面。但传统 RNN 容易遇到梯度消失、长距离依赖建模困难等问题。

Seq2Seq 模型通过编码器和解码器处理输入输出序列，常用于机器翻译。Attention 机制进一步改进了 Seq2Seq，让解码器在生成每个词时都能关注输入序列中的不同位置。

## 4. Transformer 与 GPT

Transformer 用自注意力替代循环结构，可以并行处理序列，并更好捕捉长距离依赖。GPT 属于基于 Transformer Decoder 的自回归语言模型，通过预测下一个 token 完成文本生成。

从课程材料看，Transformer 的学习不仅包括理论结构，也包括 notebook 代码实现，例如位置编码、多头注意力和训练流程。这说明理解大模型需要同时掌握公式、结构图和代码。

## 5. ChatGPT 与训练流程

资料中涉及 ChatGPT 的预训练、奖励模型和 RLHF。大语言模型通常先通过大规模语料进行预训练，再通过指令微调和人类反馈强化学习对齐用户需求。

这让我认识到，模型能力不仅来自参数规模，还来自训练数据、训练目标和后期对齐方式。一个可用的对话模型需要在知识、表达、安全性和用户意图理解之间取得平衡。

## 6. HuggingFace 与 OpenAI API

课程资料中包含 HuggingFace Transformer 和 OpenAI API 的示例。HuggingFace 提供了大量预训练模型和统一接口，适合快速调用模型完成分类、生成和嵌入等任务。

API 调用则体现了现代 AI 应用开发的方式：不一定从零训练模型，而是把模型能力作为服务接入到具体应用中。

## 7. 大模型推理效率

WW07 中还包含 KV Cache、GEMM、FlashInfer、SnapKV 等资料。大模型推理的瓶颈不仅是模型参数量，也包括长上下文带来的显存占用和注意力计算开销。

KV Cache 可以避免重复计算历史 token 的 Key 和 Value，但上下文越长，缓存越大。SnapKV 等方法尝试选择重要位置进行压缩，从而在尽量保持效果的同时降低显存和延迟。

## 8. 学习体会

第 7 周让我看到 NLP 技术的演进路径：从词袋到词向量，从 RNN 到 Attention，再到 Transformer 和 GPT。每一步都在解决前一类方法的局限。

我觉得后续需要把重点放在两个方向：一是手写或阅读 Transformer 关键模块代码，二是理解推理优化，因为实际部署大模型时，速度和显存往往和模型效果同样重要。

