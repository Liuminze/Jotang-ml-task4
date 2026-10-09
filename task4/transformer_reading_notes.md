# 《Attention Is All You Need》阅读笔记

---

## 一、我用的资料和工具

1. **论文原文**：arXiv 上的 PDF 和 HTML 版，地址 `https://arxiv.org/abs/1706.03762`。  
2. **辅助理解**：一开始看到许多专业术语的英文很懵，然后看了两遍b站的讲解视频自己跟着公式手推了一遍 attention 的矩阵维。  

---

## 二、论文最初处理的任务：机器翻译

这篇论文一开始是为了做 **sequence transduction（序列转换）**，更具体地说就是 **机器翻译**。作者在 WMT 2014 的两个数据集上做了实验：

- **英德翻译**：达到 28.4 BLEU；
- **英法翻译**：达到 41.8 BLEU。

后来他们又把模型拿到英语成分句法分析（constituency parsing）上试了试，效果也不错，说明 Transformer 不只限于翻译。

### 为什么不用 RNN 和 CNN？

**RNN 的问题**：

- RNN 是**一步一步串行**算的，第 t 步必须等第 t-1 步算完。训练时不好并行，GPU 利用率低，慢。
- 长序列时，前面的信息要一步步传到最后，容易梯度消失或信息衰减。

**CNN 的问题**：

- 卷积一次只能看一个局部窗口，要看很远的关系需要很多层或者很大的卷积核。
- 远距离 token 之间的依赖建模成本高。

**Transformer 的思路**：

- 直接用 attention 让序列中任意两个位置“互相看见”，距离是 O(1) 的（一次点积就行）。
- 计算上可以高度并行，因为每个位置的 attention 可以一起算。
- 论文原话："dispensing with recurrence and convolutions entirely"，彻底丢掉循环和卷积。

---

## 三、Self-Attention 里的 "Self" 是什么意思？

- **Self** 就是“自己注意自己”。Query、Key、Value 都来自**同一个输入序列**。
- 翻译一下：每个 token 去问序列里其他 token：“你们谁跟我有关系？关系多大？”

### 和普通 Attention 的区别

|          | Self-Attention  | 普通 Encoder-Decoder Attention |
| -------- | --------------- | ---------------------------- |
| Q、K、V 来源 | 都来自同一个序列        | Q 来自 decoder，K、V 来自 encoder  |
| 作用       | 捕捉序列内部关系        | 把 encoder 的信息“拉”进 decoder    |
| 例子       | “它”应该关注“猫”还是“狗” | 翻译时，目标词的生成依赖源句子的哪些词          |

论文里 decoder 中间那个 multi-head attention 就是普通 attention：Q 来自 decoder 上一层的输出，K、V 来自 encoder 最终输出。

---

## 四、Q、K、V 来自哪里？为什么要不同线性变换？

### 来源

Q、K、V 的“原料”都是同一个 X，但用不同的矩阵投影成了不同的空间。

### 为什么要三套不同的矩阵？

我一开始觉得直接拿 X 当 Q、K、V 不也行？后来想通：

- **Q（query）**：代表“我要查什么”；
- **K（key）**：代表“我有什么索引标签”；
- **V（value）**：代表“我真正包含的信息内容”。

让模型分别学三套投影，相当于让同一个词在“提问”“被检索”“提供内容”这三种角色下有不同的表示。如果共用一套，模型就少了这种灵活性。

---

## 五、Attention 结果为什么要 softmax？每行权重表示什么？

### 为什么用 softmax？

`QK^T` 算出来的是“相似度分数”，可以是任意实数。softmax 做两件事：

1. **把它们变成概率**：所有权重加起来等于 1；
2. **放大差异**：分数高的位置权重更大，低的位置被压低，形成“聚焦”效果。

如果不做 softmax，加权求和就变成了直接加权平均，权重可正可负可很大，不稳定。

### 每行权重表示什么？

假设 Q 是 `(seq_len, d_k)`，那么 `QK^T` 是 `(seq_len, seq_len)`，softmax 之后每行和为 1。

- **第 i 行**：表示第 i 个 token 对序列中所有 token 的“关注程度”；
- **第 i 行第 j 列**：第 i 个 token 应该花多少注意力在第 j 个 token 上。

---

## 六、为什么要除以 sqrt(d_k)？

这是 **Scaled Dot-Product Attention** 的关键。

- 当 `d_k` 很大时，两个随机向量的点积绝对值也会变大（方差和维度成正比）。
- 点积太大，softmax 函数的输入会落到梯度很小的区域，导致梯度消失，训练不动。
- 除以 `sqrt(d_k)` 可以把数值缩放到合理范围，让 softmax 不至于太“硬”。

---

## 七、Multi-Head Attention 为什么要分多个头？

### 不是简单重复

每个头用的是**不同的投影矩阵**，所以每个头看到的 Q/K/V 是在不同子空间里的表示。它们能捕捉到不同的关系：

- 一个头可能学到“主谓关系”；
- 另一个头可能学到“指代关系”；
- 还有一个头可能学到“相邻词搭配”。

论文里 `h = 8`，`d_model = 512`，所以每个头是 `d_k = d_v = 64`。8 个头并行计算，最后拼接起来再投影回 512 维。

### 为什么不分头就不行？

单头 attention 把所有信息压在一个空间里，模型很难同时关注多种类型的关系。分头相当于给模型多个“观察角度”。

---

## 八、Transformer 没有循环结构，怎么知道词语顺序？

靠 **Positional Encoding（位置编码）**。

因为 attention 本身对位置不敏感：你打乱输入顺序，只要 Q/K/V 跟着打乱，attention 算出来的结果形式上是一样的。模型必须额外知道“哪个词在前面”。

论文用的是**正弦/余弦位置编码**：

然后直接把位置编码和词嵌入相加，作为模型输入。

> 我没完全吃透为什么这样设计就能让模型学会相对位置。只知道数学上 `PE(pos+k)` 可以写成 `PE(pos)` 的线性函数，所以模型能外推到训练时没见过的长度。

---

## 九、Decoder 为什么需要 Masked Attention？

Decoder 的任务是**自回归生成**：生成第 i 个词时，只能看到已经生成的第 1 到第 i-1 个词，不能提前看后面的词。

### Mask 怎么做？

在计算 `QK^T` 之后、softmax 之前，把当前位置“不应该看”的位置设为 `-∞`。softmax 之后这些位置的权重就变成 0。

具体来说，得到一个上三角 mask：

```
位置 1 只能看位置 1
位置 2 能看位置 1、2
位置 3 能看位置 1、2、3
...
```

### 如果能提前看到后面的答案会怎样？

- 训练时会“作弊”：模型直接抄后面的正确答案，loss 很低，但根本没有学会生成；
- 推理时根本不存在后面的词，模型就懵了，生成质量暴跌；
- 本质上破坏了自回归的因果性。

---

## 十、残差连接、Layer Normalization、Feed-Forward Network 在哪里？作用是什么？

### 1. 残差连接（Residual Connection）和 Layer Normalization

在结构图里，每个子层（attention 或 feed-forward）外面都包着一个 **Add & Norm** 框。

- **Add**：`x + Sublayer(x)`，把子层的输出和输入相加，形成残差连接；
- **Norm**：对相加后的结果做 Layer Normalization。

**为什么需要残差连接？**

- 模型很深（6 层 × 2/3 个子层），直接堆叠容易导致梯度消失；
- 残差连接给梯度留了一条“高速公路”，让深层网络也能训练；
- 也帮助保留原始输入信息。

**为什么用 Layer Normalization 而不是 Batch Normalization？**

- 序列长度不一样，batch 里的样本长度可能不同，BN 不好做；
- LN 是对**一个样本的所有特征**做归一化，不受 batch 和序列长度影响，更适合 NLP。

### 2. Position-wise Feed-Forward Network

每个 encoder/decoder 层里，attention 之后都有一个 FFN：

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

也就是：`Linear(d_model=512 → d_ff=2048)` → `ReLU` → `Linear(2048 → 512)`。

**位置级（position-wise）**的意思是：对序列中的每个位置独立应用同一个 FFN，位置之间不交互。这部分提供非线性变换和特征提取能力，相当于给 attention 提取出来的特征再做一个 MLP 变换。

---

## 十一、Figure 1 中 Encoder 和 Decoder 的位置

- **左侧是 Encoder**：输入 → Input Embedding + Positional Encoding → 6 层相同的层，每层 = Multi-Head Attention + Add&Norm + Feed Forward + Add&Norm。
- **右侧是 Decoder**：Outputs (shifted right) → Output Embedding + Positional Encoding → 6 层相同的层，每层 = Masked Multi-Head Attention + Add&Norm + Multi-Head Attention（encoder-decoder）+ Add&Norm + Feed Forward + Add&Norm。
- **顶部**：Decoder 输出 → Linear → Softmax → 输出概率。

---

## 十二、仍然存在的疑问

1. 位置编码的 sin/cos 形式为什么是“最优”或“足够好”的？可学习的位置嵌入效果几乎一样，那正弦编码的优势是不是只在“外推到更长序列”这个理论上？
2. 为什么 FFN 的中间层要设成 4 倍模型维度（2048 = 4×512）？这个 4 倍是经验调出来的还是有理论依据？
3. attention 权重可视化出来，真的能看到“某个头在学主谓关系”吗？还是需要更多分析工具？

---
