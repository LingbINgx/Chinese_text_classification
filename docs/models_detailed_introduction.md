# Models 详细介绍

本文档汇总 models 目录下各分类模型的设计与接口，重点包括：

- 模型定位
- 初始化参数
- 前向输入输出
- 层结构与数据流
- 参数量构成

---

## 1. TextCNNClassifier

源码：models/textcnn_classifier.py

### 1.1 模型定位

- 使用多组不同窗口大小卷积核提取局部 n-gram 特征。
- 每个卷积分支独立池化后拼接，最后用全连接层分类。

### 1.2 初始化参数

- vocab_size：词表大小。
- num_labels：类别数。
- embed_dim=256：词向量维度。
- num_filters=128：每个卷积分支的输出通道数。
- kernel_sizes=(3, 4, 5)：卷积窗口大小集合。
- dropout=0.5：分类前 dropout 概率。
- padding_idx=0：Embedding 中 padding 的索引。

### 1.3 前向参数与返回

- input_ids：形状一般为 (batch_size, seq_len)。
- attention_mask=None：接口保留，当前实现未使用。
- labels=None：可选监督标签。
- 返回值：(loss, logits)
  - loss：仅当 labels 不为 None 时计算。
  - logits：形状为 (batch_size, num_labels)。

### 1.4 层结构与数据流

- Embedding：input_ids -> (B, L, E)
- unsqueeze：加通道维 -> (B, 1, L, E)
- 多分支 Conv2d：每个 kernel_size 对应一条卷积分支
- ReLU 激活
- AdaptiveMaxPool2d((1, 1))：每个分支得到 (B, num_filters)
- 拼接分支特征：得到 (B, num_filters \* len(kernel_sizes))
- Dropout
- Linear：输出分类 logits

### 1.5 参数量构成

- Embedding：vocab_size \* embed_dim
- 每个卷积分支：num_filters _ (kernel_size _ embed_dim + 1)
- 分类层：(num_filters _ 分支数) _ num_labels + num_labels

---

## 2. TextRNNClassifier

源码：models/textrnn_classifier.py

### 2.1 模型定位

- 基于 LSTM 编码序列上下文。
- 使用最后时间步隐状态进行分类。

### 2.2 初始化参数

- vocab_size：词表大小。
- num_labels：类别数。
- embed_dim：词向量维度。
- hidden_dim：LSTM 隐状态维度。
- num_layers=1：LSTM 堆叠层数。
- bidirectional=False：是否双向。
- dropout=0.5：分类前 dropout 概率。

### 2.3 前向参数与返回

- input_ids：形状一般为 (B, L)。
- attention_mask=None：接口保留，当前实现未使用。
- labels=None：可选监督标签。
- 返回值：(loss, logits)

### 2.4 层结构与数据流

- Embedding：input_ids -> (B, L, E)
- LSTM：输出序列特征 (B, L, H) 或 (B, L, 2H)
- 取最后时间步：x[:, -1, :]
- Dropout
- Linear：输出 logits

### 2.5 参数量构成

- Embedding：vocab_size \* embed_dim
- LSTM（单层单向近似）：4H(E + H + 2)
- 双向时参数量约翻倍，多层按层数叠加
- 分类层：(hidden_dim _ 方向数) _ num_labels + num_labels

---

## 3. TransformerClassifier

源码：models/transformer_classifier.py

### 3.1 模型定位

- 使用 Hugging Face 预训练编码器进行文本表示。
- 取首 token 表示后接分类头进行微调。

### 3.2 初始化参数

- model_name：预训练模型名或本地模型路径。
- num_labels：类别数。
- dropout_prob=0.1：分类头前 dropout 概率。
- freeze_encoder=False：是否冻结编码器参数。

### 3.3 前向参数与返回

- input_ids：token 序列。
- attention_mask：注意力掩码。
- token_type_ids=None：可选句段 id。
- labels=None：可选监督标签。
- 返回值：(loss, logits)

### 3.4 层结构与数据流

- AutoModel.from_pretrained(model_name) 作为编码器
- 取 last_hidden_state[:, 0, :] 作为句向量
- Dropout
- Linear(hidden_size -> num_labels) 输出 logits
- labels 存在时计算 CrossEntropyLoss

### 3.5 参数量构成

- 编码器参数量：由 model_name 对应的预训练模型决定
- 分类头参数：hidden_size \* num_labels + num_labels
- freeze_encoder=True 时，可训练参数主要集中在分类头

---

## 4. ScratchTransformerClassifier

源码：models/scratch_transformer_classifier.py

### 4.1 模型定位

- 从零构建 Transformer Encoder 文本分类器。
- 使用可学习 CLS token 聚合全局信息进行分类。

### 4.2 初始化参数

- vocab_size：词表大小。
- num_labels：类别数。
- max_length=256：最大位置长度。
- embed_dim=256：模型维度 d_model。
- num_heads=8：多头注意力头数。
- num_layers=4：Encoder 层数。
- ffn_dim=1024：前馈层中间维度。
- dropout=0.1：dropout 概率。

### 4.3 前向参数与返回

- input_ids：形状一般为 (B, L)。
- attention_mask：形状一般为 (B, L)，用于 padding mask。
- labels=None：可选监督标签。
- 返回值：(loss, logits)

### 4.4 层结构与数据流

- Token Embedding + Position Embedding
- 拼接 cls_token 到序列首位
- LayerNorm + Dropout
- 构造 src_key_padding_mask（为 CLS 位补非屏蔽位）
- TransformerEncoder 堆叠编码
- 取 CLS 向量 x[:, 0]
- LayerNorm + Dropout + Linear 分类
- labels 存在时计算 CrossEntropyLoss

### 4.5 参数量构成

- token_embedding：vocab_size \* embed_dim
- position_embedding：max_length \* embed_dim
- cls_token：embed_dim
- embed_layer_norm：2 \* embed_dim
- 每层 TransformerEncoder 近似：
  - MHA：4 _ embed_dim^2 + 4 _ embed_dim
  - FFN：2 _ embed_dim _ ffn_dim + ffn_dim + embed_dim
  - 两个 LayerNorm：4 \* embed_dim
- 编码器总参数近似为单层参数乘 num_layers
- 最终分类层：embed_dim \* num_labels + num_labels

---

## 5. 四个模型对比速览

- TextCNNClassifier
  - 优点：并行度高，训练快，对局部关键短语敏感。
  - 局限：长距离依赖建模较弱。

- TextRNNClassifier
  - 优点：天然顺序建模，适合时序依赖。
  - 局限：并行能力弱，长序列训练效率较低。

- TransformerClassifier
  - 优点：直接利用预训练语义，精度通常较高。
  - 局限：依赖外部预训练模型，资源开销较大。

- ScratchTransformerClassifier
  - 优点：结构可控、可解释，便于教学和研究改造。
  - 局限：无预训练加持时，通常需要更多数据与训练轮次。
