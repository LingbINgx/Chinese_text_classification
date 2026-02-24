# Transformer 文本分类任务详细文档（含注释）

## 1. 任务目标与范围

- **任务目标**：基于预训练 Transformer 模型完成中文文本分类。
- **当前默认模型**：`bert-base-chinese`。
- **训练入口**：项目根目录执行 `python main.py`。
- **输出产物**：最佳权重、tokenizer、标签映射文件。

---

## 2. 项目结构说明（与任务相关）

```text
main.py                             # 训练总入口
params.yaml                         # 训练参数配置文件
models/transformer_classifier.py    # 模型定义（编码器 + 分类头）
train/transformer_trainer.py        # 训练/验证/测试主流程
docs/transformer_task_documentation.md
```

> 注释：任务主逻辑集中在 `train/transformer_trainer.py`，其余文件分别承担“入口、参数、模型定义”。

---

## 3. 数据格式要求（非常关键）

当前训练代码在 `train/transformer_trainer.py` 内部固定要求 CSV 至少包含以下列：

- `content`：文本内容
- `label`：分类标签

### 3.1 示例数据

```csv
content,label
这部电影节奏很紧凑，演员发挥也很好,1
剧情比较平淡，后半段略拖沓,0
```

### 3.2 字段说明（详细注释）

- `content`
  - 注释：原始文本字段，会先经过 `clean_text` 清洗，再送入 tokenizer。
  - 注释：任何非字符串输入会被强制转字符串处理。
- `label`
  - 注释：可以是整数或可排序的离散类别（例如 `0/1`、`体育/财经`）。
  - 注释：训练前会自动构建 `label2id` 映射，转换为模型训练用的整数 id。

---

## 4. 配置文件详解（params.yaml）

当前配置示例：

```yaml
TRAIN_FILE_PATH: "data/news_train.csv"   # 训练集路径（相对项目根目录）
VAL_FILE_PATH: "data/news_test.csv"      # 验证集路径
TEST_FILE_PATH: "data/news_test.csv"     # 测试集路径

MODEL_NAME: "bert-base-chinese"          # Hugging Face 预训练模型名
MAX_LENGTH: 256                           # 文本最大长度；超长截断，不足补齐

TRAIN_BATCH_SIZE: 16                      # 训练 batch 大小
EVAL_BATCH_SIZE: 32                       # 验证/测试 batch 大小
LEARNING_RATE: 2.0e-5                     # AdamW 学习率
WEIGHT_DECAY: 0.01                        # AdamW 权重衰减，抑制过拟合
WARMUP_RATIO: 0.1                         # 学习率 warmup 比例
EPOCHS: 3                                 # 训练轮次

DROPOUT_PROB: 0.1                         # 分类头 dropout 概率
FREEZE_ENCODER: false                     # 是否冻结编码器参数
OUTPUT_DIR: "models/checkpoints"         # 模型与映射输出目录
```

### 4.1 关键参数调优建议（注释）

- `MAX_LENGTH`
  - 注释：值越大，保留语义越完整，但显存占用和耗时越高。
- `TRAIN_BATCH_SIZE`
  - 注释：受显存限制，显存不足优先降低该参数。
- `LEARNING_RATE`
  - 注释：Transformer 微调常用 `1e-5 ~ 5e-5`，当前 `2e-5` 是稳妥起点。
- `WARMUP_RATIO`
  - 注释：warmup 可稳定初期训练，常见在 `0.05 ~ 0.1`。
- `FREEZE_ENCODER`
  - 注释：当训练数据很少、机器资源有限时，可尝试 `true` 先只训练分类头。

---

## 5. 训练流程逐步拆解（含详细注释）

下述流程对应 `train/transformer_trainer.py`：

1. **读取配置**
   - 注释：`_load_config` 从 `params.yaml` 读取并构建 `TrainConfig`。
2. **加载数据**
   - 注释：`_load_dataframe` 读取 CSV，并校验 `content/label` 两列是否存在。
3. **构建标签映射**
   - 注释：`_build_label_mapping` 生成 `label2id` 与 `id2label`。
4. **初始化 tokenizer**
   - 注释：`AutoTokenizer.from_pretrained(MODEL_NAME)`。
5. **构建 Dataset/DataLoader**
   - 注释：`TextClassificationDataset.__getitem__` 返回 `input_ids`、`attention_mask`、`labels`，必要时附带 `token_type_ids`。
6. **初始化模型与优化器**
   - 注释：模型来自 `TransformerClassifier`，优化器使用 `AdamW`。
7. **训练循环**
   - 注释：每步执行前向、反向传播、优化器更新、学习率调度。
8. **验证并保存最佳模型**
   - 注释：按验证集准确率 `val_acc` 选优并保存 `best_model.pt`。
9. **测试评估**
   - 注释：加载最佳权重，在测试集计算最终 loss/accuracy。
10. **保存附属文件**
    - 注释：保存 tokenizer 和标签映射，便于后续推理与部署一致性。

---

## 6. 模型结构说明（transformer_classifier.py）

模型核心结构：

- 编码器：`AutoModel.from_pretrained(model_name)`
- 池化策略：取最后一层隐藏状态中 `[CLS]` 位置向量（`last_hidden_state[:, 0, :]`）
- 分类头：`Dropout + Linear(hidden_size -> num_labels)`
- 损失函数：`CrossEntropyLoss`

### 6.1 设计注释

- 使用 `[CLS]` 表示进行分类是 BERT 系列最常见范式。
- `Dropout` 用于降低过拟合风险。
- `freeze_encoder` 支持快速实验（只训练线性层）。

---

## 7. 运行方式与输出解释

## 7.1 运行命令

```bash
pip install -e .
python main.py
```

## 7.2 日志示例

```text
Epoch 1/3 | train_loss=0.4217 train_acc=0.8124 | val_loss=0.3651 val_acc=0.8453
Epoch 2/3 | train_loss=0.3014 train_acc=0.8742 | val_loss=0.3328 val_acc=0.8610
Epoch 3/3 | train_loss=0.2458 train_acc=0.9021 | val_loss=0.3214 val_acc=0.8678
Test | loss=0.3189 acc=0.8695
```

## 7.3 输出目录

`OUTPUT_DIR` 下会生成：

- `best_model.pt`
  - 注释：验证集最优模型权重。
- `tokenizer/`
  - 注释：训练时使用的分词器，推理阶段必须保持一致。
- `label_mapping.json`
  - 注释：保存 `label2id` 与 `id2label`，用于把预测 id 还原为原始标签。

---

## 8. 常见问题与排查（附注释）

1. **报错：缺少 `content` 或 `label` 列**
   - 注释：数据列名与代码约定不一致，请修改 CSV 表头或代码校验字段。
2. **显存不足（CUDA out of memory）**
   - 注释：优先降低 `TRAIN_BATCH_SIZE`，其次降低 `MAX_LENGTH`。
3. **下载预训练模型失败**
   - 注释：检查网络环境，或提前缓存 Hugging Face 模型。
4. **训练很慢**
   - 注释：确认是否使用 GPU；若仅 CPU，可先将 `EPOCHS` 降低做流程验证。
5. **验证准确率不升**
   - 注释：尝试降低学习率、增大训练数据、检查标签噪声。

---

## 9. 可维护性建议（文档注释）

- 建议将数据列名约定写入数据生成脚本，避免训练时列名不一致。
- 建议在后续版本补充推理脚本（加载 `best_model.pt` + `tokenizer` + `label_mapping.json`）。
- 建议增加随机种子与日志落盘，提升可复现性与可追溯性。

---
