# 图像描述生成模型 - 使用说明

## 项目结构

```
photo_description/
├── flickr8k_aim3/              # 数据集目录
│   ├── dataset_flickr8k.json   # 数据集标注文件
│   ├── images/                 # 图像文件夹
│   ├── vocabulary.pkl          # 词汇表（训练时生成）
│   └── statistic.py           # 数据统计脚本
├── checkpoints/                # 模型检查点（训练时生成）
├── vocabulary.py               # 词汇表构建模块
├── dataset.py                  # 数据集和数据加载器
├── encoder.py                  # 编码器（CNN/ViT）
├── decoder.py                  # 解码器（LSTM/GRU）
├── model.py                    # 完整模型
├── train.py                    # 训练脚本
├── evaluate.py                 # 评估脚本
├── inference.py                # 推理脚本
├── requirements.txt            # 依赖配置
└── README_USAGE.md            # 本文件
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- NLTK >= 3.8.0
- Pillow, numpy, tqdm, matplotlib

### 2. 下载NLTK数据（用于评估）

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## 使用流程

### 第一步：构建词汇表（首次运行）

```bash
python vocabulary.py
```

这将从训练集构建词汇表并保存到 `flickr8k_aim3/vocabulary.pkl`

### 第二步：训练模型

```bash
python train.py
```

**训练配置说明**（可在train.py中修改）：

```python
config = {
    # 模型结构
    'embed_size': 256,          # 嵌入维度
    'hidden_size': 512,         # 隐藏层维度
    'num_layers': 1,            # RNN层数
    'dropout': 0.5,             # Dropout比例
    'encoder_type': 'resnet50', # 编码器: resnet50/resnet101/vit_b_16
    'decoder_type': 'lstm',     # 解码器: lstm/gru
    
    # 训练参数
    'batch_size': 64,           # 批次大小
    'num_epochs': 30,           # 训练轮数
    'learning_rate': 1e-3,      # 学习率
    'weight_decay': 1e-5,       # 权重衰减
}
```

训练过程中会：
- 自动保存检查点到 `checkpoints/` 目录
- 保存最佳模型为 `checkpoints/best_model.pth`
- 记录训练历史到 `checkpoints/training_history.json`

### 第三步：评估模型

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --split test
```

评估参数：
- `--checkpoint`: 模型检查点路径（必需）
- `--split`: 评估数据集（train/val/test，默认test）
- `--batch_size`: 批次大小（默认32）
- `--max_length`: 最大生成长度（默认50）

评估指标：
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR

结果保存到 `evaluation_results.json`

### 第四步：推理（生成描述）

#### 单张图像推理

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image flickr8k_aim3/images/1000268201_693b08cb0e.jpg
```

#### 使用束搜索

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image flickr8k_aim3/images/1000268201_693b08cb0e.jpg \
    --method beam_search \
    --beam_width 5
```

#### 生成多个描述

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image flickr8k_aim3/images/1000268201_693b08cb0e.jpg \
    --multiple 5 \
    --temperature 1.5
```

#### 可视化结果

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image flickr8k_aim3/images/1000268201_693b08cb0e.jpg \
    --visualize
```

## 模型架构说明

### 编码器选项

1. **ResNet50**（默认）
   - 轻量级，训练快
   - 特征维度：2048 → embed_size

2. **ResNet101**
   - 更深，效果更好
   - 计算量较大

3. **Vision Transformer (ViT)**
   - 最新架构
   - 需要更多训练数据

### 解码器选项

1. **LSTM**（默认）
   - 经典选择
   - 长序列效果好

2. **GRU**
   - 参数更少
   - 训练更快

## 进阶功能

### 1. 微调编码器

在 `train.py` 中设置：

```python
model.fine_tune_encoder(True)  # 允许更新编码器参数
```

### 2. 调整学习率策略

模型使用 `ReduceLROnPlateau` 自动调整学习率：
- 验证损失不下降时，学习率减半
- 耐心值（patience）：3个epoch

### 3. 梯度裁剪

防止梯度爆炸，在 `train.py` 中：

```python
'grad_clip': 5.0  # 梯度裁剪阈值
```

## 实验建议

### 基础实验

1. **Baseline**: ResNet50 + LSTM
   ```python
   'encoder_type': 'resnet50'
   'decoder_type': 'lstm'
   'hidden_size': 512
   ```

2. **增加容量**: ResNet101 + 2层LSTM
   ```python
   'encoder_type': 'resnet101'
   'num_layers': 2
   'hidden_size': 1024
   ```

### 进阶实验

3. **使用ViT**: Vision Transformer + LSTM
   ```python
   'encoder_type': 'vit_b_16'
   'decoder_type': 'lstm'
   ```

4. **不同解码器**: ResNet50 + GRU
   ```python
   'encoder_type': 'resnet50'
   'decoder_type': 'gru'
   ```

### 超参数调优

- **批次大小**: 32, 64, 128（根据显存）
- **学习率**: 1e-4, 5e-4, 1e-3
- **嵌入维度**: 128, 256, 512
- **隐藏层维度**: 256, 512, 1024

## 常见问题

### 1. 显存不足

- 减小 `batch_size`
- 减小 `hidden_size` 或 `embed_size`
- 使用梯度累积

### 2. 训练过慢

- 增大 `batch_size`
- 减少 `num_workers`
- 使用更小的编码器（ResNet50而非ResNet101）

### 3. 过拟合

- 增大 `dropout`
- 增大 `weight_decay`
- 使用数据增强
- 减少模型容量

### 4. 生成质量差

- 训练更多epoch
- 使用束搜索（beam_search）
- 调整温度参数
- 增加模型容量

## 性能预期

在Flickr8k数据集上，典型性能：

| 模型 | BLEU-1 | BLEU-4 | 训练时间 |
|------|--------|--------|----------|
| ResNet50+LSTM | 0.60-0.65 | 0.15-0.20 | ~2小时 |
| ResNet101+LSTM | 0.65-0.70 | 0.18-0.23 | ~3小时 |
| ViT+LSTM | 0.68-0.73 | 0.20-0.25 | ~4小时 |

*注：基于单张RTX 3090，30个epoch*

## 下一步

- [ ] 添加注意力机制（Attention）
- [ ] 实现Transformer解码器
- [ ] 支持更多评估指标（ROUGE, CIDEr）
- [ ] 添加数据增强
- [ ] 实现分布式训练

## 参考资料

- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)

## 联系方式

如有问题，请查看代码注释或提issue。
