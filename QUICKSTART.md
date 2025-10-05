# 快速开始指南

## 1. 安装依赖

```bash
cd d:\programming\lab\aim3\photo_description
pip install -r requirements.txt
```

## 2. 构建词汇表

```bash
python vocabulary.py
```

## 3. 开始训练

```bash
python train.py
```

## 4. 评估模型

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --split test
```

## 5. 生成描述

```bash
python inference.py --checkpoint checkpoints/best_model.pth --image flickr8k_aim3/images/1000268201_693b08cb0e.jpg
```

---

详细文档请参考 `README_USAGE.md`
