"""
配置文件模板
可以复制并修改此文件来进行不同的实验
"""

# ============================================
# 基础配置 - ResNet50 + LSTM
# ============================================
config_baseline = {
    # 数据相关
    'dataset_path': 'flickr8k_aim3/dataset_flickr8k.json',
    'images_dir': 'flickr8k_aim3/images',
    'vocab_path': 'flickr8k_aim3/vocabulary.pkl',
    'freq_threshold': 5,
    
    # 模型结构
    'embed_size': 256,
    'hidden_size': 512,
    'num_layers': 1,
    'dropout': 0.5,
    'encoder_type': 'resnet50',
    'decoder_type': 'lstm',
    
    # 训练参数
    'batch_size': 64,
    'num_epochs': 30,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'grad_clip': 5.0,
    'num_workers': 4,
    
    # 保存和日志
    'checkpoint_dir': 'checkpoints',
    'save_step': 1,
    'log_step': 100
}


# ============================================
# 大模型配置 - ResNet101 + 2层LSTM
# ============================================
config_large = {
    **config_baseline,
    'encoder_type': 'resnet101',
    'hidden_size': 1024,
    'num_layers': 2,
    'batch_size': 32,  # 减小批次以适应显存
    'checkpoint_dir': 'checkpoints_large',
}


# ============================================
# ViT配置 - Vision Transformer + LSTM
# ============================================
config_vit = {
    **config_baseline,
    'encoder_type': 'vit_b_16',
    'hidden_size': 768,
    'embed_size': 768,
    'batch_size': 32,
    'checkpoint_dir': 'checkpoints_vit',
}


# ============================================
# GRU配置 - ResNet50 + GRU
# ============================================
config_gru = {
    **config_baseline,
    'decoder_type': 'gru',
    'checkpoint_dir': 'checkpoints_gru',
}


# ============================================
# 快速测试配置（用于调试）
# ============================================
config_debug = {
    **config_baseline,
    'batch_size': 8,
    'num_epochs': 2,
    'num_workers': 0,
    'save_step': 1,
    'log_step': 10,
    'checkpoint_dir': 'checkpoints_debug',
}


# ============================================
# Transformer配置 - ResNet50 + Transformer
# ============================================
config_transformer = {
    **config_baseline,
    'decoder_type': 'transformer',
    'embed_size': 512,  # Transformer通常需要更大的嵌入维度
    'hidden_size': 2048, # 在Transformer中用作dim_feedforward
    'num_layers': 6, # Transformer解码器层数
    'nhead': 8, # 多头注意力头数
    'batch_size': 32,
    'learning_rate': 5e-5, # Transformer通常需要更小的学习率
    'checkpoint_dir': 'checkpoints_transformer',
}


# ============================================
# 如何使用
# ============================================
"""
在 train.py 中选择配置：

from config import config_baseline, config_large, config_vit

# 选择一个配置
config = config_baseline  # 或 config_large, config_vit 等

trainer = Trainer(config)
trainer.train()
"""
