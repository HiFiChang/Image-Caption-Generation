"""
训练脚本
实现模型训练循环、损失计算、模型保存
"""
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import json

from vocabulary import Vocabulary, build_vocab_from_dataset
from dataset import get_data_loader
from model import build_model


class Trainer:
    """训练器类"""
    
    def __init__(self, config: dict):
        """
        初始化训练器
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        
        # 创建保存目录
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
        # 构建或加载词汇表
        if os.path.exists(config['vocab_path']):
            print(f"加载已有词汇表: {config['vocab_path']}")
            self.vocab = Vocabulary.load(config['vocab_path'])
        else:
            print(f"构建新词汇表...")
            self.vocab = build_vocab_from_dataset(
                config['dataset_path'],
                freq_threshold=config['freq_threshold'],
                save_path=config['vocab_path']
            )
        
        # 构建模型
        config['vocab_size'] = len(self.vocab)
        self.model = build_model(config, vocab_size=len(self.vocab))
        self.model.to(self.device)
        
        # 定义损失函数（忽略padding的交叉熵）
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab.word2idx[self.vocab.pad_token]
        )
        
        # 定义优化器
        self.optimizer = Adam(
            self.model.get_trainable_params(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # 训练状态
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, data_loader, epoch):
        """
        训练一个epoch
        Args:
            data_loader: 训练数据加载器
            epoch: 当前epoch
        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
        
        for i, (images, captions, lengths) in enumerate(pbar):
            # 移到设备
            images = images.to(self.device)
            captions = captions.to(self.device)
            lengths = lengths.to(self.device)
            
            # 前向传播
            outputs = self.model(images, captions, lengths)
            
            # 计算损失
            # outputs: [batch_size, max_length, vocab_size]
            # captions: [batch_size, max_length]
            # 我们预测captions的下一个词（从第2个位置开始预测）
            
            # 目标是captions本身（因为decoder的forward已经处理了输入偏移）
            batch_size = outputs.size(0)
            max_length = outputs.size(1)
            vocab_size = outputs.size(2)
            
            # 展平为2D以便计算损失
            outputs = outputs.reshape(batch_size * max_length, vocab_size)
            targets = captions.reshape(batch_size * max_length)
            
            # 计算损失（CrossEntropyLoss会自动忽略ignore_index的位置）
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['grad_clip']
            )
            
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})
            
            # 定期打印
            if (i + 1) % self.config['log_step'] == 0:
                avg_loss = total_loss / (i + 1)
                print(f"\nStep [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")
        
        avg_loss = total_loss / len(data_loader)
        return avg_loss
    
    def validate(self, data_loader):
        """
        验证模型
        Args:
            data_loader: 验证数据加载器
        Returns:
            平均损失
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, captions, lengths in tqdm(data_loader, desc="Validating"):
                images = images.to(self.device)
                captions = captions.to(self.device)
                lengths = lengths.to(self.device)
                
                outputs = self.model(images, captions, lengths)
                
                # 计算损失
                batch_size = outputs.size(0)
                vocab_size = outputs.size(2)
                outputs = outputs.reshape(-1, vocab_size)
                targets = captions.reshape(-1)
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """
        保存模型检查点
        Args:
            epoch: 当前epoch
            loss: 当前损失
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'vocab': self.vocab
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            f"checkpoint_epoch_{epoch+1}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")
        
        # 如果是最佳模型，额外保存
        if is_best:
            best_path = os.path.join(
                self.config['checkpoint_dir'],
                "best_model.pth"
            )
            torch.save(checkpoint, best_path)
            print(f"最佳模型已保存: {best_path}")
    
    def train(self):
        """完整训练流程"""
        print("\n" + "="*50)
        print("开始训练")
        print("="*50)
        
        # 创建数据加载器
        train_loader = get_data_loader(
            dataset_path=self.config['dataset_path'],
            images_dir=self.config['images_dir'],
            vocabulary=self.vocab,
            split='train',
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        
        val_loader = get_data_loader(
            dataset_path=self.config['dataset_path'],
            images_dir=self.config['images_dir'],
            vocabulary=self.vocab,
            split='val',
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        print(f"\n训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(val_loader.dataset)}")
        print(f"批次大小: {self.config['batch_size']}")
        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}")
        
        # 训练循环
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            print(f"{'='*50}")
            
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            print(f"\n训练损失: {train_loss:.4f}")
            
            # 验证
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            print(f"验证损失: {val_loss:.4f}")
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 保存检查点
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            if (epoch + 1) % self.config['save_step'] == 0:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            # 保存训练历史
            self.save_training_history()
        
        print("\n" + "="*50)
        print("训练完成！")
        print(f"最佳验证损失: {self.best_loss:.4f}")
        print("="*50)
    
    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        history_path = os.path.join(
            self.config['checkpoint_dir'],
            'training_history.json'
        )
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)


def main():
    """主函数"""
    # 配置参数
    config = {
        # 数据相关
        'dataset_path': 'flickr8k_aim3/dataset_flickr8k.json',
        'images_dir': 'flickr8k_aim3/images',
        'vocab_path': 'flickr8k_aim3/vocabulary.pkl',
        'freq_threshold': 5,
        
        # 模型相关
        'embed_size': 256,
        'hidden_size': 512,
        'num_layers': 1,
        'dropout': 0.5,
        'encoder_type': 'resnet50',  # 'resnet50', 'resnet101', 'vit_b_16'
        'decoder_type': 'lstm',  # 'lstm', 'gru'
        
        # 训练相关
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
    
    # 打印配置
    print("训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
