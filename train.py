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
import argparse

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
        self.config['vocab_size'] = len(self.vocab)
        # 传递pad_token索引给模型（Transformer使用）
        self.config['pad_token_idx'] = self.vocab.word2idx[self.vocab.pad_token]
        self.model = build_model(self.config)
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
        
        for i, batch in enumerate(pbar):
            # 兼容数据加载返回3元组或4元组（带image_paths）
            if len(batch) == 4:
                images, captions, lengths, _ = batch
            elif len(batch) == 3:
                images, captions, lengths = batch
            else:
                raise ValueError(f"不支持的批处理数据格式，期望3或4个元素，得到{len(batch)}个")
            # 移到设备
            images = images.to(self.device)
            captions = captions.to(self.device)
            lengths = lengths.to(self.device)
            
            # 前向传播
            outputs = self.model(images, captions, lengths)
            
            # 计算损失
            # outputs: [batch_size, seq_len, vocab_size]
            # captions: [batch_size, max_length]
            
            # 对于不同解码器的统一处理：
            # - LSTM/GRU解码器：输出长度等于输入长度
            # - Transformer解码器：输出长度 = 输入长度 - 1 (因为去掉了最后一个词)
            
            if self.config['decoder_type'] == 'transformer':
                # Transformer: decoder_input是captions[:-1], 目标是captions[1:]
                targets = captions[:, 1:outputs.size(1)+1]
            else:
                # LSTM/GRU: 输入包含图像特征+captions[:-1], 目标是captions[1:]
                targets = captions[:, 1:outputs.size(1)+1]
            
            # 确保维度匹配
            min_len = min(outputs.size(1), targets.size(1))
            outputs = outputs[:, :min_len, :]
            targets = targets[:, :min_len]
            
            # 展平为2D以便计算损失
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)
            
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
            for batch in tqdm(data_loader, desc="Validating"):
                # 兼容数据加载返回3元组或4元组（带image_paths）
                if len(batch) == 4:
                    images, captions, lengths, _ = batch
                elif len(batch) == 3:
                    images, captions, lengths = batch
                else:
                    raise ValueError(f"不支持的批处理数据格式，期望3或4个元素，得到{len(batch)}个")
                images = images.to(self.device)
                captions = captions.to(self.device)
                lengths = lengths.to(self.device)
                
                outputs = self.model(images, captions, lengths)
                
                # 计算损失：与训练时保持一致的对齐
                if self.config['decoder_type'] == 'transformer':
                    targets = captions[:, 1:outputs.size(1)+1]
                else:
                    targets = captions[:, 1:outputs.size(1)+1]
                
                # 确保维度匹配
                min_len = min(outputs.size(1), targets.size(1))
                outputs = outputs[:, :min_len, :]
                targets = targets[:, :min_len]
                
                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)
                
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练图像描述生成模型')
    
    # 模型参数
    parser.add_argument('--encoder_type', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'resnet152', 'vit_b_16', 'vit_l_16'],
                        help='编码器类型')
    parser.add_argument('--decoder_type', type=str, default='lstm',
                        choices=['lstm', 'gru', 'transformer'],
                        help='解码器类型')
    parser.add_argument('--embed_size', type=int, default=256, help='嵌入维度')
    parser.add_argument('--hidden_size', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=1, help='RNN/Transformer层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout比例')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='梯度裁剪')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    
    # 数据参数
    parser.add_argument('--dataset_path', type=str, default='flickr8k_aim3/dataset_flickr8k.json',
                        help='数据集路径')
    parser.add_argument('--images_dir', type=str, default='flickr8k_aim3/images',
                        help='图像目录')
    parser.add_argument('--vocab_path', type=str, default='flickr8k_aim3/vocabulary.pkl',
                        help='词汇表路径')
    parser.add_argument('--freq_threshold', type=int, default=5, help='词频阈值')
    
    # 保存和日志参数
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='检查点保存目录，默认根据模型类型自动生成')
    parser.add_argument('--save_step', type=int, default=1, help='保存步长')
    parser.add_argument('--log_step', type=int, default=100, help='日志步长')
    
    # Transformer特定参数
    parser.add_argument('--nhead', type=int, default=8, help='Transformer注意力头数')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 构建配置字典
    config = {
        # 数据相关
        'dataset_path': args.dataset_path,
        'images_dir': args.images_dir,
        'vocab_path': args.vocab_path,
        'freq_threshold': args.freq_threshold,
        
        # 模型相关
        'embed_size': args.embed_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'encoder_type': args.encoder_type,
        'decoder_type': args.decoder_type,
        'nhead': args.nhead,  # Transformer专用
        
        # 训练相关
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'num_workers': args.num_workers,
        
        # 保存和日志
        'checkpoint_dir': args.checkpoint_dir,
        'save_step': args.save_step,
        'log_step': args.log_step
    }
    
    # 如果没有指定checkpoint_dir，则自动生成
    if config['checkpoint_dir'] is None:
        config['checkpoint_dir'] = f"checkpoints_{config['encoder_type']}_{config['decoder_type']}"
    
    # 打印配置
    print("训练配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
