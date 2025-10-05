"""
数据集类和数据加载器
处理Flickr8k数据集的图像加载和文本预处理
"""
import json
import os
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class Flickr8kDataset(Dataset):
    """Flickr8k数据集类"""
    
    def __init__(self, 
                 dataset_path: str,
                 images_dir: str,
                 vocabulary,
                 split: str = 'train',
                 transform=None,
                 max_length: int = 50):
        """
        初始化数据集
        Args:
            dataset_path: 数据集JSON文件路径
            images_dir: 图像文件夹路径
            vocabulary: Vocabulary对象
            split: 数据集划分 ('train', 'val', 'test')
            transform: 图像变换
            max_length: 最大序列长度
        """
        self.images_dir = images_dir
        self.vocabulary = vocabulary
        self.split = split
        self.max_length = max_length
        
        # 加载数据集
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 筛选对应划分的数据
        self.samples = []
        for img_data in data['images']:
            if img_data['split'] == split:
                img_path = os.path.join(images_dir, img_data['filename'])
                for sent in img_data['sentences']:
                    self.samples.append({
                        'image_path': img_path,
                        'caption': sent['tokens'],
                        'raw_caption': sent['raw']
                    })
        
        # 图像变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        print(f"{split} 数据集加载完成: {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """
        获取数据样本
        Returns:
            image: 图像张量 [3, 224, 224]
            caption: 标注索引张量 [max_length]
            length: 标注实际长度
        """
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 处理标注
        caption_tokens = [self.vocabulary.start_token] + \
                        sample['caption'] + \
                        [self.vocabulary.end_token]
        
        # 编码为索引
        caption_indices = self.vocabulary.encode(caption_tokens)
        
        # 截断或填充到固定长度
        length = len(caption_indices)
        if length > self.max_length:
            caption_indices = caption_indices[:self.max_length]
            length = self.max_length
        else:
            caption_indices += [self.vocabulary.word2idx[self.vocabulary.pad_token]] * \
                              (self.max_length - length)
        
        caption_tensor = torch.tensor(caption_indices, dtype=torch.long)
        
        return image, caption_tensor, length, sample['image_path']


def collate_fn(batch):
    """
    自定义批处理函数
    将不同长度的标注统一处理
    """
    # 按标注长度排序（降序）
    batch.sort(key=lambda x: x[2], reverse=True)
    
    images, captions, lengths, image_paths = zip(*batch)
    
    # 堆叠图像
    images = torch.stack(images, 0)
    
    # 堆叠标注
    captions = torch.stack(captions, 0)
    
    # 转换长度为张量
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return images, captions, lengths, image_paths


def get_data_loader(dataset_path: str,
                   images_dir: str,
                   vocabulary,
                   split: str = 'train',
                   batch_size: int = 32,
                   shuffle: bool = True,
                   num_workers: int = 4) -> DataLoader:
    """
    创建数据加载器
    Args:
        dataset_path: 数据集JSON文件路径
        images_dir: 图像文件夹路径
        vocabulary: Vocabulary对象
        split: 数据集划分
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 工作进程数
    Returns:
        DataLoader对象
    """
    dataset = Flickr8kDataset(
        dataset_path=dataset_path,
        images_dir=images_dir,
        vocabulary=vocabulary,
        split=split
    )
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return data_loader


if __name__ == "__main__":
    # 测试代码
    from vocabulary import Vocabulary
    
    # 加载词汇表
    vocab = Vocabulary.load("flickr8k_aim3/vocabulary.pkl")
    
    # 创建数据集
    dataset = Flickr8kDataset(
        dataset_path="flickr8k_aim3/dataset_flickr8k.json",
        images_dir="flickr8k_aim3/images",
        vocabulary=vocab,
        split='train'
    )
    
    # 测试获取样本
    image, caption, length, image_path = dataset[0]
    print(f"图像形状: {image.shape}")
    print(f"标注形状: {caption.shape}")
    print(f"标注长度: {length}")
    print(f"图像路径: {image_path}")
    print(f"标注索引: {caption[:length]}")
    print(f"标注文本: {vocab.decode(caption[:length].tolist())}")
    
    # 测试数据加载器
    data_loader = get_data_loader(
        dataset_path="flickr8k_aim3/dataset_flickr8k.json",
        images_dir="flickr8k_aim3/images",
        vocabulary=vocab,
        split='train',
        batch_size=4,
        num_workers=0
    )
    
    # 测试一个批次
    images, captions, lengths, image_paths = next(iter(data_loader))
    print(f"\n批次图像形状: {images.shape}")
    print(f"批次标注形状: {captions.shape}")
    print(f"批次长度: {lengths}")
    print(f"批次图像路径: {image_paths}")
    
    for images, captions, lengths in data_loader:
        print(f"\n批次:")
        print(f"图像批次形状: {images.shape}")
        print(f"标注批次形状: {captions.shape}")
        print(f"长度: {lengths}")
        break
