"""
编码器模块
使用预训练的ResNet提取图像特征
"""
import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """CNN编码器，使用预训练的ResNet"""
    
    def __init__(self, embed_size: int = 256, model_name: str = 'resnet50', pretrained: bool = True):
        """
        初始化编码器
        Args:
            embed_size: 嵌入维度
            model_name: 预训练模型名称 ('resnet50', 'resnet101', 'resnet152')
            pretrained: 是否使用预训练权重
        """
        super(EncoderCNN, self).__init__()
        
        self.embed_size = embed_size
        
        # 加载预训练模型
        if model_name == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
        elif model_name == 'resnet152':
            resnet = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 移除最后的全连接层和平均池化层
        modules = list(resnet.children())[:-1]  # 去掉fc层
        self.resnet = nn.Sequential(*modules)
        
        # 添加自定义的全连接层
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
        # 初始化全连接层权重
        self.init_weights()
        
    def init_weights(self):
        """初始化权重"""
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            images: 图像张量 [batch_size, 3, 224, 224]
        Returns:
            features: 特征向量 [batch_size, embed_size]
        """
        with torch.no_grad():
            features = self.resnet(images)  # [batch_size, 2048, 1, 1]

        features = features.reshape(features.size(0), -1)  # [batch_size, 2048]
        features = self.fc(features)  # [batch_size, embed_size]
        features = self.bn(features)
        
        return features
    
    def fine_tune(self, fine_tune: bool = True):
        """
        设置是否微调ResNet
        Args:
            fine_tune: 是否允许梯度更新
        """
        for param in self.resnet.parameters():
            param.requires_grad = fine_tune


class EncoderViT(nn.Module):
    """Vision Transformer 编码器（可选）"""
    
    def __init__(self, embed_size: int = 256, model_name: str = 'vit_b_16', pretrained: bool = True):
        """
        初始化ViT编码器
        Args:
            embed_size: 嵌入维度
            model_name: 模型名称
            pretrained: 是否使用预训练权重
        """
        super(EncoderViT, self).__init__()
        
        self.embed_size = embed_size
        
        # 加载预训练的ViT模型
        if model_name == 'vit_b_16':
            vit = models.vit_b_16(pretrained=pretrained)
        elif model_name == 'vit_l_16':
            vit = models.vit_l_16(pretrained=pretrained)
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 获取特征提取部分
        self.vit = vit
        hidden_dim = vit.hidden_dim
        
        # 移除分类头
        self.vit.heads = nn.Identity()
        
        # 添加自定义的投影层
        self.projection = nn.Linear(hidden_dim, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
        self.init_weights()
        
    def init_weights(self):
        """初始化权重"""
        self.projection.weight.data.normal_(0.0, 0.02)
        self.projection.bias.data.fill_(0)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            images: 图像张量 [batch_size, 3, 224, 224]
        Returns:
            features: 特征向量 [batch_size, embed_size]
        """
        # features = self.vit(images)  # [batch_size, hidden_dim]
        with torch.no_grad():
            features = self.vit(images)  # [batch_size, hidden_dim]


        features = self.projection(features)  # [batch_size, embed_size]
        features = self.bn(features)
        
        return features
    
    def fine_tune(self, fine_tune: bool = True):
        """
        设置是否微调ViT
        Args:
            fine_tune: 是否允许梯度更新
        """
        for param in self.vit.parameters():
            param.requires_grad = fine_tune
