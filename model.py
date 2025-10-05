"""
完整的图像描述生成模型
整合编码器和解码器
"""
import torch
import torch.nn as nn
from encoder import EncoderCNN, EncoderViT
from decoder import DecoderLSTM, DecoderGRU


class ImageCaptioningModel(nn.Module):
    """图像描述生成模型"""
    
    def __init__(self,
                 embed_size: int = 256,
                 hidden_size: int = 512,
                 vocab_size: int = 5000,
                 num_layers: int = 1,
                 dropout: float = 0.5,
                 encoder_type: str = 'resnet50',
                 decoder_type: str = 'lstm'):
        """
        初始化模型
        Args:
            embed_size: 嵌入维度
            hidden_size: 隐藏层维度
            vocab_size: 词汇表大小
            num_layers: RNN层数
            dropout: Dropout比例
            encoder_type: 编码器类型 ('resnet50', 'resnet101', 'vit_b_16')
            decoder_type: 解码器类型 ('lstm', 'gru')
        """
        super(ImageCaptioningModel, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # 初始化编码器
        if encoder_type in ['resnet50', 'resnet101', 'resnet152']:
            self.encoder = EncoderCNN(
                embed_size=embed_size,
                model_name=encoder_type,
                pretrained=True
            )
        elif encoder_type in ['vit_b_16', 'vit_l_16']:
            self.encoder = EncoderViT(
                embed_size=embed_size,
                model_name=encoder_type,
                pretrained=True
            )
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
        
        # 初始化解码器
        if decoder_type == 'lstm':
            self.decoder = DecoderLSTM(
                embed_size=embed_size,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                dropout=dropout
            )
        elif decoder_type == 'gru':
            self.decoder = DecoderGRU(
                embed_size=embed_size,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"不支持的解码器类型: {decoder_type}")
        
        print(f"模型初始化完成:")
        print(f"  编码器: {encoder_type}")
        print(f"  解码器: {decoder_type}")
        print(f"  嵌入维度: {embed_size}")
        print(f"  隐藏层维度: {hidden_size}")
        print(f"  词汇表大小: {vocab_size}")
        
    def forward(self, images, captions, lengths):
        """
        训练时的前向传播
        Args:
            images: 图像张量 [batch_size, 3, 224, 224]
            captions: 标注索引 [batch_size, max_length]
            lengths: 标注长度 [batch_size]
        Returns:
            outputs: 预测输出 [batch_size, max_length, vocab_size]
        """
        # 编码图像
        features = self.encoder(images)
        
        # 解码生成标注
        outputs = self.decoder(features, captions, lengths)
        
        return outputs
    
    def generate(self, 
                images,
                start_token: int,
                end_token: int,
                max_length: int = 50,
                method: str = 'greedy',
                beam_width: int = 3,
                temperature: float = 1.0):
        """
        推理时生成标注
        Args:
            images: 图像张量 [batch_size, 3, 224, 224]
            start_token: 开始标记索引
            end_token: 结束标记索引
            max_length: 最大生成长度
            method: 生成方法 ('greedy', 'beam_search')
            beam_width: 束搜索宽度
            temperature: 温度参数
        Returns:
            generated_ids: 生成的词索引序列
        """
        # 编码图像
        with torch.no_grad():
            features = self.encoder(images)
        
        # 根据方法生成标注
        if method == 'greedy':
            generated_ids = self.decoder.sample(
                features, 
                start_token, 
                end_token,
                max_length,
                temperature
            )
        elif method == 'beam_search':
            if images.size(0) > 1:
                raise ValueError("束搜索仅支持单张图像")
            generated_ids = self.decoder.sample_beam_search(
                features,
                start_token,
                end_token,
                max_length,
                beam_width
            )
        else:
            raise ValueError(f"不支持的生成方法: {method}")
        
        return generated_ids
    
    def fine_tune_encoder(self, fine_tune: bool = True):
        """
        设置是否微调编码器
        Args:
            fine_tune: 是否允许梯度更新
        """
        self.encoder.fine_tune(fine_tune)
        
    def get_trainable_params(self):
        """获取可训练参数"""
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def count_parameters(self):
        """统计参数数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable
        }


def build_model(config: dict, vocab_size: int) -> ImageCaptioningModel:
    """
    根据配置构建模型
    Args:
        config: 配置字典
        vocab_size: 词汇表大小
    Returns:
        ImageCaptioningModel对象
    """
    model = ImageCaptioningModel(
        embed_size=config.get('embed_size', 256),
        hidden_size=config.get('hidden_size', 512),
        vocab_size=vocab_size,
        num_layers=config.get('num_layers', 1),
        dropout=config.get('dropout', 0.5),
        encoder_type=config.get('encoder_type', 'resnet50'),
        decoder_type=config.get('decoder_type', 'lstm')
    )
    
    return model


if __name__ == "__main__":
    # 测试模型
    print("测试图像描述生成模型...\n")
    
    # 配置
    config = {
        'embed_size': 256,
        'hidden_size': 512,
        'num_layers': 1,
        'dropout': 0.5,
        'encoder_type': 'resnet50',
        'decoder_type': 'lstm'
    }
    
    vocab_size = 5000
    batch_size = 4
    max_length = 20
    
    # 构建模型
    model = build_model(config, vocab_size)
    
    # 统计参数
    param_stats = model.count_parameters()
    print(f"\n参数统计:")
    print(f"  总参数: {param_stats['total']:,}")
    print(f"  可训练参数: {param_stats['trainable']:,}")
    print(f"  冻结参数: {param_stats['frozen']:,}")
    
    # 创建随机输入
    images = torch.randn(batch_size, 3, 224, 224)
    captions = torch.randint(0, vocab_size, (batch_size, max_length))
    lengths = torch.tensor([20, 18, 15, 12])
    
    # 训练模式前向传播
    print(f"\n训练模式测试:")
    outputs = model(images, captions, lengths)
    print(f"  输入图像形状: {images.shape}")
    print(f"  输入标注形状: {captions.shape}")
    print(f"  输出形状: {outputs.shape}")
    
    # 推理模式测试
    print(f"\n推理模式测试:")
    model.eval()
    generated = model.generate(
        images[:1], 
        start_token=1, 
        end_token=2,
        max_length=15,
        method='greedy'
    )
    print(f"  生成序列形状: {generated.shape}")
    print(f"  生成序列: {generated}")
