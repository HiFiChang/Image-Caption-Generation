"""
完整的图像描述生成模型
整合编码器和解码器
"""
import torch
import torch.nn as nn
from encoder import EncoderCNN, EncoderViT
from decoder import DecoderLSTM, DecoderGRU, DecoderTransformer


class ImageCaptioningModel(nn.Module):
    """图像描述生成模型"""
    
    def __init__(self,
                 embed_size: int = 256,
                 hidden_size: int = 512,
                 vocab_size: int = 5000,
                 num_layers: int = 1,
                 dropout: float = 0.5,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 pad_token_idx: int = 0,
                 encoder_type: str = 'resnet50',
                 decoder_type: str = 'lstm',
                 **kwargs):
        """
        初始化模型
        Args:
            embed_size: 嵌入维度
            hidden_size: 隐藏层维度
            vocab_size: 词汇表大小
            num_layers: RNN层数
            dropout: Dropout比例
            encoder_type: 编码器类型 ('resnet50', 'resnet101', 'vit_b_16')
            decoder_type: 解码器类型 ('lstm', 'gru', 'transformer')
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
        elif decoder_type == 'transformer':
            self.decoder = DecoderTransformer(
                embed_size=embed_size,
                vocab_size=vocab_size,
                num_layers=num_layers,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                pad_token_idx=pad_token_idx,
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
                max_length
            )
        # elif method == 'beam_search':
        #     generated_ids = self.decoder.sample_beam_search(
        #         features,
        #         start_token,
        #         end_token,
        #         max_length,
        #         beam_width
        #     )
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
        
    # def get_trainable_params(self):
    #     """获取可训练参数"""
    #     return filter(lambda p: p.requires_grad, self.parameters())
    
    # def count_parameters(self):
    #     """统计参数数量"""
    #     total = sum(p.numel() for p in self.parameters())
    #     trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
    #     return {
    #         'total': total,
    #         'trainable': trainable,
    #         'frozen': total - trainable
    #     }


def build_model(config: dict) -> ImageCaptioningModel:
    """
    根据配置构建模型
    Args:
        config: 配置字典
    Returns:
        模型实例
    """
    return ImageCaptioningModel(
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        vocab_size=config['vocab_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        encoder_type=config['encoder_type'],
        decoder_type=config['decoder_type'],
        # Transformer特定参数
        nhead=config['nhead'],
        dim_feedforward=config['dim_feedforward'],
        pad_token_idx=config['pad_token_idx']
    )
