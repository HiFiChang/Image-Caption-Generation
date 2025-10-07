"""
解码器模块
使用LSTM实现文本生成器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DecoderLSTM(nn.Module):
    """LSTM解码器"""
    
    def __init__(self, 
                 embed_size: int,
                 hidden_size: int,
                 vocab_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.5):
        """
        初始化解码器
        Args:
            embed_size: 嵌入维度
            hidden_size: 隐藏层维度
            vocab_size: 词汇表大小
            num_layers: LSTM层数
            dropout: Dropout比例
        """
        super(DecoderLSTM, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # 词嵌入层
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层（输出层）
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # 初始化权重
        self.init_weights()
        
    def init_weights(self):
        """初始化权重"""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        
    def forward(self, 
                features: torch.Tensor, 
                captions: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """
        训练时的前向传播
        Args:
            features: 图像特征 [batch_size, embed_size]
            captions: 标注索引 [batch_size, max_length]
            lengths: 标注长度 [batch_size]
        Returns:
            outputs: 预测输出 [batch_size, max_length, vocab_size]
        """
        # 保存原始max_length
        batch_size, max_length = captions.size()
        
        # 词嵌入
        embeddings = self.embed(captions)  # [batch_size, max_length, embed_size]
        
        # 将图像特征作为第一个时间步
        features = features.unsqueeze(1)  # [batch_size, 1, embed_size]
        embeddings = torch.cat([features, embeddings[:, :-1, :]], dim=1)  # [batch_size, max_length, embed_size]
        
        # 使用pack_padded_sequence处理变长序列
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, 
            lengths.cpu(), 
            batch_first=True,
            enforce_sorted=True  # collate_fn已经排序
        )
        
        # LSTM前向传播
        hiddens, _ = self.lstm(packed)
        
        # 解包，指定total_length确保输出长度一致
        hiddens, _ = nn.utils.rnn.pad_packed_sequence(
            hiddens, 
            batch_first=True,
            total_length=max_length
        )
        
        # Dropout
        hiddens = self.dropout(hiddens)
        
        # 全连接层
        outputs = self.fc(hiddens)  # [batch_size, max_length, vocab_size]
        
        return outputs
    
    def sample(self, 
               features: torch.Tensor, 
               start_token: int,
               end_token: int,
               max_length: int = 50,
               temperature: float = 1.0) -> torch.Tensor:
        """
        推理时的采样生成
        Args:
            features: 图像特征 [batch_size, embed_size]
            start_token: 开始标记索引
            end_token: 结束标记索引
            max_length: 最大生成长度
            temperature: 温度参数（控制随机性）
        Returns:
            sampled_ids: 生成的词索引序列 [batch_size, max_length]
        """
        batch_size = features.size(0)
        sampled_ids = []
        
        # 初始化输入
        inputs = features.unsqueeze(1)  # [batch_size, 1, embed_size]
        states = None
        
        for i in range(max_length):
            # LSTM前向传播
            hiddens, states = self.lstm(inputs, states)  # hiddens: [batch_size, 1, hidden_size]
            
            # 预测下一个词
            outputs = self.fc(hiddens.squeeze(1))  # [batch_size, vocab_size]
            
            # 应用温度参数
            outputs = outputs / temperature
            
            # 采样（选择概率最高的词）
            _, predicted = outputs.max(1)  # [batch_size]
            sampled_ids.append(predicted)
            
            # 准备下一个时间步的输入
            inputs = self.embed(predicted).unsqueeze(1)  # [batch_size, 1, embed_size]
        
        sampled_ids = torch.stack(sampled_ids, 1)  # [batch_size, max_length]
        return sampled_ids
    
class PositionalEncoding(nn.Module):
    """位置编码，为序列中的每个位置添加一个唯一的编码"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embed_size]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DecoderTransformer(nn.Module):
    """Transformer解码器"""
    
    def __init__(self, 
                 embed_size: int,
                 vocab_size: int,
                 num_layers: int,
                 nhead: int,
                 dim_feedforward: int,
                 pad_token_idx: int,
                 dropout: float = 0.5,
                 **kwargs):
        """
        初始化Transformer解码器
        Args:
            embed_size: 嵌入维度 (d_model)
            vocab_size: 词汇表大小
            num_layers: Transformer解码器层数
            nhead: 多头注意力头数
            dim_feedforward: 前馈网络维度
            pad_token_idx: PAD标记的索引
            dropout: Dropout比例
        """
        super(DecoderTransformer, self).__init__()
        
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.pad_token_idx = pad_token_idx
        
        # 词嵌入和位置编码
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 输出层
        self.fc = nn.Linear(embed_size, vocab_size)
        
        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, features: torch.Tensor, captions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        训练时的前向传播
        Args:
            features: 图像特征 [batch_size, embed_size]
            captions: 标注索引 [batch_size, max_length]
            lengths: 标注长度（Transformer中主要用于生成padding mask）
        Returns:
            outputs: 预测输出 [batch_size, max_length, vocab_size]
        """
        batch_size, max_len = captions.size()
        
        # 1. 准备解码器输入：右移一位，去掉最后一个词
        # captions: [<START>, w1, w2, ..., w_n, <END>]
        # decoder_input: [<START>, w1, w2, ..., w_n]
        decoder_input = captions[:, :-1]
        
        # 2. 词嵌入和位置编码
        embeddings = self.embed(decoder_input) * math.sqrt(self.embed_size)
        embeddings = self.pos_encoder(embeddings)
        
        # 图像特征作为memory (不重复，让注意力机制处理维度对齐)
        memory = features.unsqueeze(1)  # [batch_size, 1, embed_size]
        
        # 3. 生成掩码
        # 目标序列掩码 (tgt_mask): 防止看到未来的词
        tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(1)).to(features.device)
        
        # 填充掩码 (tgt_key_padding_mask): 忽略<PAD>标记
        tgt_key_padding_mask = (decoder_input == self.pad_token_idx)

        # 4. Transformer解码
        output = self.transformer_decoder(
            tgt=embeddings,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # 输出层
        outputs = self.fc(output)
        return outputs

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成一个上三角矩阵的掩码，用于防止序列中的位置关注到后续位置"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def sample(self, 
               features: torch.Tensor, 
               start_token: int,
               end_token: int,
               max_length: int = 50,
               **kwargs) -> torch.Tensor:
        """
        推理时的贪心采样
        """
        batch_size = features.size(0)
        # 初始化生成的序列，以<START>标记开始
        generated_ids = torch.full((batch_size, 1), start_token, dtype=torch.long, device=features.device)
        
        memory = features.unsqueeze(1) # [batch_size, 1, embed_size]

        for _ in range(max_length - 1):
            # 获取当前序列的嵌入和位置编码
            tgt_embed = self.embed(generated_ids) * math.sqrt(self.embed_size)
            tgt_embed = self.pos_encoder(tgt_embed)
            
            # 生成目标序列掩码
            tgt_mask = self.generate_square_subsequent_mask(generated_ids.size(1)).to(features.device)
            
            # Transformer解码
            output = self.transformer_decoder(tgt_embed, memory, tgt_mask=tgt_mask)
            
            # 取最后一个时间步的输出
            last_output = output[:, -1, :]
            
            # 预测下一个词
            logits = self.fc(last_output)
            _, next_word = torch.max(logits, dim=1)
            
            # 将新生成的词添加到序列中
            generated_ids = torch.cat([generated_ids, next_word.unsqueeze(1)], dim=1)
            
            # 如果所有批次都生成了<END>标记，则提前停止
            if (next_word == end_token).all():
                break
        
        return generated_ids


class DecoderGRU(nn.Module):
    """GRU解码器（备选方案）"""
    
    def __init__(self, 
                 embed_size: int,
                 hidden_size: int,
                 vocab_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.5):
        """
        初始化GRU解码器
        Args:
            embed_size: 嵌入维度
            hidden_size: 隐藏层维度
            vocab_size: 词汇表大小
            num_layers: GRU层数
            dropout: Dropout比例
        """
        super(DecoderGRU, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # 词嵌入层
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # GRU层
        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.init_weights()
        
    def init_weights(self):
        """初始化权重"""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths):
        """前向传播（与DecoderLSTM类似）"""
        # 保存原始max_length
        batch_size, max_length = captions.size()
        
        embeddings = self.embed(captions)
        features = features.unsqueeze(1)
        embeddings = torch.cat([features, embeddings[:, :-1, :]], dim=1)
        
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        
        hiddens, _ = self.gru(packed)
        # 指定total_length确保输出长度一致
        hiddens, _ = nn.utils.rnn.pad_packed_sequence(
            hiddens, batch_first=True, total_length=max_length
        )
        hiddens = self.dropout(hiddens)
        outputs = self.fc(hiddens)
        
        return outputs
    
    def sample(self, features, start_token, end_token, max_length=50, temperature=1.0):
        """贪心采样（与DecoderLSTM类似）"""
        batch_size = features.size(0)
        sampled_ids = []
        
        inputs = features.unsqueeze(1)
        states = None
        
        for _ in range(max_length):
            hiddens, states = self.gru(inputs, states)
            outputs = self.fc(hiddens.squeeze(1))
            outputs = outputs / temperature
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted).unsqueeze(1)
            
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids
