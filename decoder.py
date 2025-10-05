"""
解码器模块
使用LSTM实现文本生成器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
    def sample_beam_search(self,
                          features: torch.Tensor,
                          start_token: int,
                          end_token: int,
                          max_length: int = 50,
                          beam_width: int = 3) -> torch.Tensor:
        """
        使用束搜索进行采样（单个图像）
        Args:
            features: 图像特征 [1, embed_size]
            start_token: 开始标记索引
            end_token: 结束标记索引
            max_length: 最大生成长度
            beam_width: 束宽度
        Returns:
            best_sequence: 最佳序列 [max_length]
        """
        # 初始化束
        k = beam_width
        sequences = [[start_token]]
        scores = [0.0]
        
        # 初始化LSTM状态
        inputs = features.unsqueeze(1)  # [1, 1, embed_size]
        hiddens, states = self.lstm(inputs)
        
        for _ in range(max_length):
            all_candidates = []
            
            for i, seq in enumerate(sequences):
                if seq[-1] == end_token:
                    all_candidates.append((scores[i], seq))
                    continue
                
                # 获取当前词的嵌入
                current_word = torch.tensor([seq[-1]], device=features.device)
                inputs = self.embed(current_word).unsqueeze(0)  # [1, 1, embed_size]
                
                # LSTM前向传播
                hiddens, new_states = self.lstm(inputs, states)
                
                # 预测下一个词
                outputs = self.fc(hiddens.squeeze(1))  # [1, vocab_size]
                log_probs = F.log_softmax(outputs, dim=1)
                
                # 获取top-k个候选
                top_log_probs, top_indices = log_probs.topk(k)
                
                for j in range(k):
                    candidate_seq = seq + [top_indices[0, j].item()]
                    candidate_score = scores[i] + top_log_probs[0, j].item()
                    all_candidates.append((candidate_score, candidate_seq))
            
            # 选择分数最高的k个序列
            ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
            sequences = [seq for score, seq in ordered[:k]]
            scores = [score for score, seq in ordered[:k]]
            
            # 如果所有序列都结束了，提前停止
            if all(seq[-1] == end_token for seq in sequences):
                break
        
        # 返回分数最高的序列
        best_sequence = sequences[0]
        return torch.tensor(best_sequence)


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


if __name__ == "__main__":
    # 测试解码器
    print("测试LSTM解码器...")
    
    embed_size = 256
    hidden_size = 512
    vocab_size = 5000
    batch_size = 4
    max_length = 20
    
    decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, num_layers=2)
    
    # 创建随机输入
    features = torch.randn(batch_size, embed_size)
    captions = torch.randint(0, vocab_size, (batch_size, max_length))
    lengths = torch.tensor([20, 18, 15, 12])
    
    # 前向传播
    outputs = decoder(features, captions, lengths)
    print(f"图像特征形状: {features.shape}")
    print(f"标注形状: {captions.shape}")
    print(f"输出形状: {outputs.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # 测试采样
    print("\n测试采样生成...")
    sampled = decoder.sample(features[:1], start_token=1, end_token=2, max_length=10)
    print(f"生成序列形状: {sampled.shape}")
    print(f"生成序列: {sampled}")
