"""
词汇表构建模块
用于构建词汇表，实现文本到索引的转换
"""
import json
import pickle
from collections import Counter
from typing import List, Dict


class Vocabulary:
    """词汇表类"""
    
    def __init__(self, freq_threshold: int = 5):
        """
        初始化词汇表
        Args:
            freq_threshold: 词频阈值，低于此阈值的词将被替换为<UNK>
        """
        self.freq_threshold = freq_threshold
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        # 特殊标记
        self.pad_token = "<PAD>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.unk_token = "<UNK>"
        
        # 初始化特殊标记
        self.word2idx = {
            self.pad_token: 0,
            self.start_token: 1,
            self.end_token: 2,
            self.unk_token: 3
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
    def build_vocabulary(self, captions: List[List[str]]):
        """
        从标注构建词汇表
        Args:
            captions: 标注列表，每个标注是一个词的列表
        """
        # 统计词频
        for caption in captions:
            self.word_freq.update(caption)
        
        # 添加频率高于阈值的词
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.freq_threshold and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
                
        print(f"词汇表构建完成！")
        print(f"总词数: {len(self.word_freq)}")
        print(f"词汇表大小: {len(self.word2idx)}")
        print(f"词频阈值: {self.freq_threshold}")
    
    def encode(self, caption: List[str]) -> List[int]:
        """
        将词列表转换为索引列表
        Args:
            caption: 词列表
        Returns:
            索引列表
        """
        return [self.word2idx.get(word, self.word2idx[self.unk_token]) 
                for word in caption]
    
    def decode(self, indices: List[int]) -> List[str]:
        """
        将索引列表转换为词列表
        Args:
            indices: 索引列表
        Returns:
            词列表
        """
        return [self.idx2word.get(idx, self.unk_token) for idx in indices]
    
    def __len__(self):
        """返回词汇表大小"""
        return len(self.word2idx)
    
    def save(self, filepath: str):
        """
        保存词汇表
        Args:
            filepath: 保存路径
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_freq': self.word_freq,
                'freq_threshold': self.freq_threshold
            }, f)
        print(f"词汇表已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        加载词汇表
        Args:
            filepath: 词汇表路径
        Returns:
            Vocabulary对象
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls(freq_threshold=data['freq_threshold'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.word_freq = data['word_freq']
        
        print(f"词汇表已加载，大小: {len(vocab)}")
        return vocab


def build_vocab_from_dataset(dataset_path: str, 
                             freq_threshold: int = 5,
                             save_path: str = None) -> Vocabulary:
    """
    从数据集构建词汇表
    Args:
        dataset_path: 数据集JSON文件路径
        freq_threshold: 词频阈值
        save_path: 保存路径（可选）
    Returns:
        Vocabulary对象
    """
    print(f"从 {dataset_path} 构建词汇表...")
    
    # 加载数据集
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取所有训练集的标注
    captions = []
    for img in data['images']:
        if img['split'] == 'train':  # 只使用训练集构建词汇表
            for sent in img['sentences']:
                captions.append(sent['tokens'])
    
    print(f"训练集标注数量: {len(captions)}")
    
    # 构建词汇表
    vocab = Vocabulary(freq_threshold=freq_threshold)
    vocab.build_vocabulary(captions)
    
    # 保存词汇表
    if save_path:
        vocab.save(save_path)
    
    return vocab


if __name__ == "__main__":
    # 测试代码
    dataset_path = "flickr8k_aim3/dataset_flickr8k.json"
    vocab_path = "flickr8k_aim3/vocabulary.pkl"
    
    # 构建并保存词汇表
    vocab = build_vocab_from_dataset(dataset_path, 
                                     freq_threshold=5,
                                     save_path=vocab_path)
    
    # 测试编码和解码
    test_caption = ["a", "dog", "is", "running", "on", "the", "beach"]
    encoded = vocab.encode(test_caption)
    decoded = vocab.decode(encoded)
    
    print(f"\n测试:")
    print(f"原始: {test_caption}")
    print(f"编码: {encoded}")
    print(f"解码: {decoded}")
