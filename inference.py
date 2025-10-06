"""
推理脚本
用于单张或多张图像的描述生成
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os
from typing import List

from vocabulary import Vocabulary
from model import ImageCaptioningModel, build_model


class ImageCaptioner:
    """图像描述生成器"""
    
    def __init__(self, checkpoint_path: str, device=None):
        """
        初始化描述生成器
        Args:
            checkpoint_path: 模型检查点路径
            device: 设备
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        print(f"加载模型: {checkpoint_path}")
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 获取配置和词汇表
        self.config = checkpoint['config']
        self.vocab = checkpoint['vocab']
        
        # 构建模型
        self.config['vocab_size'] = len(self.vocab)
        self.config['pad_token_idx'] = self.vocab.word2idx.get(self.vocab.pad_token, 0)
        self.model = build_model(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"模型加载完成！")
        print(f"词汇表大小: {len(self.vocab)}")
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """
        加载并预处理图像
        Args:
            image_path: 图像路径
        Returns:
            图像张量 [1, 3, 224, 224]
        """
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)  # 添加批次维度
        return image
    
    def generate_caption(self, 
                        image_path: str,
                        max_length: int = 50,
                        method: str = 'greedy',
                        beam_width: int = 3,
                        temperature: float = 1.0) -> str:
        """
        为单张图像生成描述
        Args:
            image_path: 图像路径
            max_length: 最大生成长度
            method: 生成方法 ('greedy', 'beam_search')
            beam_width: 束搜索宽度
            temperature: 温度参数
        Returns:
            生成的描述文本
        """
        # 加载图像
        image = self.load_image(image_path).to(self.device)
        
        # 生成标注
        with torch.no_grad():
            generated_ids = self.model.generate(
                image,
                start_token=self.vocab.word2idx[self.vocab.start_token],
                end_token=self.vocab.word2idx[self.vocab.end_token],
                max_length=max_length,
                method=method,
                beam_width=beam_width,
                temperature=temperature
            )
        
        # 转换为文本
        if method == 'beam_search':
            generated_ids = generated_ids.unsqueeze(0)  # 添加批次维度
        
        caption = self.ids_to_caption(generated_ids[0])
        
        return caption
    
    def ids_to_caption(self, ids: torch.Tensor) -> str:
        """
        将索引序列转换为文本
        Args:
            ids: 索引张量
        Returns:
            描述文本
        """
        words = []
        for idx in ids.cpu().tolist():
            word = self.vocab.idx2word.get(idx, self.vocab.unk_token)
            
            # 遇到结束标记时停止
            if word == self.vocab.end_token:
                break
            
            # 跳过特殊标记
            if word not in [self.vocab.start_token, self.vocab.pad_token]:
                words.append(word)
        
        # 组合成句子
        caption = ' '.join(words)
        
        # 首字母大写，添加句号
        if caption:
            caption = caption[0].upper() + caption[1:] + '.'
        
        return caption
    
    def generate_multiple_captions(self,
                                  image_path: str,
                                  num_captions: int = 5,
                                  max_length: int = 50,
                                  temperature: float = 1.0) -> List[str]:
        """
        为单张图像生成多个不同的描述（通过采样）
        Args:
            image_path: 图像路径
            num_captions: 生成数量
            max_length: 最大长度
            temperature: 温度参数（越高越随机）
        Returns:
            描述列表
        """
        captions = []
        for _ in range(num_captions):
            caption = self.generate_caption(
                image_path,
                max_length=max_length,
                method='greedy',
                temperature=temperature
            )
            captions.append(caption)
        
        return captions


def visualize_result(image_path: str, caption: str):
    """
    可视化结果（显示图像和描述）
    Args:
        image_path: 图像路径
        caption: 描述文本
    """
    try:
        import matplotlib.pyplot as plt
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 显示
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.title(caption, fontsize=14, wrap=True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib未安装，跳过可视化")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='图像描述生成推理')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--image', type=str, required=True,
                       help='输入图像路径')
    parser.add_argument('--method', type=str, default='greedy',
                       choices=['greedy', 'beam_search'],
                       help='生成方法')
    parser.add_argument('--beam_width', type=int, default=3,
                       help='束搜索宽度')
    parser.add_argument('--max_length', type=int, default=50,
                       help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='温度参数')
    parser.add_argument('--visualize', action='store_true',
                       help='是否可视化结果')
    parser.add_argument('--multiple', type=int, default=0,
                       help='生成多个描述（指定数量）')
    
    args = parser.parse_args()
    
    # 检查图像是否存在
    if not os.path.exists(args.image):
        print(f"错误: 图像不存在: {args.image}")
        return
    
    # 创建描述生成器
    captioner = ImageCaptioner(args.checkpoint)
    
    print(f"\n{'='*50}")
    print(f"图像: {args.image}")
    print(f"生成方法: {args.method}")
    print(f"{'='*50}\n")
    
    # 生成描述
    if args.multiple > 0:
        print(f"生成 {args.multiple} 个不同的描述:\n")
        captions = captioner.generate_multiple_captions(
            args.image,
            num_captions=args.multiple,
            max_length=args.max_length,
            temperature=args.temperature
        )
        for i, caption in enumerate(captions, 1):
            print(f"{i}. {caption}")
    else:
        caption = captioner.generate_caption(
            args.image,
            max_length=args.max_length,
            method=args.method,
            beam_width=args.beam_width,
            temperature=args.temperature
        )
        print(f"生成的描述: {caption}")
        
        # 可视化
        if args.visualize:
            visualize_result(args.image, caption)
    
    print(f"\n{'='*50}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        print("使用示例:")
        print("python inference.py --checkpoint checkpoints/best_model.pth --image flickr8k_aim3/images/1000268201_693b08cb0e.jpg")
        print("\n其他选项:")
        print("  --method beam_search    # 使用束搜索")
        print("  --beam_width 5          # 束搜索宽度")
        print("  --visualize             # 可视化结果")
        print("  --multiple 5            # 生成5个不同的描述")
    else:
        main()
