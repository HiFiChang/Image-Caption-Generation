"""
评估脚本
实现BLEU、METEOR、ROUGE、CIDEr等评估指标
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
import json
from typing import List, Dict

# CIDEr和其他指标（需要pycocoevalcap）
try:
    from pycocoevalcap.cider.cider import Cider
    CIDER_AVAILABLE = True
except ImportError:
    CIDER_AVAILABLE = False
    print("警告: pycocoevalcap未安装，CIDEr指标不可用")

# ROUGE指标
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("警告: rouge-score未安装，ROUGE指标不可用")

from vocabulary import Vocabulary
from dataset import get_data_loader
from model import ImageCaptioningModel


# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class Evaluator:
    """评估器类"""
    
    def __init__(self, model: ImageCaptioningModel, vocab: Vocabulary, device):
        """
        初始化评估器
        Args:
            model: 图像描述生成模型
            vocab: 词汇表
            device: 设备
        """
        self.model = model
        self.vocab = vocab
        self.device = device
        self.model.eval()
        
    def generate_captions(self, data_loader, max_length: int = 50) -> tuple:
        """
        为数据集生成标注
        Args:
            data_loader: 数据加载器
            max_length: 最大生成长度
        Returns:
            references: 参考标注列表
            hypotheses: 生成标注列表
        """
        references_map = {}  # {image_path: [ref1, ref2, ...]}
        hypotheses_map = {} # {image_path: hypothesis}
        
        print("正在生成标注...")
        with torch.no_grad():
            for images, captions, lengths, image_paths in tqdm(data_loader):
                images = images.to(self.device)
                
                # 生成标注
                generated_ids = self.model.generate(
                    images,
                    start_token=self.vocab.word2idx[self.vocab.start_token],
                    end_token=self.vocab.word2idx[self.vocab.end_token],
                    max_length=max_length,
                    method='greedy'
                )
                
                # 转换为词列表
                for i in range(images.size(0)):
                    image_path = image_paths[i]

                    # 参考标注（真实标注）
                    ref_indices = captions[i, :lengths[i]].cpu().tolist()
                    ref_words = self.vocab.decode(ref_indices)
                    # 移除特殊标记
                    ref_words = [w for w in ref_words if w not in 
                                [self.vocab.start_token, self.vocab.end_token, 
                                 self.vocab.pad_token]]
                    
                    if image_path not in references_map:
                        references_map[image_path] = []
                    references_map[image_path].append(ref_words)
                    
                    # 生成标注 (只为每个图像生成一次)
                    if image_path not in hypotheses_map:
                        gen_indices = generated_ids[i].cpu().tolist()
                        gen_words = self.vocab.decode(gen_indices)
                        # 移除特殊标记，并在遇到END标记时停止
                        gen_words_clean = []
                        for w in gen_words:
                            if w == self.vocab.end_token:
                                break
                            if w not in [self.vocab.start_token, self.vocab.pad_token]:
                                gen_words_clean.append(w)
                        hypotheses_map[image_path] = gen_words_clean

        # 将map转换为评估函数所需的列表格式
        # 确保references和hypotheses的顺序一致
        references = list(references_map.values())
        hypotheses = [hypotheses_map[path] for path in references_map.keys()]

        return references, hypotheses
    
    def calculate_bleu(self, references: List, hypotheses: List) -> Dict:
        """
        计算BLEU分数
        Args:
            references: 参考标注列表
            hypotheses: 生成标注列表
        Returns:
            BLEU分数字典
        """
        print("\n计算BLEU分数...")
        
        # BLEU-1到BLEU-4
        bleu_scores = {}
        
        for n in range(1, 5):
            weights = tuple([1.0/n] * n + [0.0] * (4-n))
            score = corpus_bleu(references, hypotheses, weights=weights)
            bleu_scores[f'BLEU-{n}'] = score
            print(f"BLEU-{n}: {score:.4f}")
        
        return bleu_scores
    
    def calculate_meteor(self, references: List, hypotheses: List) -> float:
        """
        计算METEOR分数
        Args:
            references: 参考标注列表
            hypotheses: 生成标注列表
        Returns:
            METEOR分数
        """
        print("\n计算METEOR分数...")
        
        scores = []
        failed_count = 0
        for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
            # METEOR需要字符串形式
            ref_strs = [' '.join(r) for r in ref]
            hyp_str = ' '.join(hyp)
            try:
                score = meteor_score(ref_strs, hyp_str)
                scores.append(score)
            except Exception as e:
                if failed_count == 0:  # 只打印第一个错误
                    print(f"  警告: METEOR计算失败 (样本 {i}): {e}")
                failed_count += 1
                scores.append(0.0)
        
        if failed_count > 0:
            print(f"  共有 {failed_count}/{len(references)} 个样本METEOR计算失败")
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"METEOR: {avg_score:.4f}")
        
        return avg_score
    
    def calculate_cider(self, references: List, hypotheses: List) -> float:
        """
        计算CIDEr分数
        CIDEr (Consensus-based Image Description Evaluation) 是专门为图像描述任务设计的指标
        它考虑了人类标注的共识，对于捕捉图像的重要细节给予更高的权重
        Args:
            references: 参考标注列表
            hypotheses: 生成标注列表
        Returns:
            CIDEr分数
        """
        if not CIDER_AVAILABLE:
            print("\n跳过CIDEr计算（pycocoevalcap未安装）")
            return 0.0
        
        print("\n计算CIDEr分数...")
        
        try:
            # 将数据转换为CIDEr所需的格式
            # CIDEr需要字典格式: {id: [captions]}
            gts = {}  # ground truth
            res = {}  # results
            
            for i, (ref_list, hyp) in enumerate(zip(references, hypotheses)):
                # 参考标注（可以有多个）
                gts[i] = [' '.join(r) for r in ref_list]
                # 生成标注（只有一个）
                res[i] = [' '.join(hyp)]
            
            # 计算CIDEr
            cider_scorer = Cider()
            score, scores = cider_scorer.compute_score(gts, res)
            
            print(f"CIDEr: {score:.4f}")
            
            return score
        except Exception as e:
            print(f"CIDEr计算失败: {e}")
            return 0.0
    
    def calculate_rouge(self, references: List, hypotheses: List) -> Dict:
        """
        计算ROUGE分数
        ROUGE (Recall-Oriented Understudy for Gisting Evaluation) 主要用于摘要任务
        在图像描述中也可以作为辅助指标
        Args:
            references: 参考标注列表
            hypotheses: 生成标注列表
        Returns:
            ROUGE分数字典
        """
        if not ROUGE_AVAILABLE:
            print("\n跳过ROUGE计算（rouge-score未安装）")
            return {}
        
        print("\n计算ROUGE分数...")
        
        try:
            # 初始化ROUGE评分器（计算ROUGE-1, ROUGE-2, ROUGE-L）
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for ref_list, hyp in zip(references, hypotheses):
                ref_strs = [' '.join(r) for r in ref_list]
                hyp_str = ' '.join(hyp)
                
                if not ref_strs or not hyp_str:
                    continue
                
                # ROUGE-score can handle multiple references
                scores = scorer.score_multi(ref_strs, hyp_str)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            # 计算平均分
            rouge_results = {
                'ROUGE-1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
                'ROUGE-2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
                'ROUGE-L': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
            }
            
            for metric, score in rouge_results.items():
                print(f"{metric}: {score:.4f}")
            
            return rouge_results
        except Exception as e:
            print(f"ROUGE计算失败: {e}")
            return {}
    
    def evaluate(self, data_loader, max_length: int = 50, save_results: bool = True) -> Dict:
        """
        完整评估流程
        Args:
            data_loader: 数据加载器
            max_length: 最大生成长度
            save_results: 是否保存结果
        Returns:
            评估指标字典
        """
        # 生成标注
        references, hypotheses = self.generate_captions(data_loader, max_length)
        
        # 计算各项指标
        results = {}
        
        # BLEU分数
        bleu_scores = self.calculate_bleu(references, hypotheses)
        results.update(bleu_scores)
        
        # METEOR分数
        try:
            meteor = self.calculate_meteor(references, hypotheses)
            results['METEOR'] = meteor
        except Exception as e:
            print(f"METEOR计算失败: {e}")
            results['METEOR'] = 0.0
        
        # CIDEr分数
        try:
            cider = self.calculate_cider(references, hypotheses)
            results['CIDEr'] = cider
        except Exception as e:
            print(f"CIDEr计算失败: {e}")
            results['CIDEr'] = 0.0
        
        # ROUGE分数
        try:
            rouge_scores = self.calculate_rouge(references, hypotheses)
            results.update(rouge_scores)
        except Exception as e:
            print(f"ROUGE计算失败: {e}")
        
        # 打印汇总
        print("\n" + "="*50)
        print("评估结果汇总:")
        print("="*50)
        for metric, score in results.items():
            print(f"{metric}: {score:.4f}")
        print("="*50)
        
        # 保存结果
        if save_results:
            self.save_results(results, references, hypotheses)
        
        return results
    
    def save_results(self, metrics: Dict, references: List, hypotheses: List):
        """
        保存评估结果
        Args:
            metrics: 评估指标
            references: 参考标注
            hypotheses: 生成标注
        """
        results = {
            'metrics': metrics,
            'samples': []
        }
        
        # 保存一些样例
        num_samples = min(100, len(references))
        for i in range(num_samples):
            results['samples'].append({
                'references': [' '.join(r) for r in references[i]],
                'hypothesis': ' '.join(hypotheses[i])
            })
        
        # 保存到文件
        with open('evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n评估结果已保存到: evaluation_results.json")


def load_model_from_checkpoint(checkpoint_path: str, device):
    """
    从检查点加载模型
    Args:
        checkpoint_path: 检查点路径
        device: 设备
    Returns:
        model, vocab
    """
    print(f"从检查点加载模型: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取配置和词汇表
    config = checkpoint['config']
    vocab = checkpoint['vocab']
    
    # 构建模型
    from model import build_model
    model = build_model(config, vocab_size=len(vocab))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"模型加载完成 (Epoch {checkpoint['epoch']+1}, Loss: {checkpoint['loss']:.4f})")
    
    return model, vocab


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='评估图像描述生成模型')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='评估的数据集划分')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--max_length', type=int, default=50,
                       help='最大生成长度')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model, vocab = load_model_from_checkpoint(args.checkpoint, device)
    
    # 创建数据加载器
    data_loader = get_data_loader(
        dataset_path='flickr8k_aim3/dataset_flickr8k.json',
        images_dir='flickr8k_aim3/images',
        vocabulary=vocab,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"\n评估 {args.split} 数据集")
    print(f"样本数: {len(data_loader.dataset)}")
    
    # 创建评估器并评估
    evaluator = Evaluator(model, vocab, device)
    results = evaluator.evaluate(data_loader, max_length=args.max_length)


if __name__ == "__main__":
    # 如果直接运行，使用默认参数
    import sys
    if len(sys.argv) == 1:
        print("使用示例: python evaluate.py --checkpoint checkpoints/best_model.pth --split test")
        print("\n使用默认配置进行测试...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 这里仅作为示例，实际使用时需要提供检查点路径
        print("\n注意: 请先训练模型，然后使用以下命令评估:")
        print("python evaluate.py --checkpoint checkpoints/best_model.pth --split test")
    else:
        main()
