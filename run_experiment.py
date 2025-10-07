#!/usr/bin/env python
"""
单一自动化实验脚本

本脚本实现对不同编码器-解码器组合的自动化、串行化训练与评估。
每完成一个组合的训练和评估后，结果会立即追加到CSV文件中。
"""
import os
import sys
import subprocess
import csv
import re
from itertools import product
from datetime import datetime
from pathlib import Path

ENCODERS = ['resnet50', 'resnet101', 'vit_b_16', 'vit_l_16']
DECODERS = ['lstm', 'gru', 'transformer']

RESULTS_CSV_FILE = 'experiment_results.csv'

TRAIN_SCRIPT = 'train.py'
EVALUATE_SCRIPT = 'evaluate.py'


def log(message):
    """打印带时间戳的日志信息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def check_required_files():
    """检查必要的脚本是否存在"""
    if not os.path.exists(TRAIN_SCRIPT):
        log(f"错误: 训练脚本 '{TRAIN_SCRIPT}' 不存在。")
        return False
    if not os.path.exists(EVALUATE_SCRIPT):
        log(f"错误: 评估脚本 '{EVALUATE_SCRIPT}' 不存在。")
        return False
    return True

def run_command(command):
    """执行一个shell命令并返回其输出"""
    log(f"执行命令: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=False  # 不在返回码非0时抛出异常
        )
        return result
    except Exception as e:
        log(f"命令执行异常: {e}")
        return None

def parse_evaluation_output(output_text: str) -> dict:
    metrics = {}
    patterns = {
        'BLEU-1': r'BLEU-1:\s*([\d.]+)',
        'BLEU-2': r'BLEU-2:\s*([\d.]+)',
        'BLEU-3': r'BLEU-3:\s*([\d.]+)',
        'BLEU-4': r'BLEU-4:\s*([\d.]+)',
        'METEOR': r'METEOR:\s*([\d.]+)',
        'ROUGE-L': r'ROUGE-L:\s*([\d.]+)',
        'CIDEr': r'CIDEr:\s*([\d.]+)',
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, output_text)
        if match:
            metrics[name] = float(match.group(1))
        else:
            metrics[name] = 'N/A'
    return metrics

def initialize_csv():
    if not os.path.exists(RESULTS_CSV_FILE):
        log(f"创建新的结果文件: {RESULTS_CSV_FILE}")
        with open(RESULTS_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = [
                'combination', 'encoder', 'decoder', 'training_status',
                'evaluation_status', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4',
                'METEOR', 'ROUGE-L', 'CIDEr', 'timestamp'
            ]
            writer.writerow(header)

def append_to_csv(result_data: dict):
    with open(RESULTS_CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=result_data.keys())
        writer.writerow(result_data)
    log(f"结果已保存到: {RESULTS_CSV_FILE}")

def main():
    if not check_required_files():
        return

    initialize_csv()

    combinations = list(product(ENCODERS, DECODERS))
    total_combinations = len(combinations)
    log(f"实验开始，总共 {total_combinations} 种组合。")
    log("="*80)

    for i, (encoder, decoder) in enumerate(combinations):
        combination_name = f"{encoder}_{decoder}"
        log(f"开始处理组合 [{i+1}/{total_combinations}]: {combination_name}")

        result_data = {
            'combination': combination_name,
            'encoder': encoder,
            'decoder': decoder,
            'training_status': 'failed',
            'evaluation_status': 'skipped',
            'BLEU-1': 'N/A', 'BLEU-2': 'N/A', 'BLEU-3': 'N/A', 'BLEU-4': 'N/A',
            'METEOR': 'N/A', 'ROUGE-L': 'N/A', 'CIDEr': 'N/A',
            'timestamp': datetime.now().isoformat()
        }

        # --- 1. 训练阶段 ---
        train_command = [
            sys.executable, TRAIN_SCRIPT,
            '--encoder_type', encoder,
            '--decoder_type', decoder
        ]
        
        # 根据decoder类型调整参数
        if decoder == 'transformer':
            # Transformer专用参数
            train_command.extend([
                '--embed_size', '512',      # Transformer需要更大的嵌入维度
                '--dim_feedforward', '2048',    # dim_feedforward = 4 * embed_size
                '--num_layers', '6',
                '--dropout', '0.2',         # Transformer的dropout通常较小（0.1-0.3）
                '--batch_size', '64',
                '--learning_rate', '5e-5',  # Transformer需要更小的学习率（避免训练不稳定）
                '--nhead', '8'              # 8头注意力（512/8=64，标准配置）
            ])
        elif decoder in ['lstm', 'gru']:
            # RNN系列参数
            train_command.extend([
                '--embed_size', '256',      # RNN用较小维度即可
                '--hidden_size', '512',     # LSTM/GRU隐藏层大小
                '--num_layers', '1',        # 单层RNN避免过拟合
                '--dropout', '0.5',         # 标准dropout
                '--batch_size', '64',       # RNN显存需求小
                '--learning_rate', '1e-3'   # Adam标准学习率
            ])
        
        train_result = run_command(train_command)

        if train_result and train_result.returncode == 0:
            log(f"组合 {combination_name} 训练成功。")
            result_data['training_status'] = 'success'
            
            checkpoint_dir = f"checkpoints_{encoder}_{decoder}"
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

            if not os.path.exists(best_model_path):
                log(f"警告: 训练成功但未找到检查点文件: {best_model_path}")
                result_data['evaluation_status'] = 'checkpoint_not_found'
                append_to_csv(result_data)
                log("="*80)
                continue

            # --- 2. 评估阶段 ---
            log(f"开始评估模型: {best_model_path}")
            evaluate_command = [
                sys.executable, EVALUATE_SCRIPT,
                '--checkpoint', best_model_path,
                '--split', 'test'
            ]
            eval_result = run_command(evaluate_command)

            if eval_result and eval_result.returncode == 0:
                log(f"组合 {combination_name} 评估成功。")
                result_data['evaluation_status'] = 'success'
                metrics = parse_evaluation_output(eval_result.stdout)
                result_data.update(metrics)
            else:
                log(f"组合 {combination_name} 评估失败。")
                if eval_result:
                    log(f"评估错误信息:\n{eval_result.stderr[-500:]}")
                result_data['evaluation_status'] = 'failed'
        
        else:
            log(f"组合 {combination_name} 训练失败。")
            if train_result:
                log(f"训练错误信息:\n{train_result.stderr[-500:]}") # 打印最后500个字符的错误信息
        
        # --- 3. 保存结果 ---
        append_to_csv(result_data)
        log("="*80)

    log("所有实验组合已处理完毕。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n用户中断了实验。")
        sys.exit(0)
