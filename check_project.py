"""
项目状态检查脚本
用于验证项目完整性和依赖安装情况
"""
import os
import sys

def check_files():
    """检查必要文件是否存在"""
    print("=" * 60)
    print("检查项目文件...")
    print("=" * 60)
    
    required_files = [
        'vocabulary.py',
        'dataset.py',
        'encoder.py',
        'decoder.py',
        'model.py',
        'train.py',
        'evaluate.py',
        'inference.py',
        'config.py',
        'requirements.txt',
        'README_USAGE.md',
        'QUICKSTART.md',
        'PROJECT_SUMMARY.md',
    ]
    
    all_exist = True
    for file in required_files:
        exists = os.path.exists(file)
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
        if not exists:
            all_exist = False
    
    print()
    return all_exist

def check_dataset():
    """检查数据集是否存在"""
    print("=" * 60)
    print("检查数据集...")
    print("=" * 60)
    
    dataset_items = [
        ('flickr8k_aim3', '目录'),
        ('flickr8k_aim3/dataset_flickr8k.json', '标注文件'),
        ('flickr8k_aim3/images', '图像目录'),
    ]
    
    all_exist = True
    for item, desc in dataset_items:
        exists = os.path.exists(item)
        status = "✓" if exists else "✗"
        print(f"  {status} {item:<45} ({desc})")
        if not exists:
            all_exist = False
    
    # 检查图像数量
    if os.path.exists('flickr8k_aim3/images'):
        image_count = len([f for f in os.listdir('flickr8k_aim3/images') 
                          if f.endswith('.jpg')])
        print(f"\n  图像数量: {image_count}")
        if image_count > 0:
            print(f"  预期: 8091张图像")
    
    print()
    return all_exist

def check_dependencies():
    """检查Python依赖是否安装"""
    print("=" * 60)
    print("检查Python依赖...")
    print("=" * 60)
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('nltk', 'NLTK'),
        ('tqdm', 'tqdm'),
    ]
    
    all_installed = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (未安装)")
            all_installed = False
    
    print()
    return all_installed

def check_vocab():
    """检查词汇表是否已构建"""
    print("=" * 60)
    print("检查词汇表...")
    print("=" * 60)
    
    vocab_path = 'flickr8k_aim3/vocabulary.pkl'
    exists = os.path.exists(vocab_path)
    
    if exists:
        print(f"  ✓ 词汇表已构建: {vocab_path}")
        try:
            import pickle
            with open(vocab_path, 'rb') as f:
                data = pickle.load(f)
            print(f"  词汇表大小: {len(data['word2idx'])}")
        except:
            print(f"  ⚠ 词汇表文件存在但可能损坏")
    else:
        print(f"  ✗ 词汇表未构建")
        print(f"  运行命令: python vocabulary.py")
    
    print()
    return exists

def check_checkpoints():
    """检查模型检查点"""
    print("=" * 60)
    print("检查模型检查点...")
    print("=" * 60)
    
    if os.path.exists('checkpoints'):
        checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
        if checkpoints:
            print(f"  找到 {len(checkpoints)} 个检查点:")
            for ckpt in checkpoints[:5]:  # 只显示前5个
                print(f"    - {ckpt}")
            if len(checkpoints) > 5:
                print(f"    ... 还有 {len(checkpoints)-5} 个")
        else:
            print(f"  ✗ 没有找到检查点文件")
            print(f"  运行命令: python train.py")
    else:
        print(f"  ✗ checkpoints目录不存在")
        print(f"  将在训练时自动创建")
    
    print()

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("图像描述生成项目 - 状态检查")
    print("=" * 60)
    print()
    
    # 检查文件
    files_ok = check_files()
    
    # 检查数据集
    dataset_ok = check_dataset()
    
    # 检查依赖
    deps_ok = check_dependencies()
    
    # 检查词汇表
    vocab_ok = check_vocab()
    
    # 检查检查点
    check_checkpoints()
    
    # 总结
    print("=" * 60)
    print("总结")
    print("=" * 60)
    
    if files_ok and dataset_ok and deps_ok:
        print("✓ 项目配置完整，可以开始训练！")
        print()
        print("下一步:")
        if not vocab_ok:
            print("  1. 构建词汇表: python vocabulary.py")
            print("  2. 开始训练: python train.py")
        else:
            print("  1. 开始训练: python train.py")
        print("  2. 评估模型: python evaluate.py --checkpoint checkpoints/best_model.pth")
        print("  3. 生成描述: python inference.py --checkpoint checkpoints/best_model.pth --image <图像路径>")
    else:
        print("✗ 项目配置不完整，请检查以下问题:")
        if not files_ok:
            print("  - 某些代码文件缺失")
        if not dataset_ok:
            print("  - 数据集缺失或不完整")
        if not deps_ok:
            print("  - Python依赖未安装，运行: pip install -r requirements.txt")
    
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()
