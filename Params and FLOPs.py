"""
计算FISCNet_DualPath模型的参数量和FLOPs
使用实际训练配置和常见测试尺寸
"""
import torch
from basicsr.archs import build_network
from thop import profile

# 按 options/train/FISCNet_DualPath.yml 的设置
network_opt = {
    'type': 'FISCNet_DualPath',  # 使用实际训练的架构
    'vis_channels': 1,  # YCbCr的Y通道
    'inf_channels': 1,
    'n_feat': 16,
    'H': 64,   # SSM内部参数（用于初始化）
    'W': 64,   # SSM内部参数（用于初始化）
    'num_transformer_layers': 2,  # Transformer层数（空域分支）
    'num_heads': 4,  # 注意力头数
}

# 构建模型
model = build_network(network_opt)
model.eval()

print("=" * 60)
print("FISCNet_DualPath 模型复杂度分析")
print("=" * 60)

# 计算参数量（只统计可训练参数）
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
params_total = sum(p.numel() for p in model.parameters())
print(f'\n参数量统计:')
print(f'  可训练参数: {params:,}  ({params / 1e6:.3f} M)')
print(f'  总参数: {params_total:,}  ({params_total / 1e6:.3f} M)')
print(f'  不可训练参数: {params_total - params:,}  ({(params_total - params) / 1e6:.3f} M)')

# 计算FLOPs - 使用多个常见测试尺寸
print(f'\nFLOPs统计（不同输入尺寸）:')
print('-' * 60)

# 测试尺寸（仅保留训练patch尺寸）
test_sizes = [
    (128, 128),   # 训练patch尺寸
]

for H, W in test_sizes:
    # 构造输入：Y通道（模型会从YCbCr中提取Y通道，这里直接使用Y通道模拟）
    vis_y = torch.randn(1, 1, H, W)  # Y通道 [B, 1, H, W]
    ir = torch.randn(1, 1, H, W)     # 红外图像 [B, 1, H, W]
    
    try:
        flops, params_thop = profile(model, inputs=(vis_y, ir), verbose=False)
        print(f'  {H:4d} x {W:4d}: {flops:>15,} FLOPs  ({flops / 1e9:>6.3f} G)')
    except Exception as e:
        print(f'  {H:4d} x {W:4d}: 计算失败 - {e}')

print('-' * 60)
print('\n注意:')
print('  - 参数量不依赖于输入尺寸，是固定的')
print('  - FLOPs依赖于输入尺寸，实际测试时请根据图像尺寸计算')
print('  - 模型会自动padding到偶数尺寸（小波分解要求）')
print('  - 实际输入格式：可见光为YCbCr格式（3通道），模型会提取Y通道')
print('=' * 60)