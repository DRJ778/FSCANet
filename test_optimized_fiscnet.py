"""
测试优化后的FISCNet Enhanced Correct模型
加载训练好的权重文件，在测试集上进行推理和评估

评价指标：使用 evaluation_metrics.py 中的所有21个指标计算函数
包括：CE, NMI, QNCIE, TE, EI, Qy, Qcb, EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM
"""

import torch
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from basicsr.archs import build_network
from basicsr.utils.img_util import tensor2img
import yaml
# 使用 evaluation_metrics.py 中的评价指标函数
from evaluation_metrics import compute_all_metrics, compute_metrics_batch


def load_model(weight_path, device='cuda', arch_type='auto'):
    """加载训练好的模型
    
    Args:
        weight_path: 权重文件路径
        device: 设备 ('cuda' or 'cpu')
        arch_type: 架构类型 ('auto', 'FISCNet_DualPath', 'FISCNet_Enhanced_Correct_Optimized')
    """
    print(f"正在加载模型权重: {weight_path}")
    
    # 加载权重文件，检查架构类型
    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
    
    # 确定权重键名
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    else:
        state_dict = checkpoint
    
    # 自动检测架构类型（根据state_dict的键）
    if arch_type == 'auto':
        if 'spatial_branch_vis' in state_dict or 'dual_cafm' in state_dict or 'freq_branch' in state_dict:
            arch_type = 'FISCNet_DualPath'
            print("🔍 自动检测到架构类型: FISCNet_DualPath")
        elif 'ssm_processor' in state_dict and 'sc_with_cafm' in state_dict:
            arch_type = 'FISCNet_Enhanced_Correct_Optimized'
            print("🔍 自动检测到架构类型: FISCNet_Enhanced_Correct_Optimized")
        else:
            # 默认使用双路径架构（如果训练使用的是新架构）
            arch_type = 'FISCNet_DualPath'
            print("⚠️ 无法自动检测架构类型，默认使用: FISCNet_DualPath")
    
    # 根据架构类型创建模型
    if arch_type == 'FISCNet_DualPath':
        network_opt = {
            'type': 'FISCNet_DualPath',
            'vis_channels': 1,
            'inf_channels': 1,
            'n_feat': 16,
            'H': 64,
            'W': 64,
            'num_transformer_layers': 2,
            'num_heads': 4
        }
        print("📦 使用 FISCNet_DualPath 架构")
    elif arch_type == 'FISCNet_Enhanced_Correct_Optimized':
        network_opt = {
            'type': 'FISCNet_Enhanced_Correct_Optimized',
            'vis_channels': 1,
            'inf_channels': 1,
            'n_feat': 16,
            'H': 64,
            'W': 64
        }
        print("📦 使用 FISCNet_Enhanced_Correct_Optimized 架构")
    else:
        raise ValueError(f"不支持的架构类型: {arch_type}")
    
    model = build_network(network_opt)
    model.eval()
    model = model.to(device)
    
    # 加载权重
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ 使用严格模式加载权重成功")
    except RuntimeError as e:
        print("⚠️ 严格模式加载失败，尝试非严格模式...")
        # 处理尺寸不匹配的层
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        size_mismatch_keys = []
        
        for key, value in state_dict.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    size_mismatch_keys.append(f"{key}: checkpoint {value.shape} vs model {model_state_dict[key].shape}")
                    print(f"⚠️  跳过尺寸不匹配的层: {key}")
                    print(f"   checkpoint形状: {value.shape}, 模型形状: {model_state_dict[key].shape}")
            else:
                skipped_keys.append(key)
        
        # 加载过滤后的权重
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
        if skipped_keys:
            print(f"⚠️  权重文件中存在但模型不需要的键 ({len(skipped_keys)} 个):")
            for key in skipped_keys[:10]:  # 只显示前10个
                print(f"   - {key}")
            if len(skipped_keys) > 10:
                print(f"   ... 还有 {len(skipped_keys) - 10} 个键未显示")
        
        if size_mismatch_keys:
            print(f"⚠️  尺寸不匹配的层 ({len(size_mismatch_keys)} 个):")
            for msg in size_mismatch_keys[:5]:  # 只显示前5个
                print(f"   - {msg}")
            if len(size_mismatch_keys) > 5:
                print(f"   ... 还有 {len(size_mismatch_keys) - 5} 个层未显示")
        
        if missing_keys:
            print(f"⚠️  模型需要但权重文件中缺失的键 ({len(missing_keys)} 个):")
            for key in missing_keys[:10]:  # 只显示前10个
                print(f"   - {key}")
            if len(missing_keys) > 10:
                print(f"   ... 还有 {len(missing_keys) - 10} 个键未显示")
        
        print("✅ 使用非严格模式加载权重成功（部分层未加载）")
    
    print(f"✅ 模型加载成功！")
    return model


def read_image(img_path, grayscale=False):
    """读取图像"""
    if grayscale:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = img[..., None]  # [H, W, 1]
    else:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 归一化到 [0, 1]
    img = img.astype(np.float32) / 255.0
    return img


def img2tensor(img):
    """将numpy图像转为tensor
    Args:
        img: numpy array, shape [H, W, C] (RGB) or [H, W] (grayscale)
    Returns:
        tensor: [1, C, H, W]
    """
    if len(img.shape) == 3:
        img = img.transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]
    else:
        img = np.expand_dims(img, axis=0)  # [H, W] -> [1, H, W]
    
    img = torch.from_numpy(img.copy()).float()
    return img.unsqueeze(0)  # [1, C, H, W]


def tensor2img_np(tensor):
    """将tensor转为numpy图像（处理RGB输出）"""
    img = tensor.squeeze().cpu().numpy()
    
    # YCrCb2RGB输出范围应该在[0,1]，如果不在则裁剪
    img = np.clip(img, 0, 1)
    
    # 转为 [0, 255]
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    # 如果是CHW格式，转为HWC
    if len(img.shape) == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    
    return img


def RGB2YCrCb(input_im, device='cuda'):
    """RGB转YCbCr（与训练时一致）"""
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def YCrCb2RGB(input_im, device='cuda'):
    """YCbCr转RGB（与训练时一致）"""
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(device)
    temp = (im_flat + bias).mm(mat).to(device)
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def enhance_ir_in_postprocess(fused_y, ir_tensor, enhancement_strength=0.15):
    """
    后处理增强：在融合图像的Y通道中进一步增强红外信息
    Args:
        fused_y: 模型输出的融合Y通道 [B, 1, H, W]
        ir_tensor: 红外图像 [B, 1, H, W]
        enhancement_strength: 增强强度 (0.0-0.3)，控制红外信息的额外增强幅度
    Returns:
        enhanced_y: 增强后的Y通道
    """
    # 确保尺寸一致
    min_h = min(fused_y.shape[2], ir_tensor.shape[2])
    min_w = min(fused_y.shape[3], ir_tensor.shape[3])
    fused_y = fused_y[:, :, :min_h, :min_w]
    ir_aligned = ir_tensor[:, :, :min_h, :min_w]
    
    # 方法1：基于红外显著区域的自适应增强
    # 计算红外图像的显著度（高亮度区域）
    ir_mean = ir_aligned.mean(dim=[2, 3], keepdim=True)
    ir_saliency = (ir_aligned - ir_mean).abs()  # 显著性矩阵
    ir_saliency_norm = (ir_saliency - ir_saliency.min()) / (ir_saliency.max() - ir_saliency.min() + 1e-8)
    
    # 方法2：在红外高亮度区域增强融合图像的Y通道
    ir_bright_mask = (ir_aligned > ir_aligned.quantile(0.6)).float()  # 高于60%分位数的区域
    
    # 组合增强：在高显著区域和高亮度区域增强红外信息
    enhancement_mask = (ir_saliency_norm * 0.6 + ir_bright_mask * 0.4).clamp(0, 1)
    
    # 计算差异：红外图像相对于融合图像的额外信息
    ir_delta = (ir_aligned - fused_y).clamp(0, 1)  # 只保留红外更强的地方
    
    # 增强融合：在显著区域添加红外的额外信息
    enhanced_y = fused_y + enhancement_strength * enhancement_mask * ir_delta
    
    # 确保在有效范围内
    enhanced_y = torch.clamp(enhanced_y, 0, 1)
    
    return enhanced_y


def fuse_images_grayscale(model, vis_path, ir_path, device='cuda', postprocess_ir_strength=0.15):
    """
    融合可见光和红外图像，直接输出灰度图（不引入Y通道的Cb/Cr，只使用融合后的Y通道）
    Args:
        model: 训练好的模型
        vis_path: 可见光图像路径
        ir_path: 红外图像路径
        device: 计算设备
        postprocess_ir_strength: 后处理红外增强强度 (0.0=关闭, 0.1-0.3=推荐范围)
    Returns:
        fused_gray: 融合后的灰度图 [H, W] (numpy array, uint8)
        vis_img: 原始可见光图像 [H, W, 3] (归一化到[0,1])
        ir_img: 原始红外图像 [H, W, 1] (归一化到[0,1])
    """
    # 读取图像
    vis_img = read_image(vis_path, grayscale=False)  # RGB [H, W, 3]
    ir_img = read_image(ir_path, grayscale=True)  # Grayscale [H, W, 1]
    
    # 转为tensor
    vis_tensor = img2tensor(vis_img).to(device)  # RGB图像 [1, 3, H, W]
    ir_tensor = img2tensor(ir_img).to(device)  # 灰度图像 [1, 1, H, W]
    
    # 确保尺寸一致
    min_h = min(vis_tensor.shape[2], ir_tensor.shape[2])
    min_w = min(vis_tensor.shape[3], ir_tensor.shape[3])
    vis_tensor = vis_tensor[:, :, :min_h, :min_w]
    ir_tensor = ir_tensor[:, :, :min_h, :min_w]
    
    # RGB -> YCbCr -> 提取Y -> 模型处理 -> 直接输出Y通道（灰度图）
    # 1. RGB转YCbCr
    vi_ycrcb = RGB2YCrCb(vis_tensor, device)
    
    # 2. 模型推理（输入YCbCr的Y通道和IR）
    with torch.no_grad():
        outputs = model(vi_ycrcb, ir_tensor)
        # 处理三个返回值：融合图像、可见光重建、红外重建
        if isinstance(outputs, tuple) and len(outputs) == 3:
            output_y, vis_recon, ir_recon = outputs
        else:
            # 兼容旧版本：如果模型只返回一个值
            output_y = outputs
        output_y = torch.clamp(output_y, 0, 1)  # 与训练时一致
    
    # 后处理：进一步增强红外信息
    if postprocess_ir_strength > 0:
        output_y = enhance_ir_in_postprocess(output_y, ir_tensor, postprocess_ir_strength)
    
    # 3. 直接将融合后的Y通道转为灰度图（不合并Cb/Cr，不转回RGB）
    # 将tensor转为numpy
    fused_y_np = output_y.squeeze().cpu().numpy()  # [H, W]
    
    # 确保在[0, 1]范围内
    fused_y_np = np.clip(fused_y_np, 0, 1)
    
    # 转为 [0, 255] 的uint8格式
    fused_gray = (fused_y_np * 255).astype(np.uint8)
    
    return fused_gray, vis_img, ir_img


def fuse_images(model, vis_path, ir_path, device='cuda', postprocess_ir_strength=0.15):
    """
    融合可见光和红外图像（完全按照训练时的流程 + 后处理增强）
    Args:
        model: 训练好的模型
        vis_path: 可见光图像路径
        ir_path: 红外图像路径
        device: 计算设备
        postprocess_ir_strength: 后处理红外增强强度 (0.0=关闭, 0.1-0.3=推荐范围)
    """
    # 读取图像
    vis_img = read_image(vis_path, grayscale=False)  # RGB [H, W, 3]
    ir_img = read_image(ir_path, grayscale=True)  # Grayscale [H, W, 1]
    
    # 转为tensor
    vis_tensor = img2tensor(vis_img).to(device)  # RGB图像 [1, 3, H, W]
    ir_tensor = img2tensor(ir_img).to(device)  # 灰度图像 [1, 1, H, W]
    
    # 确保尺寸一致
    min_h = min(vis_tensor.shape[2], ir_tensor.shape[2])
    min_w = min(vis_tensor.shape[3], ir_tensor.shape[3])
    vis_tensor = vis_tensor[:, :, :min_h, :min_w]
    ir_tensor = ir_tensor[:, :, :min_h, :min_w]
    
    # 【关键修复】按照训练时的流程：RGB -> YCbCr -> 提取Y -> 模型处理 -> 合并Y和CrCb -> YCbCr2RGB
    # 1. RGB转YCbCr
    vi_ycrcb = RGB2YCrCb(vis_tensor, device)
    
    # 2. 模型推理（输入YCbCr的Y通道和IR）
    with torch.no_grad():
        outputs = model(vi_ycrcb, ir_tensor)
        # 【修改】处理三个返回值：融合图像、可见光重建、红外重建
        if isinstance(outputs, tuple) and len(outputs) == 3:
            output_y, vis_recon, ir_recon = outputs
        else:
            # 兼容旧版本：如果模型只返回一个值
            output_y = outputs
        output_y = torch.clamp(output_y, 0, 1)  # 与训练时一致
    
    # 【新增】后处理：进一步增强红外信息
    if postprocess_ir_strength > 0:
        output_y = enhance_ir_in_postprocess(output_y, ir_tensor, postprocess_ir_strength)
    
    # 3. 对齐尺寸（同训练时）
    oh, ow = output_y.shape[-2], output_y.shape[-1]
    ch, cw = vi_ycrcb.shape[-2], vi_ycrcb.shape[-3]
    if (oh != ch) or (ow != cw):
        top = max((ch - oh) // 2, 0)
        left = max((cw - ow) // 2, 0)
        c1 = vi_ycrcb[:, 1:2, top:top+oh, left:left+ow]
        c2 = vi_ycrcb[:, 2:, top:top+oh, left:left+ow]
    else:
        c1 = vi_ycrcb[:, 1:2, :, :]
        c2 = vi_ycrcb[:, 2:, :, :]
    
    # 4. 合并Y和CrCb
    output_ycrcb = torch.cat((output_y, c1, c2), dim=1)
    
    # 5. YCbCr转RGB
    output_rgb = YCrCb2RGB(output_ycrcb, device)
    
    # 转为numpy
    fused_img = tensor2img_np(output_rgb)
    
    return fused_img, vis_img, ir_img


def test_on_dataset(model, vis_dir, ir_dir, output_dir, device='cuda', compute_metrics=True, 
                   postprocess_ir_strength=0.15):
    """在数据集上测试并计算评价指标"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像列表
    vis_files = sorted(list(Path(vis_dir).glob('*.png')) + list(Path(vis_dir).glob('*.jpg')))
    ir_files = sorted(list(Path(ir_dir).glob('*.png')) + list(Path(ir_dir).glob('*.jpg')))
    
    if len(vis_files) != len(ir_files):
        print(f"⚠️  可见光图像({len(vis_files)})和红外图像({len(ir_files)})数量不匹配")
        min_len = min(len(vis_files), len(ir_files))
        vis_files = vis_files[:min_len]
        ir_files = ir_files[:min_len]
    
    print(f"找到 {len(vis_files)} 对图像")
    
    results = []
    all_metrics_list = []
    detailed_metrics = []  # 存储每张图像的详细指标
    
    # 定义所有指标名称
    metric_names = ['CE', 'NMI', 'QNCIE', 'TE', 'EI', 'Qy', 'Qcb', 'EN', 'MI', 
                   'SF', 'AG', 'SD', 'CC', 'SCD', 'VIF', 'MSE', 'PSNR', 
                   'Qabf', 'Nabf', 'SSIM', 'MS_SSIM']
    
    for idx, (vis_path, ir_path) in enumerate(tqdm(zip(vis_files, ir_files), total=len(vis_files))):
        try:
            # 融合图像（使用后处理增强）
            fused_img, vis_img_norm, ir_img_norm = fuse_images(model, vis_path, ir_path, device, 
                                                              postprocess_ir_strength=postprocess_ir_strength)
            
            # 保存结果（转为BGR保存）
            output_path = os.path.join(output_dir, f"{vis_path.stem}_fused.png")
            cv2.imwrite(output_path, cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR))
            
            # 计算评价指标（使用 evaluation_metrics.py 中的函数）
            if compute_metrics:
                # 读取原始图像用于评价（需要转换为uint8格式）
                vis_img_uint8 = (vis_img_norm * 255).astype(np.uint8)
                ir_img_uint8 = (ir_img_norm.squeeze() * 255).astype(np.uint8)
                if len(ir_img_uint8.shape) == 2:
                    ir_img_uint8 = np.stack([ir_img_uint8] * 3, axis=2)  # 转为3通道
                
                # 转为BGR格式（evaluation_metrics.py 使用 cv2.imread，默认是BGR格式）
                fused_img_bgr = cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR)
                vis_img_bgr = cv2.cvtColor(vis_img_uint8, cv2.COLOR_RGB2BGR)
                
                try:
                    # 使用 evaluation_metrics.py 中的 compute_all_metrics 函数
                    # 计算所有21个评价指标：CE, NMI, QNCIE, TE, EI, Qy, Qcb, EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM
                    metrics = compute_all_metrics(fused_img_bgr, vis_img_bgr, ir_img_uint8)
                    all_metrics_list.append(metrics)
                    
                    # 保存该图像的指标信息
                    image_metrics = {
                        'image_name': vis_path.stem,
                        'metrics': metrics.copy()
                    }
                    detailed_metrics.append(image_metrics)
                    
                except Exception as e:
                    print(f"⚠️  计算 {vis_path.stem} 的指标时出错: {e}")
                    import traceback
                    traceback.print_exc()
            
            results.append({
                'name': vis_path.stem,
                'vis_path': str(vis_path),
                'ir_path': str(ir_path),
                'fused_path': output_path
            })
            
        except Exception as e:
            print(f"❌ 处理 {vis_path.name} 时出错: {e}")
            continue
    
    print(f"\n✅ 测试完成！融合结果保存在: {output_dir}")
    print(f"   总共处理了 {len(results)} 对图像")
    
    # 计算并显示平均评价指标（基于 evaluation_metrics.py 的计算结果）
    if compute_metrics and all_metrics_list:
        print("\n" + "="*60)
        print("评价指标结果（平均值）- 使用 evaluation_metrics.py")
        print("="*60)
        print(f"计算了 {len(all_metrics_list)} 张图像的指标\n")
        
        avg_metrics = {}
        # evaluation_metrics.py 支持的所有21个指标（按照评价指标文件夹中的顺序）
        # 移除SSIM_a，因为它不在评价指标文件夹中
        metric_names = ['CE', 'NMI', 'QNCIE', 'TE', 'EI', 'Qy', 'Qcb', 'EN', 'MI', 
                       'SF', 'AG', 'SD', 'CC', 'SCD', 'VIF', 'MSE', 'PSNR', 
                       'Qabf', 'Nabf', 'SSIM', 'MS_SSIM']
        
        # 计算平均值
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics_list 
                     if metric_name in m and not np.isnan(m[metric_name])]
            if values:
                avg_metrics[metric_name] = np.mean(values)
                print(f"{metric_name:10}: {avg_metrics[metric_name]:.4f}")
            else:
                avg_metrics[metric_name] = np.nan
                print(f"{metric_name:10}: N/A")
        
        # 保存指标到文件（包含每张图像的详细指标和平均值）
        metrics_file = os.path.join(output_dir, "metrics_results.txt")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("评价指标结果 - 使用 evaluation_metrics.py\n")
            f.write("="*80 + "\n")
            f.write(f"处理图像数: {len(all_metrics_list)}\n\n")
            
            # 写入指标说明
            f.write("指标说明：\n")
            f.write("  CE: 交叉熵 (Cross Entropy)\n")
            f.write("  NMI: 归一化互信息 (Normalized Mutual Information)\n")
            f.write("  QNCIE: 归一化互相关熵 (Normalized Cross-Correlation Entropy)\n")
            f.write("  TE: 总熵 (Total Entropy)\n")
            f.write("  EI: 边缘强度 (Edge Intensity)\n")
            f.write("  Qy: 基于SSIM的融合质量指标\n")
            f.write("  Qcb: 基于对比度的融合质量指标\n")
            f.write("  EN: 信息熵 (Entropy)\n")
            f.write("  MI: 互信息 (Mutual Information)\n")
            f.write("  SF: 空间频率 (Spatial Frequency)\n")
            f.write("  AG: 平均梯度 (Average Gradient)\n")
            f.write("  SD: 标准差 (Standard Deviation)\n")
            f.write("  CC: 相关系数 (Correlation Coefficient)\n")
            f.write("  SCD: 差异相关和 (Sum of Correlation of Differences)\n")
            f.write("  VIF: 视觉信息保真度 (Visual Information Fidelity)\n")
            f.write("  MSE: 均方误差 (Mean Squared Error)\n")
            f.write("  PSNR: 峰值信噪比 (Peak Signal-to-Noise Ratio)\n")
            f.write("  Qabf: 边缘质量指标 (Edge-based Quality)\n")
            f.write("  Nabf: 基于边缘的融合质量指标 (Negative Artifact-based Fusion)\n")
            f.write("  SSIM: 结构相似性 (Structural Similarity Index)\n")
            f.write("  MS_SSIM: 多尺度结构相似性 (Multi-Scale SSIM)\n\n")
            
            # 写入每张图像的详细指标
            f.write("\n" + "="*80 + "\n")
            f.write("每张图像的详细指标\n")
            f.write("="*80 + "\n\n")
            
            for img_metrics in detailed_metrics:
                f.write(f"图像名称: {img_metrics['image_name']}\n")
                f.write("-" * 80 + "\n")
                for metric_name in metric_names:
                    if metric_name in img_metrics['metrics'] and not np.isnan(img_metrics['metrics'][metric_name]):
                        f.write(f"{metric_name:10}: {img_metrics['metrics'][metric_name]:.4f}\n")
                    else:
                        f.write(f"{metric_name:10}: N/A\n")
                f.write("\n")
            
            # 写入平均值
            f.write("\n" + "="*80 + "\n")
            f.write("平均值（所有图像）\n")
            f.write("="*80 + "\n")
            for metric_name in metric_names:
                if not np.isnan(avg_metrics[metric_name]):
                    f.write(f"{metric_name:10}: {avg_metrics[metric_name]:.4f}\n")
                else:
                    f.write(f"{metric_name:10}: N/A\n")
        
        print(f"\n✅ 指标已保存到: {metrics_file}")
        print("="*60)
        
        return results, avg_metrics
    else:
        return results, None


def test_single_pair(model, vis_path, ir_path, output_path=None, device='cuda', compute_metrics=True,
                    postprocess_ir_strength=0.15):
    """测试单对图像并计算评价指标"""
    print(f"可见光图像: {vis_path}")
    print(f"红外图像: {ir_path}")
    print(f"后处理红外增强强度: {postprocess_ir_strength}")
    
    fused_img, vis_img_norm, ir_img_norm = fuse_images(model, vis_path, ir_path, device, 
                                                       postprocess_ir_strength=postprocess_ir_strength)
    
    if output_path is None:
        output_path = "fused_result.png"
    
    cv2.imwrite(output_path, cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR))
    print(f"✅ 融合结果已保存: {output_path}")
    
    # 计算评价指标（使用 evaluation_metrics.py）
    if compute_metrics:
        try:
            # 转换为uint8格式
            vis_img_uint8 = (vis_img_norm * 255).astype(np.uint8)
            ir_img_uint8 = (ir_img_norm.squeeze() * 255).astype(np.uint8)
            if len(ir_img_uint8.shape) == 2:
                ir_img_uint8 = np.stack([ir_img_uint8] * 3, axis=2)  # 转为3通道
            
            # 转为BGR格式（evaluation_metrics.py 使用 cv2.imread，默认是BGR格式）
            fused_img_bgr = cv2.cvtColor(fused_img, cv2.COLOR_RGB2BGR)
            vis_img_bgr = cv2.cvtColor(vis_img_uint8, cv2.COLOR_RGB2BGR)
            
            # 使用 evaluation_metrics.py 中的 compute_all_metrics 函数
            metrics = compute_all_metrics(fused_img_bgr, vis_img_bgr, ir_img_uint8)
            
            print("\n" + "="*60)
            print("评价指标结果 - 使用 evaluation_metrics.py")
            print("="*60)
            metric_names = ['EN', 'SD', 'SF', 'AG', 'PSNR', 'SSIM', 'SSIM_a', 'CC', 'MI', 'SCD', 'VIF', 'Qabf']
            for metric_name in metric_names:
                if metric_name in metrics and not np.isnan(metrics[metric_name]):
                    print(f"{metric_name:8}: {metrics[metric_name]:.4f}")
                else:
                    print(f"{metric_name:8}: N/A")
            print("="*60)
        except Exception as e:
            print(f"⚠️  计算评价指标时出错: {e}")
            import traceback
            traceback.print_exc()
    
    return fused_img


def main():
    parser = argparse.ArgumentParser(description='测试FISCNet模型（支持双路径架构和原架构）')
    parser.add_argument('--weight', type=str, required=True, help='权重文件路径 (net_g_50000.pth)')
    parser.add_argument('--vis_dir', type=str, help='可见光图像目录')
    parser.add_argument('--ir_dir', type=str, help='红外图像目录')
    parser.add_argument('--vis_img', type=str, help='单张可见光图像路径')
    parser.add_argument('--ir_img', type=str, help='单张红外图像路径')
    parser.add_argument('--output_dir', type=str, default='results', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--arch', type=str, default='auto', 
                       choices=['auto', 'FISCNet_DualPath', 'FISCNet_Enhanced_Correct_Optimized'],
                       help='架构类型（auto表示自动检测）')
    parser.add_argument('--no_metrics', action='store_true', help='不计算评价指标（仅融合图像）')
    parser.add_argument('--ir_enhance', type=float, default=0.15, 
                       help='后处理红外增强强度 (0.0=关闭, 0.1-0.3=推荐范围, 默认0.15)')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    # 加载模型
    model = load_model(args.weight, args.device, arch_type=args.arch)
    
    # 是否计算指标
    compute_metrics = not args.no_metrics
    
    # 测试模式选择
    if args.vis_img and args.ir_img:
        # 单对图像测试
        test_single_pair(model, args.vis_img, args.ir_img, 
                        output_path=args.output_dir, device=args.device, 
                        compute_metrics=compute_metrics,
                        postprocess_ir_strength=args.ir_enhance)
    elif args.vis_dir and args.ir_dir:
        # 数据集测试
        results, metrics = test_on_dataset(model, args.vis_dir, args.ir_dir, 
                       args.output_dir, device=args.device, 
                       compute_metrics=compute_metrics,
                       postprocess_ir_strength=args.ir_enhance)
        if metrics:
            return results, metrics
    else:
        print("❌ 请提供 --vis_dir 和 --ir_dir (数据集测试) 或 --vis_img 和 --ir_img (单对图像测试)")
        parser.print_help()


if __name__ == '__main__':
    main()
