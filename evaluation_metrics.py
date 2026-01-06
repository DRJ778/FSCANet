#!/usr/bin/env python3
"""
图像融合评价指标完整代码
直接调用评价指标/文件夹中的实现
包含所有21个评价指标计算函数
适用于各种图像融合方法的评价实验
"""

import numpy as np
import cv2
import sys
import os
import warnings

# 过滤numpy的除零警告（这些警告是预期的，代码中已处理）
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')

# 添加评价指标文件夹到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
metrics_dir = os.path.join(current_dir, '评价指标')
if os.path.exists(metrics_dir):
    sys.path.insert(0, metrics_dir)
else:
    raise ImportError(f"评价指标文件夹不存在: {metrics_dir}")

# 导入评价指标文件夹中的模块
try:
    from Metric_torch import (
        EN_function, CE_function, QNCIE_function, TE_function, EI_function,
        Qy_function, Qcb_function, MI_function, SF_function, AG_function,
        SD_function, PSNR_function, MSE_function, VIF_function, CC_function,
        SCD_function, Qabf_function, Nabf_function, SSIM_function, MS_SSIM_function,
        NMI_function
    )
    from Qabf import get_Qabf
    from Nabf import get_Nabf
    from ssim import ssim, ms_ssim

    METRICS_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入评价指标模块: {e}")
    METRICS_AVAILABLE = False

# 尝试导入torch
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，部分指标可能无法计算")


def _to_gray(img):
    """将BGR图像转换为灰度图像"""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def validate_image(img, name="image"):
    """验证图像数据的有效性"""
    if img is None:
        print(f"错误: {name} 为空")
        return False
    
    img = np.array(img)
    if img.size == 0:
        print(f"错误: {name} 尺寸为0")
        return False
        
    if len(img.shape) not in [2, 3]:
        print(f"错误: {name} 维度不正确: {img.shape}")
        return False
        
    return True


def _numpy_to_tensor(img, device='cpu'):
    """将numpy数组转换为torch tensor"""
    if not TORCH_AVAILABLE:
        return None
    if isinstance(img, torch.Tensor):
        return img.to(device)
    img_array = np.array(img)
    if img_array.ndim == 2:
        return torch.from_numpy(img_array).float().to(device)
    elif img_array.ndim == 3:
        # 如果是BGR，转换为灰度
        if img_array.shape[2] == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            return torch.from_numpy(gray).float().to(device)
    return torch.from_numpy(img_array).float().to(device)


def _prepare_images_for_metrics(ir_img, vi_img, fused_img):
    """准备图像数据，转换为torch tensor或numpy数组"""
    # 转换为灰度
    ir_gray = _to_gray(ir_img)
    vi_gray = _to_gray(vi_img)
    fused_gray = _to_gray(fused_img)

    # 确保是numpy数组
    ir_gray = np.array(ir_gray)
    vi_gray = np.array(vi_gray)
    fused_gray = np.array(fused_gray)

    # 确保值在0-255范围内（如果是浮点数，假设是0-1范围，需要转换）
    if ir_gray.dtype == np.float32 or ir_gray.dtype == np.float64:
        if ir_gray.max() <= 1.0:
            ir_gray = (ir_gray * 255).astype(np.uint8)
        else:
            ir_gray = np.clip(ir_gray, 0, 255).astype(np.uint8)
    else:
        ir_gray = np.clip(ir_gray, 0, 255).astype(np.uint8)

    if vi_gray.dtype == np.float32 or vi_gray.dtype == np.float64:
        if vi_gray.max() <= 1.0:
            vi_gray = (vi_gray * 255).astype(np.uint8)
        else:
            vi_gray = np.clip(vi_gray, 0, 255).astype(np.uint8)
    else:
        vi_gray = np.clip(vi_gray, 0, 255).astype(np.uint8)

    if fused_gray.dtype == np.float32 or fused_gray.dtype == np.float64:
        if fused_gray.max() <= 1.0:
            fused_gray = (fused_gray * 255).astype(np.uint8)
        else:
            fused_gray = np.clip(fused_gray, 0, 255).astype(np.uint8)
    else:
        fused_gray = np.clip(fused_gray, 0, 255).astype(np.uint8)

    # 转换为torch tensor（如果需要）
    if TORCH_AVAILABLE:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ir_tensor = _numpy_to_tensor(ir_gray, device)
        vi_tensor = _numpy_to_tensor(vi_gray, device)
        fused_tensor = _numpy_to_tensor(fused_gray, device)

        # 同时准备int类型（用于MI, NMI等需要直方图的函数）
        ir_int = ir_gray.astype(np.int32)
        vi_int = vi_gray.astype(np.int32)
        fused_int = fused_gray.astype(np.int32)

        # 准备float32类型（用于Qabf, SSIM等，虽然变量名叫double但实际是float32）
        ir_double = ir_gray.astype(np.float32)
        vi_double = vi_gray.astype(np.float32)
        fused_double = fused_gray.astype(np.float32)

        return {
            'tensor': (ir_tensor, vi_tensor, fused_tensor),
            'int': (ir_int, vi_int, fused_int),
            'double': (ir_double, vi_double, fused_double),
            'numpy': (ir_gray, vi_gray, fused_gray)
        }
    else:
        return {
            'tensor': None,
            'int': (ir_gray.astype(np.int32), vi_gray.astype(np.int32), fused_gray.astype(np.int32)),
            'double': (ir_gray.astype(np.float32), vi_gray.astype(np.float32), fused_gray.astype(np.float32)),
            'numpy': (ir_gray, vi_gray, fused_gray)
        }


def compute_all_metrics(fused_image, visible_image, infrared_image):
    """
    计算所有图像融合质量评价指标
    直接调用评价指标/文件夹中的函数
    
    Args:
        fused_image: 融合后的图像
        visible_image: 可见光图像
        infrared_image: 红外图像
    
    Returns:
        dict: 包含所有21个评价指标的字典
    """
    if not METRICS_AVAILABLE:
        print("错误: 评价指标模块未正确导入")
        return {}

    metrics = {}
    
    # 验证输入图像
    if not validate_image(fused_image, "融合图像"):
        return {}
    if not validate_image(visible_image, "可见光图像"):
        return {}
    if not validate_image(infrared_image, "红外图像"):
        return {}

    try:
        # 准备图像数据
        img_data = _prepare_images_for_metrics(infrared_image, visible_image, fused_image)

        if not TORCH_AVAILABLE:
            print("错误: 需要PyTorch来计算评价指标")
            return {}

        ir_tensor, vi_tensor, f_tensor = img_data['tensor']
        ir_int, vi_int, f_int = img_data['int']
        ir_double, vi_double, f_double = img_data['double']

        if ir_tensor is None or vi_tensor is None or f_tensor is None:
            print("错误: 无法将图像转换为tensor")
            return {}

        # 计算所有21个指标（按照eval_torch.py中的顺序）
        try:
            metrics['CE'] = CE_function(ir_tensor, vi_tensor, f_tensor).item()
        except Exception as e:
            print(f"CE计算失败: {e}")
            metrics['CE'] = np.nan

        try:
            metrics['NMI'] = NMI_function(ir_int, vi_int, f_int, gray_level=256)
        except Exception as e:
            print(f"NMI计算失败: {e}")
            metrics['NMI'] = np.nan

        try:
            metrics['QNCIE'] = QNCIE_function(ir_tensor, vi_tensor, f_tensor)
        except Exception as e:
            print(f"QNCIE计算失败: {e}")
            metrics['QNCIE'] = np.nan

        try:
            metrics['TE'] = TE_function(ir_tensor, vi_tensor, f_tensor)
        except Exception as e:
            print(f"TE计算失败: {e}")
            metrics['TE'] = np.nan

        try:
            metrics['EI'] = EI_function(f_tensor)
        except Exception as e:
            print(f"EI计算失败: {e}")
            metrics['EI'] = np.nan

        try:
            metrics['Qy'] = Qy_function(ir_tensor, vi_tensor, f_tensor)
        except Exception as e:
            print(f"Qy计算失败: {e}")
            metrics['Qy'] = np.nan

        try:
            metrics['Qcb'] = Qcb_function(ir_tensor, vi_tensor, f_tensor)
        except Exception as e:
            print(f"Qcb计算失败: {e}")
            metrics['Qcb'] = np.nan

        try:
            metrics['EN'] = EN_function(f_tensor).item()
        except Exception as e:
            print(f"EN计算失败: {e}")
            metrics['EN'] = np.nan

        try:
            metrics['MI'] = MI_function(ir_int, vi_int, f_int, gray_level=256)
        except Exception as e:
            print(f"MI计算失败: {e}")
            metrics['MI'] = np.nan

        try:
            metrics['SF'] = SF_function(f_tensor).item()
        except Exception as e:
            print(f"SF计算失败: {e}")
            metrics['SF'] = np.nan

        try:
            metrics['AG'] = AG_function(f_tensor).item()
        except Exception as e:
            print(f"AG计算失败: {e}")
            metrics['AG'] = np.nan

        try:
            metrics['SD'] = SD_function(f_tensor).item()
        except Exception as e:
            print(f"SD计算失败: {e}")
            metrics['SD'] = np.nan

        try:
            metrics['CC'] = CC_function(ir_tensor, vi_tensor, f_tensor).item()
        except Exception as e:
            print(f"CC计算失败: {e}")
            metrics['CC'] = np.nan

        try:
            metrics['SCD'] = SCD_function(ir_tensor, vi_tensor, f_tensor).item()
        except Exception as e:
            print(f"SCD计算失败: {e}")
            metrics['SCD'] = np.nan

        try:
            metrics['VIF'] = VIF_function(ir_tensor, vi_tensor, f_tensor).item()
        except Exception as e:
            print(f"VIF计算失败: {e}")
            metrics['VIF'] = np.nan

        try:
            metrics['MSE'] = MSE_function(ir_tensor, vi_tensor, f_tensor).item()
        except Exception as e:
            print(f"MSE计算失败: {e}")
            metrics['MSE'] = np.nan

        try:
            metrics['PSNR'] = PSNR_function(ir_tensor, vi_tensor, f_tensor).item()
        except Exception as e:
            print(f"PSNR计算失败: {e}")
            metrics['PSNR'] = np.nan

        try:
            # Qabf返回numpy float，不需要.item()
            qabf_result = Qabf_function(ir_double, vi_double, f_double)
            metrics['Qabf'] = float(qabf_result) if not isinstance(qabf_result, (torch.Tensor, np.ndarray)) else float(
                qabf_result.item() if hasattr(qabf_result, 'item') else float(qabf_result))
        except Exception as e:
            print(f"Qabf计算失败: {e}")
            metrics['Qabf'] = np.nan

        try:
            # Nabf返回numpy float，但需要检查类型
            nabf_result = Nabf_function(ir_tensor, vi_tensor, f_tensor)
            metrics['Nabf'] = float(nabf_result) if not isinstance(nabf_result, (torch.Tensor, np.ndarray)) else float(
                nabf_result.item() if hasattr(nabf_result, 'item') else float(nabf_result))
        except Exception as e:
            print(f"Nabf计算失败: {e}")
            metrics['Nabf'] = np.nan

        try:
            metrics['SSIM'] = SSIM_function(ir_double, vi_double, f_double)
        except Exception as e:
            print(f"SSIM计算失败: {e}")
            metrics['SSIM'] = np.nan

        try:
            metrics['MS_SSIM'] = MS_SSIM_function(ir_double, vi_double, f_double)
        except Exception as e:
            print(f"MS_SSIM计算失败: {e}")
            metrics['MS_SSIM'] = np.nan

        # 验证指标值的合理性并统一转换为float
        for key, value in metrics.items():
            # 确保所有值都是Python float类型
            if isinstance(value, torch.Tensor):
                metrics[key] = float(value.item())
            elif isinstance(value, np.ndarray):
                metrics[key] = float(value.item() if value.size == 1 else float(value))
            elif not isinstance(value, (int, float)):
                try:
                    metrics[key] = float(value)
                except (ValueError, TypeError):
                    metrics[key] = np.nan  # 修复缩进

            # 检查异常值
            if np.isnan(metrics[key]) or np.isinf(metrics[key]):
                print(f"警告: {key} 计算结果异常: {metrics[key]}")
                
    except Exception as e:
        print(f"计算指标时出错: {e}")
        import traceback
        traceback.print_exc()
        # 返回空字典或NaN值
        metrics = {}
    
    return metrics


def compute_metrics_batch(fused_dir, vi_dir, ir_dir, output_file=None):
    """
    批量计算图像融合质量评价指标
    
    Args:
        fused_dir: 融合图像目录
        vi_dir: 可见光图像目录
        ir_dir: 红外图像目录
        output_file: 输出文件路径（可选）
    
    Returns:
        dict: 包含所有图像平均指标的字典
    """
    import glob
    
    # 获取所有图像文件
    fused_files = glob.glob(os.path.join(fused_dir, "*.png")) + glob.glob(os.path.join(fused_dir, "*.jpg"))
    vi_files = glob.glob(os.path.join(vi_dir, "*.png")) + glob.glob(os.path.join(vi_dir, "*.jpg"))
    ir_files = glob.glob(os.path.join(ir_dir, "*.png")) + glob.glob(os.path.join(ir_dir, "*.jpg"))
    
    # 按文件名排序
    fused_files.sort()
    vi_files.sort()
    ir_files.sort()
    
    all_metrics = []
    
    for i, (fused_path, vi_path, ir_path) in enumerate(zip(fused_files, vi_files, ir_files)):
        try:
            # 读取图像
            fused_img = cv2.imread(fused_path)
            vi_img = cv2.imread(vi_path)
            ir_img = cv2.imread(ir_path)
            
            if fused_img is None or vi_img is None or ir_img is None:
                print(f"跳过文件 {i + 1}: 读取失败")
                continue
            
            # 计算指标
            metrics = compute_all_metrics(fused_img, vi_img, ir_img)
            if metrics:
                all_metrics.append(metrics)  # 修复缩进
                print(f"处理完成 {i + 1}/{len(fused_files)}: {os.path.basename(fused_path)}")
            else:
                print(f"跳过文件 {i + 1}: 指标计算失败")
            
        except Exception as e:
            print(f"处理文件 {i + 1} 时出错: {e}")
            continue
    
    if not all_metrics:
        print("没有成功处理的图像")
        return {}
    
    # 计算平均指标
    avg_metrics = {}
    all_keys = set()
    for m in all_metrics:
        all_keys.update(m.keys())

    for key in all_keys:
        values = [m[key] for m in all_metrics if key in m and not np.isnan(m[key])]
        avg_metrics[key] = np.mean(values) if values else np.nan
    
    # 输出结果
    print("\n================ 指标结果 ================")
    print(f"处理图像数: {len(all_metrics)}")
    # 按照eval_torch.py中的顺序输出
    metric_order = ['CE', 'NMI', 'QNCIE', 'TE', 'EI', 'Qy', 'Qcb', 'EN', 'MI',
                    'SF', 'AG', 'SD', 'CC', 'SCD', 'VIF', 'MSE', 'PSNR',
                    'Qabf', 'Nabf', 'SSIM', 'MS_SSIM']
    for key in metric_order:
        if key in avg_metrics:
            value = avg_metrics[key]
            if not np.isnan(value):  # 修复缩进
                print(f"{key:10}: {value:.4f}")
            else:
                print(f"{key:10}: N/A")

    # 输出其他可能存在的指标
    for key in sorted(avg_metrics.keys()):
        if key not in metric_order:
            value = avg_metrics[key]
        if not np.isnan(value):
                print(f"{key:10}: {value:.4f}")
        else:
                print(f"{key:10}: N/A")
    
    # 保存到文件
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("================ 指标结果 ================\n")
                f.write(f"处理图像数: {len(all_metrics)}\n\n")
                for key in metric_order:
                    if key in avg_metrics:
                        value = avg_metrics[key]
                    if not np.isnan(value):
                        f.write(f"{key:10}: {value:.4f}\n")
                    else:
                        f.write(f"{key:10}: N/A\n")
                # 其他指标
                for key in sorted(avg_metrics.keys()):
                    if key not in metric_order:
                        value = avg_metrics[key]
                        if not np.isnan(value):  # 修复缩进
                            f.write(f"{key:10}: {value:.4f}\n")
                    else:
                            f.write(f"{key:10}: N/A\n")
            print(f"\n指标已保存到: {output_file}")
        except Exception as e:
            print(f"保存指标失败: {e}")
    
    return avg_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='计算图像融合质量评价指标')
    parser.add_argument('--fused', type=str, required=True, help='融合图像目录')
    parser.add_argument('--vi', type=str, required=True, help='可见光图像目录')
    parser.add_argument('--ir', type=str, required=True, help='红外图像目录')
    parser.add_argument('--output', type=str, help='输出文件路径')
    
    args = parser.parse_args()
    
    compute_metrics_batch(args.fused, args.vi, args.ir, args.output)