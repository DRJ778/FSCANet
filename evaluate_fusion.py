#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用融合评价脚本
- 输入：
  --fused  融合图目录（保存的真实融合结果）
  --ir     红外图像目录（与 fused 按文件名匹配）
  --vis    可见光图像目录（与 fused 按文件名匹配）
  --output 结果JSON保存路径
  计算九项指标：EN SD SF AG VIF SCD Qabf PSNR SSIM
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import cv2
from PIL import Image

try:
    from skimage.metrics import structural_similarity as sk_ssim
except Exception:
    sk_ssim = None

sewar_vifp = None
sewar_ssim = None
sewar_psnr = None
sewar_qabf = None
try:
    # primary import path
    from sewar.full_ref import vifp as sewar_vifp
    from sewar.full_ref import ssim as sewar_ssim
    from sewar.full_ref import psnr as sewar_psnr
    from sewar.full_ref import qabf as sewar_qabf
except Exception:
    try:
        # fallback import style
        from sewar import full_ref as sewar_fr
        sewar_vifp = getattr(sewar_fr, 'vifp', None)
        sewar_ssim = getattr(sewar_fr, 'ssim', None)
        sewar_psnr = getattr(sewar_fr, 'psnr', None)
        sewar_qabf = getattr(sewar_fr, 'qabf', None)
    except Exception:
        pass


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
        
    if img.dtype not in [np.uint8, np.float32, np.float64]:
        print(f"警告: {name} 数据类型为 {img.dtype}，建议使用uint8")
        
    return True


def load_image_as_gray(image_path: Path):
    try:
        img = Image.open(image_path)
        if img.mode != 'L':
            img = img.convert('L')
        return np.array(img, dtype=np.float64)
    except Exception as e:
        print(f"无法加载图像 {image_path}: {e}")
        return None


def calculate_entropy(img: np.ndarray) -> float:
    img = img.astype(np.uint8)
    hist, _ = np.histogram(img, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    hist = hist / max(1, hist.sum())
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    return float(-np.sum(hist * np.log2(hist)))


def calculate_spatial_frequency(img: np.ndarray) -> float:
    img = img.astype(np.float64)
    rf = np.sqrt(np.mean(np.diff(img, axis=1) ** 2))
    cf = np.sqrt(np.mean(np.diff(img, axis=0) ** 2))
    return float(np.sqrt(rf ** 2 + cf ** 2))


def calculate_average_gradient(img: np.ndarray) -> float:
    """计算平均梯度（AG）"""
    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    gy, gx = np.gradient(img)
    ag = np.mean(np.sqrt(gx ** 2 + gy ** 2))
    return ag


def calculate_mutual_information(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(np.uint8)
    img2 = img2.astype(np.uint8)
    hist_2d = np.histogram2d(img1.ravel(), img2.ravel(), bins=256, range=[[0, 256], [0, 256]])[0]
    if hist_2d.sum() == 0:
        return 0.0
    pxy = hist_2d / hist_2d.sum()
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    mask = (pxy > 0) & (px_py > 0)
    mi = np.sum(pxy[mask] * np.log2(pxy[mask] / px_py[mask]))
    return float(mi)


def calculate_vif_avg(fused: np.ndarray, src1: np.ndarray, src2: np.ndarray) -> float:
    """计算VIF (Visual Information Fidelity) 指标"""
    # 优先使用sewar库
    if sewar_vifp is not None:
        try:
            f = _to_gray(fused).astype(np.float64)
            s1 = _to_gray(src1).astype(np.float64)
            s2 = _to_gray(src2).astype(np.float64)
            return float((sewar_vifp(s1, f) + sewar_vifp(s2, f)) / 2.0)
        except Exception as e:
            print(f"VIF计算失败 (sewar): {e}")
    
    # 使用简化版本的VIF计算
    try:
        f = _to_gray(fused).astype(np.float64)
        s1 = _to_gray(src1).astype(np.float64)
        s2 = _to_gray(src2).astype(np.float64)
        
        def _simple_vif(img1, img2):
            """简化的VIF计算"""
            # 计算图像方差
            var1 = np.var(img1)
            var2 = np.var(img2)
            
            if var1 == 0:
                return 0.0
            
            # 简化的信息保真度计算
            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                return 1.0
            
            # 基于方差的VIF近似
            vif = min(1.0, var1 / (var1 + mse))
            return vif
        
        vif1 = _simple_vif(s1, f)
        vif2 = _simple_vif(s2, f)
        return float((vif1 + vif2) / 2.0)
        
    except Exception as e:
        print(f"VIF计算失败 (简化版): {e}")
        return None


def calculate_vif(reference: np.ndarray, distorted: np.ndarray) -> float:
    """保持向后兼容的VIF计算函数"""
    ref_var = float(np.var(reference))
    dist_var = float(np.var(distorted))
    if ref_var <= 1e-12:
        return 0.0
    corr = np.corrcoef(reference.ravel(), distorted.ravel())[0, 1]
    if np.isnan(corr):
        corr = 0.0
    vif = float(corr) * (dist_var / ref_var)
    return float(max(0.0, min(1.0, vif)))


def calculate_psnr_avg(fused: np.ndarray, src1: np.ndarray, src2: np.ndarray) -> float:
    """计算峰值信噪比 (PSNR)"""
    # compute PSNR vs. both sources (grayscale) and average
    f = _to_gray(fused).astype(np.float32)
    s1 = _to_gray(src1).astype(np.float32)
    s2 = _to_gray(src2).astype(np.float32)
    if sewar_psnr is not None:
        # sewar.psnr expects integer images to infer MAX from dtype; use uint8
        f8 = np.clip(f, 0, 255).astype(np.uint8)
        s1_8 = np.clip(s1, 0, 255).astype(np.uint8)
        s2_8 = np.clip(s2, 0, 255).astype(np.uint8)
        p1 = float(sewar_psnr(s1_8, f8))
        p2 = float(sewar_psnr(s2_8, f8))
        return (p1 + p2) / 2.0
    # manual PSNR
    def _psnr(a, b):
        mse = np.mean((a - b) ** 2)
        if mse <= 1e-12:
            return 100.0
        PIXEL_MAX = 255.0
        return float(20 * np.log10(PIXEL_MAX) - 10 * np.log10(mse))
    return (_psnr(s1, f) + _psnr(s2, f)) / 2.0


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """保持向后兼容的PSNR计算函数"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return float('inf')
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def calculate_ssim_avg(fused: np.ndarray, src1: np.ndarray, src2: np.ndarray) -> float:
    """计算结构相似性 (SSIM)"""
    f = _to_gray(fused).astype(np.float32)
    s1 = _to_gray(src1).astype(np.float32)
    s2 = _to_gray(src2).astype(np.float32)
    if sewar_ssim is not None:
        # sewar.ssim expects integer dtype to infer MAX; use uint8
        f8 = np.clip(f, 0, 255).astype(np.uint8)
        s1_8 = np.clip(s1, 0, 255).astype(np.uint8)
        s2_8 = np.clip(s2, 0, 255).astype(np.uint8)
        # sewar ssim returns (mssim, s_map)
        m1 = float(sewar_ssim(s1_8, f8)[0])
        m2 = float(sewar_ssim(s2_8, f8)[0])
        return (m1 + m2) / 2.0
    if sk_ssim is not None:
        m1 = float(sk_ssim(s1, f, data_range=255))
        m2 = float(sk_ssim(s2, f, data_range=255))
        return (m1 + m2) / 2.0
    # fallback crude: normalized cross-correlation
    def _ncc(a, b):
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)
        return float(np.mean(a * b))
    return (_ncc(s1, f) + _ncc(s2, f)) / 2.0


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """保持向后兼容的SSIM计算函数"""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1 = float(np.mean(img1))
    mu2 = float(np.mean(img2))
    sigma1_sq = float(np.var(img1))
    sigma2_sq = float(np.var(img2))
    sigma12 = float(np.mean((img1 - mu1) * (img2 - mu2)))
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _grad_mag(img: np.ndarray) -> np.ndarray:
    grad_x = np.diff(img, axis=1)
    grad_y = np.diff(img, axis=0)
    min_h = min(grad_x.shape[0], grad_y.shape[0]) if grad_x.size and grad_y.size else 0
    min_w = min(grad_x.shape[1], grad_y.shape[1]) if grad_x.size and grad_y.size else 0
    if min_h == 0 or min_w == 0:
        return np.zeros((0, 0), dtype=np.float64)
    grad_x = grad_x[:min_h, :min_w]
    grad_y = grad_y[:min_h, :min_w]
    return np.sqrt(grad_x ** 2 + grad_y ** 2)


def calculate_scd_score(fused: np.ndarray, src1: np.ndarray, src2: np.ndarray) -> float:
    """计算结构相关性 (SCD)"""
    # Structural Correlation based on gradient magnitudes
    # SCD = corr(|∇F|, |∇S1|) + corr(|∇F|, |∇S2|)
    # where corr(a,b) = sum(a*b) / sqrt(sum(a^2) * sum(b^2)) ∈ [0,1]
    def _grad_mag(x):
        x = x.astype(np.float32)
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        return np.hypot(gx, gy)
    f = _to_gray(fused)
    s1 = _to_gray(src1)
    s2 = _to_gray(src2)
    gf = _grad_mag(f)
    g1 = _grad_mag(s1)
    g2 = _grad_mag(s2)
    def _corr_pos(a, b):
        num = float((a * b).sum())
        den = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()) + 1e-12)
        return num / den
    return _corr_pos(gf, g1) + _corr_pos(gf, g2)


def calculate_scd(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    """保持向后兼容的SCD计算函数"""
    g1 = _grad_mag(img1)
    g2 = _grad_mag(img2)
    gf = _grad_mag(fused)
    if g1.size == 0 or g2.size == 0 or gf.size == 0:
        return 0.0
    scd = np.mean(np.abs(gf - (g1 + g2) / 2))
    return float(scd)


def calculate_qabf_score(fused: np.ndarray, src1: np.ndarray, src2: np.ndarray) -> float:
    """计算基于边缘的质量评估 (Qabf)"""
    # Qabf via sewar if available, else None
    def _qabf_local(f, s1, s2):
        # Xydeas-Petrovic 2000 implementation
        # gradients
        ksize = 3
        def _grad_and_angle(x):
            gx = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=ksize)
            gy = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=ksize)
            gm = np.hypot(gx, gy)
            ang = np.arctan2(gy, gx)
            return gm, ang
        gf, af = _grad_and_angle(f)
        g1, a1 = _grad_and_angle(s1)
        g2, a2 = _grad_and_angle(s2)
        T1 = 1e-6
        T2 = 1e-6
        alpha = 1.0
        beta = 1.0
        # similarity maps for source1 and source2
        qg1 = (2 * g1 * gf + T1) / (g1 * g1 + gf * gf + T1)
        qa1 = (1 + np.cos(a1 - af) + T2) / (2 + T2)
        q1 = np.power(qg1, alpha) * np.power(qa1, beta)
        w1 = np.maximum(g1, gf)
        qg2 = (2 * g2 * gf + T1) / (g2 * g2 + gf * gf + T1)
        qa2 = (1 + np.cos(a2 - af) + T2) / (2 + T2)
        q2 = np.power(qg2, alpha) * np.power(qa2, beta)
        w2 = np.maximum(g2, gf)
        num = (w1 * q1).sum() + (w2 * q2).sum()
        den = w1.sum() + w2.sum() + 1e-12
        return float(num / den)

    # try sewar first
    if sewar_qabf is not None:
        f64 = _to_gray(fused).astype(np.float64)
        s1_64 = _to_gray(src1).astype(np.float64)
        s2_64 = _to_gray(src2).astype(np.float64)
        try:
            return float(sewar_qabf(s1_64, s2_64, f64))
        except Exception:
            try:
                f8 = np.clip(f64, 0, 255).astype(np.uint8)
                s1_8 = np.clip(s1_64, 0, 255).astype(np.uint8)
                s2_8 = np.clip(s2_64, 0, 255).astype(np.uint8)
                return float(sewar_qabf(s1_8, s2_8, f8))
            except Exception:
                pass

    # fallback to local implementation
    f = _to_gray(fused).astype(np.float64)
    s1 = _to_gray(src1).astype(np.float64)
    s2 = _to_gray(src2).astype(np.float64)
    return _qabf_local(f, s1, s2)


def calculate_qabf(img1: np.ndarray, img2: np.ndarray, fused: np.ndarray) -> float:
    """保持向后兼容的Qabf计算函数"""
    g1 = _grad_mag(img1)
    g2 = _grad_mag(img2)
    gf = _grad_mag(fused)
    if g1.size == 0 or g2.size == 0 or gf.size == 0:
        return 0.0
    w1 = g1 / (g1 + g2 + 1e-10)
    w2 = g2 / (g1 + g2 + 1e-10)
    g_ideal = w1 * g1 + w2 * g2
    corr = np.corrcoef(gf.ravel(), g_ideal.ravel())[0, 1]
    if np.isnan(corr):
        corr = 0.0
    return float(max(0.0, corr))


def _collect_images(dir_path: Path) -> List[Path]:
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in exts])


def _build_index(files: List[Path]) -> Dict[str, Path]:
    # 优先 .png > .jpg/.jpeg > 其它
    priority = {'.png': 3, '.jpg': 2, '.jpeg': 2, '.bmp': 1, '.tiff': 1}
    idx: Dict[str, tuple] = {}
    for p in files:
        stem = p.stem
        score = priority.get(p.suffix.lower(), 0)
        if stem not in idx or score > idx[stem][0]:
            idx[stem] = (score, p)
    return {k: v[1] for k, v in idx.items()}


def _match_sources(fused_files: List[Path], ir_index: Dict[str, Path], vis_index: Dict[str, Path]):
    matches = []
    for f in fused_files:
        stem = f.stem
        candidates = [stem, f"FLIR_{stem}", stem.zfill(5), stem.zfill(6)]
        ir_p = vis_p = None
        for c in candidates:
            if ir_p is None and c in ir_index:
                ir_p = ir_index[c]
            if vis_p is None and c in vis_index:
                vis_p = vis_index[c]
            if ir_p is not None and vis_p is not None:
                break
        if ir_p is not None and vis_p is not None:
            matches.append({'fused': f, 'ir': ir_p, 'vis': vis_p})
    return matches


def evaluate_pair(fused_p: Path, ir_p: Path, vis_p: Path) -> Dict[str, float]:
    fused = load_image_as_gray(fused_p)
    ir = load_image_as_gray(ir_p)
    vis = load_image_as_gray(vis_p)
    if fused is None or ir is None or vis is None:
        return {}
    h = min(fused.shape[0], ir.shape[0], vis.shape[0])
    w = min(fused.shape[1], ir.shape[1], vis.shape[1])
    fused = fused[:h, :w]
    ir = ir[:h, :w]
    vis = vis[:h, :w]

    # 转换为BGR格式用于新的评价函数
    fused_bgr = np.stack([fused, fused, fused], axis=2).astype(np.uint8)
    ir_bgr = np.stack([ir, ir, ir], axis=2).astype(np.uint8)
    vis_bgr = np.stack([vis, vis, vis], axis=2).astype(np.uint8)

    res = {
        'filename': fused_p.name,
        'EN': calculate_entropy(fused),
        'SD': float(np.std(fused)),
        'SF': calculate_spatial_frequency(fused),
        'AG': calculate_average_gradient(fused_bgr),
    }

    # 使用新的评价指标函数
    try:
        # VIF和Qabf可能为None，需要特殊处理
        vif = calculate_vif_avg(fused_bgr, vis_bgr, ir_bgr)
        res['VIF'] = vif if vif is not None else np.nan
        
        qabf = calculate_qabf_score(fused_bgr, vis_bgr, ir_bgr)
        res['Qabf'] = qabf if qabf is not None else np.nan
        
        res['SCD'] = calculate_scd_score(fused_bgr, vis_bgr, ir_bgr)
        res['PSNR'] = calculate_psnr_avg(fused_bgr, vis_bgr, ir_bgr)
        res['SSIM'] = calculate_ssim_avg(fused_bgr, vis_bgr, ir_bgr)
        
        # 验证指标值的合理性
        for key, value in res.items():
            if key != 'filename' and (np.isnan(value) or np.isinf(value)):
                print(f"警告: {key} 计算结果异常: {value}")
                res[key] = np.nan
                
    except Exception as e:
        print(f"计算指标时出错: {e}")
        # 使用旧的计算方法作为备选
        mi_ir = calculate_mutual_information(fused, ir)
        mi_vis = calculate_mutual_information(fused, vis)
        res['MI'] = (mi_ir + mi_vis) / 2
        vif_ir = calculate_vif(ir, fused)
        vif_vis = calculate_vif(vis, fused)
        res['VIF'] = (vif_ir + vif_vis) / 2
        res['SCD'] = calculate_scd(ir, vis, fused)
        res['Qabf'] = calculate_qabf(ir, vis, fused)
        psnr_ir = calculate_psnr(ir, fused)
        psnr_vis = calculate_psnr(vis, fused)
        res['PSNR'] = (psnr_ir + psnr_vis) / 2
        ssim_ir = calculate_ssim(ir, fused)
        ssim_vis = calculate_ssim(vis, fused)
        res['SSIM'] = (ssim_ir + ssim_vis) / 2
    
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fused', required=True, help='融合图目录')
    parser.add_argument('--ir', required=True, help='红外目录')
    parser.add_argument('--vis', required=True, help='可见光目录')
    parser.add_argument('--output', required=True, help='结果JSON路径')
    args = parser.parse_args()

    fused_dir = Path(args.fused)
    ir_dir = Path(args.ir)
    vis_dir = Path(args.vis)
    out_path = Path(args.output)

    if not fused_dir.exists():
        print(f"错误: 融合图目录不存在 {fused_dir}")
        return
    if not ir_dir.exists() or not vis_dir.exists():
        print("错误: 源图像目录不存在")
        print(f"ir: {ir_dir}")
        print(f"vis: {vis_dir}")
        return

    fused_files = _collect_images(fused_dir)
    ir_files = _collect_images(ir_dir)
    vis_files = _collect_images(vis_dir)
    ir_index = _build_index(ir_files)
    vis_index = _build_index(vis_files)
    matches = _match_sources(fused_files, ir_index, vis_index)

    print(f"找到融合图 {len(fused_files)}，成功匹配 {len(matches)} 对")
    results = []
    for m in matches:
        res = evaluate_pair(m['fused'], m['ir'], m['vis'])
        if res:
            results.append(res)

    if not results:
        print("错误: 没有成功评价任何图像")
        return

    # 输出打印表格
    header = f"{'文件名':<20} {'EN':<8} {'SD':<8} {'SF':<8} {'AG':<8} {'VIF':<8} {'SCD':<8} {'Qabf':<8} {'PSNR':<8} {'SSIM':<8}"
    print(header)
    print('-' * len(header))
    sums: Dict[str, list] = {}
    for r in results:
        print(f"{r['filename']:<20} {r['EN']:<8.4f} {r['SD']:<8.4f} {r['SF']:<8.4f} {r['AG']:<8.4f} {r['VIF']:<8.4f} {r['SCD']:<8.4f} {r['Qabf']:<8.4f} {r['PSNR']:<8.4f} {r['SSIM']:<8.4f}")
        for k, v in r.items():
            if k == 'filename' or not isinstance(v, (int, float)):
                continue
            sums.setdefault(k, []).append(v)

    print('-' * len(header))
    # 计算平均值，过滤掉NaN值
    avg = {}
    for k, v in sums.items():
        # 过滤掉NaN值
        valid_values = [x for x in v if not (np.isnan(x) or np.isinf(x))]
        if valid_values:
            avg[k] = float(np.mean(valid_values))
        else:
            avg[k] = np.nan
    
    # 格式化输出，处理NaN值
    def format_value(value):
        if np.isnan(value) or np.isinf(value):
            return "NaN"
        else:
            return f"{value:<8.4f}"
    
    print(f"{'平均FISCNet':<20} {format_value(avg.get('EN', np.nan))} {format_value(avg.get('SD', np.nan))} {format_value(avg.get('SF', np.nan))} {format_value(avg.get('AG', np.nan))} {format_value(avg.get('VIF', np.nan))} {format_value(avg.get('SCD', np.nan))} {format_value(avg.get('Qabf', np.nan))} {format_value(avg.get('PSNR', np.nan))} {format_value(avg.get('SSIM', np.nan))}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到: {out_path}")


if __name__ == '__main__':
    main()



