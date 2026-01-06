import time
import os
import cv2
import torch
from basicsr.archs import build_network

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1) 构建模型
network_opt = {
    'type': 'FISCNet_Enhanced_Mamba',
    'vis_channels': 1,
    'inf_channels': 1,
    'n_feat': 16,
    'H': 128,
    'W': 128,
}
model = build_network(network_opt).to(device).eval()

# 2) 准备 test_data 路径，选前 20 对图像
root = 'test_data'
vi_dir = os.path.join(root, 'vi')
ir_dir = os.path.join(root, 'ir')

vi_files = sorted(os.listdir(vi_dir))[:20]  # 只取前 20 张
pair_paths = []
for name in vi_files:
    vi_path = os.path.join(vi_dir, name)
    ir_path = os.path.join(ir_dir, name)
    if os.path.isfile(ir_path):  # 两边都存在才算一对
        pair_paths.append((vi_path, ir_path))

print(f'Found {len(pair_paths)} image pairs for timing.')

def load_img_as_tensor(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 读成单通道
    img = cv2.resize(img, (128, 128))             # 保持与模型配置 H=W=128 一致
    img = img.astype('float32') / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    return tensor

# 3) 预热
with torch.no_grad():
    for vi_path, ir_path in pair_paths:
        vis = load_img_as_tensor(vi_path)
        ir  = load_img_as_tensor(ir_path)
        _ = model(vis, ir)

if device == 'cuda':
    torch.cuda.synchronize()

# 4) 正式计时
total_time = 0.0
with torch.no_grad():
    for vi_path, ir_path in pair_paths:
        vis = load_img_as_tensor(vi_path)
        ir  = load_img_as_tensor(ir_path)

        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        _ = model(vis, ir)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.time()

        total_time += (end - start)

avg_time = total_time / len(pair_paths)
print(f'Average inference time over {len(pair_paths)} image pairs: {avg_time*1000:.3f} ms')