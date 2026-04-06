"""
独立脚本：预处理输入CSV数据 -> 以最大值为中心进行裁剪 -> 绘图

功能：
  1. 读取CSV格式的光强分布数据
  2. 预处理（去除负值、阈值处理、可选高斯滤波）
  3. 以最大值位置为中心进行裁剪
  4. 绘制原始数据、预处理后数据和裁剪后数据的对比图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os


# ======================== 配置参数 ========================
# 输入文件路径（修改为实际路径）
INPUT_FILE = r'F:\实验\实验集\20250226\moniguang_0um__0009.ascii.csv'

# 裁剪尺寸（以最大值为中心的窗口大小，单位：像素）
CROP_WIDTH = 201   # x方向裁剪宽度
CROP_HEIGHT = 201  # y方向裁剪高度

# 预处理参数
THRESHOLD = 0.00          # 阈值（相对于最大值的比例），低于此比例的值置零
USE_LOWPASS_FILTER = True  # 是否启用高斯低通滤波
FILTER_SIGMA = 2.0        # 高斯滤波标准差

# 输出文件名前缀（None 表示不保存）
# 例如设为 r'F:\实验\...\moniguang_-250um_crop_0006.ascii'
# 则裁剪数据保存为 '...crop_0006.ascii.csv'，图片保存为 '...crop_0006.ascii.png'
OUTPUT_PREFIX = r'F:\实验\实验集\20250226\moniguang_0um_crop_0009.ascii'

# 绘图参数
COLORMAP = 'jet'
SHOW_PLOT = True
SAVE_PLOT = True
# ==========================================================


def load_data(file_path):
    """加载CSV数据文件"""
    print(f"Loading data from: {file_path}")
    data = pd.read_csv(file_path, header=None).values
    print(f"  Data shape: {data.shape}")
    print(f"  Data range: [{data.min():.6g}, {data.max():.6g}]")
    return data


def preprocess(data, threshold=0.01, use_lowpass=False, sigma=2.0):
    """
    预处理数据：
      1. 去除负值
      2. 阈值处理（低于 threshold * max 的值置零）
      3. 可选高斯低通滤波
    """
    F = data.copy()

    # 去除 NaN 和 Inf 值（CSV文件末尾可能包含NaN列）
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  After removing NaN/Inf: shape = {F.shape}")

    # 去除负值
    F[F < 0] = 0
    print(f"  After removing negatives: max = {F.max():.6g}")

    # 阈值处理
    max_val = F.max()
    F[F < threshold * max_val] = 0
    print(f"  After thresholding (threshold={threshold}): "
          f"non-zero pixels = {np.count_nonzero(F)}/{F.size}")

    # 高斯低通滤波
    if use_lowpass:
        F = gaussian_filter(F, sigma=sigma)
        print(f"  After Gaussian filter (sigma={sigma}): max = {F.max():.6g}")

    return F


def crop_center_max(data, crop_width, crop_height):
    """
    以最大值位置为中心裁剪数据

    :param data: 2D numpy数组
    :param crop_width: 裁剪宽度（x方向，列数）
    :param crop_height: 裁剪高度（y方向，行数）
    :return: 裁剪后的数据，以及最大值的坐标 (max_y, max_x)
    """
    # 找到最大值位置
    max_y, max_x = np.unravel_index(np.argmax(data), data.shape)
    print(f"  Maximum value location: (row={max_y}, col={max_x}), value={data[max_y, max_x]:.6g}")

    ny, nx = data.shape

    # 计算裁剪起止位置（以最大值为中心）
    startx = max_x - crop_width // 2
    starty = max_y - crop_height // 2

    # 边界修正
    if startx < 0:
        startx = 0
    if starty < 0:
        starty = 0

    endx = startx + crop_width
    endy = starty + crop_height

    if endx > nx:
        endx = nx
        startx = max(endx - crop_width, 0)
    if endy > ny:
        endy = ny
        starty = max(endy - crop_height, 0)

    cropped = data[starty:endy, startx:endx]
    print(f"  Crop region: rows[{starty}:{endy}], cols[{startx}:{endx}]")
    print(f"  Cropped shape: {cropped.shape}")

    return cropped, (max_y, max_x)


def plot_results(raw, preprocessed, cropped, max_pos, cmap='jet',
                 save_path=None, show=True):
    """
    绘制三幅对比图：原始数据、预处理后数据、裁剪后数据

    :param raw: 原始数据
    :param preprocessed: 预处理后数据
    :param cropped: 裁剪后数据
    :param max_pos: 最大值位置 (row, col)
    :param cmap: 颜色映射
    :param save_path: 图片保存路径（None则不保存）
    :param show: 是否显示图片
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- 原始数据 ---
    im0 = axes[0].imshow(raw, cmap=cmap, aspect='equal')
    axes[0].set_title('Raw Data', fontsize=14)
    axes[0].set_xlabel('X (pixels)')
    axes[0].set_ylabel('Y (pixels)')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # --- 预处理后数据 ---
    im1 = axes[1].imshow(preprocessed, cmap=cmap, aspect='equal')
    # 标记最大值位置
    axes[1].plot(max_pos[1], max_pos[0], 'w+', markersize=15, markeredgewidth=2,
                 label=f'Max @ ({max_pos[0]}, {max_pos[1]})')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].set_title('Preprocessed Data', fontsize=14)
    axes[1].set_xlabel('X (pixels)')
    axes[1].set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # --- 裁剪后数据 ---
    im2 = axes[2].imshow(cropped, cmap=cmap, aspect='equal')
    axes[2].set_title(f'Cropped ({cropped.shape[1]}×{cropped.shape[0]})', fontsize=14)
    axes[2].set_xlabel('X (pixels)')
    axes[2].set_ylabel('Y (pixels)')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    """主函数"""
    print("=" * 60)
    print("Preprocess, Crop (centered on max), and Plot")
    print("=" * 60)

    # 1. 加载数据
    print("\n[Step 1] Loading data...")
    raw_data = load_data(INPUT_FILE)

    # 2. 预处理
    print("\n[Step 2] Preprocessing...")
    preprocessed = preprocess(raw_data, threshold=THRESHOLD,
                              use_lowpass=USE_LOWPASS_FILTER,
                              sigma=FILTER_SIGMA)

    # 3. 以最大值为中心裁剪
    print("\n[Step 3] Cropping centered on maximum...")
    cropped, max_pos = crop_center_max(preprocessed, CROP_WIDTH, CROP_HEIGHT)

    # 4. 保存裁剪后的数据
    if OUTPUT_PREFIX is not None:
        # 确保输出目录存在
        output_dir = os.path.dirname(OUTPUT_PREFIX)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 保存裁剪后的数据（使用前缀 + .csv）
        cropped_path = OUTPUT_PREFIX + '.csv'
        np.savetxt(cropped_path, cropped, delimiter=",")
        print(f"\n  Cropped data saved to: {cropped_path}")

    # 5. 绘图
    print("\n[Step 4] Plotting...")
    save_path = None
    if SAVE_PLOT and OUTPUT_PREFIX is not None:
        save_path = OUTPUT_PREFIX + '.png'

    plot_results(raw_data, preprocessed, cropped, max_pos,
                 cmap=COLORMAP, save_path=save_path, show=SHOW_PLOT)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()


