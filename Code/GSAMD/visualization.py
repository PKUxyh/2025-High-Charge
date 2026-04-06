import matplotlib.pyplot as plt
import numpy as np


def plot_preprocessed_data(F_list, output_dir=None, save_plot=True, show_plot=False):
    """
    绘制GSAMD预处理后的数据（用于GSAMD和贝叶斯优化的强度分布）
    
    :param F_list: GSAMD处理后的数据列表（self.F_list）
    :param output_dir: 输出目录路径（如果提供，保存图片）
    :param save_plot: 是否保存图片（默认True）
    :param show_plot: 是否显示图片（默认False，避免阻塞进程）
    """
    import os
    plt.close('all')
    n_planes = len(F_list)
    
    # 根据平面数量调整布局
    if n_planes == 1:
        fig, axes = plt.subplots(1, 1, figsize=(6, 5))
        axes = [axes]
    elif n_planes == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        n_cols = min(3, n_planes)
        n_rows = (n_planes + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        axes = axes.flatten() if n_planes > 1 else [axes]
    
    for k, F_processed in enumerate(F_list):
        plane_num = k + 1
        ax = axes[k] if n_planes > 1 else axes[0]
        
        # 绘制强度分布的伪彩图
        im = ax.imshow(F_processed, cmap='jet', aspect='auto')
        ax.set_title(f'Preprocessed Intensity - Plane {plane_num}\n(For GSAMD & Bayesian Optimization)')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax, label='Intensity')
    
    # 隐藏多余的子图
    for k in range(n_planes, len(axes)):
        axes[k].axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    if save_plot and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'preprocessed_data_for_optimization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Preprocessed data plot saved to: {save_path}")
    
    # 是否显示图片
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_reconstruction_results(F_list, E_list):
    """
    绘制重建结果：N个平面 × 3列（预处理强度、重建强度、相位）
    
    :param F_list: 预处理后的强度列表（长度为N）
    :param E_list: 重建的电场列表（长度为N）
    """
    n_planes = len(F_list)
    plt.close('all')
    fig, axes = plt.subplots(n_planes, 3, figsize=(15, 4 * n_planes))
    
    # 当只有一个平面时，axes需要变成2D
    if n_planes == 1:
        axes = axes[np.newaxis, :]
    
    for k in range(n_planes):
        plane_num = k + 1
        
        # 第1列：预处理后的强度
        im1 = axes[k, 0].imshow(F_list[k], cmap='jet')
        axes[k, 0].set_title(f'Plane {plane_num} - Preprocessed Intensity')
        plt.colorbar(im1, ax=axes[k, 0])
        
        # 第2列：重建强度
        im2 = axes[k, 1].imshow(np.abs(E_list[k])**2, cmap='jet')
        axes[k, 1].set_title(f'Plane {plane_num} - Reconstructed Intensity')
        plt.colorbar(im2, ax=axes[k, 1])
        
        # 第3列：重建相位
        im3 = axes[k, 2].imshow(np.angle(E_list[k]), cmap='hsv')
        axes[k, 2].set_title(f'Plane {plane_num} - Reconstructed Phase')
        plt.colorbar(im3, ax=axes[k, 2])

    plt.tight_layout()
    plt.show()

