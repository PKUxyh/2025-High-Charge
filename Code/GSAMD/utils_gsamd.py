import numpy as np
from scipy.special import hermite
import math
import os


def crop_center(arr, cropx, cropy):
    arr[arr < 0] = 0  # 确保没有负值
    arr = arr[:, :-1]
    y, x = np.unravel_index(np.argmax(arr), arr.shape)
    startx = max(x - cropx // 2, 0)
    starty = max(y - cropy // 2, 0)
    endx = startx + cropx
    endy = starty + cropy
    # 防止越界
    if endx > arr.shape[1]:
        endx = arr.shape[1]
        startx = endx - cropx
    if endy > arr.shape[0]:
        endy = arr.shape[0]
        starty = endy - cropy
    return arr[starty:endy, startx:endx]


def HG_mode(m, n, x, y, x0, y0, z, w0_x, w0_y, wavelength):
    """
    定义Hermite-Gauss模式
    :param m: x方向模式阶数
    :param n: y方向模式阶数
    :param x, y: 网格坐标
    :param x0, y0: 模式中心
    :param z: 传播距离（相对于焦点）
    :param w0_x, w0_y: 初始腰斑大小
    :param wavelength: 激光波长
    """
    k0 = 2 * np.pi / wavelength
    ZR_x = np.pi * w0_x**2 / wavelength  # x方向瑞利长度
    ZR_y = np.pi * w0_y**2 / wavelength  # y方向瑞利长度
    
    # x方向模式
    wx = w0_x * np.sqrt(1 + (z/ZR_x)**2)
    Rx = z + ZR_x**2 / z if z != 0 else np.inf
    phase_x = (m + 0.5) * np.arctan(z/ZR_x)
    Hm = hermite(m)(np.sqrt(2)*(x - x0)/wx)
    Am = 1/np.sqrt(wx * 2**(m-0.5) * math.factorial(m) * np.sqrt(np.pi))
    term_x = Am * Hm * np.exp(-(x - x0)**2 / wx**2) * np.exp(-1j * k0 * (x - x0)**2 / (2*Rx))
    
    # y方向模式
    wy = w0_y * np.sqrt(1 + (z/ZR_y)**2)
    Ry = z + ZR_y**2 / z if z != 0 else np.inf
    phase_y = (n + 0.5) * np.arctan(z/ZR_y)
    Hn = hermite(n)(np.sqrt(2)*(y - y0)/wy)
    An = 1/np.sqrt(wy * 2**(n-0.5) * math.factorial(n) * np.sqrt(np.pi))
    term_y = An * Hn * np.exp(-(y - y0)**2 / wy**2) * np.exp(-1j * k0 * (y - y0)**2 / (2*Ry))
    
    # 整体模式
    # A = 1/np.sqrt(np.sqrt(np.pi) * 2**(m+n) * np.math.factorial(m) * np.math.factorial(n) * wx * wy)
    total_phase = np.exp(1j*(phase_x + phase_y))
    
    return term_x * term_y * total_phase

def apply_hg_filter(F, x_grid, y_grid, x0, y0, z, w0_x, w0_y, wavelength, dx, dy, N_modes=(20, 20)):
    """
    HG模式滤波：将光强分布投影到HG模式基上，只保留低阶模式
    
    :param F: 输入光强分布（阈值处理后的）
    :param x_grid, y_grid: 坐标网格
    :param x0, y0: 模式中心
    :param z: 传播距离
    :param w0_x, w0_y: 腰斑大小
    :param wavelength: 波长
    :param dx, dy: 网格间距
    :param N_modes: 要保留的模式阶数 (m_max, n_max)，默认(20, 20)
    :return: 滤波后的光强分布
    """
    # 从光强分布计算电场（假设初始相位为0）
    E = np.sqrt(F)
    
    # 计算HG模式系数
    C = np.zeros((N_modes[0], N_modes[1]), dtype=complex)
    for m in range(N_modes[0]):
        for n in range(N_modes[1]):
            mode = HG_mode(m, n, x_grid, y_grid, x0, y0, z, 
                          w0_x, w0_y, wavelength)
            # 计算投影系数
            C[m, n] = np.sum(E * mode.conj()) * dx * dy / np.sum(np.abs(mode)**2 * dx * dy)
    
    # 用保留的模式重建电场
    E_filtered = np.zeros_like(x_grid, dtype=complex)
    for m in range(N_modes[0]):
        for n in range(N_modes[1]):
            mode = HG_mode(m, n, x_grid, y_grid, x0, y0, z,
                          w0_x, w0_y, wavelength)
            E_filtered += C[m, n] * mode
    
    # 返回滤波后的光强分布
    F_filtered = np.abs(E_filtered)**2
    # 确保非负
    F_filtered[F_filtered < 0] = 0
    
    return F_filtered


def preprocess_fluence(F_exp, threshold=0.01, lowpass_filter=False, filter_sigma=2.0, 
                       hg_filter=False, hg_params=None, save_path=None, plane_index=None):
    """
    预处理：背景减除、阈值处理、低通滤波（或HG模式滤波）
    
    :param F_exp: 输入光强分布
    :param threshold: 阈值（相对于最大值）
    :param lowpass_filter: 是否启用低通滤波（高斯滤波，默认False）
    :param filter_sigma: 高斯滤波的标准差（默认2.0）
    :param hg_filter: 是否启用HG模式滤波（默认False）
    :param hg_params: HG滤波参数字典，包含：
        - x_grid, y_grid: 坐标网格
        - x0, y0: 模式中心
        - z: 传播距离
        - w0_x, w0_y: 腰斑大小
        - wavelength: 波长
        - dx, dy: 网格间距
        - N_modes: 要保留的模式阶数 (m_max, n_max)，默认(20, 20)
    :param save_path: 保存路径（如果提供，将保存滤波后的数据）
    :param plane_index: 平面索引（用于文件命名）
    :return: 预处理后的光强分布
    """
    from scipy.ndimage import gaussian_filter
    import matplotlib.pyplot as plt
    
    F = F_exp.copy()
    # 阈值处理
    F[F < threshold * F.max()] = 0

    # 保存滤波前的数据（用于对比绘图）
    F_before = F.copy()

    # 低通滤波（高斯滤波，在阈值处理后）
    if lowpass_filter:
        F = gaussian_filter(F, sigma=filter_sigma)
        filter_name = f'Gaussian (σ={filter_sigma})'
        
        # 保存滤波后的初始数据
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            if plane_index is not None:
                filename = os.path.join(save_path, f'preprocessed_lowpass_filtered_plane{plane_index+1}.csv')
            else:
                filename = os.path.join(save_path, 'preprocessed_lowpass_filtered.csv')
            np.savetxt(filename, F, delimiter=",")
    
    # HG模式滤波（在阈值处理后，如果启用）
    elif hg_filter and hg_params is not None:
        N_modes = hg_params.get('N_modes', (20, 20))
        F = apply_hg_filter(F, 
                           hg_params['x_grid'], 
                           hg_params['y_grid'],
                           hg_params['x0'],
                           hg_params['y0'],
                           hg_params['z'],
                           hg_params['w0_x'],
                           hg_params['w0_y'],
                           hg_params['wavelength'],
                           hg_params['dx'],
                           hg_params['dy'],
                           N_modes=N_modes)
        filter_name = f'HG Filter {N_modes}'
        
        # 保存滤波后的初始数据
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            if plane_index is not None:
                filename = os.path.join(save_path, f'preprocessed_hg_filtered_plane{plane_index+1}.csv')
            else:
                filename = os.path.join(save_path, 'preprocessed_hg_filtered.csv')
            np.savetxt(filename, F, delimiter=",")
    else:
        filter_name = None

    # 绘制滤波前后对比图
    if filter_name is not None:
        plane_label = f' (Plane {plane_index+1})' if plane_index is not None else ''
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        im0 = axes[0].imshow(F_before, cmap='jet', aspect='equal')
        axes[0].set_title(f'Before Filter{plane_label}')
        plt.colorbar(im0, ax=axes[0], shrink=0.8)
        
        im1 = axes[1].imshow(F, cmap='jet', aspect='equal')
        axes[1].set_title(f'After {filter_name}{plane_label}')
        plt.colorbar(im1, ax=axes[1], shrink=0.8)
        
        # 差值图
        diff = F_before - F
        im2 = axes[2].imshow(diff, cmap='RdBu_r', aspect='equal')
        axes[2].set_title(f'Difference (Before - After){plane_label}')
        plt.colorbar(im2, ax=axes[2], shrink=0.8)
        
        plt.tight_layout()
        
        # 保存对比图
        if save_path is not None:
            fig_path = os.path.join(save_path, f'filter_comparison_plane{plane_index+1 if plane_index is not None else 0}.png')
            plt.savefig(fig_path, dpi=200, bbox_inches='tight')
            print(f"  Filter comparison plot saved to: {fig_path}")
        
        plt.show()

    return F
