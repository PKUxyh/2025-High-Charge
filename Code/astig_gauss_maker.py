import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import os
from skimage.restoration import unwrap_phase
import numpy as np
from math import factorial
from scipy.special import comb

def zernike_radial(n, m, rho):
    R = np.zeros_like(rho)
    for k in range((n - abs(m)) // 2 + 1):
        numerator = (-1)**k * factorial(n - k)
        denominator = (factorial(k) * 
                      factorial((n + abs(m)) // 2 - k) * 
                      factorial((n - abs(m)) // 2 - k))
        R += numerator / denominator * rho**(n - 2*k)
    return R

def zernike_polynomial(n, m, rho, theta, norm_type='noll'):
    """
    标准Zernike多项式 Z_n^m(ρ,θ)
    
    参数:
    n: 径向阶数
    m: 角向频率 (|m| <= n, n-|m|为偶数)
    rho: 归一化径向坐标 [0,1]
    theta: 角向坐标
    norm_type: 归一化类型 ('noll' 或 'standard')
    """
    if (n - abs(m)) % 2 != 0:
        return np.zeros_like(rho)
    
    # 计算径向部分
    R = zernike_radial(n, m, rho)
    
    # 角向部分和归一化
    if norm_type == 'noll':
        # Noll归一化 (常用)
        if m == 0:
            norm_factor = np.sqrt(n + 1)
        else:
            norm_factor = np.sqrt(2 * (n + 1))
        
        if m > 0:
            return norm_factor * R * np.cos(m * theta)
        elif m < 0:
            return norm_factor * R * np.sin(abs(m) * theta)
        else:
            return norm_factor * R
    else:
        # 标准Zernike多项式 (未归一化)
        if m > 0:
            return R * np.cos(m * theta)
        elif m < 0:
            return R * np.sin(abs(m) * theta)
        else:
            return R

def zernike_coefficients(phase, n_max):
    """
    计算相位图的 Zernike 系数
    """
    rows, cols = phase.shape
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # 只考虑单位圆内的点
    mask = r <= 1
    r = r[mask]
    theta = theta[mask]
    phase_flat = phase[mask]

    # 构建 Zernike 基函数
    zernike_basis = []
    for n in range(n_max + 1):
        for m in range(-n, n + 1, 2):
            z = zernike_polynomial(n, m, r, theta)
            zernike_basis.append(z)

    zernike_basis = np.array(zernike_basis).T  # 形状为 (n_samples, n_coeffs)

    # 使用卷积分解 Zernike 系数
    coeffs = []
    for basis in zernike_basis.T:
        coeff = np.sum(phase_flat * basis) / np.sum(basis**2)
        coeffs.append(coeff)
    return np.array(coeffs)

def plot_zernike_modes(n_min, n_max):
    """
    绘制指定范围内的所有 Zernike 模式
    :param n_max: 最大 Zernike 阶数
    """
    rows, cols = 256, 256  # 图像分辨率
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    # 计算 Zernike 模式的数量
    num_modes = (n_max + 1) * (n_max + 2) // 2

    # 计算子图的行数和列数
    n_cols = 5  # 每行显示 5 个子图
    n_rows = (num_modes + n_cols - 1) // n_cols

    # 创建大图
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    fig.suptitle(f'Zernike Modes (n_max = {n_max})', fontsize=16)

    # 遍历所有 Zernike 模式并绘制
    idx = 0
    for n in range(n_min, n_max + 1):
        for m in range(-n, n + 1, 2):
            # 计算 Zernike 模式
            z = zernike_polynomial(n, m, r, theta)
            z[r > 1] = 0  # 只保留单位圆内的值

            # 绘制子图
            ax = axs[idx // n_cols, idx % n_cols]
            ax.imshow(z, cmap='viridis', extent=(-1, 1, -1, 1))
            ax.set_title(f'Z({n}, {m})')
            ax.set_xticks([])
            ax.set_yticks([])
            idx += 1

    # 隐藏多余的子图
    for i in range(idx, n_rows * n_cols):
        axs.flatten()[i].axis('off')

    plt.tight_layout()
    plt.show()

def reconstruct_phase(coeffs, shape):
    """
    根据 Zernike 系数重建相位图
    """
    rows, cols = shape
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    phase = np.zeros_like(X)
    idx = 0
    for n in range(int(np.sqrt(len(coeffs)))):
        for m in range(-n, n + 1, 2):
            z = zernike_polynomial(n, m, r, theta)
            phase += coeffs[idx] * z
            idx += 1

    # 只保留单位圆内的值
    phase[r > 1] = 0
    return phase

def create_focal_plane_field(sizex, sizey, radius, grid_size, zernike_coeffs=None):
    """
    在焦平面(z=0)处，生成一个高斯强度、Zernike相位的复振幅场。
    zernike_coeffs: Zernike系数数组（单位rad），长度应与zernike基底数量一致。
    """
    # 创建物理坐标网格
    x = np.linspace(-sizex//2, sizex//2, sizex) * grid_size
    y = np.linspace(-sizey//2, sizey//2, sizey) * grid_size
    X, Y = np.meshgrid(x, y)
    # 支持radius为标量或[radius_major, radius_minor, angle]
    if isinstance(radius, (list, tuple, np.ndarray)) and len(radius) == 3:
        r_major, r_minor, angle_deg = radius
        angle_rad = np.deg2rad(angle_deg)
        # 坐标旋转
        X_rot = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
        Y_rot = -X * np.sin(angle_rad) + Y * np.cos(angle_rad)
        r_sq = (X_rot / r_major)**2 + (Y_rot / r_minor)**2
        intensity = np.exp(-2 * r_sq)
    else:
        r_sq = X**2 + Y**2
        intensity = np.exp(-2 * r_sq / radius**2)
    amplitude = np.sqrt(intensity)

    # 坐标归一化到单位圆
    rows, cols = sizey, sizex
    center_row, center_col = rows // 2, cols // 2
    radius_pixel = min(rows, cols) // 2
    y_grid, x_grid = np.ogrid[:rows, :cols]
    rho = np.sqrt((x_grid - center_col)**2 + (y_grid - center_row)**2) / radius_pixel
    theta = np.arctan2(y_grid - center_row, x_grid - center_col)

    # 生成Zernike相位
    phase = np.zeros_like(rho)
    if zernike_coeffs is not None:
        idx = 0
        n_max = int(np.sqrt(len(zernike_coeffs)))  # 估算最大阶数
        for n in range(n_max + 1):
            for m in range(-n, n + 1, 2):
                z = zernike_polynomial(n, m, rho, theta)
                phase += zernike_coeffs[idx] * z
                idx += 1
                if idx >= len(zernike_coeffs):
                    break
            if idx >= len(zernike_coeffs):
                break
        phase[rho > 1] = 0  # 单位圆外设为0
    else:
        phase = np.zeros_like(rho)

    # 合成复振幅
    complex_field = amplitude * np.exp(1j * phase)
    return complex_field

def propagate_angular_spectrum(complex_field, z, grid_size, wavelength):
    """
    使用角谱法将一个复振幅场传播指定的距离z。
    
    参数:
    complex_field: 输入的复振幅矩阵
    z: 传播距离 (微米)
    grid_size: 网格物理尺寸 (微米/像素)
    wavelength: 激光波长 (微米)
    
    返回:
    propagated_field: 传播后的复振幅场
    """
    sizex, sizey = complex_field.shape[1], complex_field.shape[0]
    k = 2 * np.pi / wavelength
    
    # 1. 对源场进行傅里叶变换
    A = np.fft.fftshift(np.fft.fft2(complex_field))
    
    # 2. 创建空间频率坐标
    fx = np.fft.fftshift(np.fft.fftfreq(sizex, d=grid_size))
    fy = np.fft.fftshift(np.fft.fftfreq(sizey, d=grid_size))
    FX, FY = np.meshgrid(fx, fy)
    
    # 3. 计算传播因子 (Transfer Function)
    # 确保根号内的值为非负，以正确处理倏逝波
    sqrt_arg = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
    sqrt_arg[sqrt_arg < 0] = 0
    
    H = np.exp(1j * k * z * np.sqrt(sqrt_arg))
    
    # 4. 在频域中进行传播
    A_propagated = A * H
    
    # 5. 逆傅里叶变换回到空间域
    propagated_field = np.fft.ifft2(np.fft.ifftshift(A_propagated))
    
    return propagated_field

def save_matrix_to_csv(matrix, filename, folder=None):
    """将矩阵保存为CSV文件，可指定保存文件夹"""
    if folder is not None:
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, filename)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(matrix)

def plot_beam_profile(intensity, phase, grid_size, title="Beam Profile", radius_x=None, radius_y=None, angle=None):
    """绘制光束的强度和相位分布，并可选绘制椭圆及主轴"""
    phase_cmap = plt.get_cmap('twilight')  # 'twilight' 专为周期相位设计，区分度高

    extent = [
        -intensity.shape[1]//2 * grid_size, intensity.shape[1]//2 * grid_size,
        -intensity.shape[0]//2 * grid_size, intensity.shape[0]//2 * grid_size
    ]

    # 如果提供了半径，将其添加到标题中
    if radius_x is not None and radius_y is not None:
        title += f"\n$w_x$ = {radius_x:.2f} μm, $w_y$ = {radius_y:.2f} μm"
        if angle is not None:
            title += f", angle = {angle:.2f}°"

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axs[0].imshow(intensity, extent=extent, cmap='hot', origin='lower')
    axs[0].set_title('Intensity Distribution')
    axs[0].set_xlabel('x (μm)'); axs[0].set_ylabel('y (μm)')
    fig.colorbar(im0, ax=axs[0], label='Intensity')

    im1 = axs[1].imshow(phase, extent=extent, cmap=phase_cmap, origin='lower', vmin=-np.pi, vmax=np.pi)
    axs[1].set_title('Phase Distribution')
    axs[1].set_xlabel('x (μm)'); axs[1].set_ylabel('y (μm)')
    fig.colorbar(im1, ax=axs[1], label='Phase (rad)')
    # 在强度和相位图上绘制 x=0 和 y=0 的横线和竖线
    x0 = intensity.shape[1] // 2
    y0 = intensity.shape[0] // 2
    # 横线 y=0
    axs[0].axhline(y=0, color='cyan', linestyle='--', linewidth=1, xmin=0, xmax=1)
    axs[1].axhline(y=0, color='cyan', linestyle='--', linewidth=1, xmin=0, xmax=1)
    # 竖线 x=0
    axs[0].axvline(x=0, color='magenta', linestyle='--', linewidth=1, ymin=0, ymax=1)
    axs[1].axvline(x=0, color='magenta', linestyle='--', linewidth=1, ymin=0, ymax=1)

    # 找到强度最大值的位置
    max_idx = np.unravel_index(np.argmax(intensity), intensity.shape)
    max_y, max_x = max_idx
    # 计算物理坐标
    x_max = (max_x - intensity.shape[1] // 2) * grid_size
    y_max = (max_y - intensity.shape[0] // 2) * grid_size
    # 在强度和相位图上绘制最大值的横竖线
    axs[0].axhline(y=y_max, color='lime', linestyle='-', linewidth=1)
    axs[0].axvline(x=x_max, color='orange', linestyle='-', linewidth=1)
    axs[1].axhline(y=y_max, color='lime', linestyle='-', linewidth=1)
    axs[1].axvline(x=x_max, color='orange', linestyle='-', linewidth=1)
    # 绘制椭圆及主轴
    if radius_x is not None and radius_y is not None and angle is not None:
        from matplotlib.patches import Ellipse
        # 椭圆中心
        center_x = 0
        center_y = 0
        # 椭圆对象
        ellipse = Ellipse((center_x, center_y), width=2*radius_x, height=2*radius_y,
                          angle=angle, edgecolor='cyan', facecolor='none', lw=2, ls='--')
        axs[0].add_patch(ellipse)
        # 绘制长轴
        theta_rad = np.deg2rad(angle)
        dx = radius_x * np.cos(theta_rad)
        dy = radius_x * np.sin(theta_rad)
        axs[0].plot([center_x - dx, center_x + dx], [center_y - dy, center_y + dy], color='lime', lw=2)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def calculate_beam_radii(intensity, grid_size):
    """
    通过二阶矩方法计算强度分布的椭圆长短轴1/e²半径及主轴角度。
    这种方法对于旋转的椭圆也是准确的。
    
    参数:
    intensity: 2D强度矩阵
    grid_size: 每个像素的物理尺寸 (微米)
    
    返回:
    (radius_major, radius_minor, angle): 长轴半径、短轴半径 (微米)，主轴角度(度, x轴逆时针)
    """
    # 创建物理坐标网格
    size_y, size_x = intensity.shape
    x = np.linspace(-size_x//2, size_x//2, size_x) * grid_size
    y = np.linspace(-size_y//2, size_y//2, size_y) * grid_size
    X, Y = np.meshgrid(x, y)
    
    # 计算总功率 (用作归一化因子)
    total_power = np.sum(intensity)
    if total_power == 0:
        return 0, 0, 0
        
    # 计算质心 (一阶矩)
    x_c = np.sum(X * intensity) / total_power
    y_c = np.sum(Y * intensity) / total_power
    
    # 计算相对于质心的二阶中心矩
    var_x = np.sum((X - x_c)**2 * intensity) / total_power
    var_y = np.sum((Y - y_c)**2 * intensity) / total_power
    cov_xy = np.sum((X - x_c) * (Y - y_c) * intensity) / total_power
    
    # 构造协方差矩阵
    cov_matrix = np.array([[var_x, cov_xy], [cov_xy, var_y]])
    
    # 计算特征值和特征向量
    try:
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    except np.linalg.LinAlgError:
        return 0, 0, 0 # 如果矩阵有问题，返回0
    
    # 对于高斯光束, 1/e² 半径 w = 2 * sigma。
    # 因此，半径是特征值平方根的2倍。
    idx_major = np.argmax(eigvals)
    idx_minor = np.argmin(eigvals)
    radius_major = 2 * np.sqrt(eigvals[idx_major])
    radius_minor = 2 * np.sqrt(eigvals[idx_minor])
    
    # 主轴角度（与x轴夹角，单位：度，逆时针为正）
    major_axis_vec = eigvecs[:, idx_major]
    angle = np.arctan2(major_axis_vec[1], major_axis_vec[0]) * 180 / np.pi
    
    return radius_major, radius_minor, angle


def scan_beam_waist_vs_z(initial_complex_field, grid_size, wavelength, z_min, z_max, z_step):
    """
    扫描一系列z位置，计算每个z处的x、y方向光斑半径，并绘制半径随z变化曲线。
    """
    zs = np.arange(z_min, z_max + z_step, z_step)
    radii_x = []
    radii_y = []
    for z in zs:
        field_z = propagate_angular_spectrum(initial_complex_field, z, grid_size, wavelength)
        intensity_z = np.abs(field_z)**2
        rx, ry, angle = calculate_beam_radii(intensity_z, grid_size)
        radii_x.append(rx)
        radii_y.append(ry)
    plt.figure(figsize=(8,5))
    plt.plot(zs, radii_x, label='x')
    plt.plot(zs, radii_y, label='y')
    plt.xlabel('z (μm)')
    plt.ylabel('1/e² r (μm)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # 找到最小值及对应z
    min_idx_x = np.argmin(radii_x)
    min_idx_y = np.argmin(radii_y)
    print(f"Minimum x-waist: {radii_x[min_idx_x]:.2f} μm at z = {zs[min_idx_x]:.1f} μm")
    print(f"Minimum y-waist: {radii_y[min_idx_y]:.2f} μm at z = {zs[min_idx_y]:.1f} μm")

def crop_center_square(phase, size=100):
    """
    以相位矩阵为中心截取一个正方形区域
    
    参数:
    phase: 二维numpy数组，相位矩阵
    size: 要截取的正方形边长，默认100
    
    返回:
    cropped_phase: 截取后的正方形区域
    """
    h, w = phase.shape
    if h < size or w < size:
        raise ValueError(f"原始矩阵大小{h}x{w}小于要截取的尺寸{size}x{size}")
    
    c_h = h // 2
    c_w = w // 2
    
    # 计算截取的区域
    start_h = c_h - size//2
    end_h = c_h + size//2
    start_w = c_w - size//2
    end_w = c_w + size//2
    
    cropped_phase = phase[start_h:end_h, start_w:end_w]
    return cropped_phase

def scan_angular_fourier_coeffs_vs_z(initial_complex_field, grid_size, wavelength, z_min, z_max, z_step, num_coeffs=10):
    """
    扫描一系列 z 位置，计算每个 z 处的角向傅里叶系数，并记录前 num_coeffs 项的演化曲线。
    最后绘制这些系数随 z 变化的曲线。
    
    参数:
    initial_complex_field: 初始复振幅场
    grid_size: 每个像素的物理尺寸 (μm)
    wavelength: 激光波长 (μm)
    z_min: z 的最小值 (μm)
    z_max: z 的最大值 (μm)
    z_step: z 的步长 (μm)
    num_coeffs: 要记录的角向傅里叶系数的数量 (默认前 10 项)
    """
    zs = np.arange(z_min, z_max + z_step, z_step)
    coeffs_evolution = np.zeros((len(zs), num_coeffs))  # 存储每个 z 的前 num_coeffs 项系数

    for i, z in enumerate(zs):
        # 传播到当前 z 位置
        field_z = propagate_angular_spectrum(initial_complex_field, z, grid_size, wavelength)
        
        # 计算角向傅里叶系数
        coeffs_mean = compute_angular_fourier_coeffs(field_z)
        
        # 记录前 num_coeffs 项系数
        coeffs_evolution[i, :] = coeffs_mean[:num_coeffs]

    # 绘制前 num_coeffs 项系数随 z 的演化曲线
    plt.figure(figsize=(10, 6))
    for n in range(0, num_coeffs, 2):
        plt.plot(zs, coeffs_evolution[:, n], label=f"m={n}")
    plt.xlabel("z (μm)")
    plt.ylabel("Mean Amplitude of Angular Fourier Coefficients")
    plt.yscale("log")
    plt.title(f"Evolution of Angular Fourier Coefficients (First {num_coeffs} Terms)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_angular_fourier_coeffs(complex_field, num_theta=360, max_radius=None):
    """
    计算以图像中心为对称轴的角向Fourier系数。
    对每个半径上的复振幅做角向傅里叶变换，输出各阶角频率分量的平均幅值。
    
    参数:
    complex_field: 2D复振幅场
    num_theta: 角度采样点数（建议360）
    max_radius: 最大分析半径（像素），默认取中心到边缘最短距离
    
    返回:
    coeffs_mean: 各阶角频率分量的平均幅值（数组，长度=num_theta//2+1）
    """
    rows, cols = complex_field.shape
    center_row, center_col = rows // 2, cols // 2
    if max_radius is None:
        max_radius = min(center_row, center_col)
    radii = np.arange(1, max_radius)
    coeffs_accum = []
    for r in radii:
        # 沿圆周采样
        thetas = np.linspace(0, 2*np.pi, num_theta, endpoint=False)
        xs = center_col + r * np.cos(thetas)
        ys = center_row + r * np.sin(thetas)
        # 双线性插值采样
        values = np.array([
            complex_field[int(round(y)), int(round(x))]
            if 0 <= int(round(y)) < rows and 0 <= int(round(x)) < cols else 0
            for x, y in zip(xs, ys)
        ])
        # 对圆周上的复振幅做FFT
        coeffs = np.fft.fft(values)
        coeffs_accum.append(coeffs)
    coeffs_accum = np.array(coeffs_accum)  # shape: (num_radii, num_theta)
    # 取每阶角频率的平均幅值
    coeffs_mean = np.mean(np.abs(coeffs_accum), axis=0)[:num_theta//2+1]
    return coeffs_mean

def integrate_intensity(intensity, grid_size):
    """
    计算二维强度分布的积分值（总能量），单位为强度*面积。
    intensity: 2D强度矩阵
    grid_size: 每个像素的物理尺寸 (μm)
    返回: 积分值
    """
    area_per_pixel = grid_size ** 2
    total = np.sum(intensity) * area_per_pixel
    return total


def get_zernike_names(max_terms=50):
    """
    获取Zernike多项式名称 (OSA/ANSI标准索引) - 坐标轴表示版本
    
    参数:
    max_terms: 最大项数
    
    返回:
    zernike_names: Zernike名称列表
    """
    # OSA/ANSI标准Zernike名称 - 坐标轴表示 (前50项)
    zernike_names = [
        # 0-4阶
        "Piston",           # 0 (0,0)
        "Y-Tilt",           # 1 (1,-1)
        "X-Tilt",           # 2 (1,1)
        "45° Astig",        # 3 (2,-2)
        "Defocus",          # 4 (2,0)
        "0° Astig",         # 5 (2,2)
        
        # 3阶
        "Y-Trefoil",        # 6 (3,-3)
        "Y-Coma",           # 7 (3,-1)
        "X-Coma",           # 8 (3,1)
        "X-Trefoil",        # 9 (3,3)
        
        # 4阶
        "45° Quadrafoil",   # 10 (4,-4)
        "45° 2nd Astig",    # 11 (4,-2)
        "Primary Sph",      # 12 (4,0)
        "0° 2nd Astig",     # 13 (4,2)
        "0° Quadrafoil",    # 14 (4,4)
        
        # 5阶
        "Y-Pentafoil",      # 15 (5,-5)
        "45° 2nd Trefoil",  # 16 (5,-3)
        "45° 2nd Coma",     # 17 (5,-1)
        "Y-2nd Coma",       # 18 (5,1)
        "X-2nd Trefoil",    # 19 (5,3)
        "X-Pentafoil",      # 20 (5,5)
        
        # 6阶
        "45° Hexafoil",     # 21 (6,-6)
        "45° 2nd Quadrafoil", # 22 (6,-4)
        "45° 3rd Astig",    # 23 (6,-2)
        "Secondary Sph",    # 24 (6,0)
        "0° 3rd Astig",     # 25 (6,2)
        "0° 2nd Quadrafoil", # 26 (6,4)
        "0° Hexafoil",      # 27 (6,6)
        
        # 7阶
        "Y-Heptafoil",      # 28 (7,-7)
        "45° 2nd Pentafoil",# 29 (7,-5)
        "45° 3rd Trefoil",  # 30 (7,-3)
        "45° 3rd Coma",     # 31 (7,-1)
        "Y-3rd Coma",       # 32 (7,1)
        "X-3rd Trefoil",    # 33 (7,3)
        "Y-2nd Pentafoil",  # 34 (7,5)
        "X-Heptafoil",      # 35 (7,7)
        
        # 8阶
        "45° Octafoil",     # 36 (8,-8)
        "45° 2nd Hexafoil", # 37 (8,-6)
        "45° 3rd Quadrafoil", # 38 (8,-4)
        "45° 4th Astig",    # 39 (8,-2)
        "Tertiary Sph",     # 40 (8,0)
        "0° 4th Astig",     # 41 (8,2)
        "0° 3rd Quadrafoil", # 42 (8,4)
        "0° 2nd Hexafoil",  # 43 (8,6)
        "0° Octafoil",      # 44 (8,8)
        
        # 9阶
        "Y-Nonafoil",       # 45 (9,-9)
        "45° 2nd Heptafoil",# 46 (9,-7)
        "45° 3rd Pentafoil",# 47 (9,-5)
        "45° 4th Trefoil",  # 48 (9,-3)
        "45° 4th Coma",     # 49 (9,-1)
    ]
    return zernike_names[:max_terms]

def get_zernike_names_nm(max_terms=50):
    """
    获取Zernike多项式名称 (n,m)索引版本 - LaTeX格式
    
    参数:
    max_terms: 最大项数
    
    返回:
    zernike_names: Zernike名称列表，格式为 Z_{n}^{m}
    """
    # (n,m)索引的Zernike名称 (前50项)
    zernike_names = [
        # 0阶
        r'$Z_{0}^{0}$',      # 0 (0,0)
        
        # 1阶
        r'$Z_{1}^{-1}$',     # 1 (1,-1)
        r'$Z_{1}^{1}$',      # 2 (1,1)
        
        # 2阶
        r'$Z_{2}^{-2}$',     # 3 (2,-2)
        r'$Z_{2}^{0}$',      # 4 (2,0)
        r'$Z_{2}^{2}$',      # 5 (2,2)
        
        # 3阶
        r'$Z_{3}^{-3}$',     # 6 (3,-3)
        r'$Z_{3}^{-1}$',     # 7 (3,-1)
        r'$Z_{3}^{1}$',      # 8 (3,1)
        r'$Z_{3}^{3}$',      # 9 (3,3)
        
        # 4阶
        r'$Z_{4}^{-4}$',     # 10 (4,-4)
        r'$Z_{4}^{-2}$',     # 11 (4,-2)
        r'$Z_{4}^{0}$',      # 12 (4,0)
        r'$Z_{4}^{2}$',      # 13 (4,2)
        r'$Z_{4}^{4}$',      # 14 (4,4)
        
        # 5阶
        r'$Z_{5}^{-5}$',     # 15 (5,-5)
        r'$Z_{5}^{-3}$',     # 16 (5,-3)
        r'$Z_{5}^{-1}$',     # 17 (5,-1)
        r'$Z_{5}^{1}$',      # 18 (5,1)
        r'$Z_{5}^{3}$',      # 19 (5,3)
        r'$Z_{5}^{5}$',      # 20 (5,5)
        
        # 6阶
        r'$Z_{6}^{-6}$',     # 21 (6,-6)
        r'$Z_{6}^{-4}$',     # 22 (6,-4)
        r'$Z_{6}^{-2}$',     # 23 (6,-2)
        r'$Z_{6}^{0}$',      # 24 (6,0)
        r'$Z_{6}^{2}$',      # 25 (6,2)
        r'$Z_{6}^{4}$',      # 26 (6,4)
        r'$Z_{6}^{6}$',      # 27 (6,6)
        
        # 7阶
        r'$Z_{7}^{-7}$',     # 28 (7,-7)
        r'$Z_{7}^{-5}$',     # 29 (7,-5)
        r'$Z_{7}^{-3}$',     # 30 (7,-3)
        r'$Z_{7}^{-1}$',     # 31 (7,-1)
        r'$Z_{7}^{1}$',      # 32 (7,1)
        r'$Z_{7}^{3}$',      # 33 (7,3)
        r'$Z_{7}^{5}$',      # 34 (7,5)
        r'$Z_{7}^{7}$',      # 35 (7,7)
        
        # 8阶
        r'$Z_{8}^{-8}$',     # 36 (8,-8)
        r'$Z_{8}^{-6}$',     # 37 (8,-6)
        r'$Z_{8}^{-4}$',     # 38 (8,-4)
        r'$Z_{8}^{-2}$',     # 39 (8,-2)
        r'$Z_{8}^{0}$',      # 40 (8,0)
        r'$Z_{8}^{2}$',      # 41 (8,2)
        r'$Z_{8}^{4}$',      # 42 (8,4)
        r'$Z_{8}^{6}$',      # 43 (8,6)
        r'$Z_{8}^{8}$',      # 44 (8,8)
        
        # 9阶
        r'$Z_{9}^{-9}$',     # 45 (9,-9)
        r'$Z_{9}^{-7}$',     # 46 (9,-7)
        r'$Z_{9}^{-5}$',     # 47 (9,-5)
        r'$Z_{9}^{-3}$',     # 48 (9,-3)
        r'$Z_{9}^{-1}$',     # 49 (9,-1)
    ]
    return zernike_names[:max_terms]


def pad_field_custom_size(complex_field, target_height=None, target_width=None):
    """
    将激光场分布填充到指定尺寸，原数据置于中心，周围用0填充
    
    参数:
    complex_field: 输入的复振幅场 (2D numpy数组)
    target_height: 目标高度 (如果为None，则使用2倍原高度)
    target_width: 目标宽度 (如果为None，则使用2倍原宽度)
    
    返回:
    padded_field: 填充后的复振幅场
    """
    # 获取原始场的尺寸
    original_height, original_width = complex_field.shape
    
    # 设置目标尺寸
    if target_height is None:
        target_height = 2 * original_height
    if target_width is None:
        target_width = 2 * original_width
    
    # 确保目标尺寸不小于原始尺寸
    target_height = max(target_height, original_height)
    target_width = max(target_width, original_width)
    
    # 创建新的零矩阵
    padded_field = np.zeros((target_height, target_width), dtype=complex_field.dtype)
    
    # 计算原始场在新矩阵中的位置 (居中)
    start_row = (target_height - original_height) // 2
    start_col = (target_width - original_width) // 2
    end_row = start_row + original_height
    end_col = start_col + original_width
    
    # 将原始场复制到中心位置
    padded_field[start_row:end_row, start_col:end_col] = complex_field
    
    return padded_field


def crop_to_custom_size(padded_field, target_height=None, target_width=None):
    """
    移除外侧区域，将填充后的场裁剪到指定尺寸（中心区域）
    
    参数:
    padded_field: 填充后的复振幅场 (2D numpy数组)
    target_height: 目标高度 (如果为None，则使用一半高度)
    target_width: 目标宽度 (如果为None，则使用一半宽度)
    
    返回:
    cropped_field: 裁剪后的复振幅场
    """
    # 获取填充后场的尺寸
    padded_height, padded_width = padded_field.shape
    
    # 设置目标尺寸
    if target_height is None:
        target_height = padded_height // 2
    if target_width is None:
        target_width = padded_width // 2
    
    # 确保目标尺寸不大于原始尺寸
    target_height = min(target_height, padded_height)
    target_width = min(target_width, padded_width)
    
    # 计算中心区域的起始位置
    start_row = (padded_height - target_height) // 2
    start_col = (padded_width - target_width) // 2
    end_row = start_row + target_height
    end_col = start_col + target_width
    
    # 裁剪中心区域
    cropped_field = padded_field[start_row:end_row, start_col:end_col]
    
    return cropped_field


def propagation_with_padding_and_cropping(initial_field, z_distance, grid_size, wavelength):
    """
    带填充和裁剪的完整传播工作流程
    """
    # 1. 保存原始尺寸
    original_shape = initial_field.shape
    
    # 2. 填充场（2倍大小）
    padded_field = pad_field_custom_size(initial_field)
    
    # 3. 传播
    propagated_padded_field = propagate_angular_spectrum(
        padded_field, z_distance, grid_size, wavelength
    )
    
    # 4. 裁剪回原始尺寸
    final_field = crop_to_custom_size(propagated_padded_field)
    
    return final_field



# --- 主程序 ---
if __name__ == "__main__":
    # 1. 定义基本参数
    size_x = 201
    size_y = 201
    radius = [22.4, 29.26, 180*0.135]#25.83    # 焦平面(z=0)处的1/e²强度半径 (微米)
    wavelength = 0.8
    grid_size = 0.88
    a0 = 1.44
    # astig_strength = 0.2
    # zernile_list = [
    #     0, 0, 0, 3.8288, 0, 0, 0, 0, 0, 0  # Zernike系数
    # ]

    zernile_list = [
        0, 0, 0, 0.7447, 0, 0.6559, 0, 0, 0, 0  # Zernike系数
    ]


    # zernile_list = [
    #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # Zernike系数
    # ]
    # 2. 在焦平面(z=0)创建初始场
    print("Step 1: Creating the initial field at the focal plane (z=0)...")
    initial_complex_field = create_focal_plane_field(
        size_x, size_y, radius, grid_size, zernile_list
    )
    # 将模的平方最大值归一化为 a0 的平方
    initial_complex_field = initial_complex_field * (a0 / np.sqrt(np.max(np.abs(initial_complex_field)**2)))


    # 提取并绘制初始场的强度和相位
    initial_intensity = np.abs(initial_complex_field)**2
    initial_phase = np.angle(initial_complex_field)
    total_intensity = integrate_intensity(initial_intensity, grid_size)
    # 对初始相位进行解缠
    initial_phase_unwrapped = unwrap_phase(initial_phase)
    # 计算初始椭圆参数
    radiix0, radiiy0, angle0 = calculate_beam_radii(initial_intensity, grid_size)
    plot_beam_profile(initial_intensity, initial_phase, grid_size, 
                      title=f"Initial Beam Phase Unwrapped at Focal Plane (z=0 μm)\n",
                      radius_x=radiix0, radius_y=radiiy0, angle=angle0)
    print("Initial field created at the focal plane (z=0 μm). max intensity:", np.max(initial_intensity), " Total integrated intensity:", total_intensity)

    # 3. 将初始场传播到指定的离焦位置
    z_propagate = 0.0  # <--- 在这里设置您想要的任何离焦距离

    # print(f"\nStep 2: Propagating the field to z = {z_propagate} μm...")
    propagated_complex_field = propagate_angular_spectrum(
        initial_complex_field, z_propagate, grid_size, wavelength
    )
    # propagated_complex_field = propagation_with_padding_and_cropping(
    #     initial_complex_field, z_propagate, grid_size, wavelength
    # )

    # 4. 提取、绘制并保存传播后的结果
    final_intensity = np.abs(propagated_complex_field)**2
    final_phase = np.angle(propagated_complex_field)
    final_phase_unwrapped = unwrap_phase(final_phase)
    radiix, radiiy, angle = calculate_beam_radii(final_intensity, grid_size)

    plot_beam_profile(final_intensity, final_phase_unwrapped, grid_size,
                      title=f"Beam Profile at z={z_propagate} μm\n$w_x$ = {radiix:.2f} μm, $w_y$ = {radiiy:.2f} μm, angle = {angle:.2f}°",
                      radius_x=radiix, radius_y=radiiy, angle=angle)
    print(f"Final field propagated to z={z_propagate} μm. max intensity: {np.max(final_intensity)}")

    # # # 保存最终结果到CSV文件
    # output_folder = r'F:\code\GSAMD'
    # save_matrix_to_csv(initial_intensity, f'ea_intensity.csv', folder=output_folder)
    # save_matrix_to_csv(initial_phase_unwrapped, f'ea_phase.csv', folder=output_folder)
    # print(f"Propagated intensity and phase at z={z_propagate}μm have been saved to CSV files.")

    # # 保存最终结果到CSV文件
    # output_folder = r'F:\paper\GS_new\GSA-MD\paper_used\后处理\deflection'
    # save_matrix_to_csv(final_intensity, f'real_ellipicity_intensity.csv', folder=output_folder)
    # save_matrix_to_csv(final_phase_unwrapped, f'real_ellipicity_phase.csv', folder=output_folder)
    # print(f"Propagated intensity and phase at z={z_propagate}μm have been saved to CSV files.")


    # initial_complex_field = crop_center_square(initial_complex_field, size=0)
    scan_beam_waist_vs_z(initial_complex_field, grid_size, wavelength, -2000, 2000, 50)
    # scan_angular_fourier_coeffs_vs_z(
    #     initial_complex_field=initial_complex_field,
    #     grid_size=grid_size,
    #     wavelength=wavelength,
    #     z_min=-2000,
    #     z_max=2000,
    #     z_step=50,
    #     num_coeffs=16  # 记录前 8 项
    # )



    phase = crop_center_square(final_phase_unwrapped, size=100)
    # phase = final_phase_unwrapped
    # 中心化和裁剪
    rows, cols = phase.shape
    center_row, center_col = rows // 2, cols // 2
    radius = min(rows, cols) // 2
    y, x = np.ogrid[:rows, :cols]
    mask = (x - center_col)**2 + (y - center_row)**2 <= radius**2
    phase[~mask] = 0
    
    # 计算Zernike系数 (使用标准方法)
    max_zernike_index = 15  # 计算前36项Zernike系数
    coeffs = zernike_coefficients(phase, max_zernike_index)
    
    # 只绘制前15个系数
    num_coeffs_to_plot = 15
    plt.figure(figsize=(10, 4))
    plt.bar(range(1, num_coeffs_to_plot + 1), coeffs[:num_coeffs_to_plot])
    plt.xlabel('Zernike Index (OSA ordering)')
    plt.ylabel('Coefficient Value (rad)')
    plt.title('First 15 Zernike Coefficients')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 重建相位
    reconstructed = reconstruct_phase(coeffs, phase.shape)
    
    # 计算残差
    residual = phase - reconstructed
    residual[~mask] = 0  # 只考虑孔径内
    
    # 绘制结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axes[0].imshow(phase, cmap='viridis')
    axes[0].set_title('Original Phase')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(reconstructed, cmap='viridis')
    axes[1].set_title('Reconstructed Phase')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(residual, cmap='RdBu_r')
    axes[2].set_title('Residual (Original - Reconstructed)')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    zernike_names_osa = get_zernike_names(max_terms=len(coeffs))

    print("Important Zernike Coefficients (OSA/ANSI order):")
    for i, name in enumerate(zernike_names_osa, start=1):
        if i <= len(coeffs):
            print(f"{name:22s}: {coeffs[i-1]:.4f} rad")





    coeffs_mean = compute_angular_fourier_coeffs(propagated_complex_field)
    plt.figure(figsize=(8, 4))
    plt.scatter(range(1, 16), coeffs_mean[1:16], color='blue', s=40)
    plt.xlabel("Angular frequency (m)")
    plt.ylabel("Mean amplitude")
    plt.title("Angular FFT (centered, first 15)")
    plt.tight_layout()
    plt.show()
