import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap
import os

def generate_elliptical_gaussian_matrix(sizex, sizey, radius_x, radius_y, grid_size, wavelength, z=0):
    """
    生成具有物理意义自洽相位的椭圆高斯光束复振幅分布
    
    参数:
    sizex, sizey: 矩阵尺寸 (像素数)
    radius_x, radius_y: x和y方向的1/e²强度半径 (微米)
    grid_size: 网格物理尺寸 (微米/像素)
    wavelength: 激光波长 (微米)
    z: 离焦距离 (微米)，z=0表示束腰位置
    
    返回:
    intensity_matrix: 强度分布矩阵
    phase_matrix: 相位分布矩阵
    """
    # 创建物理坐标网格
    x = np.linspace(-sizex//2, sizex//2, sizex) * grid_size
    y = np.linspace(-sizey//2, sizey//2, sizey) * grid_size
    X, Y = np.meshgrid(x, y)
    
    # 计算瑞利范围
    zRx = np.pi * radius_x**2 / wavelength  # x方向瑞利范围
    zRy = np.pi * radius_y**2 / wavelength  # y方向瑞利范围
    
    # 计算离焦位置的光束半径
    wx = radius_x * np.sqrt(1 + (z/zRx)**2) if zRx != 0 else radius_x
    wy = radius_y * np.sqrt(1 + (z/zRy)**2) if zRy != 0 else radius_y
    
    # 计算波前曲率半径
    Rx = z * (1 + (zRx/z)**2) if z != 0 and zRx != 0 else np.inf
    Ry = z * (1 + (zRy/z)**2) if z != 0 and zRy != 0 else np.inf
    
    # 计算Gouy相位
    gouy_phase_x = np.arctan(z/zRx) if zRx != 0 else 0
    gouy_phase_y = np.arctan(z/zRy) if zRy != 0 else 0
    gouy_phase = 0.5 * (gouy_phase_x + gouy_phase_y)
    
    # 波数
    k = 2 * np.pi / wavelength
    
    # 计算复振幅
    amplitude = np.exp(-(X**2)/(wx**2) - (Y**2)/(wy**2))
    
    # 计算相位分布 (自洽)
    phase = (
        -k * z  # 传播相位
        + gouy_phase  # Gouy相位
        - (k * X**2) / (2 * Rx)  # x方向波前曲率
        - (k * Y**2) / (2 * Ry)  # y方向波前曲率
    )
    
    # 创建复振幅场
    complex_field = amplitude * np.exp(1j * phase)
    
    # 提取强度和相位
    intensity_matrix = np.abs(complex_field)**2
    phase_matrix = np.angle(complex_field)
    
    return intensity_matrix, phase_matrix, wx, wy

def save_matrix_to_csv(matrix, filename, folder=None):
    """将矩阵保存为CSV文件，可指定保存文件夹"""
    if folder is not None:
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, filename)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(matrix)

def plot_elliptical_beam(intensity, phase, grid_size, title="Elliptical Gaussian Beam"):
    """绘制椭圆高斯光束的强度和相位分布"""
    # 创建自定义相位颜色映射
    colors = [
        (0, 0, 1),    # 蓝色 (相位-π)
        (0.5, 0, 1),  # 紫色
        (1, 0, 1),    # 品红
        (1, 0, 0),    # 红色 (相位0)
        (1, 1, 0),    # 黄色
        (0, 1, 0),    # 绿色
        (0, 1, 1),    # 青色
        (0, 0, 1)     # 蓝色 (相位π)
    ]
    phase_cmap = LinearSegmentedColormap.from_list('phase_cmap', colors, N=256)
    
    # 计算物理范围
    extent = [
        -intensity.shape[1]//2 * grid_size,
        intensity.shape[1]//2 * grid_size,
        -intensity.shape[0]//2 * grid_size,
        intensity.shape[0]//2 * grid_size
    ]
    
    # 创建绘图
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # 强度分布
    im0 = axs[0].imshow(intensity, 
                       extent=extent,
                       cmap='hot', origin='lower')
    axs[0].set_title('Intensity Distribution')
    axs[0].set_xlabel('x (μm)')
    axs[0].set_ylabel('y (μm)')
    fig.colorbar(im0, ax=axs[0], label='Intensity')
    
    # 相位分布
    im1 = axs[1].imshow(phase, 
                       extent=extent,
                       cmap=phase_cmap, origin='lower', vmin=-np.pi, vmax=np.pi)
    axs[1].set_title('Phase Distribution')
    axs[1].set_xlabel('x (μm)')
    axs[1].set_ylabel('y (μm)')
    fig.colorbar(im1, ax=axs[1], label='Phase (rad)')
    
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

if __name__ == "__main__":
    # 参数设置
    size_x = 801  # x方向像素数
    size_y = 801  # y方向像素数
    radius_x = 22.4  # x方向1/e²强度半径 (微米)
    radius_y = 40  # y方向1/e²强度半径 (微米)
    wavelength = 0.8  # 波长 (微米)
    grid_size = 0.22   # 网格物理尺寸 (微米/像素)
    z = -1200.0         # 离焦距离 (微米)，z=0表示束腰位置
    
    # 生成椭圆高斯光束
    intensity, phase, wx, wy = generate_elliptical_gaussian_matrix(
        size_x, size_y, radius_x, radius_y, grid_size, wavelength, z
    )
    
    # 计算实际1/e²半径
    calc_radius_x, calc_radius_y, angle = calculate_beam_radii(intensity, grid_size)
    
    # 打印信息
    print(f"目标半径 (x方向): {radius_x:.2f} μm, 计算半径: {calc_radius_x:.2f} μm")
    print(f"目标半径 (y方向): {radius_y:.2f} μm, 计算半径: {calc_radius_y:.2f} μm")
    print(f"离焦位置光束半径 (x方向): {wx:.2f} μm")
    print(f"离焦位置光束半径 (y方向): {wy:.2f} μm")
    print(f"主轴角度: {angle:.2f} °")

    # # 保存矩阵到CSV文件
    # save_matrix_to_csv(intensity, 'elliptical_gaussian_intensity.csv', folder=r'F:\paper\GS_new\asym_laser_import')
    # save_matrix_to_csv(phase, 'elliptical_gaussian_phase.csv', folder=r'F:\paper\GS_new\asym_laser_import')
    # print("强度和相位分布已保存为CSV文件")
    
    # 绘制结果
    plot_title = f"Elliptical Gaussian Beam\nRx={radius_x:.1f}μm, Ry={radius_y:.1f}μm, λ={wavelength}μm, z={z}μm"
    plot_elliptical_beam(intensity, phase, grid_size, title=plot_title)
    
    # 添加剖面图分析
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # x方向剖面 (y=0)
    center_y = intensity.shape[0] // 2
    x_profile = intensity[center_y, :]
    x = np.linspace(-size_x//2, size_x//2, size_x) * grid_size
    axs[0].plot(x, x_profile, 'b-', label='Intensity')
    axs[0].axvline(x=-calc_radius_x, color='r', linestyle='--', label='1/e² radius')
    axs[0].axvline(x=calc_radius_x, color='r', linestyle='--')
    axs[0].set_title('Intensity Profile along x-axis (y=0)')
    axs[0].set_xlabel('x (μm)')
    axs[0].set_ylabel('Intensity')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # y方向剖面 (x=0)
    center_x = intensity.shape[1] // 2
    y_profile = intensity[:, center_x]
    y = np.linspace(-size_y//2, size_y//2, size_y) * grid_size
    axs[1].plot(y, y_profile, 'b-', label='Intensity')
    # 修正：axvline 需要 x 参数来指定位置
    axs[1].axvline(x=-calc_radius_y, color='r', linestyle='--', label='1/e² radius')
    axs[1].axvline(x=calc_radius_y, color='r', linestyle='--')
    axs[1].set_title('Intensity Profile along y-axis (x=0)')
    axs[1].set_xlabel('y (μm)')
    axs[1].set_ylabel('Intensity')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()