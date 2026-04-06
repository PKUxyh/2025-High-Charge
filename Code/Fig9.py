##########这个程序用于分析GS算法得到的相位的zernike系数，用于特定研究某些zernike项的影响。
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.special import jacobi
import pandas as pd
from scipy.special import comb
from skimage.restoration import unwrap_phase
from scipy.special import factorial
from ModuleZernike import (
    zernike_polynomial, 
    get_zernike_names,
    get_zernike_names_nm,
    crop_center_square,
    zernike_coefficients,
    reconstruct_phase
)
from ModuleDiffractionRayleighSommerfeldIntegral import RayleighSommerfeldPropagator as RSI
from scipy.ndimage import gaussian_filter, gaussian_filter1d

# 设置全局样式
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'lines.linewidth': 1,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'figure.dpi': 300,
    'figure.figsize': (6.74, 7),  # 调整为更合适的尺寸用于两行一列
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.frameon': False,
    'legend.fancybox': False,
    'legend.edgecolor': 'black',
    'legend.framealpha': 1,
    'legend.fontsize': 12,
})
# # #双色
rgb1 = '#3F77A3'
rgb2 = '#E49A5C'
rgb3 = '#EC3232'
rgb4 = '#5CE49A'

def process_phase_data(E_path, size=60):
    """
    处理单个文件的相位数据
    """
    Elec = read_csv_files(E_path)
    if E_path == r'F:\paper\GS_new\GSA-MD\paper_used\后处理\after_zernike_far_field_E.csv':
        phase = np.angle(Elec)  # 取相位部分
    else:
        phase = -np.angle(Elec)  # 取相位部分

    # 相位解缠
    phase = unwrap_phase(phase)
    
    # 计算 phase 中心的值
    center_value = phase[phase.shape[0] // 2, phase.shape[1] // 2]
    phase = phase - center_value  # 去除中心值

    # 保留中心部分phase数据
    phase = crop_center_square(phase, size=size)

    # 只保留 phase 中心一个圆以内的数据，其余变成 0
    rows, cols = phase.shape
    center_row, center_col = rows // 2, cols // 2
    radius = rows // 2
    y, x = np.ogrid[:rows, :cols]
    mask = (x - center_col)**2 + (y - center_row)**2 <= radius**2
    phase[~mask] = 0
    radius = rows // 2
    phase = phase[center_col - radius:center_col + radius, center_row - radius:center_row + radius]
    
    return phase

def plot_phase(phase, title):
    """
    绘制相位图
    """
    plt.figure()
    plt.imshow(phase, cmap='bwr')
    plt.colorbar(label='Phase (rad)')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def plot_zernike_comparison(ax, coeffs_list, labels, max_coeff=10):
    """
    在指定的axes上绘制多个数据集的Zernike系数比较（折线图）
    """
    # 设置Zernike系数名称
    zernike_names = get_zernike_names(max_coeff)
    Z_names = get_zernike_names_nm(max_coeff)
    indices = np.arange(len(zernike_names))
    
    # 定义颜色列表
    colors = [rgb1, rgb2, rgb3, rgb4]
    
    # 绘制每个数据集的折线图
    for i, (coeffs, label) in enumerate(zip(coeffs_list, labels)):
        coeffs_subset = coeffs[:max_coeff]
        color = colors[i % len(colors)]

        # 绘制折线图，带标记点
        line = ax.plot(indices, coeffs_subset, 'o-', 
                      label=label, color=color, 
                      markersize=6, linewidth=1.5,
                      markerfacecolor='none',
                      markeredgecolor=color,
                      markeredgewidth=1.5)
    
    # 设置图表属性
    ax.set_xlabel('Aberrations')
    ax.set_ylabel('Coefficients [rad]')
    ax.set_xticks(indices)
    ax.set_xticklabels(Z_names, ha='center')

    # 添加网格线
    ax.grid(True, axis='y', alpha=0.6, linestyle='--')
    ax.grid(True, axis='x', alpha=0.6, linestyle='--')

    # 创建第二个x轴（顶部）
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())  # 确保两个x轴范围一致
    ax2.set_xticks(indices)
    ax2.set_xticklabels(zernike_names, ha='center', rotation=20)
    
    # 调整两个x轴标签的位置，避免重叠
    ax.xaxis.set_label_coords(0.5, -0.15)  # 底部x轴标签位置
    ax2.xaxis.set_label_coords(0.5, 1.15)  # 顶部x轴标签位置
    
    # 设置y轴范围，留出一些空间
    all_coeffs = np.concatenate([coeffs[:max_coeff] for coeffs in coeffs_list])
    y_max = np.max(all_coeffs)
    y_min = np.min(all_coeffs)
    y_range = y_max - y_min
    margin = 0.1 * y_range if y_range > 0 else 0.1
    ax.set_ylim(y_min - margin, y_max + margin)
    
    # # 添加图例
    # ax.legend(frameon=False, loc='upper right')
    
    # 添加(a)标签
    ax.text(0.04, 0.96, 'a', transform=ax.transAxes, fontsize=14, 
            fontweight='bold', va='top')

def read_csv_files(path):
    df = pd.read_csv(path)
    data = df.values
    data = data[:, :-1]
    data = data.astype(np.complex128)  # 如果是复数
    return data

def read_csv_files_complex(path, type='complex'):
    df = pd.read_csv(path)
    data = df.values
    data = data[:, :-1]
    if type == 'complex':
        data = data.astype(np.complex128)  # 如果是复数
    else:
        data = data.astype(np.float64)  # 如果是实数
    return data

def read_txt_files(path):
    data = np.loadtxt(path)
    return data

def crop_center_square(phase, size=100):
    """
    以相位矩阵为中心截取一个正方形区域
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

def find_laser_center(image_array):
    """
    找到激光光斑中心
    """
    # 对图像进行高斯模糊，减少噪声
    blurred = gaussian_filter(image_array, sigma=5)

    # 使用阈值化方法将图像二值化
    threshold = np.max(blurred) * 0.9
    binary = blurred > threshold

    # 计算阈值内像素的加权中心（强度加权质心）
    mask = blurred > threshold
    if np.any(mask):
        y_indices, x_indices = np.nonzero(mask)
        weights = blurred[mask]
        cx = np.sum(x_indices * weights) / np.sum(weights)
        cy = np.sum(y_indices * weights) / np.sum(weights)
    else:
        # 如果没有像素超过阈值，返回全局最大值位置
        cy, cx = np.unravel_index(np.argmax(blurred), blurred.shape)

    return (cx, cy)

def plot_deflection_comparison(ax, z, BeforeDeflectionX, BeforeDeflectionY, 
                              AfterDeflectionX, AfterDeflectionY, 
                              zOsiris, OsirisX, OsirisY):
    """
    在指定的axes上绘制偏转比较图
    """
    line1 = ax.plot(z*1e3, BeforeDeflectionX*1e6, '-', 
                    label=r'x: AS, original $Z_1^{-1}$, $Z_1^{1}$', color=rgb1,
                    linewidth=1.5)
    line2 = ax.plot(z*1e3, BeforeDeflectionY*1e6, '--', 
                    label=r'y: AS, original $Z_1^{-1}$, $Z_1^{1}$', color=rgb1,
                    linewidth=1.5)
    line3 = ax.plot(z*1e3, AfterDeflectionX*1e6, '-', 
                    label=r'x: AS, modified $Z_1^{-1}$, $Z_1^{1}$', color=rgb2,
                    linewidth=1.5)
    line4 = ax.plot(z*1e3, AfterDeflectionY*1e6, '--', 
                    label=r'y: AS, modified $Z_1^{-1}$, $Z_1^{1}$', color=rgb2,
                    linewidth=1.5)
    line5 = ax.plot(zOsiris, OsirisX, '-', 
                    label=r'x: OSIRIS, modified $Z_1^{-1}$, $Z_1^{1}$', color=rgb3,
                    linewidth=1.5)
    line6 = ax.plot(zOsiris, OsirisY, '--', 
                    label=r'y: OSIRIS, modified $Z_1^{-1}$, $Z_1^{1}$', color=rgb3,
                    linewidth=1.5)

    # 设置图表属性
    ax.set_xlabel('z [mm]')
    ax.set_ylabel(r'laser center offset [$\mathrm{\mu}$m]')
    ax.legend(loc='lower left')
    ax.set_ylim([-55,15])

    ax.grid(True, axis='y', alpha=0.6, linestyle='--')
    ax.grid(True, axis='x', alpha=0.6, linestyle='--')
    
    # 添加(b)标签
    ax.text(0.04, 0.96, 'b', transform=ax.transAxes, fontsize=14, 
            fontweight='bold', va='top')

def calculate_seidel_aberrations(zernike_coeffs):
    """
    从Zernike系数计算Seidel像差对应的光程差
    """
    Z = zernike_coeffs
    seidel_aberrations = {}
    
    # 1. 倾斜 (Tilt)
    seidel_aberrations['tilt_x'] = Z[2] - 2 * Z[8]
    seidel_aberrations['tilt_y'] = Z[1] - 2 * Z[7]
    seidel_aberrations['tilt_magnitude'] = np.sqrt(seidel_aberrations['tilt_x']**2 + seidel_aberrations['tilt_y']**2)
    seidel_aberrations['tilt_angle'] = np.arctan2(seidel_aberrations['tilt_y'], seidel_aberrations['tilt_x'])
    
    # 2. 离焦 (Defocus)
    astig_mag = np.sqrt(Z[3]**2 + Z[5]**2)
    focus1 = 2*Z[4] - 6*Z[12] + astig_mag
    focus2 = 2*Z[4] - 6*Z[12] - astig_mag
    seidel_aberrations['defocus'] = focus1 if abs(focus1) < abs(focus2) else focus2
    
    # 3. 像散 (Astigmatism)
    seidel_aberrations['astigmatism_magnitude'] = 2 * astig_mag
    seidel_aberrations['astigmatism_angle'] = 0.5 * np.arctan2(Z[3], Z[5])
    
    # 4. 彗差 (Coma)
    seidel_aberrations['coma_x'] = 3 * Z[8]
    seidel_aberrations['coma_y'] = 3 * Z[7]
    seidel_aberrations['coma_magnitude'] = 3 * np.sqrt(Z[8]**2 + Z[7]**2)
    seidel_aberrations['coma_angle'] = np.arctan2(Z[7], Z[8])
    
    # 5. 球差 (Spherical)
    seidel_aberrations['spherical'] = 6 * Z[12]
    
    return seidel_aberrations

if __name__ == "__main__":
    # 第一部分：Zernike系数分析
    file_paths = [
        r'F:\paper\GS_new\GSA-MD\paper_used\E_far-二次优化.csv',
        # r'F:\paper\GS_new\GSA-MD\paper_used\后处理\after_zernike_far_field_E.csv'
    ]

    labels = [
        'before correction',
        # 'after correction'
    ]

    # 处理每个文件并计算Zernike系数
    all_coeffs = []
    all_phases = []

    for i, (file_path, label) in enumerate(zip(file_paths, labels)):
        print(f"处理文件: {label}")

        # 处理相位数据
        if file_path == r'F:\paper\GS_new\asym_laser_import\real_astigmatic_ellipicity_phase.csv':
            size = 240
        else:
            size = 199
        phase = process_phase_data(file_path, size=size)
        all_phases.append(phase)

        # 计算Zernike系数
        n_max = 20
        coeffs = zernike_coefficients(phase, n_max)
        all_coeffs.append(coeffs)

        # 打印前15项Zernike系数
        zernike_names = get_zernike_names(15)
        print(f"\n{label} Zernike项及其系数 (单位: rad):")
        for j, c in enumerate(coeffs[:len(zernike_names)]):
            print(f"  {zernike_names[j]:<25}: {c:.4f} rad")
        print("-" * 50)

    # 第二部分：光束偏转分析
    BeforeFile = r'F:\paper\GS_new\GSA-MD\paper_used\E_far-二次优化.csv'
    AfterFile = r'F:\paper\GS_new\GSA-MD\paper_used\后处理\after_zernike_far_field_E.csv'
    XOsirisFile = r'F:\code\paper1\fig9\center_x.txt'
    YOsirisFile = r'F:\code\paper1\fig9\center_y.txt'

    BeforeE = read_csv_files_complex(BeforeFile)
    AfterE = read_csv_files_complex(AfterFile)
    OsirisX = read_txt_files(XOsirisFile)
    OsirisY = read_txt_files(YOsirisFile)
    GridSize = 0.88e-6
    BeforeN = np.shape(BeforeE)
    AfterN = np.shape(AfterE)
    wavelength = 800e-9

    z = np.arange(-1200e-6,3000e-6,10e-6)

    BeforeDeflectionX = np.zeros(len(z))
    BeforeDeflectionY = np.zeros(len(z))
    AfterDeflectionX = np.zeros(len(z))
    AfterDeflectionY = np.zeros(len(z))
    
    print("zernike修正前: \n")
    propagator = RSI(wavelength, BeforeN, GridSize)
    BeforeE, _ = propagator.zero_padding(BeforeE)
    for i in range(len(z)):
        BeforeE1 = propagator.propagate_integral(
            BeforeE, z[i], None, None, method='fft'
        )
        cx, cy = find_laser_center(np.abs(BeforeE1)**2)
        BeforeDeflectionX[i] = (cx - BeforeN[0] / 2) * GridSize
        BeforeDeflectionY[i] = (cy - BeforeN[0] / 2) * GridSize

    print("zernike修正后: \n")
    propagator = RSI(wavelength, AfterN, GridSize)
    AfterE, _ = propagator.zero_padding(AfterE)
    for i in range(len(z)):
        AfterE1 = propagator.propagate_integral(
            AfterE, z[i], None, None, method='fft'
        )
        cx, cy = find_laser_center(np.abs(AfterE1)**2)
        AfterDeflectionX[i] = (cx - AfterN[0] / 2) * GridSize
        AfterDeflectionY[i] = (cy - AfterN[0] / 2) * GridSize

    zOsiris = np.linspace(-2.2, 2.2, len(OsirisX))
    z = z - 1e-3

    BeforeDeflectionX = BeforeDeflectionX - BeforeDeflectionX[0]
    BeforeDeflectionY = BeforeDeflectionY - BeforeDeflectionY[0]
    AfterDeflectionX = AfterDeflectionX - AfterDeflectionX[0]
    AfterDeflectionY = AfterDeflectionY - AfterDeflectionY[0]
    OsirisX = OsirisX - OsirisX[0]
    OsirisY = OsirisY - OsirisY[0]
    OsirisX = gaussian_filter1d(OsirisX, sigma=1)
    OsirisY = gaussian_filter1d(OsirisY, sigma=1)

    # 创建两行一列的图形
    fig, (ax1, ax2) = plt.subplots(2, 1)
    
    # 在上面的axes绘制Zernike系数比较
    plot_zernike_comparison(ax1, all_coeffs, labels, max_coeff=10)
    
    # 在下面的axes绘制偏转比较
    plot_deflection_comparison(ax2, z, BeforeDeflectionX, BeforeDeflectionY, 
                              AfterDeflectionX, AfterDeflectionY, 
                              zOsiris, OsirisX, OsirisY)

    plt.tight_layout()
    plt.savefig(r'F:\code\paper1\fig9\combined_figure.png', dpi=300, bbox_inches='tight')
    plt.show()