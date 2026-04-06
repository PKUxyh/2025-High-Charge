import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from propagator_bymyself import angular_spectrum
from scipy.ndimage import zoom

# 设置全局样式
plt.style.use('default')  # 重置为默认样式
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'lines.linewidth': 1.5,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'figure.dpi': 300,
    'figure.figsize': (3.37,3.37/0.618),  # 图像大小
    'xtick.direction': 'in',         # 刻度线朝内
    'ytick.direction': 'in',
    'xtick.major.width': 1.2,        # 主刻度线粗细
    'ytick.major.width': 1.2,
    'xtick.major.size': 6,           # 主刻度线长度
    'ytick.major.size': 6,
    'xtick.labelsize': 8,           # 刻度数字大小
    'ytick.labelsize': 8,
})



# 双色 
# rgb1 = '#3F77A3'
# rgb2 = '#E49A5C'

# #三色 
rgb1 = '#480080'
rgb2 = '#e23c5d'
rgb3 = '#ffb42c'

# rgb4 = '#501d8a'
# rgb5 = '#aa3474'
# rgb6 = '#ee8c7d'

# #四色
# rgb1 = '#A82E25'
# rgb2 = '#eb7e35'
# rgb3 = '#6c8735'
# rgb4 = '#505050'


def read_csv_files(path):
    df = pd.read_csv(path)
    data = df.values
    data = data[:, :-1]
    data = data.astype(np.complex128)  # 如果是复数
    # 或 data = data.astype(np.float64)  # 如果是实数
    return data

def fig1():
    real_focal_path = r"E:\Vscode_Py\optics\optics\GS_new\GSA-MD\paper_used\far_recon_abs.csv"
    real_near_path = r"E:\Vscode_Py\optics\optics\GS_new\GSA-MD\paper_used\near_recon_abs.csv"
    focal_spot_path = r"E:\Vscode_Py\optics\optics\GS_new\GSA-MD\paper_used\E_far-二次优化.csv"
    grid_size = 0.88  # um
    u1 = read_csv_files(focal_spot_path)
    real_focal = read_csv_files(real_focal_path)
    real_focal = np.abs(real_focal)  # 确保是实数
    real_near = read_csv_files(real_near_path)
    real_near = np.abs(real_near)  # 确保是实数
    
    grid_num = np.shape(u1)[0]
    u2_region = angular_spectrum(u1=u1, L=grid_num*grid_size, lamda=0.8, z=1000)
    focal_region = np.abs(u1)**2
    plus_1000um_region = np.abs(u2_region)**2

    # 归一化
    real_focal_norm = real_focal / np.max(real_focal)
    real_near_norm = real_near / np.max(real_near)
    focal_region_norm = focal_region / np.max(focal_region)
    plus_1000um_region_norm = plus_1000um_region / np.max(plus_1000um_region)

    # 使用gridspec创建非均匀列宽
    fig = plt.figure(figsize=(3.37, 3.37*2/3))
    gs = plt.GridSpec(2, 4, figure=fig, width_ratios=[0.5, 1, 1, 0.5], 
                      height_ratios=[1, 1], wspace=0.05, hspace=0.05)

    # 统一extent
    extent = [
        -real_focal.shape[1]//2 * grid_size, real_focal.shape[1]//2 * grid_size,
        -real_focal.shape[0]//2 * grid_size, real_focal.shape[0]//2 * grid_size
    ]

    # 创建子图 - 使用中间的两列
    ax1 = fig.add_subplot(gs[0, 1])  # 第一行，第二列 (a)
    ax2 = fig.add_subplot(gs[0, 2])  # 第一行，第三列 (b)
    ax3 = fig.add_subplot(gs[1, 1])  # 第二行，第二列 (c)
    ax4 = fig.add_subplot(gs[1, 2])  # 第二行，第三列 (d)

    axs = [[ax1, ax2], [ax3, ax4]]

    # 左上：real_focal (a)
    im1 = axs[0][0].imshow(real_focal_norm, cmap='jet', extent=extent, vmin=0, vmax=1)
    axs[0][0].text(0.05, 0.95, 'a', transform=axs[0][0].transAxes, fontweight='bold', 
                   fontsize=12, color='white', va='top', ha='left')

    # 绘制比例尺（50um，白色线，放在左下角）
    scalebar_len = 50  # um
    x0 = extent[0] + 20  # 距左边20um
    y0 = extent[2] + 10  # 距下边10um
    axs[0][0].plot([x0, x0 + scalebar_len], [y0, y0], color='white', lw=1.5, solid_capstyle='butt')
    axs[0][0].text(x0 + scalebar_len/2, y0 + 5, '50 μm', color='white', ha='center', 
                   va='bottom', fontsize=8, fontweight='bold')

    # 右上：real_near (b)
    im2 = axs[0][1].imshow(real_near_norm, cmap='jet', extent=extent, vmin=0, vmax=1)
    axs[0][1].text(0.05, 0.95, 'b', transform=axs[0][1].transAxes, fontweight='bold', 
                   fontsize=12, color='white', va='top', ha='left')

    # 左下：focal_region (c)
    im3 = axs[1][0].imshow(focal_region_norm, cmap='jet', extent=extent, vmin=0, vmax=1)
    axs[1][0].text(0.05, 0.95, 'c', transform=axs[1][0].transAxes, fontweight='bold', 
                   fontsize=12, color='white', va='top', ha='left')

    # 右下：plus_1000um_region (d)
    im4 = axs[1][1].imshow(plus_1000um_region_norm, cmap='jet', extent=extent, vmin=0, vmax=1)
    axs[1][1].text(0.05, 0.95, 'd', transform=axs[1][1].transAxes, fontweight='bold', 
                   fontsize=12, color='white', va='top', ha='left')

    # 完全关闭所有子图的坐标轴
    for i in range(2):
        for j in range(2):
            axs[i][j].axis('off')  # 完全关闭坐标轴

    # 只为需要标签的子图重新启用坐标轴并设置标签
    axs[1][0].axis('on')
    axs[1][0].set_xticks([])
    axs[1][0].set_yticks([])
    axs[1][0].set_xlabel('x [μm]')
    axs[1][0].set_ylabel('y [μm]')
    
    axs[1][1].axis('on')
    axs[1][1].set_xticks([])
    axs[1][1].set_yticks([])
    axs[1][1].set_xlabel('x [μm]')
    
    axs[0][0].axis('on')
    axs[0][0].set_xticks([])
    axs[0][0].set_yticks([])
    axs[0][0].set_ylabel('y [μm]')

    # 统一colorbar，放在右侧，调整高度与子图匹配
    cax = fig.add_axes([0.78, 0.16, 0.04, 0.74])  # 调整位置和高度
    cb = fig.colorbar(im1, cax=cax, orientation='vertical')
    cb.set_label('Normalized Intensity')

    # 极致紧密的布局
    plt.subplots_adjust(left=0.10, right=0.84, bottom=0.15, top=0.90, wspace=0.05, hspace=0.05)
    plt.show()
    
    # 保存图片到本地
    # fig.savefig(r"F:\paper\GS_new\存储生成的各种txt场\appendix_fig\fig1new.png", dpi=300, bbox_inches='tight')

    # 计算PSNR和SSIM（保持原有的计算部分不变）
    focal_region_centered = center_max_value(focal_region_norm)
    real_focal_centered = center_max_value(real_focal_norm)
    plus_1000um_region_centered = center_max_value(plus_1000um_region_norm)
    real_near_centered = center_max_value(real_near_norm)

    # 计算焦平面的PSNR和SSIM
    mse_focal = np.mean((focal_region_centered - real_focal_centered) ** 2)
    psnr_focal = 10 * np.log10(1.0 / mse_focal)
    ssim_focal = ssim(focal_region_centered, real_focal_centered, data_range=1.0)
    print(f"Focal Plane (Centered) - PSNR: {psnr_focal:.2f} dB, SSIM: {ssim_focal:.4f}")
    
    # 计算near平面的PSNR和SSIM
    mse_near = np.mean((plus_1000um_region_centered - real_near_centered) ** 2)
    psnr_near = 10 * np.log10(1.0 / mse_near)
    ssim_near = ssim(plus_1000um_region_centered, real_near_centered, data_range=1.0)
    print(f"Near Plane (Centered) - PSNR: {psnr_near:.2f} dB, SSIM: {ssim_near:.4f}")

    # 单独保存远场真实数据为图片（保持不变）
    plt.figure(figsize=(2, 2))
    plt.imshow(real_focal_norm, cmap='jet', extent=extent, vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    # 绘制比例尺
    x0 = extent[0] + 20
    y0 = extent[2] + 10
    plt.plot([x0, x0 + scalebar_len], [y0, y0], color='white', lw=1.5, solid_capstyle='butt')
    plt.text(x0 + scalebar_len/2, y0 + 5, '50 μm', color='white', ha='center', va='bottom', 
             fontsize=8, fontweight='bold')
    plt.savefig(r"F:\paper\GS_new\存储生成的各种txt场\appendix_fig\real_focal_only.png", 
                dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def center_max_value(data):
    """
    将数据的最大值移到中心位置，保持原矩阵大小，缺失部分补0
    """
    # 找到最大值的位置
    max_pos = np.unravel_index(np.argmax(data), data.shape)
    
    # 计算中心位置
    center_pos = (data.shape[0] // 2, data.shape[1] // 2)
    
    # 计算平移量
    shift_y = center_pos[0] - max_pos[0]
    shift_x = center_pos[1] - max_pos[1]
    
    # 创建新的数组，用0填充
    centered_data = np.zeros_like(data)
    
    # 计算有效区域的边界
    src_y_start = max(0, -shift_y)
    src_y_end = min(data.shape[0], data.shape[0] - shift_y)
    src_x_start = max(0, -shift_x)
    src_x_end = min(data.shape[1], data.shape[1] - shift_x)
    
    dst_y_start = max(0, shift_y)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = max(0, shift_x)
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    
    # 复制数据到新位置
    centered_data[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        data[src_y_start:src_y_end, src_x_start:src_x_end]
    
    return centered_data


def fig2():
    # 读取 CSV 文件
    focal_spot_path = r"F:\paper\GS_new\存储生成的各种txt场\appendix_fig\gauss\far.csv"
    plus_1000um_path = r"F:\paper\GS_new\存储生成的各种txt场\appendix_fig\gauss\near.csv"
    phase_path = r"F:\paper\GS_new\存储生成的各种txt场\appendix_fig\gauss\far_phase.csv"
    gs_phase_path = r"F:\paper\GS_new\存储生成的各种txt场\appendix_fig\gauss\gs_far_phase.csv"
    diffraction_path = r"F:\paper\GS_new\存储生成的各种txt场\appendix_fig\gauss\diffraction_near_use_gs_phase.csv"

    grid_size = 0.44  # 网格大小（单位：um）

    focal_spot_data = read_csv_files(focal_spot_path)
    plus_1000um_data = read_csv_files(plus_1000um_path)
    phase_data = read_csv_files(phase_path)
    gs_phase = read_csv_files(gs_phase_path)
    diffraction_data = read_csv_files(diffraction_path)


    def extract_region_around_max(data, region_size=100):
        """从数据中提取最大值附近的区域"""
        max_idx = np.unravel_index(np.argmax(data), data.shape)
        half_size = region_size // 2
        
        # 计算区域边界
        x_min = max(0, max_idx[0] - half_size)
        x_max = min(data.shape[0], max_idx[0] + half_size)
        y_min = max(0, max_idx[1] - half_size)
        y_max = min(data.shape[1], max_idx[1] + half_size)
        
        return data[x_min:x_max, y_min:y_max], (x_min, y_min)

    # 设置要提取的区域大小（像素）
    region_size = 300  # 可以根据需要调整

    # 提取焦点数据最大值附近的区域
    focal_region, focal_offset = extract_region_around_max(focal_spot_data, region_size)
    plus_1000um_region, plus_offset = extract_region_around_max(plus_1000um_data, region_size)
    diffraction_region, diffraction_offset = extract_region_around_max(diffraction_data, region_size)

    # 提取相位数据中心位置附近的区域
    center_idx = (phase_data.shape[0] // 2, phase_data.shape[1] // 2)
    half_size = region_size // 2
    # 计算区域边界
    x_min = max(0, center_idx[0] - half_size)
    x_max = min(phase_data.shape[0], center_idx[0] + half_size)
    y_min = max(0, center_idx[1] - half_size)
    y_max = min(phase_data.shape[1], center_idx[1] + half_size)
    phase_region = phase_data[x_min:x_max, y_min:y_max]
    gs_phase_region = gs_phase[x_min:x_max, y_min:y_max]



    # 创建2×2的绘图区域
    fig, axs = plt.subplots(2, 2)

    # 绘制第一个伪彩图（左上）
    im1 = axs[0, 0].imshow(
        focal_region, cmap='jet',
        extent=[
            -focal_region.shape[1]//2 * grid_size, focal_region.shape[1]//2 * grid_size,
            -focal_region.shape[0]//2 * grid_size, focal_region.shape[0]//2 * grid_size
        ]
    )
    axs[0, 0].set_xlabel('X (μm)')
    axs[0, 0].set_ylabel('Y (μm)')
    fig.colorbar(im1, ax=axs[0, 0], label='Intensity [counts]')
    axs[0, 0].text(0.09, 0.2, 'a', transform=axs[0, 0].transAxes, fontweight='bold', fontsize=16, color='white', va='top', ha='left')

    # 绘制第二个伪彩图（右上）
    im2 = axs[0, 1].imshow(
        plus_1000um_region, cmap='jet',
        extent=[
            -plus_1000um_region.shape[1]//2 * grid_size, plus_1000um_region.shape[1]//2 * grid_size,
            -plus_1000um_region.shape[0]//2 * grid_size, plus_1000um_region.shape[0]//2 * grid_size
        ]
    )
    axs[0, 1].set_xlabel('X (μm)')
    axs[0, 1].set_ylabel('Y (μm)')
    fig.colorbar(im2, ax=axs[0, 1], label='Intensity [counts]')
    axs[0, 1].text(0.09, 0.2, 'b', transform=axs[0, 1].transAxes, fontweight='bold', fontsize=16, color='white', va='top', ha='left')

    # 绘制第三个伪彩图（左下）
    im3 = axs[1, 0].imshow(
        phase_region, cmap='jet',
        extent=[
            -phase_region.shape[1]//2 * grid_size, phase_region.shape[1]//2 * grid_size,
            -phase_region.shape[0]//2 * grid_size, phase_region.shape[0]//2 * grid_size
        ]
    )
    axs[1, 0].set_xlabel('X (μm)')
    axs[1, 0].set_ylabel('Y (μm)')
    fig.colorbar(im3, ax=axs[1, 0], label='Phase [rad]')
    axs[1, 0].text(0.09, 0.2, 'c', transform=axs[1, 0].transAxes, fontweight='bold', fontsize=16, color='white', va='top', ha='left')

    # 绘制第四个伪彩图（右下）
    gs_phase_region = gs_phase_region - gs_phase_region[gs_phase_region.shape[0]//2, gs_phase_region.shape[1]//2]
    im4 = axs[1, 1].imshow(
        gs_phase_region, cmap='jet',
        extent=[
            -gs_phase_region.shape[1]//2 * grid_size, gs_phase_region.shape[1]//2 * grid_size,
            -gs_phase_region.shape[0]//2 * grid_size, gs_phase_region.shape[0]//2 * grid_size
        ]
    )
    axs[1, 1].set_xlabel('X (μm)')
    axs[1, 1].set_ylabel('Y (μm)')
    fig.colorbar(im4, ax=axs[1, 1], label='Phase [rad]')
    im4.set_clim(-0.1, 0.1)
    axs[1, 1].text(0.09, 0.2, 'd', transform=axs[1, 1].transAxes, fontweight='bold', fontsize=16, color='white', va='top', ha='left')

    plt.tight_layout()
    plt.show()

    # 对 u2_region 和 plus_1000um_region 进行归一化
    diffraction_region = (diffraction_region - diffraction_region.min()) / (diffraction_region.max() - diffraction_region.min())
    plus_1000um_region = (plus_1000um_region - plus_1000um_region.min()) / (plus_1000um_region.max() - plus_1000um_region.min())
    
    # 扩大到600x600的尺寸
    def expand_to_center(data, target_size=600):
        expanded = np.zeros((target_size, target_size))
        start_x = (target_size - data.shape[0]) // 2
        start_y = (target_size - data.shape[1]) // 2
        expanded[start_x:start_x + data.shape[0], start_y:start_y + data.shape[1]] = data
        return expanded

    diffraction_region = expand_to_center(diffraction_region, target_size=800)
    plus_1000um_region = expand_to_center(plus_1000um_region, target_size=800)

    # 计算峰值信噪比（Peak Signal-to-Noise Ratio, PSNR）
    mse = np.mean((diffraction_region - plus_1000um_region) ** 2)
    psnr = 10 * np.log10((plus_1000um_region.max() ** 2) / mse)

    # 计算结构相似性指数（SSIM）
    ssim_value = ssim(diffraction_region, plus_1000um_region, data_range=plus_1000um_region.max() - plus_1000um_region.min())

    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr} dB")
    print(f"Structural Similarity Index (SSIM): {ssim_value}")






def fig3():
    # 读取 CSV 文件
    focal_spot_path = r"F:\实验\实验集\20250226\moniguang_0um__0001.ascii.csv"
    plus_1000um_path = r"F:\实验\实验集\20250226\moniguang_-150um__0001.ascii.csv"
    phase_path = r"F:\paper\GS_new\存储生成的各种txt场\appendix_fig\mng\phase_far_four_planes.csv"
    diffraction_path = r"F:\paper\GS_new\存储生成的各种txt场\appendix_fig\mng\-150um.csv"

    grid_size = 0.44  # 网格大小（单位：um）

    focal_spot_data = read_csv_files(focal_spot_path)
    plus_1000um_data = read_csv_files(plus_1000um_path)
    phase_data = read_csv_files(phase_path)
    diffraction_data = read_csv_files(diffraction_path)


    def extract_region_around_max(data, region_size=100):
        """从数据中提取最大值附近的区域"""
        max_idx = np.unravel_index(np.argmax(data), data.shape)
        half_size = region_size // 2
        
        # 计算区域边界
        x_min = max(0, max_idx[0] - half_size)
        x_max = min(data.shape[0], max_idx[0] + half_size)
        y_min = max(0, max_idx[1] - half_size)
        y_max = min(data.shape[1], max_idx[1] + half_size)
        
        return data[x_min:x_max, y_min:y_max], (x_min, y_min)

    # 设置要提取的区域大小（像素）
    region_size = 100  # 可以根据需要调整

    # 提取焦点数据最大值附近的区域
    focal_region, focal_offset = extract_region_around_max(focal_spot_data, region_size)
    plus_1000um_region, plus_offset = extract_region_around_max(plus_1000um_data, region_size)
    diffraction_region, diffraction_offset = extract_region_around_max(diffraction_data, region_size)

    # 提取相位数据中心位置附近的区域
    center_idx = (phase_data.shape[0] // 2, phase_data.shape[1] // 2)
    half_size = region_size // 2
    # 计算区域边界
    x_min = max(0, center_idx[0] - half_size)
    x_max = min(phase_data.shape[0], center_idx[0] + half_size)
    y_min = max(0, center_idx[1] - half_size)
    y_max = min(phase_data.shape[1], center_idx[1] + half_size)
    phase_region = phase_data[x_min:x_max, y_min:y_max]



    # 创建2×2的绘图区域
    fig, axs = plt.subplots(2, 2)

    # 绘制第一个伪彩图（左上）
    im1 = axs[0, 0].imshow(
        focal_region, cmap='jet',
        extent=[
            -focal_region.shape[1]//2 * grid_size, focal_region.shape[1]//2 * grid_size,
            -focal_region.shape[0]//2 * grid_size, focal_region.shape[0]//2 * grid_size
        ]
    )
    axs[0, 0].set_xlabel('X (μm)')
    axs[0, 0].set_ylabel('Y (μm)')
    fig.colorbar(im1, ax=axs[0, 0], label='Intensity [counts]')
    axs[0, 0].text(0.09, 0.2, 'a', transform=axs[0, 0].transAxes, fontweight='bold', fontsize=16, color='white', va='top', ha='left')

    # 绘制第二个伪彩图（右上）
    im2 = axs[0, 1].imshow(
        plus_1000um_region, cmap='jet',
        extent=[
            -plus_1000um_region.shape[1]//2 * grid_size, plus_1000um_region.shape[1]//2 * grid_size,
            -plus_1000um_region.shape[0]//2 * grid_size, plus_1000um_region.shape[0]//2 * grid_size
        ]
    )
    axs[0, 1].set_xlabel('X (μm)')
    axs[0, 1].set_ylabel('Y (μm)')
    fig.colorbar(im2, ax=axs[0, 1], label='Intensity [counts]')
    axs[0, 1].text(0.09, 0.2, 'b', transform=axs[0, 1].transAxes, fontweight='bold', fontsize=16, color='white', va='top', ha='left')

    # 绘制第三个伪彩图（左下）
    im3 = axs[1, 0].imshow(
        phase_region, cmap='jet',
        extent=[
            -phase_region.shape[1]//2 * grid_size, phase_region.shape[1]//2 * grid_size,
            -phase_region.shape[0]//2 * grid_size, phase_region.shape[0]//2 * grid_size
        ]
    )
    axs[1, 0].set_xlabel('X (μm)')
    axs[1, 0].set_ylabel('Y (μm)')
    fig.colorbar(im3, ax=axs[1, 0], label='Phase [rad]')
    axs[1, 0].text(0.09, 0.2, 'c', transform=axs[1, 0].transAxes, fontweight='bold', fontsize=16, color='white', va='top', ha='left')

    # 绘制第四个伪彩图（右下）
    im4 = axs[1, 1].imshow(
        diffraction_region, cmap='jet',
        extent=[
            -diffraction_region.shape[1]//2 * grid_size, diffraction_region.shape[1]//2 * grid_size,
            -diffraction_region.shape[0]//2 * grid_size, diffraction_region.shape[0]//2 * grid_size
        ]
    )
    axs[1, 1].set_xlabel('X (μm)')
    axs[1, 1].set_ylabel('Y (μm)')
    fig.colorbar(im4, ax=axs[1, 1], label='Intensity [counts]')
    axs[1, 1].text(0.09, 0.2, 'd', transform=axs[1, 1].transAxes, fontweight='bold', fontsize=16, color='white', va='top', ha='left')

    plt.tight_layout()
    plt.show()

    # 对 u2_region 和 plus_1000um_region 进行归一化
    diffraction_region = (diffraction_region - diffraction_region.min()) / (diffraction_region.max() - diffraction_region.min())
    plus_1000um_region = (plus_1000um_region - plus_1000um_region.min()) / (plus_1000um_region.max() - plus_1000um_region.min())
    
    # 扩大到600x600的尺寸
    def expand_to_center(data, target_size=600):
        expanded = np.zeros((target_size, target_size))
        start_x = (target_size - data.shape[0]) // 2
        start_y = (target_size - data.shape[1]) // 2
        expanded[start_x:start_x + data.shape[0], start_y:start_y + data.shape[1]] = data
        return expanded

    diffraction_region = expand_to_center(diffraction_region, target_size=800)
    plus_1000um_region = expand_to_center(plus_1000um_region, target_size=800)

    # 计算峰值信噪比（Peak Signal-to-Noise Ratio, PSNR）
    mse = np.mean((diffraction_region - plus_1000um_region) ** 2)
    psnr = 10 * np.log10((plus_1000um_region.max() ** 2) / mse)

    # 计算结构相似性指数（SSIM）
    ssim_value = ssim(diffraction_region, plus_1000um_region, data_range=plus_1000um_region.max() - plus_1000um_region.min())

    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr} dB")
    print(f"Structural Similarity Index (SSIM): {ssim_value}")





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





def fig4():
    # 文件路径（示例，需替换为实际路径）
    plane1_path = r"F:\paper\GS_new\GSA-MD\E_far_mng2.csv"  # 平面1数据
    real_plane1_path = r'F:\实验\实验集\20250226\moniguang_0um__0008.ascii.csv'
    real_plane2_path = r'F:\实验\实验集\20250226\moniguang_-100um__0008.ascii.csv'
    real_plane3_path = r'F:\实验\实验集\20250226\moniguang_-250um__0008.ascii.csv'

    # 参数设置
    grid_size = 0.44  # 像素尺寸 (um)
    scalebar_len = 50  # 比例尺长度 (um)
    grid_num = 201

    plane1 = read_csv_files(plane1_path)
    real_plane1 = read_csv_files(real_plane1_path)
    real_plane2 = read_csv_files(real_plane2_path)
    real_plane3 = read_csv_files(real_plane3_path)
    real_plane1 = np.abs(real_plane1)
    real_plane2 = np.abs(real_plane2)
    real_plane3 = np.abs(real_plane3)
    real_plane1 = crop_center(real_plane1, grid_num, grid_num)
    real_plane2 = crop_center(real_plane2, grid_num, grid_num)
    real_plane3 = crop_center(real_plane3, grid_num, grid_num)

    plane2 = angular_spectrum(u1=plane1, L=grid_num*grid_size, lamda=0.785, z=100)  # 模拟平面2
    plane3 = angular_spectrum(u1=plane1, L=grid_num*grid_size, lamda=0.785, z=250)  # 模拟平面3

    recon_plane1 = np.abs(plane1) ** 2
    recon_plane2 = np.abs(plane2) ** 2
    recon_plane3 = np.abs(plane3) ** 2

    # 各自归一化所有数据
    real_plane1_norm = real_plane1 / np.max(real_plane1)
    real_plane2_norm = real_plane2 / np.max(real_plane2)
    real_plane3_norm = real_plane3 / np.max(real_plane3)
    recon_plane1_norm = recon_plane1 / np.max(recon_plane1)
    recon_plane2_norm = recon_plane2 / np.max(recon_plane2)
    recon_plane3_norm = recon_plane3 / np.max(recon_plane3)

    # 创建两行三列的子图
    fig, axs = plt.subplots(2, 3, figsize=(3.37, 3.37*2/3))

    # 统一extent（坐标范围）
    extent = [
        -plane1.shape[1] // 2 * grid_size,
        plane1.shape[1] // 2 * grid_size,
        -plane1.shape[0] // 2 * grid_size,
        plane1.shape[0] // 2 * grid_size,
    ]

    # 第一行：重建平面数据
    im1 = axs[0, 0].imshow(real_plane1_norm, cmap="jet", extent=extent, vmin=0, vmax=1)
    axs[0, 0].text(0.05, 0.95, "a", transform=axs[0, 0].transAxes, fontweight="bold", fontsize=12, color="white", va="top", ha="left")
    
    im2 = axs[0, 1].imshow(real_plane2_norm, cmap="jet", extent=extent, vmin=0, vmax=1)
    axs[0, 1].text(0.05, 0.95, "b", transform=axs[0, 1].transAxes, fontweight="bold", fontsize=12, color="white", va="top", ha="left")
    
    im3 = axs[0, 2].imshow(real_plane3_norm, cmap="jet", extent=extent, vmin=0, vmax=1)
    axs[0, 2].text(0.05, 0.95, "c", transform=axs[0, 2].transAxes, fontweight="bold", fontsize=12, color="white", va="top", ha="left")

    # 第二行：真实平面数据
    im4 = axs[1, 0].imshow(recon_plane1_norm, cmap="jet", extent=extent, vmin=0, vmax=1)
    axs[1, 0].text(0.05, 0.95, "d", transform=axs[1, 0].transAxes, fontweight="bold", fontsize=12, color="white", va="top", ha="left")
    
    im5 = axs[1, 1].imshow(recon_plane2_norm, cmap="jet", extent=extent, vmin=0, vmax=1)
    axs[1, 1].text(0.05, 0.95, "e", transform=axs[1, 1].transAxes, fontweight="bold", fontsize=12, color="white", va="top", ha="left")
    
    im6 = axs[1, 2].imshow(recon_plane3_norm, cmap="jet", extent=extent, vmin=0, vmax=1)
    axs[1, 2].text(0.05, 0.95, "f", transform=axs[1, 2].transAxes, fontweight="bold", fontsize=12, color="white", va="top", ha="left")

    # 添加比例尺（在左上角子图）
    x0 = extent[0] + 20  # 距左边20um
    y0 = extent[2] + 10  # 距下边10um
    axs[0, 0].plot([x0, x0 + scalebar_len], [y0, y0], color="white", lw=1.5, solid_capstyle='butt')
    axs[0, 0].text(x0 + scalebar_len / 2, y0 + 5, f"{scalebar_len} μm", color="white", ha="center", va="bottom", fontsize=8, fontweight='bold')

    # 关闭所有子图的默认坐标轴
    for i in range(2):
        for j in range(3):
            axs[i, j].axis('off')

    # 只为需要标签的子图重新启用坐标轴并设置标签
    # 底部行添加x轴标签
    axs[1, 0].axis('on')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    axs[1, 0].set_xlabel('x [μm]')
    axs[1, 0].set_ylabel('y [μm]')
    
    axs[1, 1].axis('on')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    axs[1, 1].set_xlabel('x [μm]')
    
    axs[1, 2].axis('on')
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])
    axs[1, 2].set_xlabel('x [μm]')
    
    # 左侧列添加y轴标签
    axs[0, 0].axis('on')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    axs[0, 0].set_ylabel('y [μm]')

    # 统一colorbar，放在右侧，调整高度与子图匹配
    cax = fig.add_axes([0.92, 0.16, 0.04, 0.74])  # 调整位置和高度
    cb = fig.colorbar(im1, cax=cax, orientation='vertical')
    cb.set_label('Normalized Intensity')

    # 极致紧密的布局，完全移除标题空间
    plt.subplots_adjust(left=0.08, right=0.90, bottom=0.15, top=0.90, wspace=0.05, hspace=0.05)
    plt.show()
    
    # 保存图片到本地
    # fig.savefig(r"F:\paper\GS_new\存储生成的各种txt场\appendix_fig\fig2new.png", dpi=300, bbox_inches='tight')

    # 计算PSNR和SSIM对比（使用center_max_value处理平移）
    # 对数据进行居中处理
    recon_plane1_centered = center_max_value(recon_plane1_norm)
    recon_plane2_centered = center_max_value(recon_plane2_norm)
    recon_plane3_centered = center_max_value(recon_plane3_norm)
    real_plane1_centered = center_max_value(real_plane1_norm)
    real_plane2_centered = center_max_value(real_plane2_norm)
    real_plane3_centered = center_max_value(real_plane3_norm)


    # 对重构数据插值到真实数据大小

    def resize_to_shape(data, target_shape):
        zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
        return zoom(data, zoom_factors, order=1)

    recon_plane1_centered = resize_to_shape(recon_plane1_centered, real_plane1_centered.shape)
    recon_plane2_centered = resize_to_shape(recon_plane2_centered, real_plane2_centered.shape)
    recon_plane3_centered = resize_to_shape(recon_plane3_centered, real_plane3_centered.shape)
    # 计算各平面的PSNR和SSIM
    # Plane1对比
    mse_plane1 = np.mean((recon_plane1_centered - real_plane1_centered) ** 2)
    if mse_plane1 == 0:
        psnr_plane1 = float('inf')
    else:
        psnr_plane1 = 10 * np.log10(1.0 / mse_plane1)
    ssim_plane1 = ssim(recon_plane1_centered, real_plane1_centered, data_range=1.0)
    print(f"Plane1 (Centered) - PSNR: {psnr_plane1:.2f} dB, SSIM: {ssim_plane1:.4f}")

    # Plane2对比
    mse_plane2 = np.mean((recon_plane2_centered - real_plane2_centered) ** 2)
    if mse_plane2 == 0:
        psnr_plane2 = float('inf')
    else:
        psnr_plane2 = 10 * np.log10(1.0 / mse_plane2)
    ssim_plane2 = ssim(recon_plane2_centered, real_plane2_centered, data_range=1.0)
    print(f"Plane2 (Centered) - PSNR: {psnr_plane2:.2f} dB, SSIM: {ssim_plane2:.4f}")

    # Plane3对比
    mse_plane3 = np.mean((recon_plane3_centered - real_plane3_centered) ** 2)
    if mse_plane3 == 0:
        psnr_plane3 = float('inf')
    else:
        psnr_plane3 = 10 * np.log10(1.0 / mse_plane3)
    ssim_plane3 = ssim(recon_plane3_centered, real_plane3_centered, data_range=1.0)
    print(f"Plane3 (Centered) - PSNR: {psnr_plane3:.2f} dB, SSIM: {ssim_plane3:.4f}")

if __name__ == '__main__':
    fig1()

