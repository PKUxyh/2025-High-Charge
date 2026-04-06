#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append(r'/public1/home/m8s000916/xyh/real_laser/')
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import matplotlib.gridspec as gridspec
import os
from matplotlib import colors,ticker,cm
from scipy.optimize import curve_fit
from fld_module import EM
from density_module import plasma
from scipy.signal import hilbert
from scipy.interpolate import griddata
from multiprocessing import Pool
from functools import partial
import time
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt
from matplotlib.patches import Arc


matplotlib.font_manager.fontManager.addfont('/public1/home/m8s000916/.fonts/arial/arial.ttf')
matplotlib.font_manager.fontManager.addfont('/public1/home/m8s000916/.fonts/arial/arialbd.ttf')
plt.rcParams['font.family'] = 'Arial'

# 设置全局样式
plt.style.use('default')  # 重置为默认样式
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'lines.linewidth': 1.5,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'figure.dpi': 300,
    'figure.figsize': (6.74, 3.8),  # 图像大小
    'xtick.direction': 'in',         # 刻度线朝内
    'ytick.direction': 'in',
    'xtick.major.width': 1.2,        # 主刻度线粗细
    'ytick.major.width': 1.2,
    'xtick.major.size': 6,           # 主刻度线长度
    'ytick.major.size': 6,
    'xtick.labelsize': 12,           # 刻度数字大小
    'ytick.labelsize': 12,
})


def find_gradient_max_points(density_rz, theta_rec, y_plot, density_threshold=0.005):
    """
    找到每个θ方向上梯度绝对值最大的点，但只考虑密度最大值超过阈值且索引大于50的角度

    参数:
    density_rz: 密度数据矩阵 (角度, 半径)
    theta_rec: 角度数组
    y_plot: 半径数组
    density_threshold: 密度最大值阈值，只有该角度上的最大密度超过此值才考虑

    返回:
    符合条件的最大梯度点列表，每个元素为(theta, r)元组
    """
    max_gradient_points = []

    for i, theta in enumerate(theta_rec):
        # 获取当前θ的密度剖面
        density_profile = density_rz[i, :]

        # 检查该角度上的最大密度是否超过阈值
        if np.max(density_profile) < density_threshold:
            continue  # 跳过这个角度

        # 计算梯度（密度从低到高，中心到边界）
        gradient = np.gradient(density_profile, y_plot)

        # 找到梯度最大的点（排除边界点且索引大于50）
        valid_indices = np.where(~np.isnan(gradient))[0]
        valid_indices = valid_indices[valid_indices > 50]
        if len(valid_indices) > 0:
            max_idx = valid_indices[np.argmax(gradient[valid_indices])]
            r = y_plot[max_idx]
            max_gradient_points.append((theta, r))

    return max_gradient_points

def ellipse_function(params, points):
    """
    椭圆函数，用于最小二乘拟合
    """
    x0, y0, a, b, phi = params
    result = []
    
    for point in points:
        x, y = point
        x_rot = (x - x0) * np.cos(phi) + (y - y0) * np.sin(phi)
        y_rot = -(x - x0) * np.sin(phi) + (y - y0) * np.cos(phi)
        result.append((x_rot/a)**2 + (y_rot/b)**2 - 1)
    
    return np.array(result)

def fit_ellipse(points):
    """
    使用最小二乘法拟合椭圆
    """
    # 初始猜测：使用点的均值作为中心，标准差作为轴长
    points_array = np.array(points)
    x_mean, y_mean = np.mean(points_array, axis=0)
    x_std, y_std = np.std(points_array, axis=0)
    
    # 初始参数猜测 [x0, y0, a, b, phi]
    initial_guess = [x_mean, y_mean, x_std, y_std, 0]
    
    # 执行最小二乘拟合
    result = least_squares(ellipse_function, initial_guess, args=(points,))
    
    if result.success:
        return result.x
    else:
        raise ValueError("椭圆拟合失败")

def get_ellipse_parameters(max_gradient_points, min_points=20):
    """
    获取椭圆参数

    参数:
    max_gradient_points: 梯度最大点列表，每个元素为(theta, r)元组
    min_points: 拟合椭圆所需的最小点数
    """
    # 检查是否有足够的点进行拟合
    if len(max_gradient_points) < min_points:
        print(f"警告: 只有{len(max_gradient_points)}个点，不足以拟合椭圆(需要至少{min_points}个点)")
        return None

    # 将极坐标点转换为笛卡尔坐标
    cartesian_points = []
    for theta, r in max_gradient_points:
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        cartesian_points.append((x, y))

    # 拟合椭圆
    try:
        x0, y0, a, b, phi = fit_ellipse(cartesian_points)

        # 确保a是长轴，b是短轴
        if a < b:
            a, b = b, a
            phi += np.pi/2

        # 将phi限制在[0, π)范围内
        phi = phi % np.pi

        return {
            "center": (x0, y0),
            "major_axis": a,
            "minor_axis": b,
            "orientation": phi,  # 弧度制，从x轴逆时针旋转的角度
            "num_points": len(max_gradient_points)  # 添加使用的点数信息
        }
    except ValueError as e:
        print(f"椭圆拟合错误: {e}")
        return None


def find_strong_regions_max(data_1d, threshold=3, min_region_length=30):
    """
    找到所有大于threshold的数据及其索引，根据索引连续性分为若干区域，
    筛选出长度大于min_region_length的区域，并返回每个区域的最大值及其索引。

    参数:
    data_1d: 一维数据数组
    threshold: 强度阈值，默认3
    min_region_length: 最小区域长度阈值，默认30

    返回:
    列表，每个元素是一个元组(max_index, max_value)，表示区域最大值及其索引
    """
    strong_indices = np.where(data_1d > threshold)[0]
    if len(strong_indices) == 0:
        return []

    # 按连续性分组
    regions = []
    region = [strong_indices[0]]
    for idx in strong_indices[1:]:
        if idx == region[-1] + 1:
            region.append(idx)
        else:
            regions.append(region)
            region = [idx]
    regions.append(region)

    # 筛选出长度大于min_region_length的区域
    long_regions = [region for region in regions if len(region) >= min_region_length]

    if len(long_regions) == 0:
        print("not find strong region")
        return []


    # 找到每个长区域的最大值及其索引
    result = []
    for region in long_regions:
        region_values = data_1d[region]
        max_idx_in_region = np.argmax(region_values)
        max_value = region_values[max_idx_in_region]
        max_index = region[max_idx_in_region]
        result.append((max_index, max_value))

    return result

def find_half_max_points(density_rz, theta_rec, y_plot):
    """
    找到每个θ方向上密度最大值的0.5倍阈值的点

    参数:
    density_rz: 密度数据矩阵 (角度, 半径)
    theta_rec: 角度数组
    y_plot: 半径数组

    返回:
    符合条件的点列表，每个元素为(theta, r)元组
    """
    half_max_points = []

    for i, theta in enumerate(theta_rec):
        # 获取当前θ的密度剖面
        density_profile = density_rz[i, :]

        # 找到密度最大值及其0.5倍的阈值
        max_density = np.max(density_profile)
        half_max_threshold = 0.5 * max_density

        # 找到第一个小于等于阈值的点
        valid_indices = np.where(density_profile <= half_max_threshold)[0]
        if len(valid_indices) > 0:
            first_idx = valid_indices[0]
            r = y_plot[first_idx]
            half_max_points.append((theta, r))

    print(f"找到 {len(half_max_points)} 个半最大值点")

    return half_max_points


def process_case(args):
    """
    处理单个 case 的计算任务
    """
    case_folder, k, max_mode, mv_xlim, mv_ylim, theta_rec = args

    EM_obj = EM(case_folder)
    ez_re, ez_im, time, xx, yy = EM_obj.get_modes(k, max_mode, 'e1')
    er_re, er_im, _, _, _ = EM_obj.get_modes(k, max_mode, 'e2')
    etheta_re, etheta_im, _, _, _ = EM_obj.get_modes(k, max_mode, 'e3')
    density = plasma(case_folder, 'electrons')
    var_re, var_im, time, xx, yy = density.get_modes(k, max_mode)
    density_0 = density.recover_fields(var_re, var_im, 0)

    # 找到激光的第一个和第二个 peak
    ex_0 = np.abs(hilbert(EM_obj.rec_fields_xyz(ez_re, er_re, etheta_re, ez_im, er_im, etheta_im, 0)[1], axis=1))
    strong_points = find_strong_regions_max(ex_0[5, :], threshold=3, min_region_length=30)

    if len(strong_points) > 2:
        sorted_points = sorted(strong_points, key=lambda x: x[1], reverse=True)[:2]
        sorted_points = sorted(sorted_points, key=lambda x: x[0], reverse=True)
        first_peak_index = sorted_points[0][0]
        second_peak_index = sorted_points[1][0]
    elif len(strong_points) == 2:
        sorted_points = sorted(strong_points, key=lambda x: x[0], reverse=True)
        first_peak_index = sorted_points[0][0]
        second_peak_index = sorted_points[1][0]
    elif len(strong_points) == 1:
        first_peak_index = strong_points[0][0]
        second_peak_index = None
    else:
        first_peak_index = None
        second_peak_index = None

    ex_rz = np.zeros((len(theta_rec), int((mv_ylim[1] - mv_ylim[0]) * 1), 2))
    for i in range(len(theta_rec)):
        _, ex, _ = EM_obj.rec_fields_xyz(ez_re, er_re, etheta_re, ez_im, er_im, etheta_im, theta_rec[i])
        ex = hilbert(ex, axis=1)
        if first_peak_index is not None and second_peak_index is not None:
            envelope_first = np.abs(ex[:, first_peak_index])**2
            envelope_second = np.abs(ex[:, second_peak_index])**2
            ex_rz[i, :, 0] = envelope_first
            ex_rz[i, :, 1] = envelope_second
        elif first_peak_index is not None:
            envelope_first = np.abs(ex[:, first_peak_index])**2
            ex_rz[i, :, 0] = envelope_first
        else:
            ex_rz[i, :, 0] = 0

    # 找到密度切片的索引，Ez=0
    ez_0 = EM_obj.rec_fields_xyz(ez_re, er_re, etheta_re, ez_im, er_im, etheta_im, 0)[0]
    ez_axis = ez_0[5, :]
    b, a = butter(N=4, Wn=1/31, btype='low', analog=False)
    ez_axis = filtfilt(b, a, ez_axis)
    zero_indices = np.where(np.isclose(ez_axis, 0, atol=1e-3))[0]

    window = 200
    valid_zero_indices = []
    for idx in zero_indices:
        left = idx - window
        right = idx + window
        if left >= 0 and right < len(ez_axis):
            if ez_axis[left] < ez_axis[idx] < ez_axis[right]:
                if ez_axis[right] - ez_axis[left] > 0.008:
                    valid_zero_indices.append(idx)

    if valid_zero_indices:
        max_zero_idx = max(valid_zero_indices)+30
    else:
        max_zero_idx = None

    density_rz = np.zeros((len(theta_rec), int((mv_ylim[1] - mv_ylim[0]) * 1)))
    for i in range(len(theta_rec)):
        density_theta = density.recover_fields(var_re, var_im, theta_rec[i])
        density_rz[i, :] = density_theta[:, max_zero_idx]



    # 插值到笛卡尔坐标系
    r_vals = np.linspace(0, ex_rz.shape[1]/(2*np.pi)*0.8, ex_rz.shape[1])
    theta_vals = theta_rec
    R, Theta = np.meshgrid(r_vals, theta_vals)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    # 目标笛卡尔网格
    grid_x = np.linspace(-13, 13, 400)
    grid_y = np.linspace(-13, 13, 400)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    # ex1插值
    ex1_cartesian = griddata(
        (X.flatten(), Y.flatten()),
        ex_rz[:, :, 0].flatten(),
        (grid_xx, grid_yy),
        method='linear',
        fill_value=0
    )

    # density插值
    density_cartesian = griddata(
        (X.flatten(), Y.flatten()),
        density_rz.flatten(),
        (grid_xx, grid_yy),
        method='linear',
        fill_value=0
    )
    # 沿x轴镜像翻转数据
    ex1_cartesian = np.flipud(ex1_cartesian)
    density_cartesian = np.flipud(density_cartesian)
    return ex1_cartesian, density_cartesian

def plot_3case():
    case_folder = [
        '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes',
        '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_gauss',
        '/public1/home/m8s000916/xyh/real_laser/real_astig_elli_6e18_1.2895a0'
    ]
    k_list = [250, 240, 250]
    max_modes = [6, 1, 3]
    mv_xlims = [[-706.8, 0], [-706.8, 0], [-706.8, 0]]
    mv_ylims = [[0, 684], [0, 684], [0, 692]]
    theta_rec = np.linspace(0, 2 * np.pi, 360)

    # 准备任务参数
    tasks = [
        (case_folder[i], k_list[i], max_modes[i], mv_xlims[i], mv_ylims[i], theta_rec)
        for i in range(3)
    ]

    # 使用多进程计算
    with Pool(processes=3) as pool:
        results = pool.map(process_case, tasks)

    # 合并结果
    ex1, density_peak = zip(*results)
    return list(ex1), list(density_peak)



if __name__ == '__main__':
    ex1, density_peak = plot_3case()

    label = ['case r', 'case g', 'case ea']
    z_label = ['z = -1.05 mm', 'z = -1.10 mm', 'z = -1.05 mm']
    fig1_num = ['b1', 'b2', 'b3']
    fig2_num = ['c1', 'c2', 'c3']
    # density_peak 从 nc 转换为 cm^-3
    for i in range(len(density_peak)):
        density_peak[i] = density_peak[i] * 1.74e21 / 1e19
    
    # 创建图形和网格布局
    fig = plt.figure(constrained_layout=True, figsize=(6.74, 3.8))
    
    # 使用更精细的网格布局
    gs = gridspec.GridSpec(
        3, 5,  # 增加一列用于更好的间距控制
        height_ratios=[0.15, 1, 1], 
        width_ratios=[0.05, 1, 1, 1, 0.005],  # 两侧增加空白列
        wspace=0.05,  # 减少子图间水平间距
        hspace=0.05,  # 减少子图间垂直间距
        figure=fig
    )

    # 绘制 ex1 的三幅笛卡尔图
    ex1_imgs = []
    vmin_ex1 = 0
    vmax_ex1 = 27
    extent = [-13, 13, -13, 13]
    
    for i in range(3):
        ax = fig.add_subplot(gs[1, i+1])  # 第二行，第2-4列
        im = ax.imshow(
            ex1[i],
            origin='lower',
            extent=extent,
            vmin=vmin_ex1, vmax=vmax_ex1,
            cmap='jet',
            aspect='equal'
        )
        ax.set_xlim(-13, 13)
        ax.set_ylim(-13, 13)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 绘制5um的横线和竖线
        scale_len = 5
        x0, y0 = -11, -11
        ax.plot([x0, x0 + scale_len], [y0, y0], color='w', lw=1.5)
        ax.plot([x0, x0], [y0, y0 + scale_len], color='w', lw=1.5)
        ax.text(x0 + scale_len/2 + 1, y0 + 2.5, r'$5~\mathrm{~\mu m}$', 
                color='w', ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x0 + scale_len + 1.5, y0, r'x', color='w', ha='center', va='center', 
                fontsize=10, fontweight='bold')
        ax.text(x0, y0 + scale_len + 1.5, r'y', color='w', ha='center', va='center', 
                fontsize=10, fontweight='bold')
        
        # 在右上角标注 z_label
        ax.text(0.95, 0.95, z_label[i], transform=ax.transAxes,
                fontsize=12, color='w', ha='right', va='top')
        
        # 在左上角标注 abc
        ax.text(0.05, 0.95, fig1_num[i], transform=ax.transAxes,
                fontsize=14, color='w', ha='left', va='top', fontweight='bold')

        # 角度标注逻辑保持不变
        if i == 0 or i == 2:
            max_idx = np.unravel_index(np.argmax(ex1[i]), ex1[i].shape)
            max_x = extent[0] + (extent[1] - extent[0]) * max_idx[1] / ex1[i].shape[1]
            max_y = extent[2] + (extent[3] - extent[2]) * max_idx[0] / ex1[i].shape[0]
            line_len = 20
            dx = line_len * np.cos(np.radians(155.7))
            dy = line_len * np.sin(np.radians(155.7))
            
            ax.plot([max_x - line_len/2, max_x + line_len/2], [max_y, max_y], 
                   linestyle='--', color='w', lw=0.5)
            ax.plot([max_x - dx/2, max_x + dx/2], [max_y - dy/2, max_y + dy/2],
                   linestyle='--', color='w', lw=0.5)
            
            arc_radius = 4
            arc = Arc((max_x, max_y), width=arc_radius*2, height=arc_radius*2,
                     angle=0, theta1=-24.3, theta2=0, color='w', lw=0.5, linestyle='-')
            ax.add_patch(arc)
            ax.text(max_x + 8, max_y+1.5, r'$\phi=24.3°$', color='yellow', fontsize=10, 
                   ha='center', va='center', fontweight='bold')
        
        ex1_imgs.append(im)

    # 为 ex1 添加 colorbar (使用第一列)
    cbar_ax_ex1 = fig.add_subplot(gs[1, 0])  # 第二行，第一列
    cbar_ex1 = plt.colorbar(ex1_imgs[0], cax=cbar_ax_ex1, orientation='vertical', extend='max')
    
    # 设置 ex1 colorbar 的刻度
    ticks_ex1 = np.linspace(vmin_ex1, vmax_ex1, 5)
    cbar_ex1.set_ticks(ticks_ex1)
    ticklabels_ex1 = [f"{np.sqrt(tick):.1f}" for tick in ticks_ex1]
    cbar_ex1.set_ticklabels(ticklabels_ex1)
    
    # 设置 ex1 colorbar 标签
    cbar_ax_ex1.set_ylabel(r'$a_0$', fontsize=12, rotation=90, labelpad=10)
    cbar_ax_ex1.yaxis.set_label_position('left')
    cbar_ax_ex1.yaxis.tick_left()
    cbar_ax_ex1.tick_params(axis='y', direction='in', length=3, pad=2)

    # 绘制 density_peak 的三幅笛卡尔图
    density_imgs = []
    vmin_density = 0
    vmax_density = 3.43
    
    for i in range(3):
        ax = fig.add_subplot(gs[2, i+1])  # 第三行，第2-4列
        im = ax.imshow(
            density_peak[i],
            origin='lower',
            extent=extent,
            vmin=vmin_density, vmax=vmax_density,
            cmap='plasma',
            aspect='equal'
        )
        ax.set_xlim(-13, 13)
        ax.set_ylim(-13, 13)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 绘制5um的横线和竖线
        scale_len = 5
        x0, y0 = -11, -11
        ax.plot([x0, x0 + scale_len], [y0, y0], color='w', lw=1.5)
        ax.plot([x0, x0], [y0, y0 + scale_len], color='w', lw=1.5)
        ax.text(x0 + scale_len/2 + 1, y0 + 2.5, r'$5~\mathrm{~\mu m}$', 
                color='w', ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x0 + scale_len + 1.5, y0, r'x', color='w', ha='center', va='center', 
                fontsize=10, fontweight='bold')
        ax.text(x0, y0 + scale_len + 1.5, r'y', color='w', ha='center', va='center', 
                fontsize=10, fontweight='bold')
        
        # 角度标注逻辑保持不变
        if i == 0 or i == 2:
            ax.plot([max_x - line_len/2, max_x + line_len/2], [max_y, max_y], 
                   linestyle='--', color='w', lw=0.5)
            ax.plot([max_x - dx/2, max_x + dx/2], [max_y - dy/2, max_y + dy/2],
                   linestyle='--', color='w', lw=0.5)
            
            arc_radius = 4
            arc = Arc((max_x, max_y), width=arc_radius*2, height=arc_radius*2,
                     angle=0, theta1=-24.3, theta2=0, color='w', lw=0.5, linestyle='-')
            ax.add_patch(arc)
            ax.text(max_x + 8, max_y+1.5, r'$\phi=24.3°$', color='yellow', fontsize=10, 
                   ha='center', va='center', fontweight='bold')

        # 在左上角标注 abc
        ax.text(0.05, 0.95, fig2_num[i], transform=ax.transAxes,
                fontsize=14, color='w', ha='left', va='top', fontweight='bold')

        density_imgs.append(im)

    # 为 density 添加 colorbar (使用第一列)
    cbar_ax_density = fig.add_subplot(gs[2, 0])  # 第三行，第一列
    cbar_density = plt.colorbar(density_imgs[0], cax=cbar_ax_density, orientation='vertical')
    
    # 设置 density colorbar 的刻度
    cbar_density.set_ticks([1, 2, 3])
    cbar_density.set_ticklabels(['1.0', '2.0', '3.0'])
    
    # 设置 density colorbar 标签
    cbar_ax_density.set_ylabel(r'$n_\mathrm{p}~[\times 10^{19} cm^{-3}]$', 
                              fontsize=12, rotation=90, labelpad=10)
    cbar_ax_density.yaxis.set_label_position('left')
    cbar_ax_density.yaxis.tick_left()
    cbar_ax_density.tick_params(axis='y', direction='in', length=3, pad=2)

    # 在第一行添加标签
    for i in range(3):
        ax = fig.add_subplot(gs[0, i+1])  # 第一行，第2-4列
        ax.axis('off')
        ax.text(0.5, 0.5, label[i], ha='center', va='center', 
                fontsize=14, transform=ax.transAxes)

    # 保存图像
    plt.savefig('/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/slice_rz/fig5b_cartesian.png', 
                dpi=300, bbox_inches='tight')
    plt.close(fig)