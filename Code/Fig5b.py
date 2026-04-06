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
from matplotlib.patches import Ellipse

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
    'figure.figsize': (6.74, 2.8),  # 图像大小
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



# def plot_3pos():
#     case_folder = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes'

#     k = 380
#     max_modes = 6
#     # mv_xlims = [-706.8, 0]
#     mv_ylim = [0, 684]
#     theta_rec = np.linspace(0, 2 * np.pi, 360)

#     EM_obj = EM(case_folder)
#     ez_re, ez_im, time, xx, yy = EM_obj.get_modes(k, max_modes, 'e1')
#     er_re, er_im, _, _, _ = EM_obj.get_modes(k, max_modes, 'e2')
#     etheta_re, etheta_im, _, _, _ = EM_obj.get_modes(k, max_modes, 'e3')
#     density = plasma(case_folder, 'electrons')
#     var_re, var_im, _, _, _ = density.get_modes(k, max_modes)


#     # 找到激光的第一个和第二个 peak
#     ex_0 = np.abs(hilbert(EM_obj.rec_fields_xyz(ez_re, er_re, etheta_re, ez_im, er_im, etheta_im, 0)[1], axis=1))
#     strong_points = find_strong_regions_max(ex_0[5, :], threshold=3, min_region_length=30)

#     if len(strong_points) > 2:
#         sorted_points = sorted(strong_points, key=lambda x: x[1], reverse=True)[:2]
#         sorted_points = sorted(sorted_points, key=lambda x: x[0], reverse=True)
#         first_peak_index = sorted_points[0][0]
#         second_peak_index = sorted_points[1][0]
#     elif len(strong_points) == 2:
#         sorted_points = sorted(strong_points, key=lambda x: x[0], reverse=True)
#         first_peak_index = sorted_points[0][0]
#         second_peak_index = sorted_points[1][0]
#     elif len(strong_points) == 1:
#         first_peak_index = strong_points[0][0]
#         second_peak_index = None
#     else:
#         first_peak_index = None
#         second_peak_index = None

#     ex_rz = np.zeros((len(theta_rec), int((mv_ylim[1] - mv_ylim[0]) * 1), 2))
#     for i in range(len(theta_rec)):
#         _, ex, _ = EM_obj.rec_fields_xyz(ez_re, er_re, etheta_re, ez_im, er_im, etheta_im, theta_rec[i])
#         ex = hilbert(ex, axis=1)
#         if first_peak_index is not None and second_peak_index is not None:
#             envelope_first = np.abs(ex[:, first_peak_index])**2
#             envelope_second = np.abs(ex[:, second_peak_index])**2
#             ex_rz[i, :, 0] = envelope_first
#             ex_rz[i, :, 1] = envelope_second
#         elif first_peak_index is not None:
#             envelope_first = np.abs(ex[:, first_peak_index])**2
#             ex_rz[i, :, 0] = envelope_first
#         else:
#             ex_rz[i, :, 0] = 0

#     # 找到密度切片的索引，Ez=0
#     ez_0 = EM_obj.rec_fields_xyz(ez_re, er_re, etheta_re, ez_im, er_im, etheta_im, 0)[0]
#     ez_axis = ez_0[5, :]
#     b, a = butter(N=4, Wn=1/31, btype='low', analog=False)
#     ez_axis = filtfilt(b, a, ez_axis)
#     zero_indices = np.where(np.isclose(ez_axis, 0, atol=1e-3))[0]

#     window = 200
#     valid_zero_indices = []
#     for idx in zero_indices:
#         left = idx - window
#         right = idx + window
#         if left >= 0 and right < len(ez_axis):
#             if ez_axis[left] < ez_axis[idx] < ez_axis[right]:
#                 if ez_axis[right] - ez_axis[left] > 0.008:
#                     valid_zero_indices.append(idx)

#     if valid_zero_indices:
#         max_zero_idx = max(valid_zero_indices)+30
#     else:
#         max_zero_idx = None

#     density_rz = np.zeros((len(theta_rec), int((mv_ylim[1] - mv_ylim[0]) * 1)))
#     for i in range(len(theta_rec)):
#         density_theta = density.recover_fields(var_re, var_im, theta_rec[i])
#         density_rz[i, :] = density_theta[:, max_zero_idx]



#     # 插值到笛卡尔坐标系
#     r_vals = np.linspace(0, ex_rz.shape[1]/(2*np.pi)*0.8, ex_rz.shape[1])
#     theta_vals = theta_rec
#     R, Theta = np.meshgrid(r_vals, theta_vals)
#     X = R * np.cos(Theta)
#     Y = R * np.sin(Theta)

#     # 目标笛卡尔网格
#     grid_x = np.linspace(-20, 20, 500)
#     grid_y = np.linspace(-20, 20, 500)
#     grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

#     # ex1插值
#     ex1_cartesian = griddata(
#         (X.flatten(), Y.flatten()),
#         ex_rz[:, :, 0].flatten(),
#         (grid_xx, grid_yy),
#         method='linear',
#         fill_value=0
#     )

#     ex2_cartesian = griddata(
#         (X.flatten(), Y.flatten()),
#         ex_rz[:, :, 1].flatten(),
#         (grid_xx, grid_yy),
#         method='linear',
#         fill_value=0
#     )
#     # density插值
#     density_cartesian = griddata(
#         (X.flatten(), Y.flatten()),
#         density_rz.flatten(),
#         (grid_xx, grid_yy),
#         method='linear',
#         fill_value=0
#     )
#     # 沿x轴镜像翻转数据
#     ex1_cartesian = np.flipud(ex1_cartesian)
#     ex2_cartesian = np.flipud(ex2_cartesian)
#     density_cartesian = np.flipud(density_cartesian)
    
#     return ex1_cartesian, ex2_cartesian, density_cartesian


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
    



def plot_2bubble():
    case_folder = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes'

    k_list = [380, 580]
    max_modes = 6
    mv_ylim = [0, 684]
    theta_rec = np.linspace(0, 2 * np.pi, 360)

    for k in k_list:
        EM_obj = EM(case_folder)
        ez_re, ez_im, time, xx, yy = EM_obj.get_modes(k, max_modes, 'e1')
        er_re, er_im, _, _, _ = EM_obj.get_modes(k, max_modes, 'e2')
        etheta_re, etheta_im, _, _, _ = EM_obj.get_modes(k, max_modes, 'e3')
        density = plasma(case_folder, 'electrons')
        var_re, var_im, _, _, _ = density.get_modes(k, max_modes)



##################################################################################################
############################找到激光的第一个和第二个peak###########################################
##################################################################################################
        ex_0 = np.abs(hilbert(EM_obj.rec_fields_xyz(ez_re, er_re, etheta_re, ez_im, er_im, etheta_im, 0)[1], axis=1))
        strong_points = find_strong_regions_max(ex_0[5, :], threshold=3, min_region_length=30)

        # 修改逻辑以确保 first_peak 是索引较大的那个
        if len(strong_points) > 2:
            sorted_points = sorted(strong_points, key=lambda x: x[1], reverse=True)[:2]
            sorted_points = sorted(sorted_points, key=lambda x: x[0], reverse=True)  # 按索引从大到小排序
            first_peak_index = sorted_points[0][0]
            second_peak_index = sorted_points[1][0]
        elif len(strong_points) == 2:
            sorted_points = sorted(strong_points, key=lambda x: x[0], reverse=True)  # 按索引从大到小排序
            first_peak_index = sorted_points[0][0]
            second_peak_index = sorted_points[1][0]
        elif len(strong_points) == 1:
            first_peak_index = strong_points[0][0]
            second_peak_index = None
        else:
            first_peak_index = None
            second_peak_index = None

        ex_rz = np.zeros((len(theta_rec), mv_ylim[1], 2))
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
        # 判断哪个peak更大，并对a0大的那个peak进行椭圆拟合
        if first_peak_index is not None and second_peak_index is not None:
            peak1_val = np.max(np.abs(ex_rz[:, :, 0]))
            peak2_val = np.max(np.abs(ex_rz[:, :, 1]))
            if peak1_val >= peak2_val:
                y_plot = np.linspace(0, ex_rz.shape[1]/(2*np.pi)*0.8, ex_rz.shape[1])
                half_max_points = find_half_max_points(ex_rz[:, :, 1], theta_rec, y_plot)
                ellipse_params_ax2 = get_ellipse_parameters(half_max_points)
            else:
                y_plot = np.linspace(0, ex_rz.shape[1]/(2*np.pi)*0.8, ex_rz.shape[1])
                half_max_points = find_half_max_points(ex_rz[:, :, 1], theta_rec, y_plot)
                ellipse_params_ax2 = get_ellipse_parameters(half_max_points)
        elif first_peak_index is not None:
            y_plot = np.linspace(0, ex_rz.shape[1]/(2*np.pi)*0.8, ex_rz.shape[1])
            half_max_points = find_half_max_points(ex_rz[:, :, 0], theta_rec, y_plot)
            ellipse_params_ax2 = get_ellipse_parameters(half_max_points)
        elif second_peak_index is not None:
            y_plot = np.linspace(0, ex_rz.shape[1]/(2*np.pi)*0.8, ex_rz.shape[1])
            half_max_points = find_half_max_points(ex_rz[:, :, 1], theta_rec, y_plot)
            ellipse_params_ax2 = get_ellipse_parameters(half_max_points)
        else:
            ellipse_params_ax2 = None
        
        if k == 380:
            ellipse_params1 = ellipse_params_ax2
        else:
            ellipse_params2 = ellipse_params_ax2









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


        # # 计算物理空间的半径坐标
        # y_plot = np.linspace(0, var_re.shape[1]/(2*np.pi)*0.8, var_re.shape[1])
        # if k == 380:
        #     max_gradient_points = find_gradient_max_points(density_rz, theta_rec, y_plot)
        #     ellipse_params1 = get_ellipse_parameters(max_gradient_points)
        # else:
        #     max_gradient_points = find_gradient_max_points(density_rz, theta_rec, y_plot, density_threshold=0.02)
        #     ellipse_params2 = get_ellipse_parameters(max_gradient_points)






        # 插值到笛卡尔坐标系
        r_vals = np.linspace(0, var_re.shape[1]/(2*np.pi)*0.8, var_re.shape[1])
        theta_vals = theta_rec
        R, Theta = np.meshgrid(r_vals, theta_vals)
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)

        # 目标笛卡尔网格
        grid_x = np.linspace(-20, 20, 500)
        grid_y = np.linspace(-20, 20, 500)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

        # density插值
        density_cartesian = griddata(
            (X.flatten(), Y.flatten()),
            density_rz.flatten(),
            (grid_xx, grid_yy),
            method='linear',
            fill_value=0
        )

        if k == 380:
            density_cartesian1 = np.flipud(density_cartesian)
        else:
            density_cartesian2 = np.flipud(density_cartesian)

    return density_cartesian1, density_cartesian2, ellipse_params1, ellipse_params2


if __name__ == '__main__':
    density_peak1, density_peak2, ellipse_params1, ellipse_params2 = plot_2bubble()
    z_label = ['z = -0.46 mm', 'z = 0.46 mm']
    fig1_num = ['b', 'c']

    # density_peak 从 nc 转换为 cm^-3
    for i in range(len(density_peak1)):
        density_peak1[i] = density_peak1[i] * 1.74e21 / 1e19

    for i in range(len(density_peak2)):
        density_peak2[i] = density_peak2[i] * 1.74e21 / 1e19



    def read_data(file_path):
        with open(file_path, 'r') as file:
            data = [float(line.strip()) for line in file]
        return data
    
    simi_file = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/ellipse_analysis/similarity_coeffs.txt'
    simi_coeffs = read_data(simi_file)

    time_file = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/ellipse_analysis/time.txt'
    time = read_data(time_file)
    z_plot = np.array(time) / (2*np.pi) * 0.8e-3 - 2.2
    # 对simi_coeffs进行低通滤波
    b, a = butter(N=3, Wn=0.1, btype='low')
    simi_fit = filtfilt(b, a, simi_coeffs)
    z_fit = z_plot



    # 创建图形和网格布局
    fig = plt.figure(constrained_layout=True)

    # 使用更精细的网格布局
    gs = gridspec.GridSpec(
        1, 4,
        width_ratios=[0.05, 1, 1, 0.02],  # 两侧增加空白列
        wspace=0.02,  # 减少子图间水平间距
        figure=fig
    )

    extent = [-20, 20, -20, 20]
    # 绘制 density_peak 的三幅笛卡尔图
    density_imgs = []
    vmin_density = 0
    vmax_density = 2


    ax1 = fig.add_subplot(gs[0, 1])  # 第三行，第2-4列
    im1 = ax1.imshow(
        density_peak1,
        origin='lower',
        extent=extent,
        vmin=vmin_density, vmax=vmax_density,
        cmap='plasma',
        aspect='equal'
    )

    ax1.set_xlim(-20, 20)
    ax1.set_ylim(-20, 20)
    ax1.set_xticks([])
    ax1.set_yticks([])
    # 绘制第一个椭圆（对x轴上下翻转）
    if ellipse_params1 is not None:
        center = ellipse_params1["center"]
        a = ellipse_params1["major_axis"]
        b = ellipse_params1["minor_axis"]
        phi = ellipse_params1["orientation"]
        # 对y坐标取反，phi取负，实现上下翻转
        flipped_center = (center[0], -center[1])
        flipped_phi = -phi
        ellipse_color = 'cyan'  # 与plasma色带对比明显
        # ellipse_patch = Ellipse(
        #     xy=flipped_center,
        #     width=2*a,
        #     height=2*b,
        #     angle=np.degrees(flipped_phi),
        #     edgecolor=ellipse_color,
        #     facecolor='none',
        #     lw=1,
        #     linestyle='--',
        #     zorder=10
        # )
        # ax1.add_patch(ellipse_patch)
        # 绘制长轴和短轴（加长到±2a, ±2b）
        x0, y0 = flipped_center
        cos_phi = np.cos(flipped_phi)
        sin_phi = np.sin(flipped_phi)
        # 长轴端点
        x1 = x0 + 3 * a * cos_phi
        y1 = y0 + 3 * a * sin_phi
        x2 = x0 - 3 * a * cos_phi
        y2 = y0 - 3 * a * sin_phi
        ax1.plot([x1, x2], [y1, y2], color=ellipse_color, lw=1, linestyle='--')
        # 长轴方向文字
        major_text_offset = a * 0.02
        ax1.text(x0 + major_text_offset * cos_phi, y0 + major_text_offset * sin_phi, 'major axis',
            color='w', fontsize=8, ha='left', va='top', rotation=np.degrees(flipped_phi))
        # 短轴端点
        x3 = x0 + 3 * b * (-sin_phi)
        y3 = y0 + 3 * b * cos_phi
        x4 = x0 - 3 * b * (-sin_phi)
        y4 = y0 - 3 * b * cos_phi
        ax1.plot([x3, x4], [y3, y4], color=ellipse_color, lw=1, linestyle='--')
        # 短轴方向文字
        minor_text_offset = b * 0.02
        minor_angle = flipped_phi + np.pi/2
        ax1.text(x0 + minor_text_offset * np.cos(minor_angle), y0 + minor_text_offset * np.sin(minor_angle), 'minor axis',
            color='w', fontsize=8, ha='left', va='bottom', rotation=np.degrees(minor_angle))


    # 绘制5um的横线和竖线
    scale_len = 5
    x0, y0 = -18, -18
    ax1.plot([x0, x0 + scale_len], [y0, y0], color='w', lw=1.5)
    ax1.plot([x0, x0], [y0, y0 + scale_len], color='w', lw=1.5)
    ax1.text(x0 + scale_len/2 + 1, y0 + 2.5, r'$5\mathrm{~\mu m}$', 
            color='w', ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.text(x0 + scale_len + 1.5, y0, r'x', color='w', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax1.text(x0, y0 + scale_len + 1.5, r'y', color='w', ha='center', va='center', 
            fontsize=10, fontweight='bold')

    # 在右上角标注 z_label
    ax1.text(0.95, 0.95, z_label[0], transform=ax1.transAxes,
            fontsize=12, color='w', ha='right', va='top')
    # 在左上角标注 abc
    ax1.text(0.05, 0.95, fig1_num[0], transform=ax1.transAxes,
            fontsize=14, color='w', ha='left', va='top', fontweight='bold')
    density_imgs.append(im1)



    ax2 = fig.add_subplot(gs[0, 2])  # 第三行，第2-4列
    im2 = ax2.imshow(
        density_peak2,
        origin='lower',
        extent=extent,
        vmin=vmin_density, vmax=vmax_density,
        cmap='plasma',
        aspect='equal'
    )

    ax2.set_xlim(-20, 20)
    ax2.set_ylim(-20, 20)
    ax2.set_xticks([])
    ax2.set_yticks([])



    # 绘制第二个椭圆（对x轴上下翻转）
    if ellipse_params2 is not None:
        center = ellipse_params2["center"]
        a = ellipse_params2["major_axis"]
        b = ellipse_params2["minor_axis"]
        phi = ellipse_params2["orientation"]
        # 对y坐标取反，phi取负，实现上下翻转
        flipped_center = (center[0], -center[1])
        flipped_phi = -phi
        ellipse_color = 'cyan'  # 与plasma色带对比明显
        # ellipse_patch = Ellipse(
        #     xy=flipped_center,
        #     width=2*a,
        #     height=2*b,
        #     angle=np.degrees(flipped_phi),
        #     edgecolor=ellipse_color,
        #     facecolor='none',
        #     lw=1,
        #     linestyle='--',
        #     zorder=10
        # )
        # ax2.add_patch(ellipse_patch)
        # 绘制长轴和短轴（加长到±2a, ±2b）
        x0, y0 = flipped_center
        cos_phi = np.cos(flipped_phi)
        sin_phi = np.sin(flipped_phi)
        # 长轴端点
        x1 = x0 + 3 * a * cos_phi
        y1 = y0 + 3 * a * sin_phi
        x2 = x0 - 3 * a * cos_phi
        y2 = y0 - 3 * a * sin_phi
        ax2.plot([x1, x2], [y1, y2], color=ellipse_color, lw=1, linestyle='--')
        # 长轴方向文字
        major_text_offset = a * 0.04
        ax2.text(x0 + major_text_offset * cos_phi, y0 + major_text_offset * sin_phi, 'major axis',
            color='w', fontsize=8, ha='left', va='top', rotation=np.degrees(flipped_phi))
        # 短轴端点
        x3 = x0 + 3 * b * (-sin_phi)
        y3 = y0 + 3 * b * cos_phi
        x4 = x0 - 3 * b * (-sin_phi)
        y4 = y0 - 3 * b * cos_phi
        ax2.plot([x3, x4], [y3, y4], color=ellipse_color, lw=1, linestyle='--')
        # 短轴方向文字
        minor_text_offset = b * 0.04
        minor_angle = flipped_phi + np.pi/2
        ax2.text(x0 + minor_text_offset * np.cos(minor_angle), y0 + minor_text_offset * np.sin(minor_angle), 'minor axis',
            color='w', fontsize=8, ha='left', va='bottom', rotation=np.degrees(minor_angle))


    # 绘制5um的横线和竖线
    scale_len = 5
    x0, y0 = -18, -18
    ax2.plot([x0, x0 + scale_len], [y0, y0], color='w', lw=1.5)
    ax2.plot([x0, x0], [y0, y0 + scale_len], color='w', lw=1.5)
    ax2.text(x0 + scale_len/2 + 1, y0 + 2.5, r'$5\mathrm{~\mu m}$', 
            color='w', ha='center', va='center', fontsize=10, fontweight='bold')
    ax2.text(x0 + scale_len + 1.5, y0, r'x', color='w', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    ax2.text(x0, y0 + scale_len + 1.5, r'y', color='w', ha='center', va='center', 
            fontsize=10, fontweight='bold')

    # 在右上角标注 z_label
    ax2.text(0.95, 0.95, z_label[1], transform=ax2.transAxes,
            fontsize=12, color='w', ha='right', va='top')

    # 在左上角标注 abc
    ax2.text(0.05, 0.95, fig1_num[1], transform=ax2.transAxes,
            fontsize=14, color='w', ha='left', va='top', fontweight='bold')
    density_imgs.append(im2)


    # 在gs[0, 2]绘制colorbar
    cbar_ax = fig.add_subplot(gs[0, 0])
    cbar = plt.colorbar(density_imgs[0], cax=cbar_ax, orientation='vertical')
    cbar.set_label(r'$n_\mathrm{p}~[\times 10^{19}~\mathrm{cm}^{-3}]$', fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks([0, 0.5, 1.0, 1.5])
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')

    # 保存图像
    plt.savefig('/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/slice_rz/fig6.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)




































    # ex1, ex2, density_peak = plot_3pos()
    # label = ['pulse 1', 'pulse 2', r'$E_\mathrm{z}=0$']
    # z_label = 'z = -0.46 mm'
    # fig1_num = ['d1', 'd2', 'd3']

    # # density_peak 从 nc 转换为 cm^-3
    # for i in range(len(density_peak)):
    #     density_peak[i] = density_peak[i] * 1.74e21 / 1e19
    
    # # 创建图形和网格布局
    # fig = plt.figure(constrained_layout=True, figsize=(6.74, 2.5))
    
    # # 使用更精细的网格布局
    # gs = gridspec.GridSpec(
    #     2, 5,  # 增加一列用于更好的间距控制
    #     height_ratios=[0.15, 1], 
    #     width_ratios=[0.05, 1, 1, 1, 0.05],  # 两侧增加空白列
    #     wspace=0.05,  # 减少子图间水平间距
    #     hspace=0.0,  # 减少子图间垂直间距
    #     figure=fig
    # )

    # # 绘制 ex1 的三幅笛卡尔图
    # ex1_imgs = []
    # ex2_imgs = []
    # vmin_ex1 = 0
    # vmax_ex1 = 20
    # extent = [-20, 20, -20, 20]
    

    # ax = fig.add_subplot(gs[1, 1])  # 第二行，第2-4列
    # im = ax.imshow(
    #     ex1,
    #     origin='lower',
    #     extent=extent,
    #     vmin=vmin_ex1, vmax=vmax_ex1,
    #     cmap='jet',
    #     aspect='equal'
    # )
    # ax.set_xlim(-20, 20)
    # ax.set_ylim(-20, 20)
    # ax.set_xticks([])
    # ax.set_yticks([])
    
    # # 绘制5um的横线和竖线
    # scale_len = 5
    # x0, y0 = -11, -11
    # ax.plot([x0, x0 + scale_len], [y0, y0], color='w', lw=1.5)
    # ax.plot([x0, x0], [y0, y0 + scale_len], color='w', lw=1.5)
    # ax.text(x0 + scale_len/2 + 1, y0 + 2.5, r'$5~\mathrm{~\mu m}$', 
    #         color='w', ha='center', va='center', fontsize=10, fontweight='bold')
    # ax.text(x0 + scale_len + 1.5, y0, r'x', color='w', ha='center', va='center', 
    #         fontsize=10, fontweight='bold')
    # ax.text(x0, y0 + scale_len + 1.5, r'y', color='w', ha='center', va='center', 
    #         fontsize=10, fontweight='bold')
    
    # # 在右上角标注 z_label
    # ax.text(0.95, 0.95, z_label, transform=ax.transAxes,
    #         fontsize=12, color='w', ha='right', va='top')
    
    # # 在左上角标注 abc
    # ax.text(0.05, 0.95, fig1_num[0], transform=ax.transAxes,
    #         fontsize=14, color='w', ha='left', va='top', fontweight='bold')

    # max_idx = np.unravel_index(np.argmax(ex1), ex1.shape)
    # max_x = extent[0] + (extent[1] - extent[0]) * max_idx[1] / ex1.shape[1]
    # max_y = extent[2] + (extent[3] - extent[2]) * max_idx[0] / ex1.shape[0]
    # line_len = 20
    # dx = line_len * np.cos(np.radians(155.7))
    # dy = line_len * np.sin(np.radians(155.7))
    
    # ax.plot([max_x - line_len/2, max_x + line_len/2], [max_y, max_y], 
    #         linestyle='--', color='w', lw=0.5)
    # ax.plot([max_x - dx/2, max_x + dx/2], [max_y - dy/2, max_y + dy/2],
    #         linestyle='--', color='w', lw=0.5)
        
    # arc_radius = 4
    # arc = Arc((max_x, max_y), width=arc_radius*2, height=arc_radius*2,
    #             angle=0, theta1=-24.3, theta2=0, color='w', lw=0.5, linestyle='-')
    # ax.add_patch(arc)
    # ax.text(max_x + 8, max_y+1.5, r'$\phi=24.3°$', color='yellow', fontsize=10, 
    #         ha='center', va='center', fontweight='bold')
    
    # ex1_imgs.append(im)

    # # 为 ex1 添加 colorbar (使用第一列)
    # cbar_ax_ex1 = fig.add_subplot(gs[1, 0])  # 第二行，第一列
    # cbar_ex1 = plt.colorbar(ex1_imgs[0], cax=cbar_ax_ex1, orientation='vertical', extend='max')
    
    # # 设置 ex1 colorbar 的刻度
    # ticks_ex1 = np.linspace(vmin_ex1, vmax_ex1, 5)
    # cbar_ex1.set_ticks(ticks_ex1)
    # ticklabels_ex1 = [f"{np.sqrt(tick):.1f}" for tick in ticks_ex1]
    # cbar_ex1.set_ticklabels(ticklabels_ex1)
    
    # # 设置 ex1 colorbar 标签
    # cbar_ax_ex1.set_ylabel(r'$a_0$', fontsize=12, rotation=90, labelpad=10)
    # cbar_ax_ex1.yaxis.set_label_position('left')
    # cbar_ax_ex1.yaxis.tick_left()
    # cbar_ax_ex1.tick_params(axis='y', direction='in', length=3, pad=2)



    # ax1 = fig.add_subplot(gs[1, 2])  # 第二行，第3列
    # im1 = ax1.imshow(
    #     ex2,
    #     origin='lower',
    #     extent=extent,
    #     vmin=vmin_ex1, vmax=vmax_ex1,
    #     cmap='jet',
    #     aspect='equal'
    # )
    # ax1.set_xlim(-20, 20)
    # ax1.set_ylim(-20, 20)
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    
    # # 绘制5um的横线和竖线
    # scale_len = 5
    # x0, y0 = -11, -11
    # ax1.plot([x0, x0 + scale_len], [y0, y0], color='w', lw=1.5)
    # ax1.plot([x0, x0], [y0, y0 + scale_len], color='w', lw=1.5)
    # ax1.text(x0 + scale_len/2 + 1, y0 + 2.5, r'$5~\mathrm{~\mu m}$', 
    #         color='w', ha='center', va='center', fontsize=10, fontweight='bold')
    # ax1.text(x0 + scale_len + 1.5, y0, r'x', color='w', ha='center', va='center', 
    #         fontsize=10, fontweight='bold')
    # ax1.text(x0, y0 + scale_len + 1.5, r'y', color='w', ha='center', va='center', 
    #         fontsize=10, fontweight='bold')
    
    # # 在右上角标注 z_label
    # ax1.text(0.95, 0.95, z_label, transform=ax1.transAxes,
    #         fontsize=12, color='w', ha='right', va='top')
    
    # # 在左上角标注 abc
    # ax1.text(0.05, 0.95, fig1_num[1], transform=ax1.transAxes,
    #         fontsize=14, color='w', ha='left', va='top', fontweight='bold')

    # max_idx = np.unravel_index(np.argmax(ex2), ex2.shape)
    # max_x = extent[0] + (extent[1] - extent[0]) * max_idx[1] / ex2.shape[1]
    # max_y = extent[2] + (extent[3] - extent[2]) * max_idx[0] / ex2.shape[0]
    # line_len = 20
    # dx = line_len * np.cos(np.radians(155.7))
    # dy = line_len * np.sin(np.radians(155.7))
    
    # ax1.plot([max_x - line_len/2, max_x + line_len/2], [max_y, max_y], 
    #         linestyle='--', color='w', lw=0.5)
    # ax1.plot([max_x - dx/2, max_x + dx/2], [max_y - dy/2, max_y + dy/2],
    #         linestyle='--', color='w', lw=0.5)
        
    # arc_radius = 4
    # arc = Arc((max_x, max_y), width=arc_radius*2, height=arc_radius*2,
    #             angle=0, theta1=-24.3, theta2=0, color='w', lw=0.5, linestyle='-')
    # ax1.add_patch(arc)
    # ax1.text(max_x + 8, max_y+1.5, r'$\phi=24.3°$', color='yellow', fontsize=10,
    #         ha='center', va='center', fontweight='bold')






    # # 绘制 density_peak 的三幅笛卡尔图
    # density_imgs = []
    # vmin_density = 0
    # vmax_density = 2
    

    # ax2 = fig.add_subplot(gs[1, 3])  # 第三行，第2-4列
    # im2 = ax2.imshow(
    #     density_peak,
    #     origin='lower',
    #     extent=extent,
    #     vmin=vmin_density, vmax=vmax_density,
    #     cmap='plasma',
    #     aspect='equal'
    # )
    # ax2.set_xlim(-20, 20)
    # ax2.set_ylim(-20, 20)
    # ax2.set_xticks([])
    # ax2.set_yticks([])
    
    # # 绘制5um的横线和竖线
    # scale_len = 5
    # x0, y0 = -11, -11
    # ax2.plot([x0, x0 + scale_len], [y0, y0], color='w', lw=1.5)
    # ax2.plot([x0, x0], [y0, y0 + scale_len], color='w', lw=1.5)
    # ax2.text(x0 + scale_len/2 + 1, y0 + 2.5, r'$5~\mathrm{~\mu m}$', 
    #         color='w', ha='center', va='center', fontsize=10, fontweight='bold')
    # ax2.text(x0 + scale_len + 1.5, y0, r'x', color='w', ha='center', va='center', 
    #         fontsize=10, fontweight='bold')
    # ax2.text(x0, y0 + scale_len + 1.5, r'y', color='w', ha='center', va='center', 
    #         fontsize=10, fontweight='bold')
    

    # ax2.plot([max_x - line_len/2, max_x + line_len/2], [max_y, max_y], 
    #         linestyle='--', color='w', lw=0.5)
    # ax2.plot([max_x - dx/2, max_x + dx/2], [max_y - dy/2, max_y + dy/2],
    #         linestyle='--', color='w', lw=0.5)
    
    # arc_radius = 4
    # arc = Arc((max_x, max_y), width=arc_radius*2, height=arc_radius*2,
    #             angle=0, theta1=-24.3, theta2=0, color='w', lw=0.5, linestyle='-')
    # ax2.add_patch(arc)
    # ax2.text(max_x + 8, max_y+1.5, r'$\phi=24.3°$', color='yellow', fontsize=10,
    #         ha='center', va='center', fontweight='bold')

    # # 在左上角标注 abc
    # ax2.text(0.05, 0.95, fig1_num[2], transform=ax2.transAxes,
    #         fontsize=14, color='w', ha='left', va='top', fontweight='bold')

    # density_imgs.append(im2)

    # # 为 density 添加 colorbar (使用第一列)
    # cbar_ax_density = fig.add_subplot(gs[1, 4])  # 第三行，第一列
    # cbar_density = plt.colorbar(density_imgs[0], cax=cbar_ax_density, orientation='vertical')
    
    # # 设置 density colorbar 的刻度
    # # cbar_density.set_ticks([1, 2, 3])
    # # cbar_density.set_ticklabels(['1.0', '2.0', '3.0'])
    # cbar_density.set_ticks([0.5, 1.0, 1.5])
    # cbar_density.set_ticklabels(['0.5', '1.0', '1.5'])
    
    # # 设置 density colorbar 标签
    # cbar_ax_density.set_ylabel(r'$n_\mathrm{e}~[\times 10^{19} cm^{-3}]$', 
    #                           fontsize=12, rotation=90, labelpad=10)
    # cbar_ax_density.yaxis.set_label_position('right')
    # cbar_ax_density.yaxis.tick_right()
    # cbar_ax_density.tick_params(axis='y', direction='in', length=3, pad=2)


    # # 在第一行添加标签
    # for i in range(3):
    #     ax = fig.add_subplot(gs[0, i+1])  # 第一行，第2-4列
    #     ax.axis('off')
    #     ax.text(0.5, 0.5, label[i], ha='center', va='center', 
    #             fontsize=14, transform=ax.transAxes)

    # # 保存图像
    # plt.savefig('/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/slice_rz/fig5d_k380r.png', 
    #             dpi=300, bbox_inches='tight')
    # plt.close(fig)