import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import pandas as pd


def read_csv_files(path, target_type):
    df = pd.read_csv(path)
    data = df.values
    data = data[:, :-1]
    if target_type == "complex":
        data = data.astype(np.complex128)  # 如果是复数
    else:
        data = data.astype(np.float64)  # 如果是实数
    return data

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

def calculate_ellipse_integral(intensity, grid_size, rx, ry, theta_deg):
    """
    以最大值为中心，绘制一个椭圆（给定x/y轴半径和长轴与x轴夹角），计算椭圆内的积分。

    参数:
    intensity: 2D强度矩阵
    grid_size: 每个像素的物理尺寸 (μm)
    rx: 椭圆x轴半径 (μm)
    ry: 椭圆y轴半径 (μm)
    theta_deg: 椭圆长轴与x轴的夹角 (度)

    返回:
    ellipse_integral: 椭圆内积分
    total_integral: 总积分
    mask: 椭圆掩码
    """
    max_idx = np.unravel_index(np.argmax(intensity), intensity.shape)
    cy, cx = max_idx

    theta = np.deg2rad(theta_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    h, w = intensity.shape
    y, x = np.indices((h, w))
    x_phys = (x - cx) * grid_size
    y_phys = (y - cy) * grid_size

    # 椭圆方程：(x', y')为旋转后的坐标
    x_rot = x_phys * cos_t + y_phys * sin_t
    y_rot = -x_phys * sin_t + y_phys * cos_t
    ellipse_mask = (x_rot / rx) ** 2 + (y_rot / ry) ** 2 <= 1

    area_per_pixel = grid_size ** 2
    ellipse_integral = np.sum(intensity[ellipse_mask]) * area_per_pixel
    total_integral = np.sum(intensity) * area_per_pixel
    # 绘制示意图
    fig, ax = plt.subplots()
    im = ax.imshow(intensity, cmap='jet', origin='lower')
    plt.colorbar(im, ax=ax, label='Intensity')
    ax.contour(ellipse_mask, levels=[0.5], colors='w', linewidths=2)
    ax.set_title('Ellipse Integral Region')
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    plt.show()
    return ellipse_integral, total_integral, ellipse_mask




if __name__ == "__main__":
    case_r_path = r"F:\paper\GS_new\GSA-MD\paper_used\后处理\after_zernike_far_field_E.csv"
    case_r = read_csv_files(case_r_path, target_type="complex")
    case_r_intensity = np.abs(case_r) ** 2
    case_r_intensity = case_r_intensity * (1.44 ** 2) / np.max(case_r_intensity)
    case_ea_path = r"F:\paper\GS_new\asym_laser_import\real_astigmatic_ellipicity_intensity.csv"
    case_ea_intensity = read_csv_files(case_ea_path, target_type="float")
    case_ea_intensity = case_ea_intensity * (1.44 ** 2) / np.max(case_ea_intensity)

    case_r_ellipse_integral, case_r_total, case_r_mask = calculate_ellipse_integral(case_r_intensity, 0.88, 22.4/2, 29.3/2, 24.3)
    print(f"Case R ellipse integral: {case_r_ellipse_integral}, total integral: {case_r_total}, percentage: {case_r_ellipse_integral/case_r_total*100:.2f}%")   
    case_ea_ellipse_integral, case_ea_total, case_ea_mask = calculate_ellipse_integral(case_ea_intensity, 0.44, 22.4/2, 29.3/2, 24.3)
    print(f"Case EA ellipse integral: {case_ea_ellipse_integral}, total integral: {case_ea_total}, percentage: {case_ea_ellipse_integral/case_ea_total*100:.2f}%")

    # base_path = r"F:\paper\GS_new\GSA-MD\paper_used\后处理\after_zernike_far_field_E.csv"
    # base = read_csv_files(base_path, target_type="complex")
    # base_intensity = np.abs(base) ** 2
    # peak_value = np.max(base_intensity)
    # base_intensity = base_intensity * (1.44 ** 2) / peak_value
    # plt.imshow(base_intensity, cmap='jet', origin='lower')
    # plt.colorbar(label='Intensity')
    # plt.title('Base Intensity Pseudocolor Map')
    # plt.xlabel('X Pixel')
    # plt.ylabel('Y Pixel')
    # plt.show()
    # base_grid_size = 0.88  # 假设每个像素的物理尺寸为 0.88 μm
    # base_total_intensity = integrate_intensity(base_intensity, base_grid_size)
    # print(f"Base total intensity: {base_total_intensity}")


    # test_path = r"F:\paper\GS_new\asym_laser_import\astigmatic_real_intensity_far.csv"
    # test_intensity = read_csv_files(test_path, target_type="float")
    # # 计算test的归一化因子，使得积分后总强度与base一致
    # test_peak_value = np.max(test_intensity)
    # # 计算归一化因子a0，使得积分后总强度一致
    # def find_a0(base_total, test_intensity, grid_size, test_peak):
    #     # 归一化后：test_intensity * (a0 ** 2) / test_peak
    #     # 积分后：np.sum(test_intensity) * (a0 ** 2) / test_peak * grid_size**2 == base_total
    #     # 解a0
    #     sum_test = np.sum(test_intensity)
    #     a0 = np.sqrt(base_total * test_peak / (sum_test * grid_size ** 2))
    #     return a0

    # test_grid_size = 0.22  # 与base一致
    # a0 = find_a0(base_total_intensity, test_intensity, test_grid_size, test_peak_value)
    # test_intensity_norm = test_intensity * (a0 ** 2) / test_peak_value

    # plt.imshow(test_intensity_norm, cmap='jet', origin='lower')
    # plt.colorbar(label='Intensity')
    # plt.title('Test Intensity Pseudocolor Map (Normalized)')
    # plt.xlabel('X Pixel')
    # plt.ylabel('Y Pixel')
    # plt.show()

    # test_total_intensity = integrate_intensity(test_intensity_norm, test_grid_size)
    # print(f"Test total intensity (normalized): {test_total_intensity}")
    # print(f"a0 for test: {a0}")

    # # 示例调用 calculate_percentage_within_threshold
    # base_percentage = calculate_percentage_within_threshold(base_intensity, base_grid_size)
    # print(f"Base intensity percentage within 1/e² threshold: {base_percentage:.2f}%")

    # test_percentage = calculate_percentage_within_threshold(test_intensity_norm, test_grid_size)
    # print(f"Test intensity percentage within 1/e² threshold: {test_percentage:.2f}%")

