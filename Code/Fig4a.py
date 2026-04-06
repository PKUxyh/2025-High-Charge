#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import os
from matplotlib import colors,ticker,cm
from fld_module import EM
from density_module import plasma
from raw_module import particles
from scipy.signal import savgol_filter
from scipy.signal import hilbert
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter1d
from pre_fig5a import read_pre_fig5_txt
import matplotlib.gridspec as gridspec

matplotlib.font_manager.fontManager.addfont('/public1/home/m8s000916/.fonts/arial/arial.ttf')
matplotlib.font_manager.fontManager.addfont('/public1/home/m8s000916/.fonts/arial/arialbd.ttf')
plt.rcParams['font.family'] = 'Arial'

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
    'figure.figsize': (6.74, 3.8),
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': 'black',
    'legend.framealpha': 1,
    'legend.fontsize': 12,
})
# # #双色
rgb1 = '#3F77A3'
rgb2 = '#E49A5C'
rgb3 = '#EC3232'
rgb_fit = '#5CE49A'


def sqrt_formatter(x, pos):
    return f'{np.sqrt(x):.2f}'




if __name__ == '__main__':
    # 定义三个数据目录
    elli_dirs = [
        '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/slice_rz/',
        '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_gauss/img/slice_rz/',
        '/public1/home/m8s000916/xyh/real_laser/real_astig_elli_6e18_1.2895a0/img/slice_rz/'
    ]
    labels = ['case r', 'case g', 'case ea']
    colors = [rgb1, rgb2, rgb3]

    mv_xlim_osiris = [-706.8, 0]
    y_lim_osiris = [0, 684]
    x_index_range = [0, 3534]
    y_index_range = [0, 684]

    vline_results = []
    z_elli_results = []
    tail_results = []

    for elli_dir, label, color in zip(elli_dirs, labels, colors):
        # 读取数据
        bubble_file = os.path.join(elli_dir, 'pre_fig5.txt')
        time, Ez0_idx, tail_idx = read_pre_fig5_txt(bubble_file)
        time = time[0:44]
        Ez0_idx = Ez0_idx[0:44]
        tail_idx = tail_idx[0:44]
        z_elli = [-2.2 + t / 6.28 * 0.8 * 1e-3 for t in time]
        if label == 'case g':
            vline = ((np.array(Ez0_idx) + 98.8*5) / x_index_range[1] * (mv_xlim_osiris[1]-mv_xlim_osiris[0]) + mv_xlim_osiris[0]) / (2*np.pi)*0.8
            tail = ((np.array(tail_idx) + 98.8*5) / x_index_range[1] * (mv_xlim_osiris[1]-mv_xlim_osiris[0]) + mv_xlim_osiris[0]) / (2*np.pi)*0.8
        else:
            vline = (np.array(Ez0_idx) / x_index_range[1] * (mv_xlim_osiris[1]-mv_xlim_osiris[0]) + mv_xlim_osiris[0]) / (2*np.pi)*0.8
            tail = (np.array(tail_idx) / x_index_range[1] * (mv_xlim_osiris[1]-mv_xlim_osiris[0]) + mv_xlim_osiris[0]) / (2*np.pi)*0.8
        # 保存用于微分的数据
        vline_results.append(vline)
        z_elli_results.append(z_elli)
        tail_results.append(tail)


    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0, 0])
    for z_elli, tail, label, color in zip(z_elli_results, tail_results, labels, colors):
        dtail = np.gradient(tail, z_elli)
        ax0.plot(z_elli, dtail/1e3, label=label, color=color, linestyle='-')
    ax0.axhline(y=-0.0052, color='k', linestyle='--', linewidth=1, label=r'$-v_{\mathrm{etch}}/c$')  # 添加y=0的虚线
    ax0.set_xlabel('z [mm]')
    ax0.set_ylabel(r'$\frac{d\xi_\mathrm{t}}{dz}$')
    ax0.legend(frameon=False)
    ax0.set_ylim([-0.025, 0.005])
    ax0.tick_params(labelbottom=True)  # 显示第一行的 x 轴刻度
    ax0.set_yticks([-0.025, -0.02, -0.015, -0.01, -0.005, 0, 0.005])
    ax0.grid(True, which='major', alpha=0.5, linestyle='--')

    # 在左下角绘制标记 a
    ax0.text(
        0.02, 0.02, 'a',
        transform=ax0.transAxes,
        fontsize=14,
        color='black',
        ha='left',
        va='bottom',
        fontweight='bold'
    )

    img_dir = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/evolution'
    # 保存图像
    plt.tight_layout()
    plt.savefig(img_dir + '/fig5.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    for label, dtail in zip(labels, [np.gradient(tail, z_elli) for z_elli, tail in zip(z_elli_results, tail_results)]):
        print(f"{label}: dtail = {dtail}")


    # fig = plt.figure()
    # gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.1)

    # ax0 = fig.add_subplot(gs[0, 0])
    # ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    # ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)

    # # 第一行：原始线条
    # for z_elli, vline, label, color in zip(z_elli_results, vline_results, labels, colors):
    #     ax0.plot(z_elli, vline, label=label, color=color, linestyle='-')
    #     # 设置第一行标签和图例
    #     ax0.set_ylabel(r'$\xi|_\mathrm{E_z=0} [\mathrm{\mu m}]$')
    # ax0.legend()

    # # 第二行：微分结果
    # for z_elli, vline, label, color in zip(z_elli_results, vline_results, labels, colors):
    #     # 计算微分
    #     dvline = np.gradient(vline, z_elli)
    #     ax1.plot(z_elli, dvline/1e3, label=label, color=color, linestyle='-')

    # ax1.axhline(y=-0.0052, color='gray', linestyle='--', linewidth=1)  # 添加y=0的虚线
    # ax1.set_xlabel('z [mm]')
    # ax1.set_ylabel(r'd$\xi|_\mathrm{E_z=0}/dz [c]$')
    # ax1.set_ylim([-0.025, 0.005])


    # # 第三行：tail速度
    # for z_elli, tail, label, color in zip(z_elli_results, tail_results, labels, colors):
    #     dtail = np.gradient(tail, z_elli)
    #     ax2.plot(z_elli, dtail/1e3, label=label, color=color, linestyle='-')
    # ax2.axhline(y=-0.0052, color='gray', linestyle='--', linewidth=1)  # 添加y=0的虚线
    # ax2.set_xlabel('z [mm]')
    # # 显示第一行和第二行的 x 轴刻度和标签
    # ax0.tick_params(labelbottom=False)  # 显示第一行的 x 轴刻度
    # ax1.tick_params(labelbottom=False)  # 显示第二行的 x 轴刻度

    # img_dir = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/evolution'
    # # 保存图像
    # plt.tight_layout()
    # plt.savefig(img_dir + '/fig5test.png', dpi=300, bbox_inches='tight')
    # plt.close(fig)

    # print(f"图像已保存到: {img_dir}/fig5test.png")