from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit

# 添加字体目录
font_path = '/public1/home/m8s000916/.conda/envs/py39/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf'
plt.rcParams["font.family"] = ["Arial", "sans-serif"]
plt.rcParams["font.serif"] = ["Arial"]

def read_data(file_path):
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file]
    return data

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


if __name__ == "__main__":
    simi_file = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/ellipse_analysis/similarity_coeffs.txt'
    simi_coeffs = read_data(simi_file)

    time_file = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/ellipse_analysis/time.txt'
    time = read_data(time_file)
    z_plot = np.array(time) / (2*np.pi) * 0.8e-3 - 2.2
    # 对simi_coeffs进行低通滤波
    b, a = butter(N=3, Wn=0.1, btype='low')
    simi_fit = filtfilt(b, a, simi_coeffs)
    z_fit = z_plot

    # 创建1行3列的GridSpec布局
    fig = plt.figure()
    gs = GridSpec(1, 3, width_ratios=[0.1, 1, 0.01], figure=fig, wspace=0)

    # 第1个子图
    ax1 = fig.add_subplot(gs[0, 1])
    # ax1.plot(z_fit[:60], simi_fit[:60], color=rgb1, linestyle='-', lw=1.5)
    ax1.scatter(z_plot[:60], simi_coeffs[:60], color=rgb1, marker='x', s=20)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlabel('z [mm]')
    ax1.set_ylabel('Pearson')
    ax1.text(0.05, 0.95, 'a', transform=ax1.transAxes,
            fontsize=12, color='k', ha='left', va='top', fontweight='bold')
    plt.tight_layout()
    plt.show()
    fig.savefig('/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/ellipse_analysis/pearson.png', dpi=300)