from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import scipy.io


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
    'figure.figsize': (6.74, 6),
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


# 文件路径
base_path = r'F:\paper\大电量电子加速文章2\chukou\20220608'
charge_file = f'{base_path}\\charge_vs_pressure.txt'
energy_file = f'{base_path}\\energy_vs_pressure.txt'
maxenergy_file = f'{base_path}\\maxenergy_vs_pressure.txt'
spread_file = f'{base_path}\\spread_vs_pressure.txt'

# 读取数据，指定分隔符为制表符
charge_data = np.loadtxt(charge_file, delimiter='\t')
energy_data = np.loadtxt(energy_file, delimiter='\t')
maxenergy_data = np.loadtxt(maxenergy_file, delimiter='\t')
spread_data = np.loadtxt(spread_file, delimiter='\t')

# 打印数据形状以确认读取成功
print('charge_vs_pressure:', charge_data.shape)
print('energy_vs_pressure:', energy_data.shape)
print('maxenergy_vs_pressure:', maxenergy_data.shape)
print('spread_vs_pressure:', spread_data.shape)

# # #双色
rgb1 = '#3F77A3'
rgb2 = '#E49A5C'
rgb3 = '#EC3232'
rgb4 = '#5CE49A'
rgb5 = '#1399B2'




fig = plt.figure(constrained_layout=True)
gs = GridSpec(2, 4, figure=fig, height_ratios=[1.2, 1], width_ratios=[0.05, 1, 1, 0.1], hspace=0.05, wspace=0.1)

ax1 = fig.add_subplot(gs[0, 1:3])


ax1.errorbar(
    charge_data[1:-1, 0], charge_data[1:-1, 1],
    yerr=charge_data[1:-1, 2] if charge_data.shape[1] > 2 else None,
    fmt='o-', color=rgb1, markerfacecolor='none', capsize=2, markersize=5, label='experiments'
)
ax1.plot(6, 249, marker='^', markerfacecolor='none', color=rgb3, markersize=4, label='simulation with a realistic laser')
# ax1.plot(6.1, 417, marker='s', markerfacecolor='none', color=rgb2, markersize=4, label='case ea')
# 连线 case g 的三个点
# ax1.plot([4, 6, 8], [451, 533, 473], color=rgb2, linestyle='--', linewidth=1)
# ax1.plot(6, 533, marker='s', markerfacecolor='none', color=rgb2, markersize=4, label='simulations')
# ax1.plot(4, 451, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
# ax1.plot(8, 473, marker='s', markerfacecolor='none', color=rgb2, markersize=4)

ax1.plot([3.7, 4, 4.7, 5.5, 6, 6.1, 6.7, 7.3, 8], [300, 451, 671, 623, 533, 567, 625, 587, 473], color=rgb2, linestyle='--', linewidth=1)
ax1.plot(6, 533, marker='s', markerfacecolor='none', color=rgb2, markersize=4, label='simulations with a Gaussian laser')
ax1.plot(3.7, 300, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax1.plot(4, 451, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax1.plot(4.7, 671, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax1.plot(5.5, 623, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax1.plot(6.1, 567, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax1.plot(6.7, 625, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax1.plot(7.3, 587, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax1.plot(8, 473, marker='s', markerfacecolor='none', color=rgb2, markersize=4)

ax1.legend(loc='center', ncol=1, frameon=False, handlelength=1.2, columnspacing=0.3, handletextpad=0.4, borderpad=0.4, labelspacing=0.3)
ax1.set_xlabel(r'$n_\mathrm{p}$ [$\times10^{18}$ cm$^{-3}$]')
ax1.set_ylabel(r'$Q$ [pC]')
ax1.set_ylim([0, 700])
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.text(
    0.025, 0.8, 'a',
    transform=ax1.transAxes,
    fontsize=14,
    color='black',
    ha='left',
    va='bottom',
    fontweight='bold'
)


ax2 = fig.add_subplot(gs[1, 1])
ax2.errorbar(
    energy_data[1:-1, 0], energy_data[1:-1, 1],
    yerr=energy_data[1:-1, 2] if energy_data.shape[1] > 2 else None,
    fmt='o-', color=rgb1, markerfacecolor='none', capsize=2, label='experiments', markersize=5,
)
# ax2.errorbar(
#     maxenergy_data[1:-1, 0], maxenergy_data[1:-1, 1],
#     yerr=maxenergy_data[1:-1, 2] if maxenergy_data.shape[1] > 2 else None,
#     fmt='o--', color=rgb2, markerfacecolor='none', capsize=2, label=r'$E_\mathrm{cutoff}$', markersize=5,
# )
ax2.plot(6, 202, marker='^', markerfacecolor='none', color=rgb3, markersize=4, label='real',zorder=100)
# ax2.plot(6.1, 242, marker='o', markerfacecolor='none', color=rgb3, markersize=4, label=r'$E_\mathrm{sim_{cut}}$',zorder=100)

# [3.7, 4, 4.7, 5.5, 6, 6.1, 6.7, 7.3, 8] 
# [300, 451, 671, 623, 533, 567, 625, 587, 473]
# [177.5, 216, 254.9, 224.8, 251, 259.2, 160.3, 156.0, 171]
# [121.0, 109, 68.5, 71.0, 66.3, 121.1, 113.7, 106.1, 93.0]

ax2.plot([3.7, 4, 4.7, 5.5, 6, 6.1, 6.7, 7.3, 8], [177.5, 216, 254.9, 224.8, 251, 259.2, 160.3, 156.0, 171], color=rgb2, linestyle='--', linewidth=1)
ax2.plot(4, 216, marker='s', markerfacecolor='none', color=rgb2, markersize=4, label='simulations')
ax2.plot(3.7, 177.5, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax2.plot(4.7, 254.9, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax2.plot(5.5, 224.8, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax2.plot(6, 251, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax2.plot(6.1, 259.2, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax2.plot(6.7, 160.3, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax2.plot(7.3, 156.0, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax2.plot(8, 171, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
# ax2.legend(loc='upper right', ncol=1, frameon=False, handlelength=1.2, columnspacing=0.3, handletextpad=0.4, borderpad=0.4, labelspacing=0.3)
ax2.set_xlabel(r'$n_\mathrm{p}$ [$\times10^{18}$ cm$^{-3}$]')
ax2.set_ylabel(r'$E$ [MeV]')
ax2.set_ylim([0, 300])
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.text(
    0.05, 0.8, 'b',
    transform=ax2.transAxes,
    fontsize=14,
    color='black',
    ha='left',
    va='bottom',
    fontweight='bold'
)


ax3 = fig.add_subplot(gs[1, 2])
ax3.errorbar(
    spread_data[1:-1, 0], spread_data[1:-1, 1],
    yerr=spread_data[1:-1, 2] if spread_data.shape[1] > 2 else None,
    fmt='o-', color=rgb1, markerfacecolor='none', capsize=2, markersize=5, label='experiments'
)
ax3.set_xlabel(r'$n_\mathrm{p}$ [$\times10^{18}$ cm$^{-3}$]')
ax3.set_ylabel(r'$\Delta E/E$ [%]')
ax3.plot(6, 57, marker='^', color=rgb3, markersize=4, label='real',zorder=100,markerfacecolor='none')
# ax3.plot([3.7, 4, 4.7, 5.5, 6, 6.1, 6.7, 7.3, 8], [121.0, 109, 68.5, 71.0, 66.3, 121.1, 113.7, 106.1, 93.0], color=rgb2, linestyle='--', linewidth=1)
# ax3.plot(4, 109, marker='s', markerfacecolor='none', color=rgb2, markersize=4, label='simulations')
# ax3.plot(3.7, 121.0, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
# ax3.plot(4.7, 68.5, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
# ax3.plot(5.5, 71.0, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
# ax3.plot(6, 66.3, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
# ax3.plot(6.1, 121.1, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
# ax3.plot(6.7, 113.7, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
# ax3.plot(7.3, 106.1, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
# ax3.plot(8, 93.0, marker='s', markerfacecolor='none', color=rgb2, markersize=4)

# ax3.plot([3.7, 4, 4.7, 5.5, 6, 6.1, 6.7, 7.3, 8], [93.1, 92.5, 61.2, 74.4, 44.7, 47.4, 122.4, 64.6, 93.3], color=rgb2, linestyle='--', linewidth=1)
# ax3.plot(4, 92.5, marker='s', markerfacecolor='none', color=rgb2, markersize=4, label='simulations')
# ax3.plot(3.7, 93.1, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
# ax3.plot(4.7, 61.2, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
# ax3.plot(5.5, 74.4, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
# ax3.plot(6, 44.7, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
# ax3.plot(6.1, 47.4, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
# ax3.plot(6.7, 122.4, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
# ax3.plot(7.3, 64.6, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
# ax3.plot(8, 93.3, marker='s', markerfacecolor='none', color=rgb2, markersize=4) 



ax3.plot([3.7, 4, 4.7, 5.5, 6, 6.1, 6.7, 7.3, 8], [103.9, 109.5, 61.8, 69.3, 58.7, 50.2, 100, 83.5, 82.5], color=rgb2, linestyle='--', linewidth=1)
ax3.plot(4, 109.5, marker='s', markerfacecolor='none', color=rgb2, markersize=4, label='simulations')
ax3.plot(3.7, 103.9, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax3.plot(4.7, 61.8, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax3.plot(5.5, 69.3, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax3.plot(6, 58.7, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax3.plot(6.1, 50.2, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax3.plot(6.7, 100, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax3.plot(7.3, 83.5, marker='s', markerfacecolor='none', color=rgb2, markersize=4)
ax3.plot(8, 82.5, marker='s', markerfacecolor='none', color=rgb2, markersize=4)     




# ax3.set_ylim([0, 50])
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.text(
    0.05, 0.8, 'c',
    transform=ax3.transAxes,
    fontsize=14,
    color='black',
    ha='left',
    va='bottom',
    fontweight='bold'
)
# ax3.legend(loc='upper right', ncol=1, frameon=False, handlelength=1.2, columnspacing=0.3, handletextpad=0.4, borderpad=0.4, labelspacing=0.3)

plt.show()
fig.savefig(os.path.join(base_path, 'fig2new.png'), bbox_inches='tight', dpi=300)


