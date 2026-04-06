import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def read_data(file_path):
    # 支持每行多个数字，自动展平成一维list
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            nums = [float(x) for x in line.strip().split() if x]
            data.extend(nums)
    return data

def read_mode_data(file_path):
    data = np.loadtxt(file_path)
    if data.ndim == 1:
        size = int(np.sqrt(data.size))
        data = data[:size*size].reshape((size, size))
    return data



# 设置全局样式
plt.style.use('default')  # 重置为默认样式
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'lines.linewidth': 1.0,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'figure.dpi': 300,
    'figure.figsize': (3.37,3.37/1.6),  # 图像大小
    'xtick.direction': 'in',         # 刻度线朝内
    'ytick.direction': 'in',
    'xtick.major.width': 1,        # 主刻度线粗细
    'ytick.major.width': 1,
    'xtick.major.size': 4,           # 主刻度线长度
    'ytick.major.size': 4,
    'xtick.labelsize': 8,           # 刻度数字大小
    'ytick.labelsize': 8,
})

# #双色 
#     rgb1 = '#3F77A3'
#     rgb2 = '#E49A5C'

# # #三色 
# rgb1 = '#480080'
# rgb2 = '#e23c5d'
# rgb3 = '#ffb42c'

# rgb4 = '#501d8a'
# rgb5 = '#aa3474'
# rgb6 = '#ee8c7d'

# #四色
rgb1 = '#A82E25'
rgb2 = '#eb7e35'
rgb3 = '#6c8735'
rgb4 = '#505050'


if __name__ == "__main__":
    psnr_path = r'F:\paper\GS_new\GSA-MD\paper_used\后处理\importOSIRIS\PSNR.txt'
    ssim_path = r'F:\paper\GS_new\GSA-MD\paper_used\后处理\importOSIRIS\ssim.txt'
    pearson_path = r'F:\paper\GS_new\GSA-MD\paper_used\后处理\importOSIRIS\pearson.txt'
    relative_path = r'F:\paper\GS_new\GSA-MD\paper_used\后处理\importOSIRIS\Relative.txt'

    mode2_path = r'F:\paper\GS_new\GSA-MD\paper_used\后处理\importOSIRIS\mode2.txt'
    mode4_path = r'F:\paper\GS_new\GSA-MD\paper_used\后处理\importOSIRIS\mode4.txt'
    mode7_path = r'F:\paper\GS_new\GSA-MD\paper_used\后处理\importOSIRIS\mode7.txt'
    mode20_path = r'F:\paper\GS_new\GSA-MD\paper_used\后处理\importOSIRIS\mode20.txt'

    psnr = read_data(psnr_path)
    ssim = read_data(ssim_path)
    pearson = read_data(pearson_path)
    relative = read_data(relative_path)

    mode2 = read_mode_data(mode2_path)
    mode4 = read_mode_data(mode4_path)
    mode7 = read_mode_data(mode7_path)
    mode20 = read_mode_data(mode20_path)

    x = np.arange(1, 21)

    fig, ax1 = plt.subplots()

    # 左y轴：PSNR, Relative
    l1, = ax1.plot(x, psnr, label='PSNR', color=rgb1)
    l2, = ax1.plot(x, relative, label='Relative error', color=rgb4)
    ax1.set_xlabel('Modes number')
    ax1.set_ylabel('PSNR [dB] / Relative [%]')
    ax1.set_xlim(1, 20)
    ax1.set_ylim(0, 100)
    ax1.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

    # 右y轴：SSIM, Pearson
    ax2 = ax1.twinx()
    l3, = ax2.plot(x, ssim, label='SSIM', color=rgb2)
    l4, = ax2.plot(x, pearson, label='Pearson', color=rgb3)
    ax2.set_ylabel('SSIM [1] / Pearson [1]')
    ax2.set_ylim(0, 1)

    # 合并图例
    lines = [l1, l2, l3, l4]
    labels = [line.get_label() for line in lines]
    
    legend = ax1.legend(
        lines,
        labels,
        loc='lower right',
        frameon=True,
        fancybox=False,
        edgecolor='w',
        framealpha=0,
        fontsize=6.5,
        # ncol=3,  # 设置为两列
        borderpad=0.3,      # 边框与内容的距离
        labelspacing=0.3,   # 标签之间的垂直间距
        handletextpad=0.3,  # 线条和标签之间的距离
        columnspacing=0.3,   # 列之间的间距
        bbox_to_anchor=(0.95, 0.015),  # 调整图例离坐标轴的距离 (右侧, 下侧)
        )
    

    # 插入小图
    mode_indices = [2, 4, 7, 20]
    mode_datas = [mode2, mode4, mode7, mode20]
    y_anchor = [psnr[0], psnr[6], psnr[12], psnr[19]]

    for i, (idx, mode_data) in enumerate(zip(mode_indices, mode_datas)):
        # 让小图贴近箭头，调整bbox_to_anchor和箭头终点
        x_anchor = -0.32 + 0.23 * i
        y_anchor_img = 0.08
        inset = inset_axes(ax1, width="22%", height="22%", 
                           loc='center', 
                           bbox_to_anchor=(x_anchor, y_anchor_img, 1, 1),
                           bbox_transform=ax1.transAxes, borderpad=0)
        im = inset.imshow(np.rot90(mode_data.T), cmap='jet', aspect='auto')
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title(f'{idx} modes ', fontsize=6.5)


    ax1.grid(True, which='major', axis='both', linestyle='--', color='#bbbbbb', linewidth=0.5, alpha=0.5, dashes=(8, 8))
    plt.tight_layout()
    # fig.savefig(r'F:\paper\GS_new\GSA-MD\paper_used\后处理\importOSIRIS\模式数对重建电场的影响.png', dpi=300, bbox_inches='tight')
    plt.show()
   


