from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager

matplotlib.font_manager.fontManager.addfont('/public1/home/m8s000916/.fonts/arial/arial.ttf')
matplotlib.font_manager.fontManager.addfont('/public1/home/m8s000916/.fonts/arial/arialbd.ttf')
plt.rcParams['font.family'] = 'Arial'





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
    'figure.figsize': (6.74, 9),
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
rgb4 = '#5CE49A'
# 三色
# rgb1 = '#480080'
# rgb2 = '#e23c5d'
# rgb3 = '#ffb42c'

# rgb1 = '#501d8a'
# rgb2 = '#aa3474'
# rgb3 = '#ee8c7d'

# #四色
# rgb1 = '#A82E25'
# rgb2 = '#eb7e35'
# rgb3 = '#6c8735'
# rgb4 = '#505050'

def read_ellipse_file(file_path):
    """
    读取椭圆参数文件，支持包含 NaN 占位符的行。

    参数:
    file_path: 文件路径

    返回:
    包含 'a', 'b', 'e', 'o', 'vline' 的字典，每个键对应一个列表
    """
    data = {'a': [], 'b': [], 'e': [], 'o': [], 'vline': [], 'max': []}
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在！")
        return data
    with open(file_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) == 5:
                try:
                    # 将 NaN 占位符转换为 numpy.nan
                    a = float(values[0]) if values[0] != 'NaN' else np.nan
                    b = float(values[1]) if values[1] != 'NaN' else np.nan
                    e = float(values[2]) if values[2] != 'NaN' else np.nan
                    o = float(values[3]) if values[3] != 'NaN' else np.nan
                    vline = float(values[4]) if values[4] != 'NaN' else np.nan
                    data['a'].append(a)
                    data['b'].append(b)
                    data['e'].append(e)
                    data['o'].append(o)
                    data['vline'].append(vline)
                except ValueError:
                    # 跳过无法解析的行
                    print(f"无法解析的行: {line.strip()}")
                    continue
            elif len(values) == 6:
                try:
                    # 将 NaN 占位符转换为 numpy.nan
                    a = float(values[0]) if values[0] != 'NaN' else np.nan
                    b = float(values[1]) if values[1] != 'NaN' else np.nan
                    e = float(values[2]) if values[2] != 'NaN' else np.nan
                    o = float(values[3]) if values[3] != 'NaN' else np.nan
                    vline = float(values[4]) if values[4] != 'NaN' else np.nan
                    max_val = float(values[5]) if values[5] != 'NaN' else np.nan
                    data['a'].append(a)
                    data['b'].append(b)
                    data['e'].append(e)
                    data['o'].append(o)
                    data['vline'].append(vline)
                    data['max'].append(max_val)
                except ValueError:
                    # 跳过无法解析的行
                    print(f"无法解析的行: {line.strip()}")
                    continue
    return data

def read_charge_evolution(file_path):
    """
    读取 charge_evolution 文件中的数据。

    参数:
    file_path: 文件路径

    返回:
    z: z[mm] 的数据列表
    charge: Q_injected[pC] 的数据列表
    """
    z = []
    charge = []
    try:
        with open(file_path, 'r') as file:
            # 跳过文件头部
            header = file.readline()
            for line in file:
                try:
                    values = line.strip().split()
                    z.append(float(values[0]))
                    charge.append(float(values[1]))
                except (ValueError, IndexError):
                    print(f"无法解析的行: {line.strip()}")
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在！")
    return z, charge

def read_last_hist2d_data(file_path):
    """
    读取保存的二维直方图数据文件中的最后一个时刻的二维数据和其坐标轴。

    参数:
    file_path: 保存的 npz 文件路径

    返回:
    hist2d_last: 最后一个时刻的二维直方图数据
    energy_edges: 能量坐标轴
    angle_edges: 散射角坐标轴
    """
    try:
        with np.load(file_path) as data:
            hist2d = data['hist2d']  # 获取所有时刻的二维直方图
            energy_edges = data['energy_edges']  # 获取能量坐标轴
            angle_edges = data['angle_edges']  # 获取散射角坐标轴

            # 获取最后一个时刻的二维直方图
            hist2d_last = hist2d[-1]

        return hist2d_last, energy_edges, angle_edges
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在！")
        return None, None, None
    except KeyError as e:
        print(f"文件中缺少关键数据: {e}")
        return None, None, None
    

# 读取csv文件
def read_csv_data(file_path, delimiter=',', skip_header=0):
    """
    读取csv文件，返回numpy数组。
    参数:
        file_path: 文件路径
        delimiter: 分隔符，默认为逗号
        skip_header: 跳过的表头行数
    返回:
        data: numpy数组
    """
    try:
        # 先读取一行，判断是否包含复数
        with open(file_path, 'r') as f:
            for _ in range(skip_header):
                next(f)
            first_line = f.readline()
        # 判断是否包含 'j' 或 'J'，即复数
        if 'j' in first_line or 'J' in first_line:
            data = np.genfromtxt(file_path, delimiter=delimiter, skip_header=skip_header, dtype=complex)
        else:
            data = np.genfromtxt(file_path, delimiter=delimiter, skip_header=skip_header)
        return data
    except Exception as e:
        print(f"读取CSV文件出错: {e}")
        return None
        
# 调用示例
if __name__ == "__main__":
    plot_time_real = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/evolution/time.txt'
    plot_time_ea = '/public1/home/m8s000916/xyh/real_laser/real_astig_elli_6e18_1.2895a0/img/evolution/time.txt'
    plot_time_gauss = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_gauss/img/evolution/time.txt'

    real_laser_a0 = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/evolution/a0_deflection.txt'
    ea_laser_a0 = '/public1/home/m8s000916/xyh/real_laser/real_astig_elli_6e18_1.2895a0/img/evolution/a0_deflection.txt'
    gauss_laser_a0 = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_gauss/img/evolution/a0_deflection.txt'

    # real_laser_radius_x = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/evolution/laser_spot_x.txt'
    # real_laser_radius_y = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/evolution/laser_spot_y.txt'
    # ea_laser_radius_x = '/public1/home/m8s000916/xyh/real_laser/real_astig_elli_6e18_1.2895a0/img/evolution/laser_spot_x.txt'
    # ea_laser_radius_y = '/public1/home/m8s000916/xyh/real_laser/real_astig_elli_6e18_1.2895a0/img/evolution/laser_spot_y.txt'
    # gauss_laser_radius_x = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_gauss/img/evolution/laser_spot_x.txt'
    # gauss_laser_radius_y = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_gauss/img/evolution/laser_spot_y.txt'
    real_laser_charge = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/evolution/charge_evolution_70MeV.txt'
    ea_laser_charge = '/public1/home/m8s000916/xyh/real_laser/real_astig_elli_6e18_1.2895a0/img/evolution/charge_evolution_70MeV.txt'
    gauss_laser_charge = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_gauss/img/evolution/charge_evolution_70MeV.txt'

    real_laser_hist2d = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/evolution/energy_angle_hist2d.npz'
    ea_laser_hist2d = '/public1/home/m8s000916/xyh/real_laser/real_astig_elli_6e18_1.2895a0/img/evolution/energy_angle_hist2d.npz'
    gauss_laser_hist2d = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_gauss/img/evolution/energy_angle_hist2d.npz'


    ea_focus = '/public1/home/m8s000916/xyh/real_laser/ea_focus.csv'
    gauss_focus = '/public1/home/m8s000916/xyh/real_laser/gauss_focus.csv'
    real_focus = '/public1/home/m8s000916/xyh/real_laser/after_zernike_far_field_E.csv'
    ea_focus_data = read_csv_data(ea_focus)
    gauss_focus_data = read_csv_data(gauss_focus)
    real_focus_data = read_csv_data(real_focus)
    ea_focus_data = np.abs(ea_focus_data)
    gauss_focus_data = np.abs(gauss_focus_data)
    real_focus_data = np.abs(real_focus_data)**2

    ea_focus_data = ea_focus_data / np.max(ea_focus_data) * (1.44 ** 2)
    gauss_focus_data = gauss_focus_data / np.max(gauss_focus_data) * (1.44 ** 2)
    real_focus_data = real_focus_data / np.max(real_focus_data) * (1.44 ** 2)

    # 读取数据
    time_real = read_data(plot_time_real)
    time_gauss = read_data(plot_time_gauss)
    time_ea = read_data(plot_time_ea)

    a0_real = read_data(real_laser_a0)
    a0_gauss = read_data(gauss_laser_a0)
    a0_ea = read_data(ea_laser_a0)

    # radius_real_x = read_data(real_laser_radius_x)
    # radius_real_y = read_data(real_laser_radius_y)
    # radius_gauss_x = read_data(gauss_laser_radius_x)
    # radius_gauss_y = read_data(gauss_laser_radius_y)
    # radius_ea_x = read_data(ea_laser_radius_x)
    # radius_ea_y = read_data(ea_laser_radius_y)

    _, charge_real = read_charge_evolution(real_laser_charge)
    _, charge_gauss = read_charge_evolution(gauss_laser_charge)
    _, charge_ea = read_charge_evolution(ea_laser_charge)
    # 读取最后一个时刻的二维直方图数据
    hist2d_real, energy_edges, angle_edges = read_last_hist2d_data(real_laser_hist2d)
    hist2d_ea, _, _ = read_last_hist2d_data(ea_laser_hist2d)
    hist2d_gauss, _, _ = read_last_hist2d_data(gauss_laser_hist2d)


    z_real = (np.array(time_real)-17262.5) / 2 / np.pi * 0.8e-3
    z_gauss = (np.array(time_gauss)-17262.5) / 2 / np.pi * 0.8e-3
    z_ea = (np.array(time_ea)-17262.5) / 2 / np.pi * 0.8e-3


    bg_density_x = (np.array([0.0, 1562.5, 32962.5, 34525, 36000])-17262.5) / 2 / np.pi * 0.8e-3
    bg_density = np.array([0. , 0.0035 ,  0.0035 , 0, 0])


    # 创建两个子图，共用x轴
    fig = plt.figure()
    gs = fig.add_gridspec(8, 4, height_ratios=[3.2, 0.3, 2.5, 2.5, 2.5, 2.5, 0.95, 3], hspace=0, width_ratios=[0.1, 1, 1, 1])

    #第一行
    # 绘制三个光斑图（无title、无刻度、无坐标轴）
    focus_data_list = [
        (real_focus_data, 'case r', rgb1),
        (gauss_focus_data, 'case g', rgb2),
        (ea_focus_data, 'case ea', rgb3)
    ]
    for i, (focus_data, label, color) in enumerate(focus_data_list):
        ax = fig.add_subplot(gs[0, i+1])
        im = ax.imshow(focus_data, aspect='equal', cmap='jet')
        ax.axis('off')
        ax.text(0.05, 0.85, f'a{i+1}', transform=ax.transAxes, fontweight='bold', fontsize=14, color='white')
        ax.text(0.55, 0.85, label, transform=ax.transAxes, fontsize=12, color='white')

        # 添加尺寸和角度信息
        focus_texts = [
            (r'$w_{\mathrm{x}}$, $w_{\mathrm{y}}=$22.4, 29.3 $\mathrm{\mu}$m'),
            (r'$w_{\mathrm{x}}$, $w_{\mathrm{y}}=$25.9, 25.9 $\mathrm{\mu}$m'),
            (r'$w_{\mathrm{x}}$, $w_{\mathrm{y}}=$22.4, 29.3 $\mathrm{\mu}$m')
        ]
        ax.text(0.02, 0.05, focus_texts[i], transform=ax.transAxes, fontsize=10, color='white', va='bottom')

        # 在左侧高度居中处绘制20um横线的左段和竖线的下段相交，并标注20um
        ny, nx = focus_data.shape
        if i == 0:
            um_per_pix = 0.88  # 图1的um_per_pix是0.88
            margin_pix = 10  # 距离左侧的像素
        else:
            um_per_pix = 0.44
            margin_pix = 20  # 距离左侧的像素
        line_len_um = 40
        line_len_pix = line_len_um / um_per_pix

        cy = ny // 2  # 高度居中
        # 竖线：中心在图片中心，长度为line_len_pix，整体下移10像素
        y_shift = 40
        if i == 0:
            y_shift = y_shift//2  # 图1的竖线下移10像素
        # 竖线
        ax.plot([margin_pix, margin_pix], [cy - line_len_pix/2 + y_shift, cy + line_len_pix/2 + y_shift], color='white', lw=1.5)
        # 横线
        ax.plot([margin_pix, margin_pix + line_len_pix], [cy + line_len_pix/2 + y_shift, cy + line_len_pix/2 + y_shift], color='white', lw=1.5)
        # 标注40um
        ax.text(margin_pix + line_len_pix / 1.4, cy - 8 + y_shift, r'40 $\mathrm{\mu m}$', color='white', fontsize=10, ha='center', va='top')
        # 横线右侧标记x
        ax.text(margin_pix + line_len_pix + 5, cy + line_len_pix/2 + y_shift, 'x', color='white', fontsize=12, ha='left', va='center', fontweight='bold')
        # 竖线上侧标记y
        ax.text(margin_pix, cy - line_len_pix/2 + y_shift - 8, 'y', color='white', fontsize=12, ha='center', va='bottom', fontweight='bold')

        # 只在a1和a3加虚线和夹角标记
        if i in [0, 2]:
            # 获取图像中心
            ny, nx = focus_data.shape
            cx, cy = nx // 2, ny // 2
            # 线长（单位：像素），假设每像素0.44um
            um_per_pix = 0.44
            # a1用30um, a3用60um
            if i == 0:
                line_len_um = 40
            else:
                line_len_um = 80  # a3长度加倍
            line_len_pix = line_len_um / um_per_pix / 2  # 一半长度
            # π/9方向的增量
            angle = -67.3 / 180 * np.pi
            dx = line_len_pix * np.cos(angle)
            dy = line_len_pix * np.sin(angle)
            # 画-π/9方向虚线
            ax.plot([cx + dx, cx - dx], [cy + dy, cy - dy], ls='--', color='white', lw=0.5)
            # 画x轴横线
            ax.plot([cx - line_len_pix, cx + line_len_pix], [cy, cy], ls='--', color='white', lw=0.5)
            # 标注夹角φ
            # 计算弧线起点终点
            arc_radius = line_len_pix * 0.7
            arc_t = np.linspace(0, angle, 30)
            arc_x = cx + arc_radius * np.cos(arc_t)
            arc_y = cy + arc_radius * np.sin(arc_t)
            ax.plot(arc_x, arc_y, color='white', lw=0.5)
            # φ文字
            ax.text(cx + arc_radius * 1.1 * np.cos(angle/2), cy + arc_radius * 1.1 * np.sin(angle/2), r'$\phi$=65.7°', color='yellow', fontsize=10, fontweight='bold')
    
    # 添加 colorbar 到 gs[0, 0]
    cax = fig.add_subplot(gs[0, 0])
    plt.colorbar(im, cax=cax, orientation='vertical', label='Intensity (a.u.)')
    cax.yaxis.set_label_position('left')
    cax.yaxis.set_ticks_position('left')
    cax.set_xticks([])
    # 设置 colorbar 的刻度为 0, 0.6, 0.9, 1.2, 1.44（对应a0），并显示a0值（即根号后的值）
    # 取强度最大值和中值，计算其开根号后作为 colorbar 刻度
    vmax = np.max(focus_data_list[0][0])
    vmed = vmax/2
    ticks = [vmed, vmax]
    ticklabels = [f'{np.sqrt(vmed):.2f}', f'{np.sqrt(vmax):.2f}']
    cax.set_yticks(ticks)
    cax.set_yticklabels(ticklabels)
    cax.set_frame_on(False)
    cax.set_ylabel(r'$a_0$', fontsize=12)
    
    
    # 第二行 空白
    # 第三行：背景密度
    ax_ne = fig.add_subplot(gs[2, :])
    ax_ne.plot(bg_density_x, bg_density/0.0035*6, color='k', linestyle='-', label=r'$n_\mathrm{p}$')
    ax_ne.set_ylabel(r'$n_\mathrm{p}$ [$\times10^{18} cm^{-3}$]')
    # ax_ne.tick_params(axis='y', labelcolor=rgb4)
    ax_ne.set_ylim(0, max(bg_density)/0.0035*6*1.2)
    ax_ne.set_yticks([3, 6])
    # 将注释放在竖线左侧
    ax_ne.text(-0.6, 1, 'vacuum focus', color='k', fontsize=12, ha='left', va='center', rotation=0)
    # 添加水平箭头指向虚线
    ax_ne.annotate(
        '', 
        xy=(-0.99, 1), xycoords='data',
        xytext=(-0.6, 1), textcoords='data',
        arrowprops=dict(arrowstyle='->', color='k', lw=1),
        annotation_clip=False
    )
    ax_ne.set_ylabel(r'$n_\mathrm{p}$ [$\times10^{18}~\mathrm{cm^{-3}}$]')
    ax_ne.set_ylim(0, 7.2)
    ax_ne.set_yticks([2, 4, 6])
    ax_ne.text(0.05, 0.8, 'b', transform=ax_ne.transAxes, fontweight='bold', fontsize=14)
    ax_ne.axvline(x=-1, color='gray', linestyle='--', linewidth=1)
    ax_ne.grid(True, which='major', alpha=0.5, linestyle='--', zorder=0)

    # 第三行：a0演化曲线
    ax1 = fig.add_subplot(gs[3, :])
    ax1.plot(z_real, a0_real, label='case r', color=rgb1)
    ax1.plot(z_gauss, a0_gauss, label='case g', color=rgb2)
    ax1.plot(z_ea, a0_ea, label='case ea', color=rgb3)
    ax1.set_ylabel(r'$a_0$')
    # 添加z=-1mm的竖直灰色虚线
    ax1.axvline(x=-1, color='gray', linestyle='--', linewidth=1)
    # 隐藏图1底部x轴刻度和标签
    ax1.tick_params(axis='x', which='both', bottom=True, labelbottom=False)
    # 可选：添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    all_lines = lines1
    all_labels = labels1
    ax1.legend(all_lines, all_labels, loc='lower center', bbox_to_anchor=(0.6, -0.05), ncol=3, frameon=False, handlelength=1.2, columnspacing=0.8, handletextpad=0.4, borderpad=0.3, labelspacing=0.3)
    ax1.text(0.05, 0.8, 'c', transform=ax1.transAxes, fontweight='bold', fontsize=14)
    ax1.grid(True, which='major', alpha=0.5, linestyle='--', zorder=0)

    # 第四行
    case_r_folder = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/evolution'
    case_g_folder = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_gauss/img/evolution'
    case_ea_folder = '/public1/home/m8s000916/xyh/real_laser/real_astig_elli_6e18_1.2895a0/img/evolution'
    folder_list = [case_r_folder, case_g_folder, case_ea_folder]

    case_g_ex1 = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_gauss/img/LaserPlasma/inset_0320.txt'
    case_g_ex2 = '/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_gauss/img/LaserPlasma/inset_0330.txt'
    ex1 = np.loadtxt(case_g_ex1)
    ex2 = np.loadtxt(case_g_ex2)



    img_dir = case_r_folder
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    time_plot = np.zeros((101, 3))
    max_xi_plot = np.zeros((101, 3))
    for file_dir in folder_list:
        max_xi_path = os.path.join(file_dir, 'max_xi.txt')
        if os.path.exists(max_xi_path):
            max_xi = np.loadtxt(max_xi_path)
        else:
            print(f"File not found: {max_xi_path}")
            max_xi = None

        time_path = os.path.join(file_dir, 'time.txt')
        if os.path.exists(time_path):
            time = np.loadtxt(time_path)
            time = time / (2*np.pi) * 0.8 * 1e-3 - 2.2 # mm
        else:
            print(f"File not found: {time_path}")
            time = None
        

        if max_xi is not None:
            max_xi = max_xi - max_xi[0]
        max_xi = max_xi / (2*np.pi) * 0.8

        idx = folder_list.index(file_dir)
        if time is not None and max_xi is not None:
            time_plot[:len(time), idx] = time
            max_xi_plot[:len(max_xi), idx] = max_xi
            ax_xi = fig.add_subplot(gs[4, :])
            ax_xi.plot(time_plot[:,0], max_xi_plot[:,0], label='case r', color=rgb1)
            ax_xi.plot(time_plot[:,1], max_xi_plot[:,1], label='case g', color=rgb2)
            ax_xi.plot(time_plot[:,2], max_xi_plot[:,2], label='case ea', color=rgb3)
            ax_xi.set_ylabel(r'$\xi_{a_0}$ [mm]')
            ax_xi.set_ylim(-30, 5)
            ax_xi.set_yticks([-20, -10, 0])
            ax_xi.tick_params(axis='x', which='both', bottom=True, labelbottom=False)
            ax_xi.text(0.05, 0.8, 'd', transform=ax_xi.transAxes, fontweight='bold', fontsize=14)
            ax_xi.grid(True, which='major', alpha=0.5, linestyle='--', zorder=0)

            extent_range = [-706.8, 0, -684, 684]

            # 左下角inset: ex1 - 使用父坐标轴数据坐标定位
            # 定义插图在父坐标轴中的位置和大小 (x0, y0, width, height)
            inset1_x0 = -2  # 父坐标轴x坐标
            inset1_y0 = -42  # 父坐标轴y坐标
            inset1_width = 1  # 父坐标轴单位宽度
            inset1_height = 20  # 父坐标轴单位高度

            ax_inset1 = inset_axes(
                ax_xi,
                width="120%", height="120%",
                bbox_to_anchor=(inset1_x0, inset1_y0, inset1_width, inset1_height),
                bbox_transform=ax_xi.transData,  # 使用数据坐标而不是轴坐标
                loc='lower left',
                borderpad=0
            )
            im1 = ax_inset1.imshow(ex1, aspect='auto', cmap='jet', origin='lower', extent=extent_range)
            ax_inset1.set_ylim(-120, 120)
            ax_inset1.set_xlim(-350, -200)
            ax_inset1.set_xticks([])
            ax_inset1.set_yticks([])
            ax_inset1.set_xlabel('')
            ax_inset1.set_ylabel('')
            for spine in ax_inset1.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)
            ax_inset1.set_aspect(0.625)

            xlim = ax_inset1.get_xlim()
            ylim = ax_inset1.get_ylim()
            # 五角星和scale bar
            max_idx = np.unravel_index(np.argmax(ex1), ex1.shape)
            max_y, max_x = max_idx
            star_x = extent_range[0] + (extent_range[1] - extent_range[0]) * max_x / ex1.shape[1]
            star_y = extent_range[2] + (extent_range[3] - extent_range[2]) * max_y / ex1.shape[0]
            ax_inset1.plot(star_x, star_y, marker='x', color='white', markersize=3, zorder=10)

            # scale bar放在左下角
            bar_len_pix = 392
            bar_x = xlim[0] + 300 * (xlim[1] - xlim[0]) / ex1.shape[1]
            bar_y = ylim[0] + 150 * (ylim[1] - ylim[0]) / ex1.shape[0]
            bar_len = bar_len_pix * (extent_range[1] - extent_range[0]) / ex1.shape[1]
            # 横线
            ax_inset1.plot([bar_x, bar_x + bar_len], [bar_y, bar_y], color='white', lw=1.5)
            ax_inset1.text(bar_x + bar_len / 2, bar_y + 10 * (ax_inset1.get_ylim()[1] - ax_inset1.get_ylim()[0]) / ex1.shape[0],
                        r'$10~\mathrm{\mu m}$', color='white', fontsize=10, ha='center', va='bottom')
            # 竖线
            ax_inset1.plot([bar_x, bar_x], [bar_y, bar_y + bar_len], color='white', lw=1.5)
            # 横线右侧标记xi
            ax_inset1.text(bar_x + bar_len + 5, bar_y, r'$\xi$', color='white', fontsize=12, ha='left', va='center', fontweight='bold')
            # 竖线上侧标记r
            ax_inset1.text(bar_x, bar_y + bar_len + 8, r'$x$', color='white', fontsize=12, ha='center', va='bottom', fontweight='bold')



            # 右上角inset: ex2 - 使用父坐标轴数据坐标定位
            # 定义插图在父坐标轴中的位置和大小
            inset2_x0 = -1  # 父坐标轴x坐标
            inset2_y0 = -42   # 父坐标轴y坐标
            inset2_width = 1  # 父坐标轴单位宽度
            inset2_height = 20  # 父坐标轴单位高度

            ax_inset2 = inset_axes(
                ax_xi, 
                width="120%", height="120%", 
                bbox_to_anchor=(inset2_x0, inset2_y0, inset2_width, inset2_height),
                bbox_transform=ax_xi.transData,  # 使用数据坐标而不是轴坐标
                loc='lower left',
                borderpad=0
            )
            im2 = ax_inset2.imshow(ex2, aspect='auto', cmap='jet', origin='lower', extent=extent_range, zorder=10)
            ax_inset2.set_ylim(-120, 120)
            ax_inset2.set_xlim(-350, -200)
            ax_inset2.set_xticks([])
            ax_inset2.set_yticks([])
            ax_inset2.set_xlabel('')
            ax_inset2.set_ylabel('')
            for spine in ax_inset2.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)
            ax_inset2.set_aspect(0.625)

            # 五角星和scale bar
            max_idx = np.unravel_index(np.argmax(ex2), ex2.shape)
            max_y, max_x = max_idx
            xlim = ax_inset2.get_xlim()
            ylim = ax_inset2.get_ylim()
            star_x2 = extent_range[0] + (extent_range[1] - extent_range[0]) * max_x / ex2.shape[1]
            star_y2 = extent_range[2] + (extent_range[3] - extent_range[2]) * max_y / ex2.shape[0]
            ax_inset2.plot(star_x2, star_y2, marker='x', color='white', markersize=3, zorder=10)
            bar_x2 = xlim[0] + 300 * (xlim[1] - xlim[0]) / ex2.shape[1]
            bar_y2 = ylim[0] + 150 * (ylim[1] - ylim[0]) / ex2.shape[0]
            bar_len2 = bar_len_pix * (extent_range[1] - extent_range[0]) / ex2.shape[1]
            # 横线
            ax_inset2.plot([bar_x2, bar_x2 + bar_len2], [bar_y2, bar_y2], color='white', lw=1.5, zorder=10)
            ax_inset2.text(bar_x2 + bar_len2 / 2, bar_y2 + 10 * (ax_inset2.get_ylim()[1] - ax_inset2.get_ylim()[0]) / ex2.shape[0],
                        r'$10~\mathrm{\mu m}$', color='white', fontsize=10, ha='center', va='bottom', zorder=10)
            # 竖线
            ax_inset2.plot([bar_x2, bar_x2], [bar_y2, bar_y2 + bar_len2], color='white', lw=1.5, zorder=10)
            ax_inset2.text(bar_x2 + bar_len2 + 5, bar_y2, r'$\xi$', color='white', fontsize=12, ha='left', va='center', fontweight='bold', zorder=10)
            # 竖线上侧标记r
            ax_inset2.text(bar_x2, bar_y2 + bar_len2 + 8, r'$x$', color='white', fontsize=12, ha='center', va='bottom', fontweight='bold', zorder=10)

            # 在创建内嵌图后添加以下代码
            ax_inset2.set_clip_on(False)
            ax_inset2.set_zorder(100)  # 设置更高的zorder
            ax_inset1.set_clip_on(False)
            ax_inset1.set_zorder(100)  # 设置更高的zorder
            # 确保父坐标轴不会裁剪内容
            ax_xi.set_clip_on(False)

        # 标记两个箭头，inset1标记到z=-0.78，inset2标记到z=-0.73，都是标到max_xi_plot[:,1]这个曲线上
        arrow_positions = [-0.78, -0.73]
        arrow_texts = ['inset1', 'inset2']
        for xpos, txt in zip(arrow_positions, arrow_texts):
            idx = (np.abs(time_plot[:,1] - xpos)).argmin()
            x = time_plot[idx,1]
            y = max_xi_plot[idx,1]
            # 箭头从inset指到对应的点
            if txt == 'inset1':
                # inset1中心坐标（以数据坐标表示）
                inset_x = inset1_x0 + 1/2  # x0 + width/2
                inset_y = inset1_y0 + 20/2  # y0 + height/2
            else:
                inset_x = inset2_x0 + 1/2
                inset_y = inset2_y0 + 20/2
            # 箭头从inset中心指向曲线点
            ax_xi.annotate(
                '',  # 不显示文本
                xy=(x, y), xycoords='data',
                xytext=(inset_x, inset_y), textcoords='data',
                arrowprops=dict(facecolor='none', edgecolor='k', arrowstyle='->', lw=1),
                annotation_clip=False, zorder=10
            )



    # 第五行：charge演化曲线（加高斯滤波）
    ax2 = fig.add_subplot(gs[5, :], zorder=1)
    charge_real_smooth = gaussian_filter1d(charge_real, sigma=1)
    charge_gauss_smooth = gaussian_filter1d(charge_gauss, sigma=1)
    charge_ea_smooth = gaussian_filter1d(charge_ea, sigma=1)
    ax2.plot(z_real, charge_real_smooth, label='case r', color=rgb1)
    ax2.plot(z_gauss, charge_gauss_smooth, label='case g', color=rgb2)
    ax2.plot(z_ea, charge_ea_smooth, label='case ea', color=rgb3)
    ax2.set_ylabel(r'$Q_{\mathrm{in}}$ [pC]')
    ax2.set_ylim(0, 1600)
    ax2.set_yticks([400, 800, 1200])
    # 在y=150到y=270区域绘制阴影
    # ax2.axhspan(150, 270, color='gray', alpha=0.3)
    # ax2.legend(loc='upper right')
    ax2.text(0.05, 0.8, 'e', transform=ax2.transAxes, fontweight='bold', fontsize=14)
    # 设置x轴只在图1顶部显示
    ax2.set_xlabel('z [mm]')
    ax2.grid(True, which='major', alpha=0.5, linestyle='--', zorder=0)

    # 添加三个箭头和注释，指向case g曲线  0.32
    arrow_positions = [-1.19, -0.55]
    arrow_texts = ['1st injection', '2nd injection']
    for xpos, txt in zip(arrow_positions, arrow_texts):
        # 找到z_gauss中最接近xpos的索引
        idx = (np.abs(z_gauss - xpos)).argmin()
        x = z_gauss[idx]
        y = charge_gauss_smooth[idx]
        # 箭头起点和终点
        ax2.annotate(txt,
                     xy=(x, y), xycoords='data',
                     xytext=(x, y+500), textcoords='data',
                     arrowprops=dict(facecolor=rgb2, edgecolor=rgb2, arrowstyle='->', lw=1.5),
                     ha='center', va='bottom', fontsize=11, color=rgb2, fontweight='bold')

    # 第六行：空白行


    # 第七、八、九行：三个能谱图
    # 在主图的某个区域添加一个小的GridSpec
    inner_gs = gs[7, :].subgridspec(3, 31, wspace=0.05, hspace=0.0)
    hist2d_list = [
        (hist2d_real, energy_edges, angle_edges, 'case r', rgb1),
        (hist2d_gauss, energy_edges, angle_edges, 'case g', rgb2),
        (hist2d_ea, energy_edges, angle_edges, 'case ea', rgb3)
    ]
    for i, (hist2d, energy_edges, angle_edges, label, color) in enumerate(hist2d_list):
        ax = fig.add_subplot(inner_gs[i, 0:29])
        if hist2d is not None:
            im = ax.imshow(hist2d.T, aspect='auto', origin='lower',
                           extent=[energy_edges[0], energy_edges[-1], angle_edges[0], angle_edges[-1]],
                           cmap='viridis')
        ax.set_xlabel('Energy [MeV]')
        if i == 1:
            ax.set_ylabel('Angle [mrad]')
        else:
            ax.set_ylabel('')
        ax.set_yticks([-6, 0, 6])
        # 隐藏第78行的x轴刻度和标签
        if i < 2:
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        # 子图标号用白色，写c1 c2 c3，往下一些
        ax.text(0.05, 0.5, f'f{i+1}', transform=ax.transAxes, fontweight='bold', fontsize=14, color='white')
        ax.text(0.85, 0.5, label, transform=ax.transAxes, fontsize=12, color='white')
        ax.set_title('')
    # 统一colorbar，竖直放置在三个能谱图的右侧，占用这三行

    # 只显示colorbar，其余图片全部透明
    # 创建一个空白的Axes用于colorbar
    ax_cb = fig.add_subplot(inner_gs[:, -1])
    # 公共colorbar
    # 取所有能谱的最大值，设置colorbar范围
    vmax = max(np.nanmax(hist2d_real), np.nanmax(hist2d_gauss), np.nanmax(hist2d_ea))
    vmin = 0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap='viridis')
    cbar = plt.colorbar(sm, cax=ax_cb, orientation='vertical')
    cbar.set_label('Counts [a.u.]', fontsize=12)
    ax_cb.set_frame_on(False)
    ax_cb.set_xticks([])

    plt.tight_layout()
    plt.savefig('/public1/home/m8s000916/xyh/real_laser/1.2895a0_6e18_7modes/img/evolution/fig4.png', dpi=300)

