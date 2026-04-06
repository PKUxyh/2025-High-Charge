import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.restoration import unwrap_phase

# 参数设置
w0 = 25.85e-6       # 束腰半径
lambda0 = 800e-9    # 波长 (800nm)
k = 2*np.pi/lambda0
z_R = np.pi * w0**2 / lambda0  # 瑞利长度
tau = 35e-15        # 脉冲宽度 (35fs)
omega0 = 2*np.pi * 3e8/lambda0 # 角频率
E0 = 1              # 场强幅值
zn = -1200e-6      # zn位置 (-1200微米)
c = 3e8             # 光速
phi_c = 0           # 附加相位常数
L_tau = c * tau     # 脉冲长度
t=0

z_box = [-1245e-6, -1155e-6]  # z轴范围
r_box = [-88e-6, 88e-6]       # r轴范围
# 计算网格
z_range = np.linspace(z_box[0], z_box[1], 200)  # z轴范围
r_range = np.linspace(r_box[0], r_box[1], 200)      # r轴范围
z, r = np.meshgrid(z_range, r_range)

# 以微米为单位
z_um = z * 1e6
r_um = r * 1e6

# ==============================================
# 原公式计算
# ==============================================
w_z = w0 * np.sqrt(1 + (z/z_R)**2)
R_z = z * (1 + (z_R/(z))**2)
gouy_phase = np.arctan((z)/z_R)
E_original = (E0 * w0 / w_z * 
             np.exp(-r**2 / w_z**2) * 
             np.exp(-(t - (z - zn)/c)**2 / tau**2) * 
             np.exp(1j * (k*(z - zn) - omega0*t + k*r**2/(2*R_z) + gouy_phase)))
I_original = np.abs(E_original)**2
I_original = I_original / np.max(I_original)

phase_original = np.angle(E_original)
phase_original = unwrap_phase(phase_original)
phase_original = phase_original - phase_original[phase_original.shape[0]//2, phase_original.shape[1]//2]

# ==============================================
# 新公式计算
# ==============================================
w_zn = w0 * np.sqrt(1 + (zn/z_R)**2)
R_zn = zn * (1 + (z_R/(zn))**2)
gouy_phase_zn = np.arctan(zn/z_R)
E_new = (E0 * w0 / w_zn * 
        np.exp(-r**2 / w_zn**2) * 
        np.exp(-(z - zn)**2 / L_tau**2) * 
        np.exp(1j * (k*(z- zn) + k*r**2/(2*R_zn) + gouy_phase_zn + phi_c)))
I_new = np.abs(E_new)**2
I_new = I_new / np.max(I_new)

phase_new = np.angle(E_new)
phase_new = unwrap_phase(phase_new)
phase_new = phase_new - phase_new[phase_new.shape[0]//2, phase_new.shape[1]//2]

# ==============================================
# 差值计算
# ==============================================
I_diff = I_new - I_original
phase_diff = phase_new - phase_original

# ==============================================
# 绘图：三行两列
# ==============================================
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'lines.linewidth': 1.5,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'figure.dpi': 200,
    'figure.figsize': (6.74, 6.5),
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

fig, axes = plt.subplots(3, 2)

# 1. 原公式 - 强度
im1 = axes[0,0].imshow(I_original, extent=[z_box[0]*1e6, z_box[1]*1e6, r_box[0]*1e6, r_box[1]*1e6], aspect='auto', cmap='hot')
axes[0,0].set_ylabel(r'r [$\mu$m]')
axes[0,0].text(0.04, 0.2, 'a', transform=axes[0,0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left', color='white')
axes[0,0].set_xticks([])
axes[0,0].set_xlabel('')
divider1 = make_axes_locatable(axes[0,0])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax1, label='Intensity [a.u.]')
# 添加峰值0.135倍的等高线
peak1 = np.max(I_original)
contour1 = axes[0,0].contour(z_um, r_um, I_original, levels=[peak1*0.135], colors='white', linewidths=0.5)
axes[0,0].clabel(contour1, inline=True, fontsize=10, fmt=' %.3f', inline_spacing=15)

# 找到等高线上的所有点
contour1_paths = contour1.collections[0].get_paths()
contour_points = []
for path in contour1_paths:
    v = path.vertices
    contour_points.append(v)
contour_points = np.vstack(contour_points)  # shape (N, 2), columns: z, r

# 找到等高线上的y坐标（r）最大的点
max_r_idx = np.argmax(contour_points[:,1])
max_z_val = contour_points[max_r_idx, 0]
max_r_val = contour_points[max_r_idx, 1]
# 找到最近的网格点索引
z_idx = np.abs(z_um[0] - max_z_val).argmin()
r_idx = np.abs(r_um[:,0] - max_r_val).argmin()
phase_diff_val = phase_diff[r_idx, z_idx]

# 计算FWHM对应的等高线
fwhm_level = peak1 * 0.5
contour_fwhm = axes[0,0].contour(z_um, r_um, I_original, levels=[fwhm_level], colors='white', linewidths=0.8, linestyles='-', alpha=0)
# axes[0,0].clabel(contour_fwhm, inline=True, fontsize=10, fmt=' %.1f', inline_spacing=15)

# 获取FWHM等高线上的所有点
fwhm_points = []
for path in contour_fwhm.collections[0].get_paths():
    v = path.vertices
    fwhm_points.append(v)
if fwhm_points:
    fwhm_points = np.vstack(fwhm_points)
    # 找到这些点在phase_diff上的值
    fwhm_phase_vals = []
    for zf, rf in fwhm_points:
        zf_idx = np.abs(z_um[0] - zf).argmin()
        rf_idx = np.abs(r_um[:,0] - rf).argmin()
        fwhm_phase_vals.append(phase_diff[rf_idx, zf_idx])
    # 取这些点的phase_diff的均值作为等高线值（也可以绘制所有等高线）
    fwhm_phase_mean = np.mean(fwhm_phase_vals)
else:
    fwhm_phase_mean = None

# 找到等高线上的最大强度点
max_idx = None
max_val = -np.inf
for pt in contour_points:
    z_val, r_val = pt
    # 找到最近的网格点索引
    z_idx = np.abs(z_um[0] - z_val).argmin()
    r_idx = np.abs(r_um[:,0] - r_val).argmin()
    val = I_original[r_idx, z_idx]
    if val > max_val:
        max_val = val
        max_idx = (r_idx, z_idx)
max_z_val = z_um[0, max_idx[1]]
max_r_val = r_um[max_idx[0], 0]

# 2. 原公式 - 相位
im2 = axes[0,1].imshow(phase_original, extent=[z_box[0]*1e6, z_box[1]*1e6, r_box[0]*1e6, r_box[1]*1e6], aspect='auto', cmap='rainbow')
contours1 = axes[0,1].contour(z_um, r_um, phase_original, levels=5, colors='white', linewidths=0.5)
axes[0,1].clabel(contours1, inline=True, fontsize=10, fmt=' %.1f', inline_spacing=15)
axes[0,1].text(0.04, 0.2, 'b', transform=axes[0,1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left', color='white')
axes[0,1].set_xticks([])
axes[0,1].set_xlabel('')
axes[0,1].set_yticks([])
axes[0,1].set_ylabel('')
divider2 = make_axes_locatable(axes[0,1])
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im2, cax=cax2, label='Phase [rad]')

# 3. 新公式 - 强度
im3 = axes[1,0].imshow(I_new, extent=[z_box[0]*1e6, z_box[1]*1e6, r_box[0]*1e6, r_box[1]*1e6], aspect='auto', cmap='hot')
axes[1,0].set_ylabel(r'r [$\mu$m]')
axes[1,0].text(0.04, 0.2, 'c', transform=axes[1,0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left', color='white')
axes[1,0].set_xticks([])
axes[1,0].set_xlabel('')
divider3 = make_axes_locatable(axes[1,0])
cax3 = divider3.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im3, cax=cax3, label='Intensity [a.u.]')
# 添加峰值0.135倍的等高线
peak3 = np.max(I_new)
contour3 = axes[1,0].contour(z_um, r_um, I_new, levels=[peak3*0.135], colors='white', linewidths=0.5)
axes[1,0].clabel(contour3, inline=True, fontsize=10, fmt=' %.3f', inline_spacing=15)

# 4. 新公式 - 相位
im4 = axes[1,1].imshow(phase_new, extent=[z_box[0]*1e6, z_box[1]*1e6, r_box[0]*1e6, r_box[1]*1e6], aspect='auto', cmap='rainbow')
contours2 = axes[1,1].contour(z_um, r_um, phase_new, levels=5, colors='white', linewidths=0.5)
axes[1,1].clabel(contours2, inline=True, fontsize=10, fmt=' % .1f', inline_spacing=15)
axes[1,1].text(0.04, 0.2, 'd', transform=axes[1,1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left', color='white')
axes[1,1].set_xticks([])
axes[1,1].set_xlabel('')
axes[1,1].set_yticks([])
axes[1,1].set_ylabel('')
divider4 = make_axes_locatable(axes[1,1])
cax4 = divider4.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im4, cax=cax4, label='Phase [rad]')

# 5. 强度差值
im5 = axes[2,0].imshow(I_diff*1000, extent=[z_box[0]*1e6, z_box[1]*1e6, r_box[0]*1e6, r_box[1]*1e6], aspect='auto', cmap='hot')
axes[2,0].set_xlabel(r'z [$\mu$m]')
axes[2,0].set_ylabel(r'r [$\mu$m]')
axes[2,0].text(0.04, 0.2, 'e', transform=axes[2,0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left', color='white')
divider5 = make_axes_locatable(axes[2,0])
cax5 = divider5.append_axes("right", size="5%", pad=0.05)
cb5 = plt.colorbar(im5, cax=cax5, label=r'Intensity [$\times 10^{-3}$ a.u.]')
cb5.formatter.set_useOffset(False)
cb5.formatter.set_scientific(False)
cb5.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
cb5.ax.figure.canvas.draw_idle()  # 强制刷新
cb5.update_ticks()


# 6. 相位差值
im6 = axes[2,1].imshow(phase_diff, extent=[z_box[0]*1e6, z_box[1]*1e6, r_box[0]*1e6, r_box[1]*1e6], aspect='auto', cmap='rainbow')
# contour6 = axes[2,1].contour(z_um, r_um, phase_diff, levels=[phase_diff_val], colors='white', linewidths=0.5)
# axes[2,1].clabel(contour6, inline=True, fontsize=10, fmt=' %.2f', inline_spacing=15)

# 绘制FWHM对应的phase_diff等高线
if fwhm_phase_mean is not None:
    contour_fwhm6 = axes[2,1].contour(z_um, r_um, phase_diff, levels=[-0.02, 0.02], colors='white', linewidths=0.5, linestyles='-')
    axes[2,1].clabel(contour_fwhm6, inline=True, fontsize=10, fmt=' %.2f', inline_spacing=15)

axes[2,1].set_xlabel(r'z [$\mu$m]')
axes[2,1].set_ylabel('')
axes[2,1].text(0.04, 0.2, 'f', transform=axes[2,1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left', color='white')
axes[2,1].set_yticks([])
divider6 = make_axes_locatable(axes[2,1])
cax6 = divider6.append_axes("right", size="5%", pad=0.05)
cb6 = plt.colorbar(im6, cax=cax6, label='Phase [rad]')
cb6.formatter.set_useOffset(False)
cb6.formatter.set_scientific(False)
cb6.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
cb6.ax.figure.canvas.draw_idle()  # 强制刷新



plt.tight_layout()
plt.show()


# 计算I_diff绝对值最大值及其对应的空间坐标
max_idx = np.unravel_index(np.abs(I_diff).argmax(), I_diff.shape)
max_I_diff = I_diff[max_idx]
max_z_um = z_um[max_idx]
max_r_um = r_um[max_idx]
print(f"I_diff绝对值最大值: {max_I_diff:.6e}，对应坐标: z={max_z_um:.2f} um, r={max_r_um:.2f} um")

# # 计算r=25.85um处对应的phase_diff值
# r_target = 25.85 # um
# r_idx = np.abs(r_um[:,0] - r_target).argmin()
# phase_diff_r = phase_diff[r_idx, :]
# print(f"r=25.85um处phase_diff值（z轴范围内）:")
# for zi, val in zip(z_um[0], phase_diff_r):
#     print(f"z={zi:.2f} um, phase_diff={val:.6f}")
print(f"相位差值最大: {np.max(np.abs(phase_diff)):.6f} rad")

