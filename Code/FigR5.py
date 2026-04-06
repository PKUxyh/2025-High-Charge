#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.signal import savgol_filter
from scipy.signal import hilbert
from scipy.interpolate import griddata, interp1d
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Attempt to import EM from fld_module
try:
    sys.path.append(r'/public23/home/sca1437/xyh/osiris2d/real_laser/')
    from fld_module import EM
except ImportError:
    print("Warning: fld_module or EM class not found. This script must be run in the environment where fld_module is available.")

# Global Style Settings (Learning from FigR5 and a0_evolution_deflection)
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.color': 'gray',
    'figure.facecolor': 'white',
    'figure.dpi': 300,
    'figure.figsize': (6.74, 5),
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.frameon': False,
    'legend.fontsize': 12,
    'legend.loc': 'best',
})

rgb1 = '#3F77A3'
rgb2 = '#E49A5C'

# --- Functions from a0_evolution_deflection [3].py ---

def polar2cartesian(f_rec_real, r_range, theta_range, **kwargs):
    interp_method = kwargs.get('InterpMethod', 'linear')
    center = kwargs.get('Center', [0, 0])
    grid_size = kwargs.get('GridSize', f_rec_real.shape)
    fill_value = kwargs.get('FillValue', 0)
    
    cx, cy = center
    Ny, Nx = grid_size

    Theta, R = np.meshgrid(theta_range, r_range)
    X_polar = R * np.cos(Theta) + cx
    Y_polar = R * np.sin(Theta) + cy

    x_min, x_max = np.min(X_polar), np.max(X_polar)
    y_min, y_max = np.min(Y_polar), np.max(Y_polar)
    x_grid = np.linspace(x_min, x_max, Nx)
    y_grid = np.linspace(y_min, y_max, Ny)
    Xq, Yq = np.meshgrid(x_grid, y_grid)

    points = np.column_stack((X_polar.ravel(), Y_polar.ravel()))
    values = f_rec_real.ravel()
    f_xy = griddata(points, values, (Xq, Yq), method=interp_method, fill_value=fill_value)

    return f_xy, Xq, Yq

# --- Functions from ScriptFigR5.py ---

def bilinear_interpolate(im, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)
    
    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]
    
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    
    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def extract_angular_profile(field, center_x, center_y, angle_rad, max_distance=None):
    height, width = field.shape
    if max_distance is None:
        max_distance = min(center_x, center_y, width-center_x, height-center_y)
    
    distances = np.linspace(-max_distance, max_distance, int(2*max_distance + 1))
    x_coords = center_x + distances * np.cos(angle_rad)
    y_coords = center_y + distances * np.sin(angle_rad)
    
    valid_mask = ((x_coords >= 0) & (x_coords < width) & 
                  (y_coords >= 0) & (y_coords < height))
    
    x_valid = x_coords[valid_mask]
    y_valid = y_coords[valid_mask]
    distances_valid = distances[valid_mask]
    
    intensities = bilinear_interpolate(field, x_valid, y_valid)
    return {'distances': distances_valid, 'intensities': intensities}

def calculate_profile_width(profile, method='linear', threshold_ratio=0.5):
    distances = profile['distances']
    intensities = profile['intensities']
    max_intensity = np.max(intensities)
    if max_intensity == 0: return 0, distances, intensities
    
    normalized_intensities = intensities / max_intensity
    interp_func = interp1d(distances, normalized_intensities, kind=method, bounds_error=False, fill_value=0)
    
    threshold = threshold_ratio
    above_threshold = normalized_intensities >= threshold
    if not np.any(above_threshold): return 0, distances, normalized_intensities
    
    threshold_indices = np.where(above_threshold)[0]
    left_idx = threshold_indices[0]
    right_idx = threshold_indices[-1]
    
    if left_idx > 0:
        x_left = np.linspace(distances[left_idx-1], distances[left_idx], 100)
        y_left = interp_func(x_left)
        left_bound = x_left[np.where(y_left >= threshold)[0][0]]
    else:
        left_bound = distances[left_idx]
    
    if right_idx < len(distances) - 1:
        x_right = np.linspace(distances[right_idx], distances[right_idx+1], 100)
        y_right = interp_func(x_right)
        right_bound = x_right[np.where(y_right >= threshold)[0][-1]]
    else:
        right_bound = distances[right_idx]
    
    return right_bound - left_bound, distances, normalized_intensities

def calculate_fwhm_angular(field_intensity, angle_direction, pixel_size=1):
    max_pos = np.unravel_index(np.argmax(field_intensity), field_intensity.shape)
    center_y, center_x = max_pos
    angle_rad = np.deg2rad(angle_direction)
    profile = extract_angular_profile(field_intensity, center_x, center_y, angle_rad)
    fwhm_pixels, positions, intensities = calculate_profile_width(profile, threshold_ratio=0.5)
    fwhm = fwhm_pixels * pixel_size
    return fwhm

# --- Simulation Processing Logic ---

def process_step(k, file_dir, max_modes, y_num, y_um, theta_rec):
    EM_obj = EM(file_dir)
    ez_re, ez_im, time_val, _, _ = EM_obj.get_modes(k, max_modes, 'e1')
    er_re, er_im, _, _, _ = EM_obj.get_modes(k, max_modes, 'e2')
    etheta_re, etheta_im, _, _, _ = EM_obj.get_modes(k, max_modes, 'e3')
    
    ex_rtheta = np.zeros((y_num, len(theta_rec)))
    theta_bias = np.pi * 0.135
    for i in range(len(theta_rec)):
        _, ex, _ = EM_obj.rec_fields_xyz(ez_re, er_re, etheta_re, ez_im, er_im, etheta_im, theta_bias + theta_rec[i])
        ex_hilbert = hilbert(ex, axis=1)
        ex_rtheta[:, i] = np.sum(np.abs(ex_hilbert)**2, axis=1)

    ex_cartesian, Xq, Yq = polar2cartesian(
        ex_rtheta,
        np.linspace(0, y_um, y_num),
        theta_rec,
        InterpMethod='linear',
        Center=[0, 0],
        GridSize=[2*y_num, 2*y_num],
        FillValue=0
    )
    
    pixel_size = y_um / y_num
    
    # Calculate FWHM at 65.7 and 155.7 degrees
    fwhm_65 = calculate_fwhm_angular(ex_cartesian, 65.7, pixel_size=pixel_size)
    fwhm_155 = calculate_fwhm_angular(ex_cartesian, 155.7, pixel_size=pixel_size)
    
    # Convert FWHM to 1/e^2 radius: w = FWHM / 1.1774 
    radius_65 = (fwhm_65 / 1.1774)
    radius_155 = (fwhm_155 / 1.1774)
    
    return time_val, radius_65, radius_155

if __name__ == '__main__':
    # Parameters from a0_evolution_deflection [3].py
    mv_ylim = [0, 692]
    y_num = (mv_ylim[1] - mv_ylim[0]) * 1
    y_um = (mv_ylim[1] - mv_ylim[0]) / 2 / np.pi * 0.8
    file_dir = '/public1/home/m8s000916/xyh/real_laser/real_astig_elli_6e18_1.2895a0_track'
    init_num = 0
    iter_num = 10
    total_num = 1010
    max_modes = 3
    theta_rec = np.linspace(0, 2 * np.pi, 360)

    k_values = range(init_num, total_num, iter_num)
    
    print(f"Starting processing {len(k_values)} steps...")
    start_t = time.time()
    num_cores = min(32, cpu_count())
    
    # Create partial function for mapping
    process_func = partial(process_step,
                         file_dir=file_dir,
                         max_modes=max_modes,
                         y_num=y_num,
                         y_um=y_um,
                         theta_rec=theta_rec)

    with Pool(processes=num_cores) as pool:
        results = pool.map(process_func, k_values)
    
    print(f"Processing completed in {time.time() - start_t:.2f} seconds.")

    # Filter out None results
    valid_results = [r for r in results if r is not None]
    plot_time = np.array([r[0] for r in valid_results])
    radius_65_list = np.array([r[1] for r in valid_results])
    radius_155_list = np.array([r[2] for r in valid_results])

    # Convert time [1/omega_L] to z [mm]
    # Assuming z = c * t. In normalized units, if t is in 1/omega_L, then z_norm = t.
    # To get z in mm, we need wavelength information. 
    # Usually in these scripts, y_um is already in microns. 
    # If t is normalized simulation time, we use the same scaling as a0_evolution_deflection.
    # However, FigR5 specifically asks for z[mm]. 
    # Let's use a standard conversion if wavelength is 0.8um (common in these scripts).
    wavelength_um = 0.8
    z_start_mm = -2.2
    # Shift z so it starts at z_start_mm
    z_mm = z_start_mm + (plot_time - plot_time[0]) * (wavelength_um / (2 * np.pi)) * 1e-3
    
    # Apply smoothing
    if len(radius_65_list) > 7:
        radius_65_smoothed = savgol_filter(radius_65_list, window_length=7, polyorder=2)
        radius_155_smoothed = savgol_filter(radius_155_list, window_length=7, polyorder=2)
    else:
        radius_65_smoothed = radius_65_list
        radius_155_smoothed = radius_155_list

    # --- Plotting (Following ScriptFigR5 format) ---
    fig = plt.figure(figsize=(6.74, 6.74/1.7))
    ax = fig.add_subplot(111)
    
    ax.plot(z_mm, radius_65_smoothed, color=rgb1, label='65.7°')
    ax.plot(z_mm, radius_155_smoothed, color=rgb2, label='155.7°')

    # (Optional: find minimums if you still need the values, but don't plot markers)
    # min_idx1 = np.argmin(radius_65_smoothed)
    # min_idx2 = np.argmin(radius_155_smoothed)
    # min_z1 = z_mm[min_idx1]
    # min_z2 = z_mm[min_idx2]
    # min_r1 = radius_65_smoothed[min_idx1]
    # min_r2 = radius_155_smoothed[min_idx2]

    ax.set_xlabel(r'$z\ [\mathrm{mm}]$')
    ax.set_ylabel(r'$r\ [\mathrm{\mu m}]$')
    ax.legend()
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(file_dir), 'FigR5_simulation.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
