import numpy as np
import os
from data_loader import load_fluence_data
# prepare_data 已移除，所有预处理都在GSAMD初始化时完成
from config_gsamd import get_default_config
from visualization import plot_preprocessed_data, plot_reconstruction_results


def save_final_results(results, output_dir):
    """
    保存最终优化结果（所有平面的电场、强度、相位等）
    
    :param results: run_optimization返回的结果字典
    :param output_dir: 输出目录路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    E_list = results.get('E_list', [])
    C = results.get('C', None)
    best_centers = results.get('best_centers', None)
    
    print("\n" + "="*60)
    print("Saving final optimization results...")
    print("="*60)
    
    # 保存每个平面的电场（强度、相位、实部、虚部）
    for k, E in enumerate(E_list):
        plane_num = k + 1
        
        # 强度分布
        intensity = np.abs(E)**2
        intensity_path = os.path.join(output_dir, f'final_plane{plane_num}_intensity.csv')
        np.savetxt(intensity_path, intensity, delimiter=",")
        print(f"Saved: {intensity_path}")
        
        # 相位分布
        phase = np.angle(E)
        phase_path = os.path.join(output_dir, f'final_plane{plane_num}_phase.csv')
        np.savetxt(phase_path, phase, delimiter=",")
        print(f"Saved: {phase_path}")
        
        # 电场实部
        E_real = np.real(E)
        E_real_path = os.path.join(output_dir, f'final_plane{plane_num}_E_real.csv')
        np.savetxt(E_real_path, E_real, delimiter=",")
        print(f"Saved: {E_real_path}")
        
        # 电场虚部
        E_imag = np.imag(E)
        E_imag_path = os.path.join(output_dir, f'final_plane{plane_num}_E_imag.csv')
        np.savetxt(E_imag_path, E_imag, delimiter=",")
        print(f"Saved: {E_imag_path}")
    
    # 保存HG模式系数C
    if C is not None:
        # 保存系数实部
        C_real_path = os.path.join(output_dir, 'final_HG_coefficients_real.csv')
        np.savetxt(C_real_path, np.real(C), delimiter=",")
        print(f"Saved: {C_real_path}")
        
        # 保存系数虚部
        C_imag_path = os.path.join(output_dir, 'final_HG_coefficients_imag.csv')
        np.savetxt(C_imag_path, np.imag(C), delimiter=",")
        print(f"Saved: {C_imag_path}")
        
        # 保存系数幅度
        C_abs_path = os.path.join(output_dir, 'final_HG_coefficients_abs.csv')
        np.savetxt(C_abs_path, np.abs(C), delimiter=",")
        print(f"Saved: {C_abs_path}")
        
        # 保存系数相位
        C_phase_path = os.path.join(output_dir, 'final_HG_coefficients_phase.csv')
        np.savetxt(C_phase_path, np.angle(C), delimiter=",")
        print(f"Saved: {C_phase_path}")
    
    # 保存最佳中心位置
    if best_centers is not None:
        centers_path = os.path.join(output_dir, 'final_best_centers.csv')
        # 保存为两列：x0, y0（每个平面一行）
        centers_array = np.array(best_centers).reshape(-1, 2) * 1e6  # 转换为微米
        np.savetxt(centers_path, centers_array, delimiter=",", 
                  header="x0(um),y0(um)", comments="", fmt='%.6f')
        print(f"Saved: {centers_path}")
    
    print("="*60)
    print("All final results saved successfully!")
    print("="*60)


def run_optimization(config):
    """
    执行优化流程（支持任意数量的测量平面）
    
    :param config: 配置参数字典
    :return: 优化结果字典
    """
    # 确定输出目录（项目文件夹下的output_med）
    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_dir, 'output_med')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据（支持任意数量的平面）
    file_paths = config['data_paths']
    data_list = load_fluence_data(file_paths)
    n_planes = len(data_list)
    print(f"\nLoaded {n_planes} measurement planes.")
    
    # 注意：不再调用prepare_data()，因为所有预处理都在GSAMD初始化时完成
    
    # 设置参数
    grid_size = config['grid_size']
    wavelength = config['wavelength']
    w0x = config['w0x']
    w0y = config['w0y']
    z_positions = config['z_positions']
    F_exp_list = data_list  # 使用原始数据，GSAMD类内部会进行预处理
    
    x0_init_list = config['x0_init_list']
    y0_init_list = config['y0_init_list']
    
    # 检查数据数量与z_positions数量是否一致
    assert len(data_list) == len(z_positions), \
        f"data_paths ({len(data_list)}) and z_positions ({len(z_positions)}) must have the same length!"
    assert len(x0_init_list) == len(z_positions), \
        f"x0_init_list ({len(x0_init_list)}) and z_positions ({len(z_positions)}) must have the same length!"
    assert len(y0_init_list) == len(z_positions), \
        f"y0_init_list ({len(y0_init_list)}) and z_positions ({len(z_positions)}) must have the same length!"
    
    # 导入GSAMD类和配置参数（从TestGSAMD.py导入）
    from TestGSAMD import GSAMD, EG_SEARCH_AREA, EG_N_MODES, EG_N_CALLS, EG_INNER_ITER, \
                          RS_SEARCH_AREA, RS_N_MODES, RS_N_CALLS, RS_INNER_ITER, \
                          PLOT_OPTIMIZATION, SKIP_RS_STAGE, SKIP_BAYESIAN
    
    # 创建GSAMD实例（启用低通滤波，并传入output_dir）
    gsamd = GSAMD(F_exp_list, z_positions, wavelength, w0x, w0y, 
                  max_iter=config['max_iter'], grid_size=(grid_size, grid_size), 
                  x0_list=x0_init_list, y0_list=y0_init_list,
                  use_lowpass_filter=True, filter_sigma=2.0, output_dir=output_dir,
                  use_hg_filter=False, hg_filter_modes=(30, 30))
    
    # 绘制GSAMD预处理后的数据
    print("\nPlotting preprocessed data for GSAMD and Bayesian optimization...")
    plot_preprocessed_data(gsamd.F_list, output_dir=output_dir, save_plot=True, show_plot=False)
    
    # 贝叶斯优化 / 直接使用初始中心
    if SKIP_BAYESIAN:
        # 跳过贝叶斯优化，直接用初始中心
        print("\n[Skip Bayesian] Using initial centers directly for GS iteration...")
        best_centers = []
        for x0, y0 in zip(x0_init_list, y0_init_list):
            best_centers.extend([x0, y0])
        optimization_history = None
        final_N_modes = EG_N_MODES
    elif SKIP_RS_STAGE:
        # 单阶段搜索（仅EG）
        print("\n[Single-stage mode] Only running EG stage...")
        _, best_centers, optimization_history = gsamd.optimize_centers_single_stage(
            search_area=EG_SEARCH_AREA,
            N_modes=EG_N_MODES,
            n_calls=EG_N_CALLS,
            inner_iter=EG_INNER_ITER,
            plot_optimization=PLOT_OPTIMIZATION,
            output_dir=output_dir,
            stage_prefix="EG_"
        )
        final_N_modes = EG_N_MODES
    else:
        # 两阶段搜索（EG + RS）
        _, best_centers, optimization_history = gsamd.optimize_centers(
            EG_search_area=EG_SEARCH_AREA,
            EG_N_modes=EG_N_MODES,
            EG_n_calls=EG_N_CALLS,
            EG_inner_iter=EG_INNER_ITER,
            RS_search_area=RS_SEARCH_AREA,
            RS_N_modes=RS_N_MODES,
            RS_n_calls=RS_N_CALLS,
            RS_inner_iter=RS_INNER_ITER,
            plot_optimization=PLOT_OPTIMIZATION,
            output_dir=output_dir
        )
        final_N_modes = RS_N_MODES
    print(f"\nOptimization completed. Best centers: {best_centers}")
    
    # GSAMD系数精细优化（GS迭代）
    C = gsamd.refine_coefficients(
        best_centers=best_centers,
        N_modes=final_N_modes,
        max_iter=config['max_iter']
    )
    
    # 用优化后的系数和最佳中心来重建各平面的电场分布
    E_list = []
    x0_list = best_centers[::2]
    y0_list = best_centers[1::2]
    for k in range(n_planes):
        E = gsamd.reconstruct_field(C, x0_list[k], y0_list[k], z_positions[k])
        E_list.append(E)
    
    # 可视化结果：使用GSAMD处理后的数据（self.F_list）进行对比
    F_processed_list = gsamd.F_list
    plot_reconstruction_results(F_processed_list, E_list)
    
    # 保存最终结果
    save_final_results({
        'C': C,
        'best_centers': best_centers,
        'E_list': E_list,
        'optimization_history': optimization_history
    }, output_dir)
    
    return {
        'C': C,
        'best_centers': best_centers,
        'E_list': E_list,
        'F_processed_list': gsamd.F_list,
        'F_original': data_list,
        'optimization_history': optimization_history
    }


def save_results(E_far, output_path):
    """
    保存结果（向后兼容的旧函数）
    
    :param E_far: 远场电场
    :param output_path: 输出文件路径
    """
    # 保存 E_far 为 CSV 文件
    np.savetxt(output_path, E_far, delimiter=",")
