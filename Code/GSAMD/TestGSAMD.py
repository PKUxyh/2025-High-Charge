import numpy as np
from scipy.special import hermite
from scipy.fft import fft2, ifft2
from skopt import gp_minimize
import pandas as pd
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase
import math
import os
import datetime
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter


# 从工具模块导入函数
from utils_gsamd import crop_center, HG_mode, preprocess_fluence

class GSAMD:
    def __init__(self, F_exp_list, z_positions, wavelength, w0_x, w0_y, max_iter=50, grid_size=(100, 100), x0_list=None, y0_list=None, 
                 use_lowpass_filter=True, filter_sigma=2.0, use_hg_filter=False, hg_filter_modes=(20, 20), output_dir=None):
        """
        :param F_exp_list: 多平面光强测量列表（已对齐）
        :param z_positions: 各平面对应的z坐标
        :param wavelength: 激光波长
        :param w0_x, w0_y: HG模式初始腰斑
        :param use_lowpass_filter: 是否在预处理时使用低通滤波（高斯滤波，默认True）
        :param filter_sigma: 高斯滤波的标准差（默认2.0）
        :param use_hg_filter: 是否在预处理时使用HG模式滤波（默认False）
        :param hg_filter_modes: HG滤波保留的模式阶数 (m_max, n_max)，默认(20, 20)
        :param output_dir: 输出目录路径（用于保存预处理数据）
        """
        self.z_list = z_positions
        self.wavelength = wavelength
        self.w0_x = w0_x
        self.w0_y = w0_y
        
        # 先创建网格（需要先知道数据大小）
        Nx = np.shape(F_exp_list[0])[1]
        Ny = np.shape(F_exp_list[0])[0]
        Lx = Nx * grid_size[0]  # 假设网格大小
        Ly = Ny * grid_size[1]
        self.x_grid, self.y_grid = np.meshgrid(
            np.linspace(-Lx/2, Lx/2, Nx),
            np.linspace(-Ly/2, Ly/2, Ny)
        )
        self.dx = grid_size[0]  # 网格间距
        self.dy = grid_size[1]
        
        # 初始化中心列表
        N = len(self.z_list)
        if x0_list is None:
            self.x0_list = [0.0] * N
        else:
            self.x0_list = x0_list
        if y0_list is None:
            self.y0_list = [0.0] * N
        else:
            self.y0_list = y0_list
        
        # 对每个平面进行预处理，使用低通滤波（或HG滤波）
        self.F_list = []
        for k, F_exp in enumerate(F_exp_list):
            if use_lowpass_filter:
                # 使用低通滤波（高斯滤波）
                F_processed = preprocess_fluence(F_exp, lowpass_filter=True, filter_sigma=filter_sigma,
                                                save_path=output_dir, plane_index=k)
            elif use_hg_filter:
                # 使用HG模式滤波（可选）
                hg_params = {
                    'x_grid': self.x_grid,
                    'y_grid': self.y_grid,
                    'x0': self.x0_list[k],
                    'y0': self.y0_list[k],
                    'z': self.z_list[k],
                    'w0_x': self.w0_x,
                    'w0_y': self.w0_y,
                    'wavelength': self.wavelength,
                    'dx': self.dx,
                    'dy': self.dy,
                    'N_modes': hg_filter_modes
                }
                F_processed = preprocess_fluence(F_exp, hg_filter=True, hg_params=hg_params, 
                                                save_path=output_dir, plane_index=k)
            else:
                # 只做阈值处理，不进行滤波
                F_processed = preprocess_fluence(F_exp)
            self.F_list.append(F_processed)
        
        self.max_iter = max_iter
        self.error = np.zeros(self.max_iter)
        
        # 添加模式缓存（用于加速）
        self._mode_cache = {}  # 缓存已计算的HG模式
        self._cache_enabled = True  # 缓存开关
    
    def clear_mode_cache(self):
        """清理模式缓存（当参数变化较大时使用）"""
        self._mode_cache.clear()
        print("Mode cache cleared.")
    
    def enable_mode_cache(self, enable=True):
        """启用或禁用模式缓存"""
        self._cache_enabled = enable
        if not enable:
            self.clear_mode_cache()

    def initialize_phase(self, delta_z=100e-6):
        """初始化相位（假设为二次曲面）"""
        w0 = np.sqrt(self.w0_x**2 + self.w0_y**2)
        k0 = 2 * np.pi / self.wavelength
        fai =  np.zeros_like(self.x_grid, dtype=float)
        fai = k0 * ((self.x_grid-self.x0_list[0])**2 + (self.y_grid-self.y0_list[0])**2) / (2 * delta_z * (1+((k0 /2) *w0**2 / delta_z)**2))

        # plt.close('all')  # 关闭旧图
        # plt.figure(figsize=(6, 5))
        # plt.imshow(fai, cmap='jet')
        # plt.title('Initial Phase (fai)')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.show()  # 非阻塞显示
        return fai

#np.sqrt(2/(c*eps0*tau)) * 
    def initialize_coefficients(self, C, N_modes):
        """初始化HG系数（假设相位为二次曲面，优化版本：使用缓存）"""
        init_phase = self.initialize_phase(delta_z=20e-6)
        E0_init = np.sqrt(self.F_list[0]) * np.exp(1j*init_phase)
        z0 = self.z_list[0]
        x0 = self.x0_list[0]
        y0 = self.y0_list[0]
        
        for m in range(N_modes[0]):
            for n in range(N_modes[1]):
                # 使用缓存加速（如果启用）
                if self._cache_enabled:
                    cache_key = (m, n, x0, y0, z0)
                    if cache_key not in self._mode_cache:
                        self._mode_cache[cache_key] = HG_mode(m, n, self.x_grid, self.y_grid,
                                                              x0, y0, z0, 
                                                              self.w0_x, self.w0_y, self.wavelength)
                    mode = self._mode_cache[cache_key]
                else:
                    mode = HG_mode(m, n, self.x_grid, self.y_grid,
                                  x0, y0, z0, 
                                  self.w0_x, self.w0_y, self.wavelength)
                mode_norm_sq = np.sum(np.abs(mode)**2) * self.dx * self.dy
                if mode_norm_sq > 0:
                    C[m,n] = np.sum(E0_init * mode.conj()) * self.dx * self.dy / mode_norm_sq
        return C
    
    def decompose_field(self, E, Modes, x0_list, y0_list, z):
        """将电场分解到HG模式"""
        C = np.zeros((Modes.shape[0], Modes.shape[1]), dtype=complex)
        for m in range(Modes.shape[0]):
            for n in range(Modes.shape[1]):
                mode = HG_mode(m, n, self.x_grid, self.y_grid,
                                x0_list, y0_list, z, 
                                self.w0_x, self.w0_y, self.wavelength)
                C[m, n] = np.sum(E * mode.conj()) * self.dx * self.dy / np.sum(np.abs(mode)**2 * self.dx * self.dy)
        return C
    
    def reconstruct_field(self, C, x0, y0, z, use_cache=None):
        """
        用当前系数重建电场（优化版本）
        
        :param use_cache: 是否使用模式缓存（None时使用self._cache_enabled）
        """
        if use_cache is None:
            use_cache = self._cache_enabled
        
        E = np.zeros_like(self.x_grid, dtype=complex)
        
        if use_cache:
            # 使用缓存加速
            for m in range(C.shape[0]):
                for n in range(C.shape[1]):
                    if C[m, n] == 0:
                        continue  # 跳过零系数
                    # 生成缓存键
                    cache_key = (m, n, x0, y0, z)
                    if cache_key not in self._mode_cache:
                        self._mode_cache[cache_key] = HG_mode(m, n, self.x_grid, self.y_grid, 
                                                              x0, y0, z, self.w0_x, self.w0_y, self.wavelength)
                    E += C[m, n] * self._mode_cache[cache_key]
        else:
            # 不使用缓存（原始方法）
            for m in range(C.shape[0]):
                for n in range(C.shape[1]):
                    if C[m, n] == 0:
                        continue  # 跳过零系数
                    mode = HG_mode(m, n, self.x_grid, self.y_grid, 
                                  x0, y0, z, self.w0_x, self.w0_y, self.wavelength)
                    E += C[m, n] * mode
        
        return E
    
    def update_coefficients(self, C, x0_list, y0_list, max_iter=50):
        """
        Algorithm 1: 迭代更新HG系数
        
        包含收敛判断机制：
        - 每隔5次迭代计算一次误差梯度 chi_grad^2
        - 如果梯度小于2%，则认为算法收敛，停止循环
        
        误差梯度定义：
        chi_grad^2 = (chi^2(iter) - chi^2(iter-5)) / chi^2(iter-5)
        
        当 |chi_grad^2| < 0.02 (2%) 时，算法收敛。
        """
        # 确保 error 数组大小足够（至少为 max_iter）
        # 如果数组太小，重新分配为 max_iter 大小
        if len(self.error) < max_iter:
            self.error = np.zeros(max_iter)
        
        for iter in range(max_iter):
            for k, z in enumerate(self.z_list):
                E = self.reconstruct_field(C, x0_list[k], y0_list[k], z)
                phase = np.angle(E)
                phase = unwrap_phase(phase)

                if k == 5:
                    # 初始平面直接使用测量数据
                    E_new = np.sqrt(self.F_list[k]) * np.exp(1j*phase)
                else:
                    E_new = np.sqrt(self.F_list[k]) * np.exp(1j*phase)
                    delta = (np.abs(E_new) - np.abs(E)) / np.max(np.abs(E_new))
                    E_new = E_new * np.exp(delta)

                # 投影到HG模式（优化版本：使用缓存）
                C_new = np.zeros_like(C)
                if self._cache_enabled:
                    # 使用缓存加速模式计算
                    for m in range(C.shape[0]):
                        for n in range(C.shape[1]):
                            cache_key = (m, n, x0_list[k], y0_list[k], z)
                            if cache_key not in self._mode_cache:
                                self._mode_cache[cache_key] = HG_mode(m, n, self.x_grid, self.y_grid,
                                                                      x0_list[k], y0_list[k], z, 
                                                                      self.w0_x, self.w0_y, self.wavelength)
                            mode = self._mode_cache[cache_key]
                            # 向量化计算投影系数
                            mode_norm_sq = np.sum(np.abs(mode)**2) * self.dx * self.dy
                            if mode_norm_sq > 0:
                                C_new[m,n] = np.sum(E_new * mode.conj()) * self.dx * self.dy / mode_norm_sq
                else:
                    # 不使用缓存
                    for m in range(C.shape[0]):
                        for n in range(C.shape[1]):
                            mode = HG_mode(m, n, self.x_grid, self.y_grid,
                                          x0_list[k], y0_list[k], z, 
                                          self.w0_x, self.w0_y, self.wavelength)
                            mode_norm_sq = np.sum(np.abs(mode)**2) * self.dx * self.dy
                            if mode_norm_sq > 0:
                                C_new[m,n] = np.sum(E_new * mode.conj()) * self.dx * self.dy / mode_norm_sq
                # 归一化并更新系数
                C = 0.5*(C + C_new)
            
            # 计算当前迭代的总误差 chi^2（使用与calculate_error相同的公式）
            # 在完成所有平面的处理后，计算总误差
            chi2_current_iter = self.calculate_error(C, x0_list, y0_list)
            self.error[iter] = chi2_current_iter

            # 收敛判断：每隔5次迭代计算误差梯度
            if (iter + 1) % 5 == 0 and iter >= 4:
                # 计算当前误差 chi^2(iter)
                chi2_current = self.error[iter]
                # 获取5次迭代前的误差 chi^2(iter-5)
                chi2_previous = self.error[iter - 5]
                
                # 计算误差梯度 chi_grad^2 = (chi^2(iter) - chi^2(iter-5)) / chi^2(iter-5)
                if chi2_previous > 0:
                    chi_grad_squared = (chi2_current - chi2_previous) / chi2_previous
                else:
                    chi_grad_squared = float('inf')  # 避免除零
                
                print(f"Iteration {iter+1}/{max_iter}, chi^2: {chi2_current:.6f}, "
                      f"chi_grad^2: {chi_grad_squared*100:.4f}%")
                
                # 如果梯度小于2%，则认为算法收敛
                if abs(chi_grad_squared) < 0.02:
                    print(f"\n{'='*60}")
                    print(f"Convergence reached at iteration {iter+1}!")
                    print(f"chi_grad^2 = {chi_grad_squared*100:.4f}% < 2%")
                    print(f"Final chi^2: {chi2_current:.6f}")
                    print(f"{'='*60}")
                    # 截断error数组，只保留实际迭代的部分
                    self.error = self.error[:iter+1]
                    break
            else:
                print(f"Iteration {iter+1}/{max_iter}, chi^2: {self.error[iter]:.6f}")
        
        return C
    
    def optimize_centers_single_stage(self, search_area=5e-6, N_modes=(10,10), n_calls=20, inner_iter=10, 
                                      plot_optimization=True, output_dir=None, stage_prefix="", initial_centers=None):
        """
        单阶段贝叶斯优化函数
        
        :param search_area: 搜索区域大小
        :param N_modes: HG模式数量
        :param n_calls: 贝叶斯优化调用次数
        :param inner_iter: 内部迭代次数
        :param plot_optimization: 是否绘制优化过程
        :param output_dir: 输出目录路径
        :param stage_prefix: 阶段前缀（用于文件命名）
        :param initial_centers: 初始中心（用于第二阶段，如果为None则使用self.x0_list和self.y0_list）
        :return: C_final, best_centers, optimization_history
        """
        # 确定输出目录
        if output_dir is None:
            project_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(project_dir, 'output_med')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 确定初始中心
        # 注意：dimensions 必须为交错格式 [x0_0, y0_0, x0_1, y0_1, ...]，与 objective 的 params[::2]/params[1::2] 及绘图解析一致
        if initial_centers is not None:
            # 使用提供的初始中心，搜索范围围绕这些中心（initial_centers 格式为 [x0_0, y0_0, x0_1, y0_1, ...]）
            x0_init = initial_centers[::2]
            y0_init = initial_centers[1::2]
            dimensions = []
            for i in range(len(self.z_list)):
                if i < len(x0_init):
                    dimensions.append((x0_init[i] - search_area, x0_init[i] + search_area))
                else:
                    dimensions.append((-search_area, search_area))
                if i < len(y0_init):
                    dimensions.append((y0_init[i] - search_area, y0_init[i] + search_area))
                else:
                    dimensions.append((-search_area, search_area))
        else:
            # 使用默认中心（0,0），搜索范围对称
            dimensions = [
                (-search_area, search_area) 
                for _ in range(2 * len(self.z_list))
            ]
        
        optimization_history = {
            'params': [],
            'errors': [],
            'iteration': []
        }
        
        def objective(params):
            x0_list = params[::2]
            y0_list = params[1::2]
            print(f"[{stage_prefix}] x0_list: {x0_list}, y0_list: {y0_list}")
            
            C = np.zeros((N_modes[0], N_modes[1]), dtype=complex)
            C = self.initialize_coefficients(C, N_modes)
            
            # 确保 error 数组大小足够（使用 inner_iter，至少需要 inner_iter 大小）
            # 注意：update_coefficients 内部也会检查，但这里提前确保更安全
            if len(self.error) < inner_iter:
                self.error = np.zeros(inner_iter)
            C = self.update_coefficients(C, x0_list, y0_list, max_iter=inner_iter)

            error_val = self.calculate_error(C, x0_list, y0_list)
            
            optimization_history['params'].append(params.copy())
            optimization_history['errors'].append(error_val)
            iteration_num = len(optimization_history['errors'])
            optimization_history['iteration'].append(iteration_num)

            # 减少日志写入频率（每10次迭代写入一次）
            if iteration_num % 10 == 0 or iteration_num == 1:
                write_optimization_log(params, C, error_val, log_dir=r"F:\paper\GS_new\GSA-MD")
            
            # 减少保存频率：只保存每10次迭代的结果，或者是最佳结果
            # 可以通过配置参数控制是否保存所有结果
            save_all_results = False  # 设置为True保存所有结果，False只保存每10次
            save_images = False  # 图片保存较慢，默认只保存CSV
            if save_all_results or iteration_num % 10 == 0 or iteration_num == 1:
                save_bayesian_result(self, C, x0_list, y0_list, iteration_num, output_dir, 
                                   stage_prefix=stage_prefix, save_images=save_images)

            return error_val  

        res = gp_minimize(
            objective,
            dimensions=dimensions,
            n_calls=n_calls,
            # random_state=42,
            verbose=True
        )
        
        # 只在RS阶段绘制优化结果和打印SUMMARY（EG阶段跳过，EG阶段在optimize_centers中单独处理）
        # 注意：RS阶段的绘图在optimize_centers中单独处理，以便传入EG的最佳位置作为圆心
        # 这里不绘制，避免重复
        
        best_centers = res.x
        
        # 贝叶斯优化只返回最佳中心位置，不进行最终的GSAMD优化
        # GSAMD优化将在贝叶斯优化完成后单独调用
        return res, best_centers, optimization_history
    
    def optimize_centers(self, EG_search_area=10e-6, EG_N_modes=(10,10), EG_n_calls=100, EG_inner_iter=15,
                         RS_search_area=2e-6, RS_N_modes=(20,20), RS_n_calls=100, RS_inner_iter=15,
                         plot_optimization=True, output_dir=None):
        """
        两阶段贝叶斯优化函数：初步估算阶段（EG）+ 精细搜索阶段（RS）
        
        :param EG_search_area: EG阶段搜索区域大小（大范围）
        :param EG_N_modes: EG阶段HG模式数量，默认(10,10)
        :param EG_n_calls: EG阶段贝叶斯优化调用次数，默认100
        :param EG_inner_iter: EG阶段内部迭代次数，默认15
        :param RS_search_area: RS阶段搜索区域大小（小范围，围绕EG结果）
        :param RS_N_modes: RS阶段HG模式数量，默认(20,20)
        :param RS_n_calls: RS阶段贝叶斯优化调用次数，默认100
        :param RS_inner_iter: RS阶段内部迭代次数，默认15
        :param plot_optimization: 是否绘制优化过程
        :param output_dir: 输出目录路径
        :return: C_final, best_centers, optimization_history
        """
        print("="*60)
        print("Stage 1: Educated Guess (EG) - 初步估算阶段")
        print("="*60)
        print(f"Search area: ±{EG_search_area*1e6:.2f} μm")
        print(f"HG modes: {EG_N_modes}")
        print(f"Bayesian calls: {EG_n_calls}")
        print(f"Inner iterations: {EG_inner_iter}")
        
        # 第一阶段：初步估算（EG）- 贝叶斯优化只找最佳中心位置
        res_EG, best_centers_EG, history_EG = self.optimize_centers_single_stage(
            search_area=EG_search_area,
            N_modes=EG_N_modes,
            n_calls=EG_n_calls,
            inner_iter=EG_inner_iter,
            plot_optimization=False,  # EG阶段不在这里绘制，避免SUMMARY输出
            output_dir=output_dir,
            stage_prefix="EG_"
        )
        
        # EG阶段完成后，绘制优化结果并保存（不显示，避免阻塞进程）
        print("\nSaving EG stage optimization results...")
        self.plot_optimization_results(res_EG, history_EG, EG_search_area, 
                                      output_dir=output_dir, stage_prefix="EG_", show_plot=False)
        
        print("\n" + "="*60)
        print("Stage 2: Refined Search (RS) - 精细搜索阶段")
        print("="*60)
        print(f"Search area: ±{RS_search_area*1e6:.2f} μm (around EG result)")
        print(f"HG modes: {RS_N_modes}")
        print(f"Bayesian calls: {RS_n_calls}")
        print(f"Inner iterations: {RS_inner_iter}")
        print(f"EG best centers: {np.array(best_centers_EG) * 1e6} μm")
        
        # 第二阶段：精细搜索（RS），基于EG结果 - 贝叶斯优化只找最佳中心位置
        # 注意：best_centers_EG格式为[x0_1, y0_1, x0_2, y0_2, ...]（单位：米）
        res_RS, best_centers_RS, history_RS = self.optimize_centers_single_stage(
            search_area=RS_search_area,
            N_modes=RS_N_modes,
            n_calls=RS_n_calls,
            inner_iter=RS_inner_iter,
            plot_optimization=False,  # RS阶段不在这里绘制，在下面单独绘制以便传入EG最佳位置
            output_dir=output_dir,
            stage_prefix="RS_",
            initial_centers=best_centers_EG
        )
        
        # RS阶段完成后，绘制优化结果并保存（使用EG的最佳位置作为圆心）
        if plot_optimization:
            print("\nSaving RS stage optimization results...")
            print(f"Using EG best centers as circle center: {np.array(best_centers_EG) * 1e6} μm")
            print(f"RS search area: ±{RS_search_area*1e6:.2f} μm")
            self.plot_optimization_results(res_RS, history_RS, RS_search_area,
                                          output_dir=output_dir, stage_prefix="RS_", 
                                          show_plot=False, circle_center=best_centers_EG)
        
        print("\n" + "="*60)
        print("Bayesian Optimization Summary")
        print("="*60)
        print(f"EG best error: {min(history_EG['errors']):.6f}")
        print(f"RS best error: {min(history_RS['errors']):.6f}")
        print(f"Final best centers: {np.array(best_centers_RS) * 1e6} μm")
        print("="*60)
        print("\nNote: Bayesian optimization only finds the best center positions.")
        print("GSAMD optimization (coefficient refinement) should be called separately.")
        print("="*60)
        
        # 贝叶斯优化只返回最佳中心位置，不返回系数
        # GSAMD优化需要单独调用
        return None, best_centers_RS, {'EG': history_EG, 'RS': history_RS}

    def plot_optimization_results(self, res, history, search_area, output_dir=None, stage_prefix="", 
                                  show_plot=False, circle_center=None):
        """
        简化的贝叶斯优化可视化
        
        :param res: 贝叶斯优化结果（skopt OptimizeResult对象，可能为None）
        :param history: 优化历史记录
        :param search_area: 搜索区域大小
        :param output_dir: 输出目录路径（如果为None，使用默认路径）
        :param stage_prefix: 阶段前缀（如"EG_"或"RS_"）
        :param show_plot: 是否显示图片（默认False，只保存）
        :param circle_center: 搜索圆的圆心位置（如果为None，使用(0,0)；对于RS阶段，应传入EG的最佳位置）
        """
        import matplotlib.pyplot as plt
        
        errors = history['errors']
        iterations = history['iteration']
        params_array = np.array(history['params'])
        
        # 计算需要的子图数量：每个平面一个参数空间图 + 一个误差图
        n_planes = len(self.z_list)
        n_cols = min(3, n_planes + 1)  # 最多3列
        n_rows = (n_planes + 1 + n_cols - 1) // n_cols  # 向上取整
        
        fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
        
        # 1. 误差收敛图
        plt.subplot(n_rows, n_cols, 1)
        plt.plot(iterations, errors, 'b-o', markersize=4)
        plt.axhline(y=min(errors), color='r', linestyle='--', alpha=0.7, 
                    label=f'Best: {min(errors):.6f}')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title('Optimization Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 每个平面的参数空间图
        for plane_idx in range(n_planes):
            plt.subplot(n_rows, n_cols, plane_idx + 2)
            
            # 提取当前平面的x0和y0参数（这些是绝对值，单位是米）
            x0_params = params_array[:, plane_idx * 2] * 1e6      # 转换为微米
            y0_params = params_array[:, plane_idx * 2 + 1] * 1e6
            
            # 散点图，颜色映射到误差
            scatter = plt.scatter(x0_params, y0_params, c=errors, cmap='viridis_r', 
                                s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
            
            # 标记最优点
            best_idx = np.argmin(errors)
            plt.scatter(x0_params[best_idx], y0_params[best_idx], 
                       c='red', s=150, marker='*', edgecolors='black', linewidth=2)
            
            # 添加搜索区域边界圆
            from matplotlib.patches import Circle
            # 确定圆心位置：如果提供了circle_center，使用它；否则使用(0,0)
            if circle_center is not None:
                # circle_center是一个数组，格式为[x0_1, y0_1, x0_2, y0_2, ...]（单位：米）
                # 确保circle_center是numpy数组
                circle_center_arr = np.array(circle_center)
                if len(circle_center_arr) > plane_idx * 2 + 1:
                    center_x = circle_center_arr[plane_idx * 2] * 1e6      # 转换为微米
                    center_y = circle_center_arr[plane_idx * 2 + 1] * 1e6
                else:
                    # 如果数组长度不够，使用(0,0)
                    center_x, center_y = 0, 0
                    print(f"Warning: circle_center array length ({len(circle_center_arr)}) is insufficient for plane {plane_idx+1}")
            else:
                center_x, center_y = 0, 0
            
            # 绘制搜索区域圆（半径是search_area）
            circle = Circle((center_x, center_y), search_area*1e6, fill=False, color='red', 
                           linestyle='--', alpha=0.5, linewidth=2, label='Search Area')
            plt.gca().add_patch(circle)
            
            # 标记圆心位置（如果不在原点）
            if circle_center is not None:
                plt.scatter(center_x, center_y, c='blue', s=100, marker='+', 
                           linewidths=3, label='Search Center (EG Best)')
                plt.legend()
            
            # 添加调试信息：显示实际搜索范围
            if circle_center is not None:
                # 计算实际搜索范围
                x_min = center_x - search_area*1e6
                x_max = center_x + search_area*1e6
                y_min = center_y - search_area*1e6
                y_max = center_y + search_area*1e6
                # 在图上添加文本说明
                plt.text(0.02, 0.98, f'Search: [{x_min:.2f}, {x_max:.2f}] μm\nCenter: ({center_x:.2f}, {center_y:.2f}) μm',
                        transform=plt.gca().transAxes, fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.xlabel('x0 (μm)')
            plt.ylabel('y0 (μm)')
            plt.title(f'Parameter Space - Plane {plane_idx+1}')
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            
            # 添加颜色条（只在最后一个子图添加）
            if plane_idx == n_planes - 1:
                plt.colorbar(scatter, label='Error')
        
        plt.tight_layout()
        
        # 确定保存路径
        if output_dir is None:
            save_path = r'F:\code\GSAMD\Figure\optimization_results.png'
        else:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f'{stage_prefix}bayesian_optimization_results.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Optimization plot saved to: {save_path}")
        
        # 只在需要时显示图片（默认不显示，避免阻塞进程）
        if show_plot:
            plt.show()
        else:
            plt.close(fig)  # 关闭图形，释放内存
        
        # 打印优化总结
        print(f"\n{'='*40}")
        print(f"OPTIMIZATION SUMMARY")
        print(f"{'='*40}")
        print(f"Total evaluations: {len(errors)}")
        print(f"Best error: {min(errors):.6f}")
        # 获取最佳参数：优先使用res.x，如果res为None则从history中获取
        if res is not None and hasattr(res, 'x'):
            best_params = np.array(res.x) * 1e6
        else:
            # 从history中找到误差最小的参数
            best_idx = np.argmin(errors)
            best_params = np.array(history['params'][best_idx]) * 1e6
        print(f"Best parameters (μm): {best_params}")
        if len(errors) > 0 and errors[0] > 0:
            print(f"Improvement: {(errors[0] - min(errors))/errors[0]*100:.1f}%")
        print(f"{'='*40}")
    
    def run(self, N_modes=(30,30)):
        """
        主流程：使用当前中心位置进行GSAMD优化
        
        :param N_modes: HG模式数量
        :return: 优化后的HG模式系数C
        """
        C = np.zeros((N_modes[0], N_modes[1]), dtype=complex)

        x0_list = self.x0_list
        y0_list = self.y0_list

        C = self.initialize_coefficients(C, N_modes)
        C = self.update_coefficients(C, x0_list, y0_list, self.max_iter)
        
        return C
    
    def refine_coefficients(self, best_centers, N_modes=(30,30), max_iter=None):
        """
        在贝叶斯优化找到的最佳中心位置基础上，进行GSAMD系数精细优化
        
        :param best_centers: 贝叶斯优化找到的最佳中心位置（一维数组，格式：[x0_1, y0_1, x0_2, y0_2, ...]）
        :param N_modes: HG模式数量
        :param max_iter: 最大迭代次数，如果为None则使用self.max_iter
        :return: 优化后的HG模式系数C
        """
        if max_iter is None:
            max_iter = self.max_iter
        
        x0_list = best_centers[::2]
        y0_list = best_centers[1::2]
        
        print("\n" + "="*60)
        print("GSAMD Coefficient Refinement")
        print("="*60)
        print(f"Using best centers from Bayesian optimization:")
        for k in range(len(x0_list)):
            print(f"  Plane {k+1}: x0={x0_list[k]*1e6:.3f} μm, y0={y0_list[k]*1e6:.3f} μm")
        print(f"HG modes: {N_modes}")
        print(f"Max iterations: {max_iter}")
        print("="*60)
        
        C = np.zeros((N_modes[0], N_modes[1]), dtype=complex)
        C = self.initialize_coefficients(C, N_modes)
        C = self.update_coefficients(C, x0_list, y0_list, max_iter=max_iter)
        
        # 计算最终误差
        final_error = self.calculate_error(C, x0_list, y0_list)
        print(f"\nFinal reconstruction error: {final_error:.6f}")
        print("="*60)
        
        return C
    
    def calculate_error(self, C, x0_list, y0_list):
        """
        计算重建误差 chi^2（用于优化目标）
        
        误差定义（Reconstruction Error）：
        $$\chi^2 = \sum_{k=0}^{N_{images}-1} \frac{\sqrt{\sum_{i=1}^{N_{pix_x} \times N_{pix_y}} (F_{exp}(x,y,z_k) - F_{fit}(x,y,z_k))^2}}{N_{images} \sum_{i=1}^{N_{pix_x} \times N_{pix_y}} F_{exp}(x,y,z_k)}$$
        
        其中：
        - F_exp: 实验测量的荧光图像（光强分布）
        - F_fit: 通过模式分解重构出的荧光图像（|E|^2）
        - N_images: 测量平面数量
        
        该公式本质上是所有测量平面上归一化残差的平均值。
        
        :param C: HG模式系数
        :param x0_list: x0参数列表
        :param y0_list: y0参数列表
        :return: chi^2 误差值
        """
        N_images = len(self.z_list)
        total_error = 0
        
        for k, z in enumerate(self.z_list):
            E = self.reconstruct_field(C, x0_list[k], y0_list[k], z)
            F_fit = np.abs(E)**2  # 重构的光强分布
            F_exp = self.F_list[k]  # 实验测量的光强分布
            
            # 计算每个平面的归一化残差
            # sqrt(sum((F_exp - F_fit)^2)) / (N_images * sum(F_exp))
            numerator = np.sqrt(np.sum((F_exp - F_fit)**2))
            denominator = N_images * np.sum(F_exp)
            
            if denominator > 0:
                error_k = numerator / denominator
            else:
                error_k = float('inf')  # 避免除零
            
            total_error += error_k
        
        return total_error



def save_bayesian_result(gsamd_instance, C, x0_list, y0_list, iteration_num, output_dir, stage_prefix="", save_images=True):
    """
    保存贝叶斯优化每次迭代的多平面场信息（适配不同数目的平面）
    
    :param gsamd_instance: GSAMD实例
    :param C: HG模式系数
    :param x0_list: x0参数列表
    :param y0_list: y0参数列表
    :param iteration_num: 迭代编号
    :param output_dir: 输出目录路径
    :param stage_prefix: 阶段前缀（如"EG_"或"RS_"）
    :param save_images: 是否保存图片（图片保存较慢，可以只保存CSV）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 重建所有平面的电场
    E_list = []
    for k in range(len(gsamd_instance.z_list)):
        E = gsamd_instance.reconstruct_field(C, x0_list[k], y0_list[k], gsamd_instance.z_list[k])
        E_list.append(E)
    
    # 保存每个平面的CSV和图片
    for k, E in enumerate(E_list):
        plane_num = k + 1
        intensity = np.abs(E)**2
        phase = np.angle(E)
        
        # 保存CSV文件（CSV保存较快）
        intensity_path = os.path.join(output_dir, f'{stage_prefix}bayesian_iter_{iteration_num}_plane{plane_num}_intensity.csv')
        phase_path = os.path.join(output_dir, f'{stage_prefix}bayesian_iter_{iteration_num}_plane{plane_num}_phase.csv')
        
        np.savetxt(intensity_path, intensity, delimiter=",")
        np.savetxt(phase_path, phase, delimiter=",")
        
        # 保存图片（可选，图片保存较慢）
        if save_images:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 强度图
            im1 = axes[0].imshow(intensity, cmap='jet')
            axes[0].set_title(f'Plane {plane_num} Intensity (Iter {iteration_num})')
            axes[0].set_xlabel('X')
            axes[0].set_ylabel('Y')
            plt.colorbar(im1, ax=axes[0])
            
            # 相位图
            im2 = axes[1].imshow(phase, cmap='hsv')
            axes[1].set_title(f'Plane {plane_num} Phase (Iter {iteration_num})')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            plt.colorbar(im2, ax=axes[1])
            
            plt.tight_layout()
            img_path = os.path.join(output_dir, f'{stage_prefix}bayesian_iter_{iteration_num}_plane{plane_num}.png')
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close(fig)


def write_optimization_log(params, C, error_val, log_dir="logs"):
    """
    优化日志记录函数（简化版）
    格式示例：
    [2023-08-20 14:30:00] x0=1.23e-6, y0=-2.34e-6, ..., error=0.045, C_real=[1.0, 0.5, ...], C_imag=[0.1, -0.2, ...]
    """
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, "optimization.log")
    
    # 生成时间戳和参数标签
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    param_labels = [f"param{i+1}" for i in range(len(params))]
    
    # 格式化参数和系数
    param_str = ", ".join([f"{k}={v:.3e}" for k, v in zip(param_labels, params)])
    C_str = ", ".join([f"{x:.3e}" for x in C.flatten()])
    
    # 写入日志（自动追加）
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{timestamp};\n {param_str};\n error={error_val:.6e};\n "
            f"C=[{C_str}];\n"
        )



# ============================================================================
# 贝叶斯优化配置参数（可在本脚本中修改）
# ============================================================================

# 初步估算阶段（Educated Guess, EG）参数
EG_SEARCH_AREA = 0.1e-6          # EG阶段搜索区域大小（米），大范围粗找
EG_N_MODES = (25, 25)           # EG阶段HG模式数量
EG_N_CALLS = 10                # EG阶段贝叶斯优化调用次数
EG_INNER_ITER = 60              # EG阶段内部迭代次数

# 精细搜索阶段（Refined Search, RS）参数
RS_SEARCH_AREA = 0.1e-6           # RS阶段搜索区域大小（米），小范围精细扫描（围绕EG结果）
RS_N_MODES = (12, 12)           # RS阶段HG模式数量
RS_N_CALLS = 10                # RS阶段贝叶斯优化调用次数
RS_INNER_ITER = 60              # RS阶段内部迭代次数

# 其他参数
PLOT_OPTIMIZATION = True         # 是否绘制优化过程
SKIP_RS_STAGE = True             # True: 只执行EG单阶段搜索; False: 执行EG+RS两阶段搜索
SKIP_BAYESIAN = True            # True: 跳过贝叶斯优化，直接用初始中心进行GS迭代


# 示例用法
if __name__ == "__main__":
    # 导入配置和运行模块
    from config_gsamd import get_default_config
    from run_gsamd import run_optimization, save_results
    
    # 获取配置
    config = get_default_config()
    
    # 运行优化（支持任意数量的测量平面）
    results = run_optimization(config)
    
    # 保存第一个平面的电场结果
    if results['E_list']:
        save_results(results['E_list'][0], config['output_path'])


