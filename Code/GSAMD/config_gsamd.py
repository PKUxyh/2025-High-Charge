def get_default_config():
    """
    返回默认参数字典
    
    :return: 包含所有配置参数的字典
    """
    config = {
        # 数据文件路径（支持任意数量的平面，与 z_positions 一一对应）
        # 'data_paths': [
        #     r'F:\实验\实验集\20250226\moniguang_0um_crop_0006.ascii.csv',
        #     r'F:\实验\实验集\20250226\moniguang_-100um_crop_0006.ascii.csv',
        #     r'F:\实验\实验集\20250226\moniguang_-250um_crop_0006.ascii.csv',
        # ],

        # 'data_paths': [
        #     r'F:\实验\实验集\20250226\moniguang_0um_crop_0009.ascii.csv',
        #     r'F:\实验\实验集\20250226\moniguang_-100um_crop_0009.ascii.csv',
        #     r'F:\实验\实验集\20250226\moniguang_-250um_crop_0009.ascii.csv',
        # ],

        'data_paths': [
            r'F:\code\GSAMD\Response-mng\intensity_z_0um.csv',
            r'F:\code\GSAMD\Response-mng\intensity_z_1000um.csv',
        ],        

        
        # 其他可选路径（已注释）
        # 'data_paths': [
        #     r'F:\paper\GS_new\GSA-MD\paper_used\far_recon_abs.csv',
        #     r'F:\paper\GS_new\GSA-MD\paper_used\near_recon_abs.csv',
        # ],

        # 其他可选路径（已注释）文章最开始用的都是0008
        # 'far_path': r'F:\实验\实验集\20250226\moniguang_0um__0006.ascii.csv',
        # 'near_path': r'F:\实验\实验集\20250226\moniguang_-100um__0006.ascii.csv',
        # 'near1_path': r'F:\实验\实验集\20250226\moniguang_-250um__0006.ascii.csv',
        # 'far_path': r'F:\paper\大电量电子加速文章2\高阶模激光模拟\真实激光参数\focal_spot.csv',
        # 'near_path': r'F:\paper\大电量电子加速文章2\高阶模激光模拟\真实激光参数\+1000um.csv',
        # 'far_path': r'F:\paper\GS_new\GSA-MD\低分辨率+贝叶斯\far_low.csv',
        # 'near_path': r'F:\paper\GS_new\GSA-MD\低分辨率+贝叶斯\near_low.csv',

        # 网格参数
        # 'grid_size': 4.4e-6 / 39.11,
        'grid_size': 4.4e-6 / 10,
        # 'grid_size': 0.88e-6,
        
        # 物理常数（虽然未直接使用，但保留在注释中）
        'c': 3e8,  # 光速
        'eps0': 8.854e-12,  # 真空介电常数
        'tau': 35e-15,  # 脉冲持续时间（假设）
        
        # 初始模式中心（列表，与 z_positions 一一对应）
        # 'x0_init_list': [0, 0, 0],
        # 'y0_init_list': [0, 0, 0],
        'x0_init_list': [0, 0],
        'y0_init_list': [0, 0],

        # 激光参数
        'wavelength': 0.8e-6,
        # 'w0x': 22.4e-6,
        # 'w0y': 29.3e-6,
        # 'w0x': 30e-6,
        # 'w0y': 30e-6,
        # 'w0x': 7e-6,
        # 'w0y': 7e-6,
        # 'wavelength': 0.785e-6
        'w0x': 10e-6,
        'w0y': 10e-6,
        
        # z坐标
        'z_positions': [0e-6, -1000e-6],
        # 'z_positions': [0e-6, -100e-6, -250e-6],
        
        # GSAMD参数
        'max_iter': 100,
        
        # 优化参数
        'search_area': 10e-6,
        'N_modes': (10, 10),
        'n_calls': 20,
        'inner_iter': 15,
        
        # 输出路径
        'output_path': r"F:\code\GSAMD\Figure\E_far.csv",
    }
    
    return config
