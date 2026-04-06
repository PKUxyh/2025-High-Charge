import pandas as pd
from scipy.ndimage import zoom
from utils_gsamd import crop_center
# preprocess_fluence 已移除，避免重复处理（GSAMD类内部会处理）


def load_fluence_data(file_paths):
    """
    加载CSV数据文件
    
    :param file_paths: 文件路径列表，例如 [far_path, near_path]
    :return: 数据数组列表
    """
    data_list = []
    for path in file_paths:
        data = pd.read_csv(path, header=None).values
        data_list.append(data)
    return data_list


def prepare_data(data_list, apply_crop=False, crop_size=None, apply_zoom=False, zoom_factor=None):
    """
    数据预处理（crop、zoom等可选操作），支持任意数量的平面
    
    :param data_list: 数据数组列表
    :param apply_crop: 是否应用crop操作
    :param crop_size: crop尺寸，例如 (401, 401)
    :param apply_zoom: 是否应用zoom操作
    :param zoom_factor: zoom因子，例如 0.5
    :return: 预处理后的数据列表
    """
    result = []
    for F in data_list:
        if apply_crop and crop_size is not None:
            F = crop_center(F, crop_size[0], crop_size[1])
        if apply_zoom and zoom_factor is not None:
            F = zoom(F, zoom_factor, order=3)
        result.append(F)
    
    return result

