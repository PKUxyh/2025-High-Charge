import numpy as np


def crop_around_max(data, crop_size):
    """
    以图像最大值为中心进行裁剪
    
    :param data: 2D numpy数组（光强分布）
    :param crop_size: 裁剪尺寸，可以是int（正方形）或tuple (rows, cols)
    :return: 裁剪后的2D numpy数组
    """
    if crop_size is None:
        return data
    
    # 统一为 (crop_h, crop_w) 格式
    if isinstance(crop_size, int):
        crop_h, crop_w = crop_size, crop_size
    else:
        crop_h, crop_w = crop_size[0], crop_size[1]
    
    h, w = data.shape
    
    # 找到最大值位置
    max_y, max_x = np.unravel_index(np.argmax(data), data.shape)
    
    # 计算裁剪起止位置
    start_y = max_y - crop_h // 2
    start_x = max_x - crop_w // 2
    
    # 防止越界
    if start_y < 0:
        start_y = 0
    if start_x < 0:
        start_x = 0
    if start_y + crop_h > h:
        start_y = h - crop_h
    if start_x + crop_w > w:
        start_x = w - crop_w
    
    return data[start_y:start_y + crop_h, start_x:start_x + crop_w]


def crop_data_list(data_list, crop_size):
    """
    对多个平面的数据进行裁剪
    
    :param data_list: 数据列表 [F1, F2, F3, ...]
    :param crop_size: 裁剪尺寸，None表示不裁剪，int或tuple表示裁剪大小
    :return: 裁剪后的数据列表
    """
    if crop_size is None:
        return data_list
    
    return [crop_around_max(d, crop_size) for d in data_list]
