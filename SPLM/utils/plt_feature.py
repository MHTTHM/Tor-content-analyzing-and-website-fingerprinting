import sys
import os
import pandas as pd

# 获取 TAM_feature.py 所在的路径并添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录（utils）
parent_dir = os.path.dirname(current_dir)  # 项目根目录
extractor_dir = os.path.join(parent_dir, "Extractor")
sys.path.append(extractor_dir)

# 现在可以导入 TAM_feature.py 中的函数
from TAM_feature import packets_perslot
from Momentum_feature import Momentum_feature

import matplotlib.pyplot as plt
import numpy as np

SPLIT_MARK = '\t' # '\t' ' '

def draw_multiple_intensity_maps(matrices):
    """
    绘制多个 2xN 整数矩阵的色块图，并将它们排列在同一张图上。

    参数：
        matrices (list of list of list of int): 多个 2xN 的矩阵，每个元素是一个 2 行的整数矩阵
    """
    num_matrices = len(matrices)

    # 创建子图，每个矩阵对应一行子图
    fig, axes = plt.subplots(num_matrices, 1, figsize=(10, 2 * num_matrices), squeeze=False)

    for idx, matrix in enumerate(matrices):
        data = np.array(matrix)

        if data.shape[0] != 2:
            raise ValueError(f"第 {idx+1} 个矩阵不是两行：实际形状为 {data.shape}")

        ax = axes[idx][0]
        im = ax.imshow(data, cmap='viridis', aspect='auto')

        # 设置 Y 轴标签为 "+1" 和 "-1"
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["+1", "-1"])

        #ax.set_title(f"Matrix {idx+1}")
        #ax.set_xlabel("Column Index")
        ax.set_ylabel("Row Index")

        # 添加 colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label('Value')

    plt.tight_layout()
    plt.show()

def read_file(filepath):
    with open(filepath, 'r') as f:
        tcp_dump = f.readlines()#[:MAX_TRACE_LENGHT]
    
    seq = pd.Series(tcp_dump).str.slice(0, -1).str.split(SPLIT_MARK, expand=True).astype("float")
    
    # 提取时间和包长序列
    times = np.array(seq.iloc[:, 0], dtype=np.float64)
    if times[0] != 0:
        times = times - times[0]
    length_seq = np.array(seq.iloc[:, 1], dtype=np.int32)
    return (times, length_seq)

def resize_to_N(matrix, N=100, fill_value=0):
    """
    将一个 2xN 的列表裁剪或填充为 2x100 的 numpy 数组。
    
    参数：
        matrix (list of list): 原始 2xN 列表
        fill_value (int): 若不足 100 列，用该值填充

    返回：
        np.ndarray: 形状为 (2, 100) 的数组
    """
    result = np.full((2, N), fill_value, dtype=int)  # 创建填充值数组
    for i in range(2):
        row = matrix[i]
        length = min(len(row), N)
        result[i, :length] = row[:length]  # 只填充前100个
    return result

if __name__=='__main__':
    files = [
        r'D:\2025HS_dataset\HS_longstream_flat\Marketplace\Black\hztsln4fi3udznlinmxnbwdey6lbehn4sinqa6ltbu4crxgqnlzdqoid\20250228_0306_57_hztsln4fi3udznlinmxnbwdey6lbehn4sinqa6ltbu4crxgqnlzdqoid286.dat',
        r'D:\2025HS_dataset\HS_longstream_flat\Marketplace\Black\hztsln4fi3udznlinmxnbwdey6lbehn4sinqa6ltbu4crxgqnlzdqoid\20250228_0035_14_hztsln4fi3udznlinmxnbwdey6lbehn4sinqa6ltbu4crxgqnlzdqoid216.dat',
        r'D:\2025HS_dataset\HS_longstream_flat\Marketplace\Black\hztsln4fi3udznlinmxnbwdey6lbehn4sinqa6ltbu4crxgqnlzdqoid\20250228_0251_47_hztsln4fi3udznlinmxnbwdey6lbehn4sinqa6ltbu4crxgqnlzdqoid279.dat',
        r'D:\2025HS_dataset\HS_longstream_flat\Marketplace\Black\hztsln4fi3udznlinmxnbwdey6lbehn4sinqa6ltbu4crxgqnlzdqoid\20250227_2233_52_hztsln4fi3udznlinmxnbwdey6lbehn4sinqa6ltbu4crxgqnlzdqoid160.dat'
    ]
    
    matrix = []

    for f in files:
        times, len_seq = read_file(f)
        TAM_feature_mtx = packets_perslot(times, len_seq)
        matrix.append(TAM_feature_mtx)
    
    params = {
        'threshold': None,
        'zscore': 0.16,
        'flat': False
    }
    for f in files:
        times, len_seq = read_file(f)
        Momentum_feature_mtx = Momentum_feature(times, len_seq, **params)
        print(len(Momentum_feature_mtx[0]))
        Momentum_feature_mtx = resize_to_N(Momentum_feature_mtx, 500)
        matrix.append(Momentum_feature_mtx)

    # 调用函数
    draw_multiple_intensity_maps(matrix)