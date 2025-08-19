import os
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import statistics
from typing import List, Optional

import sys



# 获取 TAM_feature.py 所在的路径并添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录（utils）
parent_dir = os.path.dirname(current_dir)  # 项目根目录
extractor_dir = os.path.join(parent_dir, "Extractor")
sys.path.append(extractor_dir)

# 现在可以导入 TAM_feature.py 中的函数
from TAM_feature import packets_perslot

TIME_END = 100
SPLIT_MARK = '\t'


def find_inflection_indices(
    seq: List[int],
    threshold: Optional[float] = None,
    zscore: float = 2.0
) -> List[int]:
    """
    Identify indices where the absolute difference between consecutive elements exceeds a threshold.

    Args:
        seq: List of integer values representing the sequence.
        threshold: If provided, use this absolute difference as the cutoff for inflection.
        zscore: Multiplier for standard deviation when threshold is not provided.

    Returns:
        A list of indices in `seq` where an inflection (sudden change) occurs.
    """
    if len(seq) < 2:
        return []

    # Compute absolute differences between consecutive points
    diffs = [abs(seq[i] - seq[i - 1]) for i in range(1, len(seq))]

    # Determine threshold automatically if not specified
    if threshold is None:
        mean_diff = statistics.mean(diffs)
        # Use sample standard deviation; if only one diff, stdev = 0
        std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        threshold = mean_diff + zscore * std_diff
    
    #print("threshold:", threshold)

    # An inflection at position i means seq[i] differs sharply from seq[i-1]
    inf_indices = [i for i, d in enumerate(diffs, start=1) if d > threshold]
    return inf_indices

def segment_by_inflection(
    seq: List[int],
    threshold: Optional[float] = None,
    zscore: float = 1.0
) -> List[List[int]]:
    """
    Segment the sequence into sublists between inflection points.

    Args:
        seq: List of integer values representing the sequence.
        threshold: If provided, use this absolute difference as the cutoff for inflection.
        zscore: Multiplier for standard deviation when threshold is not provided.

    Returns:
        A list of segments (each a list) where each segment runs from one inflection point (exclusive)
        up to the next inflection point (inclusive). The first segment starts at the beginning of the sequence.
    """
    inf_indices = find_inflection_indices(seq, threshold, zscore)
    segments: List[List[int]] = []
    start = 0

    for idx in inf_indices:
        # Include the inflection point at the end of the current segment
        segments.append(seq[start: idx + 1])
        start = idx + 1

    # Add the final segment after the last inflection
    if start < len(seq):
        segments.append(seq[start:])

    return segments

def count_file_seq(file_path):
    timestamps = []
    packets = []

    packet_sizes = []
    packet_sizes_abs = []
    packet_count = []
    pktcount = 0
    
    # 读取文件内容
    with open(file_path, 'r') as file:
        accu = 0
        accu_abs = 0
        pc = 0
        for line in file:
            
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    # 转换时间戳和数据包大小
                    ts = float(parts[0])
                    ps = float(parts[1])
                    
                    if ts < TIME_END:
                        pktcount+=1
                        timestamps.append(ts)
                        accu += ps
                        packet_sizes.append(accu)
                        accu_abs += abs(ps)
                        packet_sizes_abs.append(accu_abs)
                        pc += 1
                        packet_count.append(pc)
                        packets.append(ps)
                except ValueError:
                    continue
    return (timestamps, packets, packet_sizes, packet_sizes_abs, packet_count, pktcount)

def count_smooth_file_seq(file_path):
    timestamps = []
    packets = []

    packet_sizes = []
    packet_sizes_abs = []
    packet_count = []
    pktcount = 0

    # 读取文件内容
    with open(file_path, 'r') as file:
        accu = 0
        accu_abs = 0
        pc = 0

        for line in file:
            
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    # 转换时间戳和数据包大小
                    ts = float(parts[0])
                    ps = float(parts[1])
                    
                    if ts < TIME_END:
                        timestamps.append(ts)
                        packets.append(ps)
                except ValueError:
                    continue
    
    # from Momentum_feature import remove_flat_segments
    # [timestamps, packets] = remove_flat_segments(
    #         timestamps, 
    #         packets,
    #         rm_threshold=None,
    #         rm_zscore=0.4)

    for ts, ps in zip(timestamps, packets):
        if ts < TIME_END:
            pktcount+=1
            accu += ps
            packet_sizes.append(accu)
            accu_abs += abs(ps)
            packet_sizes_abs.append(accu_abs)
            pc += 1
            packet_count.append(pc)

    return (timestamps, packets, packet_sizes, packet_sizes_abs, packet_count, pktcount)

def plot_packet_series(folder_paths, webnum, show_inflections=False, inflection_params=None, flat=True):
    """
    绘制数据包时间序列图，可选是否显示分片点
    
    参数:
        folder_paths: 文件夹路径列表
        webnum: 每个文件夹要绘制的文件数
        show_inflections: 是否显示分片点 (默认为False)
        inflection_params: 分片检测参数字典 (可选)
            - window_size: 滑动窗口大小 (默认3)
            - threshold_multiplier: 阈值乘数 (默认2)
    """
    # 设置默认分片参数
    if inflection_params is None:
        #inflection_params = {'window_size': 3, 'threshold_multiplier': 2}
        inflection_params = {'threshold': None, 'zscore': 3.5}
    
    # 创建颜色映射
    colors = cm.get_cmap('tab20', len(folder_paths))
    
    plt.figure(figsize=(10, 7))

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 设置中文字体 [13, 9]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题 [8]

    tick_label_fontsize = 18
    plt.rcParams['xtick.labelsize'] = tick_label_fontsize  # 设置x轴刻度字体大小 [2, 12]
    plt.rcParams['ytick.labelsize'] = tick_label_fontsize  # 设置y轴刻度字体大小 [2, 12]
    
    # 存储每个文件夹的累积和信息
    folder_cumulative_info = {}
    
    # 存储所有拐点坐标用于后续绘制
    all_inflection_points = []
    
    for folder_idx, folder in enumerate(folder_paths):
        # 获取所有文件并过滤非文件项
        all_files = [f for f in os.listdir(folder) 
                    if os.path.isfile(os.path.join(folder, f))]
        
        # 随机选择webnum个文件
        selected_files = random.sample(all_files, webnum)

        selected_files = [
            '20250228_0306_57_hztsln4fi3udznlinmxnbwdey6lbehn4sinqa6ltbu4crxgqnlzdqoid286.dat',
            '20250228_0035_14_hztsln4fi3udznlinmxnbwdey6lbehn4sinqa6ltbu4crxgqnlzdqoid216.dat',
            '20250228_0251_47_hztsln4fi3udznlinmxnbwdey6lbehn4sinqa6ltbu4crxgqnlzdqoid279.dat',
            '20250227_2233_52_hztsln4fi3udznlinmxnbwdey6lbehn4sinqa6ltbu4crxgqnlzdqoid160.dat'
        ]
        
        # 生成当前文件夹的颜色
        folder_color = colors(folder_idx)
        folder_name = os.path.basename(folder)
        
        # 存储当前文件夹的平均累积和
        cumulative_sums = []
        cumulative_abs = []
        cumulative_pks = []
        
        segments_len_lsit = []

        # 遍历选中的文件
        for file_idx, filename in enumerate(selected_files):
            
            file_path = os.path.join(folder, filename)
            
            if flat:
                timestamps, packets, packet_sizes, packet_sizes_abs, packet_count, pktcount = count_smooth_file_seq(file_path=file_path,)
            else:
                timestamps, packets, packet_sizes, packet_sizes_abs, packet_count, pktcount = count_file_seq(file_path)
                

            #print(f"web {file_path} have {pktcount} packets.")
            
            # 记录当前文件的累积和
            if packet_sizes:
                cumulative_sums.append(packet_sizes[-1])
            if packet_sizes_abs:
                cumulative_abs.append(packet_sizes_abs[-1])
            if packet_count:
                cumulative_pks.append(packet_count[-1])
            
            # 绘制完整的时间序列
            label = folder_name if file_idx == 0 else None
            plt.plot(timestamps, packet_sizes,
                   color=folder_color,
                   alpha=0.7,
                   linewidth=3,
                   label=label)
            
            if False:
                # 在每条线末尾添加文件名标签
                if timestamps and packet_sizes:  # 确保有数据
                    filename_noext = os.path.splitext(filename)[0]  # 去掉后缀
                    short_name = filename_noext[-5:]  # 取最后5个字符
                    plt.text(timestamps[-1], packet_sizes[-1], 
                            short_name, 
                            color=folder_color,
                            fontsize=9,
                            va='center', ha='left')
            
            # 如果需要显示分片点
            # if show_inflections:

            segments = segment_by_inflection(
                packets,
                threshold=inflection_params.get('threshold'),
                zscore=inflection_params.get('zscore')
            )

            segments_counts = count_elements(segments)
            # print(f"segments counts: {segments_counts}, time stamp counts: {len(timestamps)}")
            
            start_idx = 0
            segment_len = 0
            for seg_idx, segment in enumerate(segments):
                segment_len += len(segment)
                end_idx = start_idx + len(segment)
                if seg_idx > 0:  # 跳过第一个分片的起点
                    point_time = timestamps[start_idx]
                    point_value = packet_sizes[start_idx]
                    all_inflection_points.append((point_time, point_value, folder_color))
                start_idx = end_idx
            
            #print(f"web {file_path} have {len(segments)} segments, have {segment_len} packets.")
            segments_len_lsit.append(len(segments))
        print(segments_len_lsit)
        
        # 计算并存储当前文件夹的平均累积和
        if cumulative_sums:
            avg_cumulative = sum(cumulative_sums) / len(cumulative_sums)
            avg_cumulabs = sum(cumulative_abs) / len(cumulative_abs)
            avg_cumulpks = sum(cumulative_pks) / len(cumulative_pks)
            folder_cumulative_info[folder_name] = (avg_cumulative, avg_cumulabs, avg_cumulpks)

    # 如果需要显示分片点，绘制所有拐点
    if show_inflections and all_inflection_points:
        
        for point_time, point_value, color in all_inflection_points:
            
            plt.scatter(point_time, point_value,
                      color=color,  # 使用原线条颜色
                      #edgecolors='red',  # 红色边框
                      s=30,  # 点的大小
                      linewidths=1.5,  # 边框宽度
                      zorder=5)  # 确保点在线上方

    # 图表装饰
    plt.xlabel('时间戳', fontsize=20)
    plt.ylabel('包长累积和', fontsize=20)
    title = 'Packet Size Time Series'
    if show_inflections:
        title += ' with Inflection Points'
    # plt.title(title, fontsize=16)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 智能调整横坐标范围
    if 'timestamps' in locals() and timestamps:  # 检查timestamps是否定义且非空
        plt.xlim(left=min(timestamps), right=max(timestamps) + 1)
    
    # 创建自定义图例
    legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
    
    # 添加累积和信息到图例标签
    new_labels = []
    for label in legend_labels:
        if label in folder_cumulative_info:
            avg_val = folder_cumulative_info[label]
            new_labels.append(f"{label[:5]}, Avg: {avg_val[0]:.1f}, {avg_val[1]:.1f}, {avg_val[2]:.1f}")
        else:
            new_labels.append(label)
    
    # 如果需要显示分片点，添加图例项
    if show_inflections:
        inflection_legend = plt.Line2D([20], [20], 
                                     marker='o', 
                                     color='w', 
                                     label='Inflection Points',
                                     markerfacecolor='w',
                                     markeredgecolor='red',
                                     markersize=10)
        legend_handles.append(inflection_legend)
        new_labels.append("Inflection Points")
    
    # 图例设置
    # plt.legend(
    #     legend_handles,
    #     new_labels,
    #     title='Data Categories (with average cumulative values)',
    #     bbox_to_anchor=(1.05, 1),
    #     loc='upper left',
    #     frameon=True
    # )
    
    plt.tight_layout()
    plt.show()

def find_inflection_points(sequence, window_size=3, threshold_multiplier=2):
    '''
    分析序列的局部变化来检测拐点。具体来说，我们计算每个点的变化率（差分），当某个点的变化率显著超过前面一定窗口内的平均变化率时，将其视为拐点。

    - window_size: 计算局部平均变化的窗口大小（默认为3）。
    - threshold_multiplier: 判断拐点的阈值倍数（默认为2）。

    处理步骤：
    1. 差分计算：首先计算序列的一阶差分，得到每个相邻元素的变化量。
    2. 滑动窗口检测：对于每个点，计算前面window_size个差分绝对值的平均值。如果当前点的差分绝对值超过该平均值的一定倍数（由threshold_multiplier决定），则视为拐点。
    3. 结果分割：将检测到的拐点与序列起始、结束位置结合，分割原序列为多个子序列。
    '''
    if len(sequence) < 2:
        return [sequence.copy()]
    
    diffs = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]
    inflection_indices = []
    
    for i in range(window_size, len(diffs)):
        window = diffs[i - window_size : i]
        avg_abs_diff = sum(abs(x) for x in window) / window_size
        current_abs = abs(diffs[i])
        
        if current_abs > avg_abs_diff * threshold_multiplier:
            # 拐点对应原序列的位置是i+1（即sequence[i+1]是拐点后的第一个点）
            inflection_indices.append(i + 1)
    
    # 关键修正：最后一个分割点必须是原序列长度
    split_points = [0] + inflection_indices + [len(sequence)]  # 修正右括号错误
    
    # 分割原序列并确保连续性
    segments = []
    for start, end in zip(split_points[:-1], split_points[1:]):
        segment = sequence[start:end]
        if segment:  # 避免空片段
            segments.append(segment)
    
    return segments

def get_HSdir(inputDir, format='pcap'):
    HSdir = [] # 待检测的目录
    for root, dirs, files in os.walk(inputDir):
        for file in files:
            if file.endswith('.'+format):
                if root not in HSdir: HSdir.append(root)
    
    return HSdir

def add_dat_extension(folder_path):
    """
    为指定文件夹下的所有文件添加.dat扩展名
    
    参数:
        folder_path (str): 要处理的文件夹路径
    """
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # 获取文件完整路径
            file_path = os.path.join(root, filename)
            
            # 如果文件已经有.dat扩展名，则跳过
            if filename.endswith('.dat'):
                continue
                
            # 构造新文件名
            new_filename = filename + '.dat'
            new_path = os.path.join(root, new_filename)
            
            # 重命名文件
            try:
                os.rename(file_path, new_path)
                print(f"重命名: {file_path} -> {new_path}")
            except Exception as e:
                print(f"无法重命名 {file_path}: {e}")

# 计算list有多少个元素
def count_elements(lst):
    if not isinstance(lst, list):
        return 1
    return sum(count_elements(item) for item in lst)

def count_segments_length(folder_paths, webnum, threshold, zscore):
    for folder_idx, folder in enumerate(folder_paths):
        all_files = [f for f in os.listdir(folder) 
                    if os.path.isfile(os.path.join(folder, f))]
        selected_files = random.sample(all_files, webnum)

        seg_len_list = []

        for file_idx, filename in enumerate(selected_files):
            file_path = os.path.join(folder, filename)

            timestamps, packets, packet_sizes, packet_sizes_abs, packet_count, pktcount = count_file_seq(file_path)

            segments = segment_by_inflection(
                packets,
                threshold=threshold,
                zscore=zscore
            )

            seg_len_list.append(len(segments))
        
        print(seg_len_list)


# 使用示例
if __name__ == "__main__":
    path = r'D:\2025HS_dataset\HS_longstream_flat'
    path = 'D:/2025HS_dataset/Temp/TEST_no_Flat'
    
    folders = [
        # r'D:\\2025HS_dataset\\HS_longstream\\Drugs\\Drugs\\Ilegal\\iwggpyxn6qv3b2twpwtyhi2sfvgnby2albbcotcysd5f7obrlwbdbkyd',
        # r'D:\\2025HS_dataset\\HS_longstream\\Drugs\\Drugs\\Ilegal\\rfyb5tlhiqtiavwhikdlvb3fumxgqwtg2naanxtiqibidqlox5vispqd',
        # r'D:\\2025HS_dataset\\HS_longstream\\Hacking\\Hacking\\prjd5pmbug2cnfs67s3y65ods27vamswdaw2lnwf45ys3pjl55h2gwqd',
        r'D:\2025HS_dataset\HS_longstream_flat\Marketplace\Black\hztsln4fi3udznlinmxnbwdey6lbehn4sinqa6ltbu4crxgqnlzdqoid'
        # r'D:\2025HS_dataset\Temp\TEST_no_Flat_10web\7sk2kov2xwx6cbc32phynrifegg6pklmzs7luwcggtzrnlsolxxuyfyd',
        # r'D:\2025HS_dataset\Temp\TEST_no_Flat_10web\cryptbbtg65gibadeeo2awe3j7s6evg7eklserehqr4w4e2bis5tebid'
    ]
    # folders = random.sample(get_HSdir(path, 'dat'), 2)
    # plot_packet_series(folders, webnum=5, flat=True)
    # plot_packet_series(folders, webnum=5, flat=False)

    inflection_params = {'window_size': 5, 'threshold_multiplier': 2}
    inflection_params = {'threshold': None, 'zscore': 0.1}
    # plot_packet_series(folders, webnum=4, show_inflections=True, inflection_params=inflection_params)
    plot_packet_series(folders, webnum=5, show_inflections=False, inflection_params=inflection_params, flat=False)
    #print(segments_len_lsit)

    # count_segments_length(folders, 6, inflection_params['threshold'], inflection_params['zscore'])