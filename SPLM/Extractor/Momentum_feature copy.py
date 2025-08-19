from typing import List, Optional, Tuple
import statistics
import math
import numpy as np

MAX_MMM_LENGTH = 8000

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
        try:
            mean_diff = statistics.mean(diffs)
            # Use a more robust standard deviation calculation
            if len(diffs) > 1:
                # Calculate variance manually to avoid potential issues
                # print(diffs)
                # print(mean_diff)
                sum_sq = sum((x - mean_diff) ** 2 for x in diffs)
                # Handle potential floating point precision issues
                sum_sq = max(0.0, sum_sq)
                std_diff = math.sqrt(sum_sq / (len(diffs) - 1))
            else:
                std_diff = 0.0
            threshold = mean_diff + zscore * std_diff
        except (ValueError, statistics.StatisticsError) as e:
            # Fallback to a simple threshold if calculation fails
            threshold = max(diffs) if diffs else 0.0

    # An inflection at position i means seq[i] differs sharply from seq[i-1]
    inf_indices = [i for i, d in enumerate(diffs, start=1) if d > threshold]
    return inf_indices

# from deepseek-R1
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
    # Validate input sequence
    if len(seq)<1:
        return []
    
    try:
        inf_indices = find_inflection_indices(seq, threshold, zscore)
    except Exception:
        # If inflection detection fails, return the whole sequence as one segment
        return [seq.copy()]
    
    segments: List[List[int]] = []
    start = 0

    for idx in inf_indices:
        # Include the inflection point at the end of the current segment
        segments.append(seq[start: idx])
        start = idx

    # Add the final segment after the last inflection
    if start < len(seq):
        segments.append(seq[start:])

    return segments

def Momentum_feature(times, sizes, **kwargs):
    # Validate input - properly handle numpy arrays
    if isinstance(sizes, np.ndarray):
        if sizes.size == 0:  # Check if numpy array is empty
            return [[], []]
        sizes = sizes.tolist()  # Convert to Python list for consistent processing
    elif not sizes:  # Handle regular Python sequences (list, tuple, etc.)
        return [[], []]
    
    if isinstance(times, np.ndarray):
        if times.size == 0:  # Check if numpy array is empty
            return [[], []]
        times = times.tolist()  # Convert to Python list for consistent processing
    elif not times:  # Handle regular Python sequences (list, tuple, etc.)
        return [[], []]
    
    threshold = kwargs.get('threshold', None)
    zscore = kwargs.get('zscore', 0.15)
    try:
        segments = segment_by_inflection(
            sizes,
            threshold=threshold,
            zscore=zscore
        )
    except Exception:
        # Fallback to treating the whole sequence as one segment
        segments = [sizes.copy() if isinstance(sizes, list) else sizes.tolist()]
    
    
    if len(segments) < MAX_MMM_LENGTH:
        MAX_MMM_LENGTH = len(segments)
    MMM = [[0 for _ in range(MAX_MMM_LENGTH)], [0 for _ in range(MAX_MMM_LENGTH)]]
    
    # 包长的ln(x)作为特征
    if False:
        for idx, segment in enumerate(segments):
            for i in segment:
                if i==0:
                    continue
                elif i > 0:
                    MMM[0][idx] += math.log10(i)
                    # MMM[0][idx] += 1
                else:
                    MMM[-1][idx] += math.log10(-i)
                    # MMM[-1][idx] += 1
    else:
        momentum = 0.1
        for idx, segment in enumerate(segments[:MAX_MMM_LENGTH]):
            for i in segment:
                if i==0:
                    continue
                elif i > 0:
                    MMM[0][idx] = (1-momentum)*MMM[0][idx] + momentum*i
                else:
                    MMM[-1][idx] = (1-momentum)*MMM[-1][idx] + momentum*(-i)
    
    return MMM

### 弃用
def remove_flat_segments(
    timestamps: List[float],
    lengths: List[int],
    flat_segment_threshold: Optional[float] = None,
    flat_segment_zscore: float = 0.3,
    rm_threshold: Optional[float] = None,
    rm_zscore: float = 0.4
) -> Tuple[List[float], List[int]]:
    """
    Remove leading and trailing flat segments based on rate of change.
    """
    if len(timestamps) != len(lengths):
        raise ValueError("Timestamps and lengths must have the same length.")
    if len(timestamps)<1:
        return [], []

    segments = segment_by_inflection(lengths, threshold=flat_segment_threshold, zscore=flat_segment_zscore)

    if not segments:
        return [], []

    # Calculate segment indices in original array
    seg_indices = []
    current = 0
    for seg in segments:
        seg_len = len(seg)
        end_idx = current + seg_len - 1
        seg_indices.append((current, end_idx))
        current = end_idx + 1

    # Calculate growth rate for each segment
    rates = []
    for seg, (start, end) in zip(segments, seg_indices):
        total_growth = sum(seg)
        time_span = timestamps[end] - timestamps[start]
        if time_span == 0:
            rate = float('inf') if total_growth != 0 else 0.0
        else:
            rate = total_growth / time_span
        rates.append(rate)

    # Determine rate threshold
    if rm_threshold is None:
        abs_rates = [abs(r) for r in rates if not (r == float('inf') or r == -float('inf'))]
        if not abs_rates:
            return [], []
        mean = statistics.mean(abs_rates)
        std = statistics.stdev(abs_rates) if len(abs_rates) > 1 else 0.0
        rm_threshold = mean + rm_zscore * std

    # Identify flat segments
    is_flat = []
    for rate in rates:
        if rate == float('inf') or rate == -float('inf'):
            is_flat.append(False)
        else:
            is_flat.append(abs(rate) < rm_threshold)

    # Find first and last non-flat segments
    first_non_flat = next((i for i, flat in enumerate(is_flat) if not flat), None)
    if first_non_flat is None:
        return [], []

    last_non_flat = next((i for i in reversed(range(len(is_flat))) if not is_flat[i]), None)

    # Get range in original arrays
    start_idx = seg_indices[first_non_flat][0]
    end_idx = seg_indices[last_non_flat][1]

    times = timestamps[start_idx:end_idx+1]
    times = [i-times[0] for i in times]

    return times, lengths[start_idx:end_idx+1]

# # 新的辅助函数：分段并返回带有原始索引信息的分段
# def segment_by_inflection_with_indices(
#     seq: List[int],
#     threshold: Optional[float] = None,
#     zscore: float = 1.0 # 与您原 segment_by_inflection 中的 zscore 默认值一致
# ) -> List[dict[str, any]]:
#     """
#     Segment the sequence by inflection points, returning segments along with their 
#     start and end indices in the original sequence, and their values.
#     Each returned dict: {'start_idx': int, 'end_idx': int, 'values': List[int]}
#     """
#     if not seq:
#         return []
    
#     try:
#         # inf_indices 指的是 seq[i] 与 seq[i-1] 发生显著变化的那个点 i
#         inf_indices = find_inflection_indices(seq, threshold, zscore)
#     except Exception:
#         # 如果拐点检测失败，将整个序列视为一个分段
#         return [{'start_idx': 0, 'end_idx': len(seq) - 1, 'values': seq[:]}] 
    
#     segments_info: List[dict[str, any]] = []
#     current_start_idx = 0

#     for inflection_point_idx in inf_indices:
#         # 当前分段从 current_start_idx 到 inflection_point_idx - 1
#         if inflection_point_idx > current_start_idx: # 确保分段非空
#             segments_info.append({
#                 'start_idx': current_start_idx,
#                 'end_idx': inflection_point_idx - 1, # 包含的结束索引
#                 'values': seq[current_start_idx : inflection_point_idx] # 切片不包含末尾
#             })
#         current_start_idx = inflection_point_idx # 下一个分段从拐点处开始

#     # 添加最后一个分段
#     if current_start_idx < len(seq):
#         segments_info.append({
#             'start_idx': current_start_idx,
#             'end_idx': len(seq) - 1,
#             'values': seq[current_start_idx:]
#         })
    
#     # 过滤掉可能因逻辑产生的空值分段 (尽管上面的 if inflection_point_idx > current_start_idx 应该有所帮助)
#     return [s for s in segments_info if s['values']]


# # 修改后的 Momentum_feature 函数
# def Momentum_feature_with_labels(
#     times: List[float], 
#     sizes: List[int], 
#     labels_info: Optional[List[List[any]]] = None,
#     **kwargs
# ) -> Tuple[List[List[float]], List[dict[str, any]]]:
#     """
#     Extracts Momentum features from segments of the 'sizes' sequence and annotates 
#     these segments with label information based on 'times' and 'labels_info'.

#     Args:
#         times: List of timestamps corresponding to each point in 'sizes'.
#         sizes: List of integer values (e.g., packet sizes).
#         labels_info: Optional. Expected format: [[list_of_spans], [list_of_class_ids]]
#                      where list_of_spans is [[start_t1, end_t1], [start_t2, end_t2], ...],
#                      and list_of_class_ids is [id1, id2, ...].
#         **kwargs:
#             threshold (Optional[float]): Threshold for find_inflection_indices.
#             zscore (float): Z-score for find_inflection_indices. Default is 0.15 
#                             (as in your original Momentum_feature).
#             momentum (float): Momentum factor for feature calculation. Default 0.1.
#             use_log_features (bool): If True, use log10(abs(size)) for features. 
#                                      Default False (uses momentum accumulation).

#     Returns:
#         A tuple (MMM, segment_annotations):
#         - MMM (List[List[float]]): The [2 x N] feature matrix, where N is the
#           number of segments (capped by MAX_MMM_LENGTH_CONST).
#         - segment_annotations (List[Dict[str, Any]]): A list of dictionaries,
#           each annotating a segment corresponding to a column in MMM. Annotations include:
#             'segment_idx_original': Original index of the segment (if more segments than N).
#             'segment_idx_mmm': Index in the returned MMM matrix (0 to N-1).
#             'start_idx_sizes': Start index in the original 'sizes' array.
#             'end_idx_sizes': End index in the original 'sizes' array.
#             'start_time': Start timestamp of the segment.
#             'end_time': End timestamp of the segment.
#             'num_points': Number of data points in this segment.
#             'feature_vector': The [positive_feature, negative_feature] for this segment from MMM.
#             'overlapping_labels': A list of dicts for labels overlapping this segment, each with:
#                 'class_id', 'label_start_time', 'label_end_time', 
#                 'overlap_start_time', 'overlap_end_time', 'overlap_duration',
#                 'overlap_ratio_on_segment' (0-1, how much of segment duration is covered by label),
#                 'overlap_ratio_on_label' (0-1, how much of label duration is covered by segment).
#     """
#     # 输入验证
#     if isinstance(sizes, np.ndarray):
#         sizes = sizes.tolist()
#     if isinstance(times, np.ndarray):
#         times = times.tolist()

#     if not sizes or not times or len(times) != len(sizes):
#         # print("警告: sizes/times 为空或长度不匹配。")
#         return [[], []], [] # 返回空的特征和标注

#     # 从 kwargs 获取参数
#     threshold_param = kwargs.get('threshold', None)
#     zscore_param = kwargs.get('zscore', 0.15) # 您原 Momentum_feature 中的默认值
#     momentum_val = kwargs.get('momentum', 0.1)
#     use_log_features = kwargs.get('use_log_features', False)
    
#     # 1. 使用新的分段函数获取分段及其原始索引
#     # segments_info 是一个列表，每个元素是字典: {'start_idx': int, 'end_idx': int, 'values': List[int]}
#     segments_info = segment_by_inflection_with_indices(
#         sizes,
#         threshold=threshold_param,
#         zscore=zscore_param
#     )

#     if not segments_info:
#         # print("警告: 未能从 sizes 序列中分割出任何 segments。")
#         return [[], []], []

#     # 2. 确定 MMM 矩阵的有效长度 (列数)
#     effective_num_segments = min(len(segments_info), MAX_MMM_LENGTH)
    
#     MMM = [[0.0 for _ in range(effective_num_segments)], [0.0 for _ in range(effective_num_segments)]]
#     all_segment_annotations = []

#     # 3. 解析 labels_info
#     obj_time_spans = []
#     obj_class_ids = []
#     if labels_info and len(labels_info) == 2:
#         # labels_info[0] 应该是时间跨度列表 [[start1, end1], [start2, end2], ...]
#         # labels_info[1] 应该是类别ID列表 [id1, id2, ...]
#         if isinstance(labels_info[0], list) and isinstance(labels_info[1], list):
#             obj_time_spans = labels_info[0]
#             obj_class_ids = labels_info[1]
#             if len(obj_time_spans) != len(obj_class_ids):
#                 # print("警告: 标签时间跨度数量与类别ID数量不匹配。将忽略标签信息。")
#                 obj_time_spans, obj_class_ids = [], []
#             # 进一步验证 obj_time_spans 内部结构
#             if obj_time_spans and not (isinstance(obj_time_spans[0], list) and len(obj_time_spans[0]) == 2):
#                 # print("警告: 标签时间跨度格式不正确。应为 [[start, end], ...]。将忽略标签信息。")
#                 obj_time_spans, obj_class_ids = [], []
#         else:
#             # print("警告: labels_info 格式不正确。将忽略标签信息。")
#             pass # obj_time_spans, obj_class_ids 保持为空

#     # 4. 遍历分段，计算特征并进行标注
#     for mmm_col_idx in range(effective_num_segments):
#         current_segment_info = segments_info[mmm_col_idx] # 获取原始分段信息
        
#         segment_values = current_segment_info['values']
#         original_start_idx = current_segment_info['start_idx']
#         original_end_idx = current_segment_info['end_idx']

#         # 计算 MMM 特征 (与您原代码逻辑一致)
#         # 初始化每个 segment 的累加器
#         accum_positive_feature = 0.0
#         accum_negative_feature = 0.0
#         if use_log_features:
#             for val in segment_values:
#                 if val == 0: continue
#                 log_abs_val = math.log10(abs(val)) # 确保 val 非0
#                 if val > 0:
#                     accum_positive_feature += log_abs_val
#                 else: # val < 0
#                     accum_negative_feature += log_abs_val
#         else: # 使用动量累积
#             for val in segment_values:
#                 if val == 0: continue
#                 if val > 0:
#                     accum_positive_feature = (1 - momentum_val) * accum_positive_feature + momentum_val * val
#                 else: # val < 0
#                     accum_negative_feature = (1 - momentum_val) * accum_negative_feature + momentum_val * (-val)
        
#         MMM[0][mmm_col_idx] = accum_positive_feature
#         MMM[1][mmm_col_idx] = accum_negative_feature
            
#         # 创建当前 segment 的标注信息
#         seg_start_time = times[original_start_idx]
#         seg_end_time = times[original_end_idx] # 确保 original_end_idx 有效
        
#         current_annotation = {
#             'segment_idx_original': mmm_col_idx, # 在此简化下，它与 mmm_col_idx 相同，因为我们只处理 effective_num_segments
#             'segment_idx_mmm': mmm_col_idx,
#             'start_idx_sizes': original_start_idx,
#             'end_idx_sizes': original_end_idx,
#             'start_time': round(seg_start_time, 6), # 保留6位小数
#             'end_time': round(seg_end_time, 6),
#             'num_points': len(segment_values),
#             'feature_vector': [round(MMM[0][mmm_col_idx], 6), round(MMM[1][mmm_col_idx], 6)],
#             'overlapping_labels': []
#         }

#         # 检查与已提供标签的重叠情况
#         if obj_time_spans:
#             segment_actual_duration = seg_end_time - seg_start_time
#             # 防止除以零，对于时间点上的segment（开始结束时间相同）
#             segment_duration_for_ratio = segment_actual_duration if segment_actual_duration > 1e-9 else 1e-9

#             for i, (label_start_t, label_end_t) in enumerate(obj_time_spans):
#                 class_id = obj_class_ids[i]
                
#                 # 计算重叠区间 [max(A,C), min(B,D)]
#                 # Segment: [A, B] = [seg_start_time, seg_end_time]
#                 # Label:   [C, D] = [label_start_t, label_end_t]
#                 overlap_start_t = max(seg_start_time, label_start_t)
#                 overlap_end_t = min(seg_end_time, label_end_t)
                
#                 overlap_duration = max(0, overlap_end_t - overlap_start_t)

#                 if overlap_duration > 1e-9: # 仅当存在显著重叠时记录
#                     label_actual_duration = label_end_t - label_start_t
#                     label_duration_for_ratio = label_actual_duration if label_actual_duration > 1e-9 else 1e-9

#                     overlap_details = {
#                         'class_id': class_id,
#                         'label_start_time': round(label_start_t, 6),
#                         'label_end_time': round(label_end_t, 6),
#                         'overlap_start_time': round(overlap_start_t, 6),
#                         'overlap_end_time': round(overlap_end_t, 6),
#                         'overlap_duration': round(overlap_duration, 6),
#                         'overlap_ratio_on_segment': round(overlap_duration / segment_duration_for_ratio, 4),
#                         'overlap_ratio_on_label': round(overlap_duration / label_duration_for_ratio, 4)
#                     }
#                     current_annotation['overlapping_labels'].append(overlap_details)
        
#         all_segment_annotations.append(current_annotation)

#     return MMM, all_segment_annotations


if __name__=='__main__':
    # 示例数据
    timestamps = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    sizes = [1, 1, 2, -0.5, 1, -1, 1, -1, 1]

    file = r'D:\2025HS_dataset\Temp\20250115_1111_45_juhanurmihxlp77nkq76byazcldy2hlmovfu2epvl5ankdibsot4csyd24.dat'
    with open(file, 'r') as f:
        tcp_dump = f.readlines()
    import pandas as pd
    seq = pd.Series(tcp_dump).str.slice(0, -1).str.split('\t', expand=True).astype("float")
    timestamps = np.array(seq.iloc[:, 0], dtype=np.float64)
    if timestamps[0] != 0:
        times = timestamps - timestamps[0]
    sizes = np.array(seq.iloc[:, 1], dtype=np.int32)

    # 分割
    segments = segment_by_inflection(sizes, threshold=None, zscore=0.2)
    

    # # 输出结果
    # for i, sz in enumerate(segments):
    #     #print(f"Segment {i+1}:")
    #     #print("Timestamps:", ts)
    #     if len(sz)>0:
    #         print("Sizes:", sz)

    # multidat = r'D:\博士研究\论文写作\202410类别分类论文\Momentum_WF\WF_Code\multilabel_dataset\2-0+3.dat'
    # multilabel = r'D:\博士研究\论文写作\202410类别分类论文\Momentum_WF\WF_Code\multilabel_dataset\2-0+3_labels.json'

    # with open(multidat, 'r') as f:
    #     tcp_dump = f.readlines()
    # import pandas as pd
    # seq = pd.Series(tcp_dump).str.slice(0, -1).str.split('\t', expand=True).astype("float")
    # timestamps = np.array(seq.iloc[:, 0], dtype=np.float64)
    # if timestamps[0] != 0:
    #     times = timestamps - timestamps[0]
    # sizes = np.array(seq.iloc[:, 1], dtype=np.int32)

    # import json

    # # 读取JSON文件
    # with open(multilabel, 'r', encoding='utf-8') as f:
    #     labels = json.load(f)

    # MMM, all_segment_annotations = Momentum_feature_with_labels(
    #         timestamps,
    #         sizes,
    #         labels_info=labels
    # )