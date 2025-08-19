from typing import List, Optional, Tuple, Dict, List, Any
import statistics
import math
import numpy as np

MAX_MMM_LENGTH = 30000

def find_inflection_indices(
    seq: List[float],
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
    seq: List[float],
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
    momentum_val = kwargs.get('momentum', 0.1)
    try:
        segments = segment_by_inflection(
            sizes,
            threshold=threshold,
            zscore=zscore
        )
    except Exception:
        # Fallback to treating the whole sequence as one segment
        segments = [sizes.copy() if isinstance(sizes, list) else sizes.tolist()]
    
    MAX_MMM_LENGTH = 8000
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
        for idx, segment in enumerate(segments[:MAX_MMM_LENGTH]):
            for i in segment:
                if i==0:
                    continue
                # # 动量特征
                # elif i > 0:
                #     MMM[0][idx] = (1-momentum)*MMM[0][idx] + momentum*i
                    
                # else:
                #     MMM[-1][idx] = (1-momentum)*MMM[-1][idx] + momentum*(-i)

                # 第一个包保存
                elif i > 0:
                    if MMM[0][idx] == 0:
                        MMM[0][idx] = i
                    else:
                        MMM[0][idx] = (1-momentum_val)*MMM[0][idx] + momentum_val*i
                else:
                    if MMM[-1][idx] == 0:
                        MMM[-1][idx] = -i
                    else:
                        MMM[-1][idx] = (1-momentum_val)*MMM[-1][idx] + momentum_val*(-i)
    
    return MMM

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

def segment_by_inflection_with_indices(
    seq: List[int],
    threshold: Optional[float] = None,
    zscore: float = 1.0
) -> List[Dict[str, Any]]:
    if not seq: return []
    try:
        inf_indices = find_inflection_indices(seq, threshold, zscore)
    except Exception:
        return [{'start_idx': 0, 'end_idx': len(seq) - 1, 'values': seq[:]}]
    
    segments_info: List[Dict[str, Any]] = []
    current_start_idx = 0
    for inflection_point_idx in inf_indices:
        if inflection_point_idx > current_start_idx:
            segments_info.append({
                'start_idx': current_start_idx,
                'end_idx': inflection_point_idx - 1,
                'values': seq[current_start_idx : inflection_point_idx]
            })
        current_start_idx = inflection_point_idx
    if current_start_idx < len(seq):
        segments_info.append({
            'start_idx': current_start_idx,
            'end_idx': len(seq) - 1,
            'values': seq[current_start_idx:]
        })
    return [s for s in segments_info if s['values']]

# 修改后的 Momentum_feature 函数，聚焦于标签到MMM列的映射
def Momentum_feature_label_to_MMM_indices(
    times: List[float], 
    sizes: List[int], 
    labels_info: Optional[List[List[Any]]] = None,
    **kwargs
) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    """
    Extracts Momentum features and maps original labeled sub-sequences to 
    their start and end column indices in the MMM feature matrix.

    Args:
        times: List of timestamps.
        sizes: List of integer values.
        labels_info: Optional. Format: [[[start_t1, end_t1], ...], [class_id1, ...]].
        **kwargs:
            threshold, zscore, momentum, use_log_features.

    Returns:
        A tuple (MMM, label_to_mmm_indices_map):
        - MMM (List[List[float]]): The [2 x N] feature matrix.
        - label_to_mmm_indices_map (List[Dict[str, Any]]): A list where each dictionary 
          corresponds to an original labeled sub-sequence from labels_info. Each dict contains:
            'label_index': Index of this label in the input labels_info.
            'class': The class ID of the labeled sub-sequence.
            'start_time': Original start time of the sub-sequence.
            'end_time': Original end time of the sub-sequence.
            'mmm_start_idx': The starting column index in MMM that overlaps with this label.
                                 None if no overlap or label is outside MMM range.
            'mmm_end_idx': The ending column index in MMM that overlaps with this label.
                               None if no overlap or label is outside MMM range.
    """
    if isinstance(sizes, np.ndarray): sizes = sizes.tolist()
    if isinstance(times, np.ndarray): times = times.tolist()

    if not sizes or not times or len(times) != len(sizes):
        return [[], []], []

    threshold_param = kwargs.get('threshold', None)
    zscore_param = kwargs.get('zscore', 0.15)
    momentum_val = kwargs.get('momentum', 0.1)
    use_log_features = kwargs.get('use_log_features', False)
    
    # segments_info_list was renamed from segments_info to avoid conflict in the outer scope
    segments_info_list = segment_by_inflection_with_indices(
        sizes, threshold=threshold_param, zscore=zscore_param
    )

    if not segments_info_list:
        return [[], []], []

    effective_num_segments = min(len(segments_info_list), MAX_MMM_LENGTH)
    MMM = [[0.0 for _ in range(effective_num_segments)], [0.0 for _ in range(effective_num_segments)]]
    
    segment_time_spans_for_mmm = [] # Stores {'start_time': float, 'end_time': float, 'mmm_col': int}

    for mmm_col_idx in range(effective_num_segments):
        current_segment_info = segments_info_list[mmm_col_idx]
        segment_values = current_segment_info['values']
        original_start_idx = current_segment_info['start_idx']
        original_end_idx = current_segment_info['end_idx']

        # Feature calculation (same as before)
        accum_positive_feature = 0.0
        accum_negative_feature = 0.0
        if use_log_features:
            for val in segment_values:
                if val == 0: continue
                try: log_abs_val = math.log10(abs(val))
                except ValueError: log_abs_val = 0 
                if val > 0: accum_positive_feature += log_abs_val
                else: accum_negative_feature += log_abs_val
        else:
            for val in segment_values:
                if val == 0: continue
                if val > 0:
                    accum_positive_feature = (1 - momentum_val) * accum_positive_feature + momentum_val * val
                else:
                    accum_negative_feature = (1 - momentum_val) * accum_negative_feature + momentum_val * (-val)
        MMM[0][mmm_col_idx] = accum_positive_feature
        MMM[1][mmm_col_idx] = accum_negative_feature
        
        if 0 <= original_start_idx < len(times) and 0 <= original_end_idx < len(times):
            seg_start_time = times[original_start_idx]
            seg_end_time = times[original_end_idx]
            # Ensure start_time is not after end_time for a segment
            if seg_start_time > seg_end_time:
                # print(f"Warning: Segment {mmm_col_idx} has start_time > end_time. Swapping. ({seg_start_time}, {seg_end_time})")
                seg_start_time, seg_end_time = seg_end_time, seg_start_time # Swap if needed
            segment_time_spans_for_mmm.append({'start_time': seg_start_time, 'end_time': seg_end_time, 'mmm_col': mmm_col_idx})
        else:
            segment_time_spans_for_mmm.append({'start_time': None, 'end_time': None, 'mmm_col': mmm_col_idx})

    # --- Map labels to MMM start and end column indices ---
    label_to_mmm_indices_map = []
    obj_time_spans = []
    obj_class_ids = []

    if labels_info and len(labels_info) == 2:
        if isinstance(labels_info[0], list) and isinstance(labels_info[1], list):
            obj_time_spans = labels_info[0]
            obj_class_ids = labels_info[1]
            if len(obj_time_spans) != len(obj_class_ids): obj_time_spans, obj_class_ids = [], []
            if obj_time_spans and not (isinstance(obj_time_spans[0], list) and len(obj_time_spans[0]) == 2):
                obj_time_spans, obj_class_ids = [], []
    
    if not segment_time_spans_for_mmm:
        for i, (label_start_t, label_end_t) in enumerate(obj_time_spans):
            label_to_mmm_indices_map.append({
                'label_index': i, 'class': obj_class_ids[i],
                'start_time': label_start_t, 'end_time': label_end_t,
                'mmm_start_idx': None, 'mmm_end_idx': None
            })
        return MMM, label_to_mmm_indices_map

    for i, (label_start_t, label_end_t) in enumerate(obj_time_spans):
        class_id = obj_class_ids[i]
        
        # Ensure label start_time is not after end_time
        if label_start_t > label_end_t:
            # print(f"Warning: Label {i} has start_time > end_time. Swapping. ({label_start_t}, {label_end_t})")
            label_start_t, label_end_t = label_end_t, label_start_t


        current_label_mmm_start_col = None
        current_label_mmm_end_col = None

        for seg_time_info in segment_time_spans_for_mmm:
            if seg_time_info['start_time'] is None or seg_time_info['end_time'] is None:
                continue

            seg_start_time = seg_time_info['start_time']
            seg_end_time = seg_time_info['end_time']
            mmm_col_idx = seg_time_info['mmm_col']

            # Check for overlap: seg_start <= label_end AND label_start <= seg_end
            is_overlapping = (seg_start_time <= label_end_t and label_start_t <= seg_end_time)
            
            if is_overlapping:
                # Check if there's a meaningful overlap duration
                overlap_s = max(seg_start_time, label_start_t)
                overlap_e = min(seg_end_time, label_end_t)
                if overlap_e > overlap_s: # Meaningful overlap
                    if current_label_mmm_start_col is None:
                        current_label_mmm_start_col = mmm_col_idx
                    # Always update end column if there's an overlap with current segment
                    current_label_mmm_end_col = mmm_col_idx 
                # If overlap is just a point, we might not update start_col unless it's the very first potential overlap
                elif overlap_e == overlap_s and current_label_mmm_start_col is None and \
                     (seg_start_time == label_end_t or label_start_t == seg_end_time) : # Touches
                     if current_label_mmm_start_col is None:
                         current_label_mmm_start_col = mmm_col_idx
                     current_label_mmm_end_col = mmm_col_idx


        label_to_mmm_indices_map.append({
            'label_index': i,
            'class': class_id,
            'start_time': round(label_start_t, 6),
            'end_time': round(label_end_t, 6),
            'mmm_start_idx': current_label_mmm_start_col,
            'mmm_end_idx': current_label_mmm_end_col
        })

    return MMM, label_to_mmm_indices_map


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
    sizes = np.array(seq.iloc[:, 1], dtype=np.float32)

    # 分割
    segments = segment_by_inflection(sizes, threshold=None, zscore=0.2)
    

    # # 输出结果
    # for i, sz in enumerate(segments):
    #     #print(f"Segment {i+1}:")
    #     #print("Timestamps:", ts)
    #     if len(sz)>0:
    #         print("Sizes:", sz)

    multidat = r'D:\博士研究\论文写作\202410类别分类论文\Momentum_WF\WF_Code\multilabel_dataset\2-0+3.dat'
    multilabel = r'D:\博士研究\论文写作\202410类别分类论文\Momentum_WF\WF_Code\multilabel_dataset\2-0+3_labels.json'

    with open(multidat, 'r') as f:
        tcp_dump = f.readlines()
    import pandas as pd
    seq = pd.Series(tcp_dump).str.slice(0, -1).str.split('\t', expand=True).astype("float")
    timestamps = np.array(seq.iloc[:, 0], dtype=np.float64)
    if timestamps[0] != 0:
        timestamps = timestamps - timestamps[0]
    sizes = np.array(seq.iloc[:, 1], dtype=np.int32)

    import json

    # 读取JSON文件
    with open(multilabel, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    MMM_matrix, label_mappings = Momentum_feature_label_to_MMM_indices(
        timestamps, 
        sizes, 
        labels_info=labels
    )

    print(MMM_matrix, '\n')
    print(label_mappings)