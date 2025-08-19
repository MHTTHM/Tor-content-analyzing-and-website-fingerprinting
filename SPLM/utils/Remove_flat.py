import statistics
from typing import List, Optional, Tuple

def find_inflection_indices(
    seq: List[int],
    threshold: Optional[float] = None,
    zscore: float = 2.0
) -> List[int]:
    """
    Identify indices where the absolute difference between consecutive elements exceeds a threshold.
    """
    if len(seq) < 2:
        return []

    diffs = [abs(seq[i] - seq[i - 1]) for i in range(1, len(seq))]

    if threshold is None:
        mean_diff = statistics.mean(diffs)
        std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        threshold = mean_diff + zscore * std_diff

    return [i for i, d in enumerate(diffs, start=1) if d > threshold]

def segment_by_inflection(
    seq: List[int],
    threshold: Optional[float] = None,
    zscore: float = 1
) -> List[List[int]]:
    """
    Segment the sequence into sublists between inflection points.
    """
    inf_indices = find_inflection_indices(seq, threshold, zscore)
    segments = []
    start = 0

    for idx in inf_indices:
        segments.append(seq[start: idx ])
        start = idx

    if start < len(seq):
        segments.append(seq[start:])
    return segments

def remove_flat_segments(
    timestamps: List[float],
    lengths: List[int],
    rate_threshold: Optional[float] = None,
    zscore: float = 1.0
) -> Tuple[List[float], List[int]]:
    """
    Remove leading and trailing flat segments based on rate of change.
    """
    if len(timestamps) != len(lengths):
        raise ValueError("Timestamps and lengths must have the same length.")
    if not timestamps:
        return [], []

    segments = segment_by_inflection(lengths)

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
    if rate_threshold is None:
        abs_rates = [abs(r) for r in rates if not (r == float('inf') or r == -float('inf'))]
        if not abs_rates:
            return [], []
        mean = statistics.mean(abs_rates)
        std = statistics.stdev(abs_rates) if len(abs_rates) > 1 else 0.0
        rate_threshold = mean + zscore * std

    # Identify flat segments
    is_flat = []
    for rate in rates:
        if rate == float('inf') or rate == -float('inf'):
            is_flat.append(False)
        else:
            is_flat.append(abs(rate) < rate_threshold)
    print(is_flat)

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

    return timestamps[start_idx:end_idx+1], lengths[start_idx:end_idx+1]
    # return times, lengths[start_idx:end_idx+1]

TIME_END = 90
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
    
    [timestamps, packets] = remove_flat_segments(
            timestamps, 
            packets,
            rate_threshold=None,
            zscore=1)

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

# 示例使用
if __name__ == "__main__":
    data = [
        (1.0, 1),  # 平缓
        (2.0, -1), # 平缓
        (3.0, 10), # 不平缓
        (4.0, 20), # 不平缓
        (5.0, -5), # 不平缓
        (6.0, 2),  # 平缓
        (7.0, -1)  # 平缓
    ]
    
    timestamps = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    lengths = [1, -1, 10, 20, 5, 5, 1, 2, 1, -1]

    # new_ts, new_lens = remove_flat_segments(timestamps, lengths)
    # print(new_ts)   # 输出: [4.0, 5.0, 6.0]
    # print(new_lens) # 输出: [5, 5, 5]

    filepath = r'D:\2025HS_dataset\Temp\20250313_1918_49_k7dyt6gcr7bvytefr2uksfbumtpyiiolih55i4hzvvxdfl6rrndrarid2.dat'
    timestamps, packets, packet_sizes, packet_sizes_abs, packet_count, pktcount = count_smooth_file_seq(filepath)
    print(f"overall length: {len(timestamps)}")
