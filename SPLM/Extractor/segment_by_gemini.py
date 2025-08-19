import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

def handle_timestamp_issues(timestamps, sizes, duplicate_strategy='mean'):
    """
    Sorts data by timestamp and handles duplicate timestamps.
    A new strategy 'keep_all_sequential' is added to keep all points,
    sorted by timestamp, without merging.
    """
    if len(timestamps) != len(sizes):
        raise ValueError("Timestamps and sizes must have the same length.")
    if not len(timestamps):
        return np.array([]), np.array([])

    # Pair original data points and sort by timestamp.
    # Python's sorted is stable, so for equal timestamps, the original relative order is preserved.
    # data_pairs = sorted(zip(timestamps, sizes), key=lambda x: x[0]) # dont need sort
    data_pairs = list(zip(timestamps, sizes))

    if not data_pairs:
        return np.array([]), np.array([])

    if duplicate_strategy == 'keep_all_sequential':
        # Keep all data points, sorted by timestamp. Original order for equal timestamps is preserved.
        processed_ts_list = [p[0] for p in data_pairs]
        processed_sizes_list = [p[1] for p in data_pairs]
    else:
        # Original logic for other strategies (mean, sum, first, last)
        temp_data = OrderedDict()
        for ts, size in data_pairs:
            if ts not in temp_data:
                temp_data[ts] = []
            temp_data[ts].append(size)

        processed_ts_list = []
        processed_sizes_list = []

        for ts, size_list in temp_data.items():
            processed_ts_list.append(ts)
            if not size_list:
                processed_sizes_list.append(0) # Should ideally not happen if data_pairs was not empty
                continue

            if duplicate_strategy == 'mean':
                processed_sizes_list.append(np.mean(size_list))
            elif duplicate_strategy == 'sum':
                processed_sizes_list.append(np.sum(size_list))
            elif duplicate_strategy == 'first':
                processed_sizes_list.append(size_list[0])
            elif duplicate_strategy == 'last':
                processed_sizes_list.append(size_list[-1])
            else:
                raise ValueError(f"Unknown duplicate_strategy: {duplicate_strategy}")

    final_ts = np.array(processed_ts_list, dtype=float)
    final_sizes = np.array(processed_sizes_list, dtype=float)
    
    # This check is for strategies that should result in strictly increasing timestamps.
    # For 'keep_all_sequential', timestamps can be non-strictly increasing (duplicates allowed).
    if len(final_ts) > 1 and duplicate_strategy != 'keep_all_sequential':
        dt_check = np.diff(final_ts)
        if np.any(dt_check <= 0):
            print("Warning: After processing (non-'keep_all_sequential' strategy), timestamps are not strictly increasing.")
            
    return final_ts, final_sizes

def segment_time_series_by_slope_change(
    timestamps, 
    sizes, 
    percentile_threshold=90, 
    duplicate_timestamp_strategy='keep_all_sequential',
    trim_flat_portions=False,
    flatness_slope_threshold=0.5,
    min_trimmed_segment_length=3
    ):
    if not isinstance(timestamps, (list, np.ndarray)) or not isinstance(sizes, (list, np.ndarray)):
        raise TypeError("Timestamps and sizes must be list-like or numpy arrays.")

    ts_arr_orig = np.array(timestamps, dtype=float)
    sizes_arr_orig = np.array(sizes, dtype=float)
    ts_arr, sizes_arr = handle_timestamp_issues(ts_arr_orig, sizes_arr_orig, duplicate_strategy=duplicate_timestamp_strategy)
    
    # --- BEGINNING OF INSERTED CODE FOR PRE-TRIMMING ---
    if trim_flat_portions and len(ts_arr) >= min_trimmed_segment_length:
        # Pre-trim flat portions from the beginning of the entire series
        prospective_start_index = 0 # This will be the index of the first point to keep
        for k_idx in range(len(ts_arr) - 1): # k_idx is the first point of the pair (k_idx, k_idx+1)
            # If trimming k_idx (i.e., starting the series from k_idx+1)
            # would make the remaining series too short, stop trying to trim further.
            # The length of series if we start from k_idx+1 is `len(ts_arr) - (k_idx + 1)`.
            if (len(ts_arr) - (k_idx + 1)) < min_trimmed_segment_length:
                break

            delta_t = ts_arr[k_idx+1] - ts_arr[k_idx]
            current_slope = float('inf') # Default to non-flat if delta_t is too small or zero
            if delta_t > 1e-9: # Avoid division by zero or very small number
                # Slope definition consistent with later segment trimming: size of next point / time_interval
                current_slope = sizes_arr[k_idx+1] / delta_t 
            
            if abs(current_slope) < flatness_slope_threshold:
                # Segment (k_idx, k_idx+1) is flat. This means point k_idx is part of the flat start.
                # So, the earliest the series could start is k_idx + 1.
                prospective_start_index = k_idx + 1
            else:
                # Segment (k_idx, k_idx+1) is NOT flat.
                # This means k_idx is the first point that leads into a non-flat region.
                # The current `prospective_start_index` (which is k_idx or an earlier index if k_idx=0)
                # is where the non-flat activity begins. We stop searching.
                break 
        
        # Apply the trim if prospective_start_index moved and remaining series is long enough
        if prospective_start_index > 0:
            if (len(ts_arr) - prospective_start_index) >= min_trimmed_segment_length:
                original_kept_start_ts = ts_arr[prospective_start_index]
                
                sizes_arr = sizes_arr[prospective_start_index:]
                ts_arr = ts_arr[prospective_start_index:]
                
                # Normalize timestamps to start from 0
                ts_arr = ts_arr - original_kept_start_ts
                # print(f"Info: Pre-trimmed {prospective_start_index} points from the beginning. New series starts at ts=0 (original ts={original_kept_start_ts}).")
            # else:
                # print(f"Info: Pre-trimming would make the series too short. Original series start kept.")
    # --- END OF INSERTED CODE FOR PRE-TRIMMING ---

    if len(ts_arr) < 3:
        print("Warning: Not enough unique data points after processing for segmentation (need at least 3).")
        if len(ts_arr) == 0: return []
        return [[(ts_arr[i], sizes_arr[i]) for i in range(len(ts_arr))]]

    cumulative_sizes = np.cumsum(sizes_arr)
    dt = np.diff(ts_arr)

    dt[dt <= 1e-9] = 1.0 # Handle zero or very small dt to prevent division issues
    if len(dt) == 0:
         print("Warning: Not enough time differences. Returning single segment.")
         return [[(ts_arr[i], sizes_arr[i]) for i in range(len(ts_arr))]]

    local_slopes = np.diff(cumulative_sizes) / dt

    if len(local_slopes) < 2:
        print("Warning: Not enough local slopes. Returning single segment.")
        return [[(ts_arr[i], sizes_arr[i]) for i in range(len(ts_arr))]]

    dt_for_slope_changes = (dt[:-1] + dt[1:]) / 2.0
    dt_for_slope_changes[dt_for_slope_changes <= 1e-9] = 1.0
    change_in_slopes = np.diff(local_slopes) / dt_for_slope_changes
    abs_change_in_slopes = np.abs(change_in_slopes)

    if len(abs_change_in_slopes) == 0:
        print("Warning: No changes in slope could be calculated. Returning single segment.")
        return [[(ts_arr[i], sizes_arr[i]) for i in range(len(ts_arr))]]

    finite_abs_changes = abs_change_in_slopes[np.isfinite(abs_change_in_slopes)]
    if len(finite_abs_changes) == 0:
        print("Warning: All slope changes are non-finite. Cannot determine threshold. Returning single segment.")
        return [[(ts_arr[i], sizes_arr[i]) for i in range(len(ts_arr))]]
        
    threshold_value = np.percentile(finite_abs_changes, percentile_threshold)
    
    if threshold_value == 0 and np.all(finite_abs_changes == 0):
        print("Warning: All finite slope changes are zero. No breakpoints found other than start/end.")
        
    breakpoint_indices = [0]
    for i in range(len(abs_change_in_slopes)):
        if np.isfinite(abs_change_in_slopes[i]) and abs_change_in_slopes[i] > threshold_value:
            breakpoint_indices.append(i + 1) 
                                        
    breakpoint_indices.append(len(ts_arr) - 1)
    breakpoint_indices = sorted(list(set(breakpoint_indices)))
    
    valid_bps = [idx for idx in breakpoint_indices if 0 <= idx < len(ts_arr)]
    breakpoint_indices = sorted(list(set(valid_bps))) 
    if not breakpoint_indices and len(ts_arr) > 0: 
        breakpoint_indices = [0, len(ts_arr)-1]
    elif len(ts_arr) > 0:
        if 0 not in breakpoint_indices: breakpoint_indices.insert(0,0)
        if (len(ts_arr)-1) not in breakpoint_indices : breakpoint_indices.append(len(ts_arr)-1)
        breakpoint_indices = sorted(list(set(breakpoint_indices)))

    raw_segments = []
    if len(breakpoint_indices) < 2:
         return [[(ts_arr[i], sizes_arr[i]) for i in range(len(ts_arr))]]

    for i in range(len(breakpoint_indices) - 1):
        start_idx = breakpoint_indices[i]
        end_idx = breakpoint_indices[i+1]
        segment_points_tuples = []
        for k in range(start_idx, min(end_idx + 1, len(ts_arr))):
            segment_points_tuples.append((ts_arr[k], sizes_arr[k]))
        
        if segment_points_tuples:
            if not raw_segments or segment_points_tuples[0] != raw_segments[-1][-1] or len(segment_points_tuples) > 1:
                 raw_segments.append(segment_points_tuples)
            elif raw_segments and len(segment_points_tuples) == 1 and segment_points_tuples[0] == raw_segments[-1][-1]:
                pass 
            else: 
                raw_segments.append(segment_points_tuples)

    # --- Trimming Logic Modification ---
    if not trim_flat_portions:
        return raw_segments # Return raw segments if trimming is globally off

    processed_segments_for_return = []
    num_raw_segments = len(raw_segments)

    for i, segment_data_tuple_list in enumerate(raw_segments):
        current_segment_points = list(segment_data_tuple_list) 

        is_first_segment = (i == 0)
        is_last_segment = (i == num_raw_segments - 1)
        
        # Trimming applies only if trim_flat_portions is True AND it's the first or the last segment
        apply_trim_to_this_one = (is_first_segment or is_last_segment)

        if apply_trim_to_this_one:
            if len(current_segment_points) < min_trimmed_segment_length:
                if current_segment_points: 
                    processed_segments_for_return.append(current_segment_points)
                continue 

            points_to_trim = list(current_segment_points) 
            
            seg_start_idx_in_points_to_trim = 0
            seg_end_idx_in_points_to_trim = len(points_to_trim) - 1

            # Trim Start of this segment
            current_trimmed_start = seg_start_idx_in_points_to_trim
            for k_start in range(seg_start_idx_in_points_to_trim, seg_end_idx_in_points_to_trim): 
                if (seg_end_idx_in_points_to_trim - (current_trimmed_start + 1) + 1) < min_trimmed_segment_length:
                    break
                
                p1_ts, p1_val = points_to_trim[current_trimmed_start]
                p2_ts, p2_val_orig = points_to_trim[current_trimmed_start + 1] 

                delta_t = p2_ts - p1_ts
                local_slope = float('inf') # Default to non-flat if delta_t is too small
                if delta_t > 1e-9: 
                    local_slope = p2_val_orig / delta_t
                
                if abs(local_slope) < flatness_slope_threshold:
                    current_trimmed_start += 1 
                else:
                    break 
            
            final_trimmed_start_for_segment = current_trimmed_start

            # Trim End of this segment
            current_trimmed_end = seg_end_idx_in_points_to_trim
            for k_end in range(seg_end_idx_in_points_to_trim, final_trimmed_start_for_segment, -1): 
                if ((current_trimmed_end - 1) - final_trimmed_start_for_segment + 1) < min_trimmed_segment_length:
                    break

                p2_ts, p2_val_orig = points_to_trim[current_trimmed_end] 
                p1_ts, p1_val = points_to_trim[current_trimmed_end - 1] 

                delta_t = p2_ts - p1_ts
                local_slope = float('inf')
                if delta_t > 1e-9:
                    local_slope = p2_val_orig / delta_t

                if abs(local_slope) < flatness_slope_threshold:
                    current_trimmed_end -= 1 
                else:
                    break 
            
            final_trimmed_end_for_segment = current_trimmed_end
            
            if final_trimmed_start_for_segment <= final_trimmed_end_for_segment and \
               (final_trimmed_end_for_segment - final_trimmed_start_for_segment + 1) >= min_trimmed_segment_length:
                processed_segments_for_return.append(points_to_trim[final_trimmed_start_for_segment : final_trimmed_end_for_segment + 1])
            else: # Trimming made it too short or invalid
                if len(current_segment_points) >= min_trimmed_segment_length : # Add original if it was long enough
                     processed_segments_for_return.append(current_segment_points)
                elif current_segment_points: # Or if original was short but non-empty
                     processed_segments_for_return.append(current_segment_points)
        else:
            # This is an intermediate segment, add it as is (without trimming)
            if current_segment_points: 
                processed_segments_for_return.append(current_segment_points)
    
    return [seg for seg in processed_segments_for_return if seg] # Filter out any empty lists

def Gemini_Momentum_feature(times, sizes, **kwargs):
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
    
    percentile_threshold = kwargs.get('percentile_threshold', 80)

    try:
        segments = segment_time_series_by_slope_change(times, sizes, percentile_threshold=percentile_threshold)
    except Exception:
        segments = [sizes.copy() if isinstance(sizes, list) else sizes.tolist()]
    
    MAX_MMM_LENGTH = 5000
    if len(segments) < MAX_MMM_LENGTH:
        MAX_MMM_LENGTH = len(segments)
    MMM = [[0 for _ in range(MAX_MMM_LENGTH)], [0 for _ in range(MAX_MMM_LENGTH)]]

    momentum = 0.1
    for idx, segment in enumerate(segments[:MAX_MMM_LENGTH]):
        for _, i in segment:
            if i==0:
                continue
            elif i > 0:
                MMM[0][idx] = (1-momentum)*MMM[0][idx] + momentum*i
            else:
                MMM[-1][idx] = (1-momentum)*MMM[-1][idx] + momentum*(-i)

    return MMM


# --- Example Usage ---
if __name__ == "__main__":
    # Example Data: Designed to have clear flat start/end for the *overall series*
    # to test selective trimming of first/last segments.
    np.random.seed(42) 
    # timestamps_main = np.arange(150) 
    
    # s_very_flat_start_overall = np.random.normal(0.05, 0.01, 20) 
    # s_active1 = np.random.normal(loc=15, scale=3, size=30) 
    # s_flat_intermediate1 = np.random.normal(0.1, 0.05, 15) 
    # s_active2 = np.random.normal(loc=-10, scale=2, size=30)
    # s_flat_intermediate2 = np.random.normal(0.08, 0.03, 15) 
    # s_active3 = np.random.normal(loc=8, scale=2, size=20)
    # s_very_flat_end_overall = np.random.normal(0.05, 0.01, 20)
    
    # sizes_main_list = [
    #     s_very_flat_start_overall, s_active1, s_flat_intermediate1, 
    #     s_active2, s_flat_intermediate2, s_active3, s_very_flat_end_overall
    # ]
    # sizes_main = np.concatenate(sizes_main_list)

    timestamps_main = np.array([0, 1, 1, 2, 3, 4, 5,6, 7, 8, 9 ,10])
    sizes_main = np.array([1, 1, 2, 2, 3, 6,7, 11, -1 , 2 ,3, 2])

    if len(sizes_main) > len(timestamps_main):
        sizes_main = sizes_main[:len(timestamps_main)]
    elif len(sizes_main) < len(timestamps_main):
        padding_arr = np.random.normal(0.05, 0.01, len(timestamps_main) - len(sizes_main))
        sizes_main = np.concatenate([sizes_main, padding_arr])

    # read from HS
    file = r'D:\2025HS_dataset\Temp\20250115_1111_45_juhanurmihxlp77nkq76byazcldy2hlmovfu2epvl5ankdibsot4csyd24.dat'
    file = r'D:\2025HS_dataset\Temp\20241202_0122_57_firearmh73frrpeene3bbbdpxj3pgac3yvxafqve2jss3yt6uk4sjfid1.dat'
    file = r'D:\2025HS_dataset\Temp\20250304_0612_54_gamcored5widhilqmnyv7msosxpcqsvyunyljk6sx6swnzmzy2km2oyd9.dat'
    file = r'D:\2025HS_dataset\Temp\20250313_1916_39_k7dyt6gcr7bvytefr2uksfbumtpyiiolih55i4hzvvxdfl6rrndrarid1.dat'
    file = r'D:\2025HS_dataset\Temp\20250313_1918_49_k7dyt6gcr7bvytefr2uksfbumtpyiiolih55i4hzvvxdfl6rrndrarid2.dat'
    file = r'E:\Tor公开数据集\Tik-Tok\Undefended_CW\7-450'
    file = r'D:\2025HS_dataset\HS_longstream_flat\Drugs\Drugs\Ilegal\2a2a2abbjsjcjwfuozip6idfxsxyowoi3ajqyehqzfqyxezhacur7oyd\20250221_1932_41_2a2a2abbjsjcjwfuozip6idfxsxyowoi3ajqyehqzfqyxezhacur7oyd0.dat'
    with open(file, 'r') as f:
        tcp_dump = f.readlines()
    import pandas as pd
    seq = pd.Series(tcp_dump).str.slice(0, -1).str.split('\t', expand=True).astype("float")
    timestamps = np.array(seq.iloc[:, 0], dtype=np.float64)
    if timestamps[0] != 0:
        timestamps = timestamps - timestamps[0]
    sizes = np.array(seq.iloc[:, 1], dtype=np.int32)
    timestamps_main = timestamps
    sizes_main = sizes

    print("--- Preparing Data for Plotting ---")
    ts_plot_processed, sizes_plot_processed = handle_timestamp_issues(timestamps_main, sizes_main)
    if len(ts_plot_processed) < 3 :
        print("Not enough data points after processing to proceed with example.")
        exit()
    cumulative_sizes_plot = np.cumsum(sizes_plot_processed)

    main_segmentation_threshold = 45
    trim_slope_threshold = 0.5       
    min_segment_len_after_trim = 5   

    print(f"\n--- Scenario 1: No Trimming (Main Threshold: {main_segmentation_threshold}th percentile) ---")
    segments_no_trim = segment_time_series_by_slope_change(
        timestamps_main, sizes_main, 
        percentile_threshold=main_segmentation_threshold, 
        trim_flat_portions=False
    )
    print(f"Found {len(segments_no_trim)} segments (no trimming):")
    # for i, seg in enumerate(segments_no_trim):
    #     if seg: # Ensure segment is not empty
    #         print(f"  Segment {i+1}: {len(seg)} points, t=[{seg[0][0]:.2f} .. {seg[-1][0]:.2f}]")
    #     else:
    #         print(f"  Segment {i+1}: 0 points (empty)")


    print(f"\n--- Scenario 2: With Selective Trimming (Slope Thresh: {trim_slope_threshold}, Min Len: {min_segment_len_after_trim}) ---")
    segments_with_trim_full = segment_time_series_by_slope_change(
        timestamps_main, sizes_main, 
        percentile_threshold=main_segmentation_threshold, 
        trim_flat_portions=True, 
        flatness_slope_threshold=trim_slope_threshold, 
        min_trimmed_segment_length=min_segment_len_after_trim
    )
    # The user's original code includes this line to remove the last segment before plotting.
    # We will respect this and calculate x-limits based on the segments *after* this operation.
    if segments_with_trim_full: # Ensure there's at least one segment to remove
        segments_with_trim = segments_with_trim_full[:-1]
    else:
        segments_with_trim = []

    print(segments_with_trim)


    print(f"Found {len(segments_with_trim)} segments (selectively trimmed and last segment removed for plotting):")
    # for i, seg in enumerate(segments_with_trim):
    #     if seg: # Ensure segment is not empty
    #          print(f"  Segment {i+1}: {len(seg)} points, t=[{seg[0][0]:.2f} .. {seg[-1][0]:.2f}]")
    #     else:
    #         print(f"  Segment {i+1}: 0 points (empty)")


    colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'olive', 'lime']
    plot_legend_max_items = 7

    # Figure 1: No Trimming
    fig1, axs1 = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    # fig1.suptitle(f"Segmentation without Trimming (Main Seg. Threshold: {main_segmentation_threshold}th percentile)", fontsize=16)

    axs1[0].plot(ts_plot_processed, sizes_plot_processed, 'o-', label='Processed Sizes', markersize=3, alpha=0.6)
    axs1[0].set_title('Processed Original Sizes vs. Time')
    axs1[0].set_ylabel('Size')
    axs1[0].grid(True)

    for i, segment_data in enumerate(segments_no_trim):
        if not segment_data: continue
        segment_ts_plot = np.array([p[0] for p in segment_data])
        try:
            start_idx_plot = np.where(np.isclose(ts_plot_processed, segment_ts_plot[0]))[0][0]
            end_idx_plot = np.where(np.isclose(ts_plot_processed, segment_ts_plot[-1]))[0][0]
            if start_idx_plot > end_idx_plot: continue
            
            sub_ts = ts_plot_processed[start_idx_plot : end_idx_plot + 1]
            sub_cumulative_sizes = cumulative_sizes_plot[start_idx_plot : end_idx_plot + 1]
            
            axs1[1].plot(sub_ts, sub_cumulative_sizes,
                        linewidth=2.5, linestyle='-', color=colors[i % len(colors)],
                        label=f'Seg {i+1}' if i < plot_legend_max_items else None)
        except IndexError:
            print(f"(No Trim Plot) Error mapping segment {i+1} (t={segment_ts_plot[0]:.1f}..{segment_ts_plot[-1]:.1f}) to plot.")
    axs1[1].set_title('Cumulative Sum with Detected Segments (Original)')
    axs1[1].set_ylabel('Cumulative Size')
    axs1[1].grid(True)
    if len(segments_no_trim) > 0 and any(segments_no_trim) : axs1[1].legend(loc='upper left', title=f"Segments (first {plot_legend_max_items})")

    dt_diag = np.diff(ts_plot_processed)
    dt_diag[dt_diag <= 1e-9] = 1.0 
    local_slopes_diag = np.diff(cumulative_sizes_plot) / dt_diag
    
    abs_change_in_slopes_diag_plot = np.array([])
    ts_for_slope_changes_diag_plot = np.array([])

    if len(local_slopes_diag) >= 2:
        dt_for_slope_changes_diag = (dt_diag[:-1] + dt_diag[1:]) / 2.0
        dt_for_slope_changes_diag[dt_for_slope_changes_diag <= 1e-9] = 1.0
        change_in_slopes_diag = np.diff(local_slopes_diag) / dt_for_slope_changes_diag
        abs_change_in_slopes_diag_plot = np.abs(change_in_slopes_diag)
        # Adjust index for ts_for_slope_changes_diag_plot to align with abs_change_in_slopes_diag_plot
        # abs_change_in_slopes_diag_plot corresponds to changes between slope_i and slope_{i+1}
        # local_slopes are at midpoints of original ts_plot_processed intervals
        # change_in_slopes are at midpoints of local_slopes intervals
        # So, effectively, ts_plot_processed[1] is start for first local_slope, ts_plot_processed[2] for second, etc.
        # And for change_in_slopes, it's roughly ts_plot_processed[1] for the first point if we consider its position.
        # More accurately, it's between ts_plot_processed[i+1] and ts_plot_processed[i+2]
        # The original code `ts_plot_processed[1:len(abs_change_in_slopes_diag_plot)+1]` might be slightly off.
        # A common way is to plot it at ts_plot_processed[i+2] or average of timestamps.
        # For simplicity and consistency with original, we'll use a safe indexing.
        if len(abs_change_in_slopes_diag_plot) > 0:
             ts_for_slope_changes_diag_plot = ts_plot_processed[1 : len(abs_change_in_slopes_diag_plot) + 1] # Original approach
             # A more centered approach might be:
             # ts_for_slope_changes_diag_plot = (ts_plot_processed[1:-1] + ts_plot_processed[2:]) / 2 # if len >=3
             # However, to match length of abs_change_in_slopes_diag_plot, we need to be careful.
             # abs_change_in_slopes is len(ts_arr) - 2.
             # ts_plot_processed has len(ts_arr).
             # So ts_plot_processed[2:] or ts_plot_processed[1:-1] or ts_plot_processed[0:-2] etc.
             # The original `ts_plot_processed[1:len(abs_change_in_slopes_diag_plot)+1]` means indices 1 to N-2.
             # If len(ts_plot_processed) = L, len(abs_change_in_slopes_diag_plot) = L-2.
             # So indices are ts_plot_processed[1]...ts_plot_processed[L-2]. These are L-2 points.
             # This plots the change associated with (slope_i, slope_{i+1}) at timestamp ts_{i+1}.
             # This seems a reasonable convention.

        finite_abs_changes_diag = abs_change_in_slopes_diag_plot[np.isfinite(abs_change_in_slopes_diag_plot)]
        if len(finite_abs_changes_diag) > 0:
            threshold_val_diag_plot = np.percentile(finite_abs_changes_diag, main_segmentation_threshold)
            axs1[2].axhline(threshold_val_diag_plot, color='red', linestyle='--', label=f'Threshold ({main_segmentation_threshold}th %ile)')

        if len(ts_for_slope_changes_diag_plot) == len(abs_change_in_slopes_diag_plot) and len(ts_for_slope_changes_diag_plot)>0:
            axs1[2].plot(ts_for_slope_changes_diag_plot, abs_change_in_slopes_diag_plot, 'o-', label='Abs. Change in Slope', color='green', markersize=3, alpha=0.7)
        elif len(abs_change_in_slopes_diag_plot) > 0 : # Mismatch but still try to plot if data exists
             print(f"Warning (Fig1 Diag): Mismatch plotting slope changes. ts_len={len(ts_for_slope_changes_diag_plot)}, slope_changes_len={len(abs_change_in_slopes_diag_plot)}")
             # Fallback: plot against indices if lengths mismatch but data exists
             # axs1[2].plot(np.arange(len(abs_change_in_slopes_diag_plot)), abs_change_in_slopes_diag_plot, 'o-', label='Abs. Change in Slope (index x-axis)', color='green', markersize=3, alpha=0.7)
        else:
             axs1[2].text(0.5, 0.5, "Not enough data for slope change plot.", ha='center', va='center', transform=axs1[2].transAxes)
    else:
        axs1[2].text(0.5, 0.5, "Not enough local slopes for change calculations.", ha='center', va='center', transform=axs1[2].transAxes)
    axs1[2].set_title('Magnitude of Slope Changes (Segmentation Diagnostic)')
    axs1[2].set_ylabel('Abs(Change in Slope)')
    axs1[2].set_xlabel('Timestamp')
    if axs1[2].has_data(): axs1[2].legend() # Check if legend has items
    axs1[2].grid(True)
    if np.any(abs_change_in_slopes_diag_plot[np.isfinite(abs_change_in_slopes_diag_plot)] > 0):
         axs1[2].set_yscale('log')
    plt.tight_layout(rect=[0, 0, 1, 0.96])


    # Figure 2: With Trimming
    fig2, axs2 = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig2.suptitle(f"Segmentation with Selective Trimming (Slope Th: {trim_slope_threshold}, Min Len: {min_segment_len_after_trim})", fontsize=16)

    axs2[0].plot(ts_plot_processed, sizes_plot_processed, 'o-', label='Processed Sizes', markersize=3, alpha=0.6)
    axs2[0].set_title('Processed Original Sizes vs. Time')
    axs2[0].set_ylabel('Size')
    axs2[0].grid(True)

    # print(segments_with_trim) # Kept for debugging if needed
    for i, segment_data in enumerate(segments_with_trim):
        if not segment_data: continue
        segment_ts_plot = np.array([p[0] for p in segment_data])
        try:
            start_idx_plot = np.where(np.isclose(ts_plot_processed, segment_ts_plot[0]))[0][0]
            end_idx_plot = np.where(np.isclose(ts_plot_processed, segment_ts_plot[-1]))[0][0]
            if start_idx_plot > end_idx_plot: continue
            
            sub_ts = ts_plot_processed[start_idx_plot : end_idx_plot + 1]
            sub_cumulative_sizes = cumulative_sizes_plot[start_idx_plot : end_idx_plot + 1]
            
            axs2[1].plot(sub_ts, sub_cumulative_sizes,
                        linewidth=2.5, linestyle='-', color=colors[i % len(colors)],
                        label=f'Seg {i+1}' if i < plot_legend_max_items else None)
        except IndexError:
             print(f"(Trim Plot) Error mapping segment {i+1} (t={segment_ts_plot[0]:.1f}..{segment_ts_plot[-1]:.1f}) to plot.")
    axs2[1].set_title('Cumulative Sum with Detected Segments (Selectively Trimmed)')
    axs2[1].set_ylabel('Cumulative Size')
    axs2[1].grid(True)
    if len(segments_with_trim) > 0 and any(segments_with_trim): axs2[1].legend(loc='upper left', title=f"Segments (first {plot_legend_max_items})")

    if len(local_slopes_diag) >= 2: # Re-using diagnostic data from Figure 1 calculation
        if len(ts_for_slope_changes_diag_plot) == len(abs_change_in_slopes_diag_plot) and len(ts_for_slope_changes_diag_plot) > 0:
            axs2[2].plot(ts_for_slope_changes_diag_plot, abs_change_in_slopes_diag_plot, 'o-', label='Abs. Change in Slope', color='green', markersize=3, alpha=0.7)
            if len(finite_abs_changes_diag) > 0: # finite_abs_changes_diag also from Fig 1 calc
                 threshold_val_diag_plot = np.percentile(finite_abs_changes_diag, main_segmentation_threshold) 
                 axs2[2].axhline(threshold_val_diag_plot, color='red', linestyle='--', label=f'Threshold ({main_segmentation_threshold}th %ile)')
        elif len(abs_change_in_slopes_diag_plot) > 0:
            print(f"Warning (Fig2 Diag): Mismatch plotting slope changes. ts_len={len(ts_for_slope_changes_diag_plot)}, slope_changes_len={len(abs_change_in_slopes_diag_plot)}")
        else:
            axs2[2].text(0.5, 0.5, "Not enough data for slope change plot.", ha='center', va='center', transform=axs2[2].transAxes)
    else:
        axs2[2].text(0.5, 0.5, "Not enough local slopes for change calculations.", ha='center', va='center', transform=axs2[2].transAxes)
    axs2[2].set_title('Magnitude of Slope Changes (Segmentation Diagnostic)')
    axs2[2].set_ylabel('Abs(Change in Slope)')
    axs2[2].set_xlabel('Timestamp')
    if axs2[2].has_data(): axs2[2].legend()
    axs2[2].grid(True)
    if np.any(abs_change_in_slopes_diag_plot[np.isfinite(abs_change_in_slopes_diag_plot)] > 0):
        axs2[2].set_yscale('log')
    
    # --- MODIFICATION START: Adjust x-axis for Figure 2 ---
    min_ts_for_xlim = np.inf
    max_ts_for_xlim = -np.inf
    has_segments_for_xlim = False

    if segments_with_trim: # Check if the list of segments is not empty
        for segment_data in segments_with_trim:
            if segment_data: # Check if the current segment itself is not empty
                # segment_ts_plot = np.array([p[0] for p in segment_data]) # Not strictly needed here
                min_ts_for_xlim = min(min_ts_for_xlim, segment_data[0][0])
                max_ts_for_xlim = max(max_ts_for_xlim, segment_data[-1][0])
                has_segments_for_xlim = True
    
    if has_segments_for_xlim:
        padding_percentage = 0.05  # 5% padding on each side
        span = max_ts_for_xlim - min_ts_for_xlim
        if span <= 1e-9: # Handle case with a single point or all points at effectively the same timestamp
            # If span is zero or very small, use a fixed padding amount
            # Or base padding on a typical scale if known, e.g., 1 if timestamps are integers
            padding = 1.0 if np.all(np.mod(ts_plot_processed, 1) == 0) else 0.1 * (max_ts_for_xlim if max_ts_for_xlim > 0 else 1)
            if padding == 0: padding = 1.0 # Ensure padding is not zero
        else:
            padding = span * padding_percentage
        
        # axs2[0], axs2[1], axs2[2] share the same x-axis due to sharex=True.
        # Setting xlim for one subplot will apply to all in Figure 2.
        axs2[0].set_xlim(min_ts_for_xlim - padding, max_ts_for_xlim + padding)
    # --- MODIFICATION END ---

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()