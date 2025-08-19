import os, sys
import numpy as np
from typing import List, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录（utils）
parent_dir = os.path.dirname(current_dir)  # 项目根目录
extractor_dir = os.path.join(parent_dir, "Extractor")
sys.path.append(extractor_dir)
process_dir = os.path.join(parent_dir, 'pcap_process')
sys.path.append(process_dir)

from segment import handle_timestamp_issues,segment_time_series_by_slope_change
from HSExtractor import create_directory, get_pcap_files, StreamWriter

def gemini_remove_flat(
    timestamps: List[float],
    sizes: List[int],
    percentile_threshold=85, 
    duplicate_timestamp_strategy='mean',
    trim_flat_portions=False,
    flatness_slope_threshold=0.5,
    min_trimmed_segment_length=3
) -> Tuple[List[float], List[int]]:
    segments_with_trim_full = segment_time_series_by_slope_change(
        timestamps, sizes, 
        percentile_threshold=percentile_threshold, 
        trim_flat_portions=True, 
        flatness_slope_threshold=flatness_slope_threshold, 
        min_trimmed_segment_length=min_trimmed_segment_length
    )
    if segments_with_trim_full: # Ensure there's at least one segment to remove
        segments_with_trim = segments_with_trim_full[:-1]
    else:
        segments_with_trim = []
    
    pure_segments = []
    for segment in segments_with_trim:
        for i, j in segment:
            k = str(i) + '\t' + str(j)
            pure_segments.append(k)
    
    return pure_segments

def read_dat_file(file):
    with open(file, 'r') as f:
        tcp_dump = f.readlines()
    import pandas as pd
    seq = pd.Series(tcp_dump).str.slice(0, -1).str.split('\t', expand=True).astype("float")
    timestamps = np.array(seq.iloc[:, 0], dtype=np.float64)
    if timestamps[0] != 0:
        timestamps = timestamps - timestamps[0]
    sizes = np.array(seq.iloc[:, 1], dtype=np.int32)
    return timestamps, sizes

def process_flat_dat(dat_path, rel_dir, output_dir):
    try:
        output_subdir = os.path.join(output_dir, rel_dir)
        timestamps, sizes = read_dat_file(dat_path)
        pure_segments = gemini_remove_flat(timestamps, sizes)
        writer = StreamWriter(os.path.basename(dat_path), output_subdir)
        writer.save_stream(pure_segments)
        saved_flows = 1

        return saved_flows, writer.counter

    except Exception as e:
        print(f"[ERROR] Processing {dat_path}: {str(e)}")
        return 0, 0

def main(input_dir, output_dir):
    create_directory(output_dir)

    suffix = 'dat'
    total_flows = 0
    total_files = 0

    all_pcap_files = list(get_pcap_files(input_dir, format=suffix))
    print(f"总共获取{suffix}文件 {len(all_pcap_files)} 个")

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count()*4)) as executor:
        futures = {executor.submit(
                    process_flat_dat, 
                    pcap_path,
                    rel_dir,
                    output_dir): (pcap_path, rel_dir) for pcap_path, rel_dir in all_pcap_files
                    }

        with tqdm(total=len(futures), desc="Processing PCAPs", unit="file") as pbar:
            for future in as_completed(futures):
                flows, files = future.result()
                total_flows += flows
                total_files += files
                pbar.update(1)
        
    print(f"\n成功处理TCP流 {total_flows:,} 条 (packet count > 100)")
    print(f"总生成特征文件: {total_files:,}")

if __name__=='__main__':
    input_dir = r'D:\2025HS_dataset\HS_longstream'
    output_dir = r'D:\2025HS_dataset\HS_longstream_flat'
    main(input_dir, output_dir)