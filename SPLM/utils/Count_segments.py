import os, sys, random


# 获取 segment.py 所在的路径并添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录（utils）
parent_dir = os.path.dirname(current_dir)  # 项目根目录
extractor_dir = os.path.join(parent_dir, "Extractor")
sys.path.append(extractor_dir)

TIME_END = 120

from segment import segment_time_series_by_slope_change
from Momentum_feature import Momentum_feature, segment_by_inflection

def get_HSdir(inputDir, format='pcap'):
    HSdir = [] # 待检测的目录
    for root, dirs, files in os.walk(inputDir):
        for file in files:
            if file.endswith('.'+format):
                if root not in HSdir: HSdir.append(root)
    
    return HSdir

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

def get_random_HS(folders, webnum):
    
    files = []
    
    # folders = [
    #     #r'D:\\2025HS_dataset\\HS_longstream\\Drugs\\Drugs\\Ilegal\\iwggpyxn6qv3b2twpwtyhi2sfvgnby2albbcotcysd5f7obrlwbdbkyd',
    #     #r'D:\\2025HS_dataset\\HS_longstream\\Drugs\\Drugs\\Ilegal\\rfyb5tlhiqtiavwhikdlvb3fumxgqwtg2naanxtiqibidqlox5vispqd',
    #     # r'D:\\2025HS_dataset\\HS_longstream\\Hacking\\Hacking\\prjd5pmbug2cnfs67s3y65ods27vamswdaw2lnwf45ys3pjl55h2gwqd',
    #     # r'D:\2025HS_dataset\HS_longstream\Marketplace\Black\hztsln4fi3udznlinmxnbwdey6lbehn4sinqa6ltbu4crxgqnlzdqoid'
    # ]
    

    for idx, folder in enumerate(folders):
        all_files = [f for f in os.listdir(folder) 
                    if os.path.isfile(os.path.join(folder, f))]
        
        # 随机选择webnum个文件
        selected_files = random.sample(all_files, webnum)
        for selected_file in selected_files:
            file = os.path.join(folder, selected_file)
            files.append(file)
    
    return files

if __name__=='__main__':
    path = r'D:\2025HS_dataset\HS_longstream'

    folders = random.sample(get_HSdir(path, 'dat'), 2)
    # folders = [
    #     r'D:\\2025HS_dataset\\HS_longstream\\Drugs\\Drugs\\Ilegal\\iwggpyxn6qv3b2twpwtyhi2sfvgnby2albbcotcysd5f7obrlwbdbkyd',
    #     r'D:\\2025HS_dataset\\HS_longstream\\Drugs\\Drugs\\Ilegal\\rfyb5tlhiqtiavwhikdlvb3fumxgqwtg2naanxtiqibidqlox5vispqd',
    #     r'D:\\2025HS_dataset\\HS_longstream\\Hacking\\Hacking\\prjd5pmbug2cnfs67s3y65ods27vamswdaw2lnwf45ys3pjl55h2gwqd',
    #     r'D:\2025HS_dataset\HS_longstream\Marketplace\Black\hztsln4fi3udznlinmxnbwdey6lbehn4sinqa6ltbu4crxgqnlzdqoid'
    # ]

    files = get_random_HS(folders, 1)

    for file in files:
        timestamps, packets, _, _, _, _ = count_file_seq(file)
        segments = segment_by_inflection(packets, zscore=1)
        print(f"R1 segments: {len(segments)}")
        segments = segment_time_series_by_slope_change(timestamps, packets, percentile_threshold=75)
        #print(segments)
        print(f"Gemini segments: {len(segments)}")