import os
import ctypes
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from ctypes import *

'''
1. 给定一个文件夹，深度遍历文件夹下文件，获取所有.pcap格式文件。
2. 读取pcap文件，根据五元组提取所有TCP流，注意，这里五元组的源IP、端口与目的IP、端口应该可交换，即提取流的双向流。从每条流中提取特征序列，特征序列为每个数据包的相对时间戳和数据包大小，中间用制表符隔开，根据流方向分为发送和接收端，发送的数据包大小为正数，接收的数据包为负数。第一次出现的数据包为发送方。
3. 每条流的特征序列保存为一个文件
4. 将所有保存的特征文件重命名为数字，从0开始递增。
5. 考虑到文件非常多，请使用多线程完成上述工作，并提供运行时的进度条。
'''

# 根据系统加载不同库
if sys.platform == 'win32':
    lib = cdll.LoadLibrary('./pcap_processor.dll')
else:
    lib = cdll.LoadLibrary('./pcap_processor.so')

# 设置参数类型
lib.process_pcap.argtypes = [c_char_p]
lib.process_pcap.restype = c_void_p  # 修改返回类型
lib.free_result.argtypes = [c_void_p]

class StreamWriter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.counter = 0
        self._lock = threading.Lock()
        os.makedirs(output_dir, exist_ok=True)
    
    def save_stream(self, features):
        with self._lock:
            file_id = self.counter
            self.counter += 1
        
        output_path = os.path.join(self.output_dir, f"{file_id}.dat")
        with open(output_path, 'w') as f:
            f.write('\n'.join(features))
        return file_id

def get_pcap_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.pcap'):
                yield os.path.join(dirpath, f)

def process_pcap(pcap_path, writer):
    try:
        # 调用C函数
        c_ptr = lib.process_pcap(pcap_path.encode())
        if not c_ptr:
            return 0
            
        # 转换为Python字符串
        c_result = cast(c_ptr, c_char_p).value.decode("utf-8", errors="ignore")

        if not c_result:
            return 0
        lib.free_result(c_ptr)
            
        # 解析结果
        result = c_result.split("===END_STREAM===\n")
    
        saved = 0
        for stream_str in result:
            stripped = stream_str.strip()
            if not stripped:
                continue
            features = stripped.split('\n')
            if len(features) < 1:  # 确保有至少一个特征
                continue
            writer.save_stream(features)
            saved += 1
        return saved
    except Exception as e:
        print(f"[ERROR] Error processing {pcap_path}: {e}")
        return 0

def main(input_dir, output_dir):
    writer = StreamWriter(output_dir)
    pcap_files = list(get_pcap_files(input_dir))
    
    total_flows = 0
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count()*4)) as executor:
        futures = {executor.submit(process_pcap, p, writer): p for p in pcap_files}
        
        with tqdm(total=len(futures), desc="Processing PCAPs", unit="file") as pbar:
            for future in as_completed(futures):
                total_flows += future.result()
                pbar.update(1)
    
    print(f"\nSuccessfully processed {total_flows:,} TCP flows (packet count > 100)")
    print(f"Total output files: {writer.counter:,}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python NormalExtractor.py <input_dir> <output_dir>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])