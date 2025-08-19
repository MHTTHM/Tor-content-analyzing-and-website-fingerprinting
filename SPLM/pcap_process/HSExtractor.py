import os
import ctypes
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from ctypes import *
import numpy as np
import shutil


'''
1. 给定一个文件夹，深度遍历文件夹下文件，获取所有.pcap格式文件。
2. 读取pcap文件，根据五元组提取所有TCP流，注意，这里五元组的源IP、端口与目的IP、端口应该可交换，即提取流的双向流。从每条流中提取特征序列，特征序列为每个数据包的相对时间戳和数据包大小，中间用制表符隔开，根据流方向分为发送和接收端，发送的数据包大小为正数，接收的数据包为负数。第一次出现的数据包为发送方。
3. 每条流的特征序列保存为一个文件
4. 将所有保存的特征文件重命名为数字，从0开始递增。
5. 考虑到文件非常多，请使用多线程完成上述工作，并提供运行时的进度条。
'''

# # 根据系统加载不同库
# if sys.platform == 'win32':
#     lib = cdll.LoadLibrary('./pcap_processor.dll')
# else:
#     lib = cdll.LoadLibrary('./pcap_processor.so')

# # 设置参数类型
# lib.process_pcap.argtypes = [c_char_p]
# lib.process_pcap.restype = c_void_p  # 修改返回类型
# lib.free_result.argtypes = [c_void_p]

class StreamWriter:
    def __init__(self, filename, output_dir):
        self.output_dir = output_dir
        # 提取文件名（不带扩展名）作为前缀
        self.id_prefix = self._remove_ext(os.path.basename(filename))
        self.counter = 0
        self._lock = threading.Lock()
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _remove_ext(self, filename):
        # 去除文件扩展名
        root, ext = os.path.splitext(filename)
        return root if ext else filename

    def save_stream(self, features):
        with self._lock:
            file_id = f"{self.id_prefix}"
            self.counter += 1
        
        output_path = os.path.join(self.output_dir, f"{file_id}.dat")
        # 写入特征到文件
        with open(output_path, 'w') as f:
            f.write('\n'.join(features))
        return file_id

def get_HSdir(inputDir, format='pcap'):
    HSdir = [] # 待检测的目录
    for root, dirs, files in os.walk(inputDir):
        for file in files:
            if file.endswith('.'+format):
                if root not in HSdir: HSdir.append(root)
    
    return HSdir

def delete_small_folders(inputDir, format='dat', min_files=100):
    """
    删除包含少于指定数量文件的文件夹
    
    """
    # 获取所有包含指定格式文件的文件夹
    folders_to_check = get_HSdir(inputDir, format)
    
    del_folders = 0
    for folder in folders_to_check:
        try:
            # 计算文件夹中指定格式的文件数量
            file_count = len([f for f in os.listdir(folder) if f.endswith('.' + format)])
            
            # 如果文件数量少于阈值，删除文件夹
            if file_count < min_files:
                del_folders += 1
                #print(f"删除文件夹 '{folder}' (仅包含 {file_count} 个 {format} 文件)")
                shutil.rmtree(folder)
        except Exception as e:
            print(f"处理文件夹 '{folder}' 时出错: {str(e)}")
    
    print(f"删除 '{del_folders}'  个少于 {min_files} 个文件的文件夹")

def get_pcap_files(input_dir, format='pcap'):
    """递归获取所有pcap文件及其相对路径"""
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.'+format):
                abs_path = os.path.join(root, file)
                # 计算相对于输入目录的路径
                rel_path = os.path.relpath(abs_path, input_dir)
                # 获取目录部分（去掉文件名）
                rel_dir = os.path.dirname(rel_path)
                yield (abs_path, rel_dir)

# 删除过大和过小的文件
def abnormal_files(directory):
    # 收集文件及其大小
    files = []
    for entry in os.scandir(directory):
        if entry.is_file():
            try:
                size = os.path.getsize(entry.path)
                files.append({'path': entry.path, 'size': size})
            except OSError as e:
                print(f"无法获取文件 {entry.path} 的大小：{e}")

    if not files:
        print("文件夹中没有文件。")
        return

    sizes = [f['size'] for f in files]

    # 计算四分位数和IQR
    q1 = np.percentile(sizes, 25)
    q3 = np.percentile(sizes, 75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # 找出异常文件
    small_outliers = [f for f in files if f['size'] < lower_bound]
    large_outliers = [f for f in files if f['size'] > upper_bound]
    outliers = small_outliers + large_outliers

    abnormal_files = []
    for f in outliers:
        abnormal_files.append(f['path'])
    
    return abnormal_files

def getall_abnormal(dir, format='pcap'):
    
    x = get_HSdir(dir, format)
    
    abnormals = []
    for i in x:
        try:
            abfiles = abnormal_files(i)
            abnormals = list(set(abnormals) | set(abfiles))
        except Exception as e:
            print(f"[ERROR] Processing {i}: {str(e)}")
    
    return abnormals

def process_pcap(pcap_path, rel_dir, input_dir, output_dir):
    # 根据系统加载不同库
    if sys.platform == 'win32':
        lib = cdll.LoadLibrary('./pcap_processor.dll')
    else:
        lib = cdll.LoadLibrary('./pcap_processor.so')

    # 设置参数类型
    lib.process_pcap.argtypes = [c_char_p]
    lib.process_pcap.restype = c_void_p  # 修改返回类型
    lib.free_result.argtypes = [c_void_p]
    try:
        output_subdir = os.path.join(output_dir, rel_dir)
        writer = StreamWriter(os.path.basename(pcap_path), output_subdir)

        c_ptr = lib.process_pcap(pcap_path.encode())
        if not c_ptr:
            return 0, 0
        
        c_result = cast(c_ptr, c_char_p).value.decode("utf-8", errors="ignore")
        lib.free_result(c_ptr)

        if not c_result:
            return 0, 0
        
        result = c_result.split("===END_STREAM===\n")

        max_features = None
        max_length = 0

        for stream_str in result:
            stripped = stream_str.strip()
            if not stripped:
                continue

            features = stripped.split('\n')
            current_length = len(features)
            
            # 过滤条件：包数量 >= 100（根据原逻辑保持相同条件）
            if current_length < 100:  
                continue

            # 更新最长流记录
            if current_length > max_length:
                max_length = current_length
                max_features = features

        saved_flows = 0
        # 只保存最长的流
        if max_features is not None:
            writer.save_stream(max_features)
            saved_flows = 1

        return saved_flows, writer.counter
    
    except Exception as e:
        print(f"[ERROR] Processing {pcap_path}: {str(e)}")
        return 0, 0

def create_directory(path):
    """
    处理目录存在与否
    """
    if not os.path.exists(path):
        os.makedirs(path)

        return True
    else:
        while True:
            choice = input(f"目录 '{path}' 已存在请选择操作: [O]覆盖/[M]合并/[C]取消: ").upper()
            
            if choice == 'O':  # 覆盖
                # 清空目录
                for filename in os.listdir(path):
                    file_path = os.path.join(path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"删除 {file_path} 失败: {e}")
                
                return True
                
            elif choice == 'M':  # 合并
                print("保持目录内容不变")
                return True
                
            elif choice == 'C':  # 取消
                print("操作已取消")
                return False
                
            else:
                print("无效输入，请重新选择")


def main(input_dir, output_dir):

    create_directory(output_dir)

    suffix = 'pcap'
    all_pcap_files = list(get_pcap_files(input_dir, format=suffix))
    print(f"总共获取{suffix}文件 {len(all_pcap_files)} 个")

    total_flows = 0
    total_files = 0
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count()*4)) as executor:
        futures = {executor.submit(
                    process_pcap, 
                    pcap_path,
                    rel_dir,
                    input_dir,
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

    delete_small_folders(output_dir, 'dat', 100)

    abfiles = getall_abnormal(output_dir, 'dat')
    for i in abfiles:
        os.remove(i)

    print(f"\n过滤大小异常文件 {len(abfiles)} 个")

    categories = len(get_HSdir(output_dir, 'dat'))
    print(f"最终处理数据集类别数量: {categories:,}")

if __name__ == "__main__":
    

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory containing PCAPs")
    parser.add_argument("--output", required=True, help="Output directory")
    if len(sys.argv) != 5:
        print("Usage: python pcap_filtered.py <input_dir> <output_dir>")
        sys.exit(1)
    args = parser.parse_args()
        
    main(args.input, args.output)