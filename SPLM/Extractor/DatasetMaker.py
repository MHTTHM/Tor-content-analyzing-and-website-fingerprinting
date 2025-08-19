import os
import numpy as np
import pickle
import h5py
from sklearn.model_selection import train_test_split
import multiprocessing as mp
import tqdm
import pandas as pd
import argparse
from pathlib import Path
from typing import Union, Tuple, List
from collections import Counter
import json
from functools import partial

try:
    from .Momentum_feature import Momentum_feature, remove_flat_segments, Momentum_feature_label_to_MMM_indices # 作为包导入时
except ImportError:
    from Momentum_feature import Momentum_feature, remove_flat_segments, Momentum_feature_label_to_MMM_indices  # 直接运行时

try:
    from .TAM_feature import packets_perslot, MAX_TRACE_LENGHT
except ImportError:
    from TAM_feature import packets_perslot, MAX_TRACE_LENGHT

try:
    from .segment_by_gemini import Gemini_Momentum_feature
except ImportError:
    from Extractor.segment_by_gemini import Gemini_Momentum_feature

SPLIT_MARK = '\t' # '\t' ' '
    
def get_filename_list(folder_path, suffix=None):
    filename_list = []
    
    # 遍历文件夹中的所有条目
    for entry in os.listdir(folder_path):
        # 拼接完整路径
        full_path = os.path.join(folder_path, entry)
        
        # 检查是否是文件
        if os.path.isfile(full_path):
            # 如果不指定后缀，或者文件以指定后缀结尾，则添加到列表
            if suffix is None or full_path.lower().endswith(suffix.lower()):
                filename_list.append(full_path)
    
    return filename_list

def extract_feature_wrapper(para, feature_func, **kwargs):
    """
    包装函数，用于统一调用不同的特征提取函数
    """
    file_path, label = para

    with open(file_path, 'r') as f:
        tcp_dump = f.readlines()#[:MAX_TRACE_LENGHT]
    
    seq = pd.Series(tcp_dump).str.slice(0, -1).str.split(SPLIT_MARK, expand=True).astype("float")
    
    # 提取时间和包长序列
    times = np.array(seq.iloc[:, 0], dtype=np.float64)
    if times[0] != 0:
        times = times - times[0]
    length_seq = np.array(seq.iloc[:, 1], dtype=np.int32)
    
    flat = kwargs.get('flat', False)
    if flat:
        flat_segment_threshold = kwargs.get('flat_segment_threshold', None)
        flat_segment_zscore = kwargs.get('flat_segment_zscore', None)
        rm_threshold = kwargs.get('rm_threshold', None)
        rm_zscore = kwargs.get('rm_zscore', None)
        times, length_seq = remove_flat_segments(times, 
                                length_seq, 
                                flat_segment_threshold,
                                flat_segment_zscore,
                                rm_threshold,
                                rm_zscore
                                )

    # 调用指定的特征提取函数
    if kwargs.get('multi-tab'):
        labelname = os.path.splitext(os.path.basename(file_path))[0] + '_labels.json'
        labelpath = os.path.join(os.path.dirname(file_path), labelname)

        with open(labelpath, 'r', encoding='utf-8') as f:
            labeljson = json.load(f)
        feature, label_dicts = feature_func(times,length_seq, labeljson, **kwargs)
        label = [[], []]
        for label_dict in label_dicts:
            label[0].append([label_dict['mmm_start_idx'], label_dict['mmm_end_idx']])
            label[1].append(label_dict['class'])
    else:
        feature = feature_func(times, length_seq, **kwargs)
    
    return feature, label

def parallel(func, para_list, n_jobs=16, **kwargs):
    """通用的并行处理函数"""
    with mp.Pool(n_jobs) as pool:
        func = partial(func, **kwargs)
        # 正确传递参数结构
        progress_bar = tqdm.tqdm(
            pool.imap(func, para_list),
            total=len(para_list),
            desc="Processing"
        )
        results = list(progress_bar)
    return results

def onehot_label(labels):
    labels = np.array(labels)
    num_classes = len(np.unique(labels))
    one_hot_labels = np.eye(num_classes)[labels]
    return one_hot_labels

def numeric_label(labels):
    numlabels = np.argmax(labels, axis=1)
    return numlabels

class DatasetMaker:
    def __init__(self, input_path: str, feature_func="momentum", **kwargs):
        self.features = None
        self.labels = None
        self.original_label_type = None
        self.openworld_processed = False
        self.unmon_test_features = None
        self.unmon_test_labels = None
        self.dominant_class = None
        self.MONITORED_SITE_NUM = 100

        if feature_func == "momentum":
            if kwargs.get('multi-tab'):
                self.feature_func = Momentum_feature_label_to_MMM_indices
                self.funcargs = {
                    'threshold': kwargs.get('threshold', 1000),
                    'zscore': kwargs.get('zscore', 0.15),
                    'momentum_val': kwargs.get('momentum_rate', 0.1),
                    'multi-tab': kwargs.get('multi-tab', True)
                }
                self.original_label_type = 'multi-tab'
            else:
                self.feature_func = Momentum_feature
                # 设置默认参数
                self.funcargs = {
                    'threshold': kwargs.get('threshold', None),
                    'zscore': kwargs.get('zscore', 0.15),
                    'momentum_val': kwargs.get('momentum_rate', 0.1),
                    'flat': kwargs.get('flat', False),
                    'flat_segment_threshold': kwargs.get('flat_segment_threshold', None),
                    'flat_segment_zscore': kwargs.get('flat_segment_zscore', None),
                    'rm_threshold': kwargs.get('rm_threshold', None),
                    'rm_zscore': kwargs.get('rm_zscore', None)
                }
        elif feature_func == "TAM":
            self.feature_func = packets_perslot
            self.funcargs = {}  # packets_perslot不需要参数
        elif feature_func == "segment":
            self.feature_func = Gemini_Momentum_feature
            self.funcargs = kwargs
        
        self._load_data(input_path)
        
    def _load_data(self, path: str):
        """加载数据的私有方法"""
        if os.path.isdir(path):
            self.features, self.labels = self._read_folder(path)
        else:
            self.features, self.labels = self._read_file(path)
        
        if not self.original_label_type == 'multi-tab':
            self.original_label_type = self._check_label_format()

    def _read_file(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """读取单个文件并自动识别格式"""
        # 1. 路径校验
        if not os.path.exists(path):
            raise FileNotFoundError(f"路径不存在: {path}")
        if not os.path.isfile(path):
            raise ValueError(f"路径不是文件: {path}")

        # 2. 读取文件头识别格式
        with open(path, 'rb') as f:
            header = f.read(8)

        # 3. 根据文件头选择加载器
        if len(header) >= 6 and header[:6] == b'\x93NUMPY':
            data_dict = self._load_npy(path)
        elif len(header) >= 8 and header[:4] == b'PK\x03\x04':
            data_dict = self._load_npy(path)
        elif len(header) >= 8 and header == b'\x89HDF\r\n\x1a\n':
            data_dict = self._load_h5py(path)
        elif len(header) >= 2 and header[0] == 0x80 and header[1] in {2, 3, 4, 5}:
            data_dict = self._load_pickle(path)
        else:
            raise ValueError("无法识别的文件格式")

        # 4. 统一键名查找逻辑
        feature_keys = {'dataset', 'features', 'feature', 'X', 'x'}
        label_keys = {'labels', 'label', 'targets', 'target', 'y'}

        # 查找特征数据
        features = None
        for key in feature_keys:
            if key in data_dict:
                features = data_dict[key]
                break
        if features is None:
            raise KeyError(f"未找到特征数据，尝试的键: {feature_keys}")

        # 查找标签数据
        labels = None
        for key in label_keys:
            if key in data_dict:
                labels = data_dict[key]
                break
        if labels is None:
            raise KeyError(f"未找到标签数据，尝试的键: {label_keys}")

        return np.asarray(features), np.asarray(labels)

    def _load_npy(self, path: str) -> dict:
        """加载.npy/.npz文件"""
        data = np.load(path, allow_pickle=True)
        return {key: data[key] for key in data.files} if isinstance(data, np.lib.npyio.NpzFile) else {'feature': data}

    def _load_h5py(self, path: str) -> dict:
        """加载HDF5文件"""
        data_dict = {}
        with h5py.File(path, 'r') as f:
            for key in f.keys():
                data_dict[key] = f[key][:]
        return data_dict

    def _load_pickle(self, path: str) -> dict:
        """加载Pickle文件"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise ValueError("Pickle文件内容不是字典格式")
        return data

    def select_classes(self, n_classes: int, random_state: int = 42) -> None:
        """
        Select a specified number of classes from the dataset, prioritizing non-dominant classes,
        and remap their labels to 0-(n_classes), where the dominant class (if exists) is always the last label.
        Only works when original_label_type is 'numeric'.

        Args:
            n_classes: Number of non-dominant classes to select (total classes = n_classes + 1 if dominant exists)
            random_state: Random seed for reproducibility
        """
        if self.original_label_type != 'numeric':
            raise ValueError("Class selection only works with numeric labels")

        if n_classes <= 0:
            raise ValueError("Number of classes must be positive")

        unique_labels = np.unique(self.labels)
        if len(unique_labels) < n_classes:
            raise ValueError(f"Dataset only has {len(unique_labels)} classes, cannot select {n_classes}")

        # Identify dominant class
        is_dominant, dominant_label = self.has_openworld_class()

        # Prepare class pool for selection
        if is_dominant:
            print(f"检测到主导类 {dominant_label}，将从非主导类中选择 {n_classes} 个类别，主导类将作为最后一类（标签={n_classes}）")
            class_pool = unique_labels[unique_labels != dominant_label]

            # Check if we have enough non-dominant classes
            if len(class_pool) < n_classes:
                raise ValueError(
                    f"只有 {len(class_pool)} 个非主导类，无法选择 {n_classes} 个类别。"
                    f"请减少 n_classes 或移除主导类检测逻辑。"
                )

            # Select n_classes from non-dominant classes
            np.random.seed(random_state)
            selected_classes = np.random.choice(class_pool, size=n_classes, replace=False)

            # Add dominant class to the selected classes (will be mapped to n_classes)
            selected_classes = np.append(selected_classes, dominant_label)
        else:
            print("未检测到主导类，将从所有类别中随机选择")
            class_pool = unique_labels
            np.random.seed(random_state)
            selected_classes = np.random.choice(class_pool, size=n_classes, replace=False)

        # Create mask for selected classes
        mask = np.isin(self.labels, selected_classes)

        if not np.any(mask):
            raise ValueError("No samples found for the selected classes")

        # Filter features and labels
        self.features = self.features[mask]
        original_labels = self.labels[mask]

        # Create label mapping:
        # - Non-dominant classes -> 0 to n_classes-1 (sorted)
        # - Dominant class (if exists) -> n_classes
        if is_dominant:
            # Non-dominant classes are sorted and mapped to 0, 1, ..., n_classes-1
            # Dominant class is forced to be the last label (n_classes)
            non_dominant_classes = sorted(selected_classes[:-1])  # exclude dominant class
            label_mapping = {old_label: new_label 
                            for new_label, old_label in enumerate(non_dominant_classes)}
            label_mapping[dominant_label] = n_classes  # dominant class gets highest label
        else:
            # No dominant class, just map all selected classes to 0-(n_classes-1)
            label_mapping = {old_label: new_label 
                            for new_label, old_label in enumerate(sorted(selected_classes))}

        # Remap labels
        self.labels = np.vectorize(label_mapping.get)(original_labels)
        print(f"类别数量: {len(np.unique(self.labels))}（{'包含主导类' if is_dominant else '不包含主导类'}）")

        # Update the original label type (still numeric)
        self.original_label_type = 'numeric'

    def _read_folder(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """处理文件夹数据集"""
        def find_leaf_directories(root_dir):
            """递归查找所有最底层文件夹（没有子目录的文件夹）"""
            leaf_dirs = []
            for dirpath, dirnames, filenames in os.walk(root_dir):
                if not dirnames:  # 没有子目录时加入列表
                    if filenames:
                        leaf_dirs.append(dirpath)
            return leaf_dirs
        
        para_list = []
        # 判断是否需要深度遍历（当路径包含子目录时）
        if any(os.path.isdir(os.path.join(path, name)) for name in os.listdir(path)):
            # 深度遍历模式：每个最底层文件夹作为一个类别
            leaf_dirs = find_leaf_directories(path)
            dir_labels = {os.path.basename(d): idx for idx, d in enumerate(sorted(set(leaf_dirs)))}

            # 保存路径与标签的映射关系
            label_mapping = {
                'path_to_label': {d: idx for idx, d in enumerate(sorted(set(leaf_dirs)))},
                'label_to_path': {idx: d for idx, d in enumerate(sorted(set(leaf_dirs)))}
            }
            
            # 保存到当前目录下的label_mapping.json文件
            with open('label_mapping.json', 'w') as f:
                json.dump(label_mapping, f, indent=4)
            #print("label mapping has been saved at label_mapping.json.")

            for dir_path in leaf_dirs:
                # 获取文件夹名称作为标签
                dir_name = os.path.basename(dir_path)
                label = dir_labels[dir_name]
                para_list.extend([
                    (os.path.join(dir_path, f), label)
                    for f in os.listdir(dir_path)
                    if os.path.isfile(os.path.join(dir_path, f))
                ])
        
        elif self.feature_func == Momentum_feature_label_to_MMM_indices:
            for f in get_filename_list(path, 'dat'):
                para_list.append((f, 0))

        else:
            # 普通模式：直接读取文件夹下的文件
            # 先处理所有m-n格式的文件，找出最大标签值
            max_label = -1
            m_n_files = []
            pure_number_files = []
            
            for f in get_filename_list(path):
                file_name = os.path.basename(f)
                
                # 解析文件名获取标签
                if '-' in file_name:
                    label = int(file_name.split('-')[0])
                    max_label = max(max_label, label)
                    m_n_files.append((f, label))
                else:
                    pure_number_files.append(f)
            
            # 设置MONITORED_SITE_NUM为最大标签值+1
            self.MONITORED_SITE_NUM = max_label + 1 if max_label != -1 else 0
            
            # 处理纯数字文件
            para_list = m_n_files + [
                (f, self.MONITORED_SITE_NUM) 
                for f in pure_number_files
            ]
        
        p = []
        for a,b in para_list:
            p.append(b)

        raw_data = parallel(extract_feature_wrapper, para_list, n_jobs=15, 
                            feature_func=self.feature_func, **self.funcargs)

        # 分离特征和标签
        features, labels = zip(*raw_data)

        max_len = max(len(sub) for row in features for sub in row)

        padded_features = [
            [sub + [0] * (max_len - len(sub)) for sub in row]
            for row in features
        ]

        arr_features = np.array(padded_features)

        return arr_features, np.array(labels)

    def select_samples(self, n_samples_per_class=100, random_state=42):
        """
        从每个类别中随机选取固定数量的样本，考虑主导类情况
        
        参数:
            n_samples_per_class: 每类选取的样本数
            random_state: 随机种子
            
        逻辑:
            1. 如果存在主导类(数量超过其他类10倍以上):
                - 从非主导类中选取n_samples_per_class样本
                - 保留所有主导类样本
                - 合并结果
            2. 如果不存在主导类:
                - 从每类中选取n_samples_per_class样本
        """
        np.random.seed(random_state)
        
        # 判断是否存在主导类
        is_dominant, dominant_label = self.has_openworld_class()
        
        selected_features_list = []
        selected_labels_list = []
        
        if is_dominant:
            # 情况1：存在主导类
            print(f"检测到主导类: {dominant_label}, 将保留所有主导类样本")
            
            # 处理主导类（保留全部样本）
            dominant_indices = np.where(self.labels == dominant_label)[0]
            selected_features_list.append(self.features[dominant_indices])
            selected_labels_list.append(self.labels[dominant_indices])
            
            # 处理非主导类
            other_labels = np.unique(self.labels[self.labels != dominant_label])
            for label in other_labels:
                indices = np.where(self.labels == label)[0]
                n_samples = min(n_samples_per_class, len(indices))
                
                if n_samples == 0:
                    continue
                elif n_samples == len(indices):
                    selected_indices = indices
                else:
                    selected_indices = np.random.choice(indices, size=n_samples, replace=False)
                
                selected_features_list.append(self.features[selected_indices])
                selected_labels_list.append(self.labels[selected_indices])
        else:
            # 情况2：不存在主导类，按原逻辑处理
            print("未检测到主导类，按标准采样逻辑处理")
            unique_labels = np.unique(self.labels)
            
            for label in unique_labels:
                indices = np.where(self.labels == label)[0]
                n_samples = min(n_samples_per_class, len(indices))
                
                if n_samples == 0:
                    continue
                elif n_samples == len(indices):
                    selected_indices = indices
                else:
                    selected_indices = np.random.choice(indices, size=n_samples, replace=False)
                
                selected_features_list.append(self.features[selected_indices])
                selected_labels_list.append(self.labels[selected_indices])
        
        # 合并结果并更新类属性
        if selected_features_list:  # 确保列表不为空
            self.features = np.concatenate(selected_features_list, axis=0)
            self.labels = np.concatenate(selected_labels_list, axis=0)
        else:
            self.features = np.array([])
            self.labels = np.array([])
        
        return self

    def has_openworld_class(self, threshold=10):
        """
        判断标签数组中是否存在某一类的数量超过其他类的10倍以上
        支持数字标签和One-Hot编码
        
        参数:
            labels: numpy.ndarray, 数字标签数组或One-Hot编码矩阵
            threshold: int, 倍数阈值，默认为10
            
        返回:
            bool: 是否存在主导类
            int/None: 主导类的标签(如果存在)，否则为None
        """
        # 检测输入类型
        if self._check_label_format() == "numeric":
            # 数字标签情况
            unique, counts = np.unique(self.labels, return_counts=True)
        elif self._check_label_format() == "single_onehot":
            # One-Hot编码情况，统计每列的和作为类别计数
            counts = self.labels.sum(axis=0)
            unique = np.arange(len(counts))
        else:
            return False, None
        
        # 过滤掉计数为0的类别（特别是One-Hot编码可能有全0列）
        mask = counts > 0
        counts = counts[mask]
        unique = unique[mask]
        
        # 如果没有数据或只有一个类别，返回False
        if len(counts) < 2:
            return False, None
        
        # 对计数进行排序(降序)
        sorted_indices = np.argsort(-counts)
        sorted_counts = counts[sorted_indices]
        sorted_unique = unique[sorted_indices]
        
        # 检查最大类是否超过第二类的threshold倍
        if sorted_counts[0] > threshold * sorted_counts[1]:
            return True, sorted_unique[0]
        
        return False, None

    def _check_label_format(self):
        """
        判断标签的格式：
        - 返回 'numeric'：数字标签
        - 返回 'single_onehot'：单标签 one-hot
        - 返回 'multi_onehot'：多标签 one-hot
        """
        if self.labels.ndim == 1:
            # 1D 数组，可能是数字标签
            if np.issubdtype(self.labels.dtype, np.integer):
                return 'numeric'
            else:
                raise ValueError("1D 数组的标签不是整数类型，无法确定格式。")
        elif self.labels.ndim == 2:
            # 2D 数组，可能是 one-hot 格式
            if np.all(np.logical_or(self.labels == 0, self.labels == 1)):  # 确保所有值为 0 或 1
                row_sums = np.sum(self.labels, axis=1)
                if np.all(row_sums == 1):
                    return 'single_onehot'  # 每行只有一个 1，是单标签 one-hot
                else:
                    return 'multi_onehot'  # 每行有多个 1，是多标签 one-hot
            else:
                raise ValueError("2D 数组的标签不是 one-hot 格式（值不全为 0 或 1）。")
        else:
            raise ValueError("标签格式既不是 numeric 也不是 onehot，无法确定格式。")

    def convert_labels(self, target_format: str) -> None:
        """转换标签格式"""
        if target_format == self.original_label_type:
            return

        if self.original_label_type == 'numeric':
            if target_format == 'single_onehot':
                self.labels = onehot_label(self.labels)
            else:
                raise ValueError("Cannot convert numeric labels to multi_onehot directly")

        elif self.original_label_type == 'single_onehot':
            if target_format == 'numeric':
                self.labels = numeric_label(self.labels)
            else:
                raise ValueError("Cannot convert single_onehot labels to multi_onehot directly")

        elif self.original_label_type == 'multi_onehot':
            if target_format == 'numeric':
                raise ValueError("Cannot convert multi_onehot labels to numeric directly")
            else:
                return  # 保持原格式

        self.original_label_type = target_format

    def split_dataset(self, proportions: List[float] = None, random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """划分数据集，支持开放世界场景"""
        if self.openworld_processed:
            dominant_class = self.dominant_class

            # 分离监控数据和非监控训练数据
            monitored_mask = self.labels != dominant_class  # 使用动态检测的类
            X_mon = self.features[monitored_mask]
            y_mon = self.labels[monitored_mask]
            X_unmon_train = self.features[~monitored_mask]
            y_unmon_train = self.labels[~monitored_mask]

            # 处理开放世界情况
            if proportions is None:
                proportions = [0.8, 0.2]  # 默认比例

            # 分割监控数据
            if len(proportions) == 2:
                X_mon_train, X_mon_test, y_mon_train, y_mon_test = train_test_split(
                    X_mon, y_mon, test_size=proportions[1], 
                    random_state=random_state, stratify=y_mon
                )
                # 合并训练集和测试集
                X_train = np.concatenate([X_mon_train, X_unmon_train])
                y_train = np.concatenate([y_mon_train, y_unmon_train])
                X_test = np.concatenate([X_mon_test, self.unmon_test_features])
                y_test = np.concatenate([y_mon_test, self.unmon_test_labels])

                return [(X_train, y_train), (X_test, y_test)]
            
            elif len(proportions) == 3:
                # 分割为train/val/test
                X_mon_train, X_rem, y_mon_train, y_rem = train_test_split(
                    X_mon, y_mon, test_size=proportions[1]+proportions[2],
                    random_state=random_state, stratify=y_mon
                )
                X_mon_val, X_mon_test, y_mon_val, y_mon_test = train_test_split(
                    X_rem, y_rem, test_size=proportions[2]/(proportions[1]+proportions[2]),
                    random_state=random_state
                )
                # 合并各部分
                X_train = np.concatenate([X_mon_train, X_unmon_train])
                y_train = np.concatenate([y_mon_train, y_unmon_train])
                X_val, y_val = X_mon_val, y_mon_val
                X_test = np.concatenate([X_mon_test, self.unmon_test_features])
                y_test = np.concatenate([y_mon_test, self.unmon_test_labels])
                return [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
            else:
                raise ValueError("Invalid proportions length")
        else:
            # 原有普通分割逻辑
            if not proportions:
                raise ValueError("Proportions must be specified in non-openworld mode")
            
            if not (0.999 < sum(proportions) <= 1.001):
                raise ValueError("Proportions must sum to 1")

            if len(proportions) == 2:
                X_train, X_test, y_train, y_test = train_test_split(
                    self.features, self.labels,
                    test_size=proportions[1],
                    random_state=random_state,
                    stratify=self.labels if self.original_label_type == 'numeric' else None
                )
                return [(X_train, y_train), (X_test, y_test)]

            elif len(proportions) == 3:
                X_train, X_rem, y_train, y_rem = train_test_split(
                    self.features, self.labels,
                    test_size=proportions[1]+proportions[2],
                    random_state=random_state,
                    stratify=self.labels if self.original_label_type == 'numeric' else None
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_rem, y_rem,
                    test_size=proportions[2]/(proportions[1]+proportions[2]),
                    random_state=random_state
                )
                return [(X_train, y_train), (X_val, y_val), (X_test, y_test)]

    def handle_openworld(self, openworld_params: Tuple[int, int]):
        """处理开放世界场景（自动检测主导类版本）"""
        # 检测是否存在主导类
        has_dominant, dominant_class = self.has_openworld_class()
        if not has_dominant:
            raise ValueError("No dominant class found for openworld scenario")

        # 分离监控和非监控数据
        monitored_mask = self.labels != dominant_class  # 修改判断逻辑
        unmonitored_mask = self.labels == dominant_class

        X_mon = self.features[monitored_mask]
        y_mon = self.labels[monitored_mask]
        X_unmon = self.features[unmonitored_mask]
        y_unmon = self.labels[unmonitored_mask]

        # 处理采样数量（保证不越界）
        train_unmon_num = min(openworld_params[0], X_unmon.shape[0])
        test_unmon_num = min(openworld_params[1], X_unmon.shape[0] - train_unmon_num)

        # 分割非监控数据
        train_unmon = X_unmon[:train_unmon_num]
        test_unmon = X_unmon[train_unmon_num:train_unmon_num+test_unmon_num]

        # 合并训练数据（保持主导类标签不变）
        self.features = np.concatenate([X_mon, train_unmon])
        self.labels = np.concatenate([y_mon, np.full(train_unmon_num, dominant_class)])  # 使用检测到的主导类标签
        
        # 保存测试数据
        self.unmon_test_features = test_unmon
        self.unmon_test_labels = np.full(test_unmon_num, dominant_class)  # 保持标签一致性
        
        # 存储状态信息
        self.openworld_processed = True
        self.dominant_class = dominant_class  # 新增属性存储主导类

    def save(self, output_path: str, datasets: List[Tuple[np.ndarray, np.ndarray]]):
        """保存数据集"""
        from pathlib import Path

        output_path = os.path.normpath(output_path)
        
        # 创建输出目录
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 解析基础文件名和扩展名
        base_name = Path(output_path).stem
        ext = Path(output_path).suffix
        
        # 确定数据集后缀 (train/val/test)
        if(len(datasets)==2):
            suffixes = ['train', 'test']
        else:
            suffixes = ['train', 'val', 'test']
        
        for suffix, (X, y) in zip(suffixes, datasets):
            # 构建新的文件名
            if len(datasets) > 1:
                new_filename = f"{base_name}_{suffix}{ext}"
            else:
                new_filename = f"{base_name}{ext}"
            
            full_path = output_dir / new_filename
            if os.path.exists(full_path):
                while True:
                    choice = input(f"文件 '{os.path.basename(full_path)}' 已存在，请选择:\n"
                                "1. 覆盖\n"
                                "2. 重命名\n"
                                "请输入选项(1/2): ").strip()
                    
                    if choice == '1':  # 覆盖
                        os.remove(full_path)
                        break
                        
                    elif choice == '2':  # 重命名
                        while True:
                            new_name = input("请输入新文件名(包含扩展名): ").strip()
                            if not new_name:
                                print("文件名不能为空，请重新输入。")
                                continue
                                
                            new_path = os.path.join(os.path.dirname(full_path), new_name)
                            if not os.path.exists(new_path):
                                full_path = new_path
                                break
                            else:
                                print(f"错误: '{new_name}' 也已存在，请选择其他名称。")


            # 根据扩展名保存文件
            if ext == '.npy':
                # 使用np.savez但保持.npy扩展名
                np.savez(full_path.with_suffix('.npz'), X=X, y=y)
                # 重命名文件以保持用户指定的扩展名
                full_path.with_suffix('.npz').rename(full_path)
            elif ext in ('.h5', '.hdf5'):
                with h5py.File(full_path, 'w') as f:
                    f.create_dataset('X', data=X)
                    f.create_dataset('y', data=y)
            elif ext == '.pkl':
                with open(full_path, 'wb') as f:
                    pickle.dump({'X': X, 'y': y}, f)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            

def main():
    parser = argparse.ArgumentParser(description="Website Fingerprinting Dataset Processor")
    parser.add_argument('-i', '--input', required=True, help="Input path (file or directory)")
    parser.add_argument('-o', '--output', default='dataset.npy', help="Output path")
    parser.add_argument('-l', '--label', choices=['numeric', 'single_onehot', 'multi_onehot'], 
                      default='numeric', help="Label format")
    parser.add_argument('-p', '--proportion', nargs='+', type=float, 
                      help="Dataset split proportions")
    parser.add_argument('-O', '--openworld', nargs=2, type=int,
                      help="Openworld scenario params [train_unmon, test_unmon]")
    parser.add_argument('-w', '--openworldPath', help="Openworld path (file or directory)")
    parser.add_argument('-n', '--num', type=int, 
                        help="number of monitor set instance")
    parser.add_argument('-c', '--clsnum', type=int, 
                        help="number of categories")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # 初始化数据集
    dataset = DatasetMaker(args.input)
    
    # 选取类别数
    if args.clsnum:
        dataset.select_classes(args.clsnum)

    # 选取样本数
    if args.num:
        if dataset.original_label_type == 'multi_onehot':
            raise ValueError("multi-onehot format data cannot select a fixed number of samples for each category")
        try:
            if dataset.original_label_type == 'single_onehot':
                dataset.convert_labels('numeric')
                dataset.select_samples(args.num)
                dataset.convert_labels('single_onehot')
            else:
                dataset.select_samples(args.num)
        except ValueError as e:
            print(f"select instances error: {str(e)}")
            return

    # 加入开放世界样本
    if args.openworldPath:
        OWdata = DatasetMaker(args.openworldPath)
        openclass = max(dataset.labels)+1
        OWdata.labels[:] = openclass

        if dataset.features.shape[2] != OWdata.features.shape[2]:
            max_len = max(dataset.features.shape[2], OWdata.features.shape[2])
            dataset_padded = np.pad(dataset.features, 
                                ((0, 0), (0, 0), (0, max_len - dataset.features.shape[2])))
            OWdata_padded = np.pad(OWdata.features, 
                                ((0, 0), (0, 0), (0, max_len - OWdata.features.shape[2])))
            dataset.features = np.concatenate((dataset_padded, OWdata_padded), axis=0)
        else:
            dataset.features = np.concatenate((dataset.features, OWdata.features), axis=0)
        
        dataset.labels = np.concatenate((dataset.labels, OWdata.labels), axis=0)

    # 处理标签格式
    try:
        dataset.convert_labels(args.label)
    except ValueError as e:
        print(f"Label conversion error: {str(e)}")
        return

    if args.openworld:
        try:
            dataset.handle_openworld(tuple(args.openworld))
        except ValueError as e:
            print(f"Openworld error: {str(e)}")
            return

    # 划分数据集逻辑
    if args.proportion or dataset.openworld_processed:
        # 开放世界模式可能使用默认比例
        proportions = [float(p) for p in args.proportion] if args.proportion else None
        print(f"comes to proportion, proportion is {proportions}")
        datasets = dataset.split_dataset(proportions, args.seed)
    else:
        datasets = [(dataset.features, dataset.labels)]

    # 输出结果
    for i, (X, y) in enumerate(datasets):
        print(f"Split {i+1}: Features shape {X.shape}, Labels shape {y.shape}")

    # 保存结果
    try:
        dataset.save(args.output, datasets)
        print(f"Dataset saved successfully to {args.output}")
    except Exception as e:
        print(f"Save failed: {str(e)}")

if __name__ == '__main__':
    main()