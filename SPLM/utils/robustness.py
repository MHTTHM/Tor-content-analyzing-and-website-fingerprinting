import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from tqdm import tqdm
import math,sys, os
import statistics
from typing import List, Tuple, Dict, Callable, Any

current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录（utils）
parent_dir = os.path.dirname(current_dir)  # 项目根目录
extractor_dir = os.path.join(parent_dir, "Extractor")
sys.path.append(extractor_dir)

# =============================================================================
# 步骤 1: 导入您提供的特征提取代码
# 请确保这些函数位于 'splm_feature.py' 和 'tam_feature.py' 文件中，
# 或者直接将它们粘贴到此处。
# =============================================================================

# 假设这些函数来自 splm_feature.py
from Momentum_feature import Momentum_feature, remove_flat_segments 

# 假设这个函数来自 tam_feature.py
from TAM_feature import packets_perslot

# =============================================================================
# 步骤 2: 实现核心度量函数 - MMD
# =============================================================================

def calculate_mmd_rbf(X: np.ndarray, Y: np.ndarray, gammas: List[float] = [1.0]) -> float:
    """
    计算两个样本集 X 和 Y 之间的最大均值差异 (MMD)，使用高斯核(RBF)。
    可以对多个gamma值（核宽度）的结果取平均。

    Args:
        X (np.ndarray): 第一个样本集, shape (n_samples, n_features).
        Y (np.ndarray): 第二个样本集, shape (m_samples, n_features).
        gammas (List[float]): 用于RBF核的gamma参数列表。

    Returns:
        float: MMD^2 的值。
    """
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return 0.0
        
    total_mmd = 0.0
    for gamma in gammas:
        K_XX = rbf_kernel(X, X, gamma=gamma)
        K_YY = rbf_kernel(Y, Y, gamma=gamma)
        K_XY = rbf_kernel(X, Y, gamma=gamma)

        # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
        mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        total_mmd += mmd2
        
    return total_mmd / len(gammas)

# =============================================================================
# 步骤 3: 实现鲁棒性评估的主流程
# =============================================================================

def evaluate_robustness(
    dataset: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
    feature_extractor: Callable,
    defense_func: Callable,
    defense_params: Dict[str, Any],
    mmd_gammas: List[float]
) -> float:
    """
    评估一个特征提取器在特定防御下的鲁棒性。

    Args:
        dataset: 数据集，格式为 {class_id: [(times, sizes), ...]}
        feature_extractor: 特征提取函数，输入 (times, sizes)，返回特征向量。
        defense_func: 防御应用函数，输入 (times, sizes, **params)，返回扰动后的 (times', sizes')。
        defense_params: 防御函数的参数。
        mmd_gammas: MMD计算用的gamma值列表。

    Returns:
        float: D_intra, 即所有类别上MMD的平均值。
    """
    all_mmds = []
    
    print(f"Evaluating for defense params: {defense_params}")
    for class_id, traces in tqdm(dataset.items(), desc="Processing classes"):
        # 1. 提取原始特征
        original_features = []
        for times, sizes in traces:
            # 特征提取函数可能返回非numpy数组，需要转换和填充
            feat = np.array(feature_extractor(times, sizes)).flatten()
            original_features.append(feat)

        # 2. 应用防御并提取扰动后的特征
        defended_features = []
        for times, sizes in traces:
            defended_times, defended_sizes = defense_func(times, sizes, **defense_params)
            feat = np.array(feature_extractor(defended_times, defended_sizes)).flatten()
            defended_features.append(feat)
            
        # 统一特征向量长度 (非常重要！)
        # MMD要求输入的向量维度一致。由于SPLM长度可变，需要填充到最大长度。
        max_len = max(max(len(f) for f in original_features), max(len(f) for f in defended_features))
        
        original_features_padded = np.array([np.pad(f, (0, max_len - len(f))) for f in original_features])
        defended_features_padded = np.array([np.pad(f, (0, max_len - len(f))) for f in defended_features])

        # 3. 计算该类别的MMD
        if original_features_padded.shape[0] > 0 and defended_features_padded.shape[0] > 0:
            mmd = calculate_mmd_rbf(original_features_padded, defended_features_padded, gammas=mmd_gammas)
            all_mmds.append(mmd)

    # 4. 计算所有类别的平均MMD (D_intra)
    d_intra = np.mean(all_mmds) if all_mmds else 0.0
    return d_intra

# =============================================================================
# 步骤 4: 定义特征提取器和防御函数的封装
# =============================================================================

# --- 特征提取器封装 ---
def extract_splm_feature(times: np.ndarray, sizes: np.ndarray) -> List[List[float]]:
    # SPLM流程：先移除平坦段，再提取动量特征
    # clean_times, clean_sizes = remove_flat_segments(list(times), list(sizes))
    # if not clean_times:
    #     return [[], []]
    # return Momentum_feature(np.array(clean_times), np.array(clean_sizes))
    return Momentum_feature(np.array(times), np.array(sizes))

def extract_tam_feature(times: np.ndarray, sizes: np.ndarray) -> List[List[int]]:
    return packets_perslot(list(times), list(sizes))

# --- 防御函数占位符 ---
# 您需要用真实的防御算法实现替换这些虚拟函数

def apply_dummy_padding_defense(times: np.ndarray, sizes: np.ndarray, bandwidth_overhead: float) -> Tuple[np.ndarray, np.ndarray]:
    """虚拟的填充防御。您需要替换为真实实现。"""
    num_packets_to_add = int(len(sizes) * bandwidth_overhead)
    new_sizes = list(sizes)
    new_times = list(times)
    
    for _ in range(num_packets_to_add):
        # 随机选择方向和插入位置
        direction = np.random.choice([-1, 1])
        size = direction * 1500 # 假设填充包大小为MTU
        insert_idx = np.random.randint(0, len(new_times) + 1)
        
        # 简单地在时间上插入
        time_to_insert = new_times[insert_idx-1] + 1e-4 if insert_idx > 0 else 0
        
        new_sizes.insert(insert_idx, size)
        new_times.insert(insert_idx, time_to_insert)
        
    return np.array(new_times), np.array(new_sizes)

def apply_dummy_delay_defense(times: np.ndarray, sizes: np.ndarray, time_overhead: float) -> Tuple[np.ndarray, np.ndarray]:
    """虚拟的延迟防御。您需要替换为真实实现。"""
    total_time = times[-1] - times[0] if len(times) > 1 else 0
    # 延迟从正态分布中采样，其标准差与时间开销相关
    delay_std_dev = total_time * time_overhead 
    delays = np.random.normal(0, delay_std_dev, size=len(times))
    delays[delays < 0] = 0 # 延迟不能为负
    
    new_times = np.sort(times + delays)
    return new_times, sizes

# =============================================================================
# 步骤 5: 主执行块 - 运行实验并可视化
# =============================================================================

if __name__ == '__main__':
    # --- 1. 准备数据 (使用模拟数据，请替换为真实数据加载) ---
    print("Loading/Generating dataset...")
    NUM_CLASSES = 10  # 网站数量
    TRACES_PER_CLASS = 50 # 每个网站的trace数量
    PACKETS_PER_TRACE = 300 # 每个trace的包数量

    mock_dataset = {}
    for i in range(NUM_CLASSES):
        traces = []
        for _ in range(TRACES_PER_CLASS):
            # 模拟一个简单的网页加载过程
            times = np.sort(np.random.rand(PACKETS_PER_TRACE) * 10) # 10秒加载时间
            sizes = np.random.choice([-500, -300, 1500, 1000], size=PACKETS_PER_TRACE, p=[0.2, 0.3, 0.4, 0.1])
            # 添加一些网站特有的模式
            sizes[i*10:(i+1)*10] = 1500 
            traces.append((times, sizes))
        mock_dataset[i] = traces
    print("Dataset ready.")

    # --- 2. 定义实验参数 ---
    feature_extractors = {
        "SPLM": extract_splm_feature,
        "TAM": extract_tam_feature,
    }

    # 实验A: 固定时间开销，改变带宽开销
    bandwidth_overheads = np.linspace(0.1, 1.0, 5) # 10% to 100%
    fixed_time_overhead = 0.1 # 10%

    results = {name: [] for name in feature_extractors}
    
    # 根据论文，MMD使用5个高斯核
    # 常见做法是使用基于样本间距离中位数的一系列gamma
    # 这里为简化，我们使用固定的gamma列表
    mmd_gammas = [0.1, 1.0, 10.0]

    # --- 3. 运行实验A ---
    for name, extractor in feature_extractors.items():
        print(f"\n--- Evaluating Feature: {name} ---")
        for bw_overhead in bandwidth_overheads:
            defense_params = {'bandwidth_overhead': bw_overhead}
            d_intra = evaluate_robustness(
                dataset=mock_dataset,
                feature_extractor=extractor,
                defense_func=apply_dummy_padding_defense,
                defense_params=defense_params,
                mmd_gammas=mmd_gammas
            )
            results[name].append(d_intra)
            print(f"Result for {name} @ {bw_overhead*100:.0f}% BW overhead: D_intra = {d_intra:.6f}")

    # --- 4. 可视化结果 ---
    plt.figure(figsize=(10, 6))
    for name, d_intra_values in results.items():
        plt.plot(bandwidth_overheads * 100, d_intra_values, marker='o', linestyle='-', label=name)

    plt.title('Feature Robustness against Packet Padding')
    plt.xlabel('Bandwidth Overhead (%)')
    plt.ylabel('Intra-Class Distance (D_intra via MMD)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 您可以仿照上面的流程，设计并运行实验B（固定带宽，改变时间）