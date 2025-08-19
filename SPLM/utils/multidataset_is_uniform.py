import os, sys
import random
from collections import defaultdict
import numpy as np

# 获取 TAM_feature.py 所在的路径并添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录（utils）
parent_dir = os.path.dirname(current_dir)  # 项目根目录
extractor_dir = os.path.join(parent_dir, "Extractor")
sys.path.append(extractor_dir)

from DatasetMaker import DatasetMaker


def generate_mock_data(num_samples=4000, max_val=5000, min_len=100, max_len=1000, possible_labels=[0, 1, 2, 3]):
    """
    生成模拟数据。
    每个样本的形状为 [[list1, list2], [label1, label2]]
    其中 list1 = [start, end] 且 end > start.
    """
    # 使用 dtype=object 来存储 Python 列表和数字
    data = np.empty((num_samples, 2, 2), dtype=object)

    for i in range(num_samples):
        # 数据列表1
        start1 = random.randint(0, max_val - min_len)
        length1 = random.randint(min_len, max_len)
        data[i, 0, 0] = [start1, start1 + length1]
        data[i, 1, 0] = random.choice(possible_labels)  # 数据列表1的标签

        # 数据列表2
        start2 = random.randint(0, max_val - min_len)
        length2 = random.randint(min_len, max_len)
        data[i, 0, 1] = [start2, start2 + length2]
        data[i, 1, 1] = random.choice(possible_labels)  # 数据列表2的标签

    return data

def analyze_data_distribution(dataset):
    """
    分析数据集，计算每个标签对应差值的均值及分布情况。
    """
    differences_by_label = defaultdict(list)

    # 1. 从4000个样本中筛选所有标签的值；
    # 2. 对每个标签的值，提取出每个数据list；
    # 3. 计算每个list的差值（第二个数总是大于第一个数）
    for sample_index in range(dataset.shape[0]):
        sample = dataset[sample_index]

        # 第一个数据列表和标签
        data_list1 = sample[0][0]
        label1 = sample[1][0]
        if data_list1 is not None and len(data_list1) == 2:
            difference1 = data_list1[1] - data_list1[0]
            if difference1 < 0:
                print(f"警告: 样本 {sample_index}, 数据列表1 ({data_list1}) 的差值为负。跳过此数据点。")
            else:
                differences_by_label[label1].append(difference1)
        else:
            print(f"警告: 样本 {sample_index}, 数据列表1 ({data_list1}) 格式不正确。跳过。")


        # 第二个数据列表和标签
        data_list2 = sample[0][1]
        label2 = sample[1][1]
        if data_list2 is not None and len(data_list2) == 2:
            difference2 = data_list2[1] - data_list2[0]
            if difference2 < 0:
                print(f"警告: 样本 {sample_index}, 数据列表2 ({data_list2}) 的差值为负。跳过此数据点。")
            else:
                differences_by_label[label2].append(difference2)
        else:
            print(f"警告: 样本 {sample_index}, 数据列表2 ({data_list2}) 格式不正确。跳过。")


    # 4. 评估每个标签对应的差值的均值、值分布是否均匀
    results = {}
    sorted_labels = sorted(list(differences_by_label.keys()))

    for label in sorted_labels:
        diffs = np.array(differences_by_label[label])
        if len(diffs) == 0:
            print(f"\n标签 {label}:")
            print("  没有找到有效的数据点。")
            results[label] = {
                "count": 0,
                "mean_difference": None,
                "std_difference": None,
                "min_difference": None,
                "q1_difference": None,
                "median_difference": None,
                "q3_difference": None,
                "max_difference": None,
                "unique_value_counts": {},
                "uniformity_assessment": "没有数据无法评估。"
            }
            continue

        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        min_diff = np.min(diffs)
        q1_diff = np.percentile(diffs, 25)
        median_diff = np.median(diffs)
        q3_diff = np.percentile(diffs, 75)
        max_diff = np.max(diffs)

        unique_values, counts = np.unique(diffs, return_counts=True)
        value_counts_dict = dict(zip(unique_values, counts))

        # 对分布均匀性的初步评估
        uniformity_desc = []
        if std_diff == 0 and len(unique_values) == 1:
            uniformity_desc.append(f"所有差值均为 {min_diff}，分布高度集中。")
        else:
            uniformity_desc.append(f"差值范围从 {min_diff} 到 {max_diff}。")
            if std_diff < (max_diff - min_diff) / 10 : # 这是一个启发式规则
                 uniformity_desc.append("标准差相对较小，表明数值可能存在一定程度的聚集。")
            else:
                 uniformity_desc.append("标准差相对较大，表明数值分布较广。")

            # 检查四分位距
            iqr = q3_diff - q1_diff
            range_val = max_diff - min_diff
            if range_val > 0 : # 避免除以零
                if iqr / range_val < 0.3 and len(unique_values) > 3: # 另一个启发式规则
                    uniformity_desc.append("大部分数据集中在较小的范围内 (基于IQR)。")

            num_unique = len(unique_values)
            if num_unique <= 10: # 如果唯一值较少，显示它们的计数
                uniformity_desc.append(f"共有 {num_unique} 个不同的差值。")
                # 检查计数是否大致相等
                if num_unique > 1 and np.std(counts) < np.mean(counts) * 0.5 : # 启发式：如果计数的标准差小于均值的一半
                    uniformity_desc.append("不同差值的出现频率较为接近，具有一定的离散均匀性。")
                else:
                    uniformity_desc.append("不同差值的出现频率差异较大。")
            else:
                uniformity_desc.append(f"共有 {num_unique} 个不同的差值，数量较多，建议使用直方图进行可视化评估。")


        results[label] = {
            "count": len(diffs),
            "mean_difference": mean_diff,
            "std_difference": std_diff,
            "min_difference": min_diff,
            "q1_difference": q1_diff,
            "median_difference": median_diff,
            "q3_difference": q3_diff,
            "max_difference": max_diff,
            "unique_value_counts": value_counts_dict,
            "uniformity_assessment": " ".join(uniformity_desc)
        }

        print(f"\n标签 {label}:")
        print(f"  样本数量: {len(diffs)}")
        print(f"  差值均值: {mean_diff:.2f}")
        print(f"  差值标准差: {std_diff:.2f}")
        print(f"  差值范围: [{min_diff}, {max_diff}]")
        print(f"  差值四分位数: Q1={q1_diff:.2f}, 中位数={median_diff:.2f}, Q3={q3_diff:.2f}")
        print(f"  唯一差值及其计数 (前10个，如果过多):")
        sorted_value_counts = sorted(value_counts_dict.items(), key=lambda item: item[1], reverse=True)
        # for val, count in sorted_value_counts[:10]:
        #     print(f"    差值 {val}: {count} 次")
        if len(sorted_value_counts) > 10:
            print(f"    ... (还有 {len(sorted_value_counts) - 10} 种不同的差值)")
        print(f"  分布均匀性评估: {results[label]['uniformity_assessment']}")

    return results

# --- 主程序 ---
if __name__ == "__main__":
    temp_folder = 'D:/博士研究/论文写作/202410类别分类论文/Momentum_WF/WF_Code/multi-tab/Temp_multitab'
    
    kwargs = {'threshold': 1000, 'zscore': 0.16, 'multi-tab': True}
    testdata = DatasetMaker(temp_folder, **kwargs)
    test_features, test_labels = testdata.features, testdata.labels
    # test_features = test_features[50:51]
    # test_labels = test_labels[50:51]
    print(test_features.shape, test_labels.shape)


    # 2. 分析数据分布
    analysis_results = analyze_data_distribution(test_labels)

    # analysis_results 字典中包含了所有标签的详细分析结果
    # 您可以进一步处理或保存这个字典
    # print("\n完整分析结果字典:")
    # print(analysis_results)
