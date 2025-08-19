# -----------------
# 最终修改版 T-SNE 可视化脚本 (高维插值 + 自动颜色 + 自定义图例)
# -----------------

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import ast

# --- 修改点: 添加 matplotlib 中文支持 ---
# 解决中文显示问题，'SimHei' 是一个常用的支持中文的黑体字
# 如果您的系统中没有 SimHei 字体，可以换成 'Microsoft YaHei' (微软雅黑) 或其他任何支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False


def generate_synthetic_points(embeddings, labels, points_per_category=50):
    """
    在原始高维空间中为每个类别生成合成数据点。

    Args:
        embeddings (np.ndarray): 原始的嵌入向量数组。
        labels (list): 对应的标签列表。
        points_per_category (int): 希望为每个类别生成的新点的数量。

    Returns:
        tuple: 包含 (增强后的嵌入, 增强后的标签) 的元组。
    """
    print(f"开始生成人工数据点，每个类别目标生成 {points_per_category} 个...")
    unique_labels = list(set(labels))
    
    augmented_embeddings = list(embeddings)
    augmented_labels = list(labels)
    
    for label in unique_labels:
        # 找到属于当前类别的所有点的索引
        indices = [i for i, l in enumerate(labels) if l == label]
        
        # 如果该类别只有一个点，无法插值，跳过
        if len(indices) < 2:
            print(f"警告：类别 '{label}' 的样本数少于2，无法为其生成插值点。")
            continue
            
        category_embeddings = embeddings[indices]
        
        generated_count = 0
        for _ in range(points_per_category):
            # 随机选择两个不同的点进行插值
            idx1, idx2 = np.random.choice(len(category_embeddings), 2, replace=False)
            point1 = category_embeddings[idx1]
            point2 = category_embeddings[idx2]
            
            # 创建一个随机权重 (0到1之间)
            alpha = np.random.rand()
            
            # 线性插值
            new_point = alpha * point1 + (1 - alpha) * point2
            
            augmented_embeddings.append(new_point)
            augmented_labels.append(label)
            generated_count += 1
        
        print(f"为类别 '{label}' 生成了 {generated_count} 个新数据点。")
        
    return np.array(augmented_embeddings), augmented_labels


def visualize_embeddings_by_category_with_interpolation():
    """
    一个完整的函数，加载数据，生成人工插值点，然后进行T-SNE可视化。
    此版本使用自动颜色、中文图例，并保存高分辨率图像。
    """
    try:
        # --- 步骤 1: 定义文件路径 ---
        embeddingFilePath = r"embeddings/test_v2.emb"
        labelsFilePath = "data/test_v2/test_v2.txt"

        # --- 步骤 2: 加载原始数据 ---
        print("开始加载原始嵌入文件...")
        embedding_vectors = []
        with open(embeddingFilePath, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                clean_line = line.strip()
                if clean_line:
                    try:
                        parsed_list = ast.literal_eval(clean_line)
                        vector = [float(item) for item in parsed_list]
                        embedding_vectors.append(vector)
                    except (ValueError, SyntaxError) as parse_error:
                        print(f"警告：跳过无法解析的嵌入文件第 {i+1} 行。错误: {parse_error}")
                        continue
        labels = []
        with open(labelsFilePath, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                clean_line = line.strip()
                if not clean_line: continue
                parts = clean_line.split('\t')
                if len(parts) >= 2:
                    labels.append(parts[1])
                else:
                    print(f"警告：跳过格式不正确的标签文件第 {i+1} 行。")
        
        if len(embedding_vectors) != len(labels):
            raise ValueError(f"嵌入数量 ({len(embedding_vectors)}) 与标签数量 ({len(labels)}) 不匹配。")
        if not embedding_vectors:
            raise ValueError("未找到有效的嵌入数据。")

        embedding_vectors_np = np.array(embedding_vectors, dtype=np.float64)
        print(f"原始数据加载成功！共加载了 {len(labels)} 个向量。")
        
        unique_labels_from_data = sorted(list(set(labels)))
        print("--------------------------------------------------")
        print(f"重要提示：从数据文件中找到的唯一标签为: {unique_labels_from_data}")
        print("请根据此列表，检查并修改下方的 'label_order' 变量，确保其内容和顺序正确。")
        print("--------------------------------------------------")

        # --- 步骤 3: 生成人工数据点 ---
        points_to_generate = 330
        augmented_embeddings_np, augmented_labels = generate_synthetic_points(
            embedding_vectors_np, labels, points_per_category=points_to_generate
        )
        print(f"数据增强完成。总向量数从 {len(labels)} 增加到 {len(augmented_labels)}。")

        # --- 步骤 4: 使用增强后的数据进行 t-SNE 降维 ---
        print("正在对增强后的数据进行 t-SNE 降维...")
        n_samples = augmented_embeddings_np.shape[0]
        perplexity_value = min(30, n_samples - 1)
        
        tsne_model = TSNE(n_components=2, init='pca', learning_rate='auto', 
                          n_iter=1000, random_state=42, perplexity=perplexity_value)
        compress_embedding = tsne_model.fit_transform(augmented_embeddings_np)
        print("t-SNE 降维完成。")

        # --- 步骤 5: 绘图 ---
        print("正在生成可视化图表...")
        
        # --- 修改点: 配置中文图例和标签顺序 ---
        # 1. 定义您希望在图例中看到的中文标签 (按期望顺序)
        chinese_labels = ['金融', '软件', '毒品', '论坛', '黑客', '市场', '色情', '暴力']

        # 2. 定义与中文标签顺序对应的原始标签名 (从您的 .txt 文件中读取的标签)
        #    !!! 这是您唯一需要手动修改的地方 !!!
        #    请将下面的 'Finance', 'Software' 等替换为您数据中的真实标签名。
        #    这里的顺序必须与上面的 `chinese_labels` 严格对应。
        label_order = [
            'Cards',   # 对应 '金融'
            'Directory',  # 对应 '软件'
            'Drugs',      # 对应 '毒品'
            'Forum',     # 对应 '论坛'
            'Hacking',    # 对应 '黑客'
            'Marketplace',    # 对应 '市场'
            'Porngraphy',      # 对应 '色情'
            'Violence'   # 对应 '暴力'
        ]

        # --- 恢复为自动生成颜色 ---
        # 3. 使用 Matplotlib 的 'tab10' 色彩映射表自动生成颜色
        #    'tab10' 是一个视觉上区分度很高的色彩方案，适合分类数据
        n_classes = len(label_order)
        cmap = plt.get_cmap('tab10', n_classes)
        colors = [cmap(i) for i in range(n_classes)]

        # 检查配置是否匹配
        if len(label_order) != len(chinese_labels):
            raise ValueError("错误: 'label_order' 和 'chinese_labels' 的项目数量不匹配!")

        # 创建从原始标签到中文和颜色的映射
        label_to_chinese = dict(zip(label_order, chinese_labels))
        label_to_color = dict(zip(label_order, colors))
        
        fig, ax = plt.subplots(figsize=(18, 15))

        # 按预设的顺序绘制每个类别，以确保图例顺序和颜色正确
        for label in label_order:
            if label not in unique_labels_from_data:
                print(f"警告: 预设标签 '{label}' 未在数据中找到，将跳过绘制。")
                continue

            indices = [idx for idx, l in enumerate(augmented_labels) if l == label]
            if not indices:
                continue

            x = compress_embedding[indices, 0]
            y = compress_embedding[indices, 1]
            
            # 使用自动生成的颜色和自定义的中文标签进行绘制
            ax.scatter(x, y, color=label_to_color[label], label=label_to_chinese[label], s=30, alpha=0.8)

        ax.grid(True)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(title="类别", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='large', title_fontsize='x-large', ncol=2)

        plt.tight_layout(rect=[0, 0, 0.97, 1]) 
        
        # --- 保留功能: 保存高分辨率图像并优化空白边距 ---
        output_filename = 'T-SNE_large.png'
        try:
            plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
            print(f"高分辨率图像已成功保存到: {output_filename}")
        except Exception as save_err:
            print(f"保存文件时出错: {save_err}")

        plt.show()
        print("图表已显示。")

    except FileNotFoundError as e:
        print(f"错误：找不到文件。请检查路径是否正确。详细信息: {e}")
    except Exception as e:
        print(f"脚本执行过程中发生未知错误: {e}")

# --- 运行主函数 ---
visualize_embeddings_by_category_with_interpolation()



