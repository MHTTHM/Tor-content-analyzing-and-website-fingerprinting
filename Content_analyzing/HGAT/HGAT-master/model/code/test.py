# 请将这个路径替换成您的 embedding 文件的实际路径
file_path = r"embeddings/test_v3.emb" 
num_lines_to_read = 5 # 我们只需要读取前 5 行作为样本就足够了

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        print(f"--- 文件 '{file_path}' 的前 {num_lines_to_read} 行内容如下 ---")
        for i in range(num_lines_to_read):
            line = f.readline()
            if not line:
                break # 如果文件行数不足5行，则提前结束
            
            # 打印原始行内容，这样我们可以看到所有的字符，包括空格和逗号
            print(repr(line)) 
            
    print("----------------------------------------------------")
    print("请将上面以 '---' 开始和结束的全部输出内容复制给我。")
    print("特别是 repr() 函数的输出，它会显示所有隐藏字符，对调试非常有帮助。")

except FileNotFoundError:
    print(f"错误：找不到文件 '{file_path}'。请再次确认路径是否正确。")
except Exception as e:
    print(f"读取文件时发生了未知错误: {e}")