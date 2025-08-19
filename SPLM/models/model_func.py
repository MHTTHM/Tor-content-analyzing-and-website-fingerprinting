import torch, os
from datetime import datetime

def save_model(model, save_dir='./saved_models', num_classes=None):
    """
    保存模型（改进版）
    参数：
        model_config: 模型的配置字典（包含类别数等信息）
    """
    os.makedirs(save_dir, exist_ok=True)
    time_str = datetime.now().strftime("%m%d%H%M")
    
    # 保存完整模型信息（包含配置）
    model_info = {
        'state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'numclasses': num_classes  # 包含类别数等配置
    }
    save_path = os.path.join(save_dir, f'RF_model_full_{time_str}.pth')
    torch.save(model_info, save_path)
    
    print(f"模型保存成功：{save_path}")
    return save_path

def load_model(model_path, model_class=None):
    """
    加载模型（适配新版保存方式）
    参数：
        model_path: 模型文件路径
        model_class: 模型类（如 `getRF`）
    返回：
        加载后的模型（自动设置 eval 模式）
    """
    # 加载模型文件
    data = torch.load(model_path, map_location='cuda:0')
    
    if not isinstance(data, dict) or 'state_dict' not in data:
        raise ValueError("模型文件格式不正确，应包含 state_dict 和 numclasses")
    
    # 检查是否包含 numclasses
    if 'numclasses' not in data:
        raise ValueError("模型文件缺少 numclasses，请检查保存方式")
    
    # 获取模型配置（从保存的数据中提取 num_classes）
    model_config = {'num_classes': data['numclasses']}
    
    # 实例化模型
    if model_class is None:
        raise ValueError("需要提供 model_class（如 getRF）")
    
    model = model_class(data['numclasses'])  # 自动传入 num_classes
    model.load_state_dict(data['state_dict'])
    
    print(f"成功加载模型: {model_path} (num_classes={data['numclasses']})")
    return model.eval()  # 设置为评估模式