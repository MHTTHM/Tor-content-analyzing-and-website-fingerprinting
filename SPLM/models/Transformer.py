import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from Extractor.DatasetMaker import DatasetMaker


# 自定义数据集类（需要根据实际数据路径修改）
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data    # 形状应为(N, 2, 300)
        self.labels = labels# 形状应为(N,)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])[0]

# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Transformer分类模型
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=300, d_model=128, nhead=4, 
                 num_layers=3, dim_feedforward=512, num_classes=182):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=0.1, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model*2, num_classes)  # 拼接两个时间步的输出
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 输入形状: (batch_size, seq_len=2, input_dim=300)
        x = x.permute(1, 0, 2)  # 转换为(seq_len, batch_size, input_dim)
        x = self.embedding(x)   # 转换为(seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
        x = self.dropout(x)
        # 拼接两个时间步的特征
        x = x.reshape(x.size(0), -1)  # (batch_size, seq_len*d_model)
        return self.fc(x)

# 训练参数
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=='__main__':
    # 数据准备（需要替换为真实数据）
    # 假设X_train, y_train, X_val, y_val已经准备好
    input_dir = r'D:\2025HS_dataset\HS_longstream'
    args = {'threshold': None, 'zscore': 0.16, 'vector': 300}
    #print("params: ", args)
    dataset = DatasetMaker(input_dir, feature_func="momentum", **args)
    proportions = [0.8, 0.2]
    datasets = dataset.split_dataset(proportions, 42)
    trainfile, testfile = datasets[0], datasets[1]
    X_train, y_train = trainfile
    X_train = X_train[:, :, :300]
    X_val, y_val = testfile
    X_val = X_val[:, :, :300]

    train_dataset = MyDataset(X_train, y_train)
    val_dataset = MyDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 初始化模型
    model = TransformerClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 训练循环
    best_val_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{EPOCHS} | '
            f'Train Loss: {train_loss/len(train_loader):.4f} | '
            f'Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f'Best Validation Accuracy: {best_val_acc:.2f}%')