import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerEncoderBlock(nn.Module):
    def __init__(self, head_size, num_heads, ff_dim, dropout=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(head_size)
        self.attn = nn.MultiheadAttention(head_size, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(head_size)
        self.conv1 = nn.Conv1d(head_size, ff_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(ff_dim, head_size, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention part
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = self.dropout1(x)
        x = x + residual
        
        # Feed-forward part
        residual = x
        x = self.norm2(x)
        x = x.permute(0, 2, 1)  # (batch, length, channels) -> (batch, channels, length)
        x = F.relu(self.conv1(x))
        x = self.dropout2(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)  # (batch, channels, length) -> (batch, length, channels)
        x = x + residual
        
        return x

class MixedModel(nn.Module):
    def __init__(
        self,
        input_channels=1,
        emb_size=64,
        head_size=256,  # 关键修改：与CNN输出通道一致
        num_heads=4,
        ff_dim=256,
        num_transformer_blocks=2,
        mlp_units=[128, 64],
        mlp_dropout=0.1,
        dropout=0.1,
        model_name=''
    ):
        super().__init__()
        self.model_name = model_name
        
        # CNN blocks
        self.block1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=8, stride=1, padding='same'),
            nn.ELU(alpha=1.0),
            nn.Conv1d(32, 32, kernel_size=8, stride=1, padding='same'),
            nn.ELU(alpha=1.0),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=1),  # 调整池化参数
            nn.Dropout(0.3)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=1),  # 调整池化参数
            nn.Dropout(0.3)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=1),  # 调整池化参数
            nn.Dropout(0.3)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=8, stride=1, padding='same'),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)  # 使用自适应池化确保输出长度
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(head_size, num_heads, ff_dim, dropout)
            for _ in range(num_transformer_blocks)
        ])
        
        # 调整后的池化层
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(4)  # 自适应池化避免长度问题
            for _ in range(num_transformer_blocks)
        ])
        
        # MLP head
        mlp_layers = []
        in_features = 256 * 4  # 根据实际池化后的长度调整
        for dim in mlp_units:
            mlp_layers.append(nn.Linear(in_features, dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(mlp_dropout))
            in_features = dim
        mlp_layers.append(nn.Linear(in_features, emb_size))
        self.mlp = nn.Sequential(*mlp_layers)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # CNN blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Transformer处理前调整维度
        x = x.permute(0, 2, 1)
        
        # Transformer blocks
        for block, pool in zip(self.transformer_blocks, self.pools):
            x = block(x)
            x = x.permute(0, 2, 1)
            x = pool(x)
            x = x.permute(0, 2, 1)
        
        # 展平并输入MLP
        x = x.reshape(x.size(0), -1)
        x = self.mlp(x)
        return x

def verify_model():
    batch_size = 4
    seq_length = 128
    input_channels = 1
    
    model = MixedModel(
        input_channels=input_channels,
        emb_size=64,
        head_size=256,  # 确保与CNN输出通道一致
        num_heads=4,
        ff_dim=256,
        num_transformer_blocks=2,
        mlp_units=[128, 64],
        mlp_dropout=0.1,
        dropout=0.1
    )
    
    print(model)
    
    x = torch.randn(batch_size, input_channels, seq_length)
    
    try:
        output = model(x)
        print("\nModel verification successful!")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print("\nModel verification failed with error:")
        print(e)

verify_model()