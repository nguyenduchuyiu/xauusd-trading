import torch
import torch.nn as nn
import math

class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv1d(in_channels, 32, kernel_size=1, padding='same')
        self.branch3 = nn.Conv1d(in_channels, 32, kernel_size=3, padding='same')
        self.branch5 = nn.Conv1d(in_channels, 32, kernel_size=5, padding='same')
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, 32, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch3(x), self.branch5(x), self.branch_pool(x)], dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CrossAttentionFusion(nn.Module):
    def __init__(self, cnn_dim, transformer_dim):
        super().__init__()
        self.query = nn.Linear(cnn_dim, transformer_dim)
        self.key = nn.Linear(transformer_dim, transformer_dim)
        self.value = nn.Linear(transformer_dim, transformer_dim)
        
    def forward(self, cnn_features, transformer_features):
        # cnn_features: [batch, cnn_dim]
        # transformer_features: [batch, seq_len, transformer_dim]
        Q = self.query(cnn_features).unsqueeze(1)  # [batch, 1, transformer_dim]
        K = self.key(transformer_features)         # [batch, seq_len, transformer_dim]
        V = self.value(transformer_features)       # [batch, seq_len, transformer_dim]
        
        # Tính attention weights
        attn_scores = (Q @ K.transpose(-2, -1)) / (K.size(-1) ** 0.5)  # [batch, 1, seq_len]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Weighted sum của values
        return torch.bmm(attn_weights, V).squeeze(1)  # [batch, transformer_dim]

class HighwayNetwork(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, fused, transformer):
        g = self.gate(fused)
        return g * fused + (1 - g) * transformer

class EnhancedHybridModel(nn.Module):
    def __init__(self, num_features, num_classes=3, d_model=512, nhead=16, dim_feedforward=1024, num_layers=6):
        super().__init__()
        # 1. InceptionTime Branch
        self.inception = nn.Sequential(
            InceptionModule(num_features),
            nn.ReLU(),
            nn.MaxPool1d(2),
            InceptionModule(128),
            nn.ReLU()
        )
        
        # 2. Transformer Branch
        self.transformer_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Fusion
        self.cross_attention = CrossAttentionFusion(128, d_model)
        self.highway = HighwayNetwork(d_model)
        
        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 1. Inception Path
        cnn_features = self.inception(x.permute(0, 2, 1))  # [batch, channels, seq_len//2]
        cnn_features = cnn_features.mean(dim=-1)          # [batch, channels=128]
        
        # 2. Transformer Path
        x_proj = self.pos_encoder(self.transformer_proj(x))  # [batch, seq_len, d_model]
        transformer_features = self.transformer(x_proj)      # [batch, seq_len, d_model]
        
        # 3. Fusion
        fused = self.cross_attention(cnn_features, transformer_features)  # [batch, d_model]
        output = self.highway(fused, transformer_features.mean(dim=1))   # [batch, d_model]
        
        return self.classifier(output)