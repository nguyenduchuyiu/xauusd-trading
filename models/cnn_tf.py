import torch
import torch.nn as nn


class HybridTradingModel(nn.Module):
    def __init__(self, num_features, num_classes=3, d_model=128, nhead=8, dim_feedforward=512, num_layers=3):
        super().__init__()
        
        # 1. CNN Branch (cho feature extraction)
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 2. Transformer Branch
        self.transformer_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Fusion & Classification
        self.fusion = nn.Linear(128 + d_model, 256)  # CNN output + Transformer output
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: [batch, seq_len, num_features]
        
        # CNN Path (requires [batch, channels, seq_len])
        cnn_features = self.cnn(x.permute(0, 2, 1))  # [batch, 128, seq_len//2]
        cnn_features = cnn_features.mean(dim=-1)      # Global Avg Pooling [batch, 128]
        
        # Transformer Path
        transformer_features = self.transformer_proj(x)  # [batch, seq_len, d_model]
        transformer_features = self.transformer(transformer_features)  # [batch, seq_len, d_model]
        transformer_features = transformer_features.mean(dim=1)  # Pooling [batch, d_model]
        
        # Fusion
        combined = torch.cat([cnn_features, transformer_features], dim=-1)
        return self.classifier(self.fusion(combined))