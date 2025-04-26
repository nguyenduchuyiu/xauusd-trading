import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TradingTransformer(nn.Module):
    def __init__(self, 
                 num_features, 
                 num_classes=3, 
                 d_model=512,
                 nhead=8,
                 num_layers=6,
                 dropout=0.1):
        super().__init__()
        
        # 1. Feature Projection
        self.input_proj = nn.Linear(num_features, d_model)
        
        # 2. Positional Encoding (Learnable)
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout)
        
        # 3. Transformer Encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True 
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, num_classes)
        )
        
    #     # 5. Initialize weights
    #     self.init_weights()

    # def init_weights(self):
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)

    def forward(self, src):
        """
        Args:
            src: Tensor shape [batch_size, seq_len, num_features]
        Returns:
            output: Tensor shape [batch_size, num_classes]
        """
        # Project input features
        x = self.input_proj(src)  # [B, S, D]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer processing
        memory = self.transformer_encoder(x)  # [B, S, D]
        
        # Get last time step output
        last_output = memory[:, -1, :]  # [B, D]
        
        # Classification
        return self.classifier(last_output)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_emb = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.position_emb, mean=0, std=0.02)

    def forward(self, x):
        """
        Args:
            x: Tensor shape [B, S, D]
        """
        positions = self.position_emb[:x.size(1), :]  # [S, D]
        x = x + positions.unsqueeze(0)  # [B, S, D]
        return self.dropout(x)