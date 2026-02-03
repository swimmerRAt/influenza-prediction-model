"""
PatchTST 모델 아키텍처
Multi-Scale CNN + TokenConvMixer + Transformer Encoder + AttnPool
"""

import math
from typing import List

import torch
import torch.nn as nn


class MultiScaleCNNPatchEmbed(nn.Module):
    """
    멀티스케일 CNN 패치 임베딩
    (B, P, L, F) -> [각 패치] 멀티스케일 Conv1d 분기(k=1,3,5,7) → GAP → (B, P, D)
    """
    
    def __init__(self, in_features: int, patch_len: int, 
                 d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % 4 == 0, "d_model은 4의 배수여야 합니다."
        out_ch = d_model // 4
        
        self.b2 = nn.Conv1d(in_features, out_ch, kernel_size=1, padding=0)
        self.b3 = nn.Conv1d(in_features, out_ch, kernel_size=3, padding=1)
        self.b5 = nn.Conv1d(in_features, out_ch, kernel_size=5, padding=2)
        self.bd = nn.Conv1d(in_features, out_ch, kernel_size=3, padding=2, dilation=2)

        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, P, L, F) 패치화된 입력
        Returns:
            z: (B, P, D) 임베딩된 패치
        """
        B, P, L, F = x.shape
        x = x.view(B*P, L, F).permute(0, 2, 1)  # (B*P, F, L)

        z = torch.cat([self.b2(x), self.b3(x), self.b5(x), self.bd(x)], dim=1)  # (B*P, D, L)
        z = self.act(self.bn(z))
        z = self.pool(z).squeeze(-1)  # (B*P, D)
        z = self.drop(z)
        
        return z.view(B, P, -1)  # (B, P, D)


class TokenConvMixer(nn.Module):
    """
    패치 토큰 간(P 축) 로컬 연속성 강화
    DepthwiseConv1d(P-축) + PointwiseConv1d
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, P, D)
        Returns:
            z: (B, P, D) with residual connection
        """
        y = z.permute(0, 2, 1)  # (B, D, P)
        y = self.dw(y)
        y = self.pw(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.drop(y)
        y = y.permute(0, 2, 1)  # (B, P, D)
        
        return z + y  # Residual


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div)[:, :pe[:, 1::2].shape[1]]
        else:
            pe[:, 1::2] = torch.cos(position * div)
        
        self.register_buffer("pe", pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, P, D)
        Returns:
            x + positional encoding
        """
        P = x.size(1)
        return x + self.pe[:, :P, :]


class AttnPool(nn.Module):
    """Learnable-query attention pooling over patch tokens"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model))
        self.proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, P, D)
        Returns:
            pooled: (B, D)
        """
        B, P, D = z.shape
        q = self.q.expand(B, -1, -1)  # (B, 1, D)
        k = self.proj(z)  # (B, P, D)
        attn = torch.softmax((q @ k.transpose(1, 2)) / (D ** 0.5), dim=-1)  # (B, 1, P)
        pooled = attn @ z  # (B, 1, D)
        
        return pooled.squeeze(1)  # (B, D)


class PatchTSTModel(nn.Module):
    """
    PatchTST 모델
    Multi-Scale CNN + TokenConvMixer + Transformer Encoder + Dual-head 예측
    """
    
    def __init__(self, in_features: int, patch_len: int, d_model: int, 
                 n_heads: int, n_layers: int, ff_dim: int, dropout: float,
                 pred_len: int, head_hidden: List[int]):
        """
        Parameters:
            in_features: 입력 피처 수
            patch_len: 패치 길이
            d_model: 모델 차원
            n_heads: Attention head 수
            n_layers: Encoder 레이어 수
            ff_dim: Feed-forward 차원
            dropout: Dropout 비율
            pred_len: 예측 길이
            head_hidden: Prediction head hidden layer 크기
        """
        super().__init__()
        
        # ① 멀티스케일 CNN 패치 임베딩
        self.embed = MultiScaleCNNPatchEmbed(
            in_features, patch_len, d_model, dropout=dropout * 0.5
        )
        
        # ② 패치 토큰 간 로컬 연속성 믹서
        self.mixer = nn.Sequential(
            TokenConvMixer(d_model, dropout=dropout),
            TokenConvMixer(d_model, dropout=dropout),
        )
        
        # ③ PatchTST 인코더
        self.posenc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pool = AttnPool(d_model)

        # ④ Dual-head 예측: Trend + Peak
        mlp_shared, in_dim = [], d_model
        for h in head_hidden[:2]:
            mlp_shared += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        self.shared_mlp = nn.Sequential(*mlp_shared) if mlp_shared else nn.Identity()
        
        # Dual heads
        self.head_trend = nn.Linear(in_dim, pred_len)  # 기본 트렌드
        self.head_peak = nn.Linear(in_dim, pred_len)   # 피크 보정

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, P, L, F) 패치화된 입력
        Returns:
            pred: (B, H) 예측값
        """
        z = self.embed(x)       # (B, P, D)
        z = self.mixer(z)       # (B, P, D)
        z = self.posenc(z)
        z = self.encoder(z)
        z = self.pool(z)        # (B, D)
        
        # Shared MLP
        z = self.shared_mlp(z)  # (B, hidden_dim)
        
        # Dual-head prediction with adaptive gating
        trend = self.head_trend(z)              # (B, H) - 기본 트렌드
        peak = torch.relu(self.head_peak(z))    # (B, H) - 피크 보정 (양수만)
        
        # trend가 클 때 peak 영향 증가 (sigmoid gating)
        return trend + peak * torch.sigmoid(trend)  # (B, H)


def correlation_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    Correlation Loss: 예측-실제값 상관관계 유지
    
    Args:
        pred: (B, H) 예측값
        true: (B, H) 실제값
    Returns:
        loss: scalar
    """
    pred = pred - pred.mean(dim=1, keepdim=True)
    true = true - true.mean(dim=1, keepdim=True)
    corr = (pred * true).sum(dim=1) / (
        (pred.norm(dim=1) * true.norm(dim=1)) + 1e-6
    )
    return 1 - corr.mean()
