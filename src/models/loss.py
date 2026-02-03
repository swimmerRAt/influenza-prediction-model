"""
손실 함수 모듈
Peak-Aware Loss, Horizon Weighting 등
"""

import numpy as np
import torch
import torch.nn as nn


class PeakAwareLoss(nn.Module):
    """
    고정 기준 Peak + 진폭 보존 + Horizon Weighting Loss
    
    특징:
    1. Peak 구간(상위 quantile)에 높은 가중치 적용
    2. 진폭 보존 항으로 peak flattening 방지
    3. Horizon weighting: 예측 구간별 가중치
    4. MAE 기반으로 outlier에 robust
    """
    
    def __init__(self, peak_quantile: float = 0.9, alpha: float = 4.0, 
                 beta: float = 0.3, pred_len: int = 4, 
                 horizon_mode: str = "exponential",
                 horizon_exp_scale: float = 1.2, 
                 horizon_tail_boost: float = 2.5, 
                 horizon_tail_count: int = 2):
        """
        Parameters:
            peak_quantile: 피크 기준 (상위 몇 %)
            alpha: 피크 가중치
            beta: 진폭 보존 가중치
            pred_len: 예측 길이
            horizon_mode: "exponential", "tail_boost", "uniform"
            horizon_exp_scale: exponential 모드 스케일
            horizon_tail_boost: tail_boost 모드 배수
            horizon_tail_count: tail_boost 모드 뒤쪽 개수
        """
        super().__init__()
        self.peak_quantile = peak_quantile
        self.alpha = alpha
        self.beta = beta
        self.mae = nn.L1Loss(reduction="none")
        
        # Horizon Weighting 계산
        h_weights = self._compute_horizon_weights(
            pred_len, horizon_mode, horizon_exp_scale,
            horizon_tail_boost, horizon_tail_count
        )
        self.register_buffer('horizon_weights', torch.from_numpy(h_weights).float())
        
        print(f"[Loss] Horizon weights ({horizon_mode}): {h_weights}")
    
    def _compute_horizon_weights(self, pred_len: int, mode: str, 
                                  exp_scale: float, tail_boost: float, 
                                  tail_count: int) -> np.ndarray:
        """예측 구간별 가중치 계산"""
        if mode == "exponential":
            h_weights = np.exp(np.linspace(0, exp_scale, pred_len))
        elif mode == "tail_boost":
            h_weights = np.ones(pred_len)
            h_weights[-tail_count:] *= tail_boost
        else:  # uniform
            h_weights = np.ones(pred_len)
        
        # 정규화 (합이 pred_len이 되도록)
        h_weights = h_weights / h_weights.sum() * pred_len
        return h_weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, H) 예측값
            target: (B, H) 실제값
        Returns:
            loss: scalar
        """
        # Base MAE
        base_loss = self.mae(pred, target)  # (B, H)
        
        # 피크 구간 가중 (배치별 동적 threshold)
        with torch.no_grad():
            peak_threshold = torch.quantile(target, self.peak_quantile)
            peak_mask = target >= peak_threshold
            weights = torch.ones_like(target)
            weights[peak_mask] = self.alpha
        
        # Horizon weighting 적용
        horizon_w = self.horizon_weights.view(1, -1)  # (1, H)
        weighted_mae = (base_loss * weights * horizon_w).mean()
        
        # 진폭 보존 항 (peak flattening 방지)
        pred_max = pred.max(dim=1).values  # (B,)
        target_max = target.max(dim=1).values  # (B,)
        amp_loss = torch.abs(pred_max - target_max).mean()
        
        # 총 손실
        total_loss = weighted_mae + self.beta * amp_loss
        
        return total_loss


def peak_weighted_loss(pred: torch.Tensor, target: torch.Tensor, 
                       peak_quantile: float = 0.9, 
                       alpha: float = 3.0) -> torch.Tensor:
    """
    Peak-aware weighted MAE loss (함수 버전)
    
    Args:
        pred: (B, H) 예측값
        target: (B, H) 실제값
        peak_quantile: 피크 기준
        alpha: 피크 가중치
    Returns:
        loss: scalar
    """
    with torch.no_grad():
        thresh = torch.quantile(target, peak_quantile)
        weights = torch.ones_like(target)
        weights[target >= thresh] = alpha
    
    return torch.mean(weights * torch.abs(pred - target))
