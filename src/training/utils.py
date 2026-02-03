"""
학습 관련 유틸리티 함수
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """랜덤 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def warmup_lr(epoch: int, base_lr: float, warmup_epochs: int) -> float:
    """웜업 학습률 계산"""
    if epoch <= warmup_epochs:
        return base_lr * (epoch / max(1, warmup_epochs))
    return base_lr


def batch_mae_in_original_units(pred_b: torch.Tensor, y_b: torch.Tensor, 
                                 scaler_y) -> float:
    """
    원래 스케일에서 MAE 계산
    
    Args:
        pred_b: (B,) or (B,1) or (B,H) 예측값
        y_b: (B,H) or (B,) 실제값
        scaler_y: 스케일러
    Returns:
        MAE in original units
    """
    p = pred_b.detach().cpu().numpy()
    t = y_b.detach().cpu().numpy()

    if p.ndim == 1:
        p = p[:, None]
    if t.ndim == 1:
        t = t[:, None]

    if p.shape[1] == 1 and t.shape[1] > 1:
        p = np.repeat(p, t.shape[1], axis=1)

    p = p.reshape(-1, 1)
    t = t.reshape(-1, 1)

    p_orig = scaler_y.inverse_transform(p).reshape(-1)
    t_orig = scaler_y.inverse_transform(t).reshape(-1)

    return float(np.mean(np.abs(p_orig - t_orig)))


def batch_mse_in_original_units(pred_b: torch.Tensor, y_b: torch.Tensor,
                                 scaler_y) -> float:
    """원래 스케일에서 MSE 계산"""
    p = pred_b.detach().cpu().numpy()
    t = y_b.detach().cpu().numpy()

    if p.ndim == 1:
        p = p[:, None]
    if t.ndim == 1:
        t = t[:, None]

    if p.shape[1] == 1 and t.shape[1] > 1:
        p = np.repeat(p, t.shape[1], axis=1)

    p = p.reshape(-1, 1)
    t = t.reshape(-1, 1)

    p_orig = scaler_y.inverse_transform(p).reshape(-1)
    t_orig = scaler_y.inverse_transform(t).reshape(-1)

    return float(np.mean((p_orig - t_orig) ** 2))


def batch_rmse_in_original_units(pred_b: torch.Tensor, y_b: torch.Tensor,
                                  scaler_y) -> float:
    """원래 스케일에서 RMSE 계산"""
    return float(np.sqrt(batch_mse_in_original_units(pred_b, y_b, scaler_y)))


def batch_corrcoef(pred_b: torch.Tensor, y_b: torch.Tensor,
                   scaler_y) -> float:
    """Pearson 상관계수 계산"""
    p = pred_b.detach().cpu().numpy().reshape(-1, 1)
    t = y_b.detach().cpu().numpy().reshape(-1, 1)
    
    p_orig = scaler_y.inverse_transform(p).reshape(-1)
    t_orig = scaler_y.inverse_transform(t).reshape(-1)

    if np.std(p_orig) < 1e-6 or np.std(t_orig) < 1e-6:
        return 0.0
    
    return float(np.corrcoef(p_orig, t_orig)[0, 1])
