"""
하이퍼파라미터 최적화 모듈 (Optuna)
"""

from typing import Dict, Any, Optional
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not installed. Hyperparameter optimization disabled.")

from src.config import config, DEVICE, SEED, BASE_DIR
from src.data.dataset import PatchTSTDataset
from src.data.preprocessing import get_scaler, make_splits
from src.models.patch_tst import PatchTSTModel
from src.training.utils import set_seed, batch_mae_in_original_units


def optimize_hyperparameters(X: np.ndarray, y: np.ndarray, 
                              labels: list, feat_names: list,
                              n_trials: int = 50) -> Optional[Dict[str, Any]]:
    """
    Optuna를 사용한 하이퍼파라미터 최적화
    
    Args:
        X: 입력 피처 (N, F)
        y: 타겟 변수 (N,)
        labels: 시간 라벨
        feat_names: 피처 이름
        n_trials: 최적화 시도 횟수
        
    Returns:
        best_params: 최적 하이퍼파라미터 dict
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not installed. Install with: pip install optuna")
    
    print("\n" + "=" * 70)
    print("🔍 Optuna 하이퍼파라미터 최적화 시작")
    if config.USE_DAILY_DATA:
        seq_len = config.DAILY_SEQ_LEN
        pred_len = config.DAILY_PRED_LEN
        print(f"   📅 일별 데이터 모드 (SEQ_LEN={seq_len}, PRED_LEN={pred_len})")
    else:
        seq_len = config.SEQ_LEN
        pred_len = config.PRED_LEN
        print(f"   📆 주차별 데이터 모드 (SEQ_LEN={seq_len}, PRED_LEN={pred_len})")
    print("=" * 70)
    
    # Train 데이터 기준 피크 threshold 계산
    (s0, e0), _, _ = make_splits(len(y))
    y_tr = y[s0:e0]
    
    def objective(trial: Trial) -> float:
        """Optuna objective function - validation MAE + Peak MAE 최소화"""
        
        search_space = config.OPTUNA_SEARCH_SPACE
        
        # 하이퍼파라미터 샘플링
        params = {}
        params['d_model'] = trial.suggest_categorical('d_model', search_space['d_model'])
        params['n_heads'] = trial.suggest_categorical('n_heads', search_space['n_heads'])
        params['enc_layers'] = trial.suggest_int('enc_layers', *search_space['enc_layers'])
        params['ff_dim'] = trial.suggest_categorical('ff_dim', search_space['ff_dim'])
        params['dropout'] = trial.suggest_float('dropout', *search_space['dropout'])
        params['lr'] = trial.suggest_float('lr', *search_space['lr'], log=True)
        params['weight_decay'] = trial.suggest_float('weight_decay', *search_space['weight_decay'], log=True)
        params['batch_size'] = trial.suggest_categorical('batch_size', search_space['batch_size'])
        
        # seq_len / pred_len: 일별 데이터일 때는 고정값
        if config.USE_DAILY_DATA:
            params['seq_len'] = seq_len
            params['pred_len'] = pred_len
        else:
            params['seq_len'] = seq_len
            params['pred_len'] = pred_len
        
        # patch_len
        if 'patch_len' in search_space:
            params['patch_len'] = trial.suggest_categorical('patch_len', search_space['patch_len'])
        else:
            params['patch_len'] = config.PATCH_LEN
        
        # d_model은 4의 배수여야 함
        if params['d_model'] % 4 != 0:
            params['d_model'] = (params['d_model'] // 4) * 4
        
        # n_heads는 d_model의 약수여야 함
        while params['d_model'] % params['n_heads'] != 0:
            params['n_heads'] //= 2
            if params['n_heads'] < 1:
                params['n_heads'] = 1
                break
        
        # 데이터 분할
        (s0, e0), (s1, e1), _ = make_splits(len(y))
        X_tr, X_va = X[s0:e0], X[s1:e1]
        y_tr_split, y_va_split = y[s0:e0], y[s1:e1]
        
        # 스케일링
        scaler_y = get_scaler(for_target=True)
        y_tr_sc = scaler_y.fit_transform(y_tr_split.reshape(-1, 1)).ravel()
        y_va_sc = scaler_y.transform(y_va_split.reshape(-1, 1)).ravel()
        
        scaler_x = get_scaler(for_target=False)
        X_tr_sc = scaler_x.fit_transform(X_tr)
        X_va_sc = scaler_x.transform(X_va)
        
        F = X.shape[1]
        
        # Dataset 생성
        try:
            ds_tr = PatchTSTDataset(X_tr_sc, y_tr_sc, params['seq_len'], params['pred_len'],
                                    params['patch_len'], config.STRIDE)
            ds_va = PatchTSTDataset(X_va_sc, y_va_sc, params['seq_len'], params['pred_len'],
                                    params['patch_len'], config.STRIDE)
        except:
            return float('inf')
        
        if len(ds_tr) < 1 or len(ds_va) < 1:
            return float('inf')
        
        dl_tr = DataLoader(ds_tr, batch_size=params['batch_size'], shuffle=True, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=params['batch_size'], shuffle=False)
        
        # 모델 생성
        model = PatchTSTModel(
            in_features=F, patch_len=params['patch_len'], d_model=params['d_model'],
            n_heads=params['n_heads'], n_layers=params['enc_layers'], ff_dim=params['ff_dim'],
            dropout=params['dropout'], pred_len=params['pred_len'], 
            head_hidden=config.HEAD_HIDDEN
        ).to(DEVICE)
        
        crit = nn.HuberLoss(delta=1.0)
        opt = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        
        # Early stopping
        best_val_metric = float('inf')
        patience_count = 0
        early_stop_patience = 20
        
        # 학습
        max_epochs = 50
        for ep in range(1, max_epochs + 1):
            # Train
            model.train()
            for Xb, yb, _ in dl_tr:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                pred = model(Xb)
                loss = crit(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            
            # Validation
            model.eval()
            va_mae_sum = 0
            n = 0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for Xb, yb, _ in dl_va:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    pred = model(Xb)
                    bs = yb.size(0)
                    va_mae_sum += batch_mae_in_original_units(pred, yb, scaler_y) * bs
                    n += bs
                    
                    pred_orig = scaler_y.inverse_transform(pred.cpu().numpy().reshape(-1, 1)).ravel()
                    target_orig = scaler_y.inverse_transform(yb.cpu().numpy().reshape(-1, 1)).ravel()
                    all_preds.extend(pred_orig)
                    all_targets.extend(target_orig)
            
            va_mae = va_mae_sum / max(1, n)
            
            # Peak MAE 계산
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            peak_threshold = np.quantile(y_tr, 0.9)
            peak_mask = all_targets >= peak_threshold
            
            if peak_mask.sum() > 0:
                peak_mae = np.mean(np.abs(all_preds[peak_mask] - all_targets[peak_mask]))
            else:
                peak_mae = 0.0
            
            # 복합 목적 함수
            combined_metric = va_mae + 0.6 * peak_mae
            
            # Early stopping
            if combined_metric < best_val_metric:
                best_val_metric = combined_metric
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= early_stop_patience:
                    break
            
            # Optuna pruning
            trial.report(combined_metric, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return combined_metric
    
    # Optuna study 생성 및 실행
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # 결과 출력
    print("\n" + "=" * 70)
    print("✅ Optuna 최적화 완료")
    print("=" * 70)
    print(f"\n🏆 Best Trial:")
    print(f"  - Value (Val MAE + 0.6*Peak MAE): {study.best_trial.value:.4f}")
    print(f"\n📊 Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    
    # Best parameters 저장
    import json
    best_params_file = BASE_DIR / "best_hyperparameters.json"
    with open(best_params_file, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\n💾 Best parameters saved to: {best_params_file}")
    
    return study.best_params
