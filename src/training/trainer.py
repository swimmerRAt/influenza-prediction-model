"""
모델 학습 모듈
Trainer 클래스 정의
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import config, DEVICE, SEED
from src.data.dataset import PatchTSTDataset
from src.data.preprocessing import get_scaler, make_splits
from src.models.patch_tst import PatchTSTModel
from src.models.loss import PeakAwareLoss, peak_weighted_loss
from src.training.utils import (
    set_seed, warmup_lr, batch_mae_in_original_units,
    batch_mse_in_original_units, batch_rmse_in_original_units,
    batch_corrcoef
)


class Trainer:
    """PatchTST 모델 학습기"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 labels: List[str], feat_names: List[str],
                 hyperparams: Optional[Dict[str, Any]] = None):
        """
        Parameters:
            X: (N, F) 입력 피처
            y: (N,) 타겟
            labels: 시간 라벨
            feat_names: 피처 이름
            hyperparams: Optuna 최적화된 하이퍼파라미터 (Optional)
        """
        self.X = X
        self.y = y
        self.labels = labels
        self.feat_names = feat_names
        
        # 하이퍼파라미터 설정
        self._setup_hyperparams(hyperparams)
        
        # 히스토리
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_mae": [], "val_mae": []
        }
        
        # 모델 및 데이터 초기화
        self.model = None
        self.scaler_x = None
        self.scaler_y = None
        self.best_state = None
        self.best_val = float("inf")
    
    def _setup_hyperparams(self, hyperparams: Optional[Dict] = None):
        """하이퍼파라미터 설정"""
        if hyperparams:
            self.d_model = hyperparams.get('d_model', config.D_MODEL)
            self.n_heads = hyperparams.get('n_heads', config.N_HEADS)
            self.enc_layers = hyperparams.get('enc_layers', config.ENC_LAYERS)
            self.ff_dim = hyperparams.get('ff_dim', config.FF_DIM)
            self.dropout = hyperparams.get('dropout', config.DROPOUT)
            self.lr = hyperparams.get('lr', config.LR)
            self.weight_decay = hyperparams.get('weight_decay', config.WEIGHT_DECAY)
            self.batch_size = hyperparams.get('batch_size', config.BATCH_SIZE)
            self.seq_len = hyperparams.get('seq_len', config.get_effective_seq_len())
            self.pred_len = hyperparams.get('pred_len', config.get_effective_pred_len())
            self.patch_len = hyperparams.get('patch_len', config.PATCH_LEN)
        else:
            self.d_model = config.D_MODEL
            self.n_heads = config.N_HEADS
            self.enc_layers = config.ENC_LAYERS
            self.ff_dim = config.FF_DIM
            self.dropout = config.DROPOUT
            self.lr = config.LR
            self.weight_decay = config.WEIGHT_DECAY
            self.batch_size = config.BATCH_SIZE
            self.seq_len = config.get_effective_seq_len()
            self.pred_len = config.get_effective_pred_len()
            self.patch_len = config.PATCH_LEN
        
        self.stride = config.STRIDE
        self.epochs = config.EPOCHS
        self.patience = config.PATIENCE
        self.warmup_epochs = config.WARMUP_EPOCHS
        self.head_hidden = config.HEAD_HIDDEN
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """데이터 준비 및 분할"""
        set_seed(SEED)
        
        # 데이터 분할
        (s0, e0), (s1, e1), (s2, e2) = make_splits(len(self.y))
        
        X_tr, X_va, X_te = self.X[s0:e0], self.X[s1:e1], self.X[s2:e2]
        y_tr, y_va, y_te = self.y[s0:e0], self.y[s1:e1], self.y[s2:e2]
        
        self.lab_tr = self.labels[s0:e0]
        self.lab_va = self.labels[s1:e1]
        self.lab_te = self.labels[s2:e2]
        
        self.y_tr = y_tr
        self.y_va = y_va
        self.y_te = y_te
        
        print(f"\n📊 데이터 분할 정보:")
        print(f"   Train: {self.lab_tr[0]} ~ {self.lab_tr[-1]} ({len(y_tr)}개)")
        print(f"   Val:   {self.lab_va[0]} ~ {self.lab_va[-1]} ({len(y_va)}개)")
        print(f"   Test:  {self.lab_te[0]} ~ {self.lab_te[-1]} ({len(y_te)}개)")
        
        # 스케일링
        self.scaler_y = get_scaler(for_target=True)
        y_tr_sc = self.scaler_y.fit_transform(y_tr.reshape(-1, 1)).ravel()
        y_va_sc = self.scaler_y.transform(y_va.reshape(-1, 1)).ravel()
        y_te_sc = self.scaler_y.transform(y_te.reshape(-1, 1)).ravel()
        
        self.scaler_x = get_scaler(for_target=False)
        X_tr_sc = self.scaler_x.fit_transform(X_tr)
        X_va_sc = self.scaler_x.transform(X_va)
        X_te_sc = self.scaler_x.transform(X_te)
        
        # 스케일된 데이터 저장
        self.X_va_sc = X_va_sc
        self.X_te_sc = X_te_sc
        self.y_va_sc = y_va_sc
        self.y_te_sc = y_te_sc
        
        F = self.X.shape[1]
        print(f"[Shapes] X_tr:{X_tr.shape}, X_va:{X_va.shape}, X_te:{X_te.shape} | F={F}")
        print(f"[Info] Model input feature order -> {self.feat_names}")
        
        # 데이터셋 생성
        ds_tr = PatchTSTDataset(X_tr_sc, y_tr_sc, self.seq_len, self.pred_len, 
                                self.patch_len, self.stride)
        ds_va = PatchTSTDataset(X_va_sc, y_va_sc, self.seq_len, self.pred_len,
                                self.patch_len, self.stride)
        ds_te = PatchTSTDataset(X_te_sc, y_te_sc, self.seq_len, self.pred_len,
                                self.patch_len, self.stride)
        
        # DataLoader 생성
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=self.batch_size, shuffle=False)
        dl_te = DataLoader(ds_te, batch_size=self.batch_size, shuffle=False)
        
        return dl_tr, dl_va, dl_te
    
    def build_model(self) -> PatchTSTModel:
        """모델 생성"""
        F = self.X.shape[1]
        
        self.model = PatchTSTModel(
            in_features=F,
            patch_len=self.patch_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.enc_layers,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            pred_len=self.pred_len,
            head_hidden=self.head_hidden
        ).to(DEVICE)
        
        return self.model
    
    def train(self, dl_tr: DataLoader, dl_va: DataLoader) -> Dict[str, List[float]]:
        """모델 학습"""
        if self.model is None:
            self.build_model()
        
        # Loss / Optimizer / Scheduler
        criterion = peak_weighted_loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )
        
        print(f"\n[Config] EPOCHS:{self.epochs}, BATCH_SIZE:{self.batch_size}, "
              f"SEQ_LEN:{self.seq_len}, PRED_LEN:{self.pred_len}")
        print(f"[Config] PATCH_LEN:{self.patch_len}, LR:{self.lr}, "
              f"Warmup:{self.warmup_epochs}, Patience:{self.patience}")
        
        noimp = 0
        printed_batch_info = False
        
        for ep in range(1, self.epochs + 1):
            # --- Train ---
            self.model.train()
            tr_loss_sum = 0
            tr_mae_sum = 0
            n = 0
            
            # Warmup LR
            for g in optimizer.param_groups:
                g['lr'] = warmup_lr(ep, self.lr, self.warmup_epochs)
            
            for Xb, yb, _ in dl_tr:
                if not printed_batch_info:
                    print(f"[Batch shapes] Xb:{Xb.shape}, yb:{yb.shape}")
                    printed_batch_info = True
                
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                
                optimizer.zero_grad()
                pred = self.model(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                bs = yb.size(0)
                tr_loss_sum += loss.item() * bs
                tr_mae_sum += batch_mae_in_original_units(pred, yb, self.scaler_y) * bs
                n += bs
            
            tr_loss = tr_loss_sum / max(1, n)
            tr_mae = tr_mae_sum / max(1, n)
            
            # --- Validation ---
            self.model.eval()
            va_loss_sum = 0
            va_mae_sum = 0
            m = 0
            
            with torch.no_grad():
                for Xb, yb, _ in dl_va:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    pred = self.model(Xb)
                    loss = criterion(pred, yb)
                    
                    bs = yb.size(0)
                    va_loss_sum += loss.item() * bs
                    va_mae_sum += batch_mae_in_original_units(pred, yb, self.scaler_y) * bs
                    m += bs
            
            va_loss = va_loss_sum / max(1, m)
            va_mae = va_mae_sum / max(1, m)
            
            # 히스토리 저장
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["train_mae"].append(tr_mae)
            self.history["val_mae"].append(va_mae)
            
            # 로그 출력
            if ep <= 5 or ep % 5 == 0:
                print(f"Epoch {ep:3d}/{self.epochs} | "
                      f"TrL:{tr_loss:.6f} TrMAE:{tr_mae:.6f} | "
                      f"VaL:{va_loss:.6f} VaMAE:{va_mae:.6f}")
            
            # Early stopping
            if va_mae < self.best_val:
                self.best_val = va_mae
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                noimp = 0
            else:
                noimp += 1
                if noimp >= self.patience:
                    print(f"Early stop at epoch {ep} (no improvement for {self.patience} epochs)")
                    break
            
            scheduler.step()
        
        # 최적 상태 로드
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        print(f"\nBest Val MAE: {self.best_val:.6f}")
        
        return self.history
    
    def evaluate(self, dl_te: DataLoader) -> Dict[str, float]:
        """모델 평가"""
        self.model.eval()
        
        te_mae_sum = 0
        te_mse_sum = 0
        te_rmse_sum = 0
        k = 0
        
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for Xb, yb, _ in dl_te:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                pred = self.model(Xb)
                
                te_mae_sum += batch_mae_in_original_units(pred, yb, self.scaler_y) * yb.size(0)
                te_mse_sum += batch_mse_in_original_units(pred, yb, self.scaler_y) * yb.size(0)
                te_rmse_sum += batch_rmse_in_original_units(pred, yb, self.scaler_y) * yb.size(0)
                k += yb.size(0)
                
                # 원본 스케일로 변환
                pred_np = pred.cpu().numpy()
                yb_np = yb.cpu().numpy()
                pred_orig = self.scaler_y.inverse_transform(
                    pred_np.reshape(-1, 1)
                ).reshape(-1, self.pred_len)
                yb_orig = self.scaler_y.inverse_transform(
                    yb_np.reshape(-1, 1)
                ).reshape(-1, self.pred_len)
                
                all_preds.append(pred_orig)
                all_trues.append(yb_orig)
        
        # 메트릭 계산
        metrics = {
            'mae': te_mae_sum / max(1, k),
            'mse': te_mse_sum / max(1, k),
            'rmse': te_rmse_sum / max(1, k)
        }
        
        self.all_preds = np.concatenate(all_preds, axis=0)
        self.all_trues = np.concatenate(all_trues, axis=0)
        
        print("\n" + "=" * 60)
        print("🎯 최종 테스트 성능 평가")
        print("=" * 60)
        print(f"MAE  (Mean Absolute Error):      {metrics['mae']:.6f}")
        print(f"MSE  (Mean Squared Error):       {metrics['mse']:.6f}")
        print(f"RMSE (Root Mean Squared Error):  {metrics['rmse']:.6f}")
        print("=" * 60)
        
        return metrics
    
    def get_results(self) -> Tuple:
        """학습 결과 반환"""
        return (
            self.model, 
            self.X_va_sc, self.y_va_sc,
            self.X_te_sc, self.y_te_sc,
            self.scaler_y, self.feat_names,
            self.history
        )
