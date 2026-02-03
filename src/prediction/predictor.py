"""
예측 모듈
모델 추론 및 Feature Importance 계산
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from src.config import config, DEVICE, SEED, BASE_DIR
from src.data.dataset import PatchTSTDataset
from src.models.patch_tst import PatchTSTModel


class Predictor:
    """예측 및 분석 클래스"""
    
    def __init__(self, model: PatchTSTModel, scaler_y, feat_names: List[str]):
        """
        Parameters:
            model: 학습된 PatchTST 모델
            scaler_y: 타겟 스케일러
            feat_names: 피처 이름
        """
        self.model = model
        self.scaler_y = scaler_y
        self.feat_names = feat_names
        self.pred_len = model.head_trend.out_features
    
    def predict(self, X_sc: np.ndarray, y_sc: np.ndarray,
                batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        예측 수행
        
        Args:
            X_sc: 스케일된 입력 (N, F)
            y_sc: 스케일된 타겟 (N,)
            batch_size: 배치 크기
            
        Returns:
            predictions: (N_samples, pred_len) 원본 스케일 예측값
            truths: (N_samples, pred_len) 원본 스케일 실제값
        """
        seq_len = config.get_effective_seq_len()
        
        ds = PatchTSTDataset(X_sc, y_sc, seq_len, self.pred_len,
                            config.PATCH_LEN, config.STRIDE)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        preds = []
        trues = []
        
        with torch.no_grad():
            for Xb, yb, _ in dl:
                Xb = Xb.to(DEVICE)
                pred = self.model(Xb).cpu().numpy()
                preds.append(pred)
                trues.append(yb.numpy())
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # 원본 스케일로 변환
        preds_orig = self.scaler_y.inverse_transform(
            preds.reshape(-1, 1)
        ).reshape(-1, self.pred_len)
        trues_orig = self.scaler_y.inverse_transform(
            trues.reshape(-1, 1)
        ).reshape(-1, self.pred_len)
        
        return preds_orig, trues_orig
    
    def predict_single(self, X_sc: np.ndarray) -> np.ndarray:
        """
        단일 시퀀스 예측
        
        Args:
            X_sc: 스케일된 입력 시퀀스 (seq_len, F)
            
        Returns:
            prediction: (pred_len,) 원본 스케일 예측값
        """
        seq_len = config.get_effective_seq_len()
        
        # Patchify
        patches = []
        pos = 0
        while pos + config.PATCH_LEN <= seq_len:
            patches.append(X_sc[pos:pos+config.PATCH_LEN, :])
            pos += config.STRIDE
        
        X_patch = np.stack(patches, axis=0)  # (P, patch_len, F)
        X_tensor = torch.from_numpy(X_patch).unsqueeze(0).float().to(DEVICE)  # (1, P, L, F)
        
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_tensor).cpu().numpy().ravel()
        
        # 원본 스케일로 변환
        pred_orig = self.scaler_y.inverse_transform(pred.reshape(-1, 1)).ravel()
        
        return pred_orig
    
    def get_horizon_predictions(self, X_sc: np.ndarray, y_sc: np.ndarray,
                                 batch_size: int = 64) -> pd.DataFrame:
        """
        Horizon별 예측 결과 DataFrame 반환
        
        Returns:
            DataFrame with columns: sample_idx, pred_1w, true_1w, error_1w, ...
        """
        preds, trues = self.predict(X_sc, y_sc, batch_size)
        
        results = []
        for i in range(len(preds)):
            row = {'sample_idx': i}
            for h in range(1, self.pred_len + 1):
                row[f'pred_{h}w'] = preds[i, h-1]
                row[f'true_{h}w'] = trues[i, h-1]
                row[f'error_{h}w'] = preds[i, h-1] - trues[i, h-1]
            results.append(row)
        
        return pd.DataFrame(results)


class FeatureImportance:
    """Feature Importance 계산 클래스"""
    
    def __init__(self, model: PatchTSTModel, scaler_y, feat_names: List[str]):
        """
        Parameters:
            model: 학습된 모델
            scaler_y: 타겟 스케일러
            feat_names: 피처 이름
        """
        self.model = model
        self.scaler_y = scaler_y
        self.feat_names = feat_names
        self.pred_len = model.head_trend.out_features
    
    def _eval_mse(self, X_sc: np.ndarray, y_sc: np.ndarray,
                  batch_size: int = 64) -> float:
        """MSE 계산"""
        seq_len = config.get_effective_seq_len()
        
        ds = PatchTSTDataset(X_sc, y_sc, seq_len, self.pred_len,
                            config.PATCH_LEN, config.STRIDE)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        mse_sum, n = 0.0, 0
        
        with torch.no_grad():
            for Xb, yb, _ in dl:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                pred = self.model(Xb)
                
                H = pred.shape[1]
                yb = yb[:, :H]
                
                pred_np = pred.cpu().numpy()
                yb_np = yb.cpu().numpy()
                
                pred_orig = self.scaler_y.inverse_transform(pred_np.reshape(-1, 1)).flatten()
                yb_orig = self.scaler_y.inverse_transform(yb_np.reshape(-1, 1)).flatten()
                
                mse = np.mean((pred_orig - yb_orig) ** 2)
                mse_sum += mse * yb.size(0)
                n += yb.size(0)
        
        return float(mse_sum / max(1, n))
    
    def compute(self, X_va_sc: np.ndarray, y_va_sc: np.ndarray,
                X_te_sc: np.ndarray = None, y_te_sc: np.ndarray = None) -> pd.DataFrame:
        """
        Perturbation-Based Feature Importance 계산
        
        Args:
            X_va_sc: 검증 데이터 피처
            y_va_sc: 검증 데이터 타겟
            X_te_sc: 테스트 데이터 피처 (Optional)
            y_te_sc: 테스트 데이터 타겟 (Optional)
            
        Returns:
            DataFrame with feature importance scores
        """
        # 'ili' 제외 (타겟 변수)
        feat_indices = [i for i, name in enumerate(self.feat_names) if name != 'ili']
        filtered_feat_names = [self.feat_names[i] for i in feat_indices]
        
        if len(filtered_feat_names) < len(self.feat_names):
            print(f"[FI] 'ili' 특징 제외됨 (타겟 변수)")
            print(f"[FI] Feature Importance 계산 대상: {len(filtered_feat_names)}개 특징")
        
        # Baseline MSE
        print(f"[FI] Computing Baseline MSE...")
        mse_original_val = self._eval_mse(X_va_sc, y_va_sc)
        print(f"[FI] Baseline Val MSE: {mse_original_val:.6f}")
        
        mse_original_tst = None
        if X_te_sc is not None and y_te_sc is not None:
            mse_original_tst = self._eval_mse(X_te_sc, y_te_sc)
            print(f"[FI] Baseline Test MSE: {mse_original_tst:.6f}")
        
        # Perturbation Importance
        print(f"[FI] Computing Perturbation Importance...")
        importance_val = []
        importance_tst = []
        
        for j in feat_indices:
            name = self.feat_names[j]
            
            # Validation: 피처를 평균값으로 마스킹
            X_masked_val = X_va_sc.copy()
            X_masked_val[:, j] = X_va_sc[:, j].mean()
            
            mse_masked_val = self._eval_mse(X_masked_val, y_va_sc)
            delta_mse_val = mse_masked_val - mse_original_val
            importance_val.append(delta_mse_val)
            
            print(f"  - {name}: ΔMSE={delta_mse_val:.6f}")
            
            # Test (optional)
            if X_te_sc is not None and y_te_sc is not None:
                X_masked_tst = X_te_sc.copy()
                X_masked_tst[:, j] = X_te_sc[:, j].mean()
                
                mse_masked_tst = self._eval_mse(X_masked_tst, y_te_sc)
                delta_mse_tst = mse_masked_tst - mse_original_tst
                importance_tst.append(delta_mse_tst)
        
        # Normalization
        importance_val = np.array(importance_val)
        sum_importance_val = np.abs(importance_val).sum()
        if sum_importance_val > 0:
            importance_norm_val = importance_val / sum_importance_val
        else:
            importance_norm_val = np.zeros_like(importance_val)
        
        importance_norm_tst = None
        if importance_tst:
            importance_tst = np.array(importance_tst)
            sum_importance_tst = np.abs(importance_tst).sum()
            if sum_importance_tst > 0:
                importance_norm_tst = importance_tst / sum_importance_tst
            else:
                importance_norm_tst = np.zeros_like(importance_tst)
        
        # DataFrame 생성
        from src.config import COLUMN_MAPPING
        inv_colmap = {v: k for k, v in COLUMN_MAPPING.items()}
        
        feature_disp = [f"{f} ({inv_colmap[f]})" if f in inv_colmap else f 
                        for f in filtered_feat_names]
        
        df_fi = pd.DataFrame({
            "feature": feature_disp,
            "importance_raw_val": importance_val,
            "importance_norm_val": importance_norm_val,
        })
        
        if importance_norm_tst is not None:
            df_fi["importance_raw_tst"] = importance_tst
            df_fi["importance_norm_tst"] = importance_norm_tst
        
        df_fi = df_fi.sort_values("importance_raw_val", ascending=False).reset_index(drop=True)
        
        print(f"\n[FI] Feature Importance Calculation Complete!")
        return df_fi
