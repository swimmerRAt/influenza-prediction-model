"""
설정 관리 모듈
모든 하이퍼파라미터와 설정을 중앙에서 관리
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import torch
from dotenv import load_dotenv


# =========================
# 환경 변수 로드
# =========================
def load_environment():
    """환경 변수 로드"""
    env_path = Path.cwd() / '.env'
    load_dotenv(env_path, verbose=True, override=True)
    return env_path.exists()


# =========================
# 디바이스 선택
# =========================
def pick_device() -> str:
    """사용 가능한 최적의 디바이스 선택"""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =========================
# 기본 경로 설정
# =========================
BASE_DIR = Path.cwd()
DEVICE = pick_device()
SEED = 42


@dataclass
class Config:
    """모델 설정 통합 관리"""
    
    # ===== Optuna 최적화 설정 =====
    USE_OPTUNA: bool = False
    N_TRIALS: int = 50
    OPTUNA_TIMEOUT: Optional[int] = None
    
    OPTUNA_SEARCH_SPACE: Dict[str, Any] = field(default_factory=lambda: {
        'd_model': [64, 128, 256],
        'n_heads': [2, 4, 8, 16],
        'enc_layers': (2, 8),
        'ff_dim': [64, 96, 128, 192, 256, 384, 512],
        'dropout': (0.05, 0.5),
        'lr': (1e-6, 1e-2),
        'weight_decay': (1e-6, 1e-2),
        'batch_size': [16, 32, 48, 64, 96, 128],
        'seq_len': (8, 30),
        'patch_len': [2, 3, 4, 5, 6],
    })
    
    # ===== 모델 하이퍼파라미터 =====
    EPOCHS: int = 200
    BATCH_SIZE: int = 64
    SEQ_LEN: int = 16
    PRED_LEN: int = 4
    PATCH_LEN: int = 4
    STRIDE: int = 1
    
    # 모델 아키텍처
    D_MODEL: int = 128
    N_HEADS: int = 2
    ENC_LAYERS: int = 4
    FF_DIM: int = 128
    DROPOUT: float = 0.3
    HEAD_HIDDEN: List[int] = field(default_factory=lambda: [64, 64])
    
    # ===== 학습 설정 =====
    LR: float = 5e-4
    WEIGHT_DECAY: float = 5e-4
    PATIENCE: int = 60
    WARMUP_EPOCHS: int = 30
    
    # ===== Loss 함수 설정 =====
    PEAK_THRESHOLD_QUANTILE: float = 0.85
    PEAK_WEIGHT_ALPHA: float = 12.0
    AMPLITUDE_WEIGHT_BETA: float = 0.6
    
    # Horizon Weighting
    HORIZON_WEIGHT_MODE: str = "exponential"
    HORIZON_EXP_SCALE: float = 2.0
    HORIZON_TAIL_BOOST: float = 2.5
    HORIZON_TAIL_COUNT: int = 2
    
    # ===== 데이터 설정 =====
    TRAIN_RATIO: float = 0.7
    VAL_RATIO: float = 0.15
    SCALER_TYPE: str = "robust"
    
    # Log 변환 설정
    USE_LOG_TRANSFORM: bool = True
    LOG_EPSILON: float = 0.000001
    
    # 외생 특징 사용 모드
    USE_EXOG: str = "all"
    INCLUDE_SEASONAL_FEATS: bool = True
    
    # ===== 연령대별 동학 설정 =====
    USE_AGE_GROUP_DYNAMICS: bool = False
    LEAD_AGE_GROUPS: List[str] = field(default_factory=lambda: ["0세", "1-6세", "7-12세"])
    
    # ===== 피처 제외 설정 =====
    EXCLUDE_HOSPITALIZATION: bool = True
    
    # ===== 일별 데이터 변환 설정 =====
    USE_DAILY_DATA: bool = True
    DAILY_INTERP_METHOD: str = "linear"
    GAUSSIAN_STD: float = 1.0
    DAILY_SEQ_LEN: int = 112
    DAILY_PRED_LEN: int = 28
    
    # ===== 트렌드 데이터 설정 =====
    USE_TRENDS_DATA: bool = False
    TRENDS_DB_NAME: str = "trends"
    TRENDS_TABLE_NAME: str = "trends_data"
    
    # ===== 출력 설정 =====
    OUT_CSV: str = str(BASE_DIR / "ili_predictions.csv")
    PLOT_LAST_WINDOW: str = str(BASE_DIR / "plot_last_window.png")
    PLOT_TEST_RECON: str = str(BASE_DIR / "results.png")
    PLOT_MA_CURVES: str = str(BASE_DIR / "plot_ma_curves.png")
    BEST_PARAMS_JSON: str = str(BASE_DIR / "best_hyperparameters.json")
    
    # ===== 기타 설정 =====
    RECON_W_START: float = 2.0
    RECON_W_END: float = 0.5
    
    def get_effective_seq_len(self) -> int:
        """실제 사용할 시퀀스 길이 반환"""
        return self.DAILY_SEQ_LEN if self.USE_DAILY_DATA else self.SEQ_LEN
    
    def get_effective_pred_len(self) -> int:
        """실제 사용할 예측 길이 반환"""
        return self.DAILY_PRED_LEN if self.USE_DAILY_DATA else self.PRED_LEN
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'd_model': self.D_MODEL,
            'n_heads': self.N_HEADS,
            'enc_layers': self.ENC_LAYERS,
            'ff_dim': self.FF_DIM,
            'dropout': self.DROPOUT,
            'lr': self.LR,
            'weight_decay': self.WEIGHT_DECAY,
            'batch_size': self.BATCH_SIZE,
            'seq_len': self.get_effective_seq_len(),
            'pred_len': self.get_effective_pred_len(),
            'patch_len': self.PATCH_LEN
        }


# =========================
# 하이퍼파라미터 저장/로드
# =========================
def save_hyperparameters(params: Dict[str, Any], json_path: str = None) -> bool:
    """하이퍼파라미터를 JSON 파일에 저장"""
    if json_path is None:
        json_path = str(BASE_DIR / "best_hyperparameters.json")
    
    try:
        with open(json_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"✅ 하이퍼파라미터 저장 성공: {json_path}")
        return True
    except Exception as e:
        print(f"❌ JSON 파일 저장 실패: {e}")
        return False


def load_hyperparameters(json_path: str = None) -> Optional[Dict[str, Any]]:
    """JSON 파일에서 하이퍼파라미터 로드"""
    if json_path is None:
        json_path = str(BASE_DIR / "best_hyperparameters.json")
    
    if not os.path.exists(json_path):
        return None
    
    try:
        with open(json_path, 'r') as f:
            params = json.load(f)
        
        print(f"\n{'='*70}")
        print(f"✅ 저장된 하이퍼파라미터 로드 성공: {json_path}")
        print(f"{'='*70}")
        for key, value in params.items():
            print(f"   - {key}: {value}")
        print(f"{'='*70}\n")
        
        return params
    except Exception as e:
        print(f"⚠️ JSON 파일 로드 실패 ({json_path}): {e}")
        return None


# =========================
# 연령대 매핑
# =========================
AGE_GROUP_MAPPING = {
    '0세': ['0세'],
    '1-6세': ['1-6세'],
    '0-6세': ['0-6세'],
    '7-12세': ['7-12세'],
    '13-18세': ['13-18세'],
    '19-49세': ['19-49세'],
    '50-64세': ['50-64세'],
    '65세이상': ['65세이상', '65세 이상'],
    '65-69세': ['65-69세'],
    '70-74세': ['70-74세'],
    '75세이상': ['75세 이상', '75세이상'],
}


def normalize_age_group(age_str: str) -> str:
    """데이터셋의 연령대 표기를 표준화"""
    for standard, variants in AGE_GROUP_MAPPING.items():
        if age_str in variants:
            return standard
    return age_str


# =========================
# 컬럼 매핑
# =========================
COLUMN_MAPPING = {
    '연도': 'year',
    '주차': 'week',
    '의사환자 분율': 'ili',
    '예방접종률': 'vaccine_rate',
    '입원환자 수': 'hospitalization',
    '인플루엔자 검출률': 'detection_rate',
    '응급실 인플루엔자 환자': 'emergency_patients',
    '아형': 'subtype'
}


# 전역 Config 인스턴스
config = Config()
