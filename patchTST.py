import math
from pathlib import Path
from typing import List, Tuple, Optional
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# matplotlib 한글 폰트 설정 (macOS)
import platform
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:  # Linux
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.trial import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not installed. Hyperparameter optimization disabled.")
    print("   Install with: pip install optuna")

# PostgreSQL for efficient data loading
from database.db_utils import TimeSeriesDB, load_from_postgres

# =========================
# 환경 변수 로드
# =========================
print("=" * 60)
print("🔍 환경변수 로드")
print("=" * 60)

# .env 파일 경로 확인
env_path = Path.cwd() / '.env'
print(f"1. 현재 작업 디렉토리: {Path.cwd()}")
print(f"2. .env 파일 경로: {env_path}")
print(f"3. .env 파일 존재 여부: {env_path.exists()}")

# .env 파일 로드
load_result = load_dotenv(env_path, verbose=True, override=True)
print(f"4. .env 로드 결과: {load_result}")
print("=" * 60 + "\n")

# =========================
# Paths & device
# =========================
BASE_DIR = Path.cwd()

# CSV 파일 후보 경로
CANDIDATE_CSVS = [
    BASE_DIR / "data" / "merged" / "merged_influenza_data.csv",
    BASE_DIR / "merged_influenza_data.csv",
    BASE_DIR / "data" / "merged_influenza_data.csv",
]


# =========================
# 데이터 로딩 함수 (PostgreSQL)
# =========================
def load_data_from_postgres():
    """
    PostgreSQL에서 인플루엔자 데이터를 로드하는 함수
    Returns:
        pd.DataFrame: 로드된 데이터
    """
    print("\n📊 데이터 로드: PostgreSQL에서 데이터프레임으로 불러옵니다...")
    try:
        df = load_from_postgres(table_name="influenza_data")
        print(f"✅ PostgreSQL influenza_data 로드 완료: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ PostgreSQL 로드 실패: {e}")
        raise

def load_weather_data_from_postgres():
    """
    PostgreSQL에서 날씨 데이터를 로드하는 함수
    Returns:
        pd.DataFrame: 로드된 날씨 데이터 (year, week, min_temp, max_temp, avg_humidity)
    """
    print("\n🌡️  날씨 데이터 로드: PostgreSQL weather_data 테이블")
    try:
        db = TimeSeriesDB()
        db.connect()
        df_weather = db.load_data(table_name="weather_data")
        db.close()
        print(f"✅ PostgreSQL weather_data 로드 완료: {df_weather.shape}")
        print(f"   - 컬럼: {list(df_weather.columns)}")
        return df_weather
    except Exception as e:
        print(f"⚠️  날씨 데이터 로드 실패: {e}")
        print(f"   weather_data 테이블이 없거나 업로드되지 않았습니다.")
        return None

def merge_weather_with_influenza(df_influenza, df_weather):
    """
    인플루엔자 데이터와 날씨 데이터를 year, week 기준으로 병합
    
    Parameters:
        df_influenza: 인플루엔자 데이터
        df_weather: 날씨 데이터
    
    Returns:
        pd.DataFrame: 병합된 데이터
    """
    print(f"\n🔗 데이터 병합: influenza_data + weather_data")
    print(f"   - 병합 기준: year, week")
    
    # 수치형 컬럼 확인
    df_influenza['year'] = pd.to_numeric(df_influenza['year'], errors='coerce')
    df_influenza['week'] = pd.to_numeric(df_influenza['week'], errors='coerce')
    df_weather['year'] = pd.to_numeric(df_weather['year'], errors='coerce')
    df_weather['week'] = pd.to_numeric(df_weather['week'], errors='coerce')
    
    # LEFT JOIN (influenza_data 기준)
    df_merged = pd.merge(
        df_influenza,
        df_weather,
        on=['year', 'week'],
        how='left'
    )
    
    print(f"   ✅ 병합 완료:")
    print(f"      - influenza_data 행 수: {len(df_influenza)}")
    print(f"      - weather_data 행 수: {len(df_weather)}")
    print(f"      - 병합 후 행 수: {len(df_merged)}")
    
    # 새로 추가된 컬럼 확인
    new_cols = [c for c in df_weather.columns if c not in df_influenza.columns and c not in ['year', 'week']]
    if new_cols:
        print(f"      - 추가된 날씨 컬럼: {new_cols}")
    
    return df_merged

def pick_csv_path():
    for p in CANDIDATE_CSVS:
        if p.exists():
            return p
    raise FileNotFoundError("No input CSV found among:\n" + "\n".join(map(str, CANDIDATE_CSVS)))



def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = pick_device()
SEED   = 42

print(f"🖥️ 선택된 디바이스: {DEVICE}")
print(f"🎲 랜덤 시드: {SEED}\n")


# =========================
# Configuration - 모든 설정을 여기서 관리
# =========================

class Config:
    """모델 설정 통합 관리"""
    
    # ===== Optuna 최적화 설정 =====
    USE_OPTUNA = False       # Optuna 최적화 실행
    N_TRIALS = 50          # Optuna 최적화 시도 횟수
    OPTUNA_TIMEOUT = None   # 최적화 시간 제한 (초), None이면 무제한
    
    # Optuna 최적화 범위 (USE_OPTUNA=True일 때 사용)
    OPTUNA_SEARCH_SPACE = {
        'd_model': [64, 128, 256],  # n_heads의 배수로 설정
        'n_heads': [2, 4, 8, 16],       # Attention head 개수
        'enc_layers': (2, 8),       # Encoder 레이어 개수 (범위 확장)
        'ff_dim': [64, 96, 128, 192, 256, 384, 512],  # Feed-forward 차원 (더 많은 값 추가)
        'dropout': (0.05, 0.5),                       # Dropout 비율 (범위 확장)
        'lr': (1e-6, 1e-2),                           # Learning rate (범위 확장, log scale)
        'weight_decay': (1e-6, 1e-2),                 # Weight decay (범위 확장, log scale)
        'batch_size': [16, 32, 48, 64, 96, 128],      # Batch size (더 세밀한 값 추가)
        'seq_len': (8, 30),       # Input sequence length (세밀화)
        # pred_len은 Config.PRED_LEN 사용 (Optuna에서 제외)
        'patch_len': [2, 3, 4, 5, 6],                 # Patch length (범위 확장)
    }
    
    # ===== 모델 하이퍼파라미터 (기본값) =====
    # Optuna를 사용하지 않을 때 또는 최적화 후 고정값으로 사용
    EPOCHS = 200
    BATCH_SIZE = 64
    SEQ_LEN = 16            # 입력 시퀀스 길이 (과거 몇 주)
    PRED_LEN = 4            # 예측 길이 (미래 몇 주) — 기본: 4주(한 달)
    PATCH_LEN = 4           # CNN 패치 길이
    STRIDE = 1              # 패치 스트라이드
    
    # 모델 아키텍처
    D_MODEL = 128           # 모델 차원 (4의 배수 필수)
    N_HEADS = 2             # Attention head 개수
    ENC_LAYERS = 4          # Encoder 레이어 개수
    FF_DIM = 128            # Feed-forward 차원
    DROPOUT = 0.3           # Dropout 비율
    HEAD_HIDDEN = [64, 64]  # Prediction head hidden layers
    
    # ===== 학습 설정 =====
    LR = 5e-4               # Learning rate
    WEIGHT_DECAY = 5e-4     # Weight decay (L2 regularization)
    PATIENCE = 60           # Early stopping patience
    WARMUP_EPOCHS = 30      # Learning rate warmup epochs
    
    # ===== Loss 함수 설정 =====
    PEAK_THRESHOLD_QUANTILE = 0.85  # 피크 기준 (상위 15% - 더 높은 피크만 집중)
    PEAK_WEIGHT_ALPHA = 12.0        # 피크 구간 가중치 (8.0 → 12.0으로 상향, peak 언더슈팅 감소)
    AMPLITUDE_WEIGHT_BETA = 0.6     # 진폭 보존 항 가중치 (0.3 → 0.6, 3~4주 후 예측값 상향)
    
    # Horizon Weighting (예측 구간별 가중치)
    HORIZON_WEIGHT_MODE = "exponential"  # "exponential", "tail_boost", "uniform"
    HORIZON_EXP_SCALE = 2.0              # exponential 모드 스케일 (1.2 → 2.0)
    HORIZON_TAIL_BOOST = 2.5             # tail_boost 모드: 뒤쪽 가중치 배수
    HORIZON_TAIL_COUNT = 2               # tail_boost 모드: 뒤쪽 몇 개
    
    # ===== 데이터 설정 =====
    TRAIN_RATIO = 0.7       # Train 데이터 비율
    VAL_RATIO = 0.15        # Validation 데이터 비율 (Test = 1 - TRAIN - VAL)
    SCALER_TYPE = "robust"  # Scaler 타입: "standard", "robust", "minmax"
    
    # Log 변환 설정 (피크 예측 향상)
    USE_LOG_TRANSFORM = True  # 타겟 변수에 log(1+x) 변환 적용
    LOG_EPSILON = 0.000001         # log(x + epsilon)의 epsilon 값
    
    # 외생 특징 사용 모드
    # "auto": 자동 감지, "none": 사용 안함, "vax": 백신률만, 
    # "resp": 호흡기지수만, "both": 둘 다, "all": 모든 특징
    USE_EXOG = "all"
    INCLUDE_SEASONAL_FEATS = True  # week_sin 포함 여부
    
    # ===== 연령대별 동학 설정 =====
    USE_AGE_GROUP_DYNAMICS = False  # 어린이 집단 ILI를 외생 변수로 사용 (현재 비활성화)
    # 주의: "0-6세"는 ILI 데이터가 없음! "0세"와 "1-6세"로 분리되어 있음
    LEAD_AGE_GROUPS = ["0세", "1-6세", "7-12세"]  # 선행 지표 연령대 (유행이 먼저 시작)
    
    # ===== 피처 제외 설정 =====
    EXCLUDE_HOSPITALIZATION = True  # hospitalization 피처 제외 여부
    
    # ===== 일별 데이터 변환 설정 =====
    USE_DAILY_DATA = True              # 주차별 → 일별 데이터 변환 여부
    DAILY_INTERP_METHOD = "linear"     # 일별 데이터 보간 : "gaussian" 또는 "linear"
    GAUSSIAN_STD = 1.0                 # 바우시안 커널 표준편차
    DAILY_SEQ_LEN = 112                # 일별 입력 길이 (약 16주)
    DAILY_PRED_LEN = 28                # 일별 예측 길이 (약 4주)
    
    # ===== 트렌드 데이터 설정 (Google, Naver, Twitter) =====
    # TODO: API가 메타데이터만 반환하는 문제 해결 후 True로 변경
    USE_TRENDS_DATA = False  # 트렌드 데이터 사용 여부 (현재 비활성화)
    TRENDS_DB_NAME = "trends"  # PostgreSQL 트렌드 데이터베이스 이름
    TRENDS_TABLE_NAME = "trends_data"  # 트렌드 데이터 테이블 이름
    
    # ===== 출력 설정 =====
    OUT_CSV = str(BASE_DIR / "ili_predictions.csv")
    PLOT_LAST_WINDOW = str(BASE_DIR / "plot_last_window.png")
    PLOT_TEST_RECON = str(BASE_DIR / "results.png")
    PLOT_MA_CURVES = str(BASE_DIR / "plot_ma_curves.png")
    BEST_PARAMS_JSON = str(BASE_DIR / "best_hyperparameters.json")
    
    # ===== 기타 설정 =====
    RECON_W_START = 2.0     # Overlap 재구성 시작 가중치
    RECON_W_END = 0.5       # Overlap 재구성 끝 가중치

# 전역 변수로 설정 (하위 호환성)
USE_OPTUNA = Config.USE_OPTUNA
N_TRIALS = Config.N_TRIALS

EPOCHS = Config.EPOCHS
BATCH_SIZE = Config.BATCH_SIZE
SEQ_LEN = Config.SEQ_LEN
PRED_LEN = Config.PRED_LEN
PATCH_LEN = Config.PATCH_LEN
STRIDE = Config.STRIDE

D_MODEL = Config.D_MODEL
N_HEADS = Config.N_HEADS
ENC_LAYERS = Config.ENC_LAYERS
FF_DIM = Config.FF_DIM
DROPOUT = Config.DROPOUT
HEAD_HIDDEN = Config.HEAD_HIDDEN

LR = Config.LR
WEIGHT_DECAY = Config.WEIGHT_DECAY
PATIENCE = Config.PATIENCE
WARMUP_EPOCHS = Config.WARMUP_EPOCHS

SCALER_TYPE = Config.SCALER_TYPE
USE_EXOG = Config.USE_EXOG
INCLUDE_SEASONAL_FEATS = Config.INCLUDE_SEASONAL_FEATS

OUT_CSV = Config.OUT_CSV
PLOT_LAST_WINDOW = Config.PLOT_LAST_WINDOW
PLOT_TEST_RECON = Config.PLOT_TEST_RECON
PLOT_MA_CURVES = Config.PLOT_MA_CURVES

RECON_W_START = Config.RECON_W_START
RECON_W_END = Config.RECON_W_END

# =========================
# utils
# =========================
from datetime import date

def _iso_weeks_in_year(y: int) -> int:
    # ISO 달력의 마지막 주 번호(52 또는 53)
    return date(y, 12, 28).isocalendar().week

def weekly_to_daily_interp_gaussian(
    df: pd.DataFrame,
    season_col: str = "season_norm",
    week_col: str = "week",
    target_col: str = "ili",
    method: str = "gaussian",
    gaussian_std: float = 1.0,
) -> pd.DataFrame:
    """
    주 단위 데이터를 일 단위로 확장(바우시안 또는 선형보간).
    
    Parameters:
        df: 주차별 데이터프레임
        season_col: 시즌 컬럼명
        week_col: 주차 컬럼명
        target_col: 타겟 컬럼명
        method: 보간 방법 ("gaussian" 또는 "linear")
        gaussian_std: 바우시안 커널 표준편차 (method="gaussian"일 때)
        
    Returns:
        date 컬럼 포함한 일 단위 DF
    """
    df = df.copy()
    df.columns = df.columns.str.replace("\ufeff", "", regex=True).str.strip()

    # --- 시즌/주차 확보 ---
    has_season = season_col in df.columns
    has_week   = week_col in df.columns

    if not (has_season and has_week):
        # label에서 시즌/주차 추출 시도: "2024-2025 season - W29"
        if "label" in df.columns:
            import re
            def _parse_label(lbl):
                m = re.search(r"(\d{4}-\d{4}).*W\s*([0-9]+)", str(lbl))
                if m:
                    return m.group(1), int(m.group(2))
                return None
            parsed = df["label"].map(_parse_label)
            if not has_season:
                df[season_col] = [p[0] if p else np.nan for p in parsed]
                has_season = True
            if not has_week:
                df[week_col] = [p[1] if p else np.nan for p in parsed]
                has_week = True

    # 최후의 수단: season_norm이 없으면 단일 시즌으로, week 없으면 1..N
    if not has_season:
        # 첫 행의 연도를 찾아 대체 시즌명 만들기
        # 없으면 "0000-0001"
        first_year = None
        if "date" in df.columns:
            try:
                first_year = pd.to_datetime(df["date"]).dt.year.min()
            except Exception:
                pass
        if first_year is None:
            first_year = pd.Timestamp.today().year
        df[season_col] = f"{first_year}-{first_year+1}"
        has_season = True

    if not has_week:
        df[week_col] = np.arange(1, len(df) + 1, dtype=int)
        has_week = True

    # 숫자화
    df[week_col] = pd.to_numeric(df[week_col], errors="coerce")
    # 시즌 문자열 정규화
    def _norm_season_text_local(s: str) -> str:
        ss = str(s).replace("절기", "")
        import re
        m = re.search(r"(\d{4})\s*-\s*(\d{4})", ss)
        return f"{m.group(1)}-{m.group(2)}" if m else ss.strip()
    df[season_col] = df[season_col].astype(str).map(_norm_season_text_local)

    # --- ISO 주 시작일 산출 (시즌 규칙 반영) ---
    week_starts = []
    for _, row in df.iterrows():
        season = str(row[season_col])
        try:
            y0 = int(season.split("-")[0])
        except Exception:
            y0 = pd.Timestamp.today().year
        wk = int(row[week_col]) if not pd.isna(row[week_col]) else 1
        iso_year = y0 if wk >= 36 else (y0 + 1)
        # 해당 ISO년의 실제 마지막 주 넘지 않도록 보정
        wk = min(max(1, wk), _iso_weeks_in_year(iso_year))
        # 월요일(1) 기준 주 시작일
        week_starts.append(pd.Timestamp.fromisocalendar(iso_year, wk, 1))
    df["week_start"] = week_starts

    # --- 중복 week_start 처리: 수치=mean, 비수치=first ---
    if df["week_start"].duplicated().any():
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        agg = {c: "mean" for c in num_cols}
        # 비수치 컬럼(라벨/시즌 등)은 첫 값 유지
        for c in df.columns:
            if c not in num_cols and c != "week_start":
                agg[c] = "first"
        df = df.groupby("week_start", as_index=False).agg(agg)

    # --- 일 단위 리샘플 ---
    df = df.set_index("week_start").sort_index()
    df_daily = df.resample("D").asfreq()

    # 수치형 보간
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if method.lower() == "gaussian":
        # 🔴 바우시안 보간법 (Gaussian Interpolation)
        from scipy.ndimage import gaussian_filter1d
        
        for c in num_cols:
            # 원본 주차별 데이터
            valid_mask = df[c].notna()
            if valid_mask.sum() < 2:
                # 데이터가 2개 미만이면 선형보간
                df_daily[c] = df_daily[c].interpolate(method="linear", limit_direction="both")
                continue
            
            # 먼저 선형보간으로 NaN 채우기
            temp = df_daily[c].interpolate(method="linear", limit_direction="both")
            
            # 바우시안 필터 적용 (평활 효과)
            if temp.notna().sum() > 0:
                values = temp.fillna(temp.mean()).values
                smoothed = gaussian_filter1d(values, sigma=gaussian_std)
                df_daily[c] = smoothed
            else:
                df_daily[c] = temp
    else:
        # 선형보간 (기존 방식)
        for c in num_cols:
            df_daily[c] = df_daily[c].interpolate(method="linear", limit_direction="both")

    # 범주형은 앞뒤 채움
    cat_cols = [c for c in df.columns if c not in num_cols]
    for c in cat_cols:
        df_daily[c] = df_daily[c].ffill().bfill()

    # 결과
    out = df_daily.reset_index().rename(columns={"week_start": "date"})
    # date는 datetime으로 강제
    out["date"] = pd.to_datetime(out["date"])
    
    print(f"\n✅ 일별 데이터 변환 완료 ({method.upper()} 보간법):")
    print(f"   - 입력: {len(df)} 주(week)")
    print(f"   - 출력: {len(out)} 일(day) → {len(out)/7:.1f}배 확대")
    print(f"   - 날짜 범위: {out['date'].min().date()} ~ {out['date'].max().date()}")
    
    return out
    
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def make_splits(n: int, train_ratio=None, val_ratio=None):
    """데이터 분할 (train/val/test)"""
    if train_ratio is None:
        train_ratio = Config.TRAIN_RATIO
    if val_ratio is None:
        val_ratio = Config.VAL_RATIO
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return (0, n_train), (n_train, n_train+n_val), (n_train+n_val, n)

class LogTransformScaler:
    """
    Log 변환을 적용하는 Scaler
    피크 예측 향상을 위해 log(1+x) 변환 후 스케일링
    """
    def __init__(self, base_scaler=None, epsilon=1.0):
        self.base_scaler = base_scaler or RobustScaler()
        self.epsilon = epsilon  # log(x + epsilon)
        
    def fit(self, X):
        # Log 변환 후 scaler fit
        X_log = np.log(X + self.epsilon)
        self.base_scaler.fit(X_log)
        return self
    
    def transform(self, X):
        # Log 변환 후 scaler transform
        X_log = np.log(X + self.epsilon)
        return self.base_scaler.transform(X_log)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        # scaler inverse 후 exp 변환
        X_log = self.base_scaler.inverse_transform(X_scaled)
        return np.exp(X_log) - self.epsilon


def get_scaler(name=None, for_target=False):
    """
    Scaler 생성 함수
    
    Args:
        name: scaler 타입 ("robust", "minmax", "standard")
        for_target: True이면 타겟 변수용 (Log 변환 적용 가능), False이면 피처용
    """
    s = (name or Config.SCALER_TYPE).lower()
    
    # Log 변환은 타겟 변수에만 적용
    if for_target and Config.USE_LOG_TRANSFORM:
        # Log 변환 + 기본 scaler
        if s == "robust":
            base = RobustScaler()
        elif s == "minmax":
            base = MinMaxScaler()
        else:
            base = StandardScaler()
        return LogTransformScaler(base_scaler=base, epsilon=Config.LOG_EPSILON)
    else:
        # 기존 scaler (피처 또는 Log 변환 미사용)
        if s == "robust":
            return RobustScaler()
        elif s == "minmax":
            return MinMaxScaler()
        else:
            return StandardScaler()

def _norm_season_text(s: str) -> str:
    ss = str(s).replace("절기", "")
    import re
    m = re.search(r"(\d{4})\s*-\s*(\d{4})", ss)
    return f"{m.group(1)}-{m.group(2)}" if m else ss.strip()


# =========================
# 연령대 매핑 및 데이터 로드 유틸리티
# =========================

# 연령대 그룹 정의 (데이터셋마다 연령대 표기가 다름)
# 주의: '0-6세'는 합계 연령대로 ILI 데이터가 없음! '0세'와 '1-6세'를 각각 사용해야 함
AGE_GROUP_MAPPING = {
    # 표준화된 연령대 이름 -> 각 데이터셋에서 사용되는 이름들
    '0세': ['0세'],           # 영아 - ILI 있음 (선행 지표)
    '1-6세': ['1-6세'],       # 유아 - ILI 있음 (선행 지표)
    '0-6세': ['0-6세'],       # 합계 연령대 - ILI 없음! (사용 불가)
    '7-12세': ['7-12세'],     # 초등학생 - ILI 있음 (선행 지표)
    '13-18세': ['13-18세'],
    '19-49세': ['19-49세'],
    '50-64세': ['50-64세'],
    '65세이상': ['65세이상', '65세 이상'],
    '65-69세': ['65-69세'],
    '70-74세': ['70-74세'],
    '75세이상': ['75세 이상', '75세이상'],
}

# 역방향 매핑: 데이터셋의 연령대 -> 표준화된 연령대
def normalize_age_group(age_str: str) -> str:
    """데이터셋의 연령대 표기를 표준화"""
    for standard, variants in AGE_GROUP_MAPPING.items():
        if age_str in variants:
            return standard
    return age_str  # 매핑이 없으면 원본 반환


# =========================
# 데이터 소스 비교 검증 함수
# =========================
def validate_data_sources(
    age_group: str = "19-49세",
    data_dir: str = "data/before",
    merged_csv_path: str = "merged_influenza_data.csv",
    verbose: bool = True
) -> dict:
    """
    merged_influenza_data.csv와 원본 CSV(data/before)에서 
    동일한 방식으로 필터링한 데이터를 비교하여 일관성 검증
    
    Parameters:
        age_group: 비교할 연령대
        data_dir: 원본 CSV 디렉토리
        merged_csv_path: 병합된 CSV 파일 경로
        verbose: 상세 출력 여부
    
    Returns:
        dict: 비교 결과 {'match': bool, 'details': {...}}
    """
    from pathlib import Path
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🔍 데이터 소스 비교 검증: {age_group}")
        print(f"{'='*70}")
    
    results = {
        'age_group': age_group,
        'match': False,
        'details': {}
    }
    
    # ===== 1. merged_influenza_data.csv에서 필터링 =====
    merged_path = Path(merged_csv_path)
    if not merged_path.exists():
        if verbose:
            print(f"   ⚠️ {merged_csv_path} 파일을 찾을 수 없습니다.")
        results['details']['merged_error'] = 'File not found'
        return results
    
    try:
        df_merged_all = pd.read_csv(merged_csv_path)
        
        # 연령대 변형 목록
        age_variants = AGE_GROUP_MAPPING.get(age_group, [age_group])
        
        # 필터링 (merged CSV는 age_group 컬럼 사용)
        mask = df_merged_all['age_group'].isin(age_variants)
        df_merged = df_merged_all[mask].copy()
        
        # 여러 변형이 있는 연령대는 year, week 기준으로 그룹화 필요
        # 0-6세: 0세 + 1-6세, 65세이상: 65세이상 + 65세 이상 등
        if len(age_variants) > 1 and len(df_merged) > 0:
            # 중복 (year, week) 조합이 있는지 확인
            dup_count = df_merged.duplicated(subset=['year', 'week'], keep=False).sum()
            if dup_count > 0:
                # year, week 기준으로 그룹화
                agg_dict = {}
                for col in df_merged.columns:
                    if col in ['year', 'week', 'age_group', 'subtype']:
                        continue  # 그룹화/문자열 컬럼 제외
                    elif col in ['hospitalization', 'emergency_patients']:
                        agg_dict[col] = 'sum'  # 입원/응급실은 합산
                    elif df_merged[col].dtype in ['float64', 'int64']:
                        agg_dict[col] = 'mean'  # 숫자형만 평균
                
                df_merged = df_merged.groupby(['year', 'week'], as_index=False).agg(agg_dict)
                df_merged['age_group'] = age_group
        
        # 정렬
        df_merged = df_merged.sort_values(['year', 'week']).reset_index(drop=True)
        
        if verbose:
            print(f"\n📊 소스 1: merged_influenza_data.csv")
            print(f"   - 필터 조건: age_group in {age_variants}")
            print(f"   - 결과 행 수: {len(df_merged)}")
            print(f"   - 연도 범위: {df_merged['year'].min():.0f} ~ {df_merged['year'].max():.0f}")
            print(f"   - ILI 범위: {df_merged['ili'].min():.2f} ~ {df_merged['ili'].max():.2f}" if df_merged['ili'].notna().any() else "   - ILI: 모두 결측")
        
        results['details']['merged'] = {
            'rows': len(df_merged),
            'year_range': (int(df_merged['year'].min()), int(df_merged['year'].max())),
            'ili_range': (float(df_merged['ili'].min()), float(df_merged['ili'].max())) if df_merged['ili'].notna().any() else None,
            'nulls': int(df_merged.isnull().sum().sum())
        }
        
    except Exception as e:
        if verbose:
            print(f"   ❌ merged CSV 로드 오류: {e}")
        results['details']['merged_error'] = str(e)
        return results
    
    # ===== 2. 원본 CSV(data/before)에서 필터링 =====
    try:
        df_raw = load_raw_data_by_age_group(data_dir=data_dir, age_group=age_group)
        df_raw = df_raw.sort_values(['year', 'week']).reset_index(drop=True)
        
        if verbose:
            print(f"\n📊 소스 2: data/before (원본 CSV)")
            print(f"   - 함수: load_raw_data_by_age_group('{age_group}')")
            print(f"   - 결과 행 수: {len(df_raw)}")
            print(f"   - 연도 범위: {df_raw['year'].min():.0f} ~ {df_raw['year'].max():.0f}")
            print(f"   - ILI 범위: {df_raw['ili'].min():.2f} ~ {df_raw['ili'].max():.2f}" if df_raw['ili'].notna().any() else "   - ILI: 모두 결측")
        
        results['details']['raw'] = {
            'rows': len(df_raw),
            'year_range': (int(df_raw['year'].min()), int(df_raw['year'].max())),
            'ili_range': (float(df_raw['ili'].min()), float(df_raw['ili'].max())) if df_raw['ili'].notna().any() else None,
            'nulls': int(df_raw.isnull().sum().sum())
        }
        
    except Exception as e:
        if verbose:
            print(f"   ❌ 원본 CSV 로드 오류: {e}")
        results['details']['raw_error'] = str(e)
        return results
    
    # ===== 3. 비교 =====
    if verbose:
        print(f"\n📊 비교 결과:")
    
    # 행 수 비교
    row_match = len(df_merged) == len(df_raw)
    if verbose:
        print(f"   - 행 수 일치: {'✅' if row_match else '❌'} (merged: {len(df_merged)}, raw: {len(df_raw)})")
    
    # ILI 값 비교 (공통 year/week 기준)
    common_keys = set(zip(df_merged['year'], df_merged['week'])) & set(zip(df_raw['year'], df_raw['week']))
    
    if common_keys:
        # 공통 키로 병합
        df_merged_subset = df_merged[df_merged.apply(lambda r: (r['year'], r['week']) in common_keys, axis=1)].copy()
        df_raw_subset = df_raw[df_raw.apply(lambda r: (r['year'], r['week']) in common_keys, axis=1)].copy()
        
        df_merged_subset = df_merged_subset.sort_values(['year', 'week']).reset_index(drop=True)
        df_raw_subset = df_raw_subset.sort_values(['year', 'week']).reset_index(drop=True)
        
        # ILI 비교
        ili_merged = df_merged_subset['ili'].fillna(0).values
        ili_raw = df_raw_subset['ili'].fillna(0).values
        
        if len(ili_merged) == len(ili_raw):
            ili_diff = np.abs(ili_merged - ili_raw)
            ili_match = np.allclose(ili_merged, ili_raw, rtol=1e-5, atol=1e-8, equal_nan=True)
            ili_max_diff = ili_diff.max()
            ili_mean_diff = ili_diff.mean()
            
            if verbose:
                print(f"   - ILI 값 일치: {'✅' if ili_match else '⚠️'} (최대 차이: {ili_max_diff:.6f}, 평균 차이: {ili_mean_diff:.6f})")
            
            results['details']['ili_comparison'] = {
                'match': bool(ili_match),
                'max_diff': float(ili_max_diff),
                'mean_diff': float(ili_mean_diff)
            }
        else:
            if verbose:
                print(f"   - ILI 비교 불가: 행 수 불일치")
            ili_match = False
    else:
        if verbose:
            print(f"   - 공통 키 없음")
        ili_match = False
    
    # 전체 일치 여부
    results['match'] = row_match and (ili_match if common_keys else False)
    
    if verbose:
        print(f"\n   📋 최종 결과: {'✅ 일치' if results['match'] else '⚠️ 불일치 (차이 있음)'}")
        print(f"{'='*70}")
    
    return results


def validate_all_age_groups(
    data_dir: str = "data/before",
    merged_csv_path: str = "merged_influenza_data.csv"
) -> dict:
    """
    모든 주요 연령대에 대해 데이터 소스 비교 검증 실행
    
    Returns:
        dict: 연령대별 검증 결과
    """
    print(f"\n{'🔬 '*20}")
    print("모든 연령대 데이터 소스 비교 검증")
    print(f"{'🔬 '*20}")
    
    age_groups = ['0-6세', '7-12세', '13-18세', '19-49세', '50-64세', '65세이상']
    all_results = {}
    
    for age in age_groups:
        result = validate_data_sources(
            age_group=age,
            data_dir=data_dir,
            merged_csv_path=merged_csv_path,
            verbose=True
        )
        all_results[age] = result
    
    # 요약
    print(f"\n{'='*70}")
    print("📋 전체 검증 요약")
    print(f"{'='*70}")
    
    for age, result in all_results.items():
        status = '✅' if result['match'] else '⚠️'
        details = result.get('details', {})
        merged_rows = details.get('merged', {}).get('rows', 'N/A')
        raw_rows = details.get('raw', {}).get('rows', 'N/A')
        print(f"   {status} {age}: merged={merged_rows}행, raw={raw_rows}행")
    
    return all_results


def load_raw_data_by_age_group(
    data_dir: str = "data/before",
    age_group: str = "19-49세"
) -> pd.DataFrame:
    """
    특정 연령대의 모든 데이터를 data/before 디렉토리에서 직접 로드
    PostgreSQL을 거치지 않고 원본 CSV에서 직접 로드
    
    Parameters:
        data_dir: 데이터 디렉토리 경로
        age_group: 선택할 연령대 (예: '19-49세', '65세이상')
    
    Returns:
        pd.DataFrame: 해당 연령대의 병합된 데이터
    """
    from pathlib import Path
    
    data_path = Path(data_dir)
    
    print(f"\n{'='*60}")
    print(f"📂 연령대별 원본 데이터 로드: {age_group}")
    print(f"{'='*60}")
    
    # 연령대 변형 목록
    age_variants = AGE_GROUP_MAPPING.get(age_group, [age_group])
    print(f"   - 검색할 연령대 변형: {age_variants}")
    
    # 데이터셋별 로드
    # has_age: True = 연령대 필터링 필수, False = 전국 데이터 (연령대 없음)
    # fallback_to_avg: True = 연령대 데이터 없으면 전국 평균 사용
    datasets = {
        'ds_0101': {'col': '의사환자 분율', 'target': 'ili', 'has_age': True, 'fallback_to_avg': False},
        'ds_0103': {'col': '입원환자 수', 'target': 'hospitalization_confirmed', 'has_age': True, 'fallback_to_avg': False},
        'ds_0104': {'col': '입원환자 수', 'target': 'hospitalization_suspected', 'has_age': True, 'fallback_to_avg': False},
        'ds_0106': {'col': '인플루엔자 검출률', 'target': 'detection_rate', 'has_age': True, 'fallback_to_avg': False},
        'ds_0108': {'col': '인플루엔자 검출률', 'target': 'detection_rate_alt', 'has_age': True, 'fallback_to_avg': False},
        'ds_0109': {'col': '응급실 인플루엔자 환자', 'target': 'emergency_patients', 'has_age': True, 'fallback_to_avg': False},
        'ds_0110': {'col': '예방접종률', 'target': 'vaccine_rate', 'has_age': True, 'fallback_to_avg': True},
    }
    
    all_data = {}
    
    for dsid, info in datasets.items():
        ds_num = dsid.replace('ds_', '')
        pattern = f"flu-{ds_num}-*.csv"
        files = list(data_path.glob(pattern))
        
        if not files:
            continue
        
        dfs = []
        for f in sorted(files):
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                print(f"   ⚠️ 파일 읽기 실패 ({f.name}): {e}")
        
        if not dfs:
            continue
        
        df_combined = pd.concat(dfs, ignore_index=True)
        
        # 연령대 필터링
        if info['has_age'] and '연령대' in df_combined.columns:
            # 해당 연령대만 필터링
            mask = df_combined['연령대'].isin(age_variants)
            df_filtered = df_combined[mask].copy()
            
            # 연령대 데이터가 없고 fallback_to_avg가 True인 경우 전국 평균 사용
            if df_filtered.empty and info.get('fallback_to_avg', False):
                print(f"   - {dsid}: 연령대 '{age_group}' 데이터 없음 → 전국 평균 사용")
                
                # 컬럼 표준화
                df_combined = df_combined.rename(columns={
                    '연도': 'year',
                    '주차': 'week',
                    info['col']: info['target']
                })
                
                # 수치형 변환
                df_combined[info['target']] = pd.to_numeric(df_combined[info['target']], errors='coerce')
                
                # 주차별 전국 평균 계산
                df_filtered = df_combined.groupby(['year', 'week'], as_index=False)[info['target']].mean()
                
                all_data[info['target']] = df_filtered
                print(f"   - {dsid} ({info['target']}): {len(df_filtered)}행 로드 (전국 평균)")
                continue
            
            if df_filtered.empty:
                print(f"   - {dsid}: 연령대 '{age_group}' 데이터 없음")
                continue
            
            # 컬럼 표준화
            df_filtered = df_filtered.rename(columns={
                '연도': 'year',
                '주차': 'week',
                '연령대': 'age_group',
                info['col']: info['target']
            })
            
            # 필요한 컬럼만 선택
            cols = ['year', 'week', info['target']]
            df_filtered = df_filtered[cols].copy()
            
            # 여러 연령대 변형이 있을 경우 (예: 0-6세 = 0세 + 1-6세) 합산
            if len(age_variants) > 1:
                # 수치형으로 변환
                df_filtered[info['target']] = pd.to_numeric(df_filtered[info['target']], errors='coerce')
                # year, week 기준으로 합산 (입원환자, 응급실) 또는 평균 (ILI, 검출률)
                if info['target'] in ['hospitalization_confirmed', 'hospitalization_suspected', 'emergency_patients']:
                    df_filtered = df_filtered.groupby(['year', 'week'], as_index=False)[info['target']].sum()
                else:
                    df_filtered = df_filtered.groupby(['year', 'week'], as_index=False)[info['target']].mean()
            
            all_data[info['target']] = df_filtered
            print(f"   - {dsid} ({info['target']}): {len(df_filtered)}행 로드")
    
    if not all_data:
        print(f"\n⚠️ 연령대 '{age_group}'에 해당하는 데이터가 없습니다.")
        return pd.DataFrame()
    
    # 모든 데이터 병합 (year, week 기준)
    print(f"\n📊 데이터 병합 중...")
    
    # 첫 번째 데이터프레임을 기준으로 시작
    result_df = None
    for target_name, df in all_data.items():
        if result_df is None:
            result_df = df.copy()
        else:
            # year, week 기준으로 병합
            result_df = pd.merge(result_df, df, on=['year', 'week'], how='outer')
    
    # 정렬
    result_df = result_df.sort_values(['year', 'week']).reset_index(drop=True)
    
    # hospitalization 합산 (확진 + 의심)
    if 'hospitalization_confirmed' in result_df.columns or 'hospitalization_suspected' in result_df.columns:
        confirmed = result_df.get('hospitalization_confirmed', 0).fillna(0)
        suspected = result_df.get('hospitalization_suspected', 0).fillna(0)
        result_df['hospitalization'] = confirmed + suspected
        
        # 원본 컬럼 제거
        for col in ['hospitalization_confirmed', 'hospitalization_suspected']:
            if col in result_df.columns:
                result_df = result_df.drop(columns=[col])
    
    # detection_rate 통합 (ds_0106과 ds_0108 중 하나 선택)
    if 'detection_rate' in result_df.columns and 'detection_rate_alt' in result_df.columns:
        # 우선 ds_0106 사용, 없으면 ds_0108
        result_df['detection_rate'] = result_df['detection_rate'].fillna(result_df['detection_rate_alt'])
        result_df = result_df.drop(columns=['detection_rate_alt'])
    elif 'detection_rate_alt' in result_df.columns:
        result_df = result_df.rename(columns={'detection_rate_alt': 'detection_rate'})
    
    # 연령대 컬럼 추가
    result_df['age_group'] = age_group
    
    print(f"\n✅ 연령대 '{age_group}' 데이터 로드 완료:")
    print(f"   - 총 {len(result_df)}행")
    print(f"   - 컬럼: {list(result_df.columns)}")
    print(f"   - 연도 범위: {result_df['year'].min():.0f} ~ {result_df['year'].max():.0f}")
    print(f"   - 주차 범위: {result_df['week'].min():.0f} ~ {result_df['week'].max():.0f}")
    
    return result_df


def get_available_age_groups(data_dir: str = "data/before") -> dict:
    """
    data/before 디렉토리에서 사용 가능한 연령대 목록 조회
    
    Returns:
        dict: 데이터셋별 연령대 목록
    """
    from pathlib import Path
    
    data_path = Path(data_dir)
    result = {}
    
    # 주요 데이터셋 확인
    datasets = ['0101', '0103', '0106', '0108', '0109', '0110']
    
    for ds in datasets:
        pattern = f"flu-{ds}-*.csv"
        files = list(data_path.glob(pattern))
        
        if not files:
            continue
        
        age_groups = set()
        for f in files:
            try:
                df = pd.read_csv(f)
                if '연령대' in df.columns:
                    age_groups.update(df['연령대'].dropna().unique())
            except:
                pass
        
        if age_groups:
            result[f'ds_{ds}'] = sorted(list(age_groups))
    
    return result


# =========================
# 아형별 데이터 로드 함수 (ds_0107)
# =========================
def load_subtype_data(data_dir: str = "data/before", subtype: str = "A") -> pd.DataFrame:
    """
    ds_0107 데이터에서 특정 아형(A/B)의 검출률 데이터를 로드
    
    Parameters:
        data_dir: 데이터 디렉토리 경로
        subtype: 아형 ('A', 'B', 또는 'all')
    
    Returns:
        pd.DataFrame: 아형별 검출률 데이터 (연도, 주차, 검출률)
    """
    from pathlib import Path
    
    data_path = Path(data_dir)
    flu_0107_files = list(data_path.glob("flu-0107-*.csv"))
    
    if not flu_0107_files:
        print(f"⚠️ ds_0107 파일을 찾을 수 없습니다: {data_dir}")
        return pd.DataFrame()
    
    print(f"\n📊 아형별 검출률 데이터 로드 (ds_0107)")
    print(f"   - 발견된 파일: {len(flu_0107_files)}개")
    print(f"   - 선택된 아형: {subtype}")
    
    all_dfs = []
    for filepath in sorted(flu_0107_files):
        try:
            df = pd.read_csv(filepath)
            all_dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ 파일 읽기 실패 ({filepath.name}): {e}")
    
    if not all_dfs:
        return pd.DataFrame()
    
    df_combined = pd.concat(all_dfs, ignore_index=True)
    
    # 컬럼명 매핑
    col_map = {
        '연도': 'year',
        '주차': 'week',
        '아형': 'subtype',
        '인플루엔자 검출률': 'detection_rate'
    }
    df_combined = df_combined.rename(columns=col_map)
    
    # '검출률' 행 제거 (전체 검출률)
    if 'subtype' in df_combined.columns:
        df_combined = df_combined[df_combined['subtype'] != '검출률'].copy()
    
    # 아형 필터링
    if subtype.upper() != 'ALL':
        df_combined = df_combined[df_combined['subtype'] == subtype.upper()].copy()
    
    # 정렬
    df_combined = df_combined.sort_values(['year', 'week']).reset_index(drop=True)
    
    print(f"   - 최종 데이터: {len(df_combined)}행")
    print(f"   - 연도 범위: {df_combined['year'].min()} ~ {df_combined['year'].max()}")
    print(f"   - 아형별 분포: {df_combined['subtype'].value_counts().to_dict() if 'subtype' in df_combined.columns else 'N/A'}")
    
    return df_combined


def prepare_subtype_data(
    subtype: str = "A",
    data_dir: str = "data/before"
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    아형별(A/B) 검출률 예측을 위한 데이터 준비
    ds_0107 데이터를 사용하여 특정 아형의 검출률 시계열 예측
    
    Parameters:
        subtype: 아형 ('A' 또는 'B')
        data_dir: 데이터 디렉토리 경로
    
    Returns:
        X: (N, F) features
        y: (N,) target (검출률)
        labels: list[str] for plotting
        feat_names: list[str] feature names
    """
    # 아형별 데이터 로드
    df = load_subtype_data(data_dir=data_dir, subtype=subtype)
    
    if df.empty:
        raise ValueError(f"아형 '{subtype}' 데이터를 찾을 수 없습니다.")
    
    print(f"\n📊 아형별 검출률 예측 데이터 준비")
    print(f"   - 선택된 아형: {subtype}")
    print(f"   - 데이터 포인트: {len(df)}개")
    
    # 계절성 피처 추가
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    
    # season_norm 라벨 생성
    df['season_norm'] = df.apply(
        lambda row: f"{int(row['year'])}-{int(row['year'])+1}" if row['week'] >= 36 
                   else f"{int(row['year'])-1}-{int(row['year'])}",
        axis=1
    )
    
    # 피처 구성: 검출률 + 계절성
    feat_names = ['detection_rate', 'week_sin']
    
    # 결측치 처리
    df = df.dropna(subset=['detection_rate'])
    
    X = df[feat_names].to_numpy(dtype=float)
    y = df['detection_rate'].to_numpy(dtype=float)
    labels = (df['season_norm'].astype(str) + f" ({subtype}) - W" + df['week'].astype(int).astype(str)).tolist()
    
    print(f"\n✅ 아형별 데이터 준비 완료:")
    print(f"   - X shape: {X.shape}")
    print(f"   - y shape: {y.shape}")
    print(f"   - Features: {feat_names}")
    
    return X, y, labels, feat_names


# =========================
# 연령대별 데이터 준비 (원본 CSV에서 직접 로드)
# =========================
def load_and_prepare_by_age(
    age_group: str = "19-49세",
    data_dir: str = "data/before",
    use_exog: str = "all"
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    특정 연령대의 원본 데이터를 직접 로드하여 모델 학습용으로 전처리
    PostgreSQL을 거치지 않고 data/before에서 직접 로드
    
    Parameters:
        age_group: 연령대 (예: '19-49세', '65세이상', '0-6세')
        data_dir: 데이터 디렉토리 경로
        use_exog: 외생변수 사용 모드 ('all', 'vaccine', 'resp', 'none', 'auto')
    
    Returns:
        X: (N, F) features
        y: (N,) target (ILI)
        labels: list[str] for plotting
        feat_names: list[str] feature names
    """
    # 원본 데이터 로드
    df = load_raw_data_by_age_group(data_dir=data_dir, age_group=age_group)
    
    if df.empty:
        raise ValueError(f"연령대 '{age_group}' 데이터를 찾을 수 없습니다.")
    
    # ILI 데이터가 있는지 확인
    if 'ili' not in df.columns:
        raise ValueError(f"연령대 '{age_group}'에 ILI 데이터가 없습니다.")
    
    # ===== 날씨 데이터 병합 (PostgreSQL) =====
    print(f"\n🌡️  날씨 데이터 병합 시도...")
    try:
        df_weather = load_weather_data_from_postgres()
        if df_weather is not None and not df_weather.empty:
            df = merge_weather_with_influenza(df, df_weather)
            
            # 병합 성공 확인
            weather_cols_merged = [c for c in ['min_temp', 'max_temp', 'avg_humidity'] if c in df.columns]
            print(f"\n   ✅ 날씨 데이터 병합 성공!")
            print(f"      - 병합 후 Shape: {df.shape}")
            print(f"      - 추가된 날씨 컬럼: {weather_cols_merged}")
            print(f"      - 데이터베이스에서 성공적으로 가져온 날씨 데이터가 모델에 적용됩니다.")
        else:
            print(f"   ⚠️  날씨 데이터가 비어있습니다. 인플루엔자 데이터만으로 진행합니다.")
    except Exception as e:
        print(f"   ⚠️  날씨 데이터 병합 중 오류: {e}")
        print(f"   인플루엔자 데이터만으로 진행합니다.")
    
    print(f"\n📊 연령대별 데이터 전처리: {age_group}")
    
    # ===== 일별 데이터 변환 (바우시안 보간) =====
    if Config.USE_DAILY_DATA:
        print(f"\n🔄 주차별 → 일별 데이터 변환 시작...")
        print(f"   - 보간 방법: {Config.DAILY_INTERP_METHOD.upper()}")
        print(f"   - 바우시안 표준편차: {Config.GAUSSIAN_STD}")
        
        # season_norm 생성 (먼저)
        df['season_norm'] = df.apply(
            lambda row: f"{int(row['year'])}-{int(row['year'])+1}" if row['week'] >= 36 
                       else f"{int(row['year'])-1}-{int(row['year'])}",
            axis=1
        )
        
        # 정렬 (변환 전)
        df = df.sort_values(['year', 'week']).reset_index(drop=True)
        
        # 일별 변환
        df = weekly_to_daily_interp_gaussian(
            df,
            season_col="season_norm",
            week_col="week",
            target_col="ili",
            method=Config.DAILY_INTERP_METHOD,
            gaussian_std=Config.GAUSSIAN_STD
        )
        
        # SEQ_LEN, PRED_LEN을 일별로 업데이트
        global SEQ_LEN, PRED_LEN
        SEQ_LEN = Config.DAILY_SEQ_LEN
        PRED_LEN = Config.DAILY_PRED_LEN
        
        print(f"   ✅ 일별 데이터 변환 완료!")
        print(f"   - 새로운 입력 길이 (SEQ_LEN): {SEQ_LEN}일")
        print(f"   - 새로운 예측 길이 (PRED_LEN): {PRED_LEN}일")
        print(f"   - 변환 후 데이터 포인트 수: {len(df)}")
    else:
        # 정렬 (일별 변환 미사용)
        df = df.sort_values(['year', 'week']).reset_index(drop=True)
    
    # ===== 팬데믹 기간 처리 =====
    pandemic_mask = (
        ((df['year'] == 2020) & (df['week'] >= 14)) |
        ((df['year'] == 2021)) |
        ((df['year'] == 2022) & (df['week'] <= 22))
    )
    
    pandemic_count = pandemic_mask.sum()
    print(f"   - 팬데믹 기간 데이터: {pandemic_count}행")
    
    # 팬데믹 기간 결측치 처리
    for col in ['ili', 'hospitalization', 'detection_rate', 'emergency_patients']:
        if col in df.columns:
            df.loc[pandemic_mask, col] = np.nan
    
    # ===== 계절성 패턴 기반 보간 =====
    if df['ili'].isna().sum() > 0:
        print(f"   - ILI 결측치 보간 중...")
        
        # 팬데믹 이전 데이터로 주차별 평균 계산
        pre_pandemic = df[(df['year'] >= 2017) & (df['year'] <= 2019) & df['ili'].notna()]
        
        if not pre_pandemic.empty:
            weekly_pattern = pre_pandemic.groupby('week')['ili'].mean()
            
            for idx in df[df['ili'].isna()].index:
                week = int(df.loc[idx, 'week'])
                if week in weekly_pattern.index:
                    df.loc[idx, 'ili'] = weekly_pattern[week]
    
    # 다른 컬럼도 보간
    for col in ['hospitalization', 'detection_rate', 'emergency_patients']:
        if col in df.columns and df[col].isna().sum() > 0:
            pre_pandemic = df[(df['year'] >= 2017) & (df['year'] <= 2019) & df[col].notna()]
            if not pre_pandemic.empty:
                weekly_pattern = pre_pandemic.groupby('week')[col].mean()
                for idx in df[df[col].isna()].index:
                    week = int(df.loc[idx, 'week'])
                    if week in weekly_pattern.index:
                        df.loc[idx, col] = weekly_pattern[week]
    
    # ===== 계절성 피처 추가 =====
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    
    # season_norm 라벨 생성
    df['season_norm'] = df.apply(
        lambda row: f"{int(row['year'])}-{int(row['year'])+1}" if row['week'] >= 36 
                   else f"{int(row['year'])-1}-{int(row['year'])}",
        axis=1
    )
    
    # ===== 연령대별 동학 피처 추가 (어린이 집단 ILI) =====
    if Config.USE_AGE_GROUP_DYNAMICS and age_group not in Config.LEAD_AGE_GROUPS:
        print(f"\n🔗 연령대별 동학 피처 추가 중...")
        for lead_age in Config.LEAD_AGE_GROUPS:
            try:
                lead_df = load_raw_data_by_age_group(data_dir=data_dir, age_group=lead_age)
                if not lead_df.empty and 'ili' in lead_df.columns:
                    lead_df = lead_df.sort_values(['year', 'week']).reset_index(drop=True)
                    # year, week 기준으로 병합
                    lead_ili = lead_df[['year', 'week', 'ili']].copy()
                    lead_ili = lead_ili.rename(columns={'ili': f'ili_{lead_age.replace("-", "_").replace("세", "")}'})
                    df = df.merge(lead_ili, on=['year', 'week'], how='left')
                    print(f"   ✅ {lead_age} ILI 추가: ili_{lead_age.replace('-', '_').replace('세', '')}")
            except Exception as e:
                print(f"   ⚠️  {lead_age} 데이터 로드 실패: {e}")
    
    # ===== 트렌드 데이터 병합 (PostgreSQL trends DB) =====
    if Config.USE_TRENDS_DATA:
        print(f"\n🔍 트렌드 데이터 로드 중 (PostgreSQL {Config.TRENDS_DB_NAME} DB)...")
        try:
            from database.db_utils import load_trends_from_postgres
            trends_df = load_trends_from_postgres(
                table_name=Config.TRENDS_TABLE_NAME,
                db_name=Config.TRENDS_DB_NAME
            )
            if not trends_df.empty and 'year' in trends_df.columns and 'week' in trends_df.columns:
                df = df.merge(trends_df, on=['year', 'week'], how='left')
                # Trends 컬럼명 확인 (google_, naver_, twitter_ 접두사)
                trends_cols = [c for c in trends_df.columns if c not in ['year', 'week']]
                print(f"   ✅ 트렌드 피처 추가: {len(trends_cols)}개 컬럼")
                print(f"      (Google: {len([c for c in trends_cols if c.startswith('google_')])}개, "
                      f"Naver: {len([c for c in trends_cols if c.startswith('naver_')])}개, "
                      f"Twitter: {len([c for c in trends_cols if c.startswith('twitter_')])}개)")
                # 결측치 0으로 채움 (검색량/언급량 없음 = 0)
                for col in trends_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(0)
            else:
                print(f"   ⚠️  트렌드 데이터가 비어있거나 year, week 컬럼이 없습니다.")
        except Exception as e:
            print(f"   ⚠️  트렌드 데이터 로드 실패: {e}")
            print(f"   💡 먼저 'python database/update_trends_database.py'를 실행하세요.")
    
    # ===== 피처 선택 =====
    # 기본 피처: ILI (타겟)
    chosen = ['ili']
    
    # 외생변수 설정
    has_hosp = 'hospitalization' in df.columns and df['hospitalization'].notna().any()
    has_detection = 'detection_rate' in df.columns and df['detection_rate'].notna().any()
    has_emergency = 'emergency_patients' in df.columns and df['emergency_patients'].notna().any()
    has_vaccine = 'vaccine_rate' in df.columns and df['vaccine_rate'].notna().any()
    
    # hospitalization 제외 설정 확인
    exclude_hosp = getattr(Config, 'EXCLUDE_HOSPITALIZATION', False)
    if exclude_hosp:
        has_hosp = False
        print("   ⚠️ hospitalization 피처 제외됨 (Config.EXCLUDE_HOSPITALIZATION=True)")
    
    if use_exog in ('all', 'auto'):
        if has_hosp:
            chosen.append('hospitalization')
        if has_detection:
            chosen.append('detection_rate')
        if has_emergency:
            chosen.append('emergency_patients')
        if has_vaccine:
            chosen.append('vaccine_rate')
    elif use_exog == 'vaccine' and has_vaccine:
        chosen.append('vaccine_rate')
    elif use_exog == 'resp':
        if has_hosp:
            chosen.append('hospitalization')
        if has_detection:
            chosen.append('detection_rate')
    
    # 계절성 피처 추가
    if INCLUDE_SEASONAL_FEATS:
        chosen.append('week_sin')
    
    # 연령대별 동학 피처 추가
    if Config.USE_AGE_GROUP_DYNAMICS and age_group not in Config.LEAD_AGE_GROUPS:
        for lead_age in Config.LEAD_AGE_GROUPS:
            col_name = f'ili_{lead_age.replace("-", "_").replace("세", "")}'
            if col_name in df.columns and df[col_name].notna().any():
                chosen.append(col_name)
                print(f"   ✅ 선행 지표 추가: {col_name}")
    
    # 트렌드 피처 추가 (google_, naver_, twitter_ 접두사로 자동 감지)
    if Config.USE_TRENDS_DATA:
        trends_cols = [c for c in df.columns if c.startswith(('google_', 'naver_', 'twitter_'))]
        for col in trends_cols:
            if col in df.columns and df[col].notna().any():
                chosen.append(col)
        if trends_cols:
            print(f"   ✅ 트렌드 피처 {len(trends_cols)}개 추가")
    
    # 🌡️ 날씨 피처 추가 (PostgreSQL weather_data)
    weather_cols = ['min_temp', 'max_temp', 'avg_humidity']  # weather_data 테이블의 컬럼
    added_weather_cols = []
    for col in weather_cols:
        if col in df.columns and df[col].notna().any():
            added_weather_cols.append(col)
            chosen.append(col)
    
    if added_weather_cols:
        print(f"\n🌡️  날씨 피처 모델에 적용:")
        print(f"   ✅ PostgreSQL weather_data 테이블에서 가져온 {len(added_weather_cols)}개 피처 추가")
        print(f"      - {added_weather_cols}")
        # 각 날씨 피처의 통계 출력
        for col in added_weather_cols:
            data = df[col].dropna()
            if len(data) > 0:
                print(f"      • {col}: 평균 {data.mean():.2f}, 표준편차 {data.std():.2f}")
    else:
        print(f"\n⚠️  날씨 피처 없음 (weather_data 테이블 확인 필요)")
    
    print(f"   - 선택된 피처: {chosen}")
    
    # 결측치 처리
    for col in chosen:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # ILI가 없는 행 제거
    df = df[df['ili'].notna()].copy()
    
    # X, y 구성
    feat_names = chosen[:]
    X = df[feat_names].to_numpy(dtype=float)
    y = df['ili'].to_numpy(dtype=float)
    
    # Labels 생성 (일별 데이터인지 주차별 데이터인지 구분)
    if Config.USE_DAILY_DATA and 'date' in df.columns:
        # 일별 데이터: date 컬럼 사용
        labels = (df['season_norm'].astype(str) + f" ({age_group}) - " + df['date'].astype(str)).tolist()
    else:
        # 주차별 데이터: week 컬럼 사용
        labels = (df['season_norm'].astype(str) + f" ({age_group}) - W" + df['week'].astype(int).astype(str)).tolist()
    
    print(f"\n✅ 연령대 '{age_group}' 데이터 준비 완료:")
    print(f"   - X shape: {X.shape}")
    print(f"   - y shape: {y.shape}")
    print(f"   - Features: {feat_names}")
    print(f"   - ILI 범위: [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y, labels, feat_names


# =========================
# data loader (multivariate-ready) - PostgreSQL 버전
# =========================
def load_and_prepare(
    df: pd.DataFrame, 
    use_exog: str = "auto",
    age_group: Optional[str] = None,
    subtype: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    PostgreSQL 또는 CSV 데이터를 PatchTST 모델 학습용으로 전처리
    
    Returns:
        X: (N, F) features (first column should be target variable)
        y: (N,) target (의사환자 분율)
        labels: list[str] for plotting ticks
        used_feat_names: list[str] feature column names (len=F)
    
    Parameters:
        df: PostgreSQL 또는 API에서 가져온 DataFrame
        use_exog: 외생변수 사용 모드
        age_group: 특정 연령대 선택 (예: '19-49세', '65세이상', None이면 자동 선택)
        subtype: 아형 필터링 ('A', 'B', None이면 우세 아형 사용)
    """
    if df is None:
        raise ValueError("df는 반드시 제공되어야 합니다. 먼저 데이터를 로드하세요.")
    
    df = df.copy()
    
    print(f"\n📊 원본 데이터 구조:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Columns: {list(df.columns)}")
    
    # ===== 날씨 데이터 병합 (PostgreSQL) =====
    print(f"\n🌡️  날씨 데이터 병합 시도...")
    try:
        df_weather = load_weather_data_from_postgres()
        if df_weather is not None and not df_weather.empty:
            df = merge_weather_with_influenza(df, df_weather)
            
            # 병합 성공 확인
            weather_cols_merged = [c for c in ['min_temp', 'max_temp', 'avg_humidity'] if c in df.columns]
            print(f"\n   ✅ 날씨 데이터 병합 성공!")
            print(f"      - 병합 후 Shape: {df.shape}")
            print(f"      - 추가된 날씨 컬럼: {weather_cols_merged}")
            print(f"      - 데이터베이스에서 성공적으로 가져온 날씨 데이터가 모델에 적용됩니다.")
        else:
            print(f"   ⚠️  날씨 데이터가 비어있습니다. 인플루엔자 데이터만으로 진행합니다.")
    except Exception as e:
        print(f"   ⚠️  날씨 데이터 병합 중 오류: {e}")
        print(f"   인플루엔자 데이터만으로 진행합니다.")
    
    # ===== PostgreSQL 데이터 형식 감지 및 처리 =====
    is_postgres_format = all(col in df.columns for col in ['year', 'week', 'age_group'])
    
    if is_postgres_format:
        print(f"\n🔍 PostgreSQL 데이터 형식 감지됨 - 연령대별 데이터 처리 중...")
        
        # ===== 팬데믹 기간 데이터 처리: 계절성 패턴 기반 보간 =====
        print(f"\n💡 팬데믹 기간 데이터 처리 전략: 계절성 패턴 기반 보간")
        print(f"   - 팬데믹 기간 (2020-W14 ~ 2022-W22)을 결측치로 표시")
        print(f"   - 과거 계절 패턴(2017-2019)으로 보간하여 시계열 연속성 유지")
        
        before_count = len(df)
        
        # 팬데믹 기간 마스크 생성
        pandemic_mask = (
            ((df['year'] == 2020) & (df['week'] >= 14)) |
            ((df['year'] == 2021)) |
            ((df['year'] == 2022) & (df['week'] <= 22))
        )
        
        pandemic_count = pandemic_mask.sum()
        print(f"\n   📊 팬데믹 기간 데이터: {pandemic_count:,}행 ({pandemic_count/before_count*100:.1f}%)")
        
        # 팬데믹 기간의 ILI 값을 NaN으로 설정 (결측치 표시)
        # 나중에 연령대별로 처리한 후 보간할 것임
        import numpy as np
        df.loc[pandemic_mask, 'ili'] = np.nan
        
        # 다른 수치형 컬럼도 팬데믹 기간 동안 NaN 처리
        numeric_cols_to_mask = ['hospitalization', 'detection_rate', 'emergency_patients']
        for col in numeric_cols_to_mask:
            if col in df.columns:
                df.loc[pandemic_mask, col] = np.nan
        
        print(f"   ✅ 팬데믹 기간 데이터를 결측치(NaN)로 표시 완료")
        print(f"   ⏭️  연령대별 필터링 후 보간 처리 예정")
        
        # 연령대별 데이터 확인
        age_groups = df['age_group'].unique()
        print(f"\n   - 고유 연령대: {len(age_groups)}개")
        print(f"   - 연령대 목록: {sorted(age_groups)[:5]}...")
        
        # 연령대 선택: 파라미터로 지정된 경우 우선 사용
        target_age_group = None
        
        if age_group is not None:
            # 사용자 지정 연령대
            if age_group in age_groups:
                target_age_group = age_group
                print(f"   - 사용자 지정 연령대 사용: '{age_group}'")
            else:
                print(f"   ⚠️ 지정된 연령대 '{age_group}'를 찾을 수 없습니다.")
                print(f"   ℹ️ 사용 가능한 연령대: {sorted(age_groups)}")
        
        if target_age_group is None:
            # 자동 선택: 데이터가 가장 풍부한 연령대
            # 우선순위: 19-49세 (가장 일반적) > 65세이상 > 65세 이상 > 0-6세
            candidate_age_groups = ['19-49세', '65세이상', '65세 이상', '0-6세']
            
            for candidate in candidate_age_groups:
                if candidate in age_groups:
                    # 해당 연령대의 데이터 품질 확인
                    temp_df = df[df['age_group'] == candidate].copy()
                    valid_ili = temp_df['ili'].notna().sum()
                    if valid_ili > 100:  # 최소 100개 이상의 유효 데이터
                        target_age_group = candidate
                        break
        
        if target_age_group and target_age_group in age_groups:
            print(f"   - '{target_age_group}' 연령대 데이터 사용")
            df_age = df[df['age_group'] == target_age_group].copy()
            
            # 정렬 (보간 전 필수)
            df_age = df_age.sort_values(['year', 'week']).reset_index(drop=True)
            
            # ===== 팬데믹 기간 결측치 보간: 계절성 패턴 기반 =====
            print(f"\n   🔧 팬데믹 기간 결측치 보간 시작 (계절성 패턴 기반)...")
            
            # ILI 보간 전 결측치 개수
            ili_nan_before = df_age['ili'].isna().sum()
            print(f"      - ILI 결측치: {ili_nan_before}개")
            
            if ili_nan_before > 0:
                # ✅ 계절성 패턴 기반 보간 (Seasonal Pattern Interpolation)
                print(f"      - 보간 방법: 과거 계절성 패턴 (2017-2019 기준)")
                
                # 1️⃣ 팬데믹 이전 기간의 주차별 평균 패턴 계산
                pre_pandemic_mask = (df_age['year'] >= 2017) & (df_age['year'] <= 2019)
                df_pre_pandemic = df_age[pre_pandemic_mask & df_age['ili'].notna()].copy()
                
                # 주차별 평균 ILI 계산
                weekly_pattern = df_pre_pandemic.groupby('week')['ili'].mean()
                print(f"      - 참조 데이터: 2017-2019년 ({len(df_pre_pandemic)}행)")
                print(f"      - 주차별 패턴: {len(weekly_pattern)}개 주차")
                
                # 2️⃣ 팬데믹 기간(NaN) 데이터를 주차별 평균 패턴으로 대체
                pandemic_nan_mask = df_age['ili'].isna()
                for idx in df_age[pandemic_nan_mask].index:
                    week_num = df_age.loc[idx, 'week']
                    # 해당 주차의 과거 평균값으로 채우기
                    if week_num in weekly_pattern.index:
                        df_age.loc[idx, 'ili'] = weekly_pattern[week_num]
                    else:
                        # 혹시 주차가 없으면 전체 평균 사용
                        df_age.loc[idx, 'ili'] = weekly_pattern.mean()
                
                ili_nan_after = df_age['ili'].isna().sum()
                filled_count = ili_nan_before - ili_nan_after
                print(f"      ✅ ILI 보간 완료: {filled_count}개 채워짐 (계절 패턴 기반)")
                
                # 3️⃣ 음수 값 제거 (ILI는 양수여야 함)
                negative_count = (df_age['ili'] < 0).sum()
                if negative_count > 0:
                    print(f"      ⚠️ 음수 값 {negative_count}개 발견 - 0으로 대체")
                    df_age.loc[df_age['ili'] < 0, 'ili'] = 0
            
            # 4️⃣ 다른 수치형 컬럼도 계절성 패턴으로 보간
            numeric_cols_to_interpolate = ['hospitalization', 'detection_rate', 'emergency_patients']
            for col in numeric_cols_to_interpolate:
                if col in df_age.columns:
                    nan_count = df_age[col].isna().sum()
                    if nan_count > 0:
                        # 과거 주차별 평균 패턴 계산
                        pre_pandemic_mask = (df_age['year'] >= 2017) & (df_age['year'] <= 2019)
                        df_pre_pandemic = df_age[pre_pandemic_mask & df_age[col].notna()].copy()
                        
                        if len(df_pre_pandemic) > 0:
                            weekly_pattern = df_pre_pandemic.groupby('week')[col].mean()
                            
                            # 팬데믹 기간 결측치를 패턴으로 대체
                            col_nan_mask = df_age[col].isna()
                            for idx in df_age[col_nan_mask].index:
                                week_num = df_age.loc[idx, 'week']
                                if week_num in weekly_pattern.index:
                                    df_age.loc[idx, col] = weekly_pattern[week_num]
                                else:
                                    df_age.loc[idx, col] = weekly_pattern.mean()
                            
                            # 음수 값 제거
                            if (df_age[col] < 0).sum() > 0:
                                df_age.loc[df_age[col] < 0, col] = 0
                            
                            print(f"      ✅ {col} 보간 완료 (계절 패턴 기반)")
                        else:
                            # 참조 데이터가 없으면 median으로 대체
                            median_val = df_age[col].median()
                            if pd.notna(median_val):
                                df_age[col] = df_age[col].fillna(median_val)
                            print(f"      ⚠️ {col} 보간: 참조 데이터 부족 - median 사용")
            
            print(f"\n   ✅ 팬데믹 기간 보간 완료 - 시계열 연속성 유지")
            print(f"   📊 최종 데이터: {len(df_age)}행")
            
            # 예방접종률이 모두 NaN인 경우 전체 평균으로 채우기
            if df_age['vaccine_rate'].notna().sum() == 0:
                print(f"   - '{target_age_group}' 연령대에 예방접종률 데이터 없음 - 전체 평균 사용")
                # 연도/주차별로 전체 연령대의 예방접종률 평균 계산
                vaccine_avg = df.groupby(['year', 'week'], as_index=False)['vaccine_rate'].mean()
                vaccine_avg = vaccine_avg.rename(columns={'vaccine_rate': 'vaccine_rate_avg'})
                df_age = df_age.merge(vaccine_avg, on=['year', 'week'], how='left')
                df_age['vaccine_rate'] = df_age['vaccine_rate_avg']
                df_age = df_age.drop(columns=['vaccine_rate_avg'])
            
            df = df_age
        else:
            # 적절한 단일 연령대가 없으면 연도/주차별 평균 사용
            print(f"   - 연도/주차별 전체 연령대 평균 사용")
            numeric_cols = ['ili', 'hospitalization', 'detection_rate', 'vaccine_rate', 'emergency_patients']
            agg_dict = {col: 'mean' for col in numeric_cols if col in df.columns}
            agg_dict['subtype'] = 'first'  # 아형은 첫 값 사용
            
            df = df.groupby(['year', 'week'], as_index=False).agg(agg_dict)
        
        # 정렬
        df = df.sort_values(['year', 'week']).reset_index(drop=True)
        
        # season_norm 생성 (week 36 이상은 현재 연도 시즌, 미만은 다음 연도 시즌)
        df['season_norm'] = df.apply(
            lambda row: f"{int(row['year'])}-{int(row['year'])+1}" if row['week'] >= 36 
                       else f"{int(row['year'])-1}-{int(row['year'])}",
            axis=1
        )
        
        print(f"\n✅ PostgreSQL 데이터 변환 완료:")
        print(f"   - 변환 후 Shape: {df.shape}")
        print(f"   - 연도 범위: {df['year'].min():.0f} ~ {df['year'].max():.0f}")
        print(f"   - 주차 범위: {df['week'].min():.0f} ~ {df['week'].max():.0f}")
        print(f"   - 데이터 포인트 수: {len(df)}")
    
    # ===== 일별 데이터 변환 (바우시안 보간) =====
    if Config.USE_DAILY_DATA:
        print(f"\n🔄 주차별 → 일별 데이터 변환 시작...")
        print(f"   - 보간 방법: {Config.DAILY_INTERP_METHOD.upper()}")
        print(f"   - 바우시안 표준편차: {Config.GAUSSIAN_STD}")
        
        # season_norm 생성 (아직 없으면)
        if 'season_norm' not in df.columns and {'year', 'week'}.issubset(df.columns):
            df['season_norm'] = df.apply(
                lambda row: f"{int(row['year'])}-{int(row['year'])+1}" if row['week'] >= 36 
                           else f"{int(row['year'])-1}-{int(row['year'])}",
                axis=1
            )
        
        # 일별 변환
        df = weekly_to_daily_interp_gaussian(
            df,
            season_col="season_norm",
            week_col="week",
            target_col="ili",
            method=Config.DAILY_INTERP_METHOD,
            gaussian_std=Config.GAUSSIAN_STD
        )
        
        # SEQ_LEN, PRED_LEN을 일별로 업데이트
        global SEQ_LEN, PRED_LEN
        SEQ_LEN = Config.DAILY_SEQ_LEN
        PRED_LEN = Config.DAILY_PRED_LEN
        
        print(f"   ✅ 일별 데이터 변환 완료!")
        print(f"   - 새로운 입력 길이 (SEQ_LEN): {SEQ_LEN}일")
        print(f"   - 새로운 예측 길이 (PRED_LEN): {PRED_LEN}일")
    
    # ⚠️  정렬: year, week만 사용 (season_norm 정렬 제거)
    # season_norm 기준 정렬은 시간 순서를 파괴함 (week 1이 week 36보다 앞으로 감)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
    elif {"year", "week"}.issubset(df.columns):
        # year, week만 사용하여 시간 순서 유지
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["week"] = pd.to_numeric(df["week"], errors="coerce")
        df = df.sort_values(["year", "week"]).reset_index(drop=True)
        print(f"   - 정렬: year, week 기준 (시간 순서 유지)")
        
        # 🔴 중복 제거: 같은 (year, week) 조합이 여러 개 있으면 첫 번째만 유지
        before_len = len(df)
        df = df.drop_duplicates(subset=["year", "week"], keep="first")
        after_len = len(df)
        if before_len != after_len:
            print(f"   ⚠️ 중복 {before_len - after_len}개 제거됨 (동일 year/week)")
    elif "label" in df.columns:
        df = df.sort_values(["label"]).reset_index(drop=True)

    # 타깃 변수 확인
    if "ili" not in df.columns:
        raise ValueError("데이터에 'ili' (의사환자 분율) 컬럼이 없습니다.")
    
    df["ili"] = pd.to_numeric(df["ili"], errors="coerce")
    if df["ili"].isna().any():
        print(f"   ⚠️ 'ili' 컬럼에 결측치 {df['ili'].isna().sum()}개 발견 - 보간 처리")
        df["ili"] = df["ili"].interpolate(method="linear", limit_direction="both").fillna(df["ili"].median())
    
    # --- Seasonality feature 추가 ---
    if "week" in df.columns:
        df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52.0)
    else:
        df["week_sin"] = 0.0
    
    # --- 연령대별 동학 피처 추가 (PostgreSQL 버전) ---
    if Config.USE_AGE_GROUP_DYNAMICS and age_group and age_group not in Config.LEAD_AGE_GROUPS:
        print(f"\n🔗 연령대별 동학 피처 추가 중 (PostgreSQL)...")
        # 현재 df는 필터링된 연령대만 있으므로, 전체 데이터를 다시 로드해야 함
        # 여기서는 merged CSV에서 직접 로드하는 방식으로 처리
        try:
            csv_path = pick_csv_path()
            full_df = pd.read_csv(csv_path)
            for lead_age in Config.LEAD_AGE_GROUPS:
                lead_data = full_df[full_df['age_group'] == lead_age].copy()
                if not lead_data.empty and 'ili' in lead_data.columns:
                    lead_data = lead_data.sort_values(['year', 'week']).reset_index(drop=True)
                    lead_ili = lead_data[['year', 'week', 'ili']].copy()
                    col_name = f'ili_{lead_age.replace("-", "_").replace("세", "")}'
                    lead_ili = lead_ili.rename(columns={'ili': col_name})
                    df = df.merge(lead_ili, on=['year', 'week'], how='left')
                    # 결측치 처리
                    if col_name in df.columns:
                        df[col_name] = df[col_name].fillna(0)
                    print(f"   ✅ {lead_age} ILI 추가: {col_name}")
        except Exception as e:
            print(f"   ⚠️  연령대별 동학 피처 추가 실패: {e}")
    
    # --- 트렌드 데이터 병합 (PostgreSQL trends DB) ---
    if Config.USE_TRENDS_DATA:
        print(f"\n🔍 트렌드 데이터 로드 중 (PostgreSQL {Config.TRENDS_DB_NAME} DB)...")
        try:
            from database.db_utils import load_trends_from_postgres
            trends_df = load_trends_from_postgres(
                table_name=Config.TRENDS_TABLE_NAME,
                db_name=Config.TRENDS_DB_NAME
            )
            if not trends_df.empty and 'year' in trends_df.columns and 'week' in trends_df.columns:
                df = df.merge(trends_df, on=['year', 'week'], how='left')
                trends_cols = [c for c in trends_df.columns if c not in ['year', 'week']]
                print(f"   ✅ 트렌드 피처 추가: {len(trends_cols)}개 컬럼")
                for col in trends_cols:
                    if col in df.columns:
                        df[col] = df[col].fillna(0)
            else:
                print(f"   ⚠️  트렌드 데이터가 비어있거나 year, week 컬럼이 없습니다.")
        except Exception as e:
            print(f"   ⚠️  트렌드 데이터 로드 실패: {e}")
            print(f"   💡 먼저 'python database/update_database.py'를 실행하세요.")

    # --- Alias 매핑 ---
    if "hospitalization" in df.columns and "respiratory_index" not in df.columns:
        df["respiratory_index"] = df["hospitalization"]
    if "case_count" in df.columns and "respiratory_index" not in df.columns:
        df["respiratory_index"] = df["case_count"]

    # 기후 피처 후보
    climate_feats = []
    if "wx_week_avg_temp" in df.columns:     climate_feats.append("wx_week_avg_temp")
    if "wx_week_avg_rain" in df.columns:     climate_feats.append("wx_week_avg_rain")
    if "wx_week_avg_humidity" in df.columns: climate_feats.append("wx_week_avg_humidity")
    if "detection_rate" in df.columns:       climate_feats.append("detection_rate")  # PostgreSQL 특성

    # 외생 후보 존재 여부
    has_vax  = "vaccine_rate" in df.columns
    has_resp = "respiratory_index" in df.columns or "hospitalization" in df.columns


    # 모든 column_mapping 내부명을 feature로 강제 포함
    column_mapping = {
        '연도': 'year',
        '주차': 'week',
        '의사환자 분율': 'ili',
        '예방접종률': 'vaccine_rate',
        '입원환자 수': 'hospitalization',
        '인플루엔자 검출률': 'detection_rate',
        '응급실 인플루엔자 환자': 'emergency_patients',
        '아형': 'subtype'
    }
    
    # hospitalization 제외 설정 확인
    exclude_hosp = getattr(Config, 'EXCLUDE_HOSPITALIZATION', False)
    
    # week는 week_sin으로 대체, 나머지는 그대로
    chosen = []
    for v in column_mapping.values():
        if v == "week":
            chosen.append("week_sin")
        elif v == "hospitalization" and exclude_hosp:
            # hospitalization 제외
            continue
        else:
            chosen.append(v)
    # 중복 제거 및 순서 보존
    chosen = [x for i, x in enumerate(chosen) if x not in chosen[:i]]
    
    if exclude_hosp:
        print("   ⚠️ hospitalization 피처 제외됨 (Config.EXCLUDE_HOSPITALIZATION=True)")

    # 숫자화 & 보간
    for c in chosen:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            if df[c].isna().any():
                # 선형 보간 후 median으로 남은 결측치 채우기
                df[c] = df[c].interpolate(method="linear", limit_direction="both")
                # 여전히 NaN이 있으면 median 사용 (median도 NaN이면 0 사용)
                median_val = df[c].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df[c] = df[c].fillna(median_val)
        else:
            # 컬럼이 없으면 0으로 채움
            print(f"   ⚠️ 컬럼 '{c}'가 없습니다. 0으로 채웁니다.")
            df[c] = 0.0

    # 라벨
    if "label" in df.columns and df["label"].notna().any():
        labels = df["label"].astype(str).tolist()
    elif {"season_norm","week"}.issubset(df.columns):
        labels = (df["season_norm"].astype(str) + " season - W" + df["week"].astype(int).astype(str)).tolist()
    else:
        labels = [f"idx_{i}" for i in range(len(df))]

    # X, y 구성
    feat_names = chosen[:]
    if INCLUDE_SEASONAL_FEATS and "week_sin" in df.columns:
        feat_names.append("week_sin")
    
    # 연령대별 동학 피처 추가 (PostgreSQL 버전)
    if Config.USE_AGE_GROUP_DYNAMICS and age_group and age_group not in Config.LEAD_AGE_GROUPS:
        for lead_age in Config.LEAD_AGE_GROUPS:
            col_name = f'ili_{lead_age.replace("-", "_").replace("세", "")}'
            if col_name in df.columns:
                feat_names.append(col_name)
                print(f"   ✅ 선행 지표 피처 추가: {col_name}")
    
    # 트렌드 피처 추가 (PostgreSQL 버전)
    if Config.USE_TRENDS_DATA:
        trends_cols = [c for c in df.columns if c.startswith(('google_', 'naver_', 'twitter_'))]
        for col in trends_cols:
            if col in df.columns:
                feat_names.append(col)
        if trends_cols:
            print(f"   ✅ 트렌드 피처 {len(trends_cols)}개 추가")

    # 🌡️ 날씨 피처 추가 (PostgreSQL weather_data)
    weather_cols = ['min_temp', 'max_temp', 'avg_humidity']  # weather_data 테이블의 컬럼
    added_weather_cols = []
    for col in weather_cols:
        if col in df.columns:
            added_weather_cols.append(col)
            feat_names.append(col)
    
    if added_weather_cols:
        print(f"\n🌡️  날씨 피처 모델에 적용:")
        print(f"   ✅ PostgreSQL weather_data 테이블에서 가져온 {len(added_weather_cols)}개 피처 추가")
        print(f"      - {added_weather_cols}")
        # 각 날씨 피처의 통계 출력
        for col in added_weather_cols:
            data = df[col].dropna()
            if len(data) > 0:
                print(f"      • {col}: 평균 {data.mean():.2f}, 표준편차 {data.std():.2f}")
    else:
        print(f"\n⚠️  날씨 피처 없음 (weather_data 테이블 확인 필요)")
    
    # 선택된 입력 피처 로그
    print(f"\n[Data] Exogenous detected -> vaccine_rate: {has_vax} | respiratory/hospitalization: {has_resp} | climate_feats: {climate_feats}")
    print(f"[Data] Selected feature columns (order) -> {feat_names}")

    X = df[feat_names].to_numpy(dtype=float)
    y = df["ili"].to_numpy(dtype=float)
    
    # 🔍 vaccine_rate 진단
    if 'vaccine_rate' in feat_names:
        vax_idx = feat_names.index('vaccine_rate')
        vax_data = X[:, vax_idx]
        print(f"\n🔬 vaccine_rate 데이터 분석:")
        print(f"   - 범위: [{vax_data.min():.4f}, {vax_data.max():.4f}]")
        print(f"   - 평균: {vax_data.mean():.4f}, 표준편차: {vax_data.std():.4f}")
        print(f"   - 변동계수(CV): {vax_data.std()/vax_data.mean():.4f}")
        print(f"   - 0인 값: {(vax_data == 0).sum()}개 / {len(vax_data)}개")
        print(f"   - 상관계수 (vaccine_rate vs ili): {np.corrcoef(vax_data, y)[0,1]:.4f}")
    
    print(f"\n✅ 최종 데이터 준비 완료:")
    print(f"   - X shape: {X.shape}")
    print(f"   - y shape: {y.shape}")
    print(f"   - Features: {len(feat_names)}")
    
    return X, y, labels, feat_names

# =========================
# Loss Function
# =========================
class PeakAwareLoss(nn.Module):
    """
    고정 기준 Peak + 진폭 보존 + Horizon Weighting Loss
    
    특징:
    1. Peak 구간(상위 quantile)에 높은 가중치 적용
    2. 진폭 보존 항으로 peak flattening 방지
    3. Horizon weighting: 예측 구간별 가중치 (피크가 주로 나타나는 후반부 강조)
    4. MAE 기반으로 outlier에 robust
    """
    def __init__(self, peak_quantile=0.9, alpha=4.0, beta=0.3, 
                 pred_len=4, horizon_mode="exponential", 
                 horizon_exp_scale=1.2, horizon_tail_boost=2.5, horizon_tail_count=2):
        super().__init__()
        self.peak_quantile = peak_quantile
        self.alpha = alpha  # 피크 가중치
        self.beta = beta    # 진폭 보존 가중치
        self.mae = nn.L1Loss(reduction="none")
        
        # 🔴 Horizon Weighting 계산
        h_weights = self._compute_horizon_weights(
            pred_len, horizon_mode, horizon_exp_scale, 
            horizon_tail_boost, horizon_tail_count
        )
        # tensor로 변환하여 등록 (학습되지 않는 버퍼)
        self.register_buffer('horizon_weights', torch.from_numpy(h_weights).float())
        
        print(f"[Loss] Horizon weights ({horizon_mode}): {h_weights}")
    
    def _compute_horizon_weights(self, pred_len, mode, exp_scale, tail_boost, tail_count):
        """예측 구간별 가중치 계산"""
        if mode == "exponential":
            # 지수적으로 증가 (뒤로 갈수록 가중치 증가)
            h_weights = np.exp(np.linspace(0, exp_scale, pred_len))
        elif mode == "tail_boost":
            # 뒤쪽 N개만 부스트
            h_weights = np.ones(pred_len)
            h_weights[-tail_count:] *= tail_boost
        else:  # uniform
            h_weights = np.ones(pred_len)
        
        # 정규화 (합이 pred_len이 되도록)
        h_weights = h_weights / h_weights.sum() * pred_len
        return h_weights
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, H) 예측값
            target: (B, H) 실제값
        Returns:
            loss: scalar
        """
        # Base MAE
        base_loss = self.mae(pred, target)  # (B, H)
        
        # 🔴 피크 구간 가중 (배치별 동적 threshold)
        with torch.no_grad():
            peak_threshold = torch.quantile(target, self.peak_quantile)
            peak_mask = target >= peak_threshold
            weights = torch.ones_like(target)
            weights[peak_mask] = self.alpha
        
        # 🔴 Horizon weighting 적용
        # horizon_weights: (H,) -> (1, H)로 브로드캐스트
        horizon_w = self.horizon_weights.view(1, -1)  # (1, H)
        weighted_mae = (base_loss * weights * horizon_w).mean()
        
        # 🔴 진폭 보존 항 (peak flattening 방지)
        # 각 배치 시퀀스의 최대값 차이를 패널티로 추가
        pred_max = pred.max(dim=1).values    # (B,)
        target_max = target.max(dim=1).values  # (B,)
        amp_loss = torch.abs(pred_max - target_max).mean()
        
        # 총 손실
        total_loss = weighted_mae + self.beta * amp_loss
        
        return total_loss

# =========================
# dataset
# =========================
class PatchTSTDataset(Dataset):
    """Multivariate X (N,F) + y (N,) -> (patchified) windows."""
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len:int, pred_len:int, patch_len:int, stride:int):
        assert len(X) == len(y)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len, self.pred_len = seq_len, pred_len
        self.patch_len, self.stride = patch_len, stride
        max_start = len(self.y) - (seq_len + pred_len)
        self.indices = list(range(max(0, max_start + 1)))

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        seq_X = self.X[i:i+self.seq_len, :]                      # (L, F)
        tgt_y = self.y[i+self.seq_len:i+self.seq_len+self.pred_len]  # (H,)

        # patchify along time axis
        patches = []
        pos = 0
        while pos + self.patch_len <= self.seq_len:
            patches.append(seq_X[pos:pos+self.patch_len, :])     # (patch_len, F)
            pos += self.stride
        X_patch = np.stack(patches, axis=0)                      # (P, patch_len, F)
        return torch.from_numpy(X_patch).float(), torch.from_numpy(tgt_y).float(), i

# =========================
# model (Multi-Scale CNN + TokenConvMixer + PatchTST + AttnPool)
# =========================
class MultiScaleCNNPatchEmbed(nn.Module):
    """
    (B, P, L, F) -> [각 패치] 멀티스케일 Conv1d 분기(k=2/3/5, 또 하나는 dilation=2) → GAP → (B, P, D)
    - 분기 4개 출력 concat → D_MODEL
    - 패치 내부의 급격/완만/잔진동 패턴을 동시에 포착
    """
    def __init__(self, in_features: int, patch_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % 4 == 0, "d_model은 4의 배수가 되어야 멀티스케일 분기 합산이 맞습니다."
        out_ch = d_model // 4
    # 커널 크기를 patch_len에 비례하게 설정
        self.b2 = nn.Conv1d(in_features, out_ch, kernel_size=1, padding=0)
        self.b3 = nn.Conv1d(in_features, out_ch, kernel_size=3, padding=1)
        self.b5 = nn.Conv1d(in_features, out_ch, kernel_size=5, padding=2)
        self.bd = nn.Conv1d(in_features, out_ch, kernel_size=3, padding=2, dilation=2)

        self.bn   = nn.BatchNorm1d(d_model)
        self.act  = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)   # (B*P, D, L) → (B*P, D, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, P, L, F)
        B, P, L, F = x.shape
        x = x.view(B*P, L, F).permute(0, 2, 1)        # (B*P, F, L)

        z = torch.cat([self.b2(x), self.b3(x), self.b5(x), self.bd(x)], dim=1)  # (B*P, D, L)
        z = self.act(self.bn(z))
        z = self.pool(z).squeeze(-1)                  # (B*P, D)
        z = self.drop(z)
        return z.view(B, P, -1)                       # (B, P, D)

class TokenConvMixer(nn.Module):
    """
    패치 토큰 간(P 축) 로컬 연속성 강화: DepthwiseConv1d(P-축) + PointwiseConv1d
    입력/출력: (B, P, D)
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, z):              # (B, P, D)
        y = z.permute(0, 2, 1)         # (B, D, P)
        y = self.dw(y)
        y = self.pw(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.drop(y)
        y = y.permute(0, 2, 1)         # (B, P, D)
        return z + y                   # Residual

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div)
        if d_model % 2 == 1:
            pe[:,1::2] = torch.cos(position*div)[:, :pe[:,1::2].shape[1]]
        else:
            pe[:,1::2] = torch.cos(position*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        P = x.size(1)
        return x + self.pe[:, :P, :]

class AttnPool(nn.Module):
    """Learnable-query attention pooling over patch tokens."""
    def __init__(self, d_model:int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model))
        self.proj = nn.Linear(d_model, d_model, bias=False)
    def forward(self, z):           # z: (B, P, D)
        B,P,D = z.shape
        q = self.q.expand(B, -1, -1)                       # (B,1,D)
        k = self.proj(z)                                   # (B,P,D)
        attn = torch.softmax((q @ k.transpose(1,2)) / (D**0.5), dim=-1)  # (B,1,P)
        pooled = attn @ z                                  # (B,1,D)
        return pooled.squeeze(1)                           # (B,D)

class PatchTSTModel(nn.Module):
    def __init__(self, in_features:int, patch_len:int, d_model:int, n_heads:int,
                 n_layers:int, ff_dim:int, dropout:float, pred_len:int, head_hidden:List[int]):
        super().__init__()
        # ① 멀티스케일 CNN 패치 임베딩
        self.embed = MultiScaleCNNPatchEmbed(in_features, patch_len, d_model, dropout=dropout*0.5)
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
        # head_hidden: MLP layers for feature extraction
        mlp_shared, in_dim = [], d_model
        for h in head_hidden[:2]:
            mlp_shared += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        self.shared_mlp = nn.Sequential(*mlp_shared) if mlp_shared else nn.Identity()
        
        # Dual heads
        self.head_trend = nn.Linear(in_dim, pred_len)  # 기본 트렌드
        self.head_peak = nn.Linear(in_dim, pred_len)   # 피크 보정 (양수만)

    def forward(self, x):
        # x: (B, P, L, F)
        z = self.embed(x)      # (B,P,D)
        z = self.mixer(z)      # (B,P,D)
        z = self.posenc(z)
        z = self.encoder(z)
        z = self.pool(z)       # (B,D)
        
        # Shared MLP
        z = self.shared_mlp(z)  # (B, hidden_dim)
        
        # Dual-head prediction with adaptive gating
        trend = self.head_trend(z)              # (B, H) - 기본 트렌드
        peak = torch.relu(self.head_peak(z))    # (B, H) - 피크 보정 (양수만)
        
        # trend가 클 때 peak 영향 증가 (sigmoid gating)
        return trend + peak * torch.sigmoid(trend)  # (B,H) - 최종 예측

    def correlation_loss(pred, true):
    # pred, true: (B, H)
        pred = pred - pred.mean(dim=1, keepdim=True)
        true = true - true.mean(dim=1, keepdim=True)
        corr = (pred * true).sum(dim=1) / (
            (pred.norm(dim=1) * true.norm(dim=1)) + 1e-6
        )
        return 1 - corr.mean()
    # =========================
# helpers
# =========================
def warmup_lr(ep:int, base_lr:float, warmup_epochs:int):
    if ep <= warmup_epochs:
        return base_lr * (ep / max(1, warmup_epochs))
    return base_lr

def batch_mae_in_original_units(pred_b: torch.Tensor, y_b: torch.Tensor, scaler_y) -> float:
    """
    Compute MAE in original units for single-step or multi-step prediction.

    pred_b: (B,) or (B,1) or (B,H)
    y_b:    (B,H) or (B,)
    """
    # move to numpy
    p = pred_b.detach().cpu().numpy()
    t = y_b.detach().cpu().numpy()

    # ensure 2D
    if p.ndim == 1:
        p = p[:, None]          # (B,1)
    if t.ndim == 1:
        t = t[:, None]          # (B,1)

    # if prediction is single-step but target is multi-step, broadcast
    if p.shape[1] == 1 and t.shape[1] > 1:
        p = np.repeat(p, t.shape[1], axis=1)

    # flatten to (B*H, 1)
    p = p.reshape(-1, 1)
    t = t.reshape(-1, 1)

    # inverse scaling
    p_orig = scaler_y.inverse_transform(p).reshape(-1)
    t_orig = scaler_y.inverse_transform(t).reshape(-1)

    return float(np.mean(np.abs(p_orig - t_orig)))

def batch_rmse_in_original_units(pred_b: torch.Tensor, y_b: torch.Tensor, scaler_y) -> float:
    """
    Compute RMSE in original units for single-step or multi-step prediction.
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

    return float(np.sqrt(np.mean((p_orig - t_orig)**2)))

def batch_mse_in_original_units(pred_b: torch.Tensor, y_b: torch.Tensor, scaler_y) -> float:
    """
    Compute MSE in original units for single-step or multi-step prediction.
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

    return float(np.mean((p_orig - t_orig)**2))

def batch_corrcoef(pred_b: torch.Tensor, y_b: torch.Tensor, scaler_y) -> float:
    """
    Pearson correlation coefficient (batch 평균)
    pred_b, y_b: (B, H)
    """
    p = pred_b.detach().cpu().numpy().reshape(-1, 1)
    t = y_b.detach().cpu().numpy().reshape(-1, 1)
    p_orig = scaler_y.inverse_transform(p).reshape(-1)
    t_orig = scaler_y.inverse_transform(t).reshape(-1)

    if np.std(p_orig) < 1e-6 or np.std(t_orig) < 1e-6:
        return 0.0
    return float(np.corrcoef(p_orig, t_orig)[0,1])

# =========================
# train & evaluate
# =========================
def train_and_eval(X: np.ndarray, y: np.ndarray, labels: list, feat_names: list):
    """
    X: (N,F), y: (N,), feat_names: ['ili', 'vaccine_rate', 'respiratory_index'] 등
    """
    set_seed(SEED)
    (s0,e0),(s1,e1),(s2,e2) = make_splits(len(y))
    X_tr, X_va, X_te = X[s0:e0], X[s1:e1], X[s2:e2]
    y_tr, y_va, y_te = y[s0:e0], y[s1:e1], y[s2:e2]
    lab_tr, lab_va, lab_te = labels[s0:e0], labels[s1:e1], labels[s2:e2]

    # ==== 데이터 분할 진단 ====
    print(f"\n📊 데이터 분할 정보:")
    print(f"   Train: {lab_tr[0]} ~ {lab_tr[-1]} ({len(y_tr)}개)")
    print(f"   Val:   {lab_va[0]} ~ {lab_va[-1]} ({len(y_va)}개)")
    print(f"   Test:  {lab_te[0]} ~ {lab_te[-1]} ({len(y_te)}개)")
    print(f"   Train y 범위: [{y_tr.min():.2f}, {y_tr.max():.2f}], 평균: {y_tr.mean():.2f}")
    print(f"   Val   y 범위: [{y_va.min():.2f}, {y_va.max():.2f}], 평균: {y_va.mean():.2f}")
    print(f"   Test  y 범위: [{y_te.min():.2f}, {y_te.max():.2f}], 평균: {y_te.mean():.2f}")

    # ==== Scaling ====
    # Target scaler (타겟: Log 변환 적용)
    scaler_y = get_scaler(for_target=True)
    y_tr_sc = scaler_y.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_va_sc = scaler_y.transform(y_va.reshape(-1,1)).ravel()
    y_te_sc = scaler_y.transform(y_te.reshape(-1,1)).ravel()

    # Feature scaler (피처: Log 변환 미적용)
    scaler_x = get_scaler(for_target=False)
    X_tr_sc = scaler_x.fit_transform(X_tr)
    X_va_sc = scaler_x.transform(X_va)
    X_te_sc = scaler_x.transform(X_te)

    F = X.shape[1]
    print(f"[Shapes] X_tr:{X_tr.shape}, X_va:{X_va.shape}, X_te:{X_te.shape} | F={F}")
    print(f"[Info] Model input feature order -> {feat_names}")

    ds_tr = PatchTSTDataset(X_tr_sc, y_tr_sc, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
    ds_va = PatchTSTDataset(X_va_sc, y_va_sc, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
    ds_te = PatchTSTDataset(X_te_sc, y_te_sc, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)

    # drop_last=False 로 변경(작은 데이터셋에서도 학습 배치 보장)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False)

    model = PatchTSTModel(
        in_features=F, patch_len=PATCH_LEN, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=ENC_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT,
        pred_len=PRED_LEN, head_hidden=HEAD_HIDDEN
    ).to(DEVICE)

    # Loss / Optim / Scheduler
    crit = PeakAwareLoss(
        peak_quantile=Config.PEAK_THRESHOLD_QUANTILE,
        alpha=Config.PEAK_WEIGHT_ALPHA,
        beta=Config.AMPLITUDE_WEIGHT_BETA,
        pred_len=PRED_LEN,
        horizon_mode=Config.HORIZON_WEIGHT_MODE,
        horizon_exp_scale=Config.HORIZON_EXP_SCALE,
        horizon_tail_boost=Config.HORIZON_TAIL_BOOST,
        horizon_tail_count=Config.HORIZON_TAIL_COUNT
    ).to(DEVICE)
    
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    # ---- history for curves ----
    hist = {"train_loss":[], "val_loss":[], "train_mae":[], "val_mae":[]}

    best_val = float("inf"); best_state=None; noimp=0
    printed_batch_info = False
    for ep in range(1, EPOCHS+1):
        # ---- Train ----
        model.train(); tr_loss_sum=0; tr_mae_sum=0; n=0
        # warmup
        for g in opt.param_groups:
            g['lr'] = warmup_lr(ep, LR, WARMUP_EPOCHS)

        for Xb,yb,_ in dl_tr:
            if not printed_batch_info:
                # Xb: (B, P, L, F)  ← 최종 모델 입력 텐서 구조
                print(f"[Batch] Xb.shape={tuple(Xb.shape)} (B,P,L,F), yb.shape={tuple(yb.shape)}")
                print(f"[Batch] Feature order used -> {feat_names}")
                printed_batch_info = True
            Xb=Xb.to(DEVICE); yb=yb.to(DEVICE)
            opt.zero_grad()
            pred = model(Xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs=yb.size(0)
            tr_loss_sum += loss.item()*bs; n+=bs
            tr_mae_sum  += batch_mae_in_original_units(pred, yb, scaler_y)*bs

        tr_loss = tr_loss_sum / max(1,n)
        tr_mae  = tr_mae_sum  / max(1,n)

        # ---- Validation ----
        model.eval(); va_loss_sum=0; va_mae_sum=0; va_corr_sum = 0; n=0
        with torch.no_grad():
            for Xb,yb,_ in dl_va:
                Xb=Xb.to(DEVICE); yb=yb.to(DEVICE)
                pred = model(Xb); loss = crit(pred, yb)
                bs=yb.size(0)
                va_loss_sum += loss.item()*bs; n+=bs
                va_mae_sum  += batch_mae_in_original_units(pred, yb, scaler_y)*bs
                va_corr_sum += batch_corrcoef(pred, yb, scaler_y)*bs
        va_loss = va_loss_sum / max(1,n)
        va_mae  = va_mae_sum  / max(1,n)
        va_corr = va_corr_sum / max(1,n)

        scheduler.step()

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["train_mae"].append(tr_mae)
        hist["val_mae"].append(va_mae)

        print(f"[Epoch {ep:03d}/{EPOCHS}] "
              f"LR={opt.param_groups[0]['lr']:.6f} | "
              f"Loss T/V={tr_loss:.5f}/{va_loss:.5f} | "
              f"MAE  T/V={tr_mae:.5f}/{va_mae:.5f}"
              f"Corr V={va_corr:.3f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss; noimp=0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            noimp += 1
            if noimp >= PATIENCE:
                print(f"Early stopping after {ep} epochs (no improvement {PATIENCE}).")
                break

    if best_state is not None:
        model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})

    # ---- Test & Metrics ----
    model.eval(); preds=[]; trues=[]; starts=[]
    with torch.no_grad():
        for Xb,yb,i0 in dl_te:
            Xb=Xb.to(DEVICE)
            preds.append(model(Xb).detach().cpu().numpy())
            trues.append(yb.numpy())
            starts.append(i0.numpy())
    yhat_sc = np.concatenate(preds,axis=0)
    ytrue_sc= np.concatenate(trues,axis=0)
    starts  = np.concatenate(starts,axis=0)

    # inverse scale (target only)
    yhat  = scaler_y.inverse_transform(yhat_sc.reshape(-1,1)).reshape(-1,PRED_LEN)
    ytrue = scaler_y.inverse_transform(ytrue_sc.reshape(-1,1)).reshape(-1,PRED_LEN)

    # 성능 평가 지표 계산
    mae  = float(np.mean(np.abs(yhat-ytrue)))
    mse  = float(np.mean((yhat-ytrue)**2))
    rmse = float(np.sqrt(mse))
    
    print("\n" + "="*60)
    print("🎯 최종 테스트 성능 평가")
    print("="*60)
    print(f"MAE  (Mean Absolute Error):      {mae:.6f}")
    print(f"MSE  (Mean Squared Error):       {mse:.6f}")
    print(f"RMSE (Root Mean Squared Error):  {rmse:.6f}")
    print("="*60)

    # =========================
    # Save per-window predictions
    # =========================
    cols_true = [f"true_t+{i}" for i in range(1,PRED_LEN+1)]
    cols_pred = [f"pred_t+{i}" for i in range(1,PRED_LEN+1)]
    out = pd.DataFrame(np.hstack([ytrue, yhat]), columns=cols_true+cols_pred)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved predictions -> {OUT_CSV}")

    # =========================
    # Plot_1: last window (H-step ahead)
    # =========================
    last_true = ytrue[-1]; last_pred = yhat[-1]
    weeks = np.arange(1, PRED_LEN+1)
    plt.figure(figsize=(10,4))
    plt.plot(weeks, last_true, label="Truth (last window)", linewidth=2)
    plt.plot(weeks, last_pred, label="Prediction (last window)", linewidth=2)
    plt.title("Last Test Window: Truth vs Prediction")
    plt.xlabel("Horizon (weeks ahead)")
    plt.ylabel("ILI per 1,000 Population")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_LAST_WINDOW, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight', format='png', pad_inches=0.1)
    print(f"Saved plot -> {PLOT_LAST_WINDOW}")

    # =========================
    # Plot_2: test reconstruction (val-context included)
    # =========================
    context = y_va_sc[-SEQ_LEN:]                       # 표준화 컨텍스트
    y_ct_sc = np.concatenate([context, y_te_sc])       # [SEQ_LEN + test_len]
    # 입력 특징도 컨텍스트 포함해 재구성 필요 → X도 동일하게 붙여서 예측
    X_ct_sc = np.concatenate([X_va_sc[-SEQ_LEN:], X_te_sc], axis=0)
    ds_ct = PatchTSTDataset(X_ct_sc, y_ct_sc, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
    dl_ct = DataLoader(ds_ct, batch_size=BATCH_SIZE, shuffle=False)

    model.eval(); preds_ct=[]; starts_ct=[]
    with torch.no_grad():
        for Xb, _, i0 in dl_ct:
            Xb = Xb.to(DEVICE)
            preds_ct.append(model(Xb).detach().cpu().numpy())  # (B, H)
            starts_ct.append(i0.numpy())
    yhat_ct_sc = np.concatenate(preds_ct, axis=0)
    starts_ct  = np.concatenate(starts_ct, axis=0)
    yhat_ct = scaler_y.inverse_transform(yhat_ct_sc.reshape(-1,1)).reshape(-1, PRED_LEN)

    test_len = len(y_te)
    recon_sum   = np.zeros(test_len)
    recon_count = np.zeros(test_len)
    h_weights = np.linspace(RECON_W_START, RECON_W_END, PRED_LEN)

    for k, s in enumerate(starts_ct):
        pos0_ct = int(s) + SEQ_LEN   # [context+test] 축
        pos0_te = pos0_ct - SEQ_LEN  # test 축으로 변환
        for j in range(PRED_LEN):
            idx = pos0_te + j
            if 0 <= idx < test_len:
                w = h_weights[j]
                recon_sum[idx]   += yhat_ct[k, j] * w
                recon_count[idx] += w

    recon = np.where(recon_count > 0, recon_sum / np.maximum(1, recon_count), np.nan)

    truth_test = y_te
    x_labels = lab_te
    
    # 유효한 데이터만 사용 (NaN 제거)
    valid_mask = ~np.isnan(truth_test) & ~np.isnan(recon)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) < len(truth_test):
        print(f"\n⚠️ Warning: {len(truth_test) - len(valid_indices)} NaN values removed from test reconstruction")
        truth_test = truth_test[valid_mask]
        recon = recon[valid_mask]
        x_labels = [x_labels[i] for i in valid_indices]
    
    test_len = len(truth_test)
    
    # 디버그: 테스트 데이터 범위 출력
    print(f"\n=== Test Segment Info ===")
    print(f"Test length: {test_len}")
    print(f"First label: {x_labels[0]}")
    print(f"Last label: {x_labels[-1]}")
    print(f"Truth range: [{truth_test.min():.2f}, {truth_test.max():.2f}]")
    print(f"Truth mean: {truth_test.mean():.2f}")
    print(f"Prediction range: [{np.nanmin(recon):.2f}, {np.nanmax(recon):.2f}]")
    print(f"Prediction mean: {np.nanmean(recon):.2f}")
    
    # X축 라벨을 더 자주 표시 (간격 조정)
    tick_step = max(1, test_len // 20)  # 약 20개 라벨 표시
    tick_idx  = list(range(0, test_len, tick_step))
    if tick_idx[-1] != test_len-1:
        tick_idx.append(test_len-1)
    tick_text = [x_labels[i] for i in tick_idx]

    plt.figure(figsize=(18,6))  # 그래프 크기 확대
    plt.plot(range(test_len), truth_test, linewidth=2.5, marker='o', markersize=3, 
             label=f"Truth (test segment, n={test_len})", color='darkblue')
    plt.plot(range(test_len), recon, linewidth=2.5, marker='s', markersize=3,
             label="Prediction (overlap-avg, weighted)", color='darkorange')
    plt.title(f"Test Range: Truth vs Prediction | {x_labels[0]} ~ {x_labels[-1]}", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Season - Week", fontsize=12)
    plt.ylabel("ILI per 1,000 Population", fontsize=12)
    plt.xticks(tick_idx, tick_text, rotation=45, ha="right", fontsize=9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='upper left')
    
    # Y축 범위 명시적 설정 (이상값 방지)
    y_min = min(truth_test.min(), np.nanmin(recon))
    y_max = max(truth_test.max(), np.nanmax(recon))
    plt.ylim(y_min * 0.95, y_max * 1.05)
    
    plt.tight_layout()
    plt.savefig(PLOT_TEST_RECON, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight', format='png', pad_inches=0.1)
    print(f"Saved plot -> {PLOT_TEST_RECON}")

    # =========================
    # Plot_3: Train/Val MAE curves
    # =========================
    xs = np.arange(1, len(hist["train_mae"])+1)
    plt.figure(figsize=(10,4))
    plt.plot(xs, hist["train_mae"], linewidth=2, label="Train MAE (original units)")
    plt.plot(xs, hist["val_mae"],   linewidth=2, label="Val MAE (original units)")
    plt.title("Training Curves: MAE per epoch (lower is better)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (ILI per 1,000)")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_MA_CURVES, dpi=300, facecolor='white', edgecolor='none', bbox_inches='tight', format='png', pad_inches=0.1)
    print(f"Saved plot -> {PLOT_MA_CURVES}")


    # =========================
# Feature Importance utils
# =========================
def _eval_mse_on_split(model, X_split_sc, y_split_sc, scaler_y, feat_names,
                       batch_size=BATCH_SIZE):
    """
    Feature Importance용 MSE 계산 (Perturbation-Based Method)
    """
    model.eval()

    # 실제 모델의 pred_len을 사용 (Dual-head 구조에서 head_trend 사용)
    pred_len = model.head_trend.out_features
    seq_len  = SEQ_LEN
    patch_len = PATCH_LEN
    stride = STRIDE

    ds = PatchTSTDataset(
        X_split_sc, y_split_sc,
        seq_len=seq_len,
        pred_len=pred_len,
        patch_len=patch_len,
        stride=stride
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    mse_sum, n = 0.0, 0
    with torch.no_grad():
        for Xb, yb, _ in dl:
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)

            pred = model(Xb)  # (B, H_model)

            # pred / yb shape mismatch 방지
            H = pred.shape[1]
            yb = yb[:, :H]

            # 원래 스케일로 변환
            pred_np = pred.cpu().numpy()
            yb_np = yb.cpu().numpy()
            
            # scaler_y가 2D 입력을 요구하면 reshape
            pred_orig = scaler_y.inverse_transform(pred_np.reshape(-1, 1)).flatten()[:H]
            yb_orig = scaler_y.inverse_transform(yb_np.reshape(-1, 1)).flatten()[:H]
            
            # MSE 계산
            mse = np.mean((pred_orig - yb_orig) ** 2)
            mse_sum += mse * yb.size(0)
            n += yb.size(0)

    return float(mse_sum / max(1, n))


def compute_feature_importance(model, 
                               X_va_sc, y_va_sc, 
                               X_te_sc=None, y_te_sc=None,
                               scaler_y=None, feat_names=None, 
                               random_state=42):
    """
    Perturbation-Based Feature Importance 계산
    
    방법:
    1. 각 변수를 마스킹(평균값으로 대체)
    2. MSE 증가량 측정: Importance(i) = MSE_masked(i) - MSE_original
    3. 중요도 정규화
    
    Note: 'ili'는 타겟 변수이므로 Feature Importance 계산에서 제외됩니다.
    """
    assert scaler_y is not None and feat_names is not None

    # --- 'ili' 제외: 타겟 변수는 feature importance 계산에서 제외 ---
    feat_indices = [i for i, name in enumerate(feat_names) if name != 'ili']
    filtered_feat_names = [feat_names[i] for i in feat_indices]
    
    if len(filtered_feat_names) < len(feat_names):
        print(f"[FI] 'ili' 특징 제외됨 (타겟 변수)")
        print(f"[FI] Feature Importance 계산 대상: {len(filtered_feat_names)}개 특징")

    # --- Step 1: Baseline MSE (원본 데이터) ---
    print(f"[FI] Computing Baseline MSE...")
    mse_original_val = _eval_mse_on_split(model, X_va_sc, y_va_sc, scaler_y, feat_names)
    print(f"[FI] Baseline Val MSE: {mse_original_val:.6f}")

    mse_original_tst = None
    if X_te_sc is not None and y_te_sc is not None:
        mse_original_tst = _eval_mse_on_split(model, X_te_sc, y_te_sc, scaler_y, feat_names)
        print(f"[FI] Baseline Test MSE: {mse_original_tst:.6f}")

    # --- Step 2: 각 변수를 마스킹하여 MSE 증가량 측정 ---
    print(f"[FI] Computing Perturbation Importance...")
    importance_val = []
    importance_tst = []

    for j in feat_indices:
        name = feat_names[j]
        
        # Validation set: 해당 피처를 평균값으로 마스킹
        X_masked_val = X_va_sc.copy()
        X_masked_val[:, j] = X_va_sc[:, j].mean()
        
        mse_masked_val = _eval_mse_on_split(model, X_masked_val, y_va_sc, scaler_y, feat_names)
        delta_mse_val = mse_masked_val - mse_original_val
        importance_val.append(delta_mse_val)
        
        print(f"  - {name}: ΔMSE={delta_mse_val:.6f}")

        # Test set (optional)
        if X_te_sc is not None and y_te_sc is not None:
            X_masked_tst = X_te_sc.copy()
            X_masked_tst[:, j] = X_te_sc[:, j].mean()
            
            mse_masked_tst = _eval_mse_on_split(model, X_masked_tst, y_te_sc, scaler_y, feat_names)
            delta_mse_tst = mse_masked_tst - mse_original_tst
            importance_tst.append(delta_mse_tst)

    # --- Step 3: Normalization ---
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

    # --- DataFrame 생성 ---
    column_mapping = {
        '연도': 'year',
        '주차': 'week',
        '의사환자 분율': 'ili',
        '예방접종률': 'vaccine_rate',
        '입원환자 수': 'hospitalization',
        '인플루엔자 검출률': 'detection_rate',
        '응급실 인플루엔자 환자': 'emergency_patients',
        '아형': 'subtype'
    }
    inv_colmap = {v: k for k, v in column_mapping.items()}

    feature_disp = [f"{f} ({inv_colmap[f]})" if f in inv_colmap else f for f in filtered_feat_names]

    df_fi = pd.DataFrame({
        "feature": feature_disp,
        "importance_raw_val": importance_val,
        "importance_norm_val": importance_norm_val,
    })
    
    if importance_norm_tst is not None:
        df_fi["importance_raw_tst"] = importance_tst
        df_fi["importance_norm_tst"] = importance_norm_tst

    # Raw importance 기준 내림차순 정렬
    df_fi = df_fi.sort_values("importance_raw_val", ascending=False).reset_index(drop=True)
    
    print(f"\n[FI] Feature Importance Calculation Complete!")
    return df_fi

def plot_feature_importance(fi_df, out_csv=None, out_png=None):
    """
    Perturbation-Based Feature Importance를 막대그래프로 시각화
    """
    if fi_df is None or len(fi_df) == 0:
        print("No feature importance data to plot.")
        return

    import matplotlib.pyplot as plt

    # CSV 저장
    if out_csv:
        fi_df.to_csv(out_csv, index=False)
        print(f"Feature Importance saved to {out_csv}")

    # 시각화 (2개 서브플롯: Raw & Normalized)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ① Raw Importance (ΔMSE)
    axes[0].barh(fi_df["feature"], fi_df["importance_raw_val"], color="steelblue")
    axes[0].set_xlabel("ΔMSE (MSE_masked - MSE_original)")
    axes[0].set_title("Perturbation-Based Importance (Raw)")
    axes[0].invert_yaxis()
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=0.8)

    # ② Normalized Importance
    axes[1].barh(fi_df["feature"], fi_df["importance_norm_val"], color="coral")
    axes[1].set_xlabel("Normalized Importance")
    axes[1].set_title("Perturbation-Based Importance (Normalized)")
    axes[1].invert_yaxis()

    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Feature Importance plot saved to {out_png}")
    plt.show()


# =========================
# Hyperparameter Management
# =========================
def get_default_hyperparameters() -> dict:
    """
    Config 클래스의 기본 하이퍼파라미터 반환
    
    Returns:
        기본 하이퍼파라미터 dict
    """
    return {
        'd_model': Config.D_MODEL,
        'n_heads': Config.N_HEADS,
        'enc_layers': Config.ENC_LAYERS,
        'ff_dim': Config.FF_DIM,
        'dropout': Config.DROPOUT,
        'lr': Config.LR,
        'weight_decay': Config.WEIGHT_DECAY,
        'batch_size': Config.BATCH_SIZE,
        'seq_len': Config.SEQ_LEN if not Config.USE_DAILY_DATA else Config.DAILY_SEQ_LEN,
        'patch_len': Config.PATCH_LEN
    }


def load_best_hyperparameters(json_path: str = "best_hyperparameters.json") -> Optional[dict]:
    """
    저장된 best_hyperparameters.json에서 최적 하이퍼파라미터 로드
    
    Args:
        json_path: JSON 파일 경로
        
    Returns:
        하이퍼파라미터 dict 또는 None (파일 없음)
    """
    import json
    import os
    
    if not os.path.exists(json_path):
        return None
    
    try:
        with open(json_path, 'r') as f:
            params = json.load(f)
        
        print(f"\n{'='*70}")
        print(f"✅ 저장된 최적 하이퍼파라미터 로드 성공: {json_path}")
        print(f"{'='*70}")
        print(f"📊 로드된 파라미터:")
        for key, value in params.items():
            print(f"   - {key}: {value}")
        print(f"{'='*70}\n")
        
        return params
    except Exception as e:
        print(f"⚠️  JSON 파일 로드 실패 ({json_path}): {e}")
        return None


def save_best_hyperparameters(params: dict, json_path: str = "best_hyperparameters.json") -> bool:
    """
    최적 하이퍼파라미터를 JSON 파일에 저장
    
    Args:
        params: 하이퍼파라미터 dict
        json_path: 저장할 JSON 파일 경로
        
    Returns:
        저장 성공 여부
    """
    import json
    
    try:
        with open(json_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"✅ 최적 하이퍼파라미터 저장 성공: {json_path}")
        return True
    except Exception as e:
        print(f"❌ JSON 파일 저장 실패: {e}")
        return False


# =========================
# Optuna Optimization
# =========================
def optimize_hyperparameters(X: np.ndarray, y: np.ndarray, labels: list, feat_names: list,
                            n_trials: int = 50):
    """
    Optuna를 사용한 하이퍼파라미터 최적화
    
    Args:
        X: 입력 특징 (N, F)
        y: 타겟 변수 (N,)
        labels: 시간 라벨
        feat_names: 특징 이름
        n_trials: 최적화 시도 횟수
        
    Returns:
        best_params: 최적 하이퍼파라미터 dict
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not installed. Install with: pip install optuna")
    
    print("\n" + "=" * 70)
    print("🔍 Optuna 하이퍼파라미터 최적화 시작")
    if Config.USE_DAILY_DATA:
        print(f"   📅 일별 데이터 모드 (SEQ_LEN={SEQ_LEN}, PRED_LEN={PRED_LEN})")
        print(f"   ⚙️  시퀀스 길이는 고정값 사용 (하이퍼파라미터 탐색 제외)")
    else:
        print(f"   📆 주차별 데이터 모드 (SEQ_LEN={SEQ_LEN}, PRED_LEN={PRED_LEN})")
        print(f"   ⚙️  시퀀스 길이도 하이퍼파라미터 탐색 대상")
    print("=" * 70)
    
    def objective(trial: Trial) -> float:
        """Optuna objective function - validation MAE를 최소화"""
        
        # Trial 시작 알림
        if Config.USE_DAILY_DATA and trial.number == 0:
            print(f"\n   💡 Trial {trial.number}: 일별 데이터 모드로 학습 시작 (seq_len=112 고정)")
        
        # Config에서 탐색 공간 가져오기
        search_space = Config.OPTUNA_SEARCH_SPACE
        
        # 하이퍼파라미터 샘플링
        # 하이퍼파라미터 샘플링: search_space에 키가 없으면 Config의 기본값 사용
        params = {}
        params['d_model'] = trial.suggest_categorical('d_model', search_space['d_model'])
        params['n_heads'] = trial.suggest_categorical('n_heads', search_space['n_heads'])
        params['enc_layers'] = trial.suggest_int('enc_layers', *search_space['enc_layers'])
        params['ff_dim'] = trial.suggest_categorical('ff_dim', search_space['ff_dim'])
        params['dropout'] = trial.suggest_float('dropout', *search_space['dropout'])
        params['lr'] = trial.suggest_float('lr', *search_space['lr'], log=True)
        params['weight_decay'] = trial.suggest_float('weight_decay', *search_space['weight_decay'], log=True)
        params['batch_size'] = trial.suggest_categorical('batch_size', search_space['batch_size'])

        # seq_len / pred_len: 일별 데이터일 때는 고정값 사용
        if Config.USE_DAILY_DATA:
            # 일별 데이터: SEQ_LEN, PRED_LEN 고정 (하이퍼파라미터 탐색 안 함)
            params['seq_len'] = SEQ_LEN   # 112일
            params['pred_len'] = PRED_LEN # 28일
            print(f"   [일별 데이터 모드] seq_len={SEQ_LEN}, pred_len={PRED_LEN} 고정")
        else:
            # 주차별 데이터: 하이퍼파라미터 탐색
            if 'seq_len' in search_space:
                params['seq_len'] = trial.suggest_categorical('seq_len', search_space['seq_len'])
            else:
                params['seq_len'] = SEQ_LEN

            if 'pred_len' in search_space:
                params['pred_len'] = trial.suggest_categorical('pred_len', search_space['pred_len'])
            else:
                params['pred_len'] = PRED_LEN

        # patch_len: 일별/주차별 모두 탐색
        if 'patch_len' in search_space:
            params['patch_len'] = trial.suggest_categorical('patch_len', search_space['patch_len'])
        else:
            params['patch_len'] = PATCH_LEN
        
        # d_model은 4의 배수여야 함 (MultiScaleCNN 분기 4개)
        if params['d_model'] % 4 != 0:
            params['d_model'] = (params['d_model'] // 4) * 4
        
        # n_heads는 d_model의 약수여야 함
        while params['d_model'] % params['n_heads'] != 0:
            params['n_heads'] //= 2
            if params['n_heads'] < 1:
                params['n_heads'] = 1
                break
        
        # 데이터 분할
        (s0, e0), (s1, e1), (s2, e2) = make_splits(len(y))
        X_tr, X_va = X[s0:e0], X[s1:e1]
        y_tr, y_va = y[s0:e0], y[s1:e1]
        
        # Scaling (타겟: Log 변환 적용, 피처: Log 변환 미적용)
        scaler_y = get_scaler(for_target=True)
        y_tr_sc = scaler_y.fit_transform(y_tr.reshape(-1,1)).ravel()
        y_va_sc = scaler_y.transform(y_va.reshape(-1,1)).ravel()
        
        scaler_x = get_scaler(for_target=False)
        X_tr_sc = scaler_x.fit_transform(X_tr)
        X_va_sc = scaler_x.transform(X_va)
        
        F = X.shape[1]
        
        # Dataset 생성
        try:
            ds_tr = PatchTSTDataset(X_tr_sc, y_tr_sc, params['seq_len'], params['pred_len'], 
                                   params['patch_len'], STRIDE)
            ds_va = PatchTSTDataset(X_va_sc, y_va_sc, params['seq_len'], params['pred_len'],
                                   params['patch_len'], STRIDE)
        except:
            # 데이터가 부족한 경우
            return float('inf')
        
        if len(ds_tr) < 1 or len(ds_va) < 1:
            return float('inf')
        
        dl_tr = DataLoader(ds_tr, batch_size=params['batch_size'], shuffle=True, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=params['batch_size'], shuffle=False)
        
        # 모델 생성
        model = PatchTSTModel(
            in_features=F, patch_len=params['patch_len'], d_model=params['d_model'],
            n_heads=params['n_heads'], n_layers=params['enc_layers'], ff_dim=params['ff_dim'],
            dropout=params['dropout'], pred_len=params['pred_len'], head_hidden=HEAD_HIDDEN
        ).to(DEVICE)
        
        crit = nn.HuberLoss(delta=1.0)
        opt = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        
        # Early stopping을 위한 변수
        best_val_loss = float('inf')
        patience_count = 0
        early_stop_patience = 20  # Optuna에서는 더 짧게
        
        # 학습 (Optuna에서는 적은 에포크)
        max_epochs = 50
        for ep in range(1, max_epochs + 1):
            # Train
            model.train()
            tr_loss_sum = 0
            n = 0
            for Xb, yb, _ in dl_tr:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                pred = model(Xb)
                loss = crit(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                bs = yb.size(0)
                tr_loss_sum += loss.item() * bs
                n += bs
            
            tr_loss = tr_loss_sum / max(1, n)
            
            # Validation
            model.eval()
            va_loss_sum = 0
            va_mae_sum = 0
            n = 0
            # Peak MAE 계산을 위한 예측값 수집
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for Xb, yb, _ in dl_va:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    pred = model(Xb)
                    loss = crit(pred, yb)
                    bs = yb.size(0)
                    va_loss_sum += loss.item() * bs
                    va_mae_sum += batch_mae_in_original_units(pred, yb, scaler_y) * bs
                    n += bs
                    
                    # 원본 스케일로 변환하여 저장 (Peak MAE 계산용)
                    pred_orig = scaler_y.inverse_transform(pred.cpu().numpy().reshape(-1, 1)).ravel()
                    target_orig = scaler_y.inverse_transform(yb.cpu().numpy().reshape(-1, 1)).ravel()
                    all_preds.extend(pred_orig)
                    all_targets.extend(target_orig)
            
            va_loss = va_loss_sum / max(1, n)
            va_mae = va_mae_sum / max(1, n)
            
            # 🔴 Peak MAE 계산 (train 데이터 기준 상위 10% threshold)
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            peak_threshold = np.quantile(y_tr, 0.9)  # train 데이터 기준 피크
            peak_mask = all_targets >= peak_threshold
            
            if peak_mask.sum() > 0:  # 피크 데이터가 있는 경우
                peak_mae = np.mean(np.abs(all_preds[peak_mask] - all_targets[peak_mask]))
            else:
                peak_mae = 0.0  # 피크 없으면 0
            
            # 🔴 복합 목적 함수: 전체 MAE + 피크 MAE
            combined_metric = va_mae + 0.6 * peak_mae
            
            # Early stopping (복합 metric 기준)
            if combined_metric < best_val_loss:
                best_val_loss = combined_metric
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= early_stop_patience:
                    break
            
            # Optuna pruning (중간 결과가 나쁘면 조기 종료)
            trial.report(combined_metric, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # 🔴 복합 Metric 반환 (Val MAE + 0.6 * Peak MAE)
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
    print("=" *  70)
    print(f"\n🏆 Best Trial:")
    print(f"  - Value (Val MAE + 0.6*Peak MAE): {study.best_trial.value:.4f}")
    print(f"\n📊 Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    
    # Best parameters 저장
    best_params_file = BASE_DIR / "best_hyperparameters.json"
    import json
    with open(best_params_file, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\n💾 Best parameters saved to: {best_params_file}")
    
    return study.best_params

# =========================
# train_and_eval (main)
# =========================
def train_and_eval(X: np.ndarray, y: np.ndarray, labels: list, feat_names: list,
                   compute_fi=False, save_fi=False, optuna_params=None):
    """
    통합 학습 + 평가 함수.
    compute_fi=True -> feature importance 계산
    save_fi=True -> CSV/plot 저장
    optuna_params=dict -> Optuna 최적화된 파라미터 사용
    """
    # Optuna 파라미터가 있으면 적용
    if optuna_params:
        global D_MODEL, N_HEADS, ENC_LAYERS, FF_DIM, DROPOUT, LR, WEIGHT_DECAY, BATCH_SIZE, SEQ_LEN, PRED_LEN, PATCH_LEN
        D_MODEL = optuna_params.get('d_model', D_MODEL)
        N_HEADS = optuna_params.get('n_heads', N_HEADS)
        ENC_LAYERS = optuna_params.get('enc_layers', ENC_LAYERS)
        FF_DIM = optuna_params.get('ff_dim', FF_DIM)
        DROPOUT = optuna_params.get('dropout', DROPOUT)
        LR = optuna_params.get('lr', LR)
        WEIGHT_DECAY = optuna_params.get('weight_decay', WEIGHT_DECAY)
        BATCH_SIZE = optuna_params.get('batch_size', BATCH_SIZE)
        SEQ_LEN = optuna_params.get('seq_len', SEQ_LEN)
        PRED_LEN = optuna_params.get('pred_len', PRED_LEN)
        PATCH_LEN = optuna_params.get('patch_len', PATCH_LEN)
        
        print("\n" + "=" * 70)
        print("🎯 Optuna 최적 파라미터로 최종 학습")
        print("=" * 70)
        for key, value in optuna_params.items():
            print(f"  - {key}: {value}")
        print("=" * 70 + "\n")
    
    """
    통합 학습 + 평가 함수.
    compute_fi=True -> feature importance 계산
    save_fi=True -> CSV/plot 저장
    """
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"[Config] EPOCHS:{EPOCHS}, BATCH_SIZE:{BATCH_SIZE}, SEQ_LEN:{SEQ_LEN}, PRED_LEN:{PRED_LEN}")
    print(f"[Config] PATCH_LEN:{PATCH_LEN}, STRIDE:{STRIDE}, LR:{LR}, Warmup:{WARMUP_EPOCHS}, Patience:{PATIENCE}")
    print(f"[Config] Log Transform: {Config.USE_LOG_TRANSFORM} (eps={Config.LOG_EPSILON}), Peak Weight: α={Config.PEAK_WEIGHT_ALPHA}, Quantile={Config.PEAK_THRESHOLD_QUANTILE}")

    N = len(y)
    split_tr = int(0.7*N); split_va = int(0.85*N)
    X_tr, y_tr = X[:split_tr], y[:split_tr]
    X_va, y_va = X[split_tr:split_va], y[split_tr:split_va]
    X_te, y_te = X[split_va:], y[split_va:]

    # 전역 get_scaler 함수 사용 (Log 변환은 타겟만)
    scaler_y = get_scaler(for_target=True)  # 타겟: Log 변환 적용
    y_tr_sc = scaler_y.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_va_sc = scaler_y.transform(y_va.reshape(-1,1)).ravel()
    y_te_sc = scaler_y.transform(y_te.reshape(-1,1)).ravel()

    scaler_x = get_scaler(for_target=False)  # 피처: Log 변환 미적용
    X_tr_sc = scaler_x.fit_transform(X_tr)
    X_va_sc = scaler_x.transform(X_va)
    X_te_sc = scaler_x.transform(X_te)

    F = X.shape[1]
    print(f"[Shapes] X_tr:{X_tr.shape}, X_va:{X_va.shape}, X_te:{X_te.shape} | F={F}")
    print(f"[Info] Model input feature order -> {feat_names}")

    ds_tr = PatchTSTDataset(X_tr_sc, y_tr_sc, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
    ds_va = PatchTSTDataset(X_va_sc, y_va_sc, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
    ds_te = PatchTSTDataset(X_te_sc, y_te_sc, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False)

    model = PatchTSTModel(
        in_features=F, patch_len=PATCH_LEN, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=ENC_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT,
        pred_len=PRED_LEN, head_hidden=HEAD_HIDDEN
    ).to(DEVICE)

    def peak_weighted_loss(pred, target, peak_quantile=0.9, alpha=3.0):
        """
        Peak-aware weighted MAE loss.
        pred, target: (B, H)
        """
        with torch.no_grad():
            thresh = torch.quantile(target, peak_quantile)
            weights = torch.ones_like(target)
            weights[target >= thresh] = alpha
        return torch.mean(weights * torch.abs(pred - target))

    crit = peak_weighted_loss
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    hist = {"train_loss":[], "val_loss":[], "train_mae":[], "val_mae":[]}

    best_val = float("inf"); best_state=None; noimp=0
    printed_batch_info = False
    for ep in range(1, EPOCHS+1):
        model.train(); tr_loss_sum=0; tr_mae_sum=0; n=0
        for g in opt.param_groups:
            g['lr'] = warmup_lr(ep, LR, WARMUP_EPOCHS)

        for Xb, yb, _ in dl_tr:
            Xb=Xb.to(DEVICE); yb=yb.to(DEVICE)
            if not printed_batch_info:
                print(f"[Batch shapes] Xb:{Xb.shape}, yb:{yb.shape}")
                printed_batch_info=True
            opt.zero_grad()
            pred=model(Xb)
            loss=crit(pred,yb)
            loss.backward(); opt.step()

            tr_loss_sum += loss.item()*yb.size(0)
            tr_mae_sum += batch_mae_in_original_units(pred, yb, scaler_y)*yb.size(0)
            n+=yb.size(0)

        tr_loss_avg = tr_loss_sum/max(1,n)
        tr_mae_avg  = tr_mae_sum/max(1,n)

        model.eval(); va_loss_sum=0; va_mae_sum=0; m=0
        with torch.no_grad():
            for Xb,yb,_ in dl_va:
                Xb=Xb.to(DEVICE); yb=yb.to(DEVICE)
                pred=model(Xb)
                loss=crit(pred,yb)
                va_loss_sum += loss.item()*yb.size(0)
                va_mae_sum  += batch_mae_in_original_units(pred,yb,scaler_y)*yb.size(0)
                m+=yb.size(0)
        va_loss_avg=va_loss_sum/max(1,m)
        va_mae_avg =va_mae_sum/max(1,m)

        hist["train_loss"].append(tr_loss_avg)
        hist["val_loss"].append(va_loss_avg)
        hist["train_mae"].append(tr_mae_avg)
        hist["val_mae"].append(va_mae_avg)

        if ep<=5 or ep%5==0:
            print(f"Epoch {ep:3d}/{EPOCHS} | TrL:{tr_loss_avg:.6f} TrMAE:{tr_mae_avg:.6f} | VaL:{va_loss_avg:.6f} VaMAE:{va_mae_avg:.6f}")

        if va_mae_avg < best_val:
            best_val = va_mae_avg
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            noimp=0
        else:
            noimp+=1
            if noimp>=PATIENCE:
                print(f"Early stop at epoch {ep} (no improvement for {PATIENCE} epochs)")
                break

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best Val MAE: {best_val:.6f}")

    # Test - 모든 성능 지표 계산 + Horizon별 예측값 수집
    model.eval()
    te_mae_sum=0; te_mse_sum=0; te_rmse_sum=0; k=0
    all_preds = []  # 모든 예측값 수집
    all_trues = []  # 모든 실제값 수집
    
    with torch.no_grad():
        for Xb,yb,_ in dl_te:
            Xb=Xb.to(DEVICE); yb=yb.to(DEVICE)
            pred=model(Xb)
            te_mae_sum += batch_mae_in_original_units(pred,yb,scaler_y)*yb.size(0)
            te_mse_sum += batch_mse_in_original_units(pred,yb,scaler_y)*yb.size(0)
            te_rmse_sum += batch_rmse_in_original_units(pred,yb,scaler_y)*yb.size(0)
            k+=yb.size(0)
            
            # 예측값/실제값 수집 (원본 스케일로 변환)
            pred_np = pred.cpu().numpy()
            yb_np = yb.cpu().numpy()
            pred_orig = scaler_y.inverse_transform(pred_np.reshape(-1,1)).reshape(-1, PRED_LEN)
            yb_orig = scaler_y.inverse_transform(yb_np.reshape(-1,1)).reshape(-1, PRED_LEN)
            all_preds.append(pred_orig)
            all_trues.append(yb_orig)
    
    te_mae_avg = te_mae_sum/max(1,k)
    te_mse_avg = te_mse_sum/max(1,k)
    te_rmse_avg = te_rmse_sum/max(1,k)
    
    # 모든 예측값/실제값 병합
    all_preds = np.concatenate(all_preds, axis=0)  # (N, PRED_LEN)
    all_trues = np.concatenate(all_trues, axis=0)  # (N, PRED_LEN)
    
    print("\n" + "="*60)
    print("🎯 최종 테스트 성능 평가")
    print("="*60)
    print(f"MAE  (Mean Absolute Error):      {te_mae_avg:.6f}")
    print(f"MSE  (Mean Squared Error):       {te_mse_avg:.6f}")
    print(f"RMSE (Root Mean Squared Error):  {te_rmse_avg:.6f}")
    print("="*60)
    
    # ===== Horizon별 예측값 (최신 예측 시점 기준) =====
    print("\n" + "="*60)
    print("📅 최신 예측 시점 기준 Horizon별 예측값")
    print("="*60)
    
    # 가장 최근 예측 시점 (마지막 샘플)
    last_idx = len(all_preds) - 1
    last_pred = all_preds[last_idx]  # 마지막 예측 시점의 예측값들 (PRED_LEN개)
    last_true = all_trues[last_idx]  # 마지막 예측 시점의 실제값들 (PRED_LEN개)
    
    print(f"\n📍 예측 시작 시점: 테스트 데이터 마지막 샘플 (index {last_idx})")
    print(f"   (이 시점에서 향후 {PRED_LEN}주를 예측)")
    print()
    
    horizons_to_check = [1, 2, 3, 4]  # 1주, 2주, 3주, 4주 후
    
    print(f"{'Horizon':<12} {'예측값':>12} {'실제값':>12} {'오차':>12} {'오차율':>10}")
    print("-" * 60)
    
    for h in horizons_to_check:
        if h <= PRED_LEN:
            h_idx = h - 1  # 0-indexed
            pred_val = last_pred[h_idx]
            true_val = last_true[h_idx]
            error = pred_val - true_val
            error_pct = (error / true_val * 100) if true_val != 0 else 0
            
            print(f"{h}주 후 ({h*7}일)  {pred_val:>12.2f} {true_val:>12.2f} {error:>+12.2f} {error_pct:>+9.1f}%")
    
    print("-" * 60)
    
    # 전체 테스트 기간에 대한 Horizon별 성능 통계 (참고용)
    print(f"\n📊 전체 테스트 기간 Horizon별 성능 (참고):")
    for h in horizons_to_check:
        if h <= PRED_LEN:
            h_idx = h - 1
            h_preds = all_preds[:, h_idx]
            h_trues = all_trues[:, h_idx]
            h_mae = np.mean(np.abs(h_preds - h_trues))
            print(f"   {h}주 후: MAE={h_mae:.2f}")
    
    print("\n" + "="*60)
    
    # ===== Horizon별 결과 CSV 저장 =====
    horizon_results = []
    for i in range(len(all_preds)):
        row = {'sample_idx': i}
        for h in range(1, PRED_LEN + 1):
            row[f'pred_{h}w'] = all_preds[i, h-1]
            row[f'true_{h}w'] = all_trues[i, h-1]
            row[f'error_{h}w'] = all_preds[i, h-1] - all_trues[i, h-1]
        horizon_results.append(row)
    
    df_horizon = pd.DataFrame(horizon_results)
    horizon_csv_path = str(BASE_DIR / "horizon_predictions.csv")
    df_horizon.to_csv(horizon_csv_path, index=False)
    print(f"📊 Horizon별 예측 결과 저장: {horizon_csv_path}")

    # Plot curves
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(hist["train_mae"],label="Train MAE")
    plt.plot(hist["val_mae"],label="Val MAE")
    plt.xlabel("Epoch"); plt.ylabel("MAE (original units)"); plt.legend(); plt.title("MAE curves")
    plt.subplot(1,2,2)
    plt.plot(hist["train_loss"],label="Train Loss")
    plt.plot(hist["val_loss"],label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Huber Loss"); plt.legend(); plt.title("Loss curves")
    plt.tight_layout()
    plt.savefig(PLOT_MA_CURVES, dpi=150)
    print(f"MAE/loss curves saved to {PLOT_MA_CURVES}")
    plt.close()  # 창을 닫아 메모리 절약

    # Last window
    last_seq_idx = len(y_te_sc) - SEQ_LEN - PRED_LEN
    if last_seq_idx >= 0:
        seq_X = X_te_sc[last_seq_idx:last_seq_idx+SEQ_LEN]  # (SEQ_LEN, F)
        
        # Patchify: Dataset의 __getitem__과 동일한 로직
        patches = []
        pos = 0
        while pos + PATCH_LEN <= SEQ_LEN:
            patches.append(seq_X[pos:pos+PATCH_LEN, :])  # (PATCH_LEN, F)
            pos += STRIDE
        X_patch = np.stack(patches, axis=0)  # (P, PATCH_LEN, F)
        
        # Tensor로 변환하고 batch 차원 추가
        seq_t = torch.from_numpy(X_patch).unsqueeze(0).float().to(DEVICE)  # (1, P, PATCH_LEN, F)
        
        model.eval()
        with torch.no_grad():
            p = model(seq_t).cpu().numpy().ravel()
        p_orig = scaler_y.inverse_transform(p.reshape(-1,1)).ravel()
        y_true_last = scaler_y.inverse_transform(y_te_sc[last_seq_idx+SEQ_LEN:last_seq_idx+SEQ_LEN+PRED_LEN].reshape(-1,1)).ravel()
        plt.figure(figsize=(8,4))
        plt.plot(range(len(y_true_last)), y_true_last, marker='o', label="True")
        plt.plot(range(len(p_orig)), p_orig, marker='x', label="Pred")
        plt.xlabel("Future step (horizon)"); plt.ylabel("ILI")
        plt.title(f"Last window prediction (SEQ_LEN={SEQ_LEN}, PRED_LEN={PRED_LEN})")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(PLOT_LAST_WINDOW, dpi=150)
        print(f"Last window plot saved to {PLOT_LAST_WINDOW}")
        plt.close()  # 창을 닫아 메모리 절약

    # =========================
    # Plot: Test Range (Overlap-Avg, Weighted)
    # =========================
    context = y_va_sc[-SEQ_LEN:]                       # validation context
    y_ct_sc = np.concatenate([context, y_te_sc])       # context + test
    X_ct_sc = np.concatenate([X_va_sc[-SEQ_LEN:], X_te_sc], axis=0)

    ds_ct = PatchTSTDataset(X_ct_sc, y_ct_sc, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
    dl_ct = DataLoader(ds_ct, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    preds_ct, starts_ct = [], []
    with torch.no_grad():
        for Xb, _, i0 in dl_ct:
            Xb = Xb.to(DEVICE)
            preds_ct.append(model(Xb).cpu().numpy())
            starts_ct.append(i0.numpy())

    preds_ct = np.concatenate(preds_ct, axis=0)    # (N,H)
    starts_ct = np.concatenate(starts_ct, axis=0)

    preds_ct_orig = scaler_y.inverse_transform(
        preds_ct.reshape(-1,1)
    ).reshape(-1, PRED_LEN)

    test_len = len(y_te)
    recon_sum = np.zeros(test_len)
    recon_cnt = np.zeros(test_len)

    # horizon weights (early step emphasized)
    h_weights = np.linspace(RECON_W_START, RECON_W_END, PRED_LEN)

    for k, s in enumerate(starts_ct):
        base = int(s) + SEQ_LEN - SEQ_LEN
        for h in range(PRED_LEN):
            idx = base + h
            if 0 <= idx < test_len:
                recon_sum[idx] += preds_ct_orig[k, h] * h_weights[h]
                recon_cnt[idx] += h_weights[h]

    recon = np.where(recon_cnt > 0, recon_sum / recon_cnt, np.nan)

    truth = y_te
    labels_te = labels[len(y) - len(y_te):]

    valid = ~np.isnan(recon)
    recon = recon[valid]
    truth = truth[valid]
    labels_te = [labels_te[i] for i in np.where(valid)[0]]

    plt.figure(figsize=(18,6))
    plt.plot(truth, linewidth=2.5, marker='o', markersize=3,
             label=f"Truth (test segment, n={len(truth)})", color="navy")
    plt.plot(recon, linewidth=2.5, marker='s', markersize=3,
             label="Prediction (overlap-avg, weighted)", color="darkorange")

    plt.title(
        f"Test Range: Truth vs Prediction | {labels_te[0]} ~ {labels_te[-1]}",
        fontsize=14, fontweight="bold"
    )
    plt.xlabel("Season - Week", fontsize=12)
    plt.ylabel("ILI per 1,000 Population", fontsize=12)

    tick_step = max(1, len(labels_te) // 20)
    tick_idx = list(range(0, len(labels_te), tick_step))
    if tick_idx[-1] != len(labels_te)-1:
        tick_idx.append(len(labels_te)-1)

    plt.xticks(tick_idx, [labels_te[i] for i in tick_idx],
               rotation=45, ha="right", fontsize=9)

    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(loc="upper left", fontsize=11)
    plt.tight_layout()

    plt.savefig(PLOT_TEST_RECON, dpi=300, bbox_inches="tight")
    print(f"Saved plot -> {PLOT_TEST_RECON}")
    plt.close()

    # Feature importance
    fi_df = None
    if compute_fi:
        print("\n[Computing Feature Importance...]")
        fi_df = compute_feature_importance(
            model, X_va_sc, y_va_sc, X_te_sc, y_te_sc,
            scaler_y, feat_names, random_state=SEED
        )
        print("\n[Feature Importance (sorted by mean_delta_val)]")
        print(fi_df.to_string(index=False))

        if save_fi:
            plot_feature_importance(
                fi_df,
                out_csv=str(BASE_DIR / "feature_importance.csv"),
                out_png=str(BASE_DIR / "feature_importance.png")
            )

    # 반환: 외부 셀에서 재활용 가능하도록
    return model, X_va_sc, y_va_sc, X_te_sc, y_te_sc, scaler_y, feat_names, fi_df

# =========================
# 실행부 (결과 출력)
# =========================
if __name__ == "__main__":
    import argparse
    
    # 환경변수에서 기본값 로드
    env_age_group = os.getenv('AGE_GROUP', '').strip() or None
    env_subtype = os.getenv('SUBTYPE', '').strip() or None
    env_subtype_only = os.getenv('SUBTYPE_ONLY', 'false').lower() == 'true'
    env_raw_data = os.getenv('USE_RAW_DATA', 'false').lower() == 'true'
    env_data_dir = os.getenv('DATA_DIR', 'data/before')
    
    parser = argparse.ArgumentParser(
        description='PatchTST 인플루엔자 예측 모델',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
환경변수 설정 (.env 파일):
  AGE_GROUP      연령대 (예: 19-49세, 65세이상)
  SUBTYPE        아형 (A 또는 B)
  SUBTYPE_ONLY   아형별 예측 모드 (true/false)
  USE_RAW_DATA   원본 CSV 사용 (true/false)
  DATA_DIR       원본 데이터 디렉토리

명령줄 인자가 환경변수보다 우선합니다.
""")
    parser.add_argument('--age-group', type=str, default=env_age_group,
                        help=f'연령대 선택 (예: 19-49세, 65세이상, 0-6세). 환경변수: AGE_GROUP={env_age_group}')
    parser.add_argument('--subtype', type=str, default=env_subtype,
                        help=f'아형 선택 (A, B). 환경변수: SUBTYPE={env_subtype}')
    parser.add_argument('--subtype-only', action='store_true', default=env_subtype_only,
                        help=f'아형별 검출률만 예측 (ds_0107 데이터 사용). 환경변수: SUBTYPE_ONLY={env_subtype_only}')
    parser.add_argument('--raw-data', action='store_true', default=env_raw_data,
                        help=f'원본 CSV 데이터에서 직접 로드. 환경변수: USE_RAW_DATA={env_raw_data}')
    parser.add_argument('--data-dir', type=str, default=env_data_dir,
                        help=f'원본 데이터 디렉토리. 환경변수: DATA_DIR={env_data_dir}')
    parser.add_argument('--list-options', action='store_true',
                        help='사용 가능한 연령대와 아형 목록 출력')
    parser.add_argument('--validate-data', action='store_true',
                        help='merged CSV와 원본 데이터 필터링 결과 비교 검증 (특정 연령대 또는 전체)')
    parser.add_argument('--validate-all', action='store_true',
                        help='모든 주요 연령대에 대해 데이터 소스 비교 검증 (--validate-data와 함께 사용)')
    args = parser.parse_args()
    
    # 현재 설정 출력
    print("\n" + "=" * 60)
    print("📋 현재 모델 설정")
    print("=" * 60)
    print(f"   연령대 (AGE_GROUP): {args.age_group or '전체 (미지정)'} {'[env]' if args.age_group == env_age_group and env_age_group else ''}")
    print(f"   아형 (SUBTYPE): {args.subtype or '우세 아형 자동 선택'} {'[env]' if args.subtype == env_subtype and env_subtype else ''}")
    print(f"   아형 전용 모드 (SUBTYPE_ONLY): {args.subtype_only} {'[env]' if args.subtype_only == env_subtype_only else ''}")
    print(f"   원본 데이터 사용 (USE_RAW_DATA): {args.raw_data} {'[env]' if args.raw_data == env_raw_data else ''}")
    print(f"   데이터 디렉토리 (DATA_DIR): {args.data_dir} {'[env]' if args.data_dir == env_data_dir else ''}")
    print("=" * 60)
    
    # --validate-data 옵션: 데이터 소스 비교 검증 후 종료
    if args.validate_data:
        print("\n🔍 데이터 소스 비교 검증 모드")
        if args.validate_all:
            # 모든 연령대 검증
            validate_all_age_groups(
                data_dir=args.data_dir,
                merged_csv_path="merged_influenza_data.csv"
            )
        elif args.age_group:
            # 특정 연령대만 검증
            validate_data_sources(
                age_group=args.age_group,
                data_dir=args.data_dir,
                merged_csv_path="merged_influenza_data.csv",
                verbose=True
            )
        else:
            # 환경변수에 연령대가 없으면 모든 연령대 검증
            validate_all_age_groups(
                data_dir=args.data_dir,
                merged_csv_path="merged_influenza_data.csv"
            )
        exit(0)
    
    # --list-options 옵션: 사용 가능한 옵션 출력 후 종료
    if args.list_options:
        print("\n📋 사용 가능한 옵션:")
        
        # 원본 데이터에서 연령대 목록 조회
        age_info = get_available_age_groups(args.data_dir)
        
        print(f"\n📂 원본 데이터 연령대 (--raw-data 모드):")
        for dsid, ages in age_info.items():
            print(f"   {dsid}: {ages}")
        
        # 공통 연령대 찾기
        if age_info:
            common_ages = set(age_info.get('ds_0101', []))
            for ages in age_info.values():
                common_ages &= set(ages)
            print(f"\n📊 공통 연령대 (모든 데이터셋에서 사용 가능):")
            for ag in sorted(common_ages):
                print(f"   - {ag}")
        
        print(f"\n아형 (--subtype-only --subtype <A|B>):")
        print(f"   - A: 인플루엔자 A형")
        print(f"   - B: 인플루엔자 B형")
        
        # ds_0107 아형별 데이터 미리보기
        df_subtype = load_subtype_data(subtype='all')
        if not df_subtype.empty:
            print(f"\n아형별 검출률 데이터 (ds_0107):")
            for st in df_subtype['subtype'].unique():
                count = len(df_subtype[df_subtype['subtype'] == st])
                print(f"   - {st}: {count}개 레코드")
        
        exit(0)
    
    print("\n" + "🚀 " * 20)
    print("데이터 로드 및 모델 학습 시작!")
    print("🚀 " * 20 + "\n")
    
    # 아형별 검출률만 예측하는 모드 (ds_0107)
    if args.subtype_only:
        if not args.subtype:
            print("⚠️ --subtype-only 옵션 사용 시 --subtype (A 또는 B)를 지정해야 합니다.")
            exit(1)
        
        print("=" * 60)
        print(f"🧬 아형별 검출률 예측 모드: {args.subtype}형")
        print("=" * 60)
        
        # 아형별 데이터 준비
        X, y, labels, feat_names = prepare_subtype_data(subtype=args.subtype, data_dir=args.data_dir)
        
        print(f"\n📊 아형 {args.subtype} 검출률 데이터:")
        print(f"   - Data points: {len(y)}")
        print(f"   - Features: {feat_names}")
        
        # 모델 학습 및 평가
        best_params = None
        
        # USE_OPTUNA 플래그에 따라 처리
        if USE_OPTUNA and OPTUNA_AVAILABLE:
            # Optuna로 최적화 실행
            best_params = optimize_hyperparameters(X, y, labels, feat_names, n_trials=N_TRIALS)
            # Optuna 최적화 결과 저장
            if best_params:
                save_best_hyperparameters(best_params)
        else:
            # Optuna 사용 안 함
            if USE_OPTUNA and not OPTUNA_AVAILABLE:
                print("\n⚠️ Optuna가 설치되지 않았습니다 (USE_OPTUNA=True)")
                print("   설치 명령: pip install optuna")
            
            # JSON 파일에서 로드 시도
            best_params = load_best_hyperparameters()
            
            # JSON 파일이 없으면 기본값 사용
            if best_params is None:
                print("\n📋 JSON 파일 없음 → Config의 기본 하이퍼파라미터 사용")
                best_params = get_default_hyperparameters()
                print("기본 하이퍼파라미터:")
                for key, value in best_params.items():
                    print(f"   - {key}: {value}")
        
        model, X_va_sc, y_va_sc, X_te_sc, y_te_sc, scaler_y, feat_names, fi_df = train_and_eval(
            X, y, labels, feat_names,
            compute_fi=True,
            save_fi=True,
            optuna_params=best_params
        )
        
        print(f"\n=== 아형 {args.subtype} 검출률 예측 결과 ===")
        print(f"Feature 개수: {len(feat_names)}")
        exit(0)
    
    # ===== 연령대별 원본 데이터 모드 =====
    if args.raw_data or args.age_group:
        # 연령대 지정 안 했으면 기본값 사용
        age_group = args.age_group or '19-49세'
        
        print("=" * 60)
        print(f"📂 원본 데이터 모드: 연령대 '{age_group}' 데이터 로드")
        print("=" * 60)
        
        # 원본 데이터에서 직접 로드 및 전처리
        X, y, labels, feat_names = load_and_prepare_by_age(
            age_group=age_group,
            data_dir=args.data_dir,
            use_exog=USE_EXOG
        )
        
        print(f"\n✅ 전처리 완료!")
        print(f"   - Data points: {len(y)}")
        print(f"   - Features used ({len(feat_names)}): {feat_names}")
        
        best_params = None
        
        # USE_OPTUNA 플래그에 따라 처리
        if USE_OPTUNA:
            if not OPTUNA_AVAILABLE:
                print("\n⚠️ Optuna가 설치되지 않았습니다 (USE_OPTUNA=True)")
                print("   설치 명령: pip install optuna")
                # Optuna가 없으면 기본값 사용
                best_params = get_default_hyperparameters()
            else:
                # Optuna로 최적화 실행
                best_params = optimize_hyperparameters(X, y, labels, feat_names, n_trials=N_TRIALS)
                # Optuna 최적화 결과 저장
                if best_params:
                    save_best_hyperparameters(best_params)
        else:
            # Optuna 사용 안 함 (USE_OPTUNA=False)
            # JSON 파일에서 로드 시도
            best_params = load_best_hyperparameters()
            
            # JSON 파일이 없으면 기본값 사용
            if best_params is None:
                print("\n📋 JSON 파일 없음 → Config의 기본 하이퍼파라미터 사용")
                best_params = get_default_hyperparameters()
                print("기본 하이퍼파라미터:")
                for key, value in best_params.items():
                    print(f"   - {key}: {value}")
        
        # 최종 학습 실행
        model, X_va_sc, y_va_sc, X_te_sc, y_te_sc, scaler_y, feat_names, fi_df = train_and_eval(
            X, y, labels, feat_names,
            compute_fi=True,
            save_fi=True,
            optuna_params=best_params
        )

        print(f"\n=== [결과 요약: 연령대 '{age_group}'] ===")
        print(f"Feature 개수: {len(feat_names)}")
        if fi_df is not None:
            print("\n[Top 10 Feature Importance]")
            print(fi_df.head(10).to_string(index=False))
        else:
            print("Feature Importance 계산이 수행되지 않았습니다.")
        
        exit(0)
    
    # ===== PostgreSQL 모드 (기본) =====
    print("=" * 60)
    print("💾 PostgreSQL 모드: 데이터베이스에서 데이터 로드")
    print("=" * 60)
    
    # PostgreSQL에서 데이터 로드
    df = load_data_from_postgres()
    
    print("\n" + "✅ " * 30)
    print("데이터 로드 완료!")
    print("✅ " * 30 + "\n")
    
    # 데이터 확인
    print(f"\n📊 DataFrame 정보:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Columns: {list(df.columns)}")
    
    # 날씨 데이터 포함 여부 확인
    weather_cols_in_data = [c for c in ['min_temp', 'max_temp', 'avg_humidity'] if c in df.columns]
    if weather_cols_in_data:
        print(f"\n🌡️  날씨 데이터 포함 확인:")
        print(f"   ✅ PostgreSQL weather_data 테이블에서 성공적으로 가져옴")
        print(f"   - 포함된 날씨 컬럼: {weather_cols_in_data}")
        for col in weather_cols_in_data:
            data = df[col].dropna()
            if len(data) > 0:
                print(f"      • {col}: 평균 {data.mean():.2f}, 범위 [{data.min():.2f}, {data.max():.2f}]")
    else:
        print(f"\n⚠️  날씨 데이터 미포함 (weather_data 테이블 확인 필요)")
    print(f"\n처음 5개 행:")
    print(df.head())
    print(f"\n데이터 타입:")
    print(df.dtypes)
    
    print(f"\n🔧 USE_EXOG = '{USE_EXOG}'  (auto-detects vaccine/resp columns)")
    
    # DataFrame을 직접 전달하여 전처리
    print("\n📈 데이터 전처리 및 특징 추출 중...")
    X, y, labels, feat_names = load_and_prepare(
        df=df, 
        use_exog=USE_EXOG,
        age_group=args.age_group,
        subtype=args.subtype
    )
    print(f"✅ 전처리 완료!")
    print(f"   - Data points: {len(y)}")
    print(f"   - Features used ({len(feat_names)}): {feat_names}")
    
    best_params = None
    
    # USE_OPTUNA 플래그에 따라 처리
    if USE_OPTUNA:
        if not OPTUNA_AVAILABLE:
            print("\n⚠️ Optuna가 설치되지 않았습니다 (USE_OPTUNA=True)")
            print("   설치 명령: pip install optuna")
            # Optuna가 없으면 기본값 사용
            best_params = get_default_hyperparameters()
        else:
            # Optuna로 최적화 실행
            best_params = optimize_hyperparameters(X, y, labels, feat_names, n_trials=N_TRIALS)
            # Optuna 최적화 결과 저장
            if best_params:
                save_best_hyperparameters(best_params)
    else:
        # Optuna 사용 안 함 (USE_OPTUNA=False)
        # JSON 파일에서 로드 시도
        best_params = load_best_hyperparameters()
        
        # JSON 파일이 없으면 기본값 사용
        if best_params is None:
            print("\n📋 JSON 파일 없음 → Config의 기본 하이퍼파라미터 사용")
            best_params = get_default_hyperparameters()
            print("기본 하이퍼파라미터:")
            for key, value in best_params.items():
                print(f"   - {key}: {value}")
    
    # 최종 학습 실행
    model, X_va_sc, y_va_sc, X_te_sc, y_te_sc, scaler_y, feat_names, fi_df = train_and_eval(
        X, y, labels, feat_names,
        compute_fi=True,
        save_fi=True,
        optuna_params=best_params
    )

    print("\n=== [결과 요약] ===")
    print(f"Feature 개수: {len(feat_names)}")
    if fi_df is not None:
        print("\n[Top 10 Feature Importance]")
        print(fi_df.head(10).to_string(index=False))
    else:
        print("Feature Importance 계산이 수행되지 않았습니다.")