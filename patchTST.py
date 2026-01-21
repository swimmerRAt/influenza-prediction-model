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
    PostgreSQL에서 데이터를 로드하는 함수
    Returns:
        pd.DataFrame: 로드된 데이터
    """
    print("\n📊 데이터 로드: PostgreSQL에서 데이터프레임으로 불러옵니다...")
    try:
        df = load_from_postgres(table_name="influenza_data")
        print(f"✅ PostgreSQL 로드 완료: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ PostgreSQL 로드 실패: {e}")
        raise

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
    USE_OPTUNA = True       # Optuna 최적화 실행
    N_TRIALS = 50          # Optuna 최적화 시도 횟수
    OPTUNA_TIMEOUT = None   # 최적화 시간 제한 (초), None이면 무제한
    
    # Optuna 최적화 범위 (USE_OPTUNA=True일 때 사용)
    OPTUNA_SEARCH_SPACE = {
        'd_model': [64, 128, 256],  # n_heads의 배수로 설정
        'n_heads': [2, 4, 8],       # Attention head 개수
        'enc_layers': (2, 8),       # Encoder 레이어 개수 (범위 확장)
        'ff_dim': [64, 96, 128, 192, 256, 384, 512],  # Feed-forward 차원 (더 많은 값 추가)
        'dropout': (0.05, 0.5),                       # Dropout 비율 (범위 확장)
        'lr': (1e-6, 1e-2),                           # Learning rate (범위 확장, log scale)
        'weight_decay': (1e-6, 1e-2),                 # Weight decay (범위 확장, log scale)
        'batch_size': [16, 32, 48, 64, 96, 128],      # Batch size (더 세밀한 값 추가)
        'seq_len': [8, 10, 12, 14, 16, 18, 20],       # Input sequence length (세밀화)
        'pred_len': [1, 2, 3, 4, 5],                  # Prediction horizon (세밀화)
        'patch_len': [2, 3, 4, 5, 6],                 # Patch length (범위 확장)
    }
    
    # ===== 모델 하이퍼파라미터 (기본값) =====
    # Optuna를 사용하지 않을 때 또는 최적화 후 고정값으로 사용
    EPOCHS = 100
    BATCH_SIZE = 64
    SEQ_LEN = 12            # 입력 시퀀스 길이 (과거 몇 주)
    PRED_LEN = 3            # 예측 길이 (미래 몇 주)
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
    
    # ===== 데이터 설정 =====
    TRAIN_RATIO = 0.7       # Train 데이터 비율
    VAL_RATIO = 0.15        # Validation 데이터 비율 (Test = 1 - TRAIN - VAL)
    SCALER_TYPE = "robust"  # Scaler 타입: "standard", "robust", "minmax"
    
    # 외생 특징 사용 모드
    # "auto": 자동 감지, "none": 사용 안함, "vax": 백신률만, 
    # "resp": 호흡기지수만, "both": 둘 다, "all": 모든 특징
    USE_EXOG = "all"
    INCLUDE_SEASONAL_FEATS = True  # week_sin, week_cos 포함 여부
    
    # ===== 출력 설정 =====
    OUT_CSV = str(BASE_DIR / "ili_predictions.csv")
    PLOT_LAST_WINDOW = str(BASE_DIR / "plot_last_window.png")
    PLOT_TEST_RECON = str(BASE_DIR / "plot_test_reconstruction.png")
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

def weekly_to_daily_interp(
    df: pd.DataFrame,
    season_col: str = "season_norm",
    week_col: str = "week",
    target_col: str = "ili",
) -> pd.DataFrame:
    """
    주 단위 데이터를 일 단위로 확장(선형보간). season/week 없으면 label에서 추출하거나,
    최후에는 연속 주차를 생성해 보간합니다.
    반환: date 컬럼 포함한 일 단위 DF
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

    # 수치형은 선형보간
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
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

def get_scaler(name=None):
    s = (name or SCALER_TYPE).lower()
    if s == "robust":  return RobustScaler()
    if s == "minmax":  return MinMaxScaler()
    return StandardScaler()

def _norm_season_text(s: str) -> str:
    ss = str(s).replace("절기", "")
    import re
    m = re.search(r"(\d{4})\s*-\s*(\d{4})", ss)
    return f"{m.group(1)}-{m.group(2)}" if m else ss.strip()

# =========================
# data loader (multivariate-ready)
# =========================
def load_and_prepare(df: pd.DataFrame, use_exog: str = "auto") -> Tuple[np.ndarray, np.ndarray, list, list]:
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
    """
    if df is None:
        raise ValueError("df는 반드시 제공되어야 합니다. 먼저 데이터를 로드하세요.")
    
    df = df.copy()
    
    print(f"\n📊 원본 데이터 구조:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Columns: {list(df.columns)}")
    
    # ===== PostgreSQL 데이터 형식 감지 및 처리 =====
    is_postgres_format = all(col in df.columns for col in ['year', 'week', 'age_group'])
    
    if is_postgres_format:
        print(f"\n🔍 PostgreSQL 데이터 형식 감지됨 - 연령대별 데이터 처리 중...")
        
        # ===== 팬데믹 기간 데이터 제외 (2020년 14주 ~ 2022년 22주) =====
        print(f"\n⚠️ 팬데믹 기간 데이터 제외 중 (2020년 14주 ~ 2022년 22주)...")
        before_count = len(df)
        
        # 팬데믹 기간 마스크 생성
        pandemic_mask = (
            ((df['year'] == 2020) & (df['week'] >= 14)) |
            ((df['year'] == 2021)) |
            ((df['year'] == 2022) & (df['week'] <= 22))
        )
        
        # 팬데믹 기간 제외
        df = df[~pandemic_mask].copy()
        after_count = len(df)
        removed_count = before_count - after_count
        
        print(f"   - 제거 전: {before_count:,}행")
        print(f"   - 제거 후: {after_count:,}행")
        print(f"   - 제거됨: {removed_count:,}행 ({removed_count/before_count*100:.1f}%)")
        
        if after_count == 0:
            raise ValueError("팬데믹 기간 제외 후 데이터가 없습니다. 데이터 범위를 확인하세요.")
        
        # 연령대별 데이터 확인
        age_groups = df['age_group'].unique()
        print(f"\n   - 고유 연령대: {len(age_groups)}개")
        print(f"   - 연령대 목록: {sorted(age_groups)[:5]}...")
        
        # 여러 연령대 중 데이터가 가장 풍부한 연령대 선택
        # 우선순위: 19-49세 (가장 일반적) > 65세이상 > 65세 이상 > 전체 평균
        candidate_age_groups = ['19-49세', '65세이상', '65세 이상', '0-6세']
        target_age_group = None
        
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
            
            # ⭐ 의사환자 분율이 NaN인 행 제거 (필수!)
            before_count = len(df_age)
            df_age = df_age[df_age['ili'].notna()].copy()
            after_count = len(df_age)
            if before_count > after_count:
                print(f"   - 의사환자 분율 NaN 행 제거: {before_count - after_count}개 제거됨")
            
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
    
    # ===== 기존 처리 로직 =====
    # 주 단위 -> 일 단위 보간 (선택사항)
    # df = weekly_to_daily_interp(df, season_col="season_norm", week_col="week", target_col="ili")
    
    # 정렬 확인
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
    elif {"season_norm", "week"}.issubset(df.columns):
        df["season_norm"] = df["season_norm"].astype(str).map(_norm_season_text)
        df["week"] = pd.to_numeric(df["week"], errors="coerce")
        df = df.sort_values(["season_norm", "week"]).reset_index(drop=True)
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
        df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52.0)
    else:
        df["week_sin"] = 0.0
        df["week_cos"] = 0.0

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
    # week는 week_sin/week_cos로 대체, 나머지는 그대로
    chosen = []
    for v in column_mapping.values():
        if v == "week":
            chosen += ["week_sin", "week_cos"]
        else:
            chosen.append(v)
    # 중복 제거 및 순서 보존
    chosen = [x for i, x in enumerate(chosen) if x not in chosen[:i]]

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
    if INCLUDE_SEASONAL_FEATS and {"week_sin", "week_cos"}.issubset(df.columns):
        feat_names += ["week_sin", "week_cos"]

    # 선택된 입력 피처 로그
    print(f"\n[Data] Exogenous detected -> vaccine_rate: {has_vax} | respiratory/hospitalization: {has_resp} | climate_feats: {climate_feats}")
    print(f"[Data] Selected feature columns (order) -> {feat_names}")

    X = df[feat_names].to_numpy(dtype=float)
    y = df["ili"].to_numpy(dtype=float)
    
    print(f"\n✅ 최종 데이터 준비 완료:")
    print(f"   - X shape: {X.shape}")
    print(f"   - y shape: {y.shape}")
    print(f"   - Features: {len(feat_names)}")
    
    return X, y, labels, feat_names

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

        # ④ 예측 헤드
        mlp, in_dim = [], d_model
        for h in head_hidden[:2]:
            mlp += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        mlp.append(nn.Linear(in_dim, pred_len))
        self.head = nn.Sequential(*mlp)

    def forward(self, x):
        # x: (B, P, L, F)
        z = self.embed(x)      # (B,P,D)
        z = self.mixer(z)      # (B,P,D)
        z = self.posenc(z)
        z = self.encoder(z)
        z = self.pool(z)       # (B,D)
        return self.head(z)    # (B,H)

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

    # ==== Scaling ====
    # Target scaler
    scaler_y = get_scaler()
    y_tr_sc = scaler_y.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_va_sc = scaler_y.transform(y_va.reshape(-1,1)).ravel()
    y_te_sc = scaler_y.transform(y_te.reshape(-1,1)).ravel()

    # Feature scaler (입력 특징 전체)
    scaler_x = get_scaler()
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
    def peak_weighted_loss(pred, target, peak_quantile=0.9, alpha=3.0):
        """
        Peak-aware weighted MAE/Huber-style loss.
        pred, target: (B, H)
        """
        with torch.no_grad():
            # 기준: 배치 내 target 상위 quantile
            thresh = torch.quantile(target, peak_quantile)
            weights = torch.ones_like(target)
            weights[target >= thresh] = alpha
        return torch.mean(weights * torch.abs(pred - target))

    crit = peak_weighted_loss
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
# run
# =========================
if __name__ == "__main__":
    print("\n" + "🚀 " * 30)
    print("데이터 로드 및 모델 학습 시작!")
    print("🚀 " * 30 + "\n")
    
    print("=" * 60)
    print("💾 PostgreSQL 모드: 데이터베이스에서 데이터 로드")
    print("=" * 60)
    
    # PostgreSQL에서 데이터 로드
    df = load_data_from_postgres()
    
    print("\n" + "✅ " * 30)
    print("데이터 로드 완료!")
    print("✅ " * 30 + "\n")
    
    # 데이터 확인
    print(f"📊 DataFrame 정보:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Columns: {list(df.columns)}")
    print(f"\n처음 5개 행:")
    print(df.head())
    print(f"\n데이터 타입:")
    print(df.dtypes)
    
    print(f"\n🔧 USE_EXOG = '{USE_EXOG}'  (auto-detects vaccine/resp columns)")
    
    # DataFrame을 직접 전달하여 전처리
    print("\n📈 데이터 전처리 및 특징 추출 중...")
    X, y, labels, feat_names = load_and_prepare(df=df, use_exog=USE_EXOG)
    print(f"✅ 전처리 완료!")
    print(f"   - Data points: {len(y)}")
    print(f"   - Features used ({len(feat_names)}): {feat_names}")
    
    # 모델 학습 및 평가 (baseline 학습은 삭제됨; Optuna 이후 최종 학습만 실행)

    # =========================
# Feature Importance utils
# =========================
def _eval_mae_on_split(model, X_split_sc, y_split_sc, scaler_y, feat_names,
                       batch_size=BATCH_SIZE):
    """
    Feature Importance용 MAE 계산
    → 반드시 현재 model.head 출력 차원(pred_len)을 기준으로 계산
    """
    model.eval()

    # 🔑 실제 모델의 pred_len을 사용
    pred_len = model.head[-1].out_features
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

    mae_sum, n = 0.0, 0
    with torch.no_grad():
        for Xb, yb, _ in dl:
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)

            pred = model(Xb)  # (B, H_model)

            # 🔒 pred / yb shape mismatch 방지
            H = pred.shape[1]
            yb = yb[:, :H]

            mae_sum += batch_mae_in_original_units(pred, yb, scaler_y) * yb.size(0)
            n += yb.size(0)

    return float(mae_sum / max(1, n))


def compute_feature_importance(model, 
                               X_va_sc, y_va_sc, 
                               X_te_sc=None, y_te_sc=None,
                               scaler_y=None, feat_names=None, 
                               random_state=42):
    """
    퍼뮤테이션(열 섞기) 중요도와 평균 대체(그 특징을 평균으로 고정) 중요도를 계산.
    반환: 중요도 DataFrame (ΔMAE가 클수록 중요)
    
    Note: 'ili'는 타겟 변수이므로 Feature Importance 계산에서 제외됩니다.
    """
    assert scaler_y is not None and feat_names is not None
    rng = np.random.RandomState(random_state)

    # --- 'ili' 제외: 타겟 변수는 feature importance 계산에서 제외 ---
    feat_indices = [i for i, name in enumerate(feat_names) if name != 'ili']
    filtered_feat_names = [feat_names[i] for i in feat_indices]
    
    if len(filtered_feat_names) < len(feat_names):
        print(f"[FI] 'ili' 특징 제외됨 (타겟 변수)")
        print(f"[FI] Feature Importance 계산 대상: {len(filtered_feat_names)}개 특징")

    # --- 기준선(baseline MAE) ---
    baseline_val = _eval_mae_on_split(model, X_va_sc, y_va_sc, scaler_y, feat_names)
    print(f"[FI] Baseline Val MAE: {baseline_val:.6f}")

    baseline_tst = None
    if X_te_sc is not None and y_te_sc is not None:
        baseline_tst = _eval_mae_on_split(model, X_te_sc, y_te_sc, scaler_y, feat_names)
        print(f"[FI] Baseline Test MAE: {baseline_tst:.6f}")

    perm_deltas_val, mean_deltas_val = [], []
    perm_deltas_tst, mean_deltas_tst = [], []

    # 'ili'를 제외한 특징들에 대해서만 계산
    for j in feat_indices:
        name = feat_names[j]
        
        # ① 퍼뮤테이션(열 섞기)
        Xp = X_va_sc.copy()
        col = Xp[:, j].copy()
        rng.shuffle(col)
        Xp[:, j] = col
        mae_perm_val = _eval_mae_on_split(model, Xp, y_va_sc, scaler_y, feat_names)
        perm_deltas_val.append(mae_perm_val - baseline_val)

        # ② 평균 대체(특징 제거 효과)
        Xz = X_va_sc.copy()
        Xz[:, j] = X_va_sc[:, j].mean()
        mae_mean_val = _eval_mae_on_split(model, Xz, y_va_sc, scaler_y, feat_names)
        mean_deltas_val.append(mae_mean_val - baseline_val)

        if X_te_sc is not None and y_te_sc is not None:
            Xp_te = X_te_sc.copy()
            col_te = Xp_te[:, j].copy()
            rng.shuffle(col_te)
            Xp_te[:, j] = col_te
            mae_perm_tst = _eval_mae_on_split(model, Xp_te, y_te_sc, scaler_y, feat_names)
            perm_deltas_tst.append(mae_perm_tst - baseline_tst)

            Xz_te = X_te_sc.copy()
            Xz_te[:, j] = X_te_sc[:, j].mean()
            mae_mean_tst = _eval_mae_on_split(model, Xz_te, y_te_sc, scaler_y, feat_names)
            mean_deltas_tst.append(mae_mean_tst - baseline_tst)

    # DataFrame 생성
    # 내부명 → 한글명 매핑 (column_mapping의 value→key 역전)
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
    # 역매핑: 내부명 → 한글명
    inv_colmap = {v: k for k, v in column_mapping.items()}

    # feature명 + 한글명 표시 ('ili' 제외된 특징들만)
    feature_disp = [f"{f} ({inv_colmap[f]})" if f in inv_colmap else f for f in filtered_feat_names]

    df_fi = pd.DataFrame({
        "feature": feature_disp,
        "perm_delta_val": perm_deltas_val,
        "mean_delta_val": mean_deltas_val,
    })
    if X_te_sc is not None and y_te_sc is not None:
        df_fi["perm_delta_tst"] = perm_deltas_tst
        df_fi["mean_delta_tst"] = mean_deltas_tst

    # 평균 델타 기준 내림차순 정렬
    df_fi = df_fi.sort_values("mean_delta_val", ascending=False).reset_index(drop=True)
    return df_fi

def plot_feature_importance(fi_df, out_csv=None, out_png=None):
    """
    Feature Importance를 막대그래프로 시각화
    """
    if fi_df is None or len(fi_df) == 0:
        print("No feature importance data to plot.")
        return

    import matplotlib.pyplot as plt

    # CSV 저장
    if out_csv:
        fi_df.to_csv(out_csv, index=False)
        print(f"Feature Importance saved to {out_csv}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ① Permutation Δ (Val)
    axes[0].barh(fi_df["feature"], fi_df["perm_delta_val"], color="steelblue")
    axes[0].set_xlabel("ΔMAE (Permutation, Val)")
    axes[0].set_title("Permutation Feature Importance (Val)")
    axes[0].invert_yaxis()

    # ② Mean Replacement Δ (Val)
    axes[1].barh(fi_df["feature"], fi_df["mean_delta_val"], color="coral")
    axes[1].set_xlabel("ΔMAE (Mean Replacement, Val)")
    axes[1].set_title("Mean Replacement Feature Importance (Val)")
    axes[1].invert_yaxis()

    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Feature Importance plot saved to {out_png}")
    plt.show()


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
    print("=" * 70)
    
    def objective(trial: Trial) -> float:
        """Optuna objective function - validation MAE를 최소화"""
        
        # Config에서 탐색 공간 가져오기
        search_space = Config.OPTUNA_SEARCH_SPACE
        
        # 하이퍼파라미터 샘플링
        params = {
            'd_model': trial.suggest_categorical('d_model', search_space['d_model']),
            'n_heads': trial.suggest_categorical('n_heads', search_space['n_heads']),
            'enc_layers': trial.suggest_int('enc_layers', *search_space['enc_layers']),
            'ff_dim': trial.suggest_categorical('ff_dim', search_space['ff_dim']),
            'dropout': trial.suggest_float('dropout', *search_space['dropout']),
            'lr': trial.suggest_float('lr', *search_space['lr'], log=True),
            'weight_decay': trial.suggest_float('weight_decay', *search_space['weight_decay'], log=True),
            'batch_size': trial.suggest_categorical('batch_size', search_space['batch_size']),
            'seq_len': trial.suggest_categorical('seq_len', search_space['seq_len']),
            'pred_len': trial.suggest_categorical('pred_len', search_space['pred_len']),
            'patch_len': trial.suggest_categorical('patch_len', search_space['patch_len']),
        }
        
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
        
        # Scaling
        scaler_y = get_scaler()
        y_tr_sc = scaler_y.fit_transform(y_tr.reshape(-1,1)).ravel()
        y_va_sc = scaler_y.transform(y_va.reshape(-1,1)).ravel()
        
        scaler_x = get_scaler()
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
            with torch.no_grad():
                for Xb, yb, _ in dl_va:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    pred = model(Xb)
                    loss = crit(pred, yb)
                    bs = yb.size(0)
                    va_loss_sum += loss.item() * bs
                    va_mae_sum += batch_mae_in_original_units(pred, yb, scaler_y) * bs
                    n += bs
            
            va_loss = va_loss_sum / max(1, n)
            va_mae = va_mae_sum / max(1, n)
            
            # Early stopping
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= early_stop_patience:
                    break
            
            # Optuna pruning (중간 결과가 나쁘면 조기 종료)
            trial.report(va_mae, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Validation MAE 반환
        return va_mae
    
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
    print(f"  - Value (Val MAE): {study.best_trial.value:.4f}")
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

    N = len(y)
    split_tr = int(0.7*N); split_va = int(0.85*N)
    X_tr, y_tr = X[:split_tr], y[:split_tr]
    X_va, y_va = X[split_tr:split_va], y[split_tr:split_va]
    X_te, y_te = X[split_va:], y[split_va:]

    def get_scaler():
        st = SCALER_TYPE.lower()
        if st=="robust": return RobustScaler()
        if st=="minmax": return MinMaxScaler()
        return StandardScaler()

    scaler_y = get_scaler()
    y_tr_sc = scaler_y.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_va_sc = scaler_y.transform(y_va.reshape(-1,1)).ravel()
    y_te_sc = scaler_y.transform(y_te.reshape(-1,1)).ravel()

    scaler_x = get_scaler()
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

    # Test - 모든 성능 지표 계산
    model.eval(); te_mae_sum=0; te_mse_sum=0; te_rmse_sum=0; k=0
    with torch.no_grad():
        for Xb,yb,_ in dl_te:
            Xb=Xb.to(DEVICE); yb=yb.to(DEVICE)
            pred=model(Xb)
            te_mae_sum += batch_mae_in_original_units(pred,yb,scaler_y)*yb.size(0)
            te_mse_sum += batch_mse_in_original_units(pred,yb,scaler_y)*yb.size(0)
            te_rmse_sum += batch_rmse_in_original_units(pred,yb,scaler_y)*yb.size(0)
            k+=yb.size(0)
    te_mae_avg = te_mae_sum/max(1,k)
    te_mse_avg = te_mse_sum/max(1,k)
    te_rmse_avg = te_rmse_sum/max(1,k)
    
    print("\n" + "="*60)
    print("🎯 최종 테스트 성능 평가")
    print("="*60)
    print(f"MAE  (Mean Absolute Error):      {te_mae_avg:.6f}")
    print(f"MSE  (Mean Squared Error):       {te_mse_avg:.6f}")
    print(f"RMSE (Root Mean Squared Error):  {te_rmse_avg:.6f}")
    print("="*60)

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
    best_params = None
    
    # Optuna 최적화 실행
    if USE_OPTUNA:
        if not OPTUNA_AVAILABLE:
            print("\n⚠️ Optuna가 설치되지 않았습니다.")
            print("   설치 명령: pip install optuna")
            print("   기본 하이퍼파라미터로 학습을 진행합니다.\n")
        else:
            best_params = optimize_hyperparameters(X, y, labels, feat_names, n_trials=N_TRIALS)
    
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