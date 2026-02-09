import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from functools import partial

# PostgreSQL 데이터 로딩
from database.db_utils import TimeSeriesDB, load_from_postgres


def get_pandemic_mask(df: pd.DataFrame) -> pd.Series:
    """팬데믹 기간(2020년 14주 ~ 2022년 22주) 마스크 반환."""
    if 'year' not in df.columns or 'week' not in df.columns:
        return pd.Series(False, index=df.index)
    return (
        ((df['year'] == 2020) & (df['week'] >= 14)) |
        (df['year'] == 2021) |
        ((df['year'] == 2022) & (df['week'] <= 22))
    )


def interpolate_pandemic_period(df: pd.DataFrame, target_cols: list = None) -> pd.DataFrame:
    """
    팬데믹 기간(2020년 14주 ~ 2022년 22주) 데이터를 
    2017~2019년 주차별 평균 패턴으로 보간
    
    Parameters:
        df: 원본 DataFrame (year, week 컬럼 필수)
        target_cols: 보간할 컬럼 목록 (None이면 모든 수치형 컬럼)
    
    Returns:
        보간된 DataFrame
    """
    df = df.copy()
    
    # year, week 컬럼 확인
    if 'year' not in df.columns or 'week' not in df.columns:
        print("⚠️  year, week 컬럼이 없어 팬데믹 보간 건너뜀")
        return df
    
    pandemic_mask = get_pandemic_mask(df)
    
    pandemic_count = pandemic_mask.sum()
    if pandemic_count == 0:
        print("ℹ️  팬데믹 기간 데이터 없음 - 보간 건너뜀")
        return df
    
    print(f"\n🦠 팬데믹 기간 보간 처리")
    print(f"   팬데믹 기간: 2020년 14주 ~ 2022년 22주 ({pandemic_count}건)")
    
    # 보간 대상 컬럼 결정 (year, week 제외한 수치형)
    if target_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = [c for c in numeric_cols if c not in ['year', 'week']]
    
    if len(target_cols) == 0:
        print("⚠️  보간할 수치형 컬럼이 없음")
        return df
    
    print(f"   보간 대상 컬럼: {target_cols}")
    
    # 1단계: 팬데믹 기간 데이터를 NaN으로 설정
    for col in target_cols:
        df.loc[pandemic_mask, col] = np.nan
    
    # 2단계: 2017~2019년 정상 계절성 패턴 계산
    pre_pandemic_mask = (df['year'] >= 2017) & (df['year'] <= 2019)
    
    weekly_patterns = {}
    for col in target_cols:
        df_pre = df[pre_pandemic_mask & df[col].notna()]
        if len(df_pre) > 0:
            weekly_patterns[col] = df_pre.groupby('week')[col].mean()
        else:
            weekly_patterns[col] = None
    
    # 3단계: 팬데믹 구간을 계절성 패턴으로 보간
    interpolated_counts = {col: 0 for col in target_cols}
    
    for idx in df[pandemic_mask].index:
        week_num = df.loc[idx, 'week']
        
        for col in target_cols:
            pattern = weekly_patterns.get(col)
            if pattern is not None:
                if week_num in pattern.index:
                    df.loc[idx, col] = pattern[week_num]
                else:
                    # 해당 주차 패턴이 없으면 전체 평균 사용
                    df.loc[idx, col] = pattern.mean()
                interpolated_counts[col] += 1
    
    # 결과 출력
    for col, count in interpolated_counts.items():
        if count > 0:
            print(f"   ✅ {col}: {count}건 보간 완료")
    
    return df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    # 19-49세 연령 그룹만 필터링
    if 'age_group' in df.columns:
        print(f"\n🔍 연령 그룹 필터링: 19-49세만 선택")
        print(f"   필터링 전: {len(df)}건")
        df = df[df['age_group'] == '19-49세'].copy()
        print(f"   필터링 후: {len(df)}건")
    else:
        print("⚠️  age_group 컬럼이 없습니다. 전체 데이터 사용")

    # 필요한 컬럼만 선택 (age_group은 필터링용이므로 분석에서 제외)
    available_cols = df.columns.tolist()
    desired_cols = ["year", "week", "ili", "detection_rate", "delta_ili"]

    # 실제로 존재하는 컬럼만 선택
    cols = [c for c in desired_cols if c in available_cols]
    if len(cols) < len(desired_cols):
        missing = set(desired_cols) - set(cols)
        print(f"⚠️  누락된 컬럼: {missing}")

    print(f"\n📋 선택된 컬럼 (상관계수/모델 입력용): {cols}")

    # 선택된 컬럼만 남기기
    df = df[cols].copy()

    # 팬데믹 기간 보간 (2017-2019년 계절성 패턴 사용)
    # year, week 외의 모든 수치형 컬럼에 적용
    target_cols_for_pandemic = [c for c in cols if c not in ['year', 'week']]
    df = interpolate_pandemic_period(df, target_cols=target_cols_for_pandemic)

    # 나머지 결측값 처리 (팬데믹 보간 후 남은 NaN)
    print(f"\n🔧 전처리 결측값 처리 중...")
    valid_cols = []
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # 전체가 NaN인 컬럼 체크
        if df[col].isna().all():
            print(f"   ❌ {col}: 전체 NaN - 피처에서 제외")
            continue
        
        if df[col].isna().any():
            nan_count = df[col].isna().sum()
            nan_pct = nan_count / len(df) * 100
            print(f"   {col}: {nan_count}개 ({nan_pct:.1f}%) 결측값 처리")
            df[col] = df[col].interpolate(method="linear").ffill().bfill()
            if df[col].isna().any():
                fill_val = df[col].median() if not np.isnan(df[col].median()) else 0.0
                df[col] = df[col].fillna(fill_val)
        
        valid_cols.append(col)
    
    if len(valid_cols) < len(cols):
        removed = set(cols) - set(valid_cols)
        print(f"   ⚠️  제거된 피처: {removed}")
    
    cols = valid_cols
    df = df[cols].copy()  # 유효한 컬럼만 유지

    remaining_nans = df.isna().sum().sum()
    print(f"   ✅ 결측값 처리 완료 (남은 NaN: {remaining_nans}개, 유효 피처: {len(cols)}개)")


    # =========================
    # Δili (1주 변화량) 추가 (수영)
    df["delta_ili"] = df["ili"].diff()

    # 첫 행 NaN 처리
    df["delta_ili"] = df["delta_ili"].fillna(0.0)
    # =========================

    # 소수점 둘째 자리로 반올림
    df = df.round(2)

    return df, cols

print("\n" + "="*60)
print("📊 PostgreSQL에서 인플루엔자 데이터 로드")
print("="*60)

# PostgreSQL에서 인플루엔자 데이터 로드
try:
    df = load_from_postgres(table_name="influenza_data")
    print(f"✅ PostgreSQL influenza_data 로드 완료: {df.shape}")
    print(f"   컬럼: {list(df.columns)}")
except Exception as e:
    print(f"❌ PostgreSQL 로드 실패: {e}")
    print("   CSV 파일로 대체 시도...")
    df = pd.read_csv("merged_influenza_data.csv")
    print(f"✅ CSV 파일 로드 완료: {df.shape}")

df, cols = preprocess_data(df)

# 전처리 결과 저장 후 입력 데이터로 사용
final_data_path = "final_data.csv"
df.to_csv(final_data_path, index=False, encoding="utf-8-sig")
print(f"✅ 전처리 결과 저장: {final_data_path}")
df = pd.read_csv(final_data_path)

# 상관계수 분석용 데이터
df_numeric = df[cols].copy()

# 상관계수 행렬 (숫자형 데이터만)
# print(f"\n📊 상관계수 분석 중 (age_group 제외)...")
# corr = df_numeric.corr(method="pearson")
# print(corr)

# plt.figure(figsize=(8,6))
# sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
# plt.title("Correlation Heatmap (Pearson) - Age 19-49")
# plt.tight_layout()
# plt.savefig("correlation_heatmap_19-49.png", dpi=150)
# print(f"✅ 상관계수 히트맵 저장: correlation_heatmap_19-49.png")
# plt.show()

# print("="*60 + "\n")

# ili_patchtst_train_and_plot_v4_cnnmix.py
# -*- coding: utf-8 -*-
"""
Influenza ILI forecasting with PatchTST (multivariate-ready) + Multi-Scale CNN Patching
- Data Source: PostgreSQL influenza_data table (age_group: 19-49세 필터링)
- Auto-detect columns: 'ili' (target), 'vaccine_rate', 'case_count'
- Climate features: wx_week_avg_temp, wx_week_avg_rain, wx_week_avg_humidity
- Train-only scaling: separate scaler_y (target) and scaler_x (features)
- **Multi-Scale CNN Patch Embedding + TokenConvMixer → PatchTST-style encoder + Attention Pooling**
- Loss: Huber; Optim: AdamW; Cosine LR + warmup; EarlyStopping
- Saves: predictions CSV, last-window plot, test reconstruction, MAE curves, feature importance
"""

import math
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# =========================
# Paths & device
# =========================
BASE_DIR = Path.cwd()
# PostgreSQL에서 데이터를 로드하므로 CSV 경로는 불필요
# (상단에서 이미 PostgreSQL에서 df 로드 완료)

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = pick_device()
SEED   = 42

# =========================
# Hyperparameters
# =========================
EPOCHS      = 200
BATCH_SIZE  = 32        # 소규모 시계열에서도 안정적으로 학습되도록 약간 낮춤
SEQ_LEN     = 12
PRED_LEN    = 3
PATCH_LEN   = 4          # ← CNN이 최소 3~5 커널 적용 가능하도록 확대
STRIDE      = 1

D_MODEL     = 64       # 8의 배수 (강화된 멀티스케일 분기 8개 합산) - 표현력 증가
N_HEADS     = 4        # 더 많은 attention head로 다양한 패턴 포착
ENC_LAYERS  = 3        # 인코더 깊이 증가
FF_DIM      = 64       # 피드포워드 차원 증가
DROPOUT     = 0.2        # 약간 강화
HEAD_HIDDEN = [64, 32]  # MLP 헤드 크기 증가

# Amplitude-aware loss 가중치 (수영)
AMP_WEIGHT   = 0.10
SLOPE_WEIGHT = 1.20

# tanh 활성화 스케일 (alpha: 입력 스케일, gain: 출력 스케일)
# 디폴트값 : alpha 1.5, gain 1.2
TANH_ALPHA = 1.5
TANH_GAIN = 6.0
TANH_LEARNABLE = True

LR              = 3e-4    # 더 강한 모델이므로 학습률 감소
WEIGHT_DECAY    = 5e-3
PATIENCE        = 50      # 조기 종료 기준 단축 (더 강한 모델은 빠르게 수렴)
WARMUP_EPOCHS   = 30      # Warmup 에포크 단축

SCALER_TYPE     = "standard"   # 노이즈/꼬리값 대응에 유리 (원하면 "standard"로 변경)

# 외생 특징 사용 모드: "auto"|"none"|"vax"|"resp"|"both"
USE_EXOG        = "all"

OUT_CSV          = str(BASE_DIR / "ili_predictions.csv")
PLOT_LAST_WINDOW = str(BASE_DIR / "plot_last_window.png")
PLOT_TEST_RECON  = str(BASE_DIR / "plot_test_reconstruction.png")
PLOT_MA_CURVES   = str(BASE_DIR / "plot_ma_curves.png")

# overlap 재구성 가중치 (t+1을 조금 더 신뢰)
RECON_W_START, RECON_W_END = 2.0, 0.5

# --- Feature switches ---
INCLUDE_SEASONAL_FEATS = True   # week_sin, week_cos를 입력 피처에 포함할지

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

def read_csv_kor(path: Path) -> pd.DataFrame:
    for enc in ["euc-kr", "cp949", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path, encoding="utf-8", errors="replace")

def make_splits(n: int, train_ratio=0.7, val_ratio=0.15):
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
# data loader (multivariate-ready) - PostgreSQL용으로 수정
# =========================
def load_and_prepare(df_input: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray, list, list]:   
    """
    PostgreSQL에서 로드한 데이터프레임을 모델 입력 형태로 변환
    
    Parameters:
        df_input: PostgreSQL에서 로드한 DataFrame (전역 변수 df 사용 가능)
    
    Returns:
        X: (N, F) features
        y: (N,) target (ili)
        labels: list[str] for plotting ticks
        used_feat_names: list[str] feature column names (len=F)
    """
    # 전역 변수 df 사용
    if df_input is None:
        if 'df' not in globals():
            raise ValueError("df가 정의되지 않았습니다. PostgreSQL에서 데이터를 먼저 로드하세요.")
        df = globals()['df'].copy()
    else:
        df = df_input.copy()
    
    print(f"\n📊 데이터 준비 중...")
    print(f"   입력 데이터: {df.shape}")
    print(f"   컬럼: {list(df.columns)}")
    
    # ⭐ 타깃/피처 분리: 입력 피처는 오직 ili 한 채널만 사용
    #    (X_target = ili(t-L:t-1), y = ili(t:t+H))
    target_col = "ili"
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' 컬럼이 df에 없습니다: {list(df.columns)}")

    # feat_names = [target_col] - 수영
    feat_names = ["ili", "delta_ili"]
    
    print(f"   입력 타깃 채널만 사용 (자기복사 방지): {feat_names}")
    
    # 선택된 컬럼만 사용 (이미 상단에서 결측값 처리 완료)
    # 추가 확인 및 보간 (혹시 모를 NaN 대비)
    valid_feat_names = []
    for c in feat_names:
        if c not in df.columns:
            print(f"   ⚠️  컬럼 '{c}'가 df에 없습니다. 건너뜀.")
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
        # 전체가 NaN인 컬럼 체크
        if df[c].isna().all():
            print(f"   ❌ {c}: 전체 NaN - 피처에서 제외")
            continue
        
        if df[c].isna().any():
            nan_count = df[c].isna().sum()
            print(f"   ⚠️  {c}: {nan_count}개 결측값 발견 - 보간 처리")
            df[c] = df[c].interpolate(method="linear", limit_direction="both")
            # 보간 후에도 남은 NaN은 0으로 채움 (median이 NaN일 수 있으므로)
            if df[c].isna().any():
                fill_val = df[c].median() if not np.isnan(df[c].median()) else 0.0
                df[c] = df[c].fillna(fill_val)
        
        valid_feat_names.append(c)
    
    if len(valid_feat_names) == 0:
        raise ValueError("사용 가능한 피처가 없습니다.")
    
    if len(valid_feat_names) < len(feat_names):
        removed = set(feat_names) - set(valid_feat_names)
        print(f"   ⚠️  제거된 피처: {removed}")
    
    feat_names = valid_feat_names
    
    # 라벨 생성 (인덱스 기반)
    labels = [f"idx_{i}" for i in range(len(df))]
    
    # X, y 구성
    X = df[feat_names].to_numpy(dtype=float)
    y = df["ili"].to_numpy(dtype=float)
    
    print(f"   ✅ 데이터 준비 완료: X={X.shape}, y={y.shape}")
    
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
class ScaledTanh(nn.Module):
    """tanh 출력에 스케일을 부여 (입력 스케일 alpha, 출력 스케일 gain)."""
    def __init__(self, alpha: float = 1.0, gain: float = 1.0, learnable: bool = False):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)), requires_grad=learnable)
        self.gain = nn.Parameter(torch.tensor(float(gain)), requires_grad=learnable)

    def forward(self, x):
        return torch.tanh(self.alpha * x) * self.gain

class MultiScaleCNNPatchEmbed(nn.Module):
    """
    강화된 멀티스케일 CNN 패치 임베딩: 다양한 커널 크기와 dilated convolution으로
    급격한 변화/이상치를 더 잘 포착
    (B, P, L, F) -> [각 패치] 멀티스케일 분기 → 활성화 → GAP → (B, P, D)
    - 분기 8개: k=[1,3,5,7] × dilation=[1,2]
    - 패치 내부의 급격/완만/이상 패턴 동시 포착
    """
    def __init__(self, in_features: int, patch_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % 8 == 0, "d_model은 8의 배수가 되어야 멀티스케일 분기 합산이 맞습니다."
        out_ch = d_model // 8
        
        # 분기 1-4: 다양한 커널 크기 (점진적 확대)
        # Tanh 활성화: 극값(튀는 값)에 더 민감하게 반응 [-1, 1] 범위로 제약
        self.b1 = nn.Sequential(
            nn.Conv1d(in_features, out_ch, kernel_size=1, padding=0, bias=False),
            ScaledTanh(alpha=TANH_ALPHA, gain=TANH_GAIN, learnable=TANH_LEARNABLE)
        )
        self.b3 = nn.Sequential(
            nn.Conv1d(in_features, out_ch, kernel_size=3, padding=1, bias=False),
            ScaledTanh(alpha=TANH_ALPHA, gain=TANH_GAIN, learnable=TANH_LEARNABLE)
        )
        self.b5 = nn.Sequential(
            nn.Conv1d(in_features, out_ch, kernel_size=5, padding=2, bias=False),
            ScaledTanh(alpha=TANH_ALPHA, gain=TANH_GAIN, learnable=TANH_LEARNABLE)
        )
        self.b7 = nn.Sequential(
            nn.Conv1d(in_features, out_ch, kernel_size=7, padding=3, bias=False),
            ScaledTanh(alpha=TANH_ALPHA, gain=TANH_GAIN, learnable=TANH_LEARNABLE)
        )
        
        # 분기 5-8: dilated convolution (넓은 수용장으로 이상치 포착)
        self.bd3_d1 = nn.Sequential(
            nn.Conv1d(in_features, out_ch, kernel_size=3, padding=1, dilation=1, bias=False),
            ScaledTanh(alpha=TANH_ALPHA, gain=TANH_GAIN, learnable=TANH_LEARNABLE)
        )
        self.bd3_d2 = nn.Sequential(
            nn.Conv1d(in_features, out_ch, kernel_size=3, padding=2, dilation=2, bias=False),
            ScaledTanh(alpha=TANH_ALPHA, gain=TANH_GAIN, learnable=TANH_LEARNABLE)
        )
        self.bd5_d2 = nn.Sequential(
            nn.Conv1d(in_features, out_ch, kernel_size=5, padding=4, dilation=2, bias=False),
            ScaledTanh(alpha=TANH_ALPHA, gain=TANH_GAIN, learnable=TANH_LEARNABLE)
        )
        self.bd3_d3 = nn.Sequential(
            nn.Conv1d(in_features, out_ch, kernel_size=3, padding=3, dilation=3, bias=False),
            ScaledTanh(alpha=TANH_ALPHA, gain=TANH_GAIN, learnable=TANH_LEARNABLE)
        )

        self.bn   = nn.BatchNorm1d(d_model)
        self.act  = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)   # (B*P, D, L) → (B*P, D, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, P, L, F)
        B, P, L, F = x.shape
        x = x.view(B*P, L, F).permute(0, 2, 1)        # (B*P, F, L)

        # 8개 분기 병렬 처리
        z = torch.cat([
            self.b1(x),      # 미세한 변화 포착
            self.b3(x),      # 작은 스케일 패턴
            self.b5(x),      # 중간 스케일 패턴
            self.b7(x),      # 큰 스케일 패턴
            self.bd3_d1(x),  # 넓은 수용장 (이상치 민감)
            self.bd3_d2(x),  # dilated 중간
            self.bd5_d2(x),  # dilated 큰 스케일
            self.bd3_d3(x),  # dilated 매우 넓은
        ], dim=1)  # (B*P, D, L)
        
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
        self.out_scale = nn.Parameter(torch.tensor(1.5)) # 출력 스케일 파라미터 (수영)

    def forward(self, x):
        # x: (B, P, L, F)
        z = self.embed(x)      # (B,P,D)
        z = self.mixer(z)      # (B,P,D)
        z = self.posenc(z)
        z = self.encoder(z)
        z = self.pool(z)       # (B,D)
        # return self.head(z)    # (B,H)
        return self.head(z) * self.out_scale    # (B,H) (수영)
# =========================
# helpers
# =========================
def warmup_lr(ep:int, base_lr:float, warmup_epochs:int):
    if ep <= warmup_epochs:
        return base_lr * (ep / max(1, warmup_epochs))
    return base_lr

def batch_mae_in_original_units(pred_b: torch.Tensor, y_b: torch.Tensor, scaler_y) -> float:
    p = pred_b.detach().cpu().numpy().reshape(-1, 1)
    t = y_b.detach().cpu().numpy().reshape(-1, 1)
    p_orig = scaler_y.inverse_transform(p).reshape(-1)
    t_orig = scaler_y.inverse_transform(t).reshape(-1)
    return float(np.mean(np.abs(p_orig - t_orig)))

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
# amplitude-aware MSE 함수 만들기 (수영)
# =========================
# 값과 변환율이 클 수록 예측을 틀리면 손실을 더 크게 주도록 하는 MSE 함수임
def amplitude_aware_mse_with_peak(pred, true,
                                  amp_weight,
                                  slope_weight,
                                  peak_weight=0.1):
    # 기존 amplitude-aware MSE
    amp_w = 1.0 + amp_weight * true.abs()

    if true.shape[1] > 1:
        slope = torch.relu(true[:, 1:] - true[:, :-1])
        slope = torch.cat([slope[:, :1], slope], dim=1)
        slope_w = 1.0 + slope_weight * slope
    else:
        slope_w = 1.0

    weight = amp_w * slope_w
    mse = (weight * (pred - true) ** 2).mean()

    # 🔥 peak underestimation penalty
    pred_max = pred.max(dim=1).values
    true_max = true.max(dim=1).values
    peak_penalty = torch.relu(true_max - pred_max).mean()

    return mse + peak_weight * peak_penalty

# =========================
# Feature Importance utils
# =========================
def _eval_mae_on_split(model, X_split_sc, y_split_sc, scaler_y, feat_names, 
                       seq_len=SEQ_LEN, pred_len=PRED_LEN, patch_len=PATCH_LEN, stride=STRIDE,
                       batch_size=BATCH_SIZE):
    """현재 모델로 한 분할(va/test) 세트에서 MAE(원 단위) 계산"""
    ds = PatchTSTDataset(X_split_sc, y_split_sc, seq_len, pred_len, patch_len, stride)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    mae_sum, n = 0.0, 0
    with torch.no_grad():
        for Xb, yb, _ in dl:
            Xb = Xb.to(DEVICE); yb = yb.to(DEVICE)
            pred = model(Xb)  # (B, H)
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
    """
    assert scaler_y is not None and feat_names is not None
    rng = np.random.RandomState(random_state)

    # --- 기준선(baseline MAE) ---
    baseline_val = _eval_mae_on_split(model, X_va_sc, y_va_sc, scaler_y, feat_names)
    print(f"[FI] Baseline Val MAE: {baseline_val:.6f}")

    baseline_tst = None
    if X_te_sc is not None and y_te_sc is not None:
        baseline_tst = _eval_mae_on_split(model, X_te_sc, y_te_sc, scaler_y, feat_names)
        print(f"[FI] Baseline Test MAE: {baseline_tst:.6f}")

    perm_deltas_val, mean_deltas_val = [], []
    perm_deltas_tst, mean_deltas_tst = [], []

    for j, name in enumerate(feat_names):
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

        print(f"[FI] {name:>20s} | ΔMAE(val) perm={perm_deltas_val[-1]:+.6f}  mean={mean_deltas_val[-1]:+.6f}")

    df = pd.DataFrame({
        "feature": feat_names,
        "delta_mae_val_perm": perm_deltas_val,
        "delta_mae_val_mean": mean_deltas_val,
    })
    if baseline_tst is not None:
        df["delta_mae_test_perm"] = perm_deltas_tst
        df["delta_mae_test_mean"] = mean_deltas_tst

    # ΔMAE가 클수록 중요 → 내림차순 정렬
    df = df.sort_values("delta_mae_val_perm", ascending=False).reset_index(drop=True)
    return df


def save_feature_importance(df: pd.DataFrame, out_csv="feature_importance.csv", out_png="feature_importance.png"):
    """중요도 테이블 저장 + 막대 그래프 저장"""
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[FI] Saved -> {out_csv}")

    top = min(20, len(df))
    plt.figure(figsize=(10, max(4, 0.4*top)))
    plt.barh(df["feature"][:top][::-1], df["delta_mae_val_perm"][:top][::-1])
    plt.title("Permutation Feature Importance (ΔMAE on Val)")
    plt.xlabel("ΔMAE (higher = more important)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[FI] Saved -> {out_png}")


# =========================
# train & evaluate (WITH Feature Importance)
# =========================
def train_and_eval(X: np.ndarray, y: np.ndarray, labels: list, feat_names: list,
                   compute_fi: bool = True, save_fi: bool = True):
    """
    X: (N,F), y: (N,), feat_names: ['ili', 'vaccine_rate', 'respiratory_index', ...]
    compute_fi: True면 검증/테스트 기반 피처 중요도 계산 및 저장
    save_fi: True면 feature_importance.csv/png 저장
    반환: (model, X_va_sc, y_va_sc, X_te_sc, y_te_sc, scaler_y, feat_names, fi_df)
    """
    set_seed(SEED)
    (s0,e0),(s1,e1),(s2,e2) = make_splits(len(y))
    X_tr, X_va, X_te = X[s0:e0], X[s1:e1], X[s2:e2]
    y_tr, y_va, y_te = y[s0:e0], y[s1:e1], y[s2:e2]
    lab_tr, lab_va, lab_te = labels[s0:e0], labels[s1:e1], labels[s2:e2]

    # ==== Scaling ====
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

    # crit = nn.HuberLoss(delta=1.0)
    # crit = amplitude_aware_mse # (수영)
    crit = partial(
        amplitude_aware_mse_with_peak,
        amp_weight=0.10,
        slope_weight=1.20,
        peak_weight=0.1
    )
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    hist = {"train_loss":[], "val_loss":[], "train_mae":[], "val_mae":[]}

    best_val = float("inf"); best_state=None; noimp=0
    printed_batch_info = False
    for ep in range(1, EPOCHS+1):
        model.train(); tr_loss_sum=0; tr_mae_sum=0; n=0
        for g in opt.param_groups:
            g['lr'] = warmup_lr(ep, LR, WARMUP_EPOCHS)

        for Xb,yb,_ in dl_tr:
            if not printed_batch_info:
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

        model.eval(); va_loss_sum=0; va_mae_sum=0; va_corr_sum=0; n=0
        with torch.no_grad():
            for Xb,yb,_ in dl_va:
                Xb=Xb.to(DEVICE); yb=yb.to(DEVICE)
                pred = model(Xb); loss = crit(pred,yb)
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

        wandb.log({
            "epoch": ep,
            "lr": opt.param_groups[0]["lr"],
            "loss/train": tr_loss,
            "loss/val": va_loss,
            "mae/train": tr_mae,
            "mae/val": va_mae,
            "corr/val": va_corr,
            "model/out_scale": model.out_scale.item(),
        })

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

    yhat  = scaler_y.inverse_transform(yhat_sc.reshape(-1,1)).reshape(-1,PRED_LEN)
    ytrue = scaler_y.inverse_transform(ytrue_sc.reshape(-1,1)).reshape(-1,PRED_LEN)

    mse  = float(np.mean((yhat-ytrue)**2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(yhat-ytrue)))

    wandb.log({
        "test/mae": mae,
        "test/rmse": rmse,
        "test/pred_max": float(yhat.max()),
        "test/true_max": float(ytrue.max()),
    })

    print("\n=== Final Test Metrics ===")
    print(f"MSE : {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")

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
    plt.tight_layout(); plt.savefig(PLOT_LAST_WINDOW, dpi=150)
    print(f"Saved plot -> {PLOT_LAST_WINDOW}")

    # =========================
    # Plot_2: test reconstruction (val-context included)
    # =========================
    context = y_va_sc[-SEQ_LEN:]
    y_ct_sc = np.concatenate([context, y_te_sc])              # [SEQ_LEN + test_len]
    X_ct_sc = np.concatenate([X_va_sc[-SEQ_LEN:], X_te_sc], axis=0)
    ds_ct = PatchTSTDataset(X_ct_sc, y_ct_sc, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
    dl_ct = DataLoader(ds_ct, batch_size=BATCH_SIZE, shuffle=False)

    model.eval(); preds_ct=[]; starts_ct=[]
    with torch.no_grad():
        for Xb, _, i0 in dl_ct:
            Xb = Xb.to(DEVICE)
            preds_ct.append(model(Xb).detach().cpu().numpy())
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
    tick_step = max(1, test_len // 12)
    tick_idx  = list(range(0, test_len, tick_step))
    if tick_idx[-1] != test_len-1:
        tick_idx.append(test_len-1)
    tick_text = [x_labels[i] for i in tick_idx]

    plt.figure(figsize=(12,5))
    plt.plot(range(test_len), truth_test, linewidth=2, label="Truth (test segment)")
    plt.plot(range(test_len), recon,      linewidth=2, label="Prediction (overlap-avg, weighted)")
    plt.title("Test Range: Truth vs Overlap-averaged Prediction (with context)")
    plt.xlabel("Season - Week"); plt.ylabel("ILI per 1,000 Population")
    plt.xticks(tick_idx, tick_text, rotation=45, ha="right")
    plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(PLOT_TEST_RECON, dpi=150)
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
    plt.tight_layout(); plt.savefig(PLOT_MA_CURVES, dpi=150)
    print(f"Saved plot -> {PLOT_MA_CURVES}")

    # =========================
    # Feature Importance
    # =========================
    fi_df = None
    if compute_fi:
        fi_df = compute_feature_importance(
            model,
            X_va_sc, y_va_sc,
            X_te_sc, y_te_sc,
            scaler_y=scaler_y,
            feat_names=feat_names,
            random_state=SEED
        )
        if save_fi:
            save_feature_importance(
                fi_df,
                out_csv=str(BASE_DIR / "feature_importance.csv"),
                out_png=str(BASE_DIR / "feature_importance.png")
            )

    # 반환: 외부 셀에서 재활용 가능하도록
    return model, X_va_sc, y_va_sc, X_te_sc, y_te_sc, scaler_y, feat_names, fi_df

# =========================
# 실행부 (결과 출력 + Feature Importance)
# =========================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("🚀 모델 학습 시작 (Feature Importance 포함)")
    print(f"Device: {DEVICE}")
    print(f"Data Source: PostgreSQL influenza_data (19-49세)")
    print(f"{'='*60}\n")
    
    # PostgreSQL에서 로드한 df 사용
    X, y, labels, feat_names = load_and_prepare()
    print(f"\n📊 데이터 정보:")
    print(f"   Data points: {len(y)}")
    print(f"   Features: {feat_names}")
    print(f"   Feature count: {len(feat_names)}")

    run_name = (
        f"PatchTST.v2"
        f"_amp{AMP_WEIGHT}"
        f"_slope{SLOPE_WEIGHT}"
        f"_tanh{TANH_GAIN}"
        f"_lr{LR}"
    )

    # wandb 실험 기록
    wandb.init(
        project="influenza-patchTST",
        name=run_name,
        config={
            # ===== data =====
            "seq_len": SEQ_LEN,
            "pred_len": PRED_LEN,
            "patch_len": PATCH_LEN,
            "stride": STRIDE,

            # ===== model =====
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "enc_layers": ENC_LAYERS,
            "ff_dim": FF_DIM,
            "dropout": DROPOUT,
            "head_hidden": HEAD_HIDDEN,

            # ===== training =====
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,

            # ===== loss =====
            "loss": "amplitude_aware_mse",
            "amp_weight": AMP_WEIGHT,
            "slope_weight": SLOPE_WEIGHT,

            # ===== activation =====
            "tanh_alpha": TANH_ALPHA,
            "tanh_gain": TANH_GAIN,
            "tanh_learnable": TANH_LEARNABLE,

            # ===== scaler =====
            "scaler_type": SCALER_TYPE,
            "recon_w_start": RECON_W_START,
            "recon_w_end": RECON_W_END,
            "loss_type": "amp+slope",
        }
    )
    
    model, X_va_sc, y_va_sc, X_te_sc, y_te_sc, scaler_y, feat_names, fi_df = train_and_eval(
        X, y, labels, feat_names,
        compute_fi=True,
        save_fi=True
    )

    print("\n" + "="*60)
    print("=== [최종 결과 요약] ===")
    print("="*60)
    print(f"✅ Feature 개수: {len(feat_names)}")
    if fi_df is not None:
        print("\n📊 [Top 10 Feature Importance]")
        print(fi_df.head(10).to_string(index=False))
        print(f"\n💾 저장된 파일:")
        print(f"   - feature_importance.csv")
        print(f"   - feature_importance.png")
    else:
        print("⚠️  Feature Importance 계산이 수행되지 않았습니다.")


    wandb.log({
        "plot/last_window": wandb.Image(PLOT_LAST_WINDOW),
        "plot/test_reconstruction": wandb.Image(PLOT_TEST_RECON),
        "plot/mae_curves": wandb.Image(PLOT_MA_CURVES),
    })
    wandb.finish()
    print("\n✅ 모든 작업 완료!")
    print("="*60)