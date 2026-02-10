import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def preprocess_data(
    df: pd.DataFrame,
    interpolate_pandemic: bool = True,
    drop_pandemic: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list]:
    """
    Args:
        df: 원본 DataFrame
        interpolate_pandemic: True면 팬데믹 구간을 2017~2019 계절성으로 보간
        drop_pandemic: True면 팬데믹 구간 행 제거 (test2용)
        verbose: 로그 출력 여부
    """
    def log(msg):
        if verbose:
            print(msg)

    # 19-49세 연령 그룹만 필터링
    if 'age_group' in df.columns:
        log(f"\n🔍 연령 그룹 필터링: 19-49세만 선택")
        log(f"   필터링 전: {len(df)}건")
        df = df[df['age_group'] == '19-49세'].copy()
        log(f"   필터링 후: {len(df)}건")
    else:
        log("⚠️  age_group 컬럼이 없습니다. 전체 데이터 사용")

    # 필요한 컬럼만 선택 (age_group은 필터링용이므로 분석에서 제외)
    available_cols = df.columns.tolist()
    desired_cols = ["year", "week", "ili", "detection_rate"]

    # 실제로 존재하는 컬럼만 선택
    cols = [c for c in desired_cols if c in available_cols]
    if len(cols) < len(desired_cols):
        missing = set(desired_cols) - set(cols)
        log(f"⚠️  누락된 컬럼: {missing}")

    log(f"\n📋 선택된 컬럼 (상관계수/모델 입력용): {cols}")

    # 선택된 컬럼만 남기기
    df = df[cols].copy()

    # 팬데믹 기간 생략: 해당 구간 행 제거 (test2용)
    if drop_pandemic and 'year' in df.columns and 'week' in df.columns:
        pandemic_mask = get_pandemic_mask(df)
        n_drop = pandemic_mask.sum()
        df = df[~pandemic_mask].copy()
        log(f"\n📌 팬데믹 기간 생략: {n_drop}건 제거 (2020년 14주 ~ 2022년 22주)")

    # 팬데믹 기간 보간 (2017-2019년 계절성 패턴 사용) — 생략하지 않았을 때만 적용 가능
    if interpolate_pandemic and not drop_pandemic:
        target_cols_for_pandemic = [c for c in cols if c not in ['year', 'week']]
        df = interpolate_pandemic_period(df, target_cols=target_cols_for_pandemic)

    # 나머지 결측값 처리 (팬데믹 보간 후 남은 NaN)
    log(f"\n🔧 전처리 결측값 처리 중...")
    valid_cols = []
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # 전체가 NaN인 컬럼 체크
        if df[col].isna().all():
            log(f"   ❌ {col}: 전체 NaN - 피처에서 제외")
            continue
        
        if df[col].isna().any():
            nan_count = df[col].isna().sum()
            nan_pct = nan_count / len(df) * 100
            log(f"   {col}: {nan_count}개 ({nan_pct:.1f}%) 결측값 처리")
            df[col] = df[col].interpolate(method="linear").ffill().bfill()
            if df[col].isna().any():
                fill_val = df[col].median() if not np.isnan(df[col].median()) else 0.0
                df[col] = df[col].fillna(fill_val)
        
        valid_cols.append(col)
    
    if len(valid_cols) < len(cols):
        removed = set(cols) - set(valid_cols)
        log(f"   ⚠️  제거된 피처: {removed}")
    
    cols = valid_cols
    df = df[cols].copy()  # 유효한 컬럼만 유지

    remaining_nans = df.isna().sum().sum()
    log(f"   ✅ 결측값 처리 완료 (남은 NaN: {remaining_nans}개, 유효 피처: {len(cols)}개)")

    # 소수점 둘째 자리로 반올림
    df = df.round(2)

    return df, cols

print("\n" + "="*60)
print("📊 PostgreSQL에서 인플루엔자 데이터 로드")
print("="*60)

# PostgreSQL에서 인플루엔자 데이터 로드
try:
    df_raw = load_from_postgres(table_name="influenza_data")
    print(f"✅ PostgreSQL influenza_data 로드 완료: {df_raw.shape}")
    print(f"   컬럼: {list(df_raw.columns)}")
except Exception as e:
    print(f"❌ PostgreSQL 로드 실패: {e}")
    print("   CSV 파일로 대체 시도...")
    df_raw = pd.read_csv("merged_influenza_data.csv")
    print(f"✅ CSV 파일 로드 완료: {df_raw.shape}")

df, cols = preprocess_data(df_raw.copy())

# 전처리 결과 저장 후 입력 데이터로 사용
final_data_path = "final_data.csv"
df.to_csv(final_data_path, index=False, encoding="utf-8-sig")
print(f"✅ 전처리 결과 저장: {final_data_path}")
df = pd.read_csv(final_data_path)

# 상관계수 분석용 데이터
df_numeric = df[cols].copy()

# ---- test1.csv (팬데믹 기간 생략 안 함) / test2.csv (팬데믹 기간 생략) ----
print("\n" + "="*60)
print("📁 test1.csv / test2.csv 생성 및 ILI 그래프")
print("="*60)

df_test1, _ = preprocess_data(
    df_raw.copy(), interpolate_pandemic=False, drop_pandemic=False, verbose=False
)
df_test1.to_csv("test1.csv", index=False, encoding="utf-8-sig")
print(f"✅ test1.csv 저장 (팬데믹 기간 포함): {len(df_test1)}건")

# test2: 팬데믹 구간은 강제 삭제 후 2017~2019 주차별 패턴으로 보간 (행은 유지)
df_test2, _ = preprocess_data(
    df_raw.copy(), interpolate_pandemic=True, drop_pandemic=False, verbose=False
)
df_test2.to_csv("test2.csv", index=False, encoding="utf-8-sig")
print(f"✅ test2.csv 저장 (팬데믹 기간 삭제 후 2017~2019 데이터로 보간): {len(df_test2)}건")

# ILI 시계열 그래프 (연속 시간축: year + week/52)
def time_axis(df: pd.DataFrame) -> np.ndarray:
    if "year" not in df.columns or "week" not in df.columns:
        return np.arange(len(df))
    return df["year"].values + df["week"].values / 52.0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

# test1: 팬데믹 포함
x1 = time_axis(df_test1)
ax1.plot(x1, df_test1["ili"].values, color="steelblue", linewidth=0.8, label="ILI")
ax1.set_ylabel("ILI")
ax1.set_title("test1.csv — Pandemic period included (full period)")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# test2: 팬데믹 구간 삭제 후 2017~2019로 보간
x2 = time_axis(df_test2)
ax2.plot(x2, df_test2["ili"].values, color="coral", linewidth=0.8, label="ILI")
ax2.set_ylabel("ILI")
ax2.set_xlabel("Year")
ax2.set_title("test2.csv — Pandemic period replaced by 2017–2019 linear interpolation")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("check_dataset_ili_compare.png", dpi=150, bbox_inches="tight")
print("✅ ILI 비교 그래프 저장: check_dataset_ili_compare.png")
plt.close()
