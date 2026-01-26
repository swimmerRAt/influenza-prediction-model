"""
정렬 문제 디버그 스크립트
patchTST_simple.py의 정렬 로직 문제를 단계별로 진단합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path.cwd()
CSV_PATH = BASE_DIR / "merged_influenza_data.csv"

print(f"\n{'='*80}")
print(f"🔍 정렬 로직 문제 진단")
print(f"{'='*80}\n")

# Step 1: 원본 데이터 로드
print(f"📄 STEP 1: 19-49세 데이터 로드")
print(f"{'='*80}")
df = pd.read_csv(CSV_PATH)
df = df[df['age_group'] == '19-49세'].copy()
print(f"✅ 19-49세 데이터: {df.shape}")

# Step 2: year, week 기준 정렬 (현재 방식)
print(f"\n🔧 STEP 2: year, week 기준 정렬 (1차 정렬)")
print(f"{'='*80}")
df = df.sort_values(['year', 'week']).reset_index(drop=True)
print(f"정렬 후 처음 10행:")
print(df[['year', 'week', 'ili', 'detection_rate']].head(10))

# Step 3: season_norm 생성 (문제의 로직)
print(f"\n⚠️  STEP 3: season_norm 생성 (week 36 기준)")
print(f"{'='*80}")
print(f"로직: week >= 36이면 '현재연도-다음연도', 아니면 '이전연도-현재연도'")

def _norm_season_text(s: str) -> str:
    """시즌 텍스트 정규화"""
    if pd.isna(s):
        return ""
    s = str(s).strip()
    if not s:
        return ""
    parts = s.split("-")
    if len(parts) >= 2:
        return f"{parts[0].strip()}-{parts[1].strip()}"
    return s

df['season_norm'] = df.apply(
    lambda row: f"{int(row['year'])}-{int(row['year'])+1}" if row['week'] >= 36 
               else f"{int(row['year'])-1}-{int(row['year'])}",
    axis=1
)

print(f"\nseason_norm 생성 후 처음 10행:")
print(df[['year', 'week', 'season_norm', 'ili']].head(10))

print(f"\nseason_norm 고유값:")
print(sorted(df['season_norm'].unique()))

# Step 4: season_norm + week 기준 재정렬 (2차 정렬 - 문제 발생!)
print(f"\n🚨 STEP 4: season_norm, week 기준 재정렬 (2차 정렬)")
print(f"{'='*80}")
print(f"⚠️  여기서 데이터가 뒤섞입니다!")

df["season_norm"] = df["season_norm"].astype(str).map(_norm_season_text)
df["week"] = pd.to_numeric(df["week"], errors="coerce")

print(f"\n재정렬 전 처음 10행:")
print(df[['year', 'week', 'season_norm', 'ili', 'detection_rate']].head(10))

df_sorted = df.sort_values(["season_norm", "week"]).reset_index(drop=True)

print(f"\n재정렬 후 처음 10행:")
print(df_sorted[['year', 'week', 'season_norm', 'ili', 'detection_rate']].head(10))

# Step 5: 문제 분석
print(f"\n📊 STEP 5: 문제 분석")
print(f"{'='*80}")

print(f"\n❌ 문제점:")
print(f"   1. season_norm은 문자열 정렬 → '2016-2017' < '2017-2018' < '2018-2019'")
print(f"   2. 2017-2018 시즌의 데이터 순서:")
print(f"      - week 36~53 (2017년) → week 1~35 (2018년)")
print(f"      - 이것이 올바른 시간 순서입니다!")
print(f"   3. 그런데 실제 데이터를 보면:")

# 2017-2018 시즌만 필터링
season_2017_2018 = df_sorted[df_sorted['season_norm'] == '2017-2018'].copy()
print(f"\n   2017-2018 시즌 데이터:")
print(season_2017_2018[['year', 'week', 'ili', 'detection_rate']].head(20))

print(f"\n   ✅ 정렬 자체는 정상입니다!")
print(f"      - week 1~53 순서로 정렬")
print(f"      - 하지만 season_norm 생성 로직에 문제가 있을 수 있습니다.")

# Step 6: 원본 CSV와 비교
print(f"\n🔎 STEP 6: 원본 CSV 2017-2018 시즌 확인")
print(f"{'='*80}")

# year, week 기준 정렬만 한 버전
df_simple = pd.read_csv(CSV_PATH)
df_simple = df_simple[df_simple['age_group'] == '19-49세'].copy()
df_simple = df_simple.sort_values(['year', 'week']).reset_index(drop=True)

print(f"\n원본 CSV (year, week 정렬만):")
print(f"2017년 36주부터 2018년 10주까지:")
mask = ((df_simple['year'] == 2017) & (df_simple['week'] >= 36)) | \
       ((df_simple['year'] == 2018) & (df_simple['week'] <= 10))
print(df_simple[mask][['year', 'week', 'ili', 'detection_rate']])

print(f"\npatchTST 파싱 (season_norm 정렬):")
print(f"2017-2018 시즌 처음 20행:")
print(df_sorted[df_sorted['season_norm'] == '2017-2018'][['year', 'week', 'ili', 'detection_rate']].head(20))

# Step 7: 값 비교
print(f"\n⚠️  STEP 7: 첫 10개 샘플 값 비교")
print(f"{'='*80}")

print(f"\n원본 (year, week 정렬)의 첫 10개:")
for i in range(10):
    row = df_simple.iloc[i]
    print(f"  행 {i}: {row['year']:.0f}년 {row['week']:.0f}주 - ili={row['ili']:.1f}, det={row['detection_rate']:.1f}")

print(f"\n파싱 (season_norm 정렬)의 첫 10개:")
for i in range(10):
    row = df_sorted.iloc[i]
    print(f"  행 {i}: {row['year']:.0f}년 {row['week']:.0f}주 - ili={row['ili']:.1f}, det={row['detection_rate']:.1f}")

# Step 8: 결론
print(f"\n{'='*80}")
print(f"📋 결론")
print(f"{'='*80}")

print(f"\n🔍 정렬 순서 차이:")
print(f"   원본:  2017년 36주부터 시작 (시간 순서)")
print(f"   파싱:  2017-2018 시즌 = 2018년 1주부터 시작!")
print(f"")
print(f"❌ 근본 원인:")
print(f"   season_norm 기준 정렬 시:")
print(f"   - '2017-2018' 시즌 내에서 week 1~53 순서로 정렬")
print(f"   - 즉, 2018년 1주(week=1)가 2017년 36주(week=36)보다 앞에 옴!")
print(f"   - 시간 순서가 완전히 뒤바뀜!")
print(f"")
print(f"✅ 해결 방법:")
print(f"   1. season_norm 정렬을 제거하고 year, week만 사용")
print(f"   2. 또는 시즌 내 올바른 순서를 위한 별도 정렬 키 생성")
print(f"      예: season_week = (year - 2017) * 100 + week")
