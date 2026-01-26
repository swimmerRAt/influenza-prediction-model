"""
patchTST.py 수정 검증 스크립트
정렬 수정 후 CSV 원본과 파싱 데이터가 일치하는지 확인
"""

import pandas as pd
from patchTST import load_and_prepare
from database.db_utils import load_from_postgres

print(f"\n{'='*80}")
print(f"🔍 patchTST.py 정렬 수정 검증")
print(f"{'='*80}\n")

# Step 1: CSV 원본에서 19-49세 추출
print(f"📄 STEP 1: CSV 원본 (19-49세만)")
print(f"{'='*80}")
csv_path = 'merged_influenza_data.csv'
df_csv = pd.read_csv(csv_path)
df_19_49 = df_csv[df_csv['age_group'] == '19-49세'].copy()
df_19_49 = df_19_49.sort_values(['year', 'week']).reset_index(drop=True)
print(f"✅ 19-49세 데이터: {df_19_49.shape}")
print(f"\n첫 10행:")
print(df_19_49[['year', 'week', 'ili', 'detection_rate']].head(10))

# Step 2: patchTST.py로 파싱
print(f"\n🔧 STEP 2: patchTST.py로 파싱")
print(f"{'='*80}")
df_pg = load_from_postgres('influenza_data')
X, y, labels, feat_names = load_and_prepare(df=df_pg, use_exog='all')
print(f"✅ 파싱 완료: X={X.shape}, y={y.shape}")

# y값을 DataFrame으로 변환하여 비교
import numpy as np
df_parsed = pd.DataFrame({
    'ili': y,
    'label': labels
})

# labels에서 year, week 추출
df_parsed['year'] = df_parsed['label'].str.extract(r'(\d{4})-\d{4}').astype(float)
df_parsed['week'] = df_parsed['label'].str.extract(r'W(\d+)').astype(float)

print(f"\n첫 10행:")
print(df_parsed[['year', 'week', 'ili']].head(10))

# Step 3: 값 비교
print(f"\n🎯 STEP 3: 값 일치 검증")
print(f"{'='*80}")

# ili 값만 비교
min_len = min(len(df_19_49), len(df_parsed))
ili_orig = df_19_49['ili'].iloc[:min_len].values
ili_parsed = y[:min_len]

diff = np.abs(ili_orig - ili_parsed)
max_diff = diff.max()
mean_diff = diff.mean()
num_match = (diff < 0.0001).sum()

print(f"\n📊 비교 결과:")
print(f"   비교 행 수:    {min_len}행")
print(f"   최대 차이:     {max_diff:.6f}")
print(f"   평균 차이:     {mean_diff:.6f}")
print(f"   일치하는 행:   {num_match}/{min_len} ({num_match/min_len*100:.1f}%)")

if num_match == min_len:
    print(f"\n✅ 완벽히 일치! 정렬 수정 성공!")
else:
    print(f"\n⚠️  일치하지 않는 행이 있습니다. 차이가 큰 상위 5개:")
    top_diff_idx = np.argsort(diff)[-5:][::-1]
    for idx in top_diff_idx:
        print(f"   행 {idx}: 원본={ili_orig[idx]:.2f}, 파싱={ili_parsed[idx]:.2f}, 차이={diff[idx]:.4f}")

# Step 4: 샘플 10개 상세 비교
print(f"\n🔎 STEP 4: 샘플 10개 상세 비교")
print(f"{'='*80}")
print(f"{'행':>5} {'년도':>6} {'주차':>4} {'CSV 원본':>12} {'patchTST':>12} {'차이':>12} {'상태':>6}")
print(f"{'-'*65}")

for i in range(10):
    year_orig = df_19_49.iloc[i]['year']
    week_orig = df_19_49.iloc[i]['week']
    ili_orig_val = df_19_49.iloc[i]['ili']
    ili_parsed_val = y[i]
    diff_val = abs(ili_orig_val - ili_parsed_val)
    status = "✅" if diff_val < 0.0001 else "⚠️"
    
    print(f"{i:>5} {year_orig:>6.0f} {week_orig:>4.0f} {ili_orig_val:>12.2f} {ili_parsed_val:>12.2f} {diff_val:>12.6f} {status:>6}")

print(f"\n{'='*80}")
print(f"✅ 검증 완료!")
print(f"{'='*80}")
