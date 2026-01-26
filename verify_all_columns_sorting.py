"""
전체 컬럼 정렬 검증 스크립트
ili, detection_rate, hospitalization, emergency_patients 등 모든 컬럼이 올바르게 정렬되었는지 확인
"""

import pandas as pd
import numpy as np
from patchTST import load_and_prepare
from database.db_utils import load_from_postgres

print(f"\n{'='*80}")
print(f"🔍 전체 컬럼 정렬 검증")
print(f"{'='*80}\n")

# Step 1: CSV 원본에서 19-49세 추출
print(f"📄 STEP 1: CSV 원본 (19-49세)")
print(f"{'='*80}")
csv_path = 'merged_influenza_data.csv'
df_csv = pd.read_csv(csv_path)
df_19_49 = df_csv[df_csv['age_group'] == '19-49세'].copy()
df_19_49 = df_19_49.sort_values(['year', 'week']).reset_index(drop=True)
print(f"✅ 19-49세 데이터: {df_19_49.shape}")

# Step 2: patchTST.py로 파싱
print(f"\n🔧 STEP 2: patchTST.py로 파싱")
print(f"{'='*80}")
df_pg = load_from_postgres('influenza_data')
X, y, labels, feat_names = load_and_prepare(df=df_pg, use_exog='all')
print(f"✅ 파싱 완료: X={X.shape}, y={y.shape}")
print(f"   Features: {feat_names}")

# Step 3: X를 DataFrame으로 변환
print(f"\n🔄 STEP 3: 파싱 데이터를 DataFrame으로 변환")
print(f"{'='*80}")
df_parsed = pd.DataFrame(X, columns=feat_names)
df_parsed['ili'] = y

# labels에서 year, week 추출
df_parsed['year'] = pd.Series(labels).str.extract(r'(\d{4})-\d{4}').astype(float)
df_parsed['week'] = pd.Series(labels).str.extract(r'W(\d+)').astype(float)

print(f"✅ 변환 완료: {df_parsed.shape}")
print(f"\n처음 5행:")
print(df_parsed[['year', 'week', 'ili', 'detection_rate', 'hospitalization']].head())

# Step 4: 각 컬럼별 비교
print(f"\n🎯 STEP 4: 컬럼별 값 일치 검증")
print(f"{'='*80}")

# 비교할 컬럼 목록
compare_columns = ['ili', 'detection_rate', 'hospitalization', 'emergency_patients']

results = []

for col in compare_columns:
    print(f"\n{'='*60}")
    print(f"[{col}] 검증")
    print(f"{'='*60}")
    
    if col not in df_19_49.columns:
        print(f"⚠️  CSV 원본에 {col} 컬럼 없음")
        continue
    
    if col not in df_parsed.columns:
        print(f"⚠️  파싱 데이터에 {col} 컬럼 없음")
        continue
    
    # 값 추출
    min_len = min(len(df_19_49), len(df_parsed))
    vals_csv = pd.to_numeric(df_19_49[col], errors='coerce').iloc[:min_len].values
    vals_parsed = pd.to_numeric(df_parsed[col], errors='coerce').iloc[:min_len].values
    
    # NaN 처리
    valid_mask = ~(np.isnan(vals_csv) | np.isnan(vals_parsed))
    n_valid = valid_mask.sum()
    
    if n_valid == 0:
        print(f"⚠️  비교 가능한 유효 데이터 없음")
        continue
    
    # 차이 계산
    diff = np.abs(vals_csv[valid_mask] - vals_parsed[valid_mask])
    max_diff = diff.max()
    mean_diff = diff.mean()
    median_diff = np.median(diff)
    num_match = (diff < 0.0001).sum()
    match_pct = num_match / n_valid * 100
    
    print(f"\n📊 비교 결과:")
    print(f"   비교 가능한 행: {n_valid}/{min_len}개")
    print(f"   최대 차이:      {max_diff:.6f}")
    print(f"   평균 차이:      {mean_diff:.6f}")
    print(f"   중간값 차이:    {median_diff:.6f}")
    print(f"   일치하는 행:    {num_match}/{n_valid} ({match_pct:.1f}%)")
    
    # 결과 저장
    results.append({
        'column': col,
        'valid_rows': n_valid,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'match_count': num_match,
        'match_pct': match_pct
    })
    
    # 팬데믹 기간 제외 비교
    pandemic_mask = (
        ((df_19_49['year'].iloc[:min_len] == 2020) & (df_19_49['week'].iloc[:min_len] >= 14)) |
        (df_19_49['year'].iloc[:min_len] == 2021) |
        ((df_19_49['year'].iloc[:min_len] == 2022) & (df_19_49['week'].iloc[:min_len] <= 22))
    ).values
    
    non_pandemic_mask = valid_mask & ~pandemic_mask
    n_non_pandemic = non_pandemic_mask.sum()
    
    if n_non_pandemic > 0:
        diff_non_pandemic = np.abs(vals_csv[non_pandemic_mask] - vals_parsed[non_pandemic_mask])
        num_match_non_pandemic = (diff_non_pandemic < 0.0001).sum()
        match_pct_non_pandemic = num_match_non_pandemic / n_non_pandemic * 100
        
        print(f"\n   📌 팬데믹 기간 제외 비교:")
        print(f"      비교 행:        {n_non_pandemic}개")
        print(f"      일치하는 행:    {num_match_non_pandemic}/{n_non_pandemic} ({match_pct_non_pandemic:.1f}%)")
    
    # 차이가 큰 행 (팬데믹 제외)
    if match_pct_non_pandemic < 100:
        print(f"\n   ⚠️  팬데믹 제외 시에도 불일치 발견!")
        
        # non_pandemic_mask의 인덱스를 원래 배열로 복원
        non_pandemic_indices = np.where(non_pandemic_mask)[0]
        diff_at_indices = np.abs(vals_csv[non_pandemic_mask] - vals_parsed[non_pandemic_mask])
        top_5_local_idx = np.argsort(diff_at_indices)[-min(5, len(diff_at_indices)):][::-1]
        top_5_global_idx = non_pandemic_indices[top_5_local_idx]
        
        print(f"      차이가 큰 상위 {len(top_5_global_idx)}개 행:")
        print(f"      {'행':>5} {'년도':>6} {'주차':>4} {'CSV 원본':>12} {'파싱':>12} {'차이':>12}")
        print(f"      {'-'*60}")
        
        for idx in top_5_global_idx:
            year_val = df_19_49.iloc[idx]['year']
            week_val = df_19_49.iloc[idx]['week']
            csv_val = vals_csv[idx]
            parsed_val = vals_parsed[idx]
            diff_val = abs(csv_val - parsed_val)
            print(f"      {idx:>5} {year_val:>6.0f} {week_val:>4.0f} {csv_val:>12.2f} {parsed_val:>12.2f} {diff_val:>12.4f}")

# Step 5: 처음 20개 샘플 상세 비교
print(f"\n{'='*80}")
print(f"🔎 STEP 5: 처음 20개 샘플 상세 비교")
print(f"{'='*80}")

print(f"\n{'행':>5} {'년도':>6} {'주차':>4} {'컬럼':>20} {'CSV 원본':>12} {'파싱':>12} {'차이':>12} {'상태':>6}")
print(f"{'-'*85}")

for i in range(min(20, len(df_19_49), len(df_parsed))):
    year_val = df_19_49.iloc[i]['year']
    week_val = df_19_49.iloc[i]['week']
    
    for col in compare_columns:
        if col in df_19_49.columns and col in df_parsed.columns:
            csv_val = pd.to_numeric(df_19_49.iloc[i][col], errors='coerce')
            parsed_val = pd.to_numeric(df_parsed.iloc[i][col], errors='coerce')
            
            if pd.notna(csv_val) and pd.notna(parsed_val):
                diff_val = abs(csv_val - parsed_val)
                status = "✅" if diff_val < 0.0001 else "⚠️"
                print(f"{i:>5} {year_val:>6.0f} {week_val:>4.0f} {col:>20} {csv_val:>12.2f} {parsed_val:>12.2f} {diff_val:>12.6f} {status:>6}")

# Step 6: 요약
print(f"\n{'='*80}")
print(f"📋 STEP 6: 요약")
print(f"{'='*80}")

summary_df = pd.DataFrame(results)
if len(summary_df) > 0:
    print(f"\n전체 컬럼 일치도:")
    print(summary_df.to_string(index=False))
    
    all_match = summary_df['match_pct'].min()
    if all_match >= 99.9:
        print(f"\n✅ 모든 컬럼이 거의 완벽히 일치합니다! (최소 {all_match:.1f}%)")
    elif all_match >= 90:
        print(f"\n⚠️  일부 불일치가 있지만 대부분 일치합니다. (최소 {all_match:.1f}%)")
        print(f"    불일치는 주로 팬데믹 기간 보간으로 인한 것입니다.")
    else:
        print(f"\n❌ 심각한 불일치 발견! (최소 {all_match:.1f}%)")
        print(f"    정렬 로직을 다시 확인해야 합니다.")

print(f"\n{'='*80}")
print(f"✅ 검증 완료!")
print(f"{'='*80}")
