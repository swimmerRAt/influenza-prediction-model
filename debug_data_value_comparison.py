"""
19-49세 데이터 값 비교 스크립트
merged_influenza_data.csv의 19-49세 데이터와 patchTST 파싱 데이터의 값을 정밀 비교합니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path.cwd()

# CSV 파일 경로
CSV_PATH = BASE_DIR / "merged_influenza_data.csv"
PARSED_CSV_PATH = BASE_DIR / "debug_parsed_data.csv"

print(f"\n{'='*80}")
print(f"🔍 19-49세 데이터 값 비교 분석")
print(f"{'='*80}\n")

# Step 1: merged_influenza_data.csv 로드
print(f"📄 STEP 1: merged_influenza_data.csv 로드")
print(f"{'='*80}")
df_csv = pd.read_csv(CSV_PATH)
print(f"✅ 원본 CSV 로드: {df_csv.shape}")
print(f"   컬럼: {list(df_csv.columns)}")

# Step 2: 19-49세만 필터링
print(f"\n📊 STEP 2: 19-49세 연령대 필터링")
print(f"{'='*80}")
if 'age_group' in df_csv.columns:
    df_19_49 = df_csv[df_csv['age_group'] == '19-49세'].copy()
    print(f"✅ 19-49세 데이터 추출: {df_19_49.shape}")
    
    # 정렬
    if 'year' in df_19_49.columns and 'week' in df_19_49.columns:
        df_19_49 = df_19_49.sort_values(['year', 'week']).reset_index(drop=True)
        print(f"   year, week 기준 정렬 완료")
    
    # 통계
    print(f"\n📈 19-49세 데이터 통계:")
    for col in ['ili', 'detection_rate', 'hospitalization', 'emergency_patients']:
        if col in df_19_49.columns:
            vals = pd.to_numeric(df_19_49[col], errors='coerce')
            non_na = vals.notna().sum()
            print(f"   [{col}]")
            print(f"      유효 데이터: {non_na}/{len(vals)}개 ({non_na/len(vals)*100:.1f}%)")
            if non_na > 0:
                print(f"      범위: [{vals.min():.2f}, {vals.max():.2f}]")
                print(f"      평균: {vals.mean():.2f}")
else:
    print(f"❌ age_group 컬럼이 없습니다!")
    exit(1)

# Step 3: patchTST 파싱 데이터 로드
print(f"\n🔧 STEP 3: patchTST 파싱 데이터 로드")
print(f"{'='*80}")
if PARSED_CSV_PATH.exists():
    df_parsed = pd.read_csv(PARSED_CSV_PATH)
    print(f"✅ 파싱 데이터 로드: {df_parsed.shape}")
    print(f"   컬럼: {list(df_parsed.columns)}")
else:
    print(f"❌ {PARSED_CSV_PATH} 파일이 없습니다!")
    print(f"   먼저 debug_data_parsing.py를 실행하세요.")
    exit(1)

# Step 4: Shape 비교
print(f"\n📊 STEP 4: Shape 비교")
print(f"{'='*80}")
print(f"   19-49세 원본: {df_19_49.shape}")
print(f"   파싱 데이터:  {df_parsed.shape}")

if df_19_49.shape[0] != df_parsed.shape[0]:
    print(f"\n⚠️  행 개수 불일치!")
    print(f"   차이: {abs(df_19_49.shape[0] - df_parsed.shape[0])}행")
else:
    print(f"\n✅ 행 개수 일치: {df_19_49.shape[0]}행")

# Step 5: 값 일대일 비교
print(f"\n🎯 STEP 5: 값 일대일 비교 (19-49세 원본 vs 파싱)")
print(f"{'='*80}")

comparison_cols = ['ili', 'detection_rate', 'hospitalization', 'emergency_patients', 'vaccine_rate']
min_len = min(len(df_19_49), len(df_parsed))

for col in comparison_cols:
    if col not in df_19_49.columns:
        print(f"\n[{col}] - 19-49세 원본에 없음, 건너뜀")
        continue
    if col not in df_parsed.columns:
        print(f"\n[{col}] - 파싱 데이터에 없음, 건너뜀")
        continue
    
    print(f"\n{'='*60}")
    print(f"[{col}] 비교")
    print(f"{'='*60}")
    
    # 값 추출
    vals_orig = pd.to_numeric(df_19_49[col], errors='coerce').iloc[:min_len]
    vals_parsed = pd.to_numeric(df_parsed[col], errors='coerce').iloc[:min_len]
    
    # 결측치 확인
    na_orig = vals_orig.isna().sum()
    na_parsed = vals_parsed.isna().sum()
    print(f"   결측치: 원본={na_orig}개, 파싱={na_parsed}개")
    
    # 비결측치만 비교
    both_valid = vals_orig.notna() & vals_parsed.notna()
    n_valid = both_valid.sum()
    
    if n_valid == 0:
        print(f"   ⚠️  비교 가능한 유효 데이터 없음")
        continue
    
    print(f"   비교 가능한 행: {n_valid}/{min_len}개")
    
    # 차이 계산
    diff = (vals_orig[both_valid] - vals_parsed[both_valid]).abs()
    max_diff = diff.max()
    mean_diff = diff.mean()
    median_diff = diff.median()
    num_different = (diff > 0.0001).sum()
    
    print(f"\n   📊 차이 통계:")
    print(f"      최대 차이:     {max_diff:.6f}")
    print(f"      평균 차이:     {mean_diff:.6f}")
    print(f"      중간값 차이:   {median_diff:.6f}")
    print(f"      차이 있는 행:  {num_different}개 ({num_different/n_valid*100:.1f}%)")
    
    if num_different > 0:
        print(f"\n   ⚠️  값 차이 감지!")
        
        # 차이가 큰 상위 10개 행
        top_diff_idx = diff.nlargest(min(10, num_different)).index.tolist()
        print(f"\n   차이가 큰 상위 {len(top_diff_idx)}개 행:")
        print(f"   {'행':>5} {'년도':>6} {'주차':>4} {'원본':>12} {'파싱':>12} {'차이':>12}")
        print(f"   {'-'*60}")
        
        for idx in top_diff_idx:
            # both_valid의 원래 인덱스 사용
            orig_idx = both_valid[both_valid].index[list(both_valid[both_valid].index).index(idx)]
            
            year_val = df_19_49.iloc[orig_idx].get('year', np.nan)
            week_val = df_19_49.iloc[orig_idx].get('week', np.nan)
            orig_val = vals_orig.iloc[orig_idx]
            parsed_val = vals_parsed.iloc[orig_idx]
            diff_val = abs(orig_val - parsed_val)
            
            print(f"   {orig_idx:>5} {year_val:>6.0f} {week_val:>4.0f} {orig_val:>12.4f} {parsed_val:>12.4f} {diff_val:>12.6f}")
    else:
        print(f"\n   ✅ 모든 값 완벽히 일치!")

# Step 6: 샘플 10개 상세 비교
print(f"\n🔎 STEP 6: 샘플 10개 상세 비교")
print(f"{'='*80}")

for i in range(min(10, min_len)):
    print(f"\n[샘플 {i}]")
    
    # 19-49세 원본
    orig_row = df_19_49.iloc[i]
    year = orig_row.get('year', '?')
    week = orig_row.get('week', '?')
    print(f"  {year:.0f}년 {week:.0f}주")
    
    # 파싱 데이터
    parsed_row = df_parsed.iloc[i]
    
    # 주요 컬럼 비교
    print(f"  {'컬럼':>20} {'원본':>12} {'파싱':>12} {'차이':>12} {'상태':>6}")
    print(f"  {'-'*62}")
    
    for col in ['ili', 'detection_rate', 'hospitalization', 'emergency_patients']:
        if col in orig_row.index and col in parsed_row.index:
            orig_val = pd.to_numeric(orig_row[col], errors='coerce')
            parsed_val = pd.to_numeric(parsed_row[col], errors='coerce')
            
            if pd.notna(orig_val) and pd.notna(parsed_val):
                diff_val = abs(orig_val - parsed_val)
                status = "✅" if diff_val < 0.0001 else "⚠️"
                print(f"  {col:>20} {orig_val:>12.4f} {parsed_val:>12.4f} {diff_val:>12.6f} {status:>6}")
            elif pd.isna(orig_val) and pd.isna(parsed_val):
                print(f"  {col:>20} {'NaN':>12} {'NaN':>12} {'-':>12} {'✅':>6}")
            else:
                status = "❌"
                print(f"  {col:>20} {str(orig_val):>12} {str(parsed_val):>12} {'-':>12} {status:>6}")

# Step 7: 19-49세 필터링 데이터 저장
print(f"\n💾 STEP 7: 19-49세 필터링 데이터 저장")
print(f"{'='*80}")
output_path = BASE_DIR / "debug_age_19_49_filtered.csv"
df_19_49.to_csv(output_path, index=False)
print(f"✅ 저장 완료: {output_path}")
print(f"   Shape: {df_19_49.shape}")

print(f"\n{'='*80}")
print(f"✅ 비교 분석 완료!")
print(f"{'='*80}")
print(f"\n생성된 파일:")
print(f"   - debug_age_19_49_filtered.csv: CSV에서 19-49세만 추출")
print(f"\n💡 결론:")
print(f"   - 위 차이 통계를 확인하여 데이터 변형 여부 판단")
print(f"   - 차이가 있다면 patchTST_simple.py의 보간/변환 로직 검토 필요")
