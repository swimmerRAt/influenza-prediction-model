"""
PostgreSQL 데이터베이스 검증 스크립트
- 인플루엔자 데이터: 전체 컬럼 정렬 검증 (ili, detection_rate, hospitalization, emergency_patients 등)
- 트렌드 데이터: Google, Naver, Twitter Trends 데이터 검증
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# patchTST.py import를 위해 상위 디렉토리 추가
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from patchTST import load_and_prepare
try:
    from .db_utils import load_from_postgres, load_trends_from_postgres
except ImportError:
    from db_utils import load_from_postgres, load_trends_from_postgres


def validate_influenza_data():
    """인플루엔자 데이터 검증 (전체 컬럼 정렬)"""
    
    print(f"\n{'='*80}")
    print(f"🔍 인플루엔자 데이터 정렬 검증")
    print(f"{'='*80}\n")

    # Step 1: CSV 원본에서 19-49세 추출
    print(f"📄 STEP 1: CSV 원본 (19-49세)")
    print(f"{'='*80}")
    csv_path = parent_dir / 'merged_influenza_data.csv'
    
    if not csv_path.exists():
        print(f"❌ CSV 파일이 없습니다: {csv_path}")
        print(f"   먼저 'python database/update_database.py'를 실행하세요.")
        return False
    
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
    # 존재하는 컬럼만 출력
    display_cols = ['year', 'week', 'ili']
    for col in ['detection_rate', 'hospitalization', 'emergency_patients']:
        if col in df_parsed.columns:
            display_cols.append(col)
    print(df_parsed[display_cols].head())

    # Step 4: 각 컬럼별 비교
    print(f"\n🎯 STEP 4: 컬럼별 값 일치 검증")
    print(f"{'='*80}")

    # 비교할 컬럼 목록 (존재하는 컬럼만)
    compare_columns = [col for col in ['ili', 'detection_rate', 'hospitalization', 'emergency_patients'] 
                       if col in df_parsed.columns]

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
        if n_non_pandemic > 0 and match_pct_non_pandemic < 100:
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
    print(f"📋 STEP 6: 인플루엔자 데이터 검증 요약")
    print(f"{'='*80}")

    summary_df = pd.DataFrame(results)
    if len(summary_df) > 0:
        print(f"\n전체 컬럼 일치도:")
        print(summary_df.to_string(index=False))
        
        all_match = summary_df['match_pct'].min()
        if all_match >= 99.9:
            print(f"\n✅ 모든 컬럼이 거의 완벽히 일치합니다! (최소 {all_match:.1f}%)")
            return True
        elif all_match >= 90:
            print(f"\n⚠️  일부 불일치가 있지만 대부분 일치합니다. (최소 {all_match:.1f}%)")
            print(f"    불일치는 주로 팬데믹 기간 보간으로 인한 것입니다.")
            return True
        else:
            print(f"\n❌ 심각한 불일치 발견! (최소 {all_match:.1f}%)")
            print(f"    정렬 로직을 다시 확인해야 합니다.")
            return False
    else:
        print(f"\n⚠️  검증 가능한 데이터가 없습니다.")
        return False


def validate_trends_data():
    """트렌드 데이터 검증 (Google, Naver, Twitter)"""
    
    print(f"\n{'='*80}")
    print(f"🔍 트렌드 데이터 검증")
    print(f"{'='*80}\n")
    
    # Step 1: CSV 백업 파일 로드
    print(f"📄 STEP 1: CSV 백업 파일 로드")
    print(f"{'='*80}")
    csv_path = parent_dir / 'trends_data.csv'
    
    if not csv_path.exists():
        print(f"⚠️  트렌드 CSV 파일이 없습니다: {csv_path}")
        print(f"   먼저 'python database/update_database.py'를 실행하세요.")
        return False
    
    df_csv = pd.read_csv(csv_path)
    df_csv = df_csv.sort_values(['year', 'week']).reset_index(drop=True)
    print(f"✅ CSV 백업: {df_csv.shape}")
    print(f"   컬럼 수: {len(df_csv.columns)}")
    
    # 트렌드 컬럼 분류
    google_cols = [c for c in df_csv.columns if c.startswith('google_')]
    naver_cols = [c for c in df_csv.columns if c.startswith('naver_')]
    twitter_cols = [c for c in df_csv.columns if c.startswith('twitter_')]
    
    print(f"   Google Trends: {len(google_cols)}개 컬럼")
    print(f"   Naver Trends: {len(naver_cols)}개 컬럼")
    print(f"   Twitter Trends: {len(twitter_cols)}개 컬럼")
    
    # Step 2: PostgreSQL 트렌드 DB 로드
    print(f"\n🔧 STEP 2: PostgreSQL trends DB 로드")
    print(f"{'='*80}")
    
    try:
        df_pg = load_trends_from_postgres()
        if df_pg.empty:
            print(f"❌ PostgreSQL trends DB가 비어있습니다.")
            return False
        
        df_pg = df_pg.sort_values(['year', 'week']).reset_index(drop=True)
        print(f"✅ PostgreSQL 로드: {df_pg.shape}")
        print(f"   컬럼 수: {len(df_pg.columns)}")
    except Exception as e:
        print(f"❌ PostgreSQL 로드 실패: {e}")
        print(f"   먼저 'python database/update_database.py'를 실행하세요.")
        return False
    
    # Step 3: 데이터 비교
    print(f"\n🎯 STEP 3: CSV vs PostgreSQL 비교")
    print(f"{'='*80}")
    
    # 행 수 비교
    if len(df_csv) != len(df_pg):
        print(f"⚠️  행 수 불일치:")
        print(f"   CSV: {len(df_csv)}행")
        print(f"   PostgreSQL: {len(df_pg)}행")
    else:
        print(f"✅ 행 수 일치: {len(df_csv)}행")
    
    # 컬럼 수 비교
    if len(df_csv.columns) != len(df_pg.columns):
        print(f"\n⚠️  컬럼 수 불일치:")
        print(f"   CSV: {len(df_csv.columns)}개")
        print(f"   PostgreSQL: {len(df_pg.columns)}개")
        
        csv_only = set(df_csv.columns) - set(df_pg.columns)
        pg_only = set(df_pg.columns) - set(df_csv.columns)
        
        if csv_only:
            print(f"   CSV에만 있는 컬럼: {csv_only}")
        if pg_only:
            print(f"   PostgreSQL에만 있는 컬럼: {pg_only}")
    else:
        print(f"✅ 컬럼 수 일치: {len(df_csv.columns)}개")
    
    # Step 4: 값 일치 검증
    print(f"\n🔎 STEP 4: 값 일치 검증")
    print(f"{'='*80}")
    
    common_cols = list(set(df_csv.columns) & set(df_pg.columns))
    common_cols = [c for c in common_cols if c not in ['year', 'week']]
    
    results = []
    
    for col in common_cols:
        vals_csv = pd.to_numeric(df_csv[col], errors='coerce').values
        vals_pg = pd.to_numeric(df_pg[col], errors='coerce').values
        
        min_len = min(len(vals_csv), len(vals_pg))
        vals_csv = vals_csv[:min_len]
        vals_pg = vals_pg[:min_len]
        
        # NaN 처리
        valid_mask = ~(np.isnan(vals_csv) | np.isnan(vals_pg))
        n_valid = valid_mask.sum()
        
        if n_valid == 0:
            continue
        
        # 차이 계산
        diff = np.abs(vals_csv[valid_mask] - vals_pg[valid_mask])
        max_diff = diff.max()
        mean_diff = diff.mean()
        num_match = (diff < 0.0001).sum()
        match_pct = num_match / n_valid * 100
        
        results.append({
            'column': col,
            'valid_rows': n_valid,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'match_count': num_match,
            'match_pct': match_pct
        })
    
    # Step 5: 요약
    print(f"\n📋 STEP 5: 트렌드 데이터 검증 요약")
    print(f"{'='*80}")
    
    if len(results) > 0:
        summary_df = pd.DataFrame(results)
        
        # 상위/하위 10개만 표시
        print(f"\n일치도가 낮은 상위 10개 컬럼:")
        print(summary_df.nsmallest(min(10, len(summary_df)), 'match_pct').to_string(index=False))
        
        print(f"\n일치도가 높은 상위 10개 컬럼:")
        print(summary_df.nlargest(min(10, len(summary_df)), 'match_pct').to_string(index=False))
        
        all_match = summary_df['match_pct'].min()
        avg_match = summary_df['match_pct'].mean()
        
        print(f"\n전체 통계:")
        print(f"   검증 컬럼 수: {len(summary_df)}개")
        print(f"   평균 일치도: {avg_match:.2f}%")
        print(f"   최소 일치도: {all_match:.2f}%")
        
        if all_match >= 99.9:
            print(f"\n✅ 모든 트렌드 컬럼이 거의 완벽히 일치합니다!")
            return True
        elif all_match >= 95:
            print(f"\n✅ 대부분의 트렌드 컬럼이 일치합니다.")
            return True
        else:
            print(f"\n⚠️  일부 컬럼에서 불일치가 발견되었습니다.")
            print(f"    확인이 필요합니다.")
            return False
    else:
        print(f"\n⚠️  검증 가능한 데이터가 없습니다.")
        return False


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"📊 PostgreSQL 데이터베이스 통합 검증")
    print(f"{'='*80}")
    
    success_count = 0
    total_tests = 2
    
    # 인플루엔자 데이터 검증
    if validate_influenza_data():
        success_count += 1
        print(f"\n✅ 인플루엔자 데이터 검증 완료!")
    else:
        print(f"\n❌ 인플루엔자 데이터 검증 실패!")
    
    # 트렌드 데이터 검증
    if validate_trends_data():
        success_count += 1
        print(f"\n✅ 트렌드 데이터 검증 완료!")
    else:
        print(f"\n❌ 트렌드 데이터 검증 실패!")
    
    # 최종 결과
    print(f"\n{'='*80}")
    print(f"📊 최종 검증 결과")
    print(f"{'='*80}")
    print(f"\n검증 결과: {success_count}/{total_tests} 성공")
    
    if success_count == total_tests:
        print(f"\n✅ 모든 데이터베이스 검증 완료!")
        print(f"   데이터가 올바르게 저장되어 있습니다.")
    elif success_count > 0:
        print(f"\n⚠️  일부 데이터베이스 검증 실패")
        print(f"   실패한 항목을 확인하고 다시 업데이트하세요.")
        print(f"   python database/update_database.py")
    else:
        print(f"\n❌ 모든 데이터베이스 검증 실패")
        print(f"   먼저 데이터베이스를 업데이트하세요:")
        print(f"   python database/update_database.py")
    
    print(f"\n{'='*80}")
