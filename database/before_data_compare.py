#!/usr/bin/env python3
"""
data/before 폴더와 merged_influenza_data.csv 비교 스크립트

비교 항목:
1. 원본 데이터 통계 (파일별 레코드 수)
2. 병합 데이터 통계
3. 연도/주차별 데이터 존재 여부 비교
4. 값 일치 여부 확인
"""

import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

# 데이터셋 ID → 컬럼 매핑
DATASET_COLUMN_MAPPING = {
    'ds_0101': {'value_col': '의사환자 분율', 'merged_col': 'ili', 'group_col': '연령대'},
    'ds_0103': {'value_col': '입원환자 수', 'merged_col': 'hospitalization', 'group_col': '연령대'},
    'ds_0104': {'value_col': '입원환자 수', 'merged_col': 'hospitalization', 'group_col': '연령대'},
    'ds_0105': {'value_col': '인플루엔자 검출률', 'merged_col': 'detection_rate', 'group_col': '아형'},
    'ds_0106': {'value_col': '인플루엔자 검출률', 'merged_col': 'detection_rate', 'group_col': '연령대'},
    'ds_0107': {'value_col': '검출률', 'merged_col': 'detection_rate', 'group_col': '아형'},
    'ds_0108': {'value_col': '검출률', 'merged_col': 'detection_rate', 'group_col': '연령대'},
    'ds_0109': {'value_col': '환자 수', 'merged_col': 'emergency_patients', 'group_col': '연령대'},
    'ds_0110': {'value_col': '접종률', 'merged_col': 'vaccine_rate', 'group_col': '연령대'},
}


def load_before_data(before_dir='data/before'):
    """
    data/before 폴더의 모든 CSV 파일 로드
    
    Returns:
        dict: {dsid: DataFrame}
    """
    before_path = Path(before_dir)
    if not before_path.exists():
        print(f"❌ {before_dir} 폴더가 존재하지 않습니다.")
        return {}
    
    csv_files = list(before_path.glob('*.csv'))
    print(f"\n📂 {before_dir} 폴더에서 {len(csv_files)}개 CSV 파일 발견\n")
    
    # 데이터셋별로 그룹화
    dataset_files = defaultdict(list)
    for f in csv_files:
        # flu-0101-2017.csv → ds_0101
        parts = f.stem.split('-')
        if len(parts) >= 2:
            dsid = f'ds_{parts[1]}'
            dataset_files[dsid].append(f)
    
    # 각 데이터셋별로 파일 로드 및 병합
    all_data = {}
    for dsid, files in sorted(dataset_files.items()):
        dfs = []
        total_rows = 0
        for f in sorted(files):
            try:
                df = pd.read_csv(f, encoding='utf-8-sig')
                df['source_file'] = f.name
                df['dsid'] = dsid
                dfs.append(df)
                total_rows += len(df)
            except Exception as e:
                print(f"   ⚠️  {f.name} 로드 실패: {e}")
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            all_data[dsid] = combined
            print(f"   📊 {dsid}: {len(files)}개 파일, {total_rows}행")
    
    return all_data


def load_merged_data(merged_path='merged_influenza_data.csv'):
    """
    merged_influenza_data.csv 로드
    """
    if not os.path.exists(merged_path):
        print(f"❌ {merged_path} 파일이 존재하지 않습니다.")
        return None
    
    df = pd.read_csv(merged_path)
    print(f"\n📊 병합 데이터: {len(df)}행, {len(df.columns)}개 컬럼")
    print(f"   컬럼: {list(df.columns)}")
    return df


def compare_statistics(before_data, merged_df):
    """
    기본 통계 비교
    """
    print("\n" + "="*70)
    print("📊 기본 통계 비교")
    print("="*70)
    
    # 원본 데이터 통계
    total_before = sum(len(df) for df in before_data.values())
    print(f"\n[원본 데이터 (data/before)]")
    print(f"   • 총 레코드 수: {total_before:,}")
    print(f"   • 데이터셋 수: {len(before_data)}")
    
    for dsid, df in sorted(before_data.items()):
        print(f"     - {dsid}: {len(df):,}행")
    
    # 병합 데이터 통계
    print(f"\n[병합 데이터 (merged_influenza_data.csv)]")
    print(f"   • 총 레코드 수: {len(merged_df):,}")
    print(f"   • 컬럼 수: {len(merged_df.columns)}")
    
    # 연도/주차 범위
    print(f"\n[데이터 범위]")
    print(f"   • 연도: {merged_df['year'].min()} ~ {merged_df['year'].max()}")
    print(f"   • 주차: {merged_df['week'].min()} ~ {merged_df['week'].max()}")
    
    # 연령대별 레코드 수
    print(f"\n[연령대별 레코드 수]")
    age_counts = merged_df['age_group'].value_counts().sort_index()
    for age, count in age_counts.items():
        print(f"     - {age}: {count:,}행")


def compare_year_week_coverage(before_data, merged_df):
    """
    연도/주차 커버리지 비교
    """
    print("\n" + "="*70)
    print("📅 연도/주차 커버리지 비교")
    print("="*70)
    
    # 원본 데이터의 연도/주차 집합
    before_year_weeks = set()
    for dsid, df in before_data.items():
        if '연도' in df.columns and '주차' in df.columns:
            for _, row in df.iterrows():
                before_year_weeks.add((int(row['연도']), int(row['주차'])))
    
    # 병합 데이터의 연도/주차 집합
    merged_year_weeks = set()
    for _, row in merged_df.iterrows():
        merged_year_weeks.add((int(row['year']), int(row['week'])))
    
    print(f"\n[원본 데이터]")
    print(f"   • 고유 (연도, 주차) 조합: {len(before_year_weeks)}개")
    
    print(f"\n[병합 데이터]")
    print(f"   • 고유 (연도, 주차) 조합: {len(merged_year_weeks)}개")
    
    # 누락된 연도/주차
    missing = before_year_weeks - merged_year_weeks
    extra = merged_year_weeks - before_year_weeks
    
    if missing:
        print(f"\n⚠️  원본에는 있지만 병합에 없는 (연도, 주차): {len(missing)}개")
        # 처음 10개만 표시
        for yw in sorted(missing)[:10]:
            print(f"      - {yw[0]}년 {yw[1]}주")
        if len(missing) > 10:
            print(f"      ... 외 {len(missing) - 10}개")
    else:
        print(f"\n✅ 모든 원본 연도/주차가 병합 데이터에 포함됨")
    
    if extra:
        print(f"\n📌 병합에는 있지만 원본에 없는 (연도, 주차): {len(extra)}개")
        # 이건 정상적인 경우가 많음 (API에서 추가 데이터)


def compare_values_sample(before_data, merged_df, sample_size=20):
    """
    특정 데이터셋의 값 비교 (샘플)
    """
    print("\n" + "="*70)
    print("🔍 값 비교 (ILI 데이터 샘플)")
    print("="*70)
    
    # ds_0101 (ILI 데이터)만 비교
    if 'ds_0101' not in before_data:
        print("   ds_0101 데이터가 없습니다.")
        return
    
    before_ili = before_data['ds_0101'].copy()
    before_ili = before_ili.rename(columns={
        '연도': 'year',
        '주차': 'week',
        '연령대': 'age_group',
        '의사환자 분율': 'ili_before'
    })
    
    # 숫자 타입으로 변환
    before_ili['year'] = before_ili['year'].astype(int)
    before_ili['week'] = before_ili['week'].astype(int)
    before_ili['ili_before'] = pd.to_numeric(before_ili['ili_before'], errors='coerce')
    
    # 병합 데이터에서 ILI 컬럼 추출
    merged_ili = merged_df[['year', 'week', 'age_group', 'ili']].copy()
    merged_ili['ili'] = pd.to_numeric(merged_ili['ili'], errors='coerce')
    
    # 병합하여 비교
    comparison = before_ili.merge(
        merged_ili,
        on=['year', 'week', 'age_group'],
        how='inner',
        suffixes=('_before', '_merged')
    )
    
    print(f"\n   매칭된 레코드 수: {len(comparison):,}")
    
    # 차이 계산
    comparison['diff'] = abs(comparison['ili_before'] - comparison['ili'])
    
    # 정확히 일치하는 레코드
    exact_match = (comparison['diff'] < 0.001) | (comparison['ili_before'].isna() & comparison['ili'].isna())
    exact_match_count = exact_match.sum()
    
    print(f"   정확히 일치: {exact_match_count:,} ({exact_match_count/len(comparison)*100:.1f}%)")
    
    # 차이가 있는 레코드 샘플
    mismatches = comparison[~exact_match].head(sample_size)
    if len(mismatches) > 0:
        print(f"\n   차이가 있는 레코드 샘플 (처음 {min(len(mismatches), sample_size)}개):")
        print(mismatches[['year', 'week', 'age_group', 'ili_before', 'ili', 'diff']].to_string(index=False))
    else:
        print("\n   ✅ 모든 ILI 값이 정확히 일치합니다!")


def compare_column_coverage(before_data, merged_df):
    """
    컬럼별 데이터 커버리지 비교
    """
    print("\n" + "="*70)
    print("📋 컬럼별 데이터 커버리지")
    print("="*70)
    
    columns_to_check = ['ili', 'detection_rate', 'hospitalization', 'vaccine_rate', 'emergency_patients']
    
    print(f"\n{'컬럼':<25} {'총 행':<12} {'값 있음':<12} {'비율':<10}")
    print("-" * 60)
    
    for col in columns_to_check:
        if col in merged_df.columns:
            total = len(merged_df)
            non_null = merged_df[col].notna().sum()
            ratio = non_null / total * 100
            print(f"{col:<25} {total:<12,} {non_null:<12,} {ratio:.1f}%")


def validate_data(before_data, merged_df):
    """
    데이터 무결성 검증 및 정상/비정상 판단
    
    Returns:
        dict: 검증 결과
    """
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # 1. 연도/주차 커버리지 검증
    before_year_weeks = set()
    for dsid, df in before_data.items():
        if '연도' in df.columns and '주차' in df.columns:
            for _, row in df.iterrows():
                before_year_weeks.add((int(row['연도']), int(row['주차'])))
    
    merged_year_weeks = set(zip(merged_df['year'].astype(int), merged_df['week'].astype(int)))
    
    missing_weeks = before_year_weeks - merged_year_weeks
    if missing_weeks:
        results['errors'].append(f"원본에 있는 {len(missing_weeks)}개 (연도,주차)가 병합 데이터에 누락됨")
        results['is_valid'] = False
    
    results['stats']['year_week_coverage'] = len(before_year_weeks - missing_weeks) / len(before_year_weeks) * 100 if before_year_weeks else 0
    
    # 2. ILI 값 일치율 검증
    if 'ds_0101' in before_data:
        before_ili = before_data['ds_0101'].copy()
        before_ili = before_ili.rename(columns={
            '연도': 'year', '주차': 'week', '연령대': 'age_group', '의사환자 분율': 'ili_before'
        })
        before_ili['year'] = before_ili['year'].astype(int)
        before_ili['week'] = before_ili['week'].astype(int)
        before_ili['ili_before'] = pd.to_numeric(before_ili['ili_before'], errors='coerce')
        
        merged_ili = merged_df[['year', 'week', 'age_group', 'ili']].copy()
        merged_ili['ili'] = pd.to_numeric(merged_ili['ili'], errors='coerce')
        
        comparison = before_ili.merge(merged_ili, on=['year', 'week', 'age_group'], how='inner')
        
        if len(comparison) > 0:
            comparison['diff'] = abs(comparison['ili_before'] - comparison['ili'])
            exact_match = (comparison['diff'] < 0.001) | (comparison['ili_before'].isna() & comparison['ili'].isna())
            match_rate = exact_match.sum() / len(comparison) * 100
            
            results['stats']['ili_match_rate'] = match_rate
            results['stats']['ili_matched'] = exact_match.sum()
            results['stats']['ili_total'] = len(comparison)
            
            # 값이 다른 레코드 (NaN으로 바뀐 경우 제외하고 실제 값이 다른 경우)
            value_diff = comparison[
                ~exact_match & 
                comparison['ili_before'].notna() & 
                comparison['ili'].notna()
            ]
            
            if len(value_diff) > 0:
                results['warnings'].append(f"ILI 값이 다른 레코드 {len(value_diff)}개 발견")
            
            # NaN으로 변경된 레코드
            nan_changed = comparison[
                ~exact_match & 
                comparison['ili_before'].notna() & 
                comparison['ili'].isna()
            ]
            if len(nan_changed) > 0:
                results['warnings'].append(f"원본에는 값이 있지만 병합 데이터에서 NaN인 레코드 {len(nan_changed)}개")
            
            if match_rate < 95:
                results['errors'].append(f"ILI 값 일치율이 95% 미만 ({match_rate:.1f}%)")
                results['is_valid'] = False
    
    # 3. 필수 컬럼 데이터 존재 여부
    required_cols = ['ili', 'detection_rate', 'hospitalization']
    for col in required_cols:
        if col in merged_df.columns:
            coverage = merged_df[col].notna().sum() / len(merged_df) * 100
            results['stats'][f'{col}_coverage'] = coverage
            if coverage < 30:
                results['warnings'].append(f"{col} 컬럼 데이터 커버리지가 30% 미만 ({coverage:.1f}%)")
    
    return results


def print_final_verdict(results):
    """
    최종 검증 결과 출력
    """
    print("\n" + "="*70)
    print("🏁 최종 검증 결과")
    print("="*70)
    
    # 통계 출력
    stats = results['stats']
    print("\n📊 검증 통계:")
    if 'year_week_coverage' in stats:
        print(f"   • 연도/주차 커버리지: {stats['year_week_coverage']:.1f}%")
    if 'ili_match_rate' in stats:
        print(f"   • ILI 값 일치율: {stats['ili_match_rate']:.1f}% ({stats['ili_matched']:,}/{stats['ili_total']:,})")
    for col in ['ili', 'detection_rate', 'hospitalization', 'vaccine_rate', 'emergency_patients']:
        key = f'{col}_coverage'
        if key in stats:
            print(f"   • {col} 커버리지: {stats[key]:.1f}%")
    
    # 경고 출력
    if results['warnings']:
        print(f"\n⚠️  경고 ({len(results['warnings'])}건):")
        for warning in results['warnings']:
            print(f"   • {warning}")
    
    # 오류 출력
    if results['errors']:
        print(f"\n❌ 오류 ({len(results['errors'])}건):")
        for error in results['errors']:
            print(f"   • {error}")
    
    # 최종 판정
    print("\n" + "-"*70)
    if results['is_valid']:
        if results['warnings']:
            print("📋 판정: ✅ 정상 (경고 있음)")
            print("   데이터 병합이 정상적으로 완료되었으나, 일부 경고 사항이 있습니다.")
        else:
            print("📋 판정: ✅ 정상")
            print("   데이터 병합이 완벽하게 완료되었습니다.")
    else:
        print("📋 판정: ❌ 비정상")
        print("   데이터 병합에 문제가 발견되었습니다. 오류 내용을 확인하세요.")
    print("-"*70)
    
    return results['is_valid']


def main():
    print("="*70)
    print("📊 data/before vs merged_influenza_data.csv 비교")
    print("="*70)
    
    # 데이터 로드
    before_data = load_before_data('data/before')
    merged_df = load_merged_data('merged_influenza_data.csv')
    
    if not before_data or merged_df is None:
        print("\n❌ 데이터 로드 실패")
        print("\n" + "-"*70)
        print("📋 판정: ❌ 비정상")
        print("   데이터 파일을 찾을 수 없습니다.")
        print("-"*70)
        return False
    
    # 비교 수행
    compare_statistics(before_data, merged_df)
    compare_year_week_coverage(before_data, merged_df)
    compare_values_sample(before_data, merged_df)
    compare_column_coverage(before_data, merged_df)
    
    # 검증 및 최종 판정
    results = validate_data(before_data, merged_df)
    is_valid = print_final_verdict(results)
    
    print("\n" + "="*70)
    print("✅ 비교 완료!")
    print("="*70)
    
    return is_valid


if __name__ == "__main__":
    import sys
    is_valid = main()
    sys.exit(0 if is_valid else 1)
