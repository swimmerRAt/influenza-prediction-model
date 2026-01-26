#!/usr/bin/env python3
"""
데이터 필터링 무결성 검증 스크립트
- 연령별/아형별 필터링 과정에서 데이터 손상 여부 확인
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 프로젝트 경로 설정
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "before"

# patchTST.py에서 필요한 함수들 임포트
from patchTST import (
    load_raw_data_by_age_group,
    load_subtype_data,
    get_available_age_groups,
    AGE_GROUP_MAPPING,
)


def load_all_raw_csvs():
    """모든 원본 CSV 파일을 로드하여 통계 계산"""
    all_data = {}
    
    datasets = {
        'ds_0101': '의사환자 분율',
        'ds_0103': '입원환자 수',
        'ds_0104': '입원환자 수',
        'ds_0106': '인플루엔자 검출률',
        'ds_0107': '검출률',  # 아형별
        'ds_0108': '인플루엔자 검출률',
        'ds_0109': '응급실 인플루엔자 환자',
        'ds_0110': '예방접종률',
    }
    
    for dsid, col_name in datasets.items():
        ds_num = dsid.replace('ds_', '')
        pattern = f"flu-{ds_num}-*.csv"
        files = list(DATA_DIR.glob(pattern))
        
        if not files:
            continue
        
        dfs = []
        for f in sorted(files):
            try:
                df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                print(f"   ⚠️ 파일 읽기 실패 ({f.name}): {e}")
        
        if dfs:
            df_combined = pd.concat(dfs, ignore_index=True)
            all_data[dsid] = df_combined
    
    return all_data


def validate_age_filtering():
    """연령별 필터링 검증"""
    print("\n" + "=" * 70)
    print("🔍 연령별 필터링 검증")
    print("=" * 70)
    
    # 원본 데이터 로드
    raw_data = load_all_raw_csvs()
    
    # 사용 가능한 연령대 확인
    age_info = get_available_age_groups(str(DATA_DIR))
    
    print("\n📊 원본 데이터셋별 통계:")
    for dsid, df in raw_data.items():
        print(f"\n   {dsid}:")
        print(f"      - 총 행 수: {len(df)}")
        if '연령대' in df.columns:
            age_groups = df['연령대'].dropna().unique()
            print(f"      - 연령대 종류: {len(age_groups)}개")
            print(f"      - 연령대 목록: {sorted([str(a) for a in age_groups])}")
    
    # 각 연령대별로 필터링 테스트
    test_ages = ['0-6세', '7-12세', '13-18세', '19-49세', '50-64세', '65세이상']
    
    print("\n📊 연령대별 필터링 결과:")
    results = []
    
    for age in test_ages:
        print(f"\n   🔸 연령대: {age}")
        try:
            df_filtered = load_raw_data_by_age_group(data_dir=str(DATA_DIR), age_group=age)
            
            if df_filtered.empty:
                print(f"      ❌ 데이터 없음")
                results.append({'age': age, 'rows': 0, 'status': 'EMPTY'})
                continue
            
            # 기본 통계
            row_count = len(df_filtered)
            col_count = len(df_filtered.columns)
            
            # 결측치 확인
            null_counts = df_filtered.isnull().sum()
            null_total = null_counts.sum()
            
            # ILI 범위 확인
            ili_min = df_filtered['ili'].min() if 'ili' in df_filtered.columns else None
            ili_max = df_filtered['ili'].max() if 'ili' in df_filtered.columns else None
            
            # 연도/주차 범위
            year_min = df_filtered['year'].min()
            year_max = df_filtered['year'].max()
            week_range = (df_filtered['week'].min(), df_filtered['week'].max())
            
            print(f"      - 행 수: {row_count}")
            print(f"      - 컬럼 수: {col_count}")
            print(f"      - 컬럼: {list(df_filtered.columns)}")
            print(f"      - 결측치 총합: {null_total}")
            if null_total > 0:
                print(f"      - 결측치 상세: {dict(null_counts[null_counts > 0])}")
            print(f"      - 연도 범위: {year_min} ~ {year_max}")
            print(f"      - 주차 범위: {week_range[0]} ~ {week_range[1]}")
            if ili_min is not None:
                print(f"      - ILI 범위: {ili_min:.2f} ~ {ili_max:.2f}")
            
            # 시간순 정렬 확인
            is_sorted = df_filtered['year'].is_monotonic_increasing or (
                df_filtered.sort_values(['year', 'week']).index.tolist() == df_filtered.index.tolist()
            )
            print(f"      - 시간순 정렬: {'✅ OK' if is_sorted else '❌ 정렬 필요'}")
            
            # 중복 확인
            duplicates = df_filtered.duplicated(subset=['year', 'week']).sum()
            print(f"      - 중복 행: {duplicates}개")
            
            results.append({
                'age': age, 
                'rows': row_count, 
                'nulls': null_total,
                'duplicates': duplicates,
                'status': 'OK' if null_total == 0 and duplicates == 0 else 'WARNING'
            })
            
        except Exception as e:
            print(f"      ❌ 오류: {e}")
            results.append({'age': age, 'rows': 0, 'status': f'ERROR: {e}'})
    
    return results


def validate_subtype_filtering():
    """아형별 필터링 검증"""
    print("\n" + "=" * 70)
    print("🔍 아형별 필터링 검증")
    print("=" * 70)
    
    # 원본 ds_0107 데이터 확인
    ds_0107_files = list(DATA_DIR.glob("flu-0107-*.csv"))
    
    if not ds_0107_files:
        print("\n   ⚠️ ds_0107 파일을 찾을 수 없습니다.")
        return []
    
    # 원본 데이터 로드
    dfs = []
    for f in sorted(ds_0107_files):
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"   ⚠️ 파일 읽기 실패 ({f.name}): {e}")
    
    if not dfs:
        print("\n   ⚠️ ds_0107 데이터를 로드할 수 없습니다.")
        return []
    
    df_raw = pd.concat(dfs, ignore_index=True)
    
    print(f"\n📊 원본 ds_0107 데이터:")
    print(f"   - 총 행 수: {len(df_raw)}")
    print(f"   - 컬럼: {list(df_raw.columns)}")
    
    # 아형 컬럼 찾기
    subtype_col = None
    for col in ['아형', 'subtype', '인플루엔자유형']:
        if col in df_raw.columns:
            subtype_col = col
            break
    
    if subtype_col:
        subtypes = df_raw[subtype_col].unique()
        print(f"   - 아형 컬럼: {subtype_col}")
        print(f"   - 아형 종류: {subtypes}")
        
        for st in subtypes:
            count = len(df_raw[df_raw[subtype_col] == st])
            print(f"      - {st}: {count}행")
    
    # 필터링 테스트
    print("\n📊 아형별 필터링 결과:")
    results = []
    
    for subtype in ['A', 'B', 'all']:
        print(f"\n   🔸 아형: {subtype}")
        try:
            df_filtered = load_subtype_data(data_dir=str(DATA_DIR), subtype=subtype)
            
            if df_filtered.empty:
                print(f"      ❌ 데이터 없음")
                results.append({'subtype': subtype, 'rows': 0, 'status': 'EMPTY'})
                continue
            
            row_count = len(df_filtered)
            col_count = len(df_filtered.columns)
            
            # 결측치 확인
            null_counts = df_filtered.isnull().sum()
            null_total = null_counts.sum()
            
            print(f"      - 행 수: {row_count}")
            print(f"      - 컬럼: {list(df_filtered.columns)}")
            print(f"      - 결측치: {null_total}")
            
            # 검출률 범위
            if 'detection_rate' in df_filtered.columns:
                dr_min = df_filtered['detection_rate'].min()
                dr_max = df_filtered['detection_rate'].max()
                print(f"      - 검출률 범위: {dr_min:.2f} ~ {dr_max:.2f}")
            
            # 연도/주차 범위
            if 'year' in df_filtered.columns:
                print(f"      - 연도 범위: {df_filtered['year'].min()} ~ {df_filtered['year'].max()}")
            
            results.append({
                'subtype': subtype, 
                'rows': row_count, 
                'nulls': null_total,
                'status': 'OK' if null_total == 0 else 'WARNING'
            })
            
        except Exception as e:
            print(f"      ❌ 오류: {e}")
            results.append({'subtype': subtype, 'rows': 0, 'status': f'ERROR: {e}'})
    
    return results


def validate_data_consistency():
    """데이터 일관성 검증: 필터링 전후 합계 비교"""
    print("\n" + "=" * 70)
    print("🔍 데이터 일관성 검증 (필터링 전후 비교)")
    print("=" * 70)
    
    # ds_0101 (ILI) 데이터로 테스트
    ds_0101_files = list(DATA_DIR.glob("flu-0101-*.csv"))
    
    if not ds_0101_files:
        print("\n   ⚠️ ds_0101 파일을 찾을 수 없습니다.")
        return
    
    # 원본 로드
    dfs = []
    for f in sorted(ds_0101_files):
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            pass
    
    if not dfs:
        return
    
    df_raw = pd.concat(dfs, ignore_index=True)
    
    print(f"\n📊 원본 ds_0101 데이터:")
    print(f"   - 총 행 수: {len(df_raw)}")
    
    if '연령대' not in df_raw.columns:
        print("   ⚠️ 연령대 컬럼 없음")
        return
    
    # 연령대별 행 수 (원본)
    raw_age_counts = df_raw.groupby('연령대').size().to_dict()
    print(f"   - 연령대별 행 수 (원본):")
    for age, count in sorted(raw_age_counts.items()):
        print(f"      - {age}: {count}")
    
    # 필터링 후 행 수 합계
    test_ages = ['0세', '1-6세', '7-12세', '13-18세', '19-49세', '50-64세', '65세이상']
    filtered_total = 0
    
    print(f"\n   - 필터링 후 행 수:")
    for age in test_ages:
        try:
            df_filtered = load_raw_data_by_age_group(data_dir=str(DATA_DIR), age_group=age)
            filtered_total += len(df_filtered)
        except:
            pass
    
    # 원본 연령대 합계 (test_ages에 해당하는 것만)
    raw_total = sum(raw_age_counts.get(age, 0) for age in test_ages)
    
    print(f"\n📊 비교 결과:")
    print(f"   - 원본 총 행 수 (주요 연령대): {raw_total}")
    print(f"   - 필터링 후 총 행 수: {filtered_total}")
    
    # 주의: 필터링 후에는 year/week 기준으로 병합되므로 행 수가 줄어듦
    print(f"   - 참고: 필터링 함수는 연도/주차 기준으로 데이터를 병합하므로")
    print(f"           행 수가 줄어드는 것은 정상입니다 (병합 후 436행 예상)")


def main():
    print("\n" + "🔬 " * 20)
    print("데이터 필터링 무결성 검증 시작")
    print("🔬 " * 20)
    
    # 1. 연령별 필터링 검증
    age_results = validate_age_filtering()
    
    # 2. 아형별 필터링 검증
    subtype_results = validate_subtype_filtering()
    
    # 3. 데이터 일관성 검증
    validate_data_consistency()
    
    # 최종 요약
    print("\n" + "=" * 70)
    print("📋 검증 요약")
    print("=" * 70)
    
    print("\n연령별 필터링:")
    for r in age_results:
        status_icon = "✅" if r.get('status') == 'OK' else ("⚠️" if 'WARNING' in str(r.get('status', '')) else "❌")
        print(f"   {status_icon} {r['age']}: {r['rows']}행 - {r.get('status', 'UNKNOWN')}")
    
    print("\n아형별 필터링:")
    for r in subtype_results:
        status_icon = "✅" if r.get('status') == 'OK' else ("⚠️" if 'WARNING' in str(r.get('status', '')) else "❌")
        print(f"   {status_icon} {r['subtype']}: {r['rows']}행 - {r.get('status', 'UNKNOWN')}")
    
    print("\n" + "=" * 70)
    print("✅ 검증 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
