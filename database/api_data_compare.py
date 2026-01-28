#!/usr/bin/env python3
"""
API 데이터 vs merged_influenza_data.csv 비교 스크립트

GFID API에서 직접 데이터를 가져와서 로컬 merged_influenza_data.csv와 비교합니다.

비교 항목:
1. API에서 가져온 데이터 통계
2. 병합 데이터 통계
3. 데이터 커버리지 비교
4. 값 일치 여부 확인
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pathlib import Path

# api_client 임포트
try:
    from api_client import (
        get_recent_etl_data,
        get_etl_data_by_date_range,
        is_auth_configured,
        INFLUENZA_DATASETS,
    )
except ImportError:
    from database.api_client import (
        get_recent_etl_data,
        get_etl_data_by_date_range,
        is_auth_configured,
        INFLUENZA_DATASETS,
    )


# 데이터셋 ID → 컬럼 매핑
DATASET_COLUMN_MAPPING = {
    'ds_0101': {
        'name': 'ILI (의사환자 분율)',
        'value_col': '의사환자 분율',
        'merged_col': 'ili',
        'group_col': '연령대'
    },
    'ds_0103': {
        'name': 'SARI (중증급성호흡기감염증)',
        'value_col': '입원환자 수',
        'merged_col': 'hospitalization',
        'group_col': '연령대'
    },
    'ds_0104': {
        'name': 'ARI (급성호흡기감염증)',
        'value_col': '입원환자 수',
        'merged_col': 'hospitalization',
        'group_col': '연령대'
    },
    'ds_0105': {
        'name': 'I-RISS (검사기관 검출률)',
        'value_col': '인플루엔자 검출률',
        'merged_col': 'detection_rate',
        'group_col': '아형'
    },
    'ds_0106': {
        'name': 'K-RISS (의원급 검출률)',
        'value_col': '인플루엔자 검출률',
        'merged_col': 'detection_rate',
        'group_col': '연령대'
    },
    'ds_0107': {
        'name': '호흡기병원체 검출현황',
        'value_col': '검출률',
        'merged_col': 'detection_rate',
        'group_col': '아형'
    },
    'ds_0108': {
        'name': '인플루엔자 표본감시',
        'value_col': '검출률',
        'merged_col': 'detection_rate',
        'group_col': '연령대'
    },
    'ds_0109': {
        'name': 'NEDIS (응급실 환자)',
        'value_col': '응급실 인플루엔자 환자',  # API에서 반환하는 실제 컬럼명
        'value_col_alt': ['환자 수', '인플루엔자 환자', '응급실 환자'],  # 대체 컬럼명
        'merged_col': 'emergency_patients',
        'group_col': '연령대'
    },
    'ds_0110': {
        'name': '예방접종률',
        'value_col': '예방접종률',  # API에서 반환하는 실제 컬럼명
        'value_col_alt': ['접종률', '접종율'],  # 대체 컬럼명
        'merged_col': 'vaccine_rate',
        'group_col': '연령대'
    },
}


def flatten_parsed_data(data_list):
    """
    API 응답의 parsedData 필드를 플래튼
    """
    flattened = []
    
    for item in data_list:
        if isinstance(item, dict):
            if 'parsedData' in item and isinstance(item['parsedData'], dict):
                flat_item = item['parsedData'].copy()
                if 'collectedAt' in item:
                    flat_item['collectedAt'] = item['collectedAt']
                flattened.append(flat_item)
            elif 'parsedData' in item and isinstance(item['parsedData'], str):
                # JSON 문자열인 경우 파싱
                import json
                try:
                    parsed = json.loads(item['parsedData'])
                    if isinstance(parsed, list):
                        for p in parsed:
                            if isinstance(p, dict):
                                if 'collectedAt' in item:
                                    p['collectedAt'] = item['collectedAt']
                                flattened.append(p)
                    elif isinstance(parsed, dict):
                        if 'collectedAt' in item:
                            parsed['collectedAt'] = item['collectedAt']
                        flattened.append(parsed)
                except:
                    pass
            else:
                flattened.append(item)
        else:
            flattened.append(item)
    
    return flattened


def fetch_api_data(dsid, cnt=50):
    """
    API에서 특정 데이터셋 가져오기
    (origin별로 실제 데이터 조회)
    
    Returns:
        DataFrame 또는 None
    """
    try:
        from api_client import get_etl_data_by_origin
    except ImportError:
        from database.api_client import get_etl_data_by_origin
    
    import json
    
    mapping = DATASET_COLUMN_MAPPING.get(dsid, {})
    name = mapping.get('name', dsid)
    
    print(f"   📡 [{dsid}] {name} 데이터 조회 중...")
    
    try:
        # 1단계: 메타데이터로 origin 목록 가져오기
        meta_data = get_recent_etl_data(dsid, cnt)
        
        if not meta_data:
            print(f"      ⚠️  메타데이터 없음")
            return None
        
        # 2단계: unique origin 추출
        origins = set()
        for item in meta_data:
            if isinstance(item, dict) and 'origin' in item:
                origins.add(item['origin'])
        
        if not origins:
            print(f"      ⚠️  origin 없음")
            return None
        
        print(f"      📋 {len(origins)}개 origin 발견, 데이터 조회 중...")
        
        # 3단계: 각 origin에서 실제 데이터 가져오기 (최대 10개)
        all_data = []
        max_origins = min(len(origins), 10)
        
        for i, origin in enumerate(list(origins)[:max_origins]):
            try:
                origin_data = get_etl_data_by_origin(dsid, origin)
                
                if origin_data:
                    # parsedData 추출
                    if isinstance(origin_data, list):
                        for item in origin_data:
                            if isinstance(item, dict) and 'parsedData' in item:
                                parsed = item['parsedData']
                                if isinstance(parsed, str):
                                    try:
                                        parsed = json.loads(parsed)
                                    except:
                                        continue
                                if isinstance(parsed, list):
                                    all_data.extend(parsed)
                                elif isinstance(parsed, dict):
                                    all_data.append(parsed)
                    elif isinstance(origin_data, dict) and 'parsedData' in origin_data:
                        parsed = origin_data['parsedData']
                        if isinstance(parsed, str):
                            try:
                                parsed = json.loads(parsed)
                            except:
                                continue
                        if isinstance(parsed, list):
                            all_data.extend(parsed)
                        elif isinstance(parsed, dict):
                            all_data.append(parsed)
            except Exception as e:
                pass
        
        if not all_data:
            print(f"      ⚠️  parsedData 없음")
            return None
        
        df = pd.DataFrame(all_data)
        
        # BOM 문자 제거 및 컬럼명 정규화
        df.columns = [col.replace('\ufeff', '').strip() for col in df.columns]
        
        # 연도/주차 컬럼 확인 (다양한 형식 지원)
        year_col = None
        week_col = None
        
        # 연도 컬럼 찾기
        for col in df.columns:
            if col in ['연도', 'year', '년도']:
                year_col = col
                break
        
        # 주차 컬럼 찾기
        for col in df.columns:
            if col in ['주차', 'week', '주']:
                week_col = col
                break
        
        # 수집 기간에서 연도/주차 추출 시도
        if (year_col is None or week_col is None) and '수집 기간' in df.columns:
            # "2025년 5주" 형식에서 추출
            import re
            def extract_year_week(val):
                if pd.isna(val):
                    return None, None
                match = re.match(r'(\d{4})년\s*(\d+)주', str(val))
                if match:
                    return int(match.group(1)), int(match.group(2))
                return None, None
            
            extracted = df['수집 기간'].apply(extract_year_week)
            df['year'] = extracted.apply(lambda x: x[0])
            df['week'] = extracted.apply(lambda x: x[1])
            df = df.dropna(subset=['year', 'week'])
            if len(df) > 0:
                df['year'] = df['year'].astype(int)
                df['week'] = df['week'].astype(int)
                print(f"      ✅ {len(df)}건 조회 완료 (수집 기간에서 추출)")
                return df
        
        if year_col and week_col:
            df['year'] = pd.to_numeric(df[year_col], errors='coerce').astype('Int64')
            df['week'] = pd.to_numeric(df[week_col], errors='coerce').astype('Int64')
            df = df.dropna(subset=['year', 'week'])
            if len(df) > 0:
                df['year'] = df['year'].astype(int)
                df['week'] = df['week'].astype(int)
                print(f"      ✅ {len(df)}건 조회 완료")
                return df
        
        print(f"      ⚠️  연도/주차 컬럼 없음: {list(df.columns)[:8]}")
        return None
    
    except Exception as e:
        print(f"      ❌ 조회 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_merged_data(merged_path='merged_influenza_data.csv'):
    """
    merged_influenza_data.csv 로드
    """
    if not os.path.exists(merged_path):
        print(f"❌ {merged_path} 파일이 존재하지 않습니다.")
        return None
    
    df = pd.read_csv(merged_path)
    print(f"\n📊 병합 데이터 로드: {len(df)}행, {len(df.columns)}개 컬럼")
    return df


def compare_dataset(api_df, merged_df, dsid):
    """
    특정 데이터셋의 API 데이터와 병합 데이터 비교
    """
    mapping = DATASET_COLUMN_MAPPING.get(dsid, {})
    value_col = mapping.get('value_col')
    value_col_alt = mapping.get('value_col_alt', [])  # 대체 컬럼명 리스트
    merged_col = mapping.get('merged_col')
    group_col = mapping.get('group_col')
    name = mapping.get('name', dsid)
    
    if not merged_col:
        return None
    
    result = {
        'dsid': dsid,
        'name': name,
        'api_rows': len(api_df),
        'matched': 0,
        'exact_match': 0,
        'mismatch': 0,
        'match_rate': 0.0
    }
    
    # API 데이터에서 value_col 찾기 (기본 컬럼명 + 대체 컬럼명)
    actual_value_col = None
    candidates = [value_col] + value_col_alt if value_col else value_col_alt
    
    for col in candidates:
        if col and col in api_df.columns:
            actual_value_col = col
            break
    
    if not actual_value_col:
        # 컬럼명을 찾지 못한 경우, 유사한 컬럼명 출력
        print(f"      ⚠️  [{dsid}] 값 컬럼을 찾을 수 없음. API 컬럼: {list(api_df.columns)}")
        return result
    
    # group_col 찾기
    if group_col not in api_df.columns:
        print(f"      ⚠️  [{dsid}] 그룹 컬럼 '{group_col}'이 없음. API 컬럼: {list(api_df.columns)}")
        return result
    
    api_compare = api_df[['year', 'week', group_col, actual_value_col]].copy()
    api_compare = api_compare.rename(columns={
        group_col: 'age_group',
        actual_value_col: 'api_value'
    })
    api_compare['api_value'] = pd.to_numeric(api_compare['api_value'], errors='coerce')
    
    # 병합 데이터 준비
    merged_compare = merged_df[['year', 'week', 'age_group', merged_col]].copy()
    merged_compare = merged_compare.rename(columns={merged_col: 'merged_value'})
    merged_compare['merged_value'] = pd.to_numeric(merged_compare['merged_value'], errors='coerce')
    
    # 조인
    comparison = api_compare.merge(
        merged_compare,
        on=['year', 'week', 'age_group'],
        how='inner'
    )
    
    result['matched'] = len(comparison)
    
    if len(comparison) > 0:
        # 정확 일치 확인 (차이 0.01 이하)
        comparison['diff'] = abs(comparison['api_value'] - comparison['merged_value'])
        exact_match = (comparison['diff'] < 0.01) | (comparison['api_value'].isna() & comparison['merged_value'].isna())
        result['exact_match'] = exact_match.sum()
        result['mismatch'] = len(comparison) - result['exact_match']
        result['match_rate'] = result['exact_match'] / len(comparison) * 100
    
    return result


def main():
    print("="*70)
    print("📊 API 데이터 vs merged_influenza_data.csv 비교")
    print("="*70)
    
    # 1. 인증 확인
    print("\n[1] API 인증 확인")
    if not is_auth_configured():
        print("   ❌ GFID API 인증이 설정되지 않았습니다.")
        print("   .env 파일에 GFID_CLIENT_ID, GFID_CLIENT_SECRET를 설정하세요.")
        return
    print("   ✅ API 인증 설정 완료")
    
    # 2. 병합 데이터 로드
    print("\n[2] 병합 데이터 로드")
    merged_df = load_merged_data('merged_influenza_data.csv')
    if merged_df is None:
        return
    
    print(f"   연도 범위: {merged_df['year'].min()} ~ {merged_df['year'].max()}")
    print(f"   주차 범위: {merged_df['week'].min()} ~ {merged_df['week'].max()}")
    
    # 3. API 데이터 조회 및 비교
    print("\n[3] API 데이터 조회 및 비교")
    
    results = []
    api_data_dict = {}
    
    for dsid in ['ds_0101', 'ds_0105', 'ds_0106', 'ds_0109', 'ds_0110']:
        api_df = fetch_api_data(dsid, cnt=200)
        
        if api_df is not None and len(api_df) > 0:
            api_data_dict[dsid] = api_df
            result = compare_dataset(api_df, merged_df, dsid)
            if result:
                results.append(result)
    
    # 4. 결과 출력
    print("\n" + "="*70)
    print("📋 비교 결과 요약")
    print("="*70)
    
    if not results:
        print("\n⚠️  비교할 데이터가 없습니다.")
        print("   API가 parsedData를 반환하지 않아 메타데이터만 조회되었을 수 있습니다.")
        return
    
    print(f"\n{'데이터셋':<12} {'이름':<25} {'API행':<8} {'매칭':<8} {'일치':<8} {'불일치':<8} {'일치율':<10}")
    print("-" * 85)
    
    total_matched = 0
    total_exact = 0
    
    for r in results:
        print(f"{r['dsid']:<12} {r['name']:<25} {r['api_rows']:<8} {r['matched']:<8} "
              f"{r['exact_match']:<8} {r['mismatch']:<8} {r['match_rate']:.1f}%")
        total_matched += r['matched']
        total_exact += r['exact_match']
    
    print("-" * 85)
    if total_matched > 0:
        overall_rate = total_exact / total_matched * 100
        print(f"{'합계':<12} {'':<25} {'':<8} {total_matched:<8} {total_exact:<8} "
              f"{total_matched - total_exact:<8} {overall_rate:.1f}%")
    
    # 5. 상세 분석 (ds_0101만)
    if 'ds_0101' in api_data_dict:
        print("\n" + "="*70)
        print("🔍 상세 분석: ds_0101 (ILI 데이터)")
        print("="*70)
        
        api_df = api_data_dict['ds_0101']
        
        # 연도/주차 커버리지
        api_year_weeks = set(zip(api_df['year'], api_df['week']))
        merged_year_weeks = set(zip(merged_df['year'], merged_df['week']))
        
        common = api_year_weeks & merged_year_weeks
        api_only = api_year_weeks - merged_year_weeks
        merged_only = merged_year_weeks - api_year_weeks
        
        print(f"\n[연도/주차 커버리지]")
        print(f"   API 고유 조합: {len(api_year_weeks)}개")
        print(f"   병합 고유 조합: {len(merged_year_weeks)}개")
        print(f"   공통: {len(common)}개")
        
        if api_only:
            print(f"\n   API에만 있는 조합 (처음 5개):")
            for yw in sorted(api_only)[:5]:
                print(f"      - {yw[0]}년 {yw[1]}주")
        
        if merged_only:
            print(f"\n   병합에만 있는 조합: {len(merged_only)}개")
    
    print("\n" + "="*70)
    print("✅ 비교 완료!")
    print("="*70)
    
    # 결론
    if results:
        avg_rate = sum(r['match_rate'] for r in results) / len(results)
        if avg_rate >= 99:
            print("\n🎉 데이터가 정확하게 적재되었습니다!")
        elif avg_rate >= 90:
            print("\n✅ 대부분의 데이터가 정확하게 적재되었습니다.")
        else:
            print("\n⚠️  일부 데이터 불일치가 있습니다. 상세 검토가 필요합니다.")


if __name__ == "__main__":
    main()
