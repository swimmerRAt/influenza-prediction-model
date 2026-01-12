"""
data 폴더의 모든 JSON 파일을 연도/주차별로 병합하는 스크립트
중복 데이터 제거 및 평균 처리 포함
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import re

def parse_week_from_date(date_str):
    """날짜 문자열에서 ISO 주차 추출"""
    try:
        dt = pd.to_datetime(date_str)
        # ISO 8601 주차 계산 (년도, 주차)
        iso_calendar = dt.isocalendar()
        return iso_calendar.year, iso_calendar.week
    except:
        return None, None

def process_ds_0101_to_0110(data):
    """
    ds_0101~ds_0110: 연령대별 의사환자 분율 데이터
    컬럼: 연도, 주차, 연령대, 의사환자 분율
    """
    records = []
    for item in data:
        parsed = json.loads(item.get('parsedData', '[]'))
        for record in parsed:
            year = record.get('﻿연도', record.get('연도'))
            week = record.get('주차')
            age_group = record.get('연령대')
            rate = record.get('의사환자 분율')
            
            if year and week:
                try:
                    year_int = int(year)
                    week_int = int(week)
                    # week 값이 52를 초과하면 스킵
                    if week_int > 52:
                        continue
                    records.append({
                        'year': year_int,
                        'week': week_int,
                        'dataset_id': item['dsId'],
                        'age_group': age_group,
                        'patient_rate': float(rate) if rate else 0.0,
                        'collected_at': item.get('collectedAt')
                    })
                except (ValueError, TypeError):
                    continue
    return records

def process_ds_0701(data):
    """
    ds_0701: 검색 트렌드 데이터 (Google Trends)
    컬럼: 독감, 감기, 독감 증상 등의 검색량 (주별)
    
    parsedData는 배열 형태이며, 각 요소가 한 주의 데이터를 나타냄
    수집 날짜를 기준으로 역산하여 각 레코드의 year/week를 계산
    """
    records = []
    for item in data:
        parsed = json.loads(item.get('parsedData', '[]'))
        # 수집 날짜로부터 연도/주차 추출
        collected_at = item.get('collectedAt', '')
        if not collected_at or not parsed:
            continue
            
        base_year, base_week = parse_week_from_date(collected_at)
        if not base_year or not base_week:
            continue
        
        # parsedData의 각 요소는 시간 순서대로 정렬되어 있음
        # 마지막 요소가 가장 최근(수집 날짜 기준), 첫 요소가 가장 과거
        num_records = len(parsed)
        
        for idx, record in enumerate(parsed):
            # 역순으로 주차 계산 (마지막 레코드가 수집 날짜 주차)
            week_offset = num_records - idx - 1
            target_week = base_week - week_offset
            target_year = base_year
            
            # 주차가 1보다 작으면 전년도로 이동
            while target_week < 1:
                target_week += 52
                target_year -= 1
            
            # 주차가 52보다 크면 스킵 (데이터 오류)
            if target_week > 52:
                continue
            
            records.append({
                'year': target_year,
                'week': target_week,
                'dataset_id': item['dsId'],
                'search_flu': int(record.get('﻿독감', record.get('독감', 0)) or 0),
                'search_cold': int(record.get('감기', 0) or 0),
                'search_flu_symptom': int(record.get('독감 증상', 0) or 0),
                'search_body_ache': int(record.get('몸살', 0) or 0),
                'search_flu_a': int(record.get('A형 독감', 0) or 0),
                'search_cough': int(record.get('기침', 0) or 0),
                'search_runny_nose': int(record.get('콧물', 0) or 0),
                'search_flu_vaccine': int(record.get('독감 예방접종', 0) or 0),
                'search_tamiflu': int(record.get('타미플루', 0) or 0),
                'collected_at': collected_at
            })
    return records

def process_ds_0801(data):
    """
    ds_0801: 네이버 검색 트렌드 데이터
    """
    return process_ds_0701(data)

def process_ds_0901(data):
    """
    ds_0901: 트위터 트렌드 데이터
    컬럼: day, keyword, count
    """
    records = []
    for item in data:
        parsed = json.loads(item.get('parsedData', '[]'))
        for record in parsed:
            day = record.get('﻿day', record.get('day'))
            keyword = record.get('keyword')
            count = record.get('count')
            
            if day:
                year, week = parse_week_from_date(day)
                if year and week and week <= 52:
                    records.append({
                        'year': year,
                        'week': week,
                        'dataset_id': item['dsId'],
                        'date': day,
                        'keyword': keyword,
                        'tweet_count': int(count) if count else 0,
                        'collected_at': item.get('collectedAt')
                    })
    return records

def load_all_data_files(data_dir='data'):
    """data 폴더의 모든 JSON 파일 로드"""
    data_path = Path(data_dir)
    all_data = {}
    
    # JSON 파일 목록
    json_files = list(data_path.glob('ds_*_page_1.json'))
    
    print(f"발견된 파일 수: {len(json_files)}")
    
    for file_path in sorted(json_files):
        print(f"로딩 중: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 파일명에서 데이터셋 ID 추출
            dataset_id = file_path.stem.replace('_page_1', '')
            all_data[dataset_id] = data
            print(f"  - {dataset_id}: {len(data)}개 레코드")
    
    return all_data

def merge_all_datasets():
    """모든 데이터셋을 연도/주차별로 병합"""
    print("=" * 60)
    print("데이터 병합 시작")
    print("=" * 60)
    
    # 모든 데이터 로드
    all_data = load_all_data_files()
    
    # 각 데이터셋 타입별로 처리
    patient_data = []  # ds_0101~ds_0110
    search_google = []  # ds_0701
    search_naver = []  # ds_0801
    twitter_data = []  # ds_0901
    
    for dataset_id, data in all_data.items():
        print(f"\n처리 중: {dataset_id}")
        
        if dataset_id.startswith('ds_01'):  # ds_0101 ~ ds_0110
            records = process_ds_0101_to_0110(data)
            patient_data.extend(records)
            print(f"  - 환자 데이터: {len(records)}개 레코드 추출")
            
        elif dataset_id == 'ds_0701':
            records = process_ds_0701(data)
            search_google.extend(records)
            print(f"  - Google 검색 데이터: {len(records)}개 레코드 추출")
            
        elif dataset_id == 'ds_0801':
            records = process_ds_0801(data)
            search_naver.extend(records)
            print(f"  - Naver 검색 데이터: {len(records)}개 레코드 추출")
            
        elif dataset_id == 'ds_0901':
            records = process_ds_0901(data)
            twitter_data.extend(records)
            print(f"  - Twitter 데이터: {len(records)}개 레코드 추출")
    
    # DataFrame 생성
    print("\n" + "=" * 60)
    print("DataFrame 생성")
    print("=" * 60)
    
    df_patient = pd.DataFrame(patient_data)
    df_search_google = pd.DataFrame(search_google)
    df_search_naver = pd.DataFrame(search_naver)
    df_twitter = pd.DataFrame(twitter_data)
    
    print(f"환자 데이터: {df_patient.shape}")
    print(f"Google 검색: {df_search_google.shape}")
    print(f"Naver 검색: {df_search_naver.shape}")
    print(f"Twitter: {df_twitter.shape}")
    
    # 연도/주차별로 집계
    print("\n" + "=" * 60)
    print("연도/주차별 집계 (중복 제거 및 평균)")
    print("=" * 60)
    
    # 1. 환자 데이터: 연령대별로 pivot하여 컬럼으로 변환
    if not df_patient.empty:
        # 중복 제거: 동일한 (year, week, age_group)의 평균을 먼저 계산
        df_patient_grouped = df_patient.groupby(['year', 'week', 'age_group']).agg({
            'patient_rate': 'mean'
        }).reset_index()
        
        patient_pivot = df_patient_grouped.pivot_table(
            index=['year', 'week'],
            columns='age_group',
            values='patient_rate',
            aggfunc='mean'
        ).reset_index()
        # 컬럼명 정리
        patient_pivot.columns = ['year', 'week'] + [f'patient_rate_{col}' for col in patient_pivot.columns[2:]]
        
        # week 값 확인
        max_week = patient_pivot['week'].max()
        print(f"환자 데이터 (pivot): {patient_pivot.shape}")
        print(f"  - 최소 week: {patient_pivot['week'].min()}, 최대 week: {max_week}")
        if max_week > 52:
            print(f"  경고: week 값이 52를 초과합니다!")
    else:
        patient_pivot = pd.DataFrame()
    
    # 2. Google 검색: 연도/주차별 평균 (중복 제거)
    if not df_search_google.empty:
        search_google_agg = df_search_google.groupby(['year', 'week']).agg({
            'search_flu': 'mean',
            'search_cold': 'mean',
            'search_flu_symptom': 'mean',
            'search_body_ache': 'mean',
            'search_flu_a': 'mean',
            'search_cough': 'mean',
            'search_runny_nose': 'mean',
            'search_flu_vaccine': 'mean',
            'search_tamiflu': 'mean'
        }).reset_index()
        # 컬럼명에 접두사 추가
        rename_dict = {col: f'google_{col}' for col in search_google_agg.columns if col not in ['year', 'week']}
        search_google_agg.rename(columns=rename_dict, inplace=True)
        
        # week 값 확인
        max_week = search_google_agg['week'].max()
        print(f"Google 검색 (집계): {search_google_agg.shape}")
        print(f"  - 최소 week: {search_google_agg['week'].min()}, 최대 week: {max_week}")
        if max_week > 52:
            print(f"  경고: week 값이 52를 초과합니다!")
    else:
        search_google_agg = pd.DataFrame()
    
    # 3. Naver 검색: 연도/주차별 평균 (중복 제거)
    if not df_search_naver.empty:
        search_naver_agg = df_search_naver.groupby(['year', 'week']).agg({
            'search_flu': 'mean',
            'search_cold': 'mean',
            'search_flu_symptom': 'mean',
            'search_body_ache': 'mean',
            'search_flu_a': 'mean',
            'search_cough': 'mean',
            'search_runny_nose': 'mean',
            'search_flu_vaccine': 'mean',
            'search_tamiflu': 'mean'
        }).reset_index()
        # 컬럼명에 접두사 추가
        rename_dict = {col: f'naver_{col}' for col in search_naver_agg.columns if col not in ['year', 'week']}
        search_naver_agg.rename(columns=rename_dict, inplace=True)
        
        # week 값 확인
        max_week = search_naver_agg['week'].max()
        print(f"Naver 검색 (집계): {search_naver_agg.shape}")
        print(f"  - 최소 week: {search_naver_agg['week'].min()}, 최대 week: {max_week}")
        if max_week > 52:
            print(f"  경고: week 값이 52를 초과합니다!")
    else:
        search_naver_agg = pd.DataFrame()
    
    # 4. Twitter: 키워드별로 pivot하여 연도/주차별 집계 (중복 제거)
    if not df_twitter.empty:
        # 중복 제거: 동일한 (year, week, keyword)의 합계를 먼저 계산
        df_twitter_grouped = df_twitter.groupby(['year', 'week', 'keyword']).agg({
            'tweet_count': 'sum'
        }).reset_index()
        
        twitter_pivot = df_twitter_grouped.pivot_table(
            index=['year', 'week'],
            columns='keyword',
            values='tweet_count',
            aggfunc='sum'
        ).reset_index()
        # 컬럼명 정리 (한글 키워드를 영어로 변환)
        keyword_map = {
            '독감': 'twitter_flu',
            '인플루엔자': 'twitter_influenza',
            '인플루엔자 독감': 'twitter_influenza_flu',
            '감기': 'twitter_cold',
            '근육통': 'twitter_muscle_pain'
        }
        twitter_pivot.columns = ['year', 'week'] + [keyword_map.get(col, f'twitter_{col}') for col in twitter_pivot.columns[2:]]
        
        # week 값 확인
        max_week = twitter_pivot['week'].max()
        print(f"Twitter (pivot): {twitter_pivot.shape}")
        print(f"  - 최소 week: {twitter_pivot['week'].min()}, 최대 week: {max_week}")
        if max_week > 52:
            print(f"  경고: week 값이 52를 초과합니다!")
    else:
        twitter_pivot = pd.DataFrame()
    
    # 모든 데이터 병합 (year, week 기준)
    print("\n" + "=" * 60)
    print("최종 병합")
    print("=" * 60)
    
    # 기준 DataFrame 생성 (모든 year, week 조합)
    all_dfs = [df for df in [patient_pivot, search_google_agg, search_naver_agg, twitter_pivot] if not df.empty]
    
    if not all_dfs:
        print("병합할 데이터가 없습니다!")
        return pd.DataFrame()
    
    # 첫 번째 DataFrame을 기준으로 시작
    merged_df = all_dfs[0]
    
    # 나머지 DataFrame들을 순차적으로 병합
    for df in all_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=['year', 'week'], how='outer')
    
    # year, week 기준으로 정렬
    merged_df.sort_values(['year', 'week'], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    
    # week 값 최종 검증
    max_week = merged_df['week'].max()
    if max_week > 52:
        print(f"\n경고: week 최대값이 {max_week}입니다. 52를 초과하는 week 값이 있습니다!")
        print(f"   52를 초과하는 데이터:")
        invalid_weeks = merged_df[merged_df['week'] > 52][['year', 'week']].drop_duplicates()
        print(invalid_weeks)
        
        # 52주를 초과하는 데이터 자동 제거
        print(f"\n자동으로 52주 초과 데이터 제거 중...")
        merged_df = merged_df[merged_df['week'] <= 52].copy()
        merged_df.reset_index(drop=True, inplace=True)
        print(f"✅ 52주 초과 데이터 제거 완료. 남은 데이터: {merged_df.shape}")
    
    # 결측치 처리: 0으로 채우지 않고 NaN으로 유지 (또는 forward fill)
    # 모델 학습 시 결측치 처리는 별도로 진행
    # merged_df.fillna(0, inplace=True)  # 제거: 결손값을 0으로 채우지 않음
    
    # 결손값 통계 출력
    print("\n" + "=" * 60)
    print("결손값 통계")
    print("=" * 60)
    missing_counts = merged_df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    if len(missing_counts) > 0:
        print(f"결손값이 있는 컬럼 수: {len(missing_counts)}")
        print(f"주요 결손값:")
        print(missing_counts.head(10))
        print(f"\n전체 데이터 대비 결손 비율: {merged_df.isnull().sum().sum() / (merged_df.shape[0] * merged_df.shape[1]):.2%}")
    else:
        print("결손값이 없습니다!")
    
    print(f"\n최종 병합 데이터: {merged_df.shape}")
    print(f"컬럼: {list(merged_df.columns)}")
    print(f"연도 범위: {merged_df['year'].min()} ~ {merged_df['year'].max()}")
    print(f"주차 범위: {merged_df['week'].min()} ~ {merged_df['week'].max()}")
    
    # 샘플 데이터 출력
    print("\n" + "=" * 60)
    print("샘플 데이터 (처음 5개 행)")
    print("=" * 60)
    print(merged_df.head())
    
    return merged_df

def save_merged_data(df, output_file='merged_influenza_data.csv'):
    """병합된 데이터를 CSV로 저장"""
    print("\n" + "=" * 60)
    print("데이터 저장")
    print("=" * 60)
    
    output_path = Path(output_file)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 저장 완료: {output_path}")
    print(f"   파일 크기: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"   행 수: {len(df)}")
    print(f"   열 수: {len(df.columns)}")

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 60)
    print("인플루엔자 데이터셋 병합 스크립트")
    print("=" * 60)
    
    # 데이터 병합
    merged_df = merge_all_datasets()
    
    if merged_df.empty:
        print("\n❌ 병합할 데이터가 없습니다!")
        return
    
    # 저장
    save_merged_data(merged_df)
    
    # 추가 통계
    print("\n" + "=" * 60)
    print("데이터 통계")
    print("=" * 60)
    print(merged_df.describe())
    
    print("\n✅ 모든 작업 완료!")

if __name__ == '__main__':
    main()
