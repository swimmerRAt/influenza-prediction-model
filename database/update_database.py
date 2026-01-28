
"""
PostgreSQL 데이터베이스 통합 업데이트

이 스크립트는 다음 작업을 수행합니다:
1. API에서 인플루엔자 데이터 가져오기 (ILI, 백신률, 입원환자 등)
2. data/before 폴더의 과거 데이터 로딩
3. 모든 인플루엔자 데이터 병합 및 PostgreSQL influenza DB에 저장
4. API에서 트렌드 데이터 가져오기 (Google, Naver, Twitter)
5. 트렌드 데이터 병합 및 PostgreSQL trends DB에 저장

API 클라이언트: src_jaehong/api/ 패턴을 참고하여 구현 (api_client.py)
"""

try:
    # 모듈로서 import 될 때
    from .db_utils import merge_and_update_database, TimeSeriesDB
    from .api_client import (
        get_recent_etl_data,
        get_etl_data_by_date_range,
        get_etl_data_by_season,
        fetch_trend_data_from_api,
        fetch_all_influenza_data,
        INFLUENZA_DATASETS,
        is_auth_configured,
    )
except ImportError:
    # 직접 실행될 때
    from db_utils import merge_and_update_database, TimeSeriesDB
    from api_client import (
        get_recent_etl_data,
        get_etl_data_by_date_range,
        get_etl_data_by_season,
        fetch_trend_data_from_api,
        fetch_all_influenza_data,
        INFLUENZA_DATASETS,
        is_auth_configured,
    )

import os
import pandas as pd
import requests

from dotenv import load_dotenv
import warnings
from datetime import datetime

load_dotenv()
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# =========================
# 트렌드 데이터 업데이트 함수들
# =========================

def parse_date_to_year_week(date_str):
    """
    날짜 문자열을 year, week으로 변환
    
    Args:
        date_str: 날짜 문자열 (예: "2024-01-15", "2024-W03", "2024-01-15T00:00:00.000Z")
    
    Returns:
        tuple: (year, week)
    """
    if date_str is None or pd.isna(date_str):
        return None, None
    
    date_str = str(date_str).strip()
    if not date_str:
        return None, None
    
    try:
        # ISO 주차 형식 (예: "2024-W03")
        if 'W' in date_str:
            parts = date_str.split('-W')
            return int(parts[0]), int(parts[1].split('-')[0])  # "2024-W03-1" 형식 처리
        
        # "연도-주차" 형식 (예: "2024-03" 또는 "202403")
        if len(date_str) == 7 and '-' in date_str:
            parts = date_str.split('-')
            year, week = int(parts[0]), int(parts[1])
            if 1 <= week <= 53:
                return year, week
        
        if len(date_str) == 6 and date_str.isdigit():
            year = int(date_str[:4])
            week = int(date_str[4:])
            if 1 <= week <= 53:
                return year, week
        
        # 일반 날짜/시간 형식 (예: "2024-01-15", "2024-01-15T00:00:00.000Z")
        date_obj = pd.to_datetime(date_str)
        # ISO 주차 계산 (월요일 시작)
        year = date_obj.isocalendar()[0]
        week = date_obj.isocalendar()[1]
        return int(year), int(week)
    except Exception as e:
        return None, None


def extract_year_week_from_data(df, dsid):
    """
    DataFrame에서 year, week 컬럼 추출/생성
    
    다양한 날짜 컬럼명과 형식을 지원:
    - 직접적인 year/week 컬럼
    - 날짜 컬럼 (date, datetime, collected_at, collectedAt 등)
    - 한글 컬럼 (연도, 주차, 날짜, 수집일 등)
    
    Args:
        df: 원본 DataFrame
        dsid: 데이터셋 ID (디버깅용)
    
    Returns:
        DataFrame: year, week 컬럼이 추가된 DataFrame
    """
    print(f"   🔍 [{dsid}] year/week 컬럼 추출 시작...")
    print(f"   📋 [{dsid}] 원본 컬럼: {list(df.columns)}")
    
    # 1. 이미 year/week 컬럼이 있는 경우
    if 'year' in df.columns and 'week' in df.columns:
        print(f"   ✅ [{dsid}] year/week 컬럼 이미 존재")
        return df
    
    # 1-1. 한글 컬럼인 경우
    if '연도' in df.columns and '주차' in df.columns:
        df['year'] = df['연도'].astype(int)
        df['week'] = df['주차'].astype(int)
        print(f"   ✅ [{dsid}] 연도/주차 → year/week 변환 완료")
        return df
    
    # 2. 날짜 컬럼 후보 목록 (우선순위 순)
    date_col_candidates = [
        # API 메타데이터 필드
        'collectedAt', 'collected_at', 'createdAt', 'created_at',
        'updatedAt', 'updated_at',
        # 일반 날짜 필드
        'date', 'datetime', 'time', 'timestamp',
        # 주차 관련 필드
        'year_week', 'yearWeek', 'week_date', 'weekDate',
        # 한글 필드
        '날짜', '수집일', '기준일', '조회일', '수집시간',
    ]
    
    # 3. 날짜 컬럼 찾기
    date_col = None
    for candidate in date_col_candidates:
        if candidate in df.columns:
            date_col = candidate
            break
        # 대소문자 무시하여 찾기
        for col in df.columns:
            if col.lower() == candidate.lower():
                date_col = col
                break
        if date_col:
            break
    
    if date_col:
        print(f"   🔍 [{dsid}] 날짜 컬럼 발견: '{date_col}'")
        print(f"   📋 [{dsid}] 샘플 값: {df[date_col].head(3).tolist()}")
        
        # year, week 컬럼 생성
        year_week_data = df[date_col].apply(parse_date_to_year_week)
        df['year'] = year_week_data.apply(lambda x: x[0])
        df['week'] = year_week_data.apply(lambda x: x[1])
        
        # None 값 확인
        null_count = df['year'].isna().sum()
        if null_count > 0:
            print(f"   ⚠️  [{dsid}] {null_count}개 행의 날짜 변환 실패 (제거됨)")
            df = df.dropna(subset=['year', 'week'])
        
        if len(df) > 0:
            df['year'] = df['year'].astype(int)
            df['week'] = df['week'].astype(int)
            print(f"   ✅ [{dsid}] year/week 변환 완료: {len(df)}건")
            print(f"   📊 [{dsid}] year 범위: {df['year'].min()}-{df['year'].max()}, week 범위: {df['week'].min()}-{df['week'].max()}")
        else:
            print(f"   ❌ [{dsid}] 변환된 데이터 없음")
        
        return df
    
    # 4. 날짜 컬럼이 없는 경우 - 다른 방법 시도
    print(f"   ⚠️  [{dsid}] 날짜 컬럼을 찾을 수 없음. 다른 방법 시도...")
    
    # 4-1. year만 있고 week 컬럼 후보 찾기
    if 'year' in df.columns:
        week_candidates = ['week', 'week_num', 'weekNum', '주차', 'wk']
        for candidate in week_candidates:
            if candidate in df.columns:
                df['week'] = df[candidate].astype(int)
                print(f"   ✅ [{dsid}] year + {candidate} → week 사용")
                return df
    
    # 4-2. 연도만 있는 경우
    if '연도' in df.columns:
        df['year'] = df['연도'].astype(int)
        week_candidates = ['week', 'week_num', 'weekNum', '주차', 'wk']
        for candidate in week_candidates:
            if candidate in df.columns:
                df['week'] = df[candidate].astype(int)
                print(f"   ✅ [{dsid}] 연도 → year, {candidate} → week 사용")
                return df
    
    print(f"   ❌ [{dsid}] year/week 컬럼 생성 실패. 사용 가능한 컬럼: {list(df.columns)}")
    return df


def flatten_parsed_data(data_list):
    """
    API 응답의 parsedData 필드를 플래튼하여 실제 데이터 추출
    
    API 응답 구조 예시:
    [
        {
            "dsId": "ds_0701",
            "origin": "uuid",
            "contentType": "application/json",
            "parsedData": {"검색어1": 100, "검색어2": 50, ...},
            "collectedAt": "2024-01-15T00:00:00.000Z"
        },
        ...
    ]
    
    Args:
        data_list: API에서 받은 raw 데이터 리스트
    
    Returns:
        list: 플래튼된 데이터 리스트
    """
    flattened = []
    
    for item in data_list:
        if isinstance(item, dict):
            # parsedData 필드가 있는 경우
            if 'parsedData' in item and isinstance(item['parsedData'], dict):
                flat_item = item['parsedData'].copy()
                # collectedAt 등 메타데이터 추가
                if 'collectedAt' in item:
                    flat_item['collectedAt'] = item['collectedAt']
                if 'dsId' in item:
                    flat_item['dsId'] = item['dsId']
                flattened.append(flat_item)
            else:
                # parsedData가 없으면 item 그대로 사용
                flattened.append(item)
        else:
            flattened.append(item)
    
    return flattened


def fetch_trend_data(dsid="ds_0701", cnt=100):
    """
    GFID API에서 트렌드 데이터 다운로드 및 year/week 변환
    (src_jaehong/api/etlDataApi.js 패턴 사용)
    
    1단계: recent API로 origin 목록 조회
    2단계: 각 unique origin에 대해 실제 데이터(parsedData) 조회
    3단계: 모든 데이터 병합 및 year/week 변환
    
    Args:
        dsid: 데이터셋 ID (ds_0701=Google, ds_0801=Naver, ds_0901=Twitter)
        cnt: 최근 데이터 건수 (origin 조회용)
    
    Returns:
        DataFrame: year, week 컬럼이 추가된 트렌드 데이터
    """
    try:
        from .api_client import get_etl_data_by_origin
    except ImportError:
        from api_client import get_etl_data_by_origin
    
    dsid_names = {
        'ds_0701': 'Google Trends',
        'ds_0801': 'Naver Trends', 
        'ds_0901': 'Twitter Trends'
    }
    dsid_name = dsid_names.get(dsid, dsid)
    
    print(f"\n📡 GFID API에서 {dsid} ({dsid_name}) 데이터 다운로드 중...")
    
    try:
        # 1단계: recent API로 메타데이터(origin 목록) 조회
        print(f"   [1/3] 메타데이터 조회 중...")
        meta_data = get_recent_etl_data(dsid, cnt)
        
        if not meta_data:
            print(f"   ⚠️  [{dsid}] 메타데이터 없음")
            return pd.DataFrame()
        
        print(f"   ✅ [{dsid}] 메타데이터 수신: {len(meta_data)}건")
        
        # 2단계: unique origin 목록 추출
        origins = set()
        origin_collected_at = {}  # origin -> collectedAt 매핑
        
        for item in meta_data:
            if isinstance(item, dict) and 'origin' in item:
                origin = item['origin']
                origins.add(origin)
                if 'collectedAt' in item:
                    origin_collected_at[origin] = item['collectedAt']
        
        origins = list(origins)
        print(f"   [2/3] 고유 origin 개수: {len(origins)}개")
        
        if not origins:
            print(f"   ⚠️  [{dsid}] origin이 없습니다.")
            return pd.DataFrame()
        
        # 3단계: 각 origin에서 실제 데이터(parsedData) 조회
        print(f"   [3/3] 각 origin에서 실제 데이터 조회 중...")
        all_data = []
        success_count = 0
        
        # 최대 조회 개수 제한 (너무 많으면 시간 오래 걸림)
        max_origins = min(len(origins), 50)
        
        for i, origin in enumerate(origins[:max_origins]):
            try:
                origin_data = get_etl_data_by_origin(dsid, origin)
                
                if origin_data:
                    # parsedData 추출
                    if isinstance(origin_data, list):
                        for item in origin_data:
                            if isinstance(item, dict):
                                parsed = item.get('parsedData', {})
                                if parsed and isinstance(parsed, dict):
                                    # collectedAt 추가
                                    if 'collectedAt' in item:
                                        parsed['collectedAt'] = item['collectedAt']
                                    elif origin in origin_collected_at:
                                        parsed['collectedAt'] = origin_collected_at[origin]
                                    all_data.append(parsed)
                                    success_count += 1
                    elif isinstance(origin_data, dict):
                        parsed = origin_data.get('parsedData', {})
                        if parsed and isinstance(parsed, dict):
                            if 'collectedAt' in origin_data:
                                parsed['collectedAt'] = origin_data['collectedAt']
                            elif origin in origin_collected_at:
                                parsed['collectedAt'] = origin_collected_at[origin]
                            all_data.append(parsed)
                            success_count += 1
                            
            except Exception as e:
                # 개별 origin 실패는 무시하고 계속
                pass
            
            # 진행률 표시 (10개마다)
            if (i + 1) % 10 == 0:
                print(f"      진행: {i + 1}/{max_origins} origins 처리됨")
        
        print(f"   ✅ [{dsid}] 실제 데이터 {len(all_data)}건 수집 완료")
        
        if not all_data:
            print(f"   ⚠️  [{dsid}] parsedData가 비어 있음. 메타데이터로 대체...")
            # 실패 시 메타데이터라도 사용
            df = pd.DataFrame(meta_data)
        else:
            df = pd.DataFrame(all_data)
        
        if df.empty:
            print(f"   ⚠️  [{dsid}] DataFrame이 비어 있음")
            return pd.DataFrame()
        
        print(f"   📋 [{dsid}] DataFrame 컬럼: {list(df.columns)[:10]}...")
        print(f"   📋 [{dsid}] DataFrame 크기: {df.shape}")
        
        # year, week 컬럼 추출/생성
        df = extract_year_week_from_data(df, dsid)
        
        # year/week 컬럼 검증
        if 'year' not in df.columns or 'week' not in df.columns:
            print(f"   ❌ [{dsid}] year/week 컬럼 생성 실패")
            return pd.DataFrame()
        
        # 결과 요약
        if len(df) > 0:
            print(f"   ✅ [{dsid}] 최종 데이터: {len(df)}건")
            print(f"   📊 [{dsid}] year 범위: {df['year'].min()}-{df['year'].max()}")
            print(f"   📊 [{dsid}] week 범위: {df['week'].min()}-{df['week'].max()}")
            # 실제 데이터 컬럼 확인 (year, week, collectedAt 제외)
            data_cols = [c for c in df.columns if c not in ['year', 'week', 'collectedAt', 'dsId', 'origin', 'id', 'contentType']]
            if data_cols:
                print(f"   📊 [{dsid}] 데이터 컬럼: {data_cols[:5]}{'...' if len(data_cols) > 5 else ''}")
        
        return df
    
    except Exception as e:
        print(f"   ❌ [{dsid}] 다운로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def merge_trend_data(google_df, naver_df, twitter_df):
    """
    3개 트렌드 데이터를 병합
    
    Args:
        google_df: Google Trends 데이터
        naver_df: Naver Trends 데이터
        twitter_df: Twitter Trends 데이터
    
    Returns:
        DataFrame: 병합된 트렌드 데이터
    """
    print("\n🔗 트렌드 데이터 병합 중...")
    
    # 기본 시간 컬럼 (year, week) 확인
    all_dfs = []
    
    if not google_df.empty:
        # Google Trends 컬럼명 정규화
        google_df = google_df.rename(columns=lambda x: f"google_{x}" if x not in ['year', 'week', '연도', '주차'] else x)
        all_dfs.append(google_df)
        print(f"   • Google Trends: {len(google_df)}건, 컬럼: {list(google_df.columns)[:5]}...")
    
    if not naver_df.empty:
        # Naver Trends 컬럼명 정규화
        naver_df = naver_df.rename(columns=lambda x: f"naver_{x}" if x not in ['year', 'week', '연도', '주차'] else x)
        all_dfs.append(naver_df)
        print(f"   • Naver Trends: {len(naver_df)}건, 컬럼: {list(naver_df.columns)[:5]}...")
    
    if not twitter_df.empty:
        # Twitter Trends 컬럼명 정규화
        twitter_df = twitter_df.rename(columns=lambda x: f"twitter_{x}" if x not in ['year', 'week', '연도', '주차'] else x)
        all_dfs.append(twitter_df)
        print(f"   • Twitter Trends: {len(twitter_df)}건, 컬럼: {list(twitter_df.columns)[:5]}...")
    
    if not all_dfs:
        print("   ⚠️  병합할 데이터가 없습니다.")
        return pd.DataFrame()
    
    # year, week 기준으로 병합
    merged = all_dfs[0]
    for df in all_dfs[1:]:
        # 한글/영문 컬럼명 통일
        df = df.rename(columns={'연도': 'year', '주차': 'week'})
        merged = merged.rename(columns={'연도': 'year', '주차': 'week'})
        merged = merged.merge(df, on=['year', 'week'], how='outer')
    
    # 정렬
    if 'year' in merged.columns and 'week' in merged.columns:
        merged = merged.sort_values(['year', 'week']).reset_index(drop=True)
    
    print(f"   ✅ 병합 완료: {len(merged)}건, {len(merged.columns)}개 컬럼")
    return merged


def create_trends_table(db: TimeSeriesDB, table_name="trends_data"):
    """
    트렌드 데이터 테이블 생성
    
    Args:
        db: TimeSeriesDB 인스턴스
        table_name: 테이블 이름
    """
    db.connect()
    
    # 기존 테이블 삭제
    with db.conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
        db.conn.commit()
        print(f"   ✅ 기존 {table_name} 테이블 삭제 완료")


def insert_trends_data(db: TimeSeriesDB, df: pd.DataFrame, table_name="trends_data"):
    """
    트렌드 데이터를 PostgreSQL에 삽입
    
    Args:
        db: TimeSeriesDB 인스턴스
        df: 트렌드 데이터 DataFrame
        table_name: 테이블 이름
    """
    if df.empty:
        print("   ⚠️  삽입할 데이터가 없습니다.")
        return
    
    db.connect()
    
    # 컬럼명 정규화 (한글 → 영문)
    df = df.rename(columns={'연도': 'year', '주차': 'week'})
    
    # 테이블 생성 (자동으로 컬럼 타입 추론)
    with db.conn.cursor() as cur:
        # 테이블 존재 여부 확인
        cur.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = '{table_name}'
            )
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            # 동적으로 CREATE TABLE 생성
            columns_def = []
            for col in df.columns:
                dtype = df[col].dtype
                if dtype in ['int64', 'int32']:
                    sql_type = 'INTEGER'
                elif dtype in ['float64', 'float32']:
                    sql_type = 'REAL'
                else:
                    sql_type = 'TEXT'
                columns_def.append(f'"{col}" {sql_type}')
            
            create_sql = f"CREATE TABLE {table_name} ({', '.join(columns_def)})"
            cur.execute(create_sql)
            db.conn.commit()
            print(f"   ✅ {table_name} 테이블 생성 완료 ({len(columns_def)}개 컬럼)")
    
    # 데이터 삽입
    df = df.where(pd.notnull(df), None)
    columns = list(df.columns)
    values = df.values.tolist()
    placeholders = ','.join(['%s'] * len(columns))
    col_names = ','.join([f'"{col}"' for col in columns])
    sql = f'INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})'
    
    with db.conn.cursor() as cur:
        import psycopg2.extras
        psycopg2.extras.execute_batch(cur, sql, values)
    db.conn.commit()
    
    print(f"   ✅ {table_name}에 {len(df)}건 삽입 완료")


def update_trends_database(
    db_name="trends",
    table_name="trends_data",
    cnt=500
):
    """
    트렌드 데이터베이스 전체 업데이트 프로세스
    (GFID API 직접 호출 - src_jaehong 패턴)
    
    Args:
        db_name: PostgreSQL 데이터베이스 이름
        table_name: 테이블 이름
        cnt: 각 데이터셋당 최근 건수
    """
    print("\n" + "="*60)
    print("📊 트렌드 데이터 PostgreSQL 업데이트 (GFID API 직접 호출)")
    print("="*60)
    
    # 인증 설정 확인
    if not is_auth_configured():
        print("\n⚠️  GFID API 인증 설정이 필요합니다.")
        print("   .env 파일에 다음 환경 변수를 설정하세요:")
        print("   - GFID_CLIENT_ID")
        print("   - GFID_CLIENT_SECRET")
        return False
    
    # 1. GFID API에서 데이터 다운로드
    google_df = fetch_trend_data("ds_0701", cnt)
    naver_df = fetch_trend_data("ds_0801", cnt)
    twitter_df = fetch_trend_data("ds_0901", cnt)
    
    # 2. 데이터 병합
    merged_df = merge_trend_data(google_df, naver_df, twitter_df)
    
    if merged_df.empty:
        print("\n❌ 트렌드 데이터가 없어 업데이트를 중단합니다.")
        return False
    
    # 3. PostgreSQL 연결 (trends 데이터베이스)
    print(f"\n💾 PostgreSQL '{db_name}' 데이터베이스에 저장 중...")
    db = TimeSeriesDB(dbname=db_name)
    
    try:
        # 데이터베이스가 없으면 생성
        db.connect()
        print(f"   ✅ '{db_name}' 데이터베이스 연결 완료")
    except Exception as e:
        print(f"   ⚠️  '{db_name}' 데이터베이스가 없습니다. 생성 중...")
        # postgres 기본 DB에 연결하여 trends DB 생성
        temp_db = TimeSeriesDB(dbname='postgres')
        temp_db.connect()
        temp_db.conn.autocommit = True
        with temp_db.conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE {db_name}")
        temp_db.close()
        print(f"   ✅ '{db_name}' 데이터베이스 생성 완료")
        
        # 다시 연결
        db = TimeSeriesDB(dbname=db_name)
        db.connect()
    
    # 4. 테이블 생성 및 데이터 삽입
    create_trends_table(db, table_name)
    insert_trends_data(db, merged_df, table_name)
    
    # 5. CSV 백업
    csv_path = "trends_data.csv"
    merged_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n   ✅ CSV 백업 완료: {csv_path}")
    
    db.close()
    
    print("\n" + "="*60)
    print("✅ 트렌드 데이터 업데이트 완료!")
    print("="*60)
    print(f"\n트렌드 데이터:")
    print(f"  • 데이터베이스: {db_name}")
    print(f"  • 테이블: {table_name}")
    print(f"  • 데이터 건수: {len(merged_df)}")
    print(f"  • 컬럼 수: {len(merged_df.columns)}")
    print(f"  • 백업 파일: {csv_path}")
    
    return True


# =========================
# 메인 실행 부분
# =========================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("📊 PostgreSQL 데이터베이스 통합 업데이트")
    print("="*60)
    
    # 환경 확인
    print("\n환경 확인:")
    print(f"  • GFID API 인증 설정: {'✅ 완료' if is_auth_configured() else '❌ 미완료'}")
    print(f"  • data/before 폴더 존재: {os.path.exists('data/before')}")
    print(f"  • merged_influenza_data.csv 존재: {os.path.exists('merged_influenza_data.csv')}")
    
    # GFID API 인증 설정 안내
    if not is_auth_configured():
        print("\n⚠️  GFID API 인증을 사용하려면 .env 파일에 다음을 설정하세요:")
        print("   GFID_CLIENT_ID=your_client_id")
        print("   GFID_CLIENT_SECRET=your_client_secret")
    
    # 사용자 확인
    print("\n다음 작업을 수행합니다:")
    print("\n[1단계] 인플루엔자 데이터 업데이트 (influenza DB)")
    print("  1-1. API에서 인플루엔자 데이터 가져오기")
    print("  1-2. data/before 폴더의 과거 데이터 로딩")
    print("  1-3. 모든 데이터 병합 및 중복 제거")
    print("  1-4. PostgreSQL influenza DB에 저장")
    print("  1-5. CSV로 백업 (merged_influenza_data.csv)")
    
    # 트렌드 데이터는 현재 비활성화 (API가 parsedData를 반환하지 않음)
    # 나중에 API가 수정되면 다시 활성화 가능
    print("\n[2단계] 트렌드 데이터 업데이트 (현재 비활성화)")
    print("  ⚠️  트렌드 API가 메타데이터만 반환하여 실제 데이터 수집 불가")
    print("  ⚠️  API 수정 후 다시 활성화 예정")
    
    response = input("\n계속하시겠습니까? (y/n): ").lower()
    
    if response == 'y':
        success_count = 0
        total_steps = 1  # 현재는 인플루엔자 데이터만 업데이트
        
        # 1단계: 인플루엔자 데이터 업데이트
        print("\n" + "="*60)
        print("1단계: 인플루엔자 데이터 업데이트")
        print("="*60)
        try:
            merge_and_update_database(
                table_name="influenza_data",
                fetch_latest=True,  # API에서 최신 데이터 가져오기
                before_dir='data/before',
                consolidate=True  # 수정된 병합 로직 사용
            )
            success_count += 1
        except Exception as e:
            print(f"\n❌ 인플루엔자 데이터 업데이트 실패: {e}")
            import traceback
            traceback.print_exc()
        
        # 2단계: 트렌드 데이터 업데이트 (현재 비활성화)
        # API가 parsedData를 반환하지 않아 실제 데이터 수집 불가
        # 나중에 API가 수정되면 아래 주석을 해제하여 활성화
        print("\n" + "="*60)
        print("2단계: 트렌드 데이터 업데이트 (건너뜀)")
        print("="*60)
        print("\n⚠️  트렌드 데이터 업데이트가 비활성화되어 있습니다.")
        print("   현재 API가 메타데이터만 반환하여 실제 데이터를 수집할 수 없습니다.")
        print("   API 수정 후 update_trends_database() 함수를 활성화하세요.")
        
        # === 트렌드 데이터 업데이트 (비활성화됨) ===
        # if is_auth_configured():
        #     print("\n📡 GFID API 직접 호출 방식으로 트렌드 데이터를 가져옵니다...")
        #     try:
        #         if update_trends_database():
        #             success_count += 1
        #     except Exception as e:
        #         print(f"\n⚠️  트렌드 데이터 업데이트 실패: {e}")
        #         import traceback
        #         traceback.print_exc()
        
        # 최종 결과
        print("\n" + "="*60)
        print("📊 업데이트 완료!")
        print("="*60)
        print(f"\n업데이트 결과: {success_count}/{total_steps} 성공")
        
        if success_count >= 1:
            print("\n✅ 데이터베이스 업데이트 완료!")
            print("\n다음 명령어로 모델을 실행할 수 있습니다:")
            print("  python patchTST.py")
            print("\n생성된 파일:")
            if os.path.exists('merged_influenza_data.csv'):
                print("  ✓ merged_influenza_data.csv (인플루엔자 데이터)")
            if os.path.exists('trends_data.csv'):
                print("  ✓ trends_data.csv (트렌드 데이터)")
        else:
            print("\n❌ 데이터베이스 업데이트 실패")
            print("   오류 메시지를 확인하고 다시 시도하세요.")
    else:
        print("\n취소되었습니다.")
