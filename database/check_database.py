"""
DuckDB 데이터베이스 내용 확인 스크립트
"""

try:
    # 모듈로서 import 될 때
    from .db_utils import TimeSeriesDB, load_from_duckdb
except ImportError:
    # 직접 실행될 때
    from db_utils import TimeSeriesDB, load_from_duckdb

import pandas as pd

def check_database(db_path="influenza_data.duckdb"):
    """DuckDB 데이터베이스 내용 확인"""
    
    print("\n" + "="*60)
    print("📊 DuckDB 데이터베이스 확인")
    print("="*60)
    
    # 1. 테이블 정보 확인
    with TimeSeriesDB(db_path) as db:
        db.get_table_info("influenza_data")
    
    # 2. 데이터 샘플 보기
    print("\n" + "="*60)
    print("📋 데이터 샘플 (처음 10행)")
    print("="*60)
    df_sample = load_from_duckdb(limit=10)
    print(df_sample)
    
    # 3. 연도별 데이터 개수
    print("\n" + "="*60)
    print("📈 연도별 데이터 개수")
    print("="*60)
    df_full = load_from_duckdb()
    if 'year' in df_full.columns:
        year_counts = df_full['year'].value_counts().sort_index()
        print(year_counts)
        print(f"\n총 연도 범위: {year_counts.index.min()} ~ {year_counts.index.max()}")
    
    # 4. 데이터셋 ID별 개수
    if 'dataset_id' in df_full.columns:
        print("\n" + "="*60)
        print("📊 데이터셋 ID별 개수")
        print("="*60)
        dataset_counts = df_full['dataset_id'].value_counts().sort_index()
        print(dataset_counts)
    
    # 5. 기본 통계
    print("\n" + "="*60)
    print("📐 기본 통계")
    print("="*60)
    print(f"전체 행 수: {len(df_full):,}")
    print(f"전체 컬럼 수: {len(df_full.columns)}")
    print(f"메모리 사용량: {df_full.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # 6. 결측치 확인
    if 'year' in df_full.columns and 'week' in df_full.columns:
        print("\n" + "="*60)
        print("🔍 주요 컬럼 결측치")
        print("="*60)
        key_columns = ['year', 'week', 'dataset_id'] if 'dataset_id' in df_full.columns else ['year', 'week']
        for col in key_columns[:5]:  # 처음 5개 컬럼만
            if col in df_full.columns:
                missing = df_full[col].isna().sum()
                print(f"{col}: {missing:,} ({missing/len(df_full)*100:.1f}%)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    
    # 명령행 인자로 DB 경로 지정 가능
    db_path = sys.argv[1] if len(sys.argv) > 1 else "influenza_data.duckdb"
    
    check_database(db_path)
    
    print("\n💡 추가 확인 방법:")
    print("-" * 60)
    print("""
# 특정 연도만 보기
from database.db_utils import load_from_duckdb
df = load_from_duckdb(where="year = 2023")

# 특정 컬럼만 보기
df = load_from_duckdb(columns=['year', 'week', 'dataset_id'])

# 최근 데이터 확인
df = load_from_duckdb(limit=100, order_by="year DESC, week DESC")
    """)
