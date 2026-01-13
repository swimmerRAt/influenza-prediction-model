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
from tabulate import tabulate

# Pandas 디스플레이 옵션 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 30)


def check_database(db_path="influenza_data.duckdb"):
    """DuckDB 데이터베이스 내용 확인"""
    
    print("\n" + "="*100)
    print("🗄️  DuckDB 데이터베이스 상세 조회")
    print("="*100)
    
    # 1. 테이블 정보 확인
    print("\n📋 테이블 구조 정보")
    print("-"*100)
    with TimeSeriesDB(db_path) as db:
        db.get_table_info("influenza_data")
    
    # 2. 전체 데이터 로드
    df_full = load_from_duckdb()
    
    # 3. 데이터 샘플 보기 - 모든 컬럼 표시
    print("\n" + "="*100)
    print("📋 데이터 샘플 미리보기 (처음 15행, 전체 컬럼)")
    print("="*100)
    print(f"총 {len(df_full):,}행 × {len(df_full.columns)}열\n")
    
    df_sample = df_full.head(15)
    print(tabulate(df_sample, headers='keys', tablefmt='simple', showindex=True, maxcolwidths=15))
    
    # 4. 컬럼별 정보
    print("\n" + "="*100)
    print("📊 컬럼별 상세 정보")
    print("="*100)
    col_info = pd.DataFrame({
        '컬럼명': df_full.columns,
        '타입': [str(dtype)[:10] for dtype in df_full.dtypes.values],
        '결측치': df_full.isna().sum().values,
        '결측치%': (df_full.isna().sum() / len(df_full) * 100).round(1).values,
        '고유값': [df_full[col].nunique() for col in df_full.columns]
    })
    print(tabulate(col_info, headers='keys', tablefmt='simple', showindex=False))
    
    # 5. 연도별 데이터 개수
    if '연도' in df_full.columns:
        print("\n" + "="*100)
        print("📈 연도별 데이터 분포")
        print("="*100)
        year_counts = df_full['연도'].value_counts().sort_index().reset_index()
        year_counts.columns = ['연도', '건수']
        year_counts['비율(%)'] = (year_counts['건수'] / len(df_full) * 100).round(1)
        print(tabulate(year_counts, headers='keys', tablefmt='simple', showindex=False))
        print(f"\n📅 연도 범위: {df_full['연도'].min():.0f} ~ {df_full['연도'].max():.0f}")
    
    # 6. 데이터셋 ID별 개수
    if 'dataset_id' in df_full.columns:
        print("\n" + "="*100)
        print("📊 데이터셋 ID별 분포")
        print("="*100)
        dataset_counts = df_full['dataset_id'].value_counts().sort_index().reset_index()
        dataset_counts.columns = ['데이터셋 ID', '건수']
        dataset_counts['비율(%)'] = (dataset_counts['건수'] / len(df_full) * 100).round(1)
        print(tabulate(dataset_counts, headers='keys', tablefmt='simple', showindex=False))
    
    # 7. 기본 통계 요약
    print("\n" + "="*100)
    print("📐 데이터베이스 통계 요약")
    print("="*100)
    print(f"전체 행 수:        {len(df_full):,} 행")
    print(f"전체 컬럼 수:       {len(df_full.columns)} 개")
    print(f"메모리 사용량:      {df_full.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    if '연도' in df_full.columns:
        print(f"연도 범위:         {df_full['연도'].min():.0f} ~ {df_full['연도'].max():.0f}")
    
    # 8. 수치형 컬럼 통계
    numeric_cols = [col for col in ['연도', '주차', '의사환자 분율', '입원환자 수', '인플루엔자 검출률', '예방접종률'] 
                    if col in df_full.columns]
    
    if numeric_cols:
        print("\n" + "="*100)
        print("🔢 주요 수치형 컬럼 통계")
        print("="*100)
        stats_df = df_full[numeric_cols].describe().T
        stats_df = stats_df[['count', 'mean', 'std', 'min', '50%', 'max']]
        stats_df.columns = ['개수', '평균', '표준편차', '최소', '중앙값', '최대']
        stats_df = stats_df.round(1)
        stats_df.insert(0, '컬럼명', stats_df.index)
        stats_df = stats_df.reset_index(drop=True)
        print(tabulate(stats_df, headers='keys', tablefmt='simple', showindex=False))
    
    print("\n" + "="*100)
    print("✅ 데이터베이스 조회 완료!")
    print("="*100)


if __name__ == "__main__":
    import sys
    
    # 명령행 인자로 DB 경로 지정 가능
    db_path = sys.argv[1] if len(sys.argv) > 1 else "influenza_data.duckdb"
    
    check_database(db_path)
    
    print("\n💡 추가 확인 방법:")
    print("="*100)
    print("""
✨ 데이터를 더 자세히 보려면 다음 코드를 사용하세요:

from database.db_utils import load_from_duckdb
from tabulate import tabulate

# 특정 연도만 테이블로 보기
df = load_from_duckdb(where="연도 = 2023")
print(tabulate(df.head(50), headers='keys', tablefmt='simple', showindex=True))

# 특정 컬럼만 선택해서 보기
df = load_from_duckdb()
df_select = df[['연도', '주차', 'dataset_id', '의사환자 분율']]
print(tabulate(df_select.head(50), headers='keys', tablefmt='simple', showindex=True))

# CSV로 내보내기
df = load_from_duckdb()
df.to_csv('exported_data.csv', index=False, encoding='utf-8-sig')

# Excel로 내보내기 (openpyxl 설치 필요: pip install openpyxl)
df.to_excel('exported_data.xlsx', index=False)
    """)
    print("="*100)
