#!/usr/bin/env python3
"""
weather_for_influenza.csv를 PostgreSQL weather_data 테이블에 업로드하는 스크립트
"""

import sys
import pandas as pd
from pathlib import Path

# database 모듈 임포트
sys.path.insert(0, str(Path(__file__).parent))
from database.db_utils import TimeSeriesDB

def upload_weather_data():
    """날씨 데이터를 PostgreSQL에 업로드"""
    
    print("=" * 70)
    print("🌡️  weather_for_influenza.csv → PostgreSQL 업로드")
    print("=" * 70)
    
    # CSV 파일 경로
    csv_path = Path(__file__).parent / "data" / "weather_for_influenza.csv"
    
    # 1. CSV 파일 확인
    print(f"\n1️⃣ CSV 파일 확인: {csv_path}")
    if not csv_path.exists():
        print(f"   ❌ 파일을 찾을 수 없습니다: {csv_path}")
        sys.exit(1)
    print(f"   ✅ 파일 존재 확인")
    
    # 2. CSV 파일 로드
    print(f"\n2️⃣ CSV 파일 로드 중...")
    try:
        df = pd.read_csv(csv_path)
        print(f"   ✅ 로드 완료: {df.shape}")
        print(f"   - 컬럼: {list(df.columns)}")
        print(f"   - 행 수: {len(df)}")
        print(f"\n   샘플 데이터:")
        print(df.head().to_string(index=False))
    except Exception as e:
        print(f"   ❌ 파일 로드 실패: {e}")
        sys.exit(1)
    
    # 3. 데이터 타입 확인
    print(f"\n3️⃣ 데이터 타입 확인:")
    print(df.dtypes.to_string())
    
    # 4. 결측치 확인
    print(f"\n4️⃣ 결측치 확인:")
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"   ⚠️ 결측치 발견:")
        print(null_counts[null_counts > 0].to_string())
    else:
        print(f"   ✅ 결측치 없음")
    
    # 5. PostgreSQL 연결 및 업로드
    print(f"\n5️⃣ PostgreSQL에 업로드 중...")
    try:
        db = TimeSeriesDB()
        db.connect()
        
        # 테이블 생성
        print(f"   - 테이블 생성: weather_data")
        db.create_table_from_dataframe(df, "weather_data", if_exists="replace")
        
        # 데이터 삽입
        print(f"   - 데이터 삽입: {len(df)}건")
        db.insert_dataframe(df, "weather_data")
        
        db.close()
        print(f"   ✅ 업로드 완료")
    except Exception as e:
        print(f"   ❌ 업로드 실패: {e}")
        print(f"\n   💡 확인 사항:")
        print(f"      1. PostgreSQL이 실행 중인가요?")
        print(f"      2. .env 파일이 올바르게 설정되어 있나요?")
        print(f"         - PG_HOST, PG_DB, PG_USER, PG_PASSWORD 확인")
        sys.exit(1)
    
    # 6. 업로드 결과 확인
    print(f"\n6️⃣ 업로드 결과 확인...")
    try:
        db = TimeSeriesDB()
        db.connect()
        
        result_df = db.load_data("weather_data")
        print(f"   ✅ 조회 완료: {result_df.shape}")
        print(f"   - 컬럼: {list(result_df.columns)}")
        print(f"   - 년도 범위: {result_df['year'].min():.0f} ~ {result_df['year'].max():.0f}")
        print(f"   - 주차 범위: {result_df['week'].min():.0f} ~ {result_df['week'].max():.0f}")
        
        # 통계 정보
        print(f"\n   📊 데이터 통계:")
        for col in result_df.select_dtypes(include=['float64', 'int64']).columns:
            if col not in ['year', 'week']:
                data = result_df[col].dropna()
                print(f"      {col:20} | 평균: {data.mean():8.2f} | 범위: [{data.min():8.2f}, {data.max():8.2f}]")
        
        db.close()
    except Exception as e:
        print(f"   ⚠️ 결과 확인 실패: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 업로드 완료!")
    print("=" * 70)
    print("\n📚 다음 단계:")
    print("   1. patchTST.py 실행하여 날씨 데이터 확인")
    print("      python patchTST.py")
    print("\n   2. PostgreSQL에서 직접 확인")
    print("      psql -U postgres -d influenza -c \"SELECT * FROM weather_data LIMIT 5;\"")

if __name__ == "__main__":
    upload_weather_data()
