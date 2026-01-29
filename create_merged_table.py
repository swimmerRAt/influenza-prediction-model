#!/usr/bin/env python3
"""
PostgreSQL에서 influenza_data와 weather_data를 병합하여 merged_data 테이블 생성

이 스크립트는:
1. PostgreSQL에서 influenza_data 테이블 로드
2. PostgreSQL에서 weather_data 테이블 로드
3. year, week 기준으로 LEFT JOIN 병합
4. 병합된 데이터를 merged_data 테이블로 PostgreSQL에 저장
"""

import sys
import pandas as pd
from pathlib import Path

# database 모듈 임포트
sys.path.insert(0, str(Path(__file__).parent))
from database.db_utils import TimeSeriesDB


def load_influenza_data(db: TimeSeriesDB) -> pd.DataFrame:
    """PostgreSQL에서 influenza_data 테이블 로드"""
    print("\n📊 1. influenza_data 테이블 로드")
    print("=" * 70)
    
    df = db.load_data(table_name="influenza_data")
    
    print(f"✅ influenza_data 로드 완료")
    print(f"   - 행 수: {len(df):,}")
    print(f"   - 컬럼: {list(df.columns)}")
    print(f"   - 기간: {df['year'].min()}-{df['year'].max()}")
    
    return df


def load_weather_data(db: TimeSeriesDB) -> pd.DataFrame:
    """PostgreSQL에서 weather_data 테이블 로드"""
    print("\n🌡️  2. weather_data 테이블 로드")
    print("=" * 70)
    
    try:
        df = db.load_data(table_name="weather_data")
        
        print(f"✅ weather_data 로드 완료")
        print(f"   - 행 수: {len(df):,}")
        print(f"   - 컬럼: {list(df.columns)}")
        print(f"   - 기간: {df['year'].min()}-{df['year'].max()}")
        
        return df
    except Exception as e:
        print(f"❌ weather_data 로드 실패: {e}")
        print(f"   weather_data 테이블이 존재하지 않습니다.")
        return None


def merge_data(df_influenza: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """influenza_data와 weather_data를 year, week 기준으로 병합"""
    print("\n🔗 3. 데이터 병합")
    print("=" * 70)
    
    # 데이터 타입 확인 및 변환
    df_influenza['year'] = pd.to_numeric(df_influenza['year'], errors='coerce')
    df_influenza['week'] = pd.to_numeric(df_influenza['week'], errors='coerce')
    df_weather['year'] = pd.to_numeric(df_weather['year'], errors='coerce')
    df_weather['week'] = pd.to_numeric(df_weather['week'], errors='coerce')
    
    # LEFT JOIN (influenza_data 기준)
    print(f"   - 병합 방식: LEFT JOIN")
    print(f"   - 병합 키: year, week")
    
    df_merged = pd.merge(
        df_influenza,
        df_weather,
        on=['year', 'week'],
        how='left'
    )
    
    print(f"\n✅ 병합 완료:")
    print(f"   - influenza_data 행 수: {len(df_influenza):,}")
    print(f"   - weather_data 행 수: {len(df_weather):,}")
    print(f"   - 병합 후 행 수: {len(df_merged):,}")
    
    # 추가된 날씨 컬럼 확인
    new_cols = [c for c in df_weather.columns if c not in df_influenza.columns and c not in ['year', 'week']]
    if new_cols:
        print(f"   - 추가된 날씨 컬럼: {new_cols}")
        
        # 날씨 데이터 결측치 확인
        print(f"\n   📈 날씨 데이터 결측치:")
        for col in new_cols:
            null_count = df_merged[col].isna().sum()
            null_pct = (null_count / len(df_merged)) * 100
            print(f"      - {col}: {null_count:,}건 ({null_pct:.1f}%)")
    
    # 최종 컬럼 순서 정렬
    print(f"\n   📋 최종 컬럼 ({len(df_merged.columns)}개):")
    print(f"      {list(df_merged.columns)}")
    
    return df_merged


def save_to_postgres(db: TimeSeriesDB, df: pd.DataFrame, table_name: str = "merged_data", 
                     if_exists: str = "replace"):
    """병합된 데이터를 PostgreSQL에 저장"""
    print(f"\n💾 4. PostgreSQL에 저장")
    print("=" * 70)
    print(f"   - 테이블명: {table_name}")
    print(f"   - 저장 모드: {if_exists}")
    
    # 테이블 생성 (기존 테이블이 있으면 삭제)
    db.create_table_from_dataframe(df, table_name=table_name, if_exists=if_exists)
    
    # 데이터 삽입
    db.insert_dataframe(df, table_name=table_name, if_exists="append")
    
    print(f"\n✅ 저장 완료!")
    print(f"   - 테이블: {table_name}")
    print(f"   - 행 수: {len(df):,}")
    print(f"   - 컬럼 수: {len(df.columns)}")


def export_to_csv(df: pd.DataFrame, output_path: str = "merged_influenza_data.csv"):
    """병합된 데이터를 CSV로도 내보내기 (백업용)"""
    print(f"\n📁 5. CSV 백업 저장")
    print("=" * 70)
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ CSV 저장 완료: {output_path}")
    print(f"   - 크기: {Path(output_path).stat().st_size / 1024:.1f} KB")


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 70)
    print("🔄 PostgreSQL 데이터 병합 스크립트")
    print("=" * 70)
    print("influenza_data + weather_data → merged_data 테이블 생성")
    print("=" * 70)
    
    # TimeSeriesDB 인스턴스 생성 (환경변수에서 자동 로드)
    db = TimeSeriesDB()
    
    try:
        # PostgreSQL 연결
        db.connect()
        
        # 1. influenza_data 로드
        df_influenza = load_influenza_data(db)
        
        # 2. weather_data 로드
        df_weather = load_weather_data(db)
        
        if df_weather is None or df_weather.empty:
            print("\n⚠️  weather_data 테이블이 없습니다.")
            print("   influenza_data만 merged_data로 저장합니다.")
            df_merged = df_influenza
        else:
            # 3. 데이터 병합
            df_merged = merge_data(df_influenza, df_weather)
        
        # 4. PostgreSQL에 저장
        save_to_postgres(db, df_merged, table_name="merged_data", if_exists="replace")
        
        # 5. CSV 백업 (선택사항)
        export_to_csv(df_merged, output_path="merged_influenza_data.csv")
        
        print("\n" + "=" * 70)
        print("✅ 모든 작업 완료!")
        print("=" * 70)
        print(f"\n📊 생성된 테이블:")
        print(f"   - PostgreSQL: merged_data ({len(df_merged):,}행 × {len(df_merged.columns)}컬럼)")
        print(f"   - CSV 백업: merged_influenza_data.csv")
        print("\n💡 사용 방법:")
        print("   from database.db_utils import TimeSeriesDB")
        print("   db = TimeSeriesDB()")
        print("   db.connect()")
        print("   df = db.load_data(table_name='merged_data')")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # PostgreSQL 연결 종료
        db.close()
    
    return 0


if __name__ == "__main__":
    exit(main())
