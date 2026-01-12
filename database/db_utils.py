"""
DuckDB를 사용한 시계열 데이터 관리 유틸리티

DuckDB는 OLAP(분석) 워크로드에 최적화된 임베디드 데이터베이스로,
대용량 CSV 파일을 빠르게 처리하고 효율적으로 쿼리할 수 있습니다.

주요 기능:
- CSV를 DuckDB/Parquet으로 변환하여 저장 공간 절약 및 로딩 속도 향상
- SQL 쿼리를 통한 유연한 데이터 필터링
- Pandas와의 원활한 통합
- 메모리 효율적인 대용량 데이터 처리
- API를 통한 최신 데이터 자동 업데이트
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, List
import time
import os
import json
import requests
from dotenv import load_dotenv
import warnings

# 환경변수 로드
load_dotenv()

# SSL 경고 무시
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class TimeSeriesDB:
    """시계열 데이터를 DuckDB로 관리하는 클래스"""
    
    def __init__(self, db_path: str = "influenza_data.duckdb"):
        """
        Parameters:
        -----------
        db_path : str
            DuckDB 데이터베이스 파일 경로
        """
        self.db_path = Path(db_path)
        self.conn = None
        
    def connect(self):
        """데이터베이스 연결"""
        if self.conn is None:
            self.conn = duckdb.connect(str(self.db_path))
            print(f"✅ DuckDB 연결됨: {self.db_path}")
        return self.conn
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self.conn:
            self.conn.close()
            self.conn = None
            print("✅ DuckDB 연결 종료됨")
    
    def __enter__(self):
        """컨텍스트 매니저 진입"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료"""
        self.close()
    
    def import_csv_to_db(
        self, 
        csv_path: str, 
        table_name: str = "influenza_data",
        chunk_size: int = 100000,
        show_progress: bool = True
    ):
        """
        대용량 CSV를 DuckDB 테이블로 임포트
        
        Parameters:
        -----------
        csv_path : str
            CSV 파일 경로
        table_name : str
            생성할 테이블 이름
        chunk_size : int
            한 번에 읽을 행 수 (메모리 관리용)
        show_progress : bool
            진행 상황 표시 여부
        """
        self.connect()
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
        
        print(f"\n{'='*60}")
        print(f"📥 CSV → DuckDB 임포트 시작")
        print(f"{'='*60}")
        print(f"원본 파일: {csv_path.name}")
        print(f"테이블명: {table_name}")
        
        start_time = time.time()
        
        # DuckDB는 CSV를 직접 읽어서 테이블로 만들 수 있습니다 (매우 빠름)
        self.conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS 
            SELECT * FROM read_csv_auto('{csv_path}', 
                header=true,
                sample_size=100000
            )
        """)
        
        # 테이블 정보 확인
        row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ 임포트 완료!")
        print(f"   • 총 행 수: {row_count:,}")
        print(f"   • 소요 시간: {elapsed:.2f}초")
        print(f"   • 초당 처리: {row_count/elapsed:,.0f} 행/초")
        print(f"{'='*60}\n")
        
        # 데이터베이스 파일 크기 확인
        if self.db_path.exists():
            db_size_mb = self.db_path.stat().st_size / (1024 * 1024)
            csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
            compression_ratio = (1 - db_size_mb / csv_size_mb) * 100
            
            print(f"💾 저장 공간:")
            print(f"   • 원본 CSV: {csv_size_mb:.1f} MB")
            print(f"   • DuckDB: {db_size_mb:.1f} MB")
            print(f"   • 압축률: {compression_ratio:.1f}% 절약\n")
    
    def export_to_parquet(
        self, 
        table_name: str = "influenza_data",
        parquet_path: str = "influenza_data.parquet"
    ):
        """
        DuckDB 테이블을 Parquet 파일로 내보내기
        Parquet은 컬럼 기반 포맷으로 분석 쿼리에 매우 효율적
        
        Parameters:
        -----------
        table_name : str
            내보낼 테이블 이름
        parquet_path : str
            저장할 Parquet 파일 경로
        """
        self.connect()
        
        print(f"\n📤 Parquet 내보내기: {table_name} → {parquet_path}")
        start_time = time.time()
        
        self.conn.execute(f"""
            COPY {table_name} TO '{parquet_path}' 
            (FORMAT PARQUET, COMPRESSION 'ZSTD')
        """)
        
        elapsed = time.time() - start_time
        parquet_size_mb = Path(parquet_path).stat().st_size / (1024 * 1024)
        
        print(f"✅ 완료! ({elapsed:.2f}초)")
        print(f"   • 파일 크기: {parquet_size_mb:.1f} MB\n")
    
    def load_data(
        self,
        table_name: str = "influenza_data",
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        DuckDB 테이블에서 데이터를 Pandas DataFrame으로 로드
        
        Parameters:
        -----------
        table_name : str
            로드할 테이블 이름
        columns : List[str], optional
            로드할 컬럼 리스트 (None이면 모든 컬럼)
        where : str, optional
            WHERE 절 조건 (예: "season_norm >= 2020")
        limit : int, optional
            로드할 최대 행 수
        order_by : str, optional
            정렬 기준 (예: "date DESC")
        
        Returns:
        --------
        pd.DataFrame
            로드된 데이터
        """
        self.connect()
        
        # SQL 쿼리 구성 (컬럼명을 따옴표로 감싸서 공백/특수문자 처리)
        if columns is None:
            select_cols = "*"
        else:
            # 컬럼명을 따옴표로 감싸기
            select_cols = ", ".join([f'"{col}"' for col in columns])
        
        query = f"SELECT {select_cols} FROM {table_name}"
        
        if where:
            query += f" WHERE {where}"
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        print(f"\n📊 데이터 로드 중...")
        print(f"쿼리: {query[:100]}{'...' if len(query) > 100 else ''}")
        
        start_time = time.time()
        df = self.conn.execute(query).df()
        elapsed = time.time() - start_time
        
        print(f"✅ 로드 완료: {df.shape[0]:,} 행 × {df.shape[1]} 열 ({elapsed:.2f}초)\n")
        
        return df
    
    def get_table_info(self, table_name: str = "influenza_data"):
        """
        테이블 정보 출력 (컬럼명, 데이터 타입, 샘플 데이터)
        
        Parameters:
        -----------
        table_name : str
            확인할 테이블 이름
        """
        self.connect()
        
        print(f"\n{'='*60}")
        print(f"📋 테이블 정보: {table_name}")
        print(f"{'='*60}")
        
        # 테이블 스키마
        schema = self.conn.execute(f"DESCRIBE {table_name}").df()
        print("\n컬럼 정보:")
        print(schema.to_string(index=False))
        
        # 행 수
        row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"\n총 행 수: {row_count:,}")
        
        # 샘플 데이터
        print("\n샘플 데이터 (처음 5행):")
        sample = self.conn.execute(f"SELECT * FROM {table_name} LIMIT 5").df()
        print(sample.to_string())
        
        # 날짜 범위 (label 컬럼이 있다고 가정)
        try:
            date_range = self.conn.execute(f"""
                SELECT 
                    MIN(label) as start_period,
                    MAX(label) as end_period
                FROM {table_name}
            """).df()
            print("\n기간:")
            print(date_range.to_string(index=False))
        except:
            pass
        
        print(f"{'='*60}\n")
    
    def create_indices(self, table_name: str = "influenza_data"):
        """
        자주 쿼리하는 컬럼에 인덱스 생성 (쿼리 속도 향상)
        
        Parameters:
        -----------
        table_name : str
            인덱스를 생성할 테이블 이름
        """
        self.connect()
        
        print(f"\n🔍 인덱스 생성 중...")
        
        # DuckDB는 자동으로 쿼리 최적화를 하지만,
        # 명시적으로 인덱스를 생성할 수도 있습니다
        # 참고: DuckDB는 인메모리 최적화를 자동으로 수행
        
        print(f"✅ DuckDB는 자동 최적화를 사용합니다\n")
    
    def optimize_database(self):
        """데이터베이스 최적화 (VACUUM, ANALYZE)"""
        self.connect()
        
        print(f"\n🔧 데이터베이스 최적화 중...")
        
        # ANALYZE: 통계 정보 업데이트로 쿼리 최적화
        self.conn.execute("ANALYZE")
        
        print(f"✅ 최적화 완료\n")


def convert_csv_to_duckdb(
    csv_path: str,
    db_path: str = "influenza_data.duckdb",
    table_name: str = "influenza_data"
):
    """
    편의 함수: CSV를 DuckDB로 변환
    
    Parameters:
    -----------
    csv_path : str
        변환할 CSV 파일 경로
    db_path : str
        생성할 DuckDB 파일 경로
    table_name : str
        테이블 이름
    """
    with TimeSeriesDB(db_path) as db:
        db.import_csv_to_db(csv_path, table_name)
        db.get_table_info(table_name)
        db.optimize_database()
    
    print(f"✅ 변환 완료: {db_path}")
    return db_path


def load_from_duckdb(
    db_path: str = "influenza_data.duckdb",
    table_name: str = "influenza_data",
    **kwargs
) -> pd.DataFrame:
    """
    편의 함수: DuckDB에서 데이터 로드
    
    Parameters:
    -----------
    db_path : str
        DuckDB 파일 경로
    table_name : str
        로드할 테이블 이름
    **kwargs
        load_data() 함수에 전달할 추가 인자
    
    Returns:
    --------
    pd.DataFrame
        로드된 데이터
    """
    with TimeSeriesDB(db_path) as db:
        return db.load_data(table_name, **kwargs)


def fetch_latest_data_from_api(api_url: str = None, dataset_ids: List[str] = None) -> pd.DataFrame:
    """
    API를 통해 최신 데이터를 가져옴
    
    Parameters:
    -----------
    api_url : str, optional
        API 서버 URL (기본값: 환경변수 API_URL)
    dataset_ids : List[str], optional
        가져올 데이터셋 ID 리스트
    
    Returns:
    --------
    pd.DataFrame
        API에서 가져온 최신 데이터
    """
    if api_url is None:
        api_url = os.getenv('API_URL', 'http://localhost:3000')
    
    if dataset_ids is None:
        # 기본 데이터셋 ID 리스트
        dataset_ids = [
            'ds_0101', 'ds_0102', 'ds_0103', 'ds_0104', 'ds_0105', 
            'ds_0106', 'ds_0107', 'ds_0108', 'ds_0109', 'ds_0110',
            'ds_0701', 'ds_0801', 'ds_0901'
        ]
    
    print(f"\n{'='*60}")
    print(f"🌐 API에서 최신 데이터 가져오기")
    print(f"{'='*60}")
    print(f"API URL: {api_url}")
    print(f"데이터셋 개수: {len(dataset_ids)}")
    
    all_dataframes = []
    
    for idx, dsid in enumerate(dataset_ids, 1):
        print(f"\n[{idx}/{len(dataset_ids)}] {dsid} 로딩 중...")
        
        try:
            request_url = f"{api_url}/download"
            request_body = {"dsid": dsid, "returnData": True}  # 데이터를 직접 반환 요청
            
            response = requests.post(
                request_url,
                json=request_body,
                timeout=300
            )
            
            if response.status_code != 200:
                print(f"  ⚠️ API 요청 실패: {response.status_code}")
                continue
            
            result = response.json()
            if not result.get('ok'):
                print(f"  ⚠️ API 에러: {result.get('error')}")
                continue
            
            # API 응답에서 직접 데이터 가져오기 (파일 저장하지 않음)
            api_data = result.get('result', {}).get('data', [])
            
            if api_data:
                df = pd.DataFrame(api_data)
                df['dataset_id'] = dsid
                all_dataframes.append(df)
                print(f"  ✅ 완료: {len(api_data)} 레코드 (메모리에서 직접 처리)")
        
        except Exception as e:
            print(f"  ⚠️ 오류: {e}")
            continue
        
        # 서버 부하 방지를 위한 대기
        time.sleep(0.5)
    
    if not all_dataframes:
        print(f"\n⚠️ 가져온 데이터가 없습니다!")
        return pd.DataFrame()
    
    # 모든 데이터 병합
    df_latest = pd.concat(all_dataframes, ignore_index=True)
    print(f"\n✅ 최신 데이터 병합 완료: {df_latest.shape}")
    
    return df_latest


def load_historical_data(before_dir: str = 'data/before') -> pd.DataFrame:
    """
    과거 데이터 CSV 파일들을 로드하고 병합
    
    Parameters:
    -----------
    before_dir : str
        과거 데이터가 저장된 디렉토리
    
    Returns:
    --------
    pd.DataFrame
        병합된 과거 데이터
    """
    print(f"\n{'='*60}")
    print(f"📂 과거 데이터 로딩")
    print(f"{'='*60}")
    print(f"디렉토리: {before_dir}")
    
    before_path = Path(before_dir)
    if not before_path.exists():
        print(f"⚠️ 디렉토리가 없습니다: {before_dir}")
        return pd.DataFrame()
    
    csv_files = list(before_path.glob("*.csv"))
    print(f"발견된 CSV 파일: {len(csv_files)}개")
    
    if not csv_files:
        return pd.DataFrame()
    
    # 데이터셋 ID별로 분류
    data_by_dsid = {}
    
    for filepath in csv_files:
        filename = filepath.name
        # 파일명 파싱: flu-0105-2022.csv -> dsid=0105, year=2022
        parts = filename.replace('.csv', '').split('-')
        if len(parts) != 3:
            continue
        
        dsid = f"ds_{parts[1]}"
        
        try:
            df = pd.read_csv(filepath)
            if dsid not in data_by_dsid:
                data_by_dsid[dsid] = []
            data_by_dsid[dsid].append(df)
        except Exception as e:
            print(f"  ⚠️ 파일 읽기 실패 ({filename}): {e}")
    
    # 데이터셋별로 병합
    all_dataframes = []
    for dsid, df_list in data_by_dsid.items():
        df_combined = pd.concat(df_list, ignore_index=True)
        df_combined['dataset_id'] = dsid
        all_dataframes.append(df_combined)
        print(f"  {dsid}: {len(df_list)}개 파일, {len(df_combined)} 레코드")
    
    if not all_dataframes:
        return pd.DataFrame()
    
    df_historical = pd.concat(all_dataframes, ignore_index=True)
    print(f"\n✅ 과거 데이터 병합 완료: {df_historical.shape}")
    
    return df_historical


def merge_and_update_database(
    db_path: str = "influenza_data.duckdb",
    table_name: str = "influenza_data",
    fetch_latest: bool = True,
    api_url: str = None,
    before_dir: str = 'data/before'
):
    """
    1. API로 최신 데이터 가져오기
    2. 과거 데이터 로딩
    3. 데이터 병합
    4. DuckDB에 저장
    
    Parameters:
    -----------
    db_path : str
        DuckDB 데이터베이스 파일 경로
    table_name : str
        테이블 이름
    fetch_latest : bool
        API에서 최신 데이터를 가져올지 여부
    api_url : str, optional
        API 서버 URL
    before_dir : str
        과거 데이터 디렉토리
    """
    print("\n" + "="*60)
    print("🔄 데이터 병합 및 DuckDB 업데이트 프로세스")
    print("="*60)
    
    all_data = []
    
    # 1. API에서 최신 데이터 가져오기
    if fetch_latest:
        print("\n[단계 1/3] API에서 최신 데이터 가져오기")
        df_latest = fetch_latest_data_from_api(api_url=api_url)
        if not df_latest.empty:
            all_data.append(df_latest)
            print(f"✅ 최신 데이터: {df_latest.shape}")
        else:
            print("⚠️ 최신 데이터 없음")
    else:
        print("\n[단계 1/3] 최신 데이터 가져오기 건너뜀")
    
    # 2. 과거 데이터 로딩
    print("\n[단계 2/2] 과거 데이터 로딩")
    df_historical = load_historical_data(before_dir=before_dir)
    if not df_historical.empty:
        all_data.append(df_historical)
        print(f"✅ 과거 데이터: {df_historical.shape}")
    else:
        print("⚠️ 과거 데이터 없음")
    
    # 3. 모든 데이터 병합
    print("\n[단계 3/3] 데이터 병합 및 DuckDB 저장")
    if not all_data:
        print("⚠️ 병합할 데이터가 없습니다!")
        return
    
    df_merged = pd.concat(all_data, ignore_index=True)
    
    # 중복 제거 (year, week, dataset_id 기준)
    if all(['year' in df_merged.columns, 'week' in df_merged.columns]):
        print(f"중복 제거 전: {len(df_merged)} 행")
        df_merged = df_merged.drop_duplicates(
            subset=['year', 'week'] if 'dataset_id' not in df_merged.columns else ['year', 'week', 'dataset_id'],
            keep='last'
        )
        print(f"중복 제거 후: {len(df_merged)} 행")
    
    print(f"\n최종 병합 데이터: {df_merged.shape}")
    
    # DuckDB에 저장
    with TimeSeriesDB(db_path) as db:
        print(f"\nDuckDB에 저장 중...")
        start_time = time.time()
        
        # 기존 테이블 삭제 후 새로 생성
        db.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        db.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df_merged")
        
        elapsed = time.time() - start_time
        row_count = db.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        print(f"✅ DuckDB 저장 완료!")
        print(f"   • 테이블: {table_name}")
        print(f"   • 행 수: {row_count:,}")
        print(f"   • 소요 시간: {elapsed:.2f}초")
        
        db.optimize_database()
    
    # CSV로도 저장 (백업)
    csv_output_dir = Path("data/merged")
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    csv_output = csv_output_dir / "merged_influenza_data.csv"
    print(f"\nCSV 백업 저장 중: {csv_output}")
    df_merged.to_csv(csv_output, index=False)
    csv_size_mb = csv_output.stat().st_size / (1024 * 1024)
    print(f"✅ CSV 저장 완료: {csv_size_mb:.1f} MB")
    
    print("\n" + "="*60)
    print("✅ 모든 작업 완료!")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # 사용 예제
    print("=" * 60)
    print("DuckDB 시계열 데이터 관리 유틸리티")
    print("=" * 60)
    
    # 명령행 인자 확인
    if len(sys.argv) > 1 and sys.argv[1] == "--update":
        # 업데이트 모드: API + 과거 데이터 + 기존 데이터 병합
        print("\n🔄 업데이트 모드: API에서 최신 데이터 가져와서 병합")
        merge_and_update_database(
            db_path="influenza_data.duckdb",
            table_name="influenza_data",
            fetch_latest=True,
            before_dir='data/before'
        )
    else:
        # 기본 모드: 기존 CSV를 DuckDB로 변환
        csv_file = "data/merged/merged_influenza_data.csv"
        if Path(csv_file).exists():
            db_path = convert_csv_to_duckdb(
                csv_path=csv_file,
                db_path="influenza_data.duckdb",
                table_name="influenza_data"
            )
            
            print("\n" + "=" * 60)
            print("사용 예제:")
            print("=" * 60)
            print("""
# 전체 데이터 로드
df = load_from_duckdb()

# 특정 컬럼만 로드
df = load_from_duckdb(columns=['year', 'week', '의사환자 분율'])

# 조건부 로드
df = load_from_duckdb(where="year >= 2020")

# 최근 1000개 데이터만
df = load_from_duckdb(limit=1000, order_by="year DESC, week DESC")

# API + 과거 데이터 + 기존 데이터 병합하여 업데이트
python db_utils.py --update
            """)
        else:
            print(f"\n⚠️ CSV 파일을 찾을 수 없습니다: {csv_file}")
            print("\n💡 API에서 데이터를 가져와 DuckDB를 생성하려면:")
            print("   python db_utils.py --update")
