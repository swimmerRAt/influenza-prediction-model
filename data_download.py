"""
인플루엔자 예측 모델을 위한 데이터 다운로드 스크립트
GFID API로부터 데이터를 다운로드하여 로컬에 저장
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import requests
import pandas as pd
from dotenv import load_dotenv
import warnings

# SSL 경고 무시
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =========================
# 데이터셋 ID 리스트 정의
# =========================
DATASET_IDS = [
    'ds_0101', 'ds_0102', 'ds_0103', 'ds_0104', 'ds_0105', 
    'ds_0106', 'ds_0107', 'ds_0108', 'ds_0109', 'ds_0110',
    'ds_0701', 'ds_0801', 'ds_0901'
]

# =========================
# 서버 과부하 방지 Rate Limiter
# =========================
class AdaptiveRateLimiter:
    """
    서버 상태를 모니터링하고 자동으로 요청 속도를 조절하는 클래스
    
    Features:
    - 응답 시간 기반 자동 딜레이 조절
    - Exponential backoff 재시도 로직
    - 에러율 모니터링
    - 서버 상태 기반 adaptive throttling
    """
    
    def __init__(self, 
                 initial_delay=1.0,      # 초기 요청 간 딜레이 (초)
                 max_delay=30.0,         # 최대 딜레이 (초)
                 min_delay=0.5,          # 최소 딜레이 (초)
                 backoff_factor=2.0,     # 백오프 증가율
                 max_retries=5,          # 최대 재시도 횟수
                 slow_threshold=5.0,     # 느린 응답 판단 기준 (초)
                 error_threshold=0.3):   # 에러율 임계값 (30%)
        
        self.current_delay = initial_delay
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.min_delay = min_delay
        self.backoff_factor = backoff_factor
        self.max_retries = max_retries
        self.slow_threshold = slow_threshold
        self.error_threshold = error_threshold
        
        # 통계 정보
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.recent_response_times = []  # 최근 10개 응답 시간
        self.consecutive_errors = 0
        
    def get_stats(self):
        """현재 통계 정보 반환"""
        avg_response_time = (self.total_response_time / self.request_count 
                            if self.request_count > 0 else 0)
        error_rate = (self.error_count / self.request_count 
                     if self.request_count > 0 else 0)
        
        return {
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': error_rate,
            'avg_response_time': avg_response_time,
            'current_delay': self.current_delay,
            'consecutive_errors': self.consecutive_errors
        }
    
    def print_stats(self):
        """통계 정보 출력"""
        stats = self.get_stats()
        print(f"\n📊 [Rate Limiter 통계]")
        print(f"   총 요청: {stats['request_count']}")
        print(f"   에러 발생: {stats['error_count']}")
        print(f"   에러율: {stats['error_rate']:.1%}")
        print(f"   평균 응답 시간: {stats['avg_response_time']:.2f}초")
        print(f"   현재 딜레이: {stats['current_delay']:.2f}초")
        print(f"   연속 에러: {stats['consecutive_errors']}회")
    
    def record_success(self, response_time):
        """성공한 요청 기록 및 딜레이 감소"""
        self.request_count += 1
        self.total_response_time += response_time
        self.recent_response_times.append(response_time)
        
        # 최근 10개만 유지
        if len(self.recent_response_times) > 10:
            self.recent_response_times.pop(0)
        
        # 연속 에러 카운터 리셋
        self.consecutive_errors = 0
        
        # 응답이 빠르면 딜레이 감소 (점진적 회복)
        if response_time < self.slow_threshold * 0.5:
            self.current_delay = max(self.min_delay, self.current_delay * 0.9)
            print(f"   ⚡ 빠른 응답 감지 → 딜레이 감소: {self.current_delay:.2f}초")
        # 응답이 느리면 딜레이 증가
        elif response_time > self.slow_threshold:
            old_delay = self.current_delay
            self.current_delay = min(self.max_delay, self.current_delay * 1.2)
            print(f"   🐢 느린 응답 감지 ({response_time:.2f}초) → 딜레이 증가: {old_delay:.2f}초 → {self.current_delay:.2f}초")
    
    def record_error(self, error_type="unknown"):
        """에러 발생 기록 및 딜레이 증가"""
        self.request_count += 1
        self.error_count += 1
        self.consecutive_errors += 1
        
        # 에러 발생 시 딜레이 증가 (exponential backoff)
        old_delay = self.current_delay
        self.current_delay = min(self.max_delay, 
                                self.current_delay * self.backoff_factor)
        
        print(f"   ⚠️ 에러 발생 ({error_type}) → 딜레이 증가: {old_delay:.2f}초 → {self.current_delay:.2f}초")
        
        # 에러율이 높으면 경고
        error_rate = self.error_count / self.request_count
        if error_rate > self.error_threshold:
            print(f"   🚨 높은 에러율 감지: {error_rate:.1%} (임계값: {self.error_threshold:.1%})")
    
    def wait(self):
        """다음 요청 전 대기"""
        if self.request_count > 0:  # 첫 요청은 대기 안 함
            print(f"   ⏳ 서버 보호를 위해 {self.current_delay:.2f}초 대기 중...")
            time.sleep(self.current_delay)
    
    def execute_with_retry(self, func, *args, **kwargs):
        """
        재시도 로직이 포함된 함수 실행
        
        Parameters:
        -----------
        func : callable
            실행할 함수
        *args, **kwargs
            함수에 전달할 인자
        
        Returns:
        --------
        함수 실행 결과
        """
        for attempt in range(self.max_retries):
            try:
                # 요청 전 대기
                self.wait()
                
                # 함수 실행 시간 측정
                start_time = time.time()
                result = func(*args, **kwargs)
                response_time = time.time() - start_time
                
                # 성공 기록
                self.record_success(response_time)
                print(f"   ✅ 요청 성공 (응답 시간: {response_time:.2f}초)")
                
                return result
                
            except requests.exceptions.Timeout as e:
                self.record_error("timeout")
                if attempt < self.max_retries - 1:
                    wait_time = self.current_delay * (self.backoff_factor ** attempt)
                    print(f"   🔄 타임아웃 발생 - {wait_time:.1f}초 후 재시도 ({attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ 최대 재시도 횟수 초과 - 타임아웃")
                    raise
                    
            except requests.exceptions.RequestException as e:
                self.record_error("connection")
                if attempt < self.max_retries - 1:
                    wait_time = self.current_delay * (self.backoff_factor ** attempt)
                    print(f"   🔄 연결 에러 - {wait_time:.1f}초 후 재시도 ({attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ 최대 재시도 횟수 초과 - 연결 에러")
                    raise
                    
            except Exception as e:
                self.record_error("other")
                # 일반 예외는 재시도하지 않음
                print(f"   ❌ 예상치 못한 에러: {str(e)}")
                raise


# =========================
# Keycloak 인증
# =========================
class KeycloakAuth:
    """Keycloak 인증 관리 클래스"""
    
    def __init__(self):
        self.server_url = os.getenv('SERVER_URL', 'https://keycloak.211.238.12.60.nip.io:8100')
        self.realm = os.getenv('REALM', 'gfid-api')
        self.client_id = os.getenv('CLIENT_ID')
        self.client_secret = os.getenv('CLIENT_SECRET')
        
        # 토큰 캐시
        self.cached = {
            'access_token': None,
            'expires_at': 0
        }
        
        if not all([self.server_url, self.realm, self.client_id]):
            print("⚠️ Missing Keycloak env vars. Check .env file")
    
    def fetch_token(self):
        """
        Keycloak 서버에서 토큰 발급
        .env에 ACCESS_TOKEN이 있으면 우선 사용
        """
        # .env에 수동으로 설정된 ACCESS_TOKEN이 있는지 확인
        manual_token = os.getenv('ACCESS_TOKEN')
        if manual_token:
            print("📌 .env 파일의 ACCESS_TOKEN 사용 (수동 설정)")
            now = int(time.time())
            # 수동 토큰은 만료 시간을 알 수 없으므로 1시간(3600초)로 가정
            self.cached['access_token'] = manual_token
            self.cached['expires_at'] = now + 3600
            return self.cached
        
        # Keycloak 토큰 엔드포인트 URL 생성
        token_url = f"{self.server_url.rstrip('/')}/realms/{self.realm}/protocol/openid-connect/token"
        
        # OAuth2 Client Credentials 방식으로 요청 파라미터 구성
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id
        }
        if self.client_secret:
            data['client_secret'] = self.client_secret
        
        print(f"🔐 Keycloak 서버에 토큰 요청 중...")
        print(f"   URL: {token_url}")
        
        try:
            # Keycloak 서버에 POST 요청
            response = requests.post(
                token_url,
                data=data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=60,
                verify=False  # SSL 인증서 검증 비활성화
            )
            
            if response.status_code == 200:
                # 응답에서 토큰 추출 및 캐시 저장
                token_data = response.json()
                now = int(time.time())
                self.cached['access_token'] = token_data.get('access_token')
                self.cached['expires_at'] = now + token_data.get('expires_in', 300)
                
                print(f"✅ 자동 토큰 발급 성공!")
                return self.cached
            else:
                print(f"❌ Keycloak token fetch failed: {response.status_code}")
                if response.text:
                    print(f"   Response: {response.text}")
                raise Exception(f"Keycloak token request failed with status {response.status_code}")
                
        except requests.exceptions.Timeout:
            print("❌ Keycloak token fetch timeout")
            print("💡 해결 방법: Postman에서 토큰을 받아 .env 파일에 ACCESS_TOKEN으로 추가하세요")
            raise Exception("Keycloak 서버 연결 타임아웃. .env에 ACCESS_TOKEN을 수동으로 설정하세요.")
        except requests.exceptions.RequestException as e:
            print(f"❌ Keycloak token fetch error: {str(e)}")
            print("💡 해결 방법: Postman에서 토큰을 받아 .env 파일에 ACCESS_TOKEN으로 추가하세요")
            raise Exception(f"Keycloak 연결 실패. .env에 ACCESS_TOKEN을 수동으로 설정하세요.")
    
    def get_token(self):
        """
        토큰 조회 - 캐시된 토큰 반환 또는 새로 발급
        """
        now = int(time.time())
        
        # 캐시된 토큰이 유효한지 확인 (만료 30초 전까지 유효)
        if self.cached['access_token'] and self.cached['expires_at'] - 30 > now:
            return self.cached['access_token']
        
        # 토큰이 없거나 만료되었으면 새로 발급
        self.fetch_token()
        return self.cached['access_token']


# =========================
# API 데이터 다운로드 함수
# =========================
def fetch_data_from_api(dsid: str, api_url: str = None) -> pd.DataFrame:
    """
    Node.js API 서버를 통해 단일 데이터셋을 가져오는 함수
    
    Parameters:
    -----------
    dsid : str
        데이터셋 ID
    api_url : str, optional
        API 서버 URL (기본값: http://localhost:3000)
    
    Returns:
    --------
    pd.DataFrame
        API로부터 가져온 데이터프레임
    """
    if api_url is None:
        api_url = os.getenv('API_URL', 'http://localhost:3000')
    
    print(f"   API URL: {api_url}")
    print(f"   Dataset ID: {dsid}")
    
    try:
        # API 서버에 데이터 다운로드 요청
        request_url = f"{api_url}/download"
        request_body = {"dsid": dsid}
        
        response = requests.post(
            request_url,
            json=request_body,
            timeout=300  # 5분 타임아웃
        )
        
        if response.status_code != 200:
            print(f"   ❌ API 요청 실패: {response.status_code}")
            print(f"   응답: {response.text}")
            raise Exception(f"API 요청 실패: {response.status_code}")
        
        result = response.json()
        
        if not result.get('ok'):
            print(f"   ❌ API 에러: {result.get('error', 'Unknown error')}")
            raise Exception(f"API 에러: {result.get('error')}")
        
        # 페이지 파일들에서 데이터 읽기
        page_files = result.get('result', {}).get('pageFiles', [])
        print(f"   받은 페이지 파일 수: {len(page_files)}")
        
        if not page_files:
            raise Exception("페이지 파일이 없습니다")
        
        # 모든 페이지의 데이터를 합치기
        all_data = []
        for page_file in page_files:
            with open(page_file, 'r', encoding='utf-8') as f:
                page_data = json.load(f)
                all_data.extend(page_data)
        
        print(f"   총 레코드 수: {len(all_data)}")
        
        # DataFrame으로 변환
        df = pd.DataFrame(all_data)
        print(f"   DataFrame 크기: {df.shape}")
        
        # 날짜 컬럼 자동 변환
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_columns:
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        return df
        
    except Exception as e:
        print(f"   ❌ 데이터 가져오기 실패: {str(e)}")
        raise


def download_all_datasets(dataset_ids: List[str] = None, api_url: str = None) -> pd.DataFrame:
    """
    여러 데이터셋을 다운로드하여 병합
    
    Parameters:
    -----------
    dataset_ids : List[str], optional
        다운로드할 데이터셋 ID 리스트 (기본값: DATASET_IDS)
    api_url : str, optional
        API 서버 URL
    
    Returns:
    --------
    pd.DataFrame
        병합된 데이터프레임
    """
    print("\n" + "=" * 60)
    print("🌐 데이터셋 다운로드 시작")
    print("=" * 60)
    
    if dataset_ids is None:
        dataset_ids = DATASET_IDS
    
    if api_url is None:
        api_url = os.getenv('API_URL', 'http://localhost:3000')
    
    print(f"📋 다운로드할 데이터셋 개수: {len(dataset_ids)}")
    print(f"   API 서버: {api_url}")
    
    # Rate Limiter 초기화
    rate_limiter = AdaptiveRateLimiter(
        initial_delay=float(os.getenv('RATE_LIMIT_INITIAL_DELAY', '1.0')),
        max_delay=float(os.getenv('RATE_LIMIT_MAX_DELAY', '30.0')),
        min_delay=float(os.getenv('RATE_LIMIT_MIN_DELAY', '0.5')),
        max_retries=int(os.getenv('RATE_LIMIT_MAX_RETRIES', '5'))
    )
    print(f"🛡️ Rate Limiter 활성화")
    
    all_dataframes = []
    
    for idx, dsid in enumerate(dataset_ids, 1):
        print(f"\n{'='*60}")
        print(f"📥 [{idx}/{len(dataset_ids)}] 데이터셋 다운로드: {dsid}")
        print(f"{'='*60}")
        
        try:
            # Rate Limiter를 사용하여 데이터 다운로드
            df_single = rate_limiter.execute_with_retry(
                fetch_data_from_api,
                dsid=dsid,
                api_url=api_url
            )
            
            if df_single is not None and not df_single.empty:
                # 데이터셋 ID를 컬럼으로 추가
                df_single['dataset_id'] = dsid
                all_dataframes.append(df_single)
                print(f"   ✅ {dsid} 다운로드 완료: {df_single.shape}")
            else:
                print(f"   ⚠️ {dsid} 데이터가 비어있음")
                
        except Exception as e:
            print(f"   ⚠️ {dsid} 다운로드 중 오류: {str(e)}")
            
            # 연속 에러가 많으면 중단 여부 확인
            if rate_limiter.consecutive_errors >= 3:
                print(f"\n🚨 연속 {rate_limiter.consecutive_errors}회 에러 발생!")
                user_input = input("계속 진행하시겠습니까? (y/n): ").lower()
                if user_input != 'y':
                    print("사용자 요청으로 중단합니다.")
                    break
            continue
    
    # 최종 통계 출력
    rate_limiter.print_stats()
    
    if not all_dataframes:
        raise ValueError("다운로드된 데이터셋이 없습니다!")
    
    print(f"\n{'='*60}")
    print(f"📊 데이터 병합 중...")
    print(f"{'='*60}")
    
    # 모든 데이터프레임 병합
    df_merged = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"✅ 병합 완료!")
    print(f"   - 다운로드된 데이터셋: {len(all_dataframes)}")
    print(f"   - 최종 데이터 크기: {df_merged.shape}")
    print(f"   - 컬럼: {list(df_merged.columns)}")
    print(f"="*60 + "\n")
    
    return df_merged


def save_data_to_local(df: pd.DataFrame, save_dir: str = None, filename: str = None) -> str:
    """
    DataFrame을 로컬에 저장
    
    Parameters:
    -----------
    df : pd.DataFrame
        저장할 데이터프레임
    save_dir : str, optional
        저장 디렉토리 (기본값: data/raw/{YYYY-MM-DD}/)
    filename : str, optional
        파일명 (기본값: raw_data.json)
    
    Returns:
    --------
    str
        저장된 파일 경로
    """
    print("\n" + "=" * 60)
    print("💾 데이터 로컬 저장")
    print("=" * 60)
    
    # 저장 디렉토리 설정
    if save_dir is None:
        data_dir = os.getenv('DATA_DIR', 'data')
        today = datetime.now().strftime('%Y-%m-%d')
        save_dir = Path(data_dir) / 'raw' / today
    else:
        save_dir = Path(save_dir)
    
    # 디렉토리 생성
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"   저장 디렉토리: {save_dir}")
    
    # 파일명 설정
    if filename is None:
        filename = 'raw_data.json'
    
    # 파일 경로
    file_path = save_dir / filename
    
    # JSON으로 저장
    print(f"   파일명: {filename}")
    print(f"   데이터 크기: {df.shape}")
    
    # DataFrame을 JSON으로 변환 (날짜 형식 처리)
    df.to_json(file_path, orient='records', date_format='iso', indent=2)
    
    print(f"✅ 저장 완료: {file_path}")
    print(f"   파일 크기: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 60 + "\n")
    
    return str(file_path)


def load_data_from_local(load_dir: str = None, filename: str = None) -> pd.DataFrame:
    """
    로컬에서 데이터 로드
    
    Parameters:
    -----------
    load_dir : str, optional
        로드 디렉토리 (기본값: .env의 LOAD_DATE 또는 최신)
    filename : str, optional
        파일명 (기본값: raw_data.json)
    
    Returns:
    --------
    pd.DataFrame
        로드된 데이터프레임
    """
    print("\n" + "=" * 60)
    print("📂 로컬 데이터 로드")
    print("=" * 60)
    
    # 로드 디렉토리 설정
    if load_dir is None:
        data_dir = Path(os.getenv('DATA_DIR', 'data'))
        load_date = os.getenv('LOAD_DATE')
        
        if load_date:
            load_dir = data_dir / 'raw' / load_date
        else:
            # 최신 날짜 폴더 찾기
            raw_dir = data_dir / 'raw'
            if raw_dir.exists():
                date_folders = sorted([d for d in raw_dir.iterdir() if d.is_dir()], reverse=True)
                if date_folders:
                    load_dir = date_folders[0]
                    print(f"   최신 데이터 폴더 사용: {load_dir.name}")
                else:
                    raise FileNotFoundError("data/raw/ 디렉토리에 데이터가 없습니다")
            else:
                raise FileNotFoundError("data/raw/ 디렉토리가 존재하지 않습니다")
    else:
        load_dir = Path(load_dir)
    
    print(f"   로드 디렉토리: {load_dir}")
    
    # 파일명 설정
    if filename is None:
        filename = 'raw_data.json'
    
    # 파일 경로
    file_path = load_dir / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    print(f"   파일명: {filename}")
    print(f"   파일 크기: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # JSON 파일 로드
    df = pd.read_json(file_path, orient='records')
    
    print(f"✅ 로드 완료: {df.shape}")
    print(f"   컬럼: {list(df.columns)}")
    print("=" * 60 + "\n")
    
    return df


# =========================
# 메인 실행
# =========================
def main():
    """메인 실행 함수"""
    print("\n" + "=" * 60)
    print("🚀 인플루엔자 데이터 다운로드 스크립트")
    print("=" * 60)
    
    # 환경변수 로드
    env_path = Path.cwd() / '.env'
    print(f"📂 .env 파일: {env_path}")
    print(f"   존재 여부: {env_path.exists()}")
    
    load_dotenv(env_path, verbose=True, override=True)
    
    # 환경변수 확인
    api_url = os.getenv('API_URL', 'http://localhost:3000')
    data_dir = os.getenv('DATA_DIR', 'data')
    
    print(f"\n⚙️  설정:")
    print(f"   API_URL: {api_url}")
    print(f"   DATA_DIR: {data_dir}")
    print("=" * 60)
    
    # 데이터 다운로드
    try:
        df = download_all_datasets(
            dataset_ids=DATASET_IDS,
            api_url=api_url
        )
        
        # 로컬에 저장
        saved_path = save_data_to_local(df, filename='raw_data.json')
        
        print(f"\n✅ 전체 프로세스 완료!")
        print(f"   다운로드된 데이터: {df.shape}")
        print(f"   저장 위치: {saved_path}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        raise


if __name__ == '__main__':
    main()
