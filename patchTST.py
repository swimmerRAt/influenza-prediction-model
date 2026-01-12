import math
from pathlib import Path
from typing import List, Tuple, Optional
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
import os
from dotenv import load_dotenv
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# SSL 경고 무시
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =========================
# 데이터셋 ID 리스트 정의
# =========================
DATASET_IDS = [
    'ds_0101', 'ds_0102', 'ds_0103', 'ds_0104', 'ds_0105', 'ds_0106', 'ds_0107', 'ds_0108', 'ds_0109', 'ds_0110',
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

# 전역 Rate Limiter 인스턴스
_rate_limiter = None

def get_rate_limiter():
    """전역 AdaptiveRateLimiter 인스턴스 반환"""
    global _rate_limiter
    if _rate_limiter is None:
        # 환경변수에서 설정 가져오기
        initial_delay = float(os.getenv('RATE_LIMIT_INITIAL_DELAY', '1.0'))
        max_delay = float(os.getenv('RATE_LIMIT_MAX_DELAY', '30.0'))
        min_delay = float(os.getenv('RATE_LIMIT_MIN_DELAY', '0.5'))
        max_retries = int(os.getenv('RATE_LIMIT_MAX_RETRIES', '5'))
        
        _rate_limiter = AdaptiveRateLimiter(
            initial_delay=initial_delay,
            max_delay=max_delay,
            min_delay=min_delay,
            max_retries=max_retries
        )
        print(f"\n🛡️ Adaptive Rate Limiter 초기화")
        print(f"   초기 딜레이: {initial_delay}초")
        print(f"   최대 딜레이: {max_delay}초")
        print(f"   최소 딜레이: {min_delay}초")
        print(f"   최대 재시도: {max_retries}회")
    return _rate_limiter

# =========================
# Keycloak 인증 (auth.js와 동일한 구조)
# =========================
class KeycloakAuth:
    """Keycloak 인증 관리 클래스 (auth.js 구조를 Python으로 구현)"""
    
    def __init__(self):
        self.server_url = os.getenv('SERVER_URL', 'https://keycloak.211.238.12.60.nip.io:8100')
        self.realm = os.getenv('REALM', 'gfid-api')
        self.client_id = os.getenv('CLIENT_ID')
        self.client_secret = os.getenv('CLIENT_SECRET')
        
        # 토큰 캐시 (auth.js의 cached 객체와 동일)
        self.cached = {
            'access_token': None,
            'expires_at': 0
        }
        
        if not all([self.server_url, self.realm, self.client_id]):
            print("⚠️ Missing Keycloak env vars. Check .env file")
    
    def fetch_token(self):
        """
        Keycloak 서버에서 토큰 발급 (auth.js의 fetchToken()과 동일)
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
        
        # ① Keycloak 토큰 엔드포인트 URL 생성
        token_url = f"{self.server_url.rstrip('/')}/realms/{self.realm}/protocol/openid-connect/token"
        
        # ② OAuth2 Client Credentials 방식으로 요청 파라미터 구성
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id
        }
        if self.client_secret:
            data['client_secret'] = self.client_secret
        
        print(f"🔐 Keycloak 서버에 토큰 요청 중...")
        print(f"   URL: {token_url}")
        
        try:
            # ③ Keycloak 서버에 POST 요청
            response = requests.post(
                token_url,
                data=data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=60,
                verify=False  # SSL 인증서 검증 비활성화
            )
            
            if response.status_code == 200:
                # ④ 응답에서 토큰 추출 및 캐시 저장
                token_data = response.json()
                now = int(time.time())
                self.cached['access_token'] = token_data.get('access_token')
                self.cached['expires_at'] = now + token_data.get('expires_in', 300)
                
                print(f"✅ 자동 토큰 발급 성공!")
                return self.cached
            else:
                # ⑤ 에러 처리
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
        토큰 조회 - 캐시된 토큰 반환 또는 새로 발급 (auth.js의 getToken()과 동일)
        """
        now = int(time.time())
        
        # 캐시된 토큰이 유효한지 확인 (만료 30초 전까지 유효)
        if self.cached['access_token'] and self.cached['expires_at'] - 30 > now:
            return self.cached['access_token']
        
        # 토큰이 없거나 만료되었으면 새로 발급
        self.fetch_token()
        return self.cached['access_token']
    
    def get_token_info(self):
        """
        토큰 정보 조회 (auth.js의 getTokenInfo()와 동일)
        """
        now = int(time.time())
        return {
            'hasToken': bool(self.cached['access_token']),
            'expiresAt': self.cached['expires_at'],
            'secondsUntilExpiry': max(0, self.cached['expires_at'] - now)
        }


# 전역 인증 객체 생성
_auth = None

def get_auth():
    """전역 KeycloakAuth 인스턴스 반환"""
    global _auth
    if _auth is None:
        _auth = KeycloakAuth()
    return _auth


def fetch_data_directly_from_gfid(dsid=None, dsid_list=None):
    """
    Keycloak 인증 후 GFID API에서 직접 데이터를 가져오는 함수
    (gfidClient.js의 downloadDataset()과 유사한 구조)
    
    Parameters:
    -----------
    dsid : str, optional
        단일 데이터셋 ID (하나만 로드할 경우)
    dsid_list : list, optional
        여러 데이터셋 ID 리스트 (여러 개 로드할 경우, 기본값: DATASET_IDS)
    
    Returns:
    --------
    pd.DataFrame
        병합된 데이터프레임
    """
    print("\n" + "=" * 60)
    print("🌐 GFID API에서 직접 데이터 로딩 (Python 방식)")
    print("=" * 60)
    
    # 데이터셋 ID 설정
    # USE_SINGLE_DATASET=true일 때만 DSID 환경변수 사용, 아니면 전체 리스트 사용
    use_single = os.getenv('USE_SINGLE_DATASET', 'false').lower() == 'true'
    
    if dsid_list is None and dsid is None:
        if use_single:
            # 명시적으로 단일 데이터셋 사용 설정된 경우에만 DSID 환경변수 확인
            env_dsid = os.getenv('DSID')
            if env_dsid:
                dsid_list = [env_dsid]
                print(f"⚙️  USE_SINGLE_DATASET=true → 단일 데이터셋 모드")
            else:
                print(f"⚠️  USE_SINGLE_DATASET=true이지만 DSID 미설정 → 전체 데이터셋 사용")
                dsid_list = DATASET_IDS
        else:
            # 기본값: 전체 데이터셋 리스트 사용
            print(f"⚙️  기본 모드 → 전체 데이터셋 로드")
            dsid_list = DATASET_IDS
    elif dsid is not None:
        dsid_list = [dsid]
    
    print(f"📋 로드할 데이터셋 개수: {len(dsid_list)}")
    if len(dsid_list) <= 10:
        print(f"   데이터셋 리스트: {dsid_list}")
    else:
        print(f"   데이터셋 리스트: {dsid_list[:5]} ... {dsid_list[-5:]}")
    
    from_date = os.getenv('FROM', '2025-01-01')
    to_date = os.getenv('TO', '2025-12-31')
    
    print(f"   - 날짜 범위: {from_date} ~ {to_date}")
    
    # 여러 데이터셋 로딩 및 병합
    # Node.js API 서버(localhost:3000)를 통해 데이터 로드
    all_dataframes = []
    
    api_url = os.getenv('API_URL', 'http://localhost:3000')
    print(f"   API 서버: {api_url}")
    
    # Rate Limiter 활성화
    rate_limiter = get_rate_limiter()
    print(f"\n🛡️ Rate Limiter 활성화 - 서버 과부하 방지 모드")
    
    for idx, current_dsid in enumerate(dsid_list, 1):
        print(f"\n{'='*60}")
        print(f"📥 [{idx}/{len(dsid_list)}] 데이터셋 로딩: {current_dsid}")
        print(f"{'='*60}")
        
        try:
            # Rate Limiter를 사용하여 fetch_data_from_api() 호출
            df_single = rate_limiter.execute_with_retry(
                fetch_data_from_api, 
                dsid=current_dsid, 
                api_url=api_url,
                _skip_rate_limiter=True  # 내부 호출이므로 중복 적용 방지
            )
            
            if df_single is not None and not df_single.empty:
                # 데이터셋 ID를 컬럼으로 추가 (어떤 데이터셋에서 왔는지 추적)
                df_single['dataset_id'] = current_dsid
                all_dataframes.append(df_single)
                print(f"   ✅ {current_dsid} 로드 완료: {df_single.shape}")
            else:
                print(f"   ⚠️ {current_dsid} 데이터가 비어있거나 로드 실패")
        except Exception as e:
            print(f"   ⚠️ {current_dsid} 로드 중 오류 발생: {str(e)}")
            # 연속 에러가 많으면 중단 고려
            if rate_limiter.consecutive_errors >= 3:
                print(f"\n🚨 연속 {rate_limiter.consecutive_errors}회 에러 발생!")
                user_input = input("계속 진행하시겠습니까? (y/n): ").lower()
                if user_input != 'y':
                    print("사용자 요청으로 중단합니다.")
                    break
            continue
    
    # 최종 통계 출력
    rate_limiter.print_stats()
    
    # 모든 데이터프레임 병합
    if not all_dataframes:
        raise ValueError("로드된 데이터셋이 없습니다!")
    
    print(f"\n{'='*60}")
    print(f"📊 데이터 병합 중...")
    print(f"{'='*60}")
    
    # 모든 데이터를 하나로 병합 (행 방향 concatenation)
    df_merged = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"✅ 전체 병합 완료!")
    print(f"   - 로드된 데이터셋 개수: {len(all_dataframes)}")
    print(f"   - 최종 데이터 크기: {df_merged.shape}")
    print(f"   - 컬럼: {list(df_merged.columns)}")
    print(f"="*60 + "\n")
    
    return df_merged


def _fetch_single_dataset(dsid, from_date, to_date, access_token):
    """
    단일 데이터셋을 GFID API에서 가져오는 내부 함수
    
    Parameters:
    -----------
    dsid : str
        데이터셋 ID
    from_date : str
        시작 날짜
    to_date : str
        종료 날짜
    access_token : str
        Keycloak 액세스 토큰
    
    Returns:
    --------
    pd.DataFrame or None
        가져온 데이터프레임 또는 None (실패 시)
    """
    # GFID API 호출
    api_base = os.getenv('GFID_API_BASE', 'http://211.238.12.60:8084/data/api/v1')
    api_path = f"/etl_data/id/{dsid}/from/{from_date}/to/{to_date}"
    api_url = api_base + api_path
    
    print(f"   URL: {api_url}")
    
    headers = {'Authorization': f'Bearer {access_token}'}
    
    try:
        response = requests.get(api_url, headers=headers, timeout=300, verify=False)
        
        print(f"   응답 코드: {response.status_code}")
        
        if response.status_code != 200:
            print(f"   ⚠️ API 요청 실패! (상태 코드: {response.status_code})")
            print(f"   응답: {response.text[:500]}")
            return None  # 실패 시 None 반환
        
        result = response.json()
        
        # GFID_ITEMS_KEY 경로로 데이터 추출 (예: 'body.data')
        items_key = os.getenv('GFID_ITEMS_KEY', 'body.data')
        data = result
        for key in items_key.split('.'):
            data = data.get(key, {})
        
        if not data:
            print(f"   ⚠️ 데이터가 비어있습니다!")
            return None
        
        print(f"   - 받은 레코드 수: {len(data)}")
        
        # DataFrame으로 변환
        df = pd.DataFrame(data)
        print(f"   - DataFrame 크기: {df.shape}")
        
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
        print(f"   ⚠️ 데이터 가져오기 실패: {str(e)}")
        return None

# =========================
# 환경 변수 로드 (디버깅 로그 추가)
# =========================
print("=" * 60)
print("🔍 환경변수 로드 과정 디버깅")
print("=" * 60)

# .env 파일 경로 확인
env_path = Path.cwd() / '.env'
print(f"1. 현재 작업 디렉토리: {Path.cwd()}")
print(f"2. .env 파일 경로: {env_path}")
print(f"3. .env 파일 존재 여부: {env_path.exists()}")

# .env 파일 로드
load_result = load_dotenv(env_path, verbose=True, override=True)
print(f"4. .env 로드 결과: {load_result}")

# 환경변수 확인
use_api_raw = os.getenv('USE_API')
use_api_bool = os.getenv('USE_API', 'false').lower() == 'true'
dsid = os.getenv('DSID')
api_url = os.getenv('API_URL')

print(f"\n📋 환경변수 값:")
print(f"   - USE_API (원본): '{use_api_raw}'")
print(f"   - USE_API (boolean): {use_api_bool}")
print(f"   - DSID: '{dsid}'")
print(f"   - API_URL: '{api_url}'")
print("=" * 60 + "\n")

# =========================
# Paths & device
# =========================
BASE_DIR = Path.cwd()
# 우선순위로 탐색 (새 파일 -> 구 파일들)
CANDIDATE_CSVS = [
    BASE_DIR / "suyeong/3_merged_influenza_vaccine_respiratory_weather.csv",
]

# =========================
# API 데이터 로딩 함수
# =========================
def fetch_data_from_api(dsid=None, api_url=None, _skip_rate_limiter=False):
    """
    Node.js API 서버를 통해 데이터를 가져오는 함수
    
    Parameters:
    -----------
    dsid : str, optional
        데이터셋 ID (기본값은 환경변수 DSID 사용)
    api_url : str, optional
        API 서버 URL (기본값: http://localhost:3000)
    _skip_rate_limiter : bool, optional
        Rate Limiter 중복 적용 방지용 (내부용)
    
    Returns:
    --------
    pd.DataFrame
        API로부터 가져온 데이터를 DataFrame으로 변환
    """
    print("\n" + "=" * 60)
    print("🌐 API 데이터 로딩 시작 (Node.js 서버 경유)")
    print("=" * 60)
    
    # API 서버 URL 설정
    if api_url is None:
        api_url = os.getenv('API_URL', 'http://localhost:3000')
    print(f"1. API URL: {api_url}")
    
    # 데이터셋 ID 설정
    if dsid is None:
        dsid = os.getenv('DSID')
    print(f"2. Dataset ID: {dsid}")
    
    if not dsid:
        raise ValueError("dsid가 제공되지 않았습니다. 환경변수 DSID를 설정하거나 인자로 전달하세요.")
    
    print(f"3. API에서 데이터 가져오는 중... (dsid: {dsid})")
    
    try:
        # API 서버에 데이터 다운로드 요청
        request_url = f"{api_url}/download"
        request_body = {"dsid": dsid}
        print(f"4. 요청 URL: {request_url}")
        print(f"5. 요청 Body: {request_body}")
        
        response = requests.post(
            request_url,
            json=request_body,
            timeout=300  # 5분 타임아웃
        )
        
        print(f"6. 응답 상태 코드: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ API 요청 실패!")
            print(f"   응답 내용: {response.text}")
            raise Exception(f"API 요청 실패: {response.status_code} - {response.text}")
        
        result = response.json()
        print(f"7. 응답 성공 여부: {result.get('ok')}")
        
        if not result.get('ok'):
            print(f"❌ API 에러 발생!")
            print(f"   에러 메시지: {result.get('error', 'Unknown error')}")
            raise Exception(f"API 에러: {result.get('error', 'Unknown error')}")
        
        # 페이지 파일들에서 데이터 읽기
        page_files = result.get('result', {}).get('pageFiles', [])
        print(f"8. 받은 페이지 파일 수: {len(page_files)}")
        
        if not page_files:
            print(f"❌ 페이지 파일이 없습니다!")
            raise Exception("API로부터 받은 데이터 파일이 없습니다.")
        
        print(f"9. 페이지 파일 목록:")
        for i, pf in enumerate(page_files, 1):
            print(f"   {i}. {pf}")
        
        # 모든 페이지의 데이터를 합치기
        all_data = []
        for idx, page_file in enumerate(page_files, 1):
            print(f"10-{idx}. 파일 읽는 중: {page_file}")
            with open(page_file, 'r', encoding='utf-8') as f:
                page_data = json.load(f)
                print(f"      레코드 수: {len(page_data)}")
                all_data.extend(page_data)
        
        print(f"11. 총 레코드 수: {len(all_data)}")
        
        # DataFrame으로 변환
        df = pd.DataFrame(all_data)
        print(f"12. DataFrame 생성 완료: {df.shape}")
        print(f"13. 컬럼 목록: {list(df.columns)}")
        
        # 날짜 컬럼이 있으면 datetime으로 변환
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_columns:
            print(f"14. 날짜 컬럼 발견: {date_columns}")
            for col in date_columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    print(f"    ✅ {col} → datetime 변환 완료")
                except Exception as e:
                    print(f"    ⚠️ {col} → datetime 변환 실패: {e}")
        
        print(f"✅ API 데이터 로딩 완료!")
        print("=" * 60 + "\n")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API 서버 연결 실패!")
        print(f"   에러: {str(e)}")
        print("=" * 60 + "\n")
        raise Exception(f"API 서버 연결 실패: {str(e)}. API 서버가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"❌ 데이터 가져오기 실패!")
        print(f"   에러: {str(e)}")
        print("=" * 60 + "\n")
        raise Exception(f"데이터 가져오기 실패: {str(e)}")


def load_data_from_api_or_csv(use_api=None, dsid=None, csv_path=None):
    """
    API 또는 로컬 CSV 파일에서 데이터를 로드하는 통합 함수
    
    Parameters:
    -----------
    use_api : bool, optional
        True면 API 사용, False면 CSV 파일 사용 (기본값: 환경변수 USE_API)
    dsid : str, optional
        API 사용 시 데이터셋 ID
    csv_path : Path, optional
        CSV 사용 시 파일 경로
    
    Returns:
    --------
    pd.DataFrame
        로드된 데이터
    """
    # 환경변수에서 USE_API 설정 확인
    if use_api is None:
        use_api = os.getenv('USE_API', 'false').lower() == 'true'
    
    print(f"\n📊 데이터 로드 모드 결정: use_api={use_api}")
    
    if use_api:
        print("=" * 50)
        print("🌐 API 모드: 서버에서 데이터를 가져옵니다...")
        print("=" * 50)
        df = fetch_data_from_api(dsid=dsid)
        print(f"✅ API로부터 데이터 로드 완료: {df.shape}")
        return df
    else:
        print("=" * 50)
        print("📁 CSV 모드: 로컬 파일에서 데이터를 로드합니다...")
        print("=" * 50)
        if csv_path is None:
            csv_path = pick_csv_path()
        df = pd.read_csv(csv_path)
        print(f"✅ CSV 파일 로드 완료: {csv_path}, {df.shape}")
        return df

def pick_csv_path():
    for p in CANDIDATE_CSVS:
        if p.exists():
            return p
    raise FileNotFoundError("No input CSV found among:\n" + "\n".join(map(str, CANDIDATE_CSVS)))

# CSV_PATH는 필요할 때만 설정 (API 모드가 아닐 때만)
print("\n" + "=" * 60)
print("📂 CSV 파일 경로 설정")
print("=" * 60)

USE_API_MODE = os.getenv('USE_API', 'false').lower() == 'true'
print(f"USE_API_MODE 결정: {USE_API_MODE}")
print(f"   - 원본 환경변수 값: '{os.getenv('USE_API')}'")
print(f"   - 소문자 변환: '{os.getenv('USE_API', 'false').lower()}'")
print(f"   - 'true' 비교 결과: {os.getenv('USE_API', 'false').lower() == 'true'}")

if not USE_API_MODE:
    print("\n➡️ CSV 모드 - CSV 파일을 찾습니다...")
    try:
        CSV_PATH = pick_csv_path()
        print(f"✅ CSV 파일 발견: {CSV_PATH.name}")
    except FileNotFoundError as e:
        print(f"⚠️ CSV 파일을 찾을 수 없습니다.")
        print(f"   검색한 경로: {CANDIDATE_CSVS}")
        print(f"💡 API 모드를 사용하려면 .env에서 USE_API=true로 설정하세요.")
        CSV_PATH = None
else:
    print("\n➡️ API 모드 - CSV 파일 검색을 생략합니다")
    CSV_PATH = None
    print("🌐 API 모드 활성화됨 - CSV 파일 검색 생략")

print("=" * 60 + "\n")

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = pick_device()
SEED   = 42

print(f"🖥️ 선택된 디바이스: {DEVICE}")
print(f"🎲 랜덤 시드: {SEED}\n")


# =========================
# Hyperparameters
# =========================
EPOCHS      = 100
BATCH_SIZE  = 64        # 소규모 시계열에서도 안정적으로 학습되도록 약간 낮춤
SEQ_LEN     = 12
PRED_LEN    = 3
PATCH_LEN   = 4          # ← CNN이 최소 3~5 커널 적용 가능하도록 확대
STRIDE      = 1

D_MODEL     = 128        # 4의 배수(멀티스케일 분기 4개 합산)
N_HEADS     = 2
ENC_LAYERS  = 4
FF_DIM      = 128
DROPOUT     = 0.3        # 약간 강화
HEAD_HIDDEN = [64, 64]

LR              = 5e-4
WEIGHT_DECAY    = 5e-4
PATIENCE        = 60
WARMUP_EPOCHS   = 30

SCALER_TYPE     = "robust"   # 노이즈/꼬리값 대응에 유리 (원하면 "standard"로 변경)

# 외생 특징 사용 모드: "auto"|"none"|"vax"|"resp"|"both"
USE_EXOG        = "all"

OUT_CSV          = str(BASE_DIR / "ili_predictions.csv")
PLOT_LAST_WINDOW = str(BASE_DIR / "plot_last_window.png")
PLOT_TEST_RECON  = str(BASE_DIR / "plot_test_reconstruction.png")
PLOT_MA_CURVES   = str(BASE_DIR / "plot_ma_curves.png")

# overlap 재구성 가중치 (t+1을 조금 더 신뢰)
RECON_W_START, RECON_W_END = 2.0, 0.5

# --- Feature switches ---
INCLUDE_SEASONAL_FEATS = True   # week_sin, week_cos를 입력 피처에 포함할지

# =========================
# utils
# =========================
from datetime import date

def _iso_weeks_in_year(y: int) -> int:
    # ISO 달력의 마지막 주 번호(52 또는 53)
    return date(y, 12, 28).isocalendar().week

def weekly_to_daily_interp(
    df: pd.DataFrame,
    season_col: str = "season_norm",
    week_col: str = "week",
    target_col: str = "ili",
) -> pd.DataFrame:
    """
    주 단위 데이터를 일 단위로 확장(선형보간). season/week 없으면 label에서 추출하거나,
    최후에는 연속 주차를 생성해 보간합니다.
    반환: date 컬럼 포함한 일 단위 DF
    """
    df = df.copy()
    df.columns = df.columns.str.replace("\ufeff", "", regex=True).str.strip()

    # --- 시즌/주차 확보 ---
    has_season = season_col in df.columns
    has_week   = week_col in df.columns

    if not (has_season and has_week):
        # label에서 시즌/주차 추출 시도: "2024-2025 season - W29"
        if "label" in df.columns:
            import re
            def _parse_label(lbl):
                m = re.search(r"(\d{4}-\d{4}).*W\s*([0-9]+)", str(lbl))
                if m:
                    return m.group(1), int(m.group(2))
                return None
            parsed = df["label"].map(_parse_label)
            if not has_season:
                df[season_col] = [p[0] if p else np.nan for p in parsed]
                has_season = True
            if not has_week:
                df[week_col] = [p[1] if p else np.nan for p in parsed]
                has_week = True

    # 최후의 수단: season_norm이 없으면 단일 시즌으로, week 없으면 1..N
    if not has_season:
        # 첫 행의 연도를 찾아 대체 시즌명 만들기
        # 없으면 "0000-0001"
        first_year = None
        if "date" in df.columns:
            try:
                first_year = pd.to_datetime(df["date"]).dt.year.min()
            except Exception:
                pass
        if first_year is None:
            first_year = pd.Timestamp.today().year
        df[season_col] = f"{first_year}-{first_year+1}"
        has_season = True

    if not has_week:
        df[week_col] = np.arange(1, len(df) + 1, dtype=int)
        has_week = True

    # 숫자화
    df[week_col] = pd.to_numeric(df[week_col], errors="coerce")
    # 시즌 문자열 정규화
    def _norm_season_text_local(s: str) -> str:
        ss = str(s).replace("절기", "")
        import re
        m = re.search(r"(\d{4})\s*-\s*(\d{4})", ss)
        return f"{m.group(1)}-{m.group(2)}" if m else ss.strip()
    df[season_col] = df[season_col].astype(str).map(_norm_season_text_local)

    # --- ISO 주 시작일 산출 (시즌 규칙 반영) ---
    week_starts = []
    for _, row in df.iterrows():
        season = str(row[season_col])
        try:
            y0 = int(season.split("-")[0])
        except Exception:
            y0 = pd.Timestamp.today().year
        wk = int(row[week_col]) if not pd.isna(row[week_col]) else 1
        iso_year = y0 if wk >= 36 else (y0 + 1)
        # 해당 ISO년의 실제 마지막 주 넘지 않도록 보정
        wk = min(max(1, wk), _iso_weeks_in_year(iso_year))
        # 월요일(1) 기준 주 시작일
        week_starts.append(pd.Timestamp.fromisocalendar(iso_year, wk, 1))
    df["week_start"] = week_starts

    # --- 중복 week_start 처리: 수치=mean, 비수치=first ---
    if df["week_start"].duplicated().any():
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        agg = {c: "mean" for c in num_cols}
        # 비수치 컬럼(라벨/시즌 등)은 첫 값 유지
        for c in df.columns:
            if c not in num_cols and c != "week_start":
                agg[c] = "first"
        df = df.groupby("week_start", as_index=False).agg(agg)

    # --- 일 단위 리샘플 ---
    df = df.set_index("week_start").sort_index()
    df_daily = df.resample("D").asfreq()

    # 수치형은 선형보간
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        df_daily[c] = df_daily[c].interpolate(method="linear", limit_direction="both")

    # 범주형은 앞뒤 채움
    cat_cols = [c for c in df.columns if c not in num_cols]
    for c in cat_cols:
        df_daily[c] = df_daily[c].ffill().bfill()

    # 결과
    out = df_daily.reset_index().rename(columns={"week_start": "date"})
    # date는 datetime으로 강제
    out["date"] = pd.to_datetime(out["date"])
    return out
    
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_csv_kor(path: Path) -> pd.DataFrame:
    for enc in ["euc-kr", "cp949", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path, encoding="utf-8", errors="replace")

def make_splits(n: int, train_ratio=0.7, val_ratio=0.15):
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return (0, n_train), (n_train, n_train+n_val), (n_train+n_val, n)

def get_scaler(name=None):
    s = (name or SCALER_TYPE).lower()
    if s == "robust":  return RobustScaler()
    if s == "minmax":  return MinMaxScaler()
    return StandardScaler()

def _norm_season_text(s: str) -> str:
    ss = str(s).replace("절기", "")
    import re
    m = re.search(r"(\d{4})\s*-\s*(\d{4})", ss)
    return f"{m.group(1)}-{m.group(2)}" if m else ss.strip()

# =========================
# data loader (multivariate-ready)
# =========================
def load_and_prepare(csv_path: Path = None, use_exog: str = "auto", df: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    Returns:
        X: (N, F) features (first column should be 'ili' to align with univariate fallback)
        y: (N,) target (ili)
        labels: list[str] for plotting ticks
        used_feat_names: list[str] feature column names (len=F)
    
    Parameters:
        csv_path: CSV 파일 경로 (df가 None일 때 사용)
        use_exog: 외생변수 사용 모드
        df: 이미 로드된 DataFrame (API에서 가져온 경우)
    """
    if df is None:
        if csv_path is None:
            raise ValueError("csv_path와 df 중 하나는 반드시 제공되어야 합니다.")
        df = read_csv_kor(csv_path).copy()
    else:
        df = df.copy()
    df = weekly_to_daily_interp(df, season_col="season_norm", week_col="week", target_col="ili")
    # 정렬
# 정렬: 주→일 변환 후에는 date 기준으로만 정렬
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)
    else:
        # (극히 드문 fallback) date가 없을 때만 기존 로직
        if {"season_norm", "week"}.issubset(df.columns):
            df["season_norm"] = df["season_norm"].astype(str).map(_norm_season_text)
            df["week"] = pd.to_numeric(df["week"], errors="coerce")
            df = df.sort_values(["season_norm", "week"]).copy()
        elif "label" in df.columns:
            df = df.sort_values(["label"]).copy()

    # 타깃
    if "ili" not in df.columns:
        raise ValueError("CSV에 'ili' 컬럼이 없습니다.")
    df["ili"] = pd.to_numeric(df["ili"], errors="coerce")
    if df["ili"].isna().any():
        df["ili"] = df["ili"].interpolate(method="linear", limit_direction="both").fillna(df["ili"].median())
    
    # --- ✅ Seasonality feature 추가 ---
    if "week" in df.columns:
        df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52.0)
        df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52.0)
    else:
        df["week_sin"] = 0.0
        df["week_cos"] = 0.0

    # --- ✅ Alias 매핑 ---
    if "case_count" in df.columns and "respiratory_index" not in df.columns:
        df["respiratory_index"] = df["case_count"]

    # 기후 피처 후보
    climate_feats = []
    if "wx_week_avg_temp" in df.columns:     climate_feats.append("wx_week_avg_temp")
    if "wx_week_avg_rain" in df.columns:     climate_feats.append("wx_week_avg_rain")
    if "wx_week_avg_humidity" in df.columns: climate_feats.append("wx_week_avg_humidity")

    # 외생 후보 존재 여부
    has_vax  = "vaccine_rate" in df.columns
    has_resp = "respiratory_index" in df.columns

    # 어떤 특징을 쓸지 결정
    mode = use_exog.lower()
    if mode == "auto":
        chosen = ["ili"]
        if has_vax:  chosen.append("vaccine_rate")
        if has_resp: chosen.append("respiratory_index")
        chosen += climate_feats
    elif mode == "none":
        chosen = ["ili"]
    elif mode == "vax":
        chosen = ["ili"] + (["vaccine_rate"] if has_vax else [])
    elif mode == "resp":
        chosen = ["ili"] + (["respiratory_index"] if has_resp else [])
    elif mode == "both":
        chosen = ["ili"]
        if has_vax:  chosen.append("vaccine_rate")
        if has_resp: chosen.append("respiratory_index")
        chosen += climate_feats
    elif mode == "climate":
        chosen = ["ili"] + climate_feats
    elif mode == "all":
        chosen = ["ili"]
        if has_vax:  chosen.append("vaccine_rate")
        if has_resp: chosen.append("respiratory_index")
        chosen += climate_feats
    else:
        raise ValueError(f"Unknown USE_EXOG mode: {use_exog}")

    # 숫자화 & 보간
    for c in chosen:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            df[c] = df[c].interpolate(method="linear", limit_direction="both").fillna(df[c].median())

    # 라벨
    if "label" in df.columns and df["label"].notna().any():
        labels = df["label"].astype(str).tolist()
    elif {"season_norm","week"}.issubset(df.columns):
        labels = (df["season_norm"].astype(str) + " season - W" + df["week"].astype(int).astype(str)).tolist()
    else:
        labels = [f"idx_{i}" for i in range(len(df))]

    # X, y 구성
    feat_names = chosen[:]
    if INCLUDE_SEASONAL_FEATS and {"week_sin", "week_cos"}.issubset(df.columns):
        feat_names += ["week_sin", "week_cos"]

    # 선택된 입력 피처 로그 찍기
    print("[Data] Exogenous detected -> vaccine_rate:", has_vax, "| respiratory_index:", has_resp, "| climate_feats:", climate_feats)
    print("[Data] Selected feature columns (order) ->", feat_names)

    X = df[feat_names].to_numpy(dtype=float)
    y = df["ili"].to_numpy(dtype=float)
    return X, y, labels, feat_names

# =========================
# dataset
# =========================
class PatchTSTDataset(Dataset):
    """Multivariate X (N,F) + y (N,) -> (patchified) windows."""
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len:int, pred_len:int, patch_len:int, stride:int):
        assert len(X) == len(y)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len, self.pred_len = seq_len, pred_len
        self.patch_len, self.stride = patch_len, stride
        max_start = len(self.y) - (seq_len + pred_len)
        self.indices = list(range(max(0, max_start + 1)))

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        seq_X = self.X[i:i+self.seq_len, :]                      # (L, F)
        tgt_y = self.y[i+self.seq_len:i+self.seq_len+self.pred_len]  # (H,)

        # patchify along time axis
        patches = []
        pos = 0
        while pos + self.patch_len <= self.seq_len:
            patches.append(seq_X[pos:pos+self.patch_len, :])     # (patch_len, F)
            pos += self.stride
        X_patch = np.stack(patches, axis=0)                      # (P, patch_len, F)
        return torch.from_numpy(X_patch).float(), torch.from_numpy(tgt_y).float(), i

# =========================
# model (Multi-Scale CNN + TokenConvMixer + PatchTST + AttnPool)
# =========================
class MultiScaleCNNPatchEmbed(nn.Module):
    """
    (B, P, L, F) -> [각 패치] 멀티스케일 Conv1d 분기(k=2/3/5, 또 하나는 dilation=2) → GAP → (B, P, D)
    - 분기 4개 출력 concat → D_MODEL
    - 패치 내부의 급격/완만/잔진동 패턴을 동시에 포착
    """
    def __init__(self, in_features: int, patch_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % 4 == 0, "d_model은 4의 배수가 되어야 멀티스케일 분기 합산이 맞습니다."
        out_ch = d_model // 4
    # 커널 크기를 patch_len에 비례하게 설정
        self.b2 = nn.Conv1d(in_features, out_ch, kernel_size=1, padding=0)
        self.b3 = nn.Conv1d(in_features, out_ch, kernel_size=3, padding=1)
        self.b5 = nn.Conv1d(in_features, out_ch, kernel_size=5, padding=2)
        self.bd = nn.Conv1d(in_features, out_ch, kernel_size=3, padding=2, dilation=2)

        self.bn   = nn.BatchNorm1d(d_model)
        self.act  = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)   # (B*P, D, L) → (B*P, D, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, P, L, F)
        B, P, L, F = x.shape
        x = x.view(B*P, L, F).permute(0, 2, 1)        # (B*P, F, L)

        z = torch.cat([self.b2(x), self.b3(x), self.b5(x), self.bd(x)], dim=1)  # (B*P, D, L)
        z = self.act(self.bn(z))
        z = self.pool(z).squeeze(-1)                  # (B*P, D)
        z = self.drop(z)
        return z.view(B, P, -1)                       # (B, P, D)

class TokenConvMixer(nn.Module):
    """
    패치 토큰 간(P 축) 로컬 연속성 강화: DepthwiseConv1d(P-축) + PointwiseConv1d
    입력/출력: (B, P, D)
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, z):              # (B, P, D)
        y = z.permute(0, 2, 1)         # (B, D, P)
        y = self.dw(y)
        y = self.pw(y)
        y = self.bn(y)
        y = self.act(y)
        y = self.drop(y)
        y = y.permute(0, 2, 1)         # (B, P, D)
        return z + y                   # Residual

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div)
        if d_model % 2 == 1:
            pe[:,1::2] = torch.cos(position*div)[:, :pe[:,1::2].shape[1]]
        else:
            pe[:,1::2] = torch.cos(position*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        P = x.size(1)
        return x + self.pe[:, :P, :]

class AttnPool(nn.Module):
    """Learnable-query attention pooling over patch tokens."""
    def __init__(self, d_model:int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model))
        self.proj = nn.Linear(d_model, d_model, bias=False)
    def forward(self, z):           # z: (B, P, D)
        B,P,D = z.shape
        q = self.q.expand(B, -1, -1)                       # (B,1,D)
        k = self.proj(z)                                   # (B,P,D)
        attn = torch.softmax((q @ k.transpose(1,2)) / (D**0.5), dim=-1)  # (B,1,P)
        pooled = attn @ z                                  # (B,1,D)
        return pooled.squeeze(1)                           # (B,D)

class PatchTSTModel(nn.Module):
    def __init__(self, in_features:int, patch_len:int, d_model:int, n_heads:int,
                 n_layers:int, ff_dim:int, dropout:float, pred_len:int, head_hidden:List[int]):
        super().__init__()
        # ① 멀티스케일 CNN 패치 임베딩
        self.embed = MultiScaleCNNPatchEmbed(in_features, patch_len, d_model, dropout=dropout*0.5)
        # ② 패치 토큰 간 로컬 연속성 믹서
        self.mixer = nn.Sequential(
            TokenConvMixer(d_model, dropout=dropout),
            TokenConvMixer(d_model, dropout=dropout),
        )
        # ③ PatchTST 인코더
        self.posenc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pool = AttnPool(d_model)

        # ④ 예측 헤드
        mlp, in_dim = [], d_model
        for h in head_hidden[:2]:
            mlp += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        mlp.append(nn.Linear(in_dim, pred_len))
        self.head = nn.Sequential(*mlp)

    def forward(self, x):
        # x: (B, P, L, F)
        z = self.embed(x)      # (B,P,D)
        z = self.mixer(z)      # (B,P,D)
        z = self.posenc(z)
        z = self.encoder(z)
        z = self.pool(z)       # (B,D)
        return self.head(z)    # (B,H)

    def correlation_loss(pred, true):
    # pred, true: (B, H)
        pred = pred - pred.mean(dim=1, keepdim=True)
        true = true - true.mean(dim=1, keepdim=True)
        corr = (pred * true).sum(dim=1) / (
            (pred.norm(dim=1) * true.norm(dim=1)) + 1e-6
        )
        return 1 - corr.mean()
    # =========================
# helpers
# =========================
def warmup_lr(ep:int, base_lr:float, warmup_epochs:int):
    if ep <= warmup_epochs:
        return base_lr * (ep / max(1, warmup_epochs))
    return base_lr

def batch_mae_in_original_units(pred_b: torch.Tensor, y_b: torch.Tensor, scaler_y) -> float:
    p = pred_b.detach().cpu().numpy().reshape(-1, 1)
    t = y_b.detach().cpu().numpy().reshape(-1, 1)
    p_orig = scaler_y.inverse_transform(p).reshape(-1)
    t_orig = scaler_y.inverse_transform(t).reshape(-1)
    return float(np.mean(np.abs(p_orig - t_orig)))

def batch_corrcoef(pred_b: torch.Tensor, y_b: torch.Tensor, scaler_y) -> float:
    """
    Pearson correlation coefficient (batch 평균)
    pred_b, y_b: (B, H)
    """
    p = pred_b.detach().cpu().numpy().reshape(-1, 1)
    t = y_b.detach().cpu().numpy().reshape(-1, 1)
    p_orig = scaler_y.inverse_transform(p).reshape(-1)
    t_orig = scaler_y.inverse_transform(t).reshape(-1)

    if np.std(p_orig) < 1e-6 or np.std(t_orig) < 1e-6:
        return 0.0
    return float(np.corrcoef(p_orig, t_orig)[0,1])

# =========================
# train & evaluate
# =========================
def train_and_eval(X: np.ndarray, y: np.ndarray, labels: list, feat_names: list):
    """
    X: (N,F), y: (N,), feat_names: ['ili', 'vaccine_rate', 'respiratory_index'] 등
    """
    set_seed(SEED)
    (s0,e0),(s1,e1),(s2,e2) = make_splits(len(y))
    X_tr, X_va, X_te = X[s0:e0], X[s1:e1], X[s2:e2]
    y_tr, y_va, y_te = y[s0:e0], y[s1:e1], y[s2:e2]
    lab_tr, lab_va, lab_te = labels[s0:e0], labels[s1:e1], labels[s2:e2]

    # ==== Scaling ====
    # Target scaler
    scaler_y = get_scaler()
    y_tr_sc = scaler_y.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_va_sc = scaler_y.transform(y_va.reshape(-1,1)).ravel()
    y_te_sc = scaler_y.transform(y_te.reshape(-1,1)).ravel()

    # Feature scaler (입력 특징 전체)
    scaler_x = get_scaler()
    X_tr_sc = scaler_x.fit_transform(X_tr)
    X_va_sc = scaler_x.transform(X_va)
    X_te_sc = scaler_x.transform(X_te)

    F = X.shape[1]
    print(f"[Shapes] X_tr:{X_tr.shape}, X_va:{X_va.shape}, X_te:{X_te.shape} | F={F}")
    print(f"[Info] Model input feature order -> {feat_names}")

    ds_tr = PatchTSTDataset(X_tr_sc, y_tr_sc, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
    ds_va = PatchTSTDataset(X_va_sc, y_va_sc, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
    ds_te = PatchTSTDataset(X_te_sc, y_te_sc, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)

    # drop_last=False 로 변경(작은 데이터셋에서도 학습 배치 보장)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False)

    model = PatchTSTModel(
        in_features=F, patch_len=PATCH_LEN, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=ENC_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT,
        pred_len=PRED_LEN, head_hidden=HEAD_HIDDEN
    ).to(DEVICE)

    # Loss / Optim / Scheduler
    crit = nn.HuberLoss(delta=1.0)
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    # ---- history for curves ----
    hist = {"train_loss":[], "val_loss":[], "train_mae":[], "val_mae":[]}

    best_val = float("inf"); best_state=None; noimp=0
    printed_batch_info = False
    for ep in range(1, EPOCHS+1):
        # ---- Train ----
        model.train(); tr_loss_sum=0; tr_mae_sum=0; n=0
        # warmup
        for g in opt.param_groups:
            g['lr'] = warmup_lr(ep, LR, WARMUP_EPOCHS)

        for Xb,yb,_ in dl_tr:
            if not printed_batch_info:
                # Xb: (B, P, L, F)  ← 최종 모델 입력 텐서 구조
                print(f"[Batch] Xb.shape={tuple(Xb.shape)} (B,P,L,F), yb.shape={tuple(yb.shape)}")
                print(f"[Batch] Feature order used -> {feat_names}")
                printed_batch_info = True
            Xb=Xb.to(DEVICE); yb=yb.to(DEVICE)
            opt.zero_grad()
            pred = model(Xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs=yb.size(0)
            tr_loss_sum += loss.item()*bs; n+=bs
            tr_mae_sum  += batch_mae_in_original_units(pred, yb, scaler_y)*bs

        tr_loss = tr_loss_sum / max(1,n)
        tr_mae  = tr_mae_sum  / max(1,n)

        # ---- Validation ----
        model.eval(); va_loss_sum=0; va_mae_sum=0; va_corr_sum = 0; n=0
        with torch.no_grad():
            for Xb,yb,_ in dl_va:
                Xb=Xb.to(DEVICE); yb=yb.to(DEVICE)
                pred = model(Xb); loss = crit(pred,yb)
                bs=yb.size(0)
                va_loss_sum += loss.item()*bs; n+=bs
                va_mae_sum  += batch_mae_in_original_units(pred, yb, scaler_y)*bs
                va_corr_sum += batch_corrcoef(pred, yb, scaler_y)*bs
        va_loss = va_loss_sum / max(1,n)
        va_mae  = va_mae_sum  / max(1,n)
        va_corr = va_corr_sum / max(1,n)

        scheduler.step()

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["train_mae"].append(tr_mae)
        hist["val_mae"].append(va_mae)

        print(f"[Epoch {ep:03d}/{EPOCHS}] "
              f"LR={opt.param_groups[0]['lr']:.6f} | "
              f"Loss T/V={tr_loss:.5f}/{va_loss:.5f} | "
              f"MAE  T/V={tr_mae:.5f}/{va_mae:.5f}"
              f"Corr V={va_corr:.3f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss; noimp=0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            noimp += 1
            if noimp >= PATIENCE:
                print(f"Early stopping after {ep} epochs (no improvement {PATIENCE}).")
                break

    if best_state is not None:
        model.load_state_dict({k:v.to(DEVICE) for k,v in best_state.items()})

    # ---- Test & Metrics ----
    model.eval(); preds=[]; trues=[]; starts=[]
    with torch.no_grad():
        for Xb,yb,i0 in dl_te:
            Xb=Xb.to(DEVICE)
            preds.append(model(Xb).detach().cpu().numpy())
            trues.append(yb.numpy())
            starts.append(i0.numpy())
    yhat_sc = np.concatenate(preds,axis=0)
    ytrue_sc= np.concatenate(trues,axis=0)
    starts  = np.concatenate(starts,axis=0)

    # inverse scale (target only)
    yhat  = scaler_y.inverse_transform(yhat_sc.reshape(-1,1)).reshape(-1,PRED_LEN)
    ytrue = scaler_y.inverse_transform(ytrue_sc.reshape(-1,1)).reshape(-1,PRED_LEN)

    mse  = float(np.mean((yhat-ytrue)**2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(yhat-ytrue)))
    print("\n=== Final Test Metrics ===")
    print(f"MSE : {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")

    # =========================
    # Save per-window predictions
    # =========================
    cols_true = [f"true_t+{i}" for i in range(1,PRED_LEN+1)]
    cols_pred = [f"pred_t+{i}" for i in range(1,PRED_LEN+1)]
    out = pd.DataFrame(np.hstack([ytrue, yhat]), columns=cols_true+cols_pred)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved predictions -> {OUT_CSV}")

    # =========================
    # Plot_1: last window (H-step ahead)
    # =========================
    last_true = ytrue[-1]; last_pred = yhat[-1]
    weeks = np.arange(1, PRED_LEN+1)
    plt.figure(figsize=(10,4))
    plt.plot(weeks, last_true, label="Truth (last window)", linewidth=2)
    plt.plot(weeks, last_pred, label="Prediction (last window)", linewidth=2)
    plt.title("Last Test Window: Truth vs Prediction")
    plt.xlabel("Horizon (weeks ahead)")
    plt.ylabel("ILI per 1,000 Population")
    plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(PLOT_LAST_WINDOW, dpi=150)
    print(f"Saved plot -> {PLOT_LAST_WINDOW}")

    # =========================
    # Plot_2: test reconstruction (val-context included)
    # =========================
    context = y_va_sc[-SEQ_LEN:]                       # 표준화 컨텍스트
    y_ct_sc = np.concatenate([context, y_te_sc])       # [SEQ_LEN + test_len]
    # 입력 특징도 컨텍스트 포함해 재구성 필요 → X도 동일하게 붙여서 예측
    X_ct_sc = np.concatenate([X_va_sc[-SEQ_LEN:], X_te_sc], axis=0)
    ds_ct = PatchTSTDataset(X_ct_sc, y_ct_sc, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
    dl_ct = DataLoader(ds_ct, batch_size=BATCH_SIZE, shuffle=False)

    model.eval(); preds_ct=[]; starts_ct=[]
    with torch.no_grad():
        for Xb, _, i0 in dl_ct:
            Xb = Xb.to(DEVICE)
            preds_ct.append(model(Xb).detach().cpu().numpy())  # (B, H)
            starts_ct.append(i0.numpy())
    yhat_ct_sc = np.concatenate(preds_ct, axis=0)
    starts_ct  = np.concatenate(starts_ct, axis=0)
    yhat_ct = scaler_y.inverse_transform(yhat_ct_sc.reshape(-1,1)).reshape(-1, PRED_LEN)

    test_len = len(y_te)
    recon_sum   = np.zeros(test_len)
    recon_count = np.zeros(test_len)
    h_weights = np.linspace(RECON_W_START, RECON_W_END, PRED_LEN)

    for k, s in enumerate(starts_ct):
        pos0_ct = int(s) + SEQ_LEN   # [context+test] 축
        pos0_te = pos0_ct - SEQ_LEN  # test 축으로 변환
        for j in range(PRED_LEN):
            idx = pos0_te + j
            if 0 <= idx < test_len:
                w = h_weights[j]
                recon_sum[idx]   += yhat_ct[k, j] * w
                recon_count[idx] += w

    recon = np.where(recon_count > 0, recon_sum / np.maximum(1, recon_count), np.nan)

    truth_test = y_te
    x_labels = lab_te
    tick_step = max(1, test_len // 12)
    tick_idx  = list(range(0, test_len, tick_step))
    if tick_idx[-1] != test_len-1:
        tick_idx.append(test_len-1)
    tick_text = [x_labels[i] for i in tick_idx]

    plt.figure(figsize=(12,5))
    plt.plot(range(test_len), truth_test, linewidth=2, label="Truth (test segment)")
    plt.plot(range(test_len), recon,      linewidth=2, label="Prediction (overlap-avg, weighted)")
    plt.title("Test Range: Truth vs Overlap-averaged Prediction (with context)")
    plt.xlabel("Season - Week"); plt.ylabel("ILI per 1,000 Population")
    plt.xticks(tick_idx, tick_text, rotation=45, ha="right")
    plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(PLOT_TEST_RECON, dpi=150)
    print(f"Saved plot -> {PLOT_TEST_RECON}")

    # =========================
    # Plot_3: Train/Val MAE curves
    # =========================
    xs = np.arange(1, len(hist["train_mae"])+1)
    plt.figure(figsize=(10,4))
    plt.plot(xs, hist["train_mae"], linewidth=2, label="Train MAE (original units)")
    plt.plot(xs, hist["val_mae"],   linewidth=2, label="Val MAE (original units)")
    plt.title("Training Curves: MAE per epoch (lower is better)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (ILI per 1,000)")
    plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(PLOT_MA_CURVES, dpi=150)
    print(f"Saved plot -> {PLOT_MA_CURVES}")


# =========================
# run
# =========================
if __name__ == "__main__":
    print("\n" + "🚀 " * 30)
    print("데이터 로드 및 모델 학습 시작!")
    print("🚀 " * 30 + "\n")
    
    # API 또는 CSV에서 데이터 로드
    # 환경변수 USE_API=true로 설정하면 API 사용, 아니면 CSV 사용
    USE_API_MODE = os.getenv('USE_API', 'false').lower() == 'true'
    
    if USE_API_MODE:
        print("=" * 60)
        print("🌐 API 모드: Python에서 직접 GFID API 호출")
        print("=" * 60)
        
        # Python에서 직접 Keycloak 인증 후 GFID API 호출
        df = fetch_data_directly_from_gfid()
        
        print("\n" + "✅ " * 30)
        print("API 데이터 로드 완료!")
        print("✅ " * 30 + "\n")
        
        # 데이터 확인
        print(f"📊 DataFrame 정보:")
        print(f"   - Shape: {df.shape}")
        print(f"   - Columns: {list(df.columns)}")
        print(f"\n처음 5개 행:")
        print(df.head())
        print(f"\n데이터 타입:")
        print(df.dtypes)
        
        print(f"\n🔧 USE_EXOG = '{USE_EXOG}'  (auto-detects vaccine/resp columns)")
        
        # DataFrame을 직접 전달하여 전처리
        print("\n📈 데이터 전처리 및 특징 추출 중...")
        X, y, labels, feat_names = load_and_prepare(df=df, use_exog=USE_EXOG)
        print(f"✅ 전처리 완료!")
        print(f"   - Data points: {len(y)}")
        print(f"   - Features used ({len(feat_names)}): {feat_names}")
        
    else:
        print("=" * 60)
        print("📁 CSV 모드: 로컬 파일에서 데이터를 로드합니다.")
        print("=" * 60)
        
        if CSV_PATH is None:
            raise FileNotFoundError("CSV 파일을 찾을 수 없습니다. USE_API=true로 설정하거나 CSV 파일을 준비하세요.")
        
        print(f"   - CSV 파일: {CSV_PATH.name}")
        print(f"   - Device: {DEVICE}")
        print(f"   - USE_EXOG: '{USE_EXOG}'")
        
        print("\n📈 데이터 로드 및 전처리 중...")
        X, y, labels, feat_names = load_and_prepare(CSV_PATH, USE_EXOG)
        print(f"✅ CSV 로드 및 전처리 완료!")
        print(f"   - Data points: {len(y)}")
        print(f"   - Features used ({len(feat_names)}): {feat_names}")
    
    # 모델 학습 및 평가
    print("\n" + "🎯 " * 30)
    print("모델 학습 시작!")
    print("🎯 " * 30 + "\n")
    train_and_eval(X, y, labels, feat_names)

    # =========================
# Feature Importance utils
# =========================
def _eval_mae_on_split(model, X_split_sc, y_split_sc, scaler_y, feat_names, 
                       seq_len=SEQ_LEN, pred_len=PRED_LEN, patch_len=PATCH_LEN, stride=STRIDE,
                       batch_size=BATCH_SIZE):
    """현재 모델로 한 분할(va/test) 세트에서 MAE(원 단위) 계산"""
    ds = PatchTSTDataset(X_split_sc, y_split_sc, seq_len, pred_len, patch_len, stride)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    mae_sum, n = 0.0, 0
    with torch.no_grad():
        for Xb, yb, _ in dl:
            Xb = Xb.to(DEVICE); yb = yb.to(DEVICE)
            pred = model(Xb)  # (B, H)
            mae_sum += batch_mae_in_original_units(pred, yb, scaler_y) * yb.size(0)
            n += yb.size(0)
    return float(mae_sum / max(1, n))


def compute_feature_importance(model, 
                               X_va_sc, y_va_sc, 
                               X_te_sc=None, y_te_sc=None,
                               scaler_y=None, feat_names=None, 
                               random_state=42):
    """
    퍼뮤테이션(열 섞기) 중요도와 평균 대체(그 특징을 평균으로 고정) 중요도를 계산.
    반환: 중요도 DataFrame (ΔMAE가 클수록 중요)
    """
    assert scaler_y is not None and feat_names is not None
    rng = np.random.RandomState(random_state)

    # --- 기준선(baseline MAE) ---
    baseline_val = _eval_mae_on_split(model, X_va_sc, y_va_sc, scaler_y, feat_names)
    print(f"[FI] Baseline Val MAE: {baseline_val:.6f}")

    baseline_tst = None
    if X_te_sc is not None and y_te_sc is not None:
        baseline_tst = _eval_mae_on_split(model, X_te_sc, y_te_sc, scaler_y, feat_names)
        print(f"[FI] Baseline Test MAE: {baseline_tst:.6f}")

    perm_deltas_val, mean_deltas_val = [], []
    perm_deltas_tst, mean_deltas_tst = [], []

    for j, name in enumerate(feat_names):
        # ① 퍼뮤테이션(열 섞기)
        Xp = X_va_sc.copy()
        col = Xp[:, j].copy()
        rng.shuffle(col)
        Xp[:, j] = col
        mae_perm_val = _eval_mae_on_split(model, Xp, y_va_sc, scaler_y, feat_names)
        perm_deltas_val.append(mae_perm_val - baseline_val)

        # ② 평균 대체(특징 제거 효과)
        Xz = X_va_sc.copy()
        Xz[:, j] = X_va_sc[:, j].mean()
        mae_mean_val = _eval_mae_on_split(model, Xz, y_va_sc, scaler_y, feat_names)
        mean_deltas_val.append(mae_mean_val - baseline_val)

        if X_te_sc is not None and y_te_sc is not None:
            Xp_te = X_te_sc.copy()
            col_te = Xp_te[:, j].copy()
            rng.shuffle(col_te)
            Xp_te[:, j] = col_te
            mae_perm_tst = _eval_mae_on_split(model, Xp_te, y_te_sc, scaler_y, feat_names)
            perm_deltas_tst.append(mae_perm_tst - baseline_tst)

            Xz_te = X_te_sc.copy()
            Xz_te[:, j] = X_te_sc[:, j].mean()
            mae_mean_tst = _eval_mae_on_split(model, Xz_te, y_te_sc, scaler_y, feat_names)
            mean_deltas_tst.append(mae_mean_tst - baseline_tst)

    # DataFrame 생성
    df_fi = pd.DataFrame({
        "feature": feat_names,
        "perm_delta_val": perm_deltas_val,
        "mean_delta_val": mean_deltas_val,
    })
    if X_te_sc is not None and y_te_sc is not None:
        df_fi["perm_delta_tst"] = perm_deltas_tst
        df_fi["mean_delta_tst"] = mean_deltas_tst

    # 평균 델타 기준 내림차순 정렬
    df_fi = df_fi.sort_values("mean_delta_val", ascending=False).reset_index(drop=True)
    return df_fi

def plot_feature_importance(fi_df, out_csv=None, out_png=None):
    """
    Feature Importance를 막대그래프로 시각화
    """
    if fi_df is None or len(fi_df) == 0:
        print("No feature importance data to plot.")
        return

    import matplotlib.pyplot as plt

    # CSV 저장
    if out_csv:
        fi_df.to_csv(out_csv, index=False)
        print(f"Feature Importance saved to {out_csv}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ① Permutation Δ (Val)
    axes[0].barh(fi_df["feature"], fi_df["perm_delta_val"], color="steelblue")
    axes[0].set_xlabel("ΔMAE (Permutation, Val)")
    axes[0].set_title("Permutation Feature Importance (Val)")
    axes[0].invert_yaxis()

    # ② Mean Replacement Δ (Val)
    axes[1].barh(fi_df["feature"], fi_df["mean_delta_val"], color="coral")
    axes[1].set_xlabel("ΔMAE (Mean Replacement, Val)")
    axes[1].set_title("Mean Replacement Feature Importance (Val)")
    axes[1].invert_yaxis()

    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Feature Importance plot saved to {out_png}")
    plt.show()


# =========================
# train_and_eval (main)
# =========================
def train_and_eval(X: np.ndarray, y: np.ndarray, labels: list, feat_names: list,
                   compute_fi=False, save_fi=False):
    """
    통합 학습 + 평가 함수.
    compute_fi=True -> feature importance 계산
    save_fi=True -> CSV/plot 저장
    """
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"[Config] EPOCHS:{EPOCHS}, BATCH_SIZE:{BATCH_SIZE}, SEQ_LEN:{SEQ_LEN}, PRED_LEN:{PRED_LEN}")
    print(f"[Config] PATCH_LEN:{PATCH_LEN}, STRIDE:{STRIDE}, LR:{LR}, Warmup:{WARMUP_EPOCHS}, Patience:{PATIENCE}")

    N = len(y)
    split_tr = int(0.7*N); split_va = int(0.85*N)
    X_tr, y_tr = X[:split_tr], y[:split_tr]
    X_va, y_va = X[split_tr:split_va], y[split_tr:split_va]
    X_te, y_te = X[split_va:], y[split_va:]

    def get_scaler():
        st = SCALER_TYPE.lower()
        if st=="robust": return RobustScaler()
        if st=="minmax": return MinMaxScaler()
        return StandardScaler()

    scaler_y = get_scaler()
    y_tr_sc = scaler_y.fit_transform(y_tr.reshape(-1,1)).ravel()
    y_va_sc = scaler_y.transform(y_va.reshape(-1,1)).ravel()
    y_te_sc = scaler_y.transform(y_te.reshape(-1,1)).ravel()

    scaler_x = get_scaler()
    X_tr_sc = scaler_x.fit_transform(X_tr)
    X_va_sc = scaler_x.transform(X_va)
    X_te_sc = scaler_x.transform(X_te)

    F = X.shape[1]
    print(f"[Shapes] X_tr:{X_tr.shape}, X_va:{X_va.shape}, X_te:{X_te.shape} | F={F}")
    print(f"[Info] Model input feature order -> {feat_names}")

    ds_tr = PatchTSTDataset(X_tr_sc, y_tr_sc, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
    ds_va = PatchTSTDataset(X_va_sc, y_va_sc, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)
    ds_te = PatchTSTDataset(X_te_sc, y_te_sc, SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False)

    model = PatchTSTModel(
        in_features=F, patch_len=PATCH_LEN, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=ENC_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT,
        pred_len=PRED_LEN, head_hidden=HEAD_HIDDEN
    ).to(DEVICE)

    crit = nn.HuberLoss(delta=1.0)
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    hist = {"train_loss":[], "val_loss":[], "train_mae":[], "val_mae":[]}

    best_val = float("inf"); best_state=None; noimp=0
    printed_batch_info = False
    for ep in range(1, EPOCHS+1):
        model.train(); tr_loss_sum=0; tr_mae_sum=0; n=0
        for g in opt.param_groups:
            g['lr'] = warmup_lr(ep, LR, WARMUP_EPOCHS)

        for Xb, yb, _ in dl_tr:
            Xb=Xb.to(DEVICE); yb=yb.to(DEVICE)
            if not printed_batch_info:
                print(f"[Batch shapes] Xb:{Xb.shape}, yb:{yb.shape}")
                printed_batch_info=True
            opt.zero_grad()
            pred=model(Xb)
            loss=crit(pred,yb)
            loss.backward(); opt.step()

            tr_loss_sum += loss.item()*yb.size(0)
            tr_mae_sum += batch_mae_in_original_units(pred, yb, scaler_y)*yb.size(0)
            n+=yb.size(0)

        tr_loss_avg = tr_loss_sum/max(1,n)
        tr_mae_avg  = tr_mae_sum/max(1,n)

        model.eval(); va_loss_sum=0; va_mae_sum=0; m=0
        with torch.no_grad():
            for Xb,yb,_ in dl_va:
                Xb=Xb.to(DEVICE); yb=yb.to(DEVICE)
                pred=model(Xb)
                loss=crit(pred,yb)
                va_loss_sum += loss.item()*yb.size(0)
                va_mae_sum  += batch_mae_in_original_units(pred,yb,scaler_y)*yb.size(0)
                m+=yb.size(0)
        va_loss_avg=va_loss_sum/max(1,m)
        va_mae_avg =va_mae_sum/max(1,m)

        hist["train_loss"].append(tr_loss_avg)
        hist["val_loss"].append(va_loss_avg)
        hist["train_mae"].append(tr_mae_avg)
        hist["val_mae"].append(va_mae_avg)

        if ep<=5 or ep%5==0:
            print(f"Epoch {ep:3d}/{EPOCHS} | TrL:{tr_loss_avg:.6f} TrMAE:{tr_mae_avg:.6f} | VaL:{va_loss_avg:.6f} VaMAE:{va_mae_avg:.6f}")

        if va_mae_avg < best_val:
            best_val = va_mae_avg
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            noimp=0
        else:
            noimp+=1
            if noimp>=PATIENCE:
                print(f"Early stop at epoch {ep} (no improvement for {PATIENCE} epochs)")
                break

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best Val MAE: {best_val:.6f}")

    # Test
    model.eval(); te_mae_sum=0; k=0
    with torch.no_grad():
        for Xb,yb,_ in dl_te:
            Xb=Xb.to(DEVICE); yb=yb.to(DEVICE)
            pred=model(Xb)
            te_mae_sum += batch_mae_in_original_units(pred,yb,scaler_y)*yb.size(0)
            k+=yb.size(0)
    te_mae_avg = te_mae_sum/max(1,k)
    print(f"Test MAE (original units): {te_mae_avg:.6f}")

    # Plot curves
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(hist["train_mae"],label="Train MAE")
    plt.plot(hist["val_mae"],label="Val MAE")
    plt.xlabel("Epoch"); plt.ylabel("MAE (original units)"); plt.legend(); plt.title("MAE curves")
    plt.subplot(1,2,2)
    plt.plot(hist["train_loss"],label="Train Loss")
    plt.plot(hist["val_loss"],label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Huber Loss"); plt.legend(); plt.title("Loss curves")
    plt.tight_layout()
    plt.savefig(PLOT_MA_CURVES, dpi=150)
    print(f"MAE/loss curves saved to {PLOT_MA_CURVES}")
    plt.show()

    # Last window
    last_seq_idx = len(y_te_sc) - SEQ_LEN
    if last_seq_idx>=0:
        seq = X_te_sc[last_seq_idx:last_seq_idx+SEQ_LEN]
        seq_t = torch.from_numpy(seq).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            p=model(seq_t).cpu().numpy().ravel()
        p_orig = scaler_y.inverse_transform(p.reshape(-1,1)).ravel()
        y_true_last = scaler_y.inverse_transform(y_te_sc[last_seq_idx+SEQ_LEN:last_seq_idx+SEQ_LEN+PRED_LEN].reshape(-1,1)).ravel()
        plt.figure(figsize=(8,4))
        plt.plot(range(len(y_true_last)), y_true_last, marker='o', label="True")
        plt.plot(range(len(p_orig)), p_orig, marker='x', label="Pred")
        plt.xlabel("Future step (horizon)"); plt.ylabel("ILI")
        plt.title(f"Last window prediction (SEQ_LEN={SEQ_LEN}, PRED_LEN={PRED_LEN})")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(PLOT_LAST_WINDOW, dpi=150)
        print(f"Last window plot saved to {PLOT_LAST_WINDOW}")
        plt.show()

    # Test reconstruction
    all_p_te = []
    model.eval()
    with torch.no_grad():
        for Xb,_,_ in dl_te:
            Xb=Xb.to(DEVICE)
            p_b=model(Xb).cpu().numpy()
            all_p_te.append(p_b)
    all_p_te = np.concatenate(all_p_te, axis=0)
    pred_orig = scaler_y.inverse_transform(all_p_te).ravel()
    y_te_orig = scaler_y.inverse_transform(y_te_sc.reshape(-1,1)).ravel()

    plt.figure(figsize=(12,5))
    plt.plot(y_te_orig, label="True", alpha=0.7)
    plt.plot(pred_orig[:len(y_te_orig)], label="Pred", alpha=0.7)
    plt.xlabel("Test set index"); plt.ylabel("ILI")
    plt.title("Test set reconstruction (multi-step predictions)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(PLOT_TEST_RECON, dpi=150)
    print(f"Test reconstruction plot saved to {PLOT_TEST_RECON}")
    plt.show()

    # Feature importance
    fi_df = None
    if compute_fi:
        print("\n[Computing Feature Importance...]")
        fi_df = compute_feature_importance(
            model, X_va_sc, y_va_sc, X_te_sc, y_te_sc,
            scaler_y, feat_names, random_state=SEED
        )
        print("\n[Feature Importance (sorted by mean_delta_val)]")
        print(fi_df.to_string(index=False))

        if save_fi:
            plot_feature_importance(
                fi_df,
                out_csv=str(BASE_DIR / "feature_importance.csv"),
                out_png=str(BASE_DIR / "feature_importance.png")
            )

    # 반환: 외부 셀에서 재활용 가능하도록
    return model, X_va_sc, y_va_sc, X_te_sc, y_te_sc, scaler_y, feat_names, fi_df

# =========================
# 실행부 (결과 출력)
# =========================
if __name__ == "__main__":
    model, X_va_sc, y_va_sc, X_te_sc, y_te_sc, scaler_y, feat_names, fi_df = train_and_eval(
        X, y, labels, feat_names,
        compute_fi=True,
        save_fi=True
    )

    print("\n=== [결과 요약] ===")
    print(f"Feature 개수: {len(feat_names)}")
    if fi_df is not None:
        print("\n[Top 10 Feature Importance]")
        print(fi_df.head(10).to_string(index=False))
    else:
        print("Feature Importance 계산이 수행되지 않았습니다.")

        
