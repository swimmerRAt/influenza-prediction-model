"""
GFID API 클라이언트 (Python)

src_jaehong/api/ 패턴을 참고하여 구현
- auth.js -> get_access_token()
- config.js -> apiClient 설정
- etlDataApi.js -> ETL 데이터 조회 함수
"""

import os
import time
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# 설정
# =============================================================================

# API 서버 URL
API_BASE_URL = os.getenv('GFID_API_URL', 'http://211.238.12.60:8084/data/api/v1')

# Keycloak 인증 설정
KEYCLOAK_SERVER_URL = os.getenv('KEYCLOAK_SERVER_URL', 'http://211.238.12.60:8080')
KEYCLOAK_REALM = os.getenv('KEYCLOAK_REALM', 'gfid')
CLIENT_ID = os.getenv('GFID_CLIENT_ID')
CLIENT_SECRET = os.getenv('GFID_CLIENT_SECRET')

# 토큰 캐시
_token_cache = {
    'token': None,
    'expiry': 0
}

# 요청 타임아웃 (초)
REQUEST_TIMEOUT = 30


# =============================================================================
# 인증 (auth.js 패턴)
# =============================================================================

def is_auth_configured():
    """인증 설정이 완료되었는지 확인"""
    return all([
        KEYCLOAK_SERVER_URL,
        KEYCLOAK_REALM,
        CLIENT_ID,
        CLIENT_SECRET,
        '{{' not in str(CLIENT_ID),
        '{{' not in str(CLIENT_SECRET)
    ])


def get_access_token():
    """
    Keycloak에서 액세스 토큰 가져오기
    
    Returns:
        str: 액세스 토큰 (인증 실패 시 None)
    """
    global _token_cache
    
    # 환경 변수 확인
    if not is_auth_configured():
        print("⚠️  Keycloak 인증 설정이 완료되지 않았습니다. .env 파일을 확인하세요.")
        print("   필요한 환경 변수: GFID_CLIENT_ID, GFID_CLIENT_SECRET")
        return None
    
    # 캐시된 토큰이 있고 아직 유효한지 확인
    if _token_cache['token'] and time.time() < _token_cache['expiry']:
        return _token_cache['token']
    
    # 새 토큰 요청
    try:
        token_url = f"{KEYCLOAK_SERVER_URL}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/token"
        
        data = {
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
            'grant_type': 'client_credentials'
        }
        
        response = requests.post(
            token_url,
            data=data,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=10,
            verify=False  # SSL 인증서 검증 비활성화 (개발 환경)
        )
        response.raise_for_status()
        
        result = response.json()
        access_token = result.get('access_token')
        expires_in = result.get('expires_in', 300)  # 기본 5분
        
        # 토큰 캐시 (만료 1분 전까지 유효)
        _token_cache['token'] = access_token
        _token_cache['expiry'] = time.time() + expires_in - 60
        
        print(f"✅ Keycloak 토큰 발급 완료 (유효기간: {expires_in}초)")
        return access_token
    
    except requests.exceptions.RequestException as e:
        print(f"❌ 토큰 발급 실패: {e}")
        return None


def clear_token():
    """캐시된 토큰 제거"""
    global _token_cache
    _token_cache = {'token': None, 'expiry': 0}


# =============================================================================
# API 클라이언트 (config.js 패턴)
# =============================================================================

def api_request(method, endpoint, params=None, json_data=None, retry_auth=True):
    """
    API 요청 수행 (인증 토큰 자동 추가)
    
    Args:
        method: HTTP 메소드 ('GET', 'POST')
        endpoint: API 엔드포인트 ('/etl_data/id/...')
        params: 쿼리 파라미터
        json_data: POST 바디 데이터
        retry_auth: 401 에러 시 토큰 갱신 후 재시도 여부
    
    Returns:
        dict: API 응답 데이터
    """
    url = f"{API_BASE_URL}{endpoint}"
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    # 인증 토큰 추가
    token = get_access_token()
    if token:
        headers['Authorization'] = f'Bearer {token}'
    
    try:
        if method.upper() == 'GET':
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
                verify=False
            )
        else:
            response = requests.post(
                url,
                params=params,
                json=json_data,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
                verify=False
            )
        
        # 401 Unauthorized - 토큰 만료
        if response.status_code == 401 and retry_auth:
            print("⚠️  인증 토큰이 만료되었습니다. 새 토큰을 요청합니다.")
            clear_token()
            token = get_access_token()
            if token:
                headers['Authorization'] = f'Bearer {token}'
                return api_request(method, endpoint, params, json_data, retry_auth=False)
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"❌ API 요청 실패 ({url}): {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   응답: {e.response.text[:500]}")
        raise


# =============================================================================
# ETL 데이터 API (etlDataApi.js 패턴)
# =============================================================================

def get_recent_etl_data(dsid, cnt=100):
    """
    자동수집 데이터 중 특정 id의 최근 n건 데이터 조회
    
    Args:
        dsid: 데이터셋 ID (예: 'ds_0101', 'ds_0701')
        cnt: 조회할 데이터 건수
    
    Returns:
        list: 데이터 리스트
    """
    print(f"🔵 [recent API] dsid={dsid}, cnt={cnt}")
    
    endpoint = f"/etl_data/id/{dsid}/recent/{cnt}"
    result = api_request('GET', endpoint)
    
    # 데이터 추출 (body.data 또는 data 또는 직접)
    data = result.get('body', {}).get('data') or result.get('data') or result
    
    if isinstance(data, list):
        print(f"✅ [recent API] {len(data)}건 조회 완료")
    
    return data


def get_etl_data_by_date_range(dsid, from_date, to_date):
    """
    자동수집 데이터 중 특정 id의 특정 기간 데이터 조회
    
    Args:
        dsid: 데이터셋 ID
        from_date: 시작 날짜 (YYYY-MM-DD 형식)
        to_date: 종료 날짜 (YYYY-MM-DD 형식)
    
    Returns:
        list: 데이터 리스트
    """
    print(f"🔵 [날짜 범위 API] dsid={dsid}, from={from_date}, to={to_date}")
    
    endpoint = f"/etl_data/id/{dsid}/from/{from_date}/to/{to_date}"
    result = api_request('GET', endpoint)
    
    # 데이터 추출
    data = result.get('body', {}).get('data') or result.get('data') or result
    
    if isinstance(data, list):
        print(f"✅ [날짜 범위 API] {len(data)}건 조회 완료")
        if len(data) > 0 and isinstance(data[0], dict):
            print(f"   📦 첫 번째 항목 키: {list(data[0].keys())[:10]}")
    
    return data


def get_etl_data_by_origin(dsid, origin):
    """
    자동수집 데이터 중 특정 id와 origin으로 데이터 조회
    
    Args:
        dsid: 데이터셋 ID
        origin: origin 값 (UUID)
    
    Returns:
        dict or list: 데이터
    """
    print(f"🔵 [origin API] dsid={dsid}, origin={origin}")
    
    endpoint = f"/etl_data/id/{dsid}/origin/{origin}"
    result = api_request('GET', endpoint)
    
    # 데이터 추출
    data = result.get('body', {}).get('data') or result.get('data') or result
    
    if isinstance(data, list):
        print(f"✅ [origin API] {len(data)}건 조회 완료")
    
    return data


def get_etl_data_statistics():
    """
    자동수집 데이터 중 id별 총 데이터 수 조회
    
    Returns:
        dict: 통계 데이터
    """
    print("🔵 [통계 API] 전체 통계 조회")
    
    endpoint = "/etl_data/statistics"
    result = api_request('GET', endpoint)
    
    data = result.get('body', {}).get('data') or result.get('data') or result
    print(f"✅ [통계 API] 조회 완료")
    
    return data


def get_date_range_from_season(season):
    """
    절기를 날짜 범위로 변환
    
    Args:
        season: 절기 (예: '25/26')
    
    Returns:
        tuple: (from_date, to_date) - YYYY-MM-DD 형식
    """
    # 절기 정의: XX/YY절기 = XX년 36주 ~ YY년 35주
    # 예: 25/26절기 = 2025년 36주 ~ 2026년 35주
    parts = season.split('/')
    year1 = int('20' + parts[0])
    year2 = int('20' + parts[1])
    
    # XX년 36주 시작일 (대략 9월 첫째 주)
    from_date = f"{year1}-09-01"
    
    # YY년 35주 종료일 (대략 8월 마지막 주)
    to_date = f"{year2}-08-31"
    
    return from_date, to_date


def get_etl_data_by_season(dsid, season, origins=None):
    """
    자동수집 데이터 중 특정 id의 절기별 데이터 조회
    
    25/26절기는 origin별로 요청, 나머지는 날짜 범위로 요청
    
    Args:
        dsid: 데이터셋 ID
        season: 절기 (예: '25/26')
        origins: origin 목록 (25/26절기인 경우 필수)
    
    Returns:
        list: 데이터 리스트
    """
    print(f"🔵 [{season}절기 API] dsid={dsid}")
    
    # 25/26절기는 origin별로 요청
    if season == '25/26' and origins:
        print(f"   origin별 요청 시작 ({len(origins)}개)")
        
        all_data = []
        for origin in origins:
            try:
                origin_data = get_etl_data_by_origin(dsid, origin)
                if isinstance(origin_data, list):
                    all_data.extend(origin_data)
                elif origin_data:
                    all_data.append(origin_data)
            except Exception as e:
                print(f"   ⚠️  origin {origin} 요청 실패: {e}")
        
        print(f"✅ [{season}절기 API] origin별 요청 완료: 총 {len(all_data)}건")
        return all_data
    else:
        # 나머지 절기는 날짜 범위로 요청
        from_date, to_date = get_date_range_from_season(season)
        print(f"   날짜 범위: {from_date} ~ {to_date}")
        
        return get_etl_data_by_date_range(dsid, from_date, to_date)


# =============================================================================
# 트렌드 데이터 조회 (ds_0701: Google, ds_0801: Naver, ds_0901: Twitter)
# =============================================================================

def fetch_trend_data_from_api(dsid, season=None, cnt=500):
    """
    트렌드 데이터 API에서 직접 가져오기
    
    Args:
        dsid: 데이터셋 ID (ds_0701, ds_0801, ds_0901)
        season: 절기 (선택, 예: '25/26')
        cnt: 최근 건수 (season 미지정 시)
    
    Returns:
        list: 트렌드 데이터 리스트
    """
    dsid_names = {
        'ds_0701': 'Google Trends',
        'ds_0801': 'Naver Trends',
        'ds_0901': 'Twitter Trends'
    }
    dsid_name = dsid_names.get(dsid, dsid)
    
    print(f"\n📡 API에서 {dsid} ({dsid_name}) 데이터 다운로드 중...")
    
    try:
        if season:
            data = get_etl_data_by_season(dsid, season)
        else:
            data = get_recent_etl_data(dsid, cnt)
        
        if not data:
            print(f"   ⚠️  [{dsid}] 데이터 없음")
            return []
        
        print(f"   ✅ [{dsid}] {len(data)}건 조회 완료")
        return data
    
    except Exception as e:
        print(f"   ❌ [{dsid}] 조회 실패: {e}")
        return []


# =============================================================================
# 인플루엔자 데이터 조회
# =============================================================================

# 데이터셋 ID 매핑
INFLUENZA_DATASETS = {
    'ds_0101': 'ILI (인플루엔자 의사환자 분율)',
    'ds_0103': 'SARI (중증급성호흡기감염증 입원환자)',
    'ds_0104': 'ARI (급성호흡기감염증 입원환자)',
    'ds_0105': 'I-RISS (검사기관 인플루엔자 검출률)',
    'ds_0106': 'K-RISS (의원급 인플루엔자 검출률)',
    'ds_0107': '호흡기병원체 검출현황',
    'ds_0108': '인플루엔자 표본감시 현황',
    'ds_0109': 'NEDIS (응급실 인플루엔자 환자)',
    'ds_0110': '예방접종률',
}


def fetch_influenza_data_from_api(dsid, cnt=500):
    """
    인플루엔자 데이터 API에서 직접 가져오기
    
    Args:
        dsid: 데이터셋 ID
        cnt: 최근 건수
    
    Returns:
        list: 인플루엔자 데이터 리스트
    """
    dsid_name = INFLUENZA_DATASETS.get(dsid, dsid)
    
    print(f"\n📡 API에서 {dsid} ({dsid_name}) 데이터 다운로드 중...")
    
    try:
        data = get_recent_etl_data(dsid, cnt)
        
        if not data:
            print(f"   ⚠️  [{dsid}] 데이터 없음")
            return []
        
        print(f"   ✅ [{dsid}] {len(data)}건 조회 완료")
        return data
    
    except Exception as e:
        print(f"   ❌ [{dsid}] 조회 실패: {e}")
        return []


def fetch_all_influenza_data(cnt=500):
    """
    모든 인플루엔자 데이터셋 조회
    
    Args:
        cnt: 각 데이터셋당 최근 건수
    
    Returns:
        dict: {dsid: data_list}
    """
    print("\n" + "="*60)
    print("📊 인플루엔자 데이터 전체 조회 (GFID API 직접 호출)")
    print("="*60)
    
    all_data = {}
    
    for dsid in INFLUENZA_DATASETS.keys():
        try:
            data = fetch_influenza_data_from_api(dsid, cnt)
            if data:
                all_data[dsid] = data
        except Exception as e:
            print(f"   ⚠️  {dsid} 건너뜀: {e}")
    
    print(f"\n✅ 총 {len(all_data)}개 데이터셋 조회 완료")
    return all_data


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    print("="*60)
    print("🧪 GFID API 클라이언트 테스트")
    print("="*60)
    
    print("\n[1] 인증 설정 확인")
    print(f"   API_BASE_URL: {API_BASE_URL}")
    print(f"   KEYCLOAK_SERVER_URL: {KEYCLOAK_SERVER_URL}")
    print(f"   CLIENT_ID: {'설정됨' if CLIENT_ID else '미설정'}")
    print(f"   CLIENT_SECRET: {'설정됨' if CLIENT_SECRET else '미설정'}")
    print(f"   인증 설정 완료: {is_auth_configured()}")
    
    print("\n[2] 토큰 발급 테스트")
    token = get_access_token()
    if token:
        print(f"   토큰: {token[:50]}...")
    else:
        print("   토큰 발급 실패 (인증 없이 계속)")
    
    print("\n[3] ETL 통계 조회 테스트")
    try:
        stats = get_etl_data_statistics()
        print(f"   통계: {stats}")
    except Exception as e:
        print(f"   실패: {e}")
    
    print("\n[4] 최근 데이터 조회 테스트 (ds_0101)")
    try:
        data = get_recent_etl_data('ds_0101', 5)
        if data:
            print(f"   조회된 데이터: {len(data)}건")
            if isinstance(data, list) and len(data) > 0:
                print(f"   첫 번째 항목 키: {list(data[0].keys()) if isinstance(data[0], dict) else type(data[0])}")
    except Exception as e:
        print(f"   실패: {e}")
    
    print("\n" + "="*60)
    print("✅ 테스트 완료")
    print("="*60)
