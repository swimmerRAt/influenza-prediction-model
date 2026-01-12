# Influenza API Pipeline Project

간단한 Node/Express 서버로 Keycloak 인증 토큰을 관리하고 GFID 데이터를 로컬에 저장하거나 Python 모델에 직접 전달하는 파이프라인입니다.

## 빠른 시작

### 1. 환경 설정

`.env` 파일을 생성하고 필요한 값을 설정합니다. `.env.example`의 값을 참고하거나 Postman 환경 파일 `KEYCLOAK.postman_environment.json`의 값을 사용할 수 있습니다.

주요 환경 변수:
```bash
# API 서버 설정
SERVER_URL=https://keycloak.211.238.12.60.nip.io:8100
REALM=gfid-api
CLIENT_ID=Reporting_accessor
CLIENT_SECRET=your_client_secret
DSID=ds_0101
PORT=3000

# Python 모델 설정
USE_API=true          # true: API에서 데이터 가져오기, false: 로컬 CSV 사용
API_URL=http://localhost:3000
```

### 2. Node.js 서버 설치 및 실행

```bash
# 패키지 설치
npm install

# 서버 시작
npm start
```

### 3. Python 환경 설정

```bash
# Python 패키지 설치
pip install -r requirements.txt
```

## 사용 방법

### Node.js API 엔드포인트

#### 인증 상태 확인
```bash
GET /auth/status
```
현재 토큰 상태 확인

#### 개별 데이터 다운로드
```bash
POST /download
Content-Type: application/json

{
  "dsid": "ds_0101"
}
```
요청 바디의 `dsid` 또는 `.env`의 `DSID` 사용하여 `data/` 폴더에 파일 저장

예시:
```bash
curl -X POST http://localhost:3000/download \
  -H "Content-Type: application/json" \
  -d '{"dsid":"ds_0202"}'
```

#### 일괄 다운로드
```bash
POST /download-all
Content-Type: application/json

{
  "dsids": ["ds_0101", "ds_0202"]  # 선택적
}
```

예시:
```bash
curl -X POST http://localhost:3000/download-all \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Python 모델 실행

#### 주의사항
1. 랩실 서버에서 돌리지 말것 - 로컬에서 돌리기
2. 와이파이는 JBNU를 사용하기

#### 방법 1: API에서 직접 데이터 가져오기 (권장)

1. `.env` 파일에서 `USE_API=true` 설정
2. Node.js API 서버가 실행 중인지 확인
3. Python 스크립트 실행:

```bash
python patchTST.py
```

모델은 API를 통해 데이터를 자동으로 가져와 변수에 파싱한 후 학습을 진행합니다.

#### 방법 2: 로컬 CSV 파일 사용

1. `.env` 파일에서 `USE_API=false` 설정
2. 필요한 CSV 파일이 프로젝트 디렉토리에 있는지 확인
3. Python 스크립트 실행:

```bash
python patchTST.py
```

## 주요 기능

### API 모드의 장점

- ✅ **실시간 데이터**: 최신 데이터를 API에서 직접 가져옴
- ✅ **자동 파싱**: JSON 데이터를 pandas DataFrame으로 자동 변환
- ✅ **디스크 절약**: 로컬에 파일을 저장하지 않고 메모리에서 직접 처리
- ✅ **유연성**: 다양한 데이터셋(dsid)을 쉽게 전환 가능

### patchTST.py의 주요 함수

#### `fetch_data_from_api(dsid=None, api_url=None)`
API 서버를 통해 데이터를 가져오는 함수
- Parameters:
  - `dsid`: 데이터셋 ID (기본값은 환경변수 DSID)
  - `api_url`: API 서버 URL (기본값: http://localhost:3000)
- Returns: pandas DataFrame

#### `load_and_prepare(csv_path=None, use_exog="auto", df=None)`
데이터를 로드하고 전처리하는 함수
- Parameters:
  - `csv_path`: CSV 파일 경로
  - `use_exog`: 외생변수 사용 모드
  - `df`: API에서 가져온 DataFrame (API 모드에서 사용)
- Returns: (X, y, labels, feat_names)

## 워크플로우

```
API 모드:
1. patchTST.py 실행
2. fetch_data_from_api() 호출
3. Node.js API 서버에 POST /download 요청
4. API 서버가 GFID에서 데이터 다운로드
5. 다운로드된 JSON 파일을 DataFrame으로 변환
6. 모델 학습 진행

CSV 모드:
1. patchTST.py 실행
2. 로컬 CSV 파일 로드
3. 모델 학습 진행
```

## 추후 작업

- 토큰 만료 자동 갱신 로직 개선 (현재는 만료 30초 전 갱신)
- 다양한 데이터 소스 지원
- 에러 핸들링 강화
