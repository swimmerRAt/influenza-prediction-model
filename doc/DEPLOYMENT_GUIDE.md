# 📦 PatchTST 모델 배포 가이드

## 목차
- [필수 파일](#필수-파일)
- [배포 시나리오](#배포-시나리오)
- [환경 변수 설정](#환경-변수-설정)
- [배포 체크리스트](#배포-체크리스트)

---

## 필수 파일

### 1️⃣ Python 모델 관련

| 파일명 | 필수 여부 | 설명 |
|--------|----------|------|
| `patchTST.py` | ✅ 필수 | 메인 모델 스크립트 |
| `requirements.txt` | ✅ 필수 | Python 의존성 패키지 목록 |
| `.env` | ✅ 필수 | 환경 변수 설정 파일 (`.env.example` 복사 후 수정) |

### 2️⃣ Node.js API 서버 관련 (API 모드 사용 시)

| 파일명 | 필수 여부 | 설명 |
|--------|----------|------|
| `server.js` | ✅ 필수 | Express API 서버 |
| `package.json` | ✅ 필수 | Node.js 의존성 및 스크립트 |
| `src/auth.js` | ✅ 필수 | Keycloak 인증 모듈 |
| `src/gfidClient.js` | ✅ 필수 | GFID API 클라이언트 |

### 3️⃣ 문서 (선택)

| 파일명 | 필수 여부 | 설명 |
|--------|----------|------|
| `RATE_LIMITER_GUIDE.md` | 📄 선택 | Rate Limiter 사용 가이드 |
| `API_USAGE_GUIDE.md` | 📄 선택 | API 사용 가이드 |
| `doc/README.md` | 📄 선택 | 프로젝트 설명서 |
| `.env.example` | 📄 선택 | 환경 변수 예시 템플릿 |

### 4️⃣ 개발/테스트 파일 (선택)

| 파일명 | 필수 여부 | 설명 |
|--------|----------|------|
| `patchTST_suyeong.ipynb` | 📄 선택 | Jupyter 노트북 (개발/테스트용) |
| `GFID-DATA-API.postman_collection.json` | 📄 선택 | Postman API 테스트 컬렉션 |
| `KEYCLOAK.postman_environment.json` | 📄 선택 | Postman 환경 설정 |

### 5️⃣ 데이터 파일 (CSV 모드 사용 시)

| 파일명 | 필수 여부 | 설명 |
|--------|----------|------|
| `3_merged_influenza_vaccine_respiratory_weather.csv` | 📄 선택 | CSV 모드 사용 시 필수 |
| `merge_data.csv` | 📄 선택 | 대체 데이터 파일 |
| `data/` | 📄 선택 | 데이터 폴더 |

---

## 배포 시나리오

### 시나리오 1: API 모드 배포 (권장) 🌐

서버에서 실시간으로 데이터를 가져오는 방식

#### 폴더 구조
```
배포_폴더/
├── patchTST.py                    ✅ 필수
├── requirements.txt               ✅ 필수
├── server.js                      ✅ 필수
├── package.json                   ✅ 필수
├── src/
│   ├── auth.js                    ✅ 필수
│   └── gfidClient.js              ✅ 필수
├── .env                           ✅ 필수 (직접 생성)
├── RATE_LIMITER_GUIDE.md          📄 선택
└── API_USAGE_GUIDE.md             📄 선택
```

#### 배포 단계

**1단계: 파일 복사**
```bash
# 필수 파일들을 배포 서버로 복사
scp patchTST.py requirements.txt server.js package.json user@server:/app/
scp -r src/ user@server:/app/
```

**2단계: Node.js 의존성 설치**
```bash
cd /app
npm install
```

**3단계: Python 의존성 설치**
```bash
pip install -r requirements.txt
```

**4단계: 환경 변수 설정**
```bash
cp .env.example .env
nano .env  # 또는 vi .env
# Keycloak 정보, API 설정 등을 입력
```

**5단계: API 서버 시작**
```bash
# 백그라운드로 실행
npm start &

# 또는 PM2 사용 (권장)
pm2 start server.js --name "gfid-api-server"
```

**6단계: Python 모델 실행**
```bash
python patchTST.py
```

---

### 시나리오 2: CSV 모드 배포 📁

로컬 CSV 파일을 사용하는 방식

#### 폴더 구조
```
배포_폴더/
├── patchTST.py                    ✅ 필수
├── requirements.txt               ✅ 필수
├── 3_merged_influenza_vaccine_respiratory_weather.csv  ✅ 필수
├── .env                           ✅ 필수 (USE_API=false)
└── RATE_LIMITER_GUIDE.md          📄 선택
```

#### 배포 단계

**1단계: 파일 복사**
```bash
scp patchTST.py requirements.txt user@server:/app/
scp 3_merged_influenza_vaccine_respiratory_weather.csv user@server:/app/
```

**2단계: Python 의존성 설치**
```bash
cd /app
pip install -r requirements.txt
```

**3단계: 환경 변수 설정**
```bash
echo "USE_API=false" > .env
```

**4단계: Python 모델 실행**
```bash
python patchTST.py
```

---

### 시나리오 3: Docker 배포 (권장) 🐳

컨테이너화된 배포 방식

#### Dockerfile 생성

`Dockerfile` 파일을 생성하세요:

```dockerfile
FROM python:3.9-slim

# Node.js 설치 (API 모드용)
RUN apt-get update && \
    apt-get install -y nodejs npm && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 파일 복사
COPY patchTST.py .
COPY requirements.txt .
COPY server.js .
COPY package.json .
COPY src/ src/
COPY .env .

# Python 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# Node.js 의존성 설치
RUN npm install

# 포트 노출
EXPOSE 3000

# 실행 명령
CMD ["sh", "-c", "npm start & python patchTST.py"]
```

#### docker-compose.yml 생성 (선택)

`docker-compose.yml` 파일을 생성하세요:

```yaml
version: '3.8'

services:
  patchtst:
    build: .
    container_name: patchtst-model
    ports:
      - "3000:3000"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./output:/app/output
    restart: unless-stopped
```

#### 배포 단계

**1단계: Docker 이미지 빌드**
```bash
docker build -t patchtst-model:latest .
```

**2단계: 컨테이너 실행**
```bash
# 단일 컨테이너
docker run -d --name patchtst -p 3000:3000 --env-file .env patchtst-model:latest

# 또는 Docker Compose 사용
docker-compose up -d
```

**3단계: 로그 확인**
```bash
docker logs -f patchtst
```

---

## 환경 변수 설정

### .env 파일 예시

배포 전 `.env` 파일을 반드시 생성하고 다음 값들을 설정하세요:

```bash
# =========================================
# Keycloak 인증 정보
# =========================================
SERVER_URL=https://keycloak.211.238.12.60.nip.io:8100
REALM=gfid-api
CLIENT_ID=your_client_id_here
CLIENT_SECRET=your_client_secret_here

# 또는 수동 토큰 사용 (Keycloak 연결 불가 시)
# ACCESS_TOKEN=your_manual_access_token_here

# =========================================
# 데이터 설정
# =========================================
# API 모드 사용 여부 (true: API에서 로드, false: CSV 파일 사용)
USE_API=true

# 데이터셋 설정
USE_SINGLE_DATASET=false  # true: 단일 데이터셋, false: 전체 데이터셋
DSID=ds_0101              # 단일 데이터셋 사용 시 ID
FROM=2025-01-01           # 시작 날짜
TO=2025-12-31             # 종료 날짜

# =========================================
# API 서버 설정
# =========================================
API_URL=http://localhost:3000
PORT=3000
GFID_API_BASE=http://211.238.12.60:8084/data/api/v1
GFID_ITEMS_KEY=body.data

# =========================================
# 서버 과부하 방지 설정 (Rate Limiting)
# =========================================
# 요청 간 초기 대기 시간 (초) - 기본값: 1.0
RATE_LIMIT_INITIAL_DELAY=1.0

# 최대 대기 시간 (초) - 서버 과부하 시 점진적으로 증가 - 기본값: 30.0
RATE_LIMIT_MAX_DELAY=30.0

# 최소 대기 시간 (초) - 서버가 안정적일 때 최소값 - 기본값: 0.5
RATE_LIMIT_MIN_DELAY=0.5

# 최대 재시도 횟수 - 네트워크 오류 시 자동 재시도 - 기본값: 5
RATE_LIMIT_MAX_RETRIES=5
```

### 환경별 권장 설정

#### 개발 환경
```bash
USE_API=true
RATE_LIMIT_INITIAL_DELAY=0.5
RATE_LIMIT_MAX_DELAY=10.0
RATE_LIMIT_MAX_RETRIES=3
```

#### 테스트 환경
```bash
USE_API=true
RATE_LIMIT_INITIAL_DELAY=1.0
RATE_LIMIT_MAX_DELAY=30.0
RATE_LIMIT_MAX_RETRIES=5
```

#### 프로덕션 환경 (안정적인 서버)
```bash
USE_API=true
RATE_LIMIT_INITIAL_DELAY=1.0
RATE_LIMIT_MAX_DELAY=60.0
RATE_LIMIT_MIN_DELAY=0.5
RATE_LIMIT_MAX_RETRIES=10
```

#### 프로덕션 환경 (불안정한 서버)
```bash
USE_API=true
RATE_LIMIT_INITIAL_DELAY=3.0
RATE_LIMIT_MAX_DELAY=120.0
RATE_LIMIT_MIN_DELAY=1.0
RATE_LIMIT_MAX_RETRIES=15
```

---

## 배포 체크리스트

### 사전 준비

- [ ] **파일 준비 완료**
  - [ ] `patchTST.py` 파일 확인
  - [ ] `requirements.txt` 파일 확인
  - [ ] API 모드 사용 시: `server.js`, `package.json`, `src/` 폴더 확인
  - [ ] CSV 모드 사용 시: 데이터 CSV 파일 확인

- [ ] **환경 설정**
  - [ ] `.env.example`을 `.env`로 복사
  - [ ] Keycloak 인증 정보 입력 (CLIENT_ID, CLIENT_SECRET)
  - [ ] 데이터 설정 확인 (USE_API, DSID, FROM, TO)
  - [ ] Rate Limiter 설정 확인

### 의존성 설치

- [ ] **Python 의존성**
  ```bash
  pip install -r requirements.txt
  ```
  - [ ] pandas
  - [ ] numpy
  - [ ] torch
  - [ ] scikit-learn
  - [ ] requests
  - [ ] python-dotenv

- [ ] **Node.js 의존성** (API 모드 사용 시)
  ```bash
  npm install
  ```
  - [ ] express
  - [ ] axios
  - [ ] dotenv

### 서버 실행

- [ ] **API 서버 시작** (API 모드 사용 시)
  ```bash
  npm start
  # 또는
  pm2 start server.js --name "gfid-api-server"
  ```

- [ ] **API 서버 동작 확인**
  ```bash
  curl http://localhost:3000/health
  ```

### 테스트 실행

- [ ] **모델 테스트 실행**
  ```bash
  python patchTST.py
  ```

- [ ] **출력 확인**
  - [ ] 데이터 로딩 성공 확인
  - [ ] 모델 학습 진행 확인
  - [ ] 에러 없이 완료 확인

### 배포 후 점검

- [ ] **로그 모니터링**
  - [ ] API 서버 로그 확인
  - [ ] Python 실행 로그 확인
  - [ ] 에러 발생 여부 점검

- [ ] **성능 모니터링**
  - [ ] Rate Limiter 통계 확인
  - [ ] 평균 응답 시간 확인
  - [ ] 에러율 확인 (< 30% 유지)

- [ ] **보안 점검**
  - [ ] `.env` 파일 권한 확인 (600)
  - [ ] 인증 토큰 만료 시간 확인
  - [ ] API 접근 제한 설정 확인

---

## 최소 배포 파일 (Quick Start)

정말 빠르게 배포하려면 다음 7개 파일만 있으면 됩니다:

```
최소_배포/
├── patchTST.py          # 1. Python 모델
├── requirements.txt     # 2. Python 의존성
├── server.js            # 3. API 서버
├── package.json         # 4. Node.js 의존성
├── src/
│   ├── auth.js          # 5. 인증 모듈
│   └── gfidClient.js    # 6. GFID 클라이언트
└── .env                 # 7. 환경 변수
```

### Quick Start 명령어

```bash
# 1. 의존성 설치
npm install && pip install -r requirements.txt

# 2. 환경 변수 설정
cp .env.example .env
# .env 파일 편집

# 3. 실행
npm start &
python patchTST.py
```

---

## 트러블슈팅

### 문제: "No response from Keycloak token endpoint"

**원인:** Keycloak 서버 연결 실패

**해결 방법:**
1. `.env` 파일에서 `SERVER_URL` 확인
2. 방화벽/네트워크 설정 확인
3. 수동 토큰 사용:
   ```bash
   # Postman에서 토큰 발급 후
   ACCESS_TOKEN=your_token_here
   ```

### 문제: "API 서버 연결 실패"

**원인:** Node.js 서버가 실행되지 않음

**해결 방법:**
```bash
# 서버 상태 확인
ps aux | grep node

# 서버 재시작
npm start &
```

### 문제: "높은 에러율 감지"

**원인:** 서버 과부하 또는 네트워크 불안정

**해결 방법:**
```bash
# .env 파일에서 딜레이 증가
RATE_LIMIT_INITIAL_DELAY=3.0
RATE_LIMIT_MAX_DELAY=60.0
```

---

## 프로덕션 배포 권장 사항

### 1. 프로세스 관리자 사용

**PM2 사용 (권장)**
```bash
# 설치
npm install -g pm2

# API 서버 시작
pm2 start server.js --name "gfid-api"

# 재시작 설정
pm2 startup
pm2 save

# 모니터링
pm2 monit
```

### 2. 로그 관리

```bash
# PM2 로그
pm2 logs gfid-api

# Python 로그를 파일로 저장
python patchTST.py > logs/patchtst_$(date +%Y%m%d).log 2>&1
```

### 3. 자동 재시작 스크립트

`start.sh` 파일 생성:
```bash
#!/bin/bash

# API 서버 시작
pm2 start server.js --name "gfid-api"

# Python 모델 실행 (재시작 설정)
while true; do
    python patchTST.py
    echo "모델 실행 완료. 10초 후 재시작..."
    sleep 10
done
```

### 4. 모니터링 설정

- 서버 상태 모니터링 (Prometheus, Grafana)
- 로그 수집 (ELK Stack)
- 알림 설정 (에러 발생 시 이메일/슬랙 알림)

---

## 지원 및 문의

- 📧 이메일: your-email@example.com
- 📖 문서: [doc/README.md](doc/README.md)
- 🔧 이슈: GitHub Issues

---

**최종 업데이트:** 2026년 1월 12일
