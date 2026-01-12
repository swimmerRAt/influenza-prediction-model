# API 모드 사용 예시

## 1. 환경 설정

`.env` 파일 설정:
```bash
# API 모드 활성화
USE_API=true

# API 서버 URL (Node.js 서버 주소)
API_URL=http://localhost:3000

# 데이터셋 ID
DSID=ds_0101
```

## 2. 서버 실행

터미널 1에서 Node.js API 서버 실행:
```bash
npm start
```

출력 예시:
```
Server listening on port 3000
```

## 3. Python 모델 실행

터미널 2에서 Python 스크립트 실행:
```bash
python patchTST.py
```

출력 예시:
```
============================================================
🌐 API 모드: 서버에서 데이터를 실시간으로 가져옵니다.
============================================================
API에서 데이터 가져오는 중... (dsid: ds_0101)
총 3개의 페이지 파일을 받았습니다.
총 1500개의 레코드를 가져왔습니다.
✅ API로부터 데이터 로드 완료: (1500, 10)
📊 컬럼: ['date', 'ili', 'vaccine_rate', 'case_count', ...]
USE_EXOG = 'all'  (auto-detects vaccine/resp columns)
Data points: 1500 | Features used (8): ['ili', 'vaccine_rate', ...]
```

## 4. CSV 모드로 전환하기

`.env` 파일 수정:
```bash
# CSV 모드로 변경
USE_API=false
```

Python 스크립트 재실행:
```bash
python patchTST.py
```

출력 예시:
```
============================================================
📁 CSV 모드: 로컬 파일에서 데이터를 로드합니다.
============================================================
Using CSV: 3_merged_influenza_vaccine_respiratory_weather.csv | Device: cpu
CSV 파일 로드 완료: .../3_merged_influenza_vaccine_respiratory_weather.csv, (1500, 10)
```

## 5. 다른 데이터셋 사용하기

`.env` 파일에서 DSID 변경:
```bash
DSID=ds_0202
USE_API=true
```

또는 Python 코드에서 직접 지정:
```python
# patchTST.py 내에서
df = fetch_data_from_api(dsid='ds_0202')
```

## 6. 트러블슈팅

### API 서버 연결 실패
```
API 서버 연결 실패: Connection refused
```
**해결 방법**: Node.js 서버가 실행 중인지 확인
```bash
npm start
```

### 데이터셋 ID 오류
```
dsid가 제공되지 않았습니다.
```
**해결 방법**: `.env` 파일에 DSID 설정 또는 함수 인자로 전달

### 패키지 누락 오류
```
ModuleNotFoundError: No module named 'requests'
```
**해결 방법**: 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 7. 고급 사용법

### 프로그래밍 방식으로 API 데이터 가져오기

```python
from patchTST import fetch_data_from_api, load_and_prepare

# API에서 데이터 가져오기
df = fetch_data_from_api(dsid='ds_0101')

# 데이터 확인
print(df.head())
print(df.columns)
print(df.shape)

# 모델 입력 데이터로 변환
X, y, labels, feat_names = load_and_prepare(df=df, use_exog='all')
```

### 여러 데이터셋 비교

```python
# 여러 데이터셋 로드
datasets = {}
for dsid in ['ds_0101', 'ds_0202', 'ds_0303']:
    try:
        df = fetch_data_from_api(dsid=dsid)
        datasets[dsid] = df
        print(f"{dsid}: {df.shape}")
    except Exception as e:
        print(f"{dsid} 로드 실패: {e}")

# 각 데이터셋으로 모델 학습
for dsid, df in datasets.items():
    print(f"\n{'='*60}")
    print(f"Training model with {dsid}")
    print(f"{'='*60}")
    X, y, labels, feat_names = load_and_prepare(df=df)
    # train_and_eval(X, y, labels, feat_names)
```
