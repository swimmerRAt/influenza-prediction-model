# 인플루엔자 예측 모델 (PatchTST)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://www.postgresql.org/)

PostgreSQL에서 인플루엔자 데이터를 로드해 19-49세 연령대 ILI를 예측하는 PatchTST 기반 모델입니다. 멀티스케일 CNN 패치 임베딩, Transformer 인코더, attention pooling, 피크 민감 손실을 결합해 주간 ILI 3주 앞을 예측합니다.

## 핵심 요약

- 메인 실행 파일: [suyeong_patchTST.py](suyeong_patchTST.py)
- 입력 피처: `ili`, `delta_ili`, `peak_gap`
- 예측 길이: 3주 (`PRED_LEN=3`)
- 팬데믹 기간 처리: 2020년 14주 ~ 2022년 22주 구간을 2017-2019 주차 평균 패턴으로 보간
- 출력: 예측 CSV, 학습 곡선/재구성 플롯, 피처 중요도

## 데이터 흐름

1) PostgreSQL에서 `influenza_data` 로드 (실패 시 [merged_influenza_data.csv](merged_influenza_data.csv) 사용)
2) 19-49세 연령대 필터링
3) 팬데믹 기간 보간 → 결측값 보정
4) 파생 피처 생성
   - `delta_ili`: 주간 변화량
   - `peak_gap`: 직전 8주 최대값 대비 현재 ILI 차이
5) PatchTST 학습 및 평가

## 실행 방법

```bash
python suyeong_patchTST.py
```

## 의존성

```bash
pip install -r requirements.txt
```

PostgreSQL 연결 설정은 [database/db_utils.py](database/db_utils.py)의 환경 변수를 사용합니다. 연결이 실패하면 CSV로 자동 대체합니다.

## 출력 파일

- [ili_predictions.csv](ili_predictions.csv) - 예측 결과
- [feature_importance.csv](feature_importance.csv) - 피처 중요도 테이블
- [feature_importance.png](feature_importance.png) - 피처 중요도 시각화
- [plot_last_window.png](plot_last_window.png) - 마지막 윈도우 예측
- [plot_test_reconstruction.png](plot_test_reconstruction.png) - 테스트 재구성
- [plot_ma_curves.png](plot_ma_curves.png) - MAE 학습 곡선

## 모델 개요

- 멀티스케일 CNN 패치 임베딩 (k=1/3/5/7 + dilation 분기)
- TokenConvMixer로 패치 토큰 간 로컬 연속성 강화
- Transformer Encoder + attention pooling
- 피크 과소예측 패널티 포함한 amplitude-aware MSE

## 주요 하이퍼파라미터

```python
# 시퀀스 설정
SEQ_LEN = 12
PRED_LEN = 3
PATCH_LEN = 4
STRIDE = 1

# 모델 구조
D_MODEL = 64
N_HEADS = 4
ENC_LAYERS = 3
FF_DIM = 64
DROPOUT = 0.2

# 학습 설정
EPOCHS = 200
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 5e-3
```

## 팬데믹 기간 처리

팬데믹 구간은 제거하지 않고 보간합니다. 2017-2019년 주차별 평균 패턴으로 대체하여 계절성을 유지합니다. 관련 로직은 [suyeong_patchTST.py](suyeong_patchTST.py) 내 `interpolate_pandemic_period()` 함수에 구현되어 있습니다.

## 실험 로깅

WandB 로깅이 기본 활성화입니다. 실행 환경에 따라 `wandb` 설정이 필요할 수 있습니다.

- **2017년 ~ 2025년** (9년간)
- **주간 단위** 시계열 데이터
- **13개 데이터셋** 통합

### 데이터 로딩 프로세스

#### 1. PostgreSQL에서 데이터 로드 (기본)

모델은 자동으로 PostgreSQL 데이터베이스를 사용합니다:

```python
from database.db_utils import load_from_postgres

# patchTST.py에서 자동 호출
df = load_data_from_postgres()

# 환경 변수를 통해 PostgreSQL 연결 정보 설정 (.env 파일):
# PG_HOST=localhost
# PG_PORT=5432
# PG_DB=influenza
# PG_USER=postgres
# PG_PASSWORD=postgres
```

**로딩 흐름**:
```
1. 환경 변수 확인 (USE_DUCKDB=true)
   ↓
2. DuckDB 파일 존재 확인
   ↓
3. SQL 쿼리로 데이터 로드
   SELECT * FROM influenza_data
   ↓
4. Pandas DataFrame 반환 (0.78초, 300만 행)
   ↓
5. 실패 시 CSV 폴백
```

#### 2. 데이터베이스 업데이트 프로세스

새로운 데이터를 데이터베이스에 추가하는 방법:

```bash
# update_database.py 실행
python database/update_database.py
```

**업데이트 흐름**:
```
1. data/before 폴더의 CSV 파일 스캔
   (flu-0101-2017.csv, flu-0101-2018.csv, ...)
   ↓
2. 모든 CSV 파일 로드 및 병합
   - pandas.concat() 사용
   - ignore_index=True로 재인덱싱
   ↓
3. 중복 제거
   - drop_duplicates() 적용
   - year, week 기준 정렬
   ↓
4. PostgreSQL에 저장
   - CREATE TABLE IF NOT EXISTS
   - INSERT ON CONFLICT DO NOTHING
   - 인덱스 생성 (year, week)
   ↓
5. 데이터베이스 최적화
   - VACUUM 명령 실행
   - 통계 업데이트
```

### 데이터베이스 전처리 파이프라인

#### 1. 원본 데이터 → PostgreSQL 변환

```python
from database.db_utils import TimeSeriesDB

# CSV를 PostgreSQL로 변환
with TimeSeriesDB() as db:  # 환경 변수에서 연결 정보 자동 로드
    db.import_csv_to_db(
        csv_path="data/merged/merged_influenza_data.csv",
        table_name="influenza_data"
    )
```

**변환 과정**:
```
CSV 파일
   ↓
1. pandas.read_csv() 
   - 청크 단위 읽기 (메모리 효율)
   ↓
2. 데이터 타입 최적화
   - int64 → int32 (메모리 절약)
   - object → category (문자열 압축)
   ↓
3. PostgreSQL INSERT
   - Batch insert
   - 트랜잭션 사용
   ↓
4. 인덱싱
   - CREATE INDEX ON year, week
   ↓
PostgreSQL 테이블 (influenza_data)
```

#### 2. 데이터베이스 내 전처리 (SQL 기반)

PostgreSQL에서 SQL로 직접 전처리 가능:

```python
# 특정 연도만 필터링
df = load_from_postgres(
    where="year >= 2020 AND year <= 2023"
)

# 특정 컬럼만 선택 (메모리 절약)
df = load_from_postgres(
    columns=['year', 'week', 'ili', 'vaccine_rate']
)

# 집계 쿼리 (연도별 평균)
with TimeSeriesDB("database/influenza_data.duckdb") as db:
    result = db.conn.execute("""
        SELECT year, AVG(ili) as avg_ili
        FROM influenza_data
        GROUP BY year
        ORDER BY year
    """).fetchdf()
```

**SQL 전처리의 장점**:
- 메모리 효율: 필요한 데이터만 로드
- 속도: 데이터베이스 엔진 최적화
- 유연성: 복잡한 필터링 및 집계

### 모델 입력을 위한 전처리

DuckDB에서 로드한 후 모델 학습을 위한 추가 전처리:

#### 1. 주간 → 일간 보간 (`weekly_to_daily_interp`)

```python
# patchTST.py의 load_and_prepare() 함수에서 수행

# 주간 데이터를 일간으로 변환
df_daily = weekly_to_daily_interp(
    df,
    date_col="label",
    target_col="ili",
    method="cubic"  # Cubic spline interpolation
)
```

**보간 과정**:
```
주간 데이터 (52 rows/year)
   ↓
1. 날짜 파싱 (2023-2024 W15 → datetime)
   ↓
2. Cubic Spline 보간
   - scipy.interpolate.CubicSpline
   - 부드러운 곡선 생성
   ↓
3. 일간 데이터 생성 (365 rows/year)
   ↓
4. 누락값 처리 (forward fill)
```

#### 2. 특징 선택 (Feature Engineering)

```python
# 자동 특징 선택 (use_exog="auto")
if use_exog == "auto":
    # 백신 데이터 확인
    has_vax = "vaccine_rate" in df.columns
    
    # 호흡기 데이터 확인
    has_resp = "respiratory_index" in df.columns
    
    # 기후 특징 추출
    climate_feats = [c for c in df.columns 
                     if any(k in c.lower() for k in 
                     ['temp', 'humid', 'rain', 'wind'])]
    
    # 최종 특징 조합
    features = ["ili"]
    if has_vax: features.append("vaccine_rate")
    if has_resp: features.append("respiratory_index")
    features.extend(climate_feats)
```

**특징 선택 전략**:
- `use_exog="auto"`: 사용 가능한 모든 특징 (기본값)
- `use_exog="none"`: ILI만 사용
- `use_exog="vax"`: ILI + 백신
- `use_exog="resp"`: ILI + 호흡기
- `use_exog="both"`: ILI + 백신 + 호흡기
- `use_exog="all"`: 모든 특징 + 기후

#### 3. 정규화 (Normalization)

```python
from sklearn.preprocessing import RobustScaler

# Train/Val/Test 분할 후 정규화
scaler_x = RobustScaler()  # 특징 정규화
scaler_y = RobustScaler()  # 타겟 정규화

# Train 데이터로 fit
X_train_scaled = scaler_x.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

# Val/Test는 transform만
X_val_scaled = scaler_x.transform(X_val)
y_val_scaled = scaler_y.transform(y_val)
```

**RobustScaler 사용 이유**:
- 중앙값과 IQR 사용 (이상치에 강건)
- 공식: `(X - median) / IQR`
- StandardScaler보다 안정적

#### 4. 시퀀스 생성 (Sequence Generation)

```python
# PatchTSTDataset에서 시퀀스 생성
class PatchTSTDataset:
    def __getitem__(self, i):
        # 입력 시퀀스 (12주)
        seq_X = self.X[i:i+self.seq_len, :]  # (12, F)
        
      # 타겟 (4주)
      seq_y = self.y[i+self.seq_len:i+self.seq_len+self.pred_len]  # (4,)
        
        # 패치 분할 (12 → 3 patches × 4 timesteps)
        patches = []
        for j in range(0, self.seq_len, self.patch_len):
            patch = seq_X[j:j+self.patch_len, :]  # (4, F)
            patches.append(patch)
        
        return X_patch, seq_y, label
```

**시퀀스 예시**:
```
원본 데이터: [Week 1, Week 2, ..., Week 100]
              ↓
시퀀스 1: 
  입력: [Week 1-12]  (12주)
   타겟: [Week 13-16] (4주)
  
시퀀스 2:
  입력: [Week 2-13]  (12주)
   타겟: [Week 14-17] (4주)
  
... (슬라이딩 윈도우)
```

### 전체 데이터 파이프라인 요약

```
📂 원본 데이터 (CSV files in data/before/)
   ↓
💾 [데이터베이스 업데이트]
   - 병합 및 중복 제거
   - PostgreSQL 저장
   ↓
🔍 [데이터 로딩]
   - PostgreSQL에서 SQL 쿼리
   - DataFrame 반환
   ↓
🚨 [데이터 필터링]
   - 팬데믹 기간 자동 제외
   - (2020-W14 ~ 2022-W22)
   ↓
📊 [전처리 1: 시간 변환]
   - 주간 → 일간 보간
   - Cubic spline 사용
   ↓
🎯 [전처리 2: 특징 선택]
   - ILI, 백신, 호흡기, 기후
   - use_exog 설정에 따라
   ↓
📈 [전처리 3: 정규화]
   - RobustScaler 적용
   - Train/Val/Test 분할
   ↓
🔢 [전처리 4: 시퀀스 생성]
   - 12주 입력 → 4주 예측
   - 패치 분할 (4 timesteps)
   ↓
🤖 [모델 학습]
   - PatchTST 모델
   - Transformer 기반
   ↓
📉 [예측 결과]
   - ili_predictions.csv
   - 시각화 그래프
```

### 데이터 품질 관리

#### 결측값 처리

```python
# 1. 수치형 컬럼: Forward fill
df_numeric = df.select_dtypes(include=[np.number])
df_numeric = df_numeric.fillna(method='ffill')

# 2. 그 이후: Backward fill
df_numeric = df_numeric.fillna(method='bfill')

# 3. 남은 결측값: 중앙값
df_numeric = df_numeric.fillna(df_numeric.median())
```

#### 이상치 탐지

```python
# IQR 방식으로 이상치 탐지
Q1 = df['ili'].quantile(0.25)
Q3 = df['ili'].quantile(0.75)
IQR = Q3 - Q1

# 이상치 플래그 (제거하지 않고 플래깅만)
outliers = (df['ili'] < Q1 - 1.5*IQR) | (df['ili'] > Q3 + 1.5*IQR)
```

#### 데이터 검증

```bash
# 데이터베이스 무결성 확인
python database/check_database.py

# 출력:
# - 총 행 수
# - 컬럼 정보
# - 연도별 데이터 개수
# - 결측값 통계
# - 데이터 샘플
```

### 데이터 분할

```python
# 시계열 순서 유지하며 분할
Train:      70%  (초기 데이터)
Validation: 15%  (중간 데이터)
Test:       15%  (최신 데이터)
```

### 주요 컬럼 설명

| 컬럼명 | 설명 | 타입 |
|--------|------|------|
| `year` | 연도 | int |
| `week` | 주차 (1-52) | int |
| `ili` | 인플루엔자 유사질환 비율 | float |
| `vaccine_rate` | 백신 접종률 | float |
| `respiratory_index` | 호흡기 질환 지수 | float |
| `temperature` | 평균 온도 (°C) | float |
| `humidity` | 상대 습도 (%) | float |
| `rainfall` | 강수량 (mm) | float |
| `dataset_id` | 데이터 출처 (ds_0101 등) | str |

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터베이스 준비

```bash
# 데이터베이스 업데이트 (선택사항 - 이미 최신 DB 포함)
python database/update_database.py

# 데이터베이스 무결성 검증
python database/validate_database.py

# 데이터베이스 내용 확인
python database/check_database.py
```

### 3. PatchTST 모델 학습

**모델은 PostgreSQL에서 데이터를 자동으로 로드하고 전처리합니다:**

```bash
# 기본 학습 (PostgreSQL 사용, 전체 데이터)
python patchTST.py

# 사용 가능한 연령대/아형 확인
python patchTST.py --list-options

# 특정 연령대로 학습 (원본 CSV 사용)
python patchTST.py --age-group 19-49세 --raw-data

# 특정 아형으로 학습
python patchTST.py --subtype A --raw-data

# 아형별 검출률 예측 모드 (ds_0107 데이터만 사용)
python patchTST.py --subtype-only --subtype A
```

**명령줄 옵션:**
| 옵션 | 설명 | 예시 |
|------|------|------|
| `--age-group` | 연령대 선택 | `--age-group 65세이상` |
| `--subtype` | 아형 선택 (A/B) | `--subtype A` |
| `--subtype-only` | 아형별 검출률만 예측 | `--subtype-only --subtype B` |
| `--raw-data` | 원본 CSV 사용 | `--raw-data` |
| `--data-dir` | 원본 데이터 디렉토리 | `--data-dir data/before` |
| `--list-options` | 사용 가능한 옵션 확인 | `--list-options` |

**데이터 파이프라인 (자동 처리)**:
1. **데이터 로드**: 
   - PostgreSQL (기본): 4,983행 × 9열 데이터
   - 로컬 아카이브 (`--raw-data`): `data/before` 폴더에 저장된 과거 API로 수집한 CSV를 사용합니다. 이 데이터를 PostgreSQL 데이터와 병합해 시계열을 확장하거나, PostgreSQL 접근이 불가능한 경우 대체 데이터로 사용할 수 있습니다.
2. **연령대 선택**: 환경변수 `AGE_GROUP` 또는 `--age-group` 옵션으로 지정
   - 0-6세, 7-12세, 13-18세, 19-49세, 50-64세, 65세이상
   - 미지정 시 전체 데이터 사용
3. **아형 선택**: 환경변수 `SUBTYPE` 또는 `--subtype` 옵션으로 지정
   - A, B (미지정 시 우세 아형 자동 선택)
4. **팬데믹 기간 제외**: 2020년 14주 ~ 2022년 22주 자동 필터링
5. **컨럼 매핑**: 한국어 → 영어
   - `연도` → `year`, `주차` → `week`
   - `의사환자 분율` → `ili` (target variable)
   - `예방접종률` → `vaccine_rate`
   - `입원환자 수` → `hospitalization`
   - `인플루엔자 검출률` → `detection_rate`
   - `응급실 인플루엔자 환자` → `emergency_patients`
6. **예방접종률 Fallback**: 연령대별 데이터 없으면 전국 평균 사용
7. **결측치 처리**: 선형 보간 + median 채우기
8. **주기성 특징**: `week_sin`, `week_cos` 추가
9. **모델 학습**: PatchTST Transformer 학습 (100 에포크)
10. **예측 및 평가**: Test set에서 성능 평가
11. **Feature Importance**: 특징 중요도 계산
12. **자동 종료**: 모든 결과 저장 후 프로그램 자동 종료

**최종 특징 벡터** (7차원):
- `ili`: 의사환자 분율 (타겟)
- `hospitalization`: 입원환자 수
- `detection_rate`: 인플루엔자 검출률
- `emergency_patients`: 응급실 인플루엔자 환자
- `vaccine_rate`: 예방접종률 (연령대별 또는 전국 평균)
- `week_sin`, `week_cos`: 주기성 특징

**학습 시간**: 약 5~10분 (MPS/GPU 사용 시)

### 4. 예측 결과 확인

**학습 완료 확인:**
- 터미널 마지막 줄에 `Feature Importance saved to ...` 출력
- 프로그램이 자동으로 종료되며 터미널 프롬프트 복귀
- **수동 종료 불필요** (이전 버전과 달리 자동 종료됨)

> **⚠️ 중요**: 이전 버전에서는 `plt.show()` 때문에 Ctrl+C로 수동 종료가 필요했으나,
> 현재 버전은 모든 그래프를 파일로 저장 후 **자동으로 종료**됩니다.

**생성된 파일:**

```
/Volumes/ExternalSSD/Workspace/influenza-prediction-model/
├── ili_predictions.csv              # 예측 결과 CSV
├── plot_ma_curves.png               # MAE/Loss 곡선
├── plot_last_window.png             # 마지막 윈도우 예측
├── plot_test_reconstruction.png     # 테스트 재구성
├── feature_importance.csv           # Feature Importance 데이터
└── feature_importance.png           # Feature Importance 그래프
```

**성능 지표:**
```
Best Validation MAE: 11.95
Test MAE: 18.00
```

## 📊 예측 결과 예시

```python
# ili_predictions.csv 구조
date,actual,predicted,residual
2024-11-01,0.023,0.025,-0.002
2024-11-08,0.028,0.027,0.001
2024-11-15,0.031,0.030,0.001
```

## 🔧 환경 변수 (.env)

```bash
# ========================================
# 모델 설정
# ========================================
# 연령대 선택: 0-6세, 7-12세, 13-18세, 19-49세, 50-64세, 65세이상
# 비워두면 전체 데이터 사용 (기본값)
AGE_GROUP=

# 아형 선택: A, B
# 비워두면 우세 아형 자동 선택 (기본값)
SUBTYPE=

# 아형별 예측 모드: true/false
# true시 ds_0107 데이터만 사용
SUBTYPE_ONLY=false

# 원본 CSV 데이터 사용 여부: true/false
# true시 PostgreSQL 대신 data/before 폴더의 CSV 직접 사용
USE_RAW_DATA=false

# 원본 데이터 디렉토리
DATA_DIR=data/before

# ========================================
# 데이터베이스 설정 (PostgreSQL)
# ========================================
PG_HOST=localhost
PG_PORT=5432
PG_DB=influenza
PG_USER=postgres
PG_PASSWORD=postgres
```

### 환경변수 vs 명령줄 인자

명령줄 인자가 환경변수보다 우선합니다.

```bash
# 환경변수로 설정 (.env 파일)
AGE_GROUP=19-49세
USE_RAW_DATA=true

# 또는 명령줄로 설정 (환경변수 무시)
python patchTST.py --age-group 65세이상 --raw-data
```

## 📚 추가 문서

- [USAGE.md](USAGE.md) - 상세 사용 가이드
- [doc/DUCKDB_GUIDE.md](doc/DUCKDB_GUIDE.md) - DuckDB 사용법
- [doc/QUICKSTART.md](doc/QUICKSTART.md) - 빠른 시작 가이드

## 🛠️ 기술 스택

- **언어**: Python 3.10
- **딥러닝**: PyTorch 2.0+
- **데이터 처리**: Pandas, NumPy
- **데이터베이스**: DuckDB 1.4.3
- **시각화**: Matplotlib
- **환경 관리**: Conda, python-dotenv

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.

## 🤝 기여

이슈 및 풀 리퀘스트를 환영합니다!

---

**개발 환경**: macOS, M-series chip  
**마지막 업데이트**: 2026년 1월 26일
