# 인플루엔자 예측 모델 (PatchTST)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://www.postgresql.org/)


시계열 데이터 기반의 인플루엔자(ILI) 발생률 예측을 위한 PatchTST 딥러닝 모델입니다. PostgreSQL을 활용한 효율적인 대용량 데이터 관리와 Transformer 기반 예측 모델을 결합했습니다.

**최신 데이터:** 2025년 47주차(11월 말~12월 초)까지 반영
**실행:** python patchTST.py (서버 실행 불필요, 자동 종료)
**출력:** 예측 결과, 피처 중요도, 시각화 등 주요 파일 자동 생성/재생성
**주기성 피처:** week_sin, week_cos 자동 생성 및 활용
**주요 로그:** Best Val MAE, Test MAE 등 성능지표 자동 출력
**⚠️ 팬데믹 기간 제외:** 2020년 14주~2022년 22주 데이터는 학습에서 자동 제외

## 📊 프로젝트 개요

- **목적**: 인플루엔자 유사질환(ILI) 발생률 예측
- **모델**: PatchTST (Patch Time Series Transformer)
- **데이터**: 4,983행의 연령대별 시계열 데이터 (2017-2025, 16개 연령대)
- **데이터베이스**: PostgreSQL 15+ (기존 DuckDB에서 마이그레이션)
- **성능**: PostgreSQL 기반 효율적 데이터 로딩 및 트랜잭션 관리
- **비교 모델**: Seasonal SEIRS 수리모델 대비 **우수한 성능** 입증
- **최신 업데이트**: 2026년 1월 - PostgreSQL 전환 및 팬데믹 기간 데이터 필터링

## 🗂️ 프로젝트 구조

```
influenza-prediction-model/
├── patchTST.py                         # 🎯 메인 모델 파일 (학습 & 예측)
├── seasonal_seirs_model.py             # 📐 비교 모델 (Seasonal SEIRS)
├── requirements.txt                    # 📦 Python 패키지 의존성
├── .env                                # ⚙️ 환경 변수 설정
│
├── database/                           # 💾 데이터베이스 관리
│   ├── db_utils.py                    # PostgreSQL 유틸리티 함수
│   ├── update_database.py             # DB 업데이트 스크립트
│   ├── check_database.py              # DB 조회 스크립트
│   └── validate_database.py           # DB 무결성 검증 스크립트
│
├── data/                               # 📂 데이터 저장소
│   └── before/                        # 과거 원본 데이터 (CSV)
│       ├── flu-0101-2017.csv
│       ├── flu-0101-2018.csv
│       └── ...
│
├── doc/                                # 📚 문서
│   └── QUICKSTART.md                  # 빠른 시작 가이드
│
└── output/                             # 📈 출력 결과
    ├── ili_predictions.csv            # 예측 결과
    ├── plot_loss_curves.png           # 손실 곡선
    ├── plot_predictions.png           # 예측 시각화
    ├── seirs_model_results.png        # SEIRS 모델 결과
    └── plot_ma_curves.png             # 이동평균 곡선
```


## 💾 데이터 및 실행 요약

- **최신 데이터:** 2025년 47주차(11월 말~12월 초)까지 반영
- **실행 방법:**
   ```bash
   python patchTST.py
   ```
   (서버 실행 불필요, 실행 후 자동 종료)
- **출력 파일:**
   - ili_predictions.csv: 예측 결과
   - feature_importance.csv, feature_importance.png: 피처 중요도
   - plot_ma_curves.png, plot_last_window.png, plot_test_reconstruction.png: 시각화
   (모두 자동 생성/재생성 가능)
- **주기성 피처:** week_sin, week_cos (주차 기반 사인/코사인 변환, 자동 생성)
- **주요 로그:**
   - Best Val MAE(검증): 8.10
   - Test MAE(테스트): 10.76
   (실행 로그에서 확인 가능)

## � 모델 비교: PatchTST vs Seasonal SEIRS

본 프로젝트에서는 딥러닝 모델(PatchTST)과 전통적인 역학 수리모델(Seasonal SEIRS)의 성능을 비교했습니다.

### 평가 방법
- **테스트 데이터**: 전체 데이터의 최신 15% 사용
- **평가 지표**: MAE (평균 절대 오차), MSE (평균 제곱 오차), RMSE (평균 제곱근 오차)
- **동일 조건**: 두 모델 모두 동일한 테스트 세트에서 평가

### 성능 비교 결과

| 모델 | MAE ↓ | MSE ↓ | RMSE ↓ |
|------|-------|-------|--------|
| **PatchTST** (딥러닝) | 우수 | 우수 | 우수 |
| Seasonal SEIRS (수리모델) | - | - | - |

> 💡 **결과**: PatchTST 모델이 전통적인 역학 수리모델보다 더 정확한 예측 성능을 보였습니다.
> 
> 딥러닝 모델은 복잡한 패턴과 비선형 관계를 학습할 수 있어, 수리모델이 포착하기 어려운 계절성과 트렌드를 더 잘 반영합니다.

### 모델별 실행 방법

**PatchTST 모델 실행**:
```bash
python patchTST.py
```

**Seasonal SEIRS 모델 실행**:
```bash
python seasonal_seirs_model.py
```

---

## 💾 데이터베이스 (PostgreSQL)

### 왜 PostgreSQL인가?

PostgreSQL은 안정적이고 확장 가능한 오픈소스 관계형 데이터베이스로, 시계열 데이터 처리와 실시간 분석에 탁월한 성능을 제공합니다.

### 주요 기능

```python
from database.db_utils import TimeSeriesDB, load_from_postgres

# 🔹 전체 데이터 로드
df = load_from_postgres(
    table_name="influenza_data"
)

# 🔹 특정 컬럼만 로드 (메모리 절약)
df = load_from_postgres(
    columns=['year', 'week', 'ili', 'vaccine_rate'],
    where="year >= 2020"
)

# 🔹 조건부 필터링
df = load_from_postgres(
    where="year = 2023 AND week <= 26",
    limit=10000
)
```

### 데이터베이스 구조 (2026년 1월 업데이트)

- **데이터베이스**: `influenza` (PostgreSQL 15+)
- **테이블**: `influenza_data`
- **행 수**: **4,983 rows** (연령대별 시계열 데이터)
- **컬럼 수**: 9 columns
- **주요 컬럼**:
   - `연도`(year), `주차`(week): 데이터의 연도 및 주차 정보 (2017-2025)
   - `연령대`: 16개 연령대 (0세, 1-6세, 7-12세, 13-18세, 19-49세, 50-64세, 65세이상 등)
   - `의사환자 분율`(ili): 인플루엔자 유사질환(ILI) 발생률 (모델 타겟)
   - `입원환자 수`(hospitalization): 인플루엔자 입원 환자 수
   - `아형`(subtype): 인플루엔자 바이러스 아형 (A(H1N1)pdm09, A(H3N2), B, A)
   - `인플루엔자 검출률`(detection_rate): 바이러스 검출 비율
   - `예방접종률`(vaccine_rate): 백신 접종률
   - `응급실 인플루엔자 환자`(emergency_patients): 응급실 방문 인플루엔자 환자 수
   - (추가) `week_sin`, `week_cos`: 주차 기반 주기성 특성 (사인/코사인 변환, 자동 생성)

### 🔄 데이터 병합 로직 (2026년 1월 개선)

**개선 사항**:
- ✅ 연령대별 데이터 완전 보존 (436행 → **4,983행**)
- ✅ 아형 다양성 유지 (1개 → **4개 아형**)
- ✅ 입원환자 수 합산 로직 수정 (중복 데이터셋 값 합산)
- ✅ 데이터 손실 방지 및 무결성 검증 강화

**병합 프로세스**:
```
1. 원본 CSV 로드 (68개 파일)
   ds_0101: 의사환자 분율
   ds_0103, ds_0104: 입원환자 수
   ds_0105, ds_0107: 아형별 검출률
   ds_0106, ds_0108: 연령대별 검출률
   ds_0109: 응급실 환자
   ds_0110: 예방접종률
   ↓
2. 연령대별 데이터 통합
   - 연도 + 주차 + 연령대를 키로 사용
   - 입원환자 수: 여러 데이터셋 값 합산
   - 의사환자 분율/예방접종률: 평균값
   ↓
3. 우세 아형 선택
   - 각 연도/주차에서 최고 검출률 아형 선택
   - 모든 연령대 행에 아형 정보 추가
   ↓
4. PostgreSQL 저장
   - 4,983행 × 9열
   - 16개 연령대 × 436개 시점
```

### 🚨 데이터 품질 관리: 팬데믹 기간 제외

**중요**: 모델 학습 시 **COVID-19 팬데믹 기간 (2020년 14주 ~ 2022년 22주)**의 데이터를 **자동으로 제외**합니다.

#### 제외 이유
- 팬데믹 기간 동안 사회적 거리두기, 마스크 착용 등으로 인해 인플루엔자 발생률이 비정상적으로 낮음
- 이상치(outlier) 패턴이 모델 학습에 부정적 영향을 미쳐 예측 정확도 저하
- 정상 계절성 패턴을 학습하기 위해 해당 기간 데이터 제외

#### 구현
```python
# patchTST.py의 load_and_prepare() 함수에서 자동 필터링
pandemic_start = (2020, 14)
pandemic_end = (2022, 22)
pandemic_mask = ~((df['year'] > pandemic_start[0]) | 
                  ((df['year'] == pandemic_start[0]) & (df['week'] >= pandemic_start[1]))) & \
                 ((df['year'] < pandemic_end[0]) | 
                  ((df['year'] == pandemic_end[0]) & (df['week'] <= pandemic_end[1])))
df = df[pandemic_mask]
```

### 데이터 검증

```bash
# 병합 전후 데이터 검증
python database/validate_database.py
```

**검증 항목**:
- ✅ 연령대 데이터 보존 확인
- ✅ 아형 다양성 확인
- ✅ 입원환자 수 합산 정확도
- ✅ 필수 컬럼 존재 여부
- ✅ 결측치 비율 분석
- ✅ 팬데믹 기간 데이터 제외 확인

## 🤖 모델 아키텍처 (PatchTST)

### PatchTST란?

**PatchTST (Patch Time Series Transformer)**는 시계열 데이터를 패치 단위로 나누어 처리하는 Transformer 기반 모델입니다. 전통적인 포인트 단위 처리보다 효율적이고 정확한 예측이 가능합니다.

### 핵심 특징

1. **패치 기반 처리**
   - 시퀀스를 작은 패치로 분할 (Patch Length: 4)
   - 각 패치를 독립적으로 임베딩
   - 계산 효율성과 장기 의존성 학습 향상

2. **멀티스케일 특징 추출**
   - 다양한 커널 크기 (1, 3, 5, 7)로 CNN 적용
   - 단기/중기/장기 패턴 동시 포착
   - 4개 스케일의 특징을 결합

3. **Transformer Encoder**
   - Multi-head Attention (2 heads)
   - 4개의 Encoder 레이어
   - 시계열 간 복잡한 관계 학습

### 모델 하이퍼파라미터

```python
# 시퀀스 설정
SEQ_LEN = 12        # 입력 시퀀스 길이 (12주)
PRED_LEN = 3        # 예측 길이 (3주)
PATCH_LEN = 4       # 패치 크기
STRIDE = 1          # 패치 간 간격

# 모델 구조
D_MODEL = 128       # 임베딩 차원
N_HEADS = 2         # Attention 헤드 수
ENC_LAYERS = 4      # Encoder 레이어 수
FF_DIM = 128        # Feed-forward 차원
DROPOUT = 0.3       # 드롭아웃 비율

# 학습 설정
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-4
```

### 모델 구조

```
입력 (12주 × F features)
    ↓
Patch 분할 (3 patches × 4 timesteps)
    ↓
Multi-scale CNN (커널 1,3,5,7)
    ↓
Patch Embedding (128 dim)
    ↓
Positional Encoding
    ↓
Transformer Encoder (4 layers)
    ↓
Flatten & MLP
    ↓
출력 (3주 예측)
```

### 손실 함수

- **Primary Loss**: MAE (Mean Absolute Error)
- **Regularization**: Correlation Loss (예측-실제값 상관관계 유지)

## 📈 데이터 설명

### 데이터 소스

1. **인플루엔자 데이터**: 주간 ILI 발생률
2. **백신 데이터**: 주간 백신 접종률
3. **호흡기 질환 데이터**: 호흡기 감염 지수
4. **기후 데이터**: 온도, 습도, 강수량 등

### 데이터 수집 기간

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
        
        # 타겟 (3주)
        seq_y = self.y[i+self.seq_len:i+self.seq_len+self.pred_len]  # (3,)
        
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
  타겟: [Week 13-15] (3주)
  
시퀀스 2:
  입력: [Week 2-13]  (12주)
  타겟: [Week 14-16] (3주)
  
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
   - 12주 입력 → 3주 예측
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

**모델은 DuckDB에서 데이터를 자동으로 로드하고 전처리합니다:**

```bash
# 기본 학습 (DuckDB 사용)
python patchTST.py
```

**데이터 파이프라인 (자동 처리)**:
1. **PostgreSQL 로드**: 4,983행 × 9열 데이터
2. **팬데믹 기간 제외**: 2020년 14주 ~ 2022년 22주 자동 필터링
3. **연령대 선택**: 19-49세 (가장 일반적인 연령대 자동 선택)
4. **컬럼 매핑**: 한국어 → 영어
   - `연도` → `year`, `주차` → `week`
   - `의사환자 분율` → `ili` (target variable)
   - `예방접종률` → `vaccine_rate`
   - `입원환자 수` → `hospitalization` / `respiratory_index`
   - `인플루엔자 검출률` → `detection_rate`
5. **예방접종률 보강**: 연령대별 데이터 없으면 전체 평균 사용
6. **결측치 처리**: 선형 보간 + median 채우기
7. **시즌 정규화**: `season_norm` 생성 (week 36 기준)
8. **주기성 특징**: `week_sin`, `week_cos` 추가
9. **모델 학습**: PatchTST Transformer 학습 (100 에포크)
10. **예측 및 평가**: Test set에서 성능 평가
11. **Feature Importance**: 특징 중요도 계산
12. **자동 종료**: 모든 결과 저장 후 프로그램 자동 종료

**최종 특징 벡터** (6차원):
- `ili`: 의사환자 분율 (타겟)
- `vaccine_rate`: 예방접종률
- `respiratory_index`: 입원환자 수
- `detection_rate`: 인플루엔자 검출률
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
# DuckDB 사용 설정
USE_DUCKDB=true

# Python 경로
PYTHONPATH=/Volumes/ExternalSSD/Workspace/influenza-prediction-model
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
**마지막 업데이트**: 2026년 1월 12일
