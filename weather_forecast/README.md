# Weather Forecast - 기상 데이터 수집 및 예측

기상청 ASOS(종관기상관측) API를 활용한 기상 데이터 수집 및 TCN(Temporal Convolutional Network) 기반 기상 예측 시스템

## 📁 프로젝트 구조

```
weather_forecast/
├── README.md                    # 프로젝트 설명서
├── requirements.txt             # Python 패키지 의존성
├── weatherAPI_download.py       # 기상청 API 데이터 수집
├── TCN.py                       # TCN 모델 및 예측 파이프라인
└── data/                        # 데이터 저장 폴더
    ├── weather_asos_서울_YYYY.csv   # 연도별 일별 기상 데이터
    ├── weather_for_influenza.csv    # 주간 기상 데이터 (전처리됨)
    └── weather_forecast.csv         # 미래 예측 결과
```

## 🚀 설치 방법

### 1. 가상환경 생성 및 활성화

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 필요 패키지
- `requests`: API 호출
- `pandas`: 데이터 처리
- `torch`: TCN 모델 학습

## 📊 데이터 수집 (weatherAPI_download.py)

기상청 공공데이터포털 ASOS 일자료 API를 사용하여 과거 기상 데이터를 수집합니다.

### 사용 방법

```python
from weatherAPI_download import ASOSCollector

# 수집기 초기화
collector = ASOSCollector()

# 2017년부터 현재까지 서울 기상 데이터 수집 (연도별 파일 저장)
filepaths = collector.collect_historical_data(
    station="서울",
    start_year=2017,
    end_year=2026
)
```

### 지원 관측소

| 지역 | 지점번호 | 지역 | 지점번호 |
|------|----------|------|----------|
| 서울 | 108 | 부산 | 159 |
| 대구 | 143 | 인천 | 112 |
| 광주 | 156 | 대전 | 133 |
| 울산 | 152 | 제주 | 184 |

### 수집 데이터

일별 관측 데이터로 약 40개 이상의 기상 변수 포함:
- 기온 (평균, 최저, 최고)
- 습도 (평균, 최저)
- 기압, 풍속, 강수량
- 지면/지중 온도 등

## 🔮 기상 예측 (TCN.py)

### 1. 데이터 전처리

일별 기상 데이터를 주간 데이터로 변환하고 인플루엔자 분석에 필요한 변수만 추출합니다.

```bash
python TCN.py --mode preprocess
```

### 2. 미래 기상 예측

TCN 모델을 학습하고 미래 N주의 기상을 예측합니다.

```bash
# 기본 실행 (4주 예측, 100 에폭)
python TCN.py

# 8주 예측
python TCN.py --weeks 8

# 200 에폭으로 학습
python TCN.py --epochs 200
```

### 명령줄 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--mode` | forecast | 실행 모드: `forecast`(예측), `preprocess`(전처리) |
| `--weeks` | 4 | 예측할 주 수 (4주 ≈ 1개월) |
| `--epochs` | 100 | 모델 학습 에폭 수 |

### 모델 구조

```
WeatherForecastModel
├── TCN Encoder (3 layers: 64 → 128 → 64 channels)
│   └── TemporalBlock × 3 (dilated causal convolutions)
└── FC Layer (64 → 13 features)
```

- **입력**: 과거 8주 기상 데이터 (13개 특성)
- **출력**: 다음 주 기상 예측 (13개 특성)

## 📋 데이터 컬럼 설명

### 영어 컬럼명 (weather_for_influenza.csv, weather_forecast.csv)

| 컬럼명 | 설명 | 단위 |
|--------|------|------|
| `year` | 연도 | - |
| `week` | ISO 주차 | - |
| `avg_temp` | 평균기온 | ℃ |
| `min_temp` | 최저기온 | ℃ |
| `max_temp` | 최고기온 | ℃ |
| `avg_ground_temp` | 평균지면온도 | ℃ |
| `avg_soil_temp_5cm` | 평균 5cm 지중온도 | ℃ |
| `avg_soil_temp_10cm` | 평균 10cm 지중온도 | ℃ |
| `avg_soil_temp_20cm` | 평균 20cm 지중온도 | ℃ |
| `avg_soil_temp_30cm` | 평균 30cm 지중온도 | ℃ |
| `avg_humidity` | 평균상대습도 | % |
| `min_humidity` | 최저상대습도 | % |
| `avg_dew_point` | 평균이슬점온도 | ℃ |
| `avg_vapor_pressure` | 평균증기압 | hPa |
| `temp_range` | 일교차 (최고-최저) | ℃ |

## 🔗 인플루엔자 데이터와 병합

```python
import pandas as pd
from TCN import merge_with_influenza_data

# 인플루엔자 데이터 로드
influenza_df = pd.read_csv('influenza_data.csv')

# 기상 데이터와 자동 병합 (year, week 기준)
merged_df = merge_with_influenza_data(influenza_df)

# 또는 직접 병합
weather_df = pd.read_csv('data/weather_for_influenza.csv')
merged_df = influenza_df.merge(weather_df, on=['year', 'week'], how='left')
```

## 📈 예측 결과 예시

```
============================================================
예측 결과 (2026년 6~9주차)
============================================================
 year  week  avg_temp  min_temp  max_temp  avg_humidity
 2026     6     -0.71     -5.24      3.37         58.50
 2026     7     -0.52     -5.05      3.57         58.01
 2026     8     -0.49     -5.04      3.61         58.13
 2026     9     -0.51     -5.06      3.57         57.44
```

## ⚠️ 주의사항

1. **API 키**: `weatherAPI_download.py`의 `KMA_API_KEY` 변수에 공공데이터포털에서 발급받은 API 키를 설정해야 합니다.

2. **API 호출 제한**: 기상청 API는 호출 제한이 있으므로 대량 수집 시 적절한 딜레이를 유지합니다.

3. **데이터 범위**: ASOS 일자료는 전일(D-1)까지의 데이터만 제공됩니다.

4. **ISO 주차**: 연도 시작/끝 부분에서 ISO 주차 기준으로 인해 이전/다음 연도의 주차가 포함될 수 있습니다.

## 📚 참고 자료

- [공공데이터포털 - 기상청 종관기상관측](https://www.data.go.kr/data/15057210/openapi.do)
- [TCN Paper - An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)

## 📝 라이선스

이 프로젝트는 연구/교육 목적으로 사용할 수 있습니다.
