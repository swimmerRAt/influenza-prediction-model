# 🦠 인플루엔자 예측 모델 (PatchTST)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)](https://www.postgresql.org/)

시계열 데이터 기반의 인플루엔자(ILI) 발생률 예측을 위한 **PatchTST 딥러닝 모델**입니다.

## 📊 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목적** | 인플루엔자 유사질환(ILI) 발생률 예측 |
| **모델** | PatchTST (Patch Time Series Transformer) |
| **데이터** | 연령대별 시계열 데이터 (2017-2025, 16개 연령대) |
| **데이터베이스** | PostgreSQL 15+ |
| **예측 기간** | 4주 (주별) 또는 28일 (일별) |

---

## 🗂️ 프로젝트 구조

```
influenza-prediction-model/
├── main.py                      # 🎯 메인 진입점 (CLI)
├── patchTST.py                  # 📦 원본 모델 (모든 기능 포함)
├── requirements.txt             # 패키지 의존성
├── .env                         # 환경 변수 설정
│
├── src/                         # 📁 모듈화된 소스 코드
│   ├── config.py               # ⚙️ 설정 및 하이퍼파라미터
│   ├── data/                   # 데이터 처리 모듈
│   │   ├── data_loader.py     # PostgreSQL/CSV 데이터 로딩
│   │   ├── preprocessing.py   # 스케일링, 보간, 팬데믹 처리
│   │   └── dataset.py         # PatchTSTDataset 클래스
│   ├── models/                 # 모델 모듈
│   │   ├── patch_tst.py       # PatchTST 모델 아키텍처
│   │   └── loss.py            # PeakAwareLoss 손실 함수
│   ├── training/               # 학습 모듈
│   │   ├── trainer.py         # Trainer 클래스
│   │   ├── optimizer.py       # Optuna 하이퍼파라미터 최적화
│   │   └── utils.py           # 학습 유틸리티
│   ├── prediction/             # 예측 모듈
│   │   └── predictor.py       # 예측 및 Feature Importance
│   └── utils/                  # 유틸리티 모듈
│       └── visualization.py   # 시각화 함수
│
├── database/                    # 💾 데이터베이스 관리
│   ├── db_utils.py             # PostgreSQL 유틸리티
│   ├── update_database.py      # DB 업데이트
│   └── validate_database.py    # DB 검증
│
└── doc/                         # 📚 문서
    └── QUICKSTART.md           # 빠른 시작 가이드
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
conda create -n pydev310 python=3.10
conda activate pydev310

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일 생성:
```env
PG_HOST=localhost
PG_PORT=5432
PG_DB=influenza
PG_USER=postgres
PG_PASSWORD=your_password
```

### 3. 실행

```bash
# 기본 실행 (PostgreSQL에서 데이터 로드)
python main.py

# 특정 연령 그룹 예측
python main.py --age-group 65세이상

# 원본 CSV에서 직접 로드
python main.py --use-raw-data --age-group 19-49세

# 아형별 검출률 예측
python main.py --subtype A

# Optuna 최적화 스킵
python main.py --skip-optuna
```

---

## ⚙️ 주요 설정 (Config)

`patchTST.py`의 `Config` 클래스에서 설정 변경:

```python
class Config:
    # 모델 하이퍼파라미터
    SEQ_LEN = 16          # 입력 시퀀스 길이 (주)
    PRED_LEN = 4          # 예측 길이 (주)
    D_MODEL = 128         # 모델 차원
    N_HEADS = 2           # Attention heads
    
    # 일별 데이터 모드
    USE_DAILY_DATA = True
    DAILY_SEQ_LEN = 112   # 입력 길이 (일)
    DAILY_PRED_LEN = 28   # 예측 길이 (일)
    
    # Optuna 최적화
    USE_OPTUNA = False
    N_TRIALS = 50
```

---

## 📈 모델 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    PatchTST Model                       │
├─────────────────────────────────────────────────────────┤
│  Input: (B, P, L, F) - Patchified Time Series           │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Multi-Scale CNN Patch Embed                    │    │
│  │  - Conv1d (k=1, 3, 5, dilation=2)               │    │
│  │  - GAP → (B, P, D)                              │    │ 
│  └─────────────────────────────────────────────────┘    │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  TokenConvMixer × 2                             │    │
│  │  - DepthwiseConv1d + PointwiseConv1d            │    │
│  └─────────────────────────────────────────────────┘    │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Transformer Encoder                            │    │
│  │  - Positional Encoding                          │    │
│  │  - Multi-Head Self-Attention × N_LAYERS         │    │
│  └─────────────────────────────────────────────────┘    │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Attention Pooling                              │    │
│  │  - Learnable Query → (B, D)                     │    │
│  └─────────────────────────────────────────────────┘    │
│                         ↓                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Dual-Head Prediction                           │    │
│  │  - Trend Head + Peak Head                       │    │
│  │  - Output: (B, PRED_LEN)                        │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 핵심 기능

### 1. Peak-Aware Loss
피크 구간에 높은 가중치를 부여하여 인플루엔자 유행 시기 예측 정확도 향상:
```python
PeakAwareLoss(
    peak_quantile=0.85,    # 상위 15%를 피크로 정의
    alpha=12.0,            # 피크 가중치
    beta=0.6,              # 진폭 보존 가중치
    horizon_mode="exponential"  # 예측 구간별 가중치
)
```

### 2. 일별 데이터 변환
주간 데이터를 일간으로 보간하여 더 세밀한 예측:
```python
Config.USE_DAILY_DATA = True
Config.DAILY_INTERP_METHOD = "gaussian"  # 또는 "linear"
```

### 3. 팬데믹 기간 처리
COVID-19 기간(2020-2022) 데이터는 계절성 패턴 기반으로 보간:
```python
# 2020-W14 ~ 2022-W22 자동 처리
```

### 4. 외생 변수 지원
- 날씨 데이터 (기온, 습도)
- 예방접종률
- 검출률
- 입원환자 수

---

## 📊 출력 파일

| 파일 | 설명 |
|------|------|
| `ili_predictions.csv` | 예측 결과 (4주 ahead) |
| `feature_importance.csv` | Feature Importance |
| `plot_last_window.png` | 마지막 윈도우 예측 시각화 |
| `results.png` | 테스트 구간 재구성 |
| `best_hyperparameters.json` | 최적 하이퍼파라미터 |

---

## 🔍 CLI 옵션

```bash
python main.py --help

옵션:
  --use-raw-data        data/before에서 원본 CSV 직접 로드
  --data-dir PATH       원본 CSV 디렉토리 경로
  --age-group GROUP     연령 그룹 (예: 19-49세, 65세이상)
  --subtype TYPE        인플루엔자 아형 (A 또는 B)
  --skip-optuna         Optuna 최적화 스킵
  --optuna-trials N     Optuna 시도 횟수 (기본: 50)
  --seed N              랜덤 시드 (기본: 42)
```

---

## 📦 의존성

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
psycopg2-binary>=2.9.0
python-dotenv>=1.0.0
optuna>=3.0.0  # 선택사항
```

---

## 📝 라이선스

MIT License
