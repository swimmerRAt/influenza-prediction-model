# 코드 비교 분석: 제공 코드 vs 프로젝트 코드

## 📋 개요

이 문서는 제공된 `ili_patchtst_train_and_plot_v4_cnnmix.py` 코드와 현재 프로젝트의 `patchTST.py` 코드 간의 차이점을 상세히 분석하고, 성능 차이의 원인을 파악합니다.

---

## 🔍 주요 차이점 분석

### 1. 데이터 로딩 및 소스

#### 제공 코드
```python
# CSV 파일 직접 로드
CANDIDATE_CSVS = [
    BASE_DIR / "data/processed/3_merged_influenza_vaccine_respiratory_weather_filled.csv",
    BASE_DIR / "data/processed/3_merged_influenza_vaccine_respiratory_weather.csv",
]
CSV_PATH = pick_csv_path()
df = read_csv_kor(CSV_PATH)
```

#### 프로젝트 코드
```python
# PostgreSQL에서 로드
df = load_data_from_postgres()  # database.db_utils 사용
# - 연령대별 데이터 완전 보존 (4,983행)
# - 팬데믹 기간 특별 처리 포함
```

**차이점:**
- 제공 코드: CSV 파일 직접 읽기 (단순)
- 프로젝트: PostgreSQL 데이터베이스 활용 (확장성, 데이터 무결성)

**성능 영향:**
- ✅ 프로젝트: 더 많은 데이터 (4,983행 vs CSV의 약 436행)
- ✅ 프로젝트: 연령대별 세분화된 데이터로 더 풍부한 패턴 학습 가능

---

### 2. 데이터 전처리: 주 단위 → 일 단위 변환

#### 제공 코드
```python
# 주 단위 데이터를 일 단위로 확장 (선형보간)
df = weekly_to_daily_interp(df, season_col="season_norm", week_col="week", target_col="ili")
```

#### 프로젝트 코드
```python
# 주 단위 데이터 유지 (주석 처리됨)
# df = weekly_to_daily_interp(df, season_col="season_norm", week_col="week", target_col="ili")
```

**차이점:**
- 제공 코드: 주 단위 → 일 단위 변환 (데이터 포인트 증가)
- 프로젝트: 주 단위 데이터 유지 (원본 데이터 구조 보존)

**성능 영향:**
- ⚠️ 제공 코드: 일 단위로 변환하면 데이터 포인트가 약 7배 증가하지만, **인위적으로 생성된 데이터**로 인한 오버피팅 위험
- ✅ 프로젝트: 원본 주 단위 데이터 유지로 **실제 패턴**에 집중

---

### 3. 팬데믹 기간 데이터 처리

#### 제공 코드
```python
# 팬데믹 기간 처리 없음
# 모든 데이터를 그대로 사용
```

#### 프로젝트 코드
```python
# 팬데믹 기간 (2020-W14 ~ 2022-W22) 특별 처리
pandemic_mask = (
    ((df['year'] == 2020) & (df['week'] >= 14)) |
    ((df['year'] == 2021)) |
    ((df['year'] == 2022) & (df['week'] <= 22))
)

# 계절성 패턴 기반 보간
# 1. 팬데믹 이전 기간(2017-2019)의 주차별 평균 패턴 계산
# 2. 팬데믹 기간 데이터를 주차별 평균 패턴으로 대체
weekly_pattern = df_pre_pandemic.groupby('week')['ili'].mean()
```

**차이점:**
- 제공 코드: 팬데믹 기간 데이터를 그대로 사용 (이상치 포함)
- 프로젝트: 팬데믹 기간 데이터를 계절성 패턴으로 보간 (정상 패턴 학습)

**성능 영향:**
- ❌ 제공 코드: 팬데믹 기간의 비정상적으로 낮은 ILI 값이 모델 학습을 방해
- ✅ 프로젝트: 정상 계절성 패턴 학습으로 **더 안정적인 예측**

---

### 4. Feature 선택 로직

#### 제공 코드
```python
# USE_EXOG 모드에 따라 선택적 feature 선택
mode = use_exog.lower()
if mode == "auto":
    chosen = ["ili"]
    if has_vax:  chosen.append("vaccine_rate")
    if has_resp: chosen.append("respiratory_index")
    chosen += climate_feats
elif mode == "all":
    chosen = ["ili"]
    if has_vax:  chosen.append("vaccine_rate")
    if has_resp: chosen.append("respiratory_index")
    chosen += climate_feats
```

#### 프로젝트 코드
```python
# column_mapping을 통한 강제 feature 포함
column_mapping = {
    '연도': 'year',
    '주차': 'week',
    '의사환자 분율': 'ili',
    '예방접종률': 'vaccine_rate',
    '입원환자 수': 'hospitalization',
    '인플루엔자 검출률': 'detection_rate',
    '응급실 인플루엔자 환자': 'emergency_patients',
    '아형': 'subtype'
}
chosen = []
for v in column_mapping.values():
    if v == "week":
        chosen += ["week_sin", "week_cos"]
    else:
        chosen.append(v)
```

**차이점:**
- 제공 코드: 조건부 feature 선택 (USE_EXOG 모드에 따라)
- 프로젝트: **모든 가능한 feature 강제 포함** (year, detection_rate, emergency_patients, subtype 등)

**성능 영향:**
- ⚠️ 제공 코드: feature가 부족할 수 있음 (예: detection_rate, emergency_patients 누락 가능)
- ✅ 프로젝트: **더 풍부한 feature set**으로 모델 성능 향상 가능
- ⚠️ 프로젝트: 일부 feature가 유용하지 않으면 노이즈로 작용할 수 있음

---

### 5. 손실 함수 (Loss Function)

#### 제공 코드
```python
crit = nn.HuberLoss(delta=1.0)
```

#### 프로젝트 코드
```python
def peak_weighted_loss(pred, target, peak_quantile=0.9, alpha=3.0):
    """
    Peak-aware weighted MAE loss.
    상위 90% quantile 이상의 값에 3배 가중치 부여
    """
    with torch.no_grad():
        thresh = torch.quantile(target, peak_quantile)
        weights = torch.ones_like(target)
        weights[target >= thresh] = alpha
    return torch.mean(weights * torch.abs(pred - target))

crit = peak_weighted_loss
```

**차이점:**
- 제공 코드: HuberLoss (이상치에 robust하지만 모든 시점 동일 가중치)
- 프로젝트: **Peak-weighted Loss** (인플루엔자 유행기(peak)에 집중)

**성능 영향:**
- ⚠️ 제공 코드: 유행기와 비유행기를 동일하게 학습 (유행기 예측 정확도 낮을 수 있음)
- ✅ 프로젝트: **유행기 예측에 집중**하여 실제로 중요한 시점의 예측 정확도 향상
- 💡 인플루엔자 예측에서는 유행기 예측이 더 중요하므로 프로젝트 접근이 더 적합

---

### 6. Feature 중복 처리

#### 제공 코드
```python
# feat_names에 week_sin, week_cos 추가 (중복 가능)
feat_names = chosen[:]
if INCLUDE_SEASONAL_FEATS and {"week_sin", "week_cos"}.issubset(df.columns):
    feat_names += ["week_sin", "week_cos"]  # 중복 추가 가능
```

#### 프로젝트 코드
```python
# 중복 제거 및 순서 보존
chosen = [x for i, x in enumerate(chosen) if x not in chosen[:i]]

feat_names = chosen[:]
if INCLUDE_SEASONAL_FEATS and {"week_sin", "week_cos"}.issubset(df.columns):
    feat_names += ["week_sin", "week_cos"]  # 여전히 중복 가능
```

**차이점:**
- 제공 코드: 중복 체크 없음
- 프로젝트: chosen 리스트 내 중복만 제거 (feat_names에 추가 시 중복 가능)

**성능 영향:**
- ⚠️ 양쪽 모두: week_sin/week_cos가 중복될 수 있음 (버그 가능성)
- ⚠️ 중복 feature는 모델 학습에 혼란을 줄 수 있음

---

### 7. 데이터 분할 (Train/Val/Test Split)

#### 제공 코드
```python
def make_splits(n: int, train_ratio=0.7, val_ratio=0.15):
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return (0, n_train), (n_train, n_train+n_val), (n_train+n_val, n)
```

#### 프로젝트 코드
```python
def make_splits(n: int, train_ratio=None, val_ratio=None):
    if train_ratio is None:
        train_ratio = Config.TRAIN_RATIO  # 0.7
    if val_ratio is None:
        val_ratio = Config.VAL_RATIO  # 0.15
    # 동일한 로직
```

**차이점:**
- 거의 동일 (70% train, 15% val, 15% test)

**성능 영향:**
- ✅ 동일한 분할 비율로 공정한 비교 가능

---

### 8. 모델 아키텍처

#### 제공 코드 & 프로젝트 코드
```python
# 동일한 아키텍처
class MultiScaleCNNPatchEmbed:
    # 멀티스케일 CNN (kernel_size=1,3,5, dilation=2)
    
class TokenConvMixer:
    # 패치 토큰 간 로컬 연속성 강화
    
class PatchTSTModel:
    # MultiScaleCNN → TokenConvMixer → Transformer → AttnPool → Head
```

**차이점:**
- 모델 아키텍처는 **완전히 동일**

**성능 영향:**
- ✅ 아키텍처 차이는 성능 차이의 원인이 아님

---

### 9. 하이퍼파라미터

#### 제공 코드
```python
EPOCHS = 100
BATCH_SIZE = 64
SEQ_LEN = 12
PRED_LEN = 3
PATCH_LEN = 4
D_MODEL = 128
N_HEADS = 2
ENC_LAYERS = 4
FF_DIM = 128
DROPOUT = 0.3
LR = 5e-4
WEIGHT_DECAY = 5e-4
PATIENCE = 60
WARMUP_EPOCHS = 30
```

#### 프로젝트 코드
```python
EPOCHS = 200  # 더 긴 학습
BATCH_SIZE = 64  # 동일
SEQ_LEN = 12  # 동일
PRED_LEN = 3  # 동일
PATCH_LEN = 4  # 동일
D_MODEL = 128  # 동일
N_HEADS = 2  # 동일
ENC_LAYERS = 4  # 동일
FF_DIM = 128  # 동일
DROPOUT = 0.3  # 동일
LR = 5e-4  # 동일
WEIGHT_DECAY = 5e-4  # 동일
PATIENCE = 60  # 동일
WARMUP_EPOCHS = 30  # 동일
```

**차이점:**
- 프로젝트: EPOCHS = 200 (제공 코드의 2배)

**성능 영향:**
- ✅ 프로젝트: 더 긴 학습으로 **더 나은 수렴** 가능
- ⚠️ 단, Early Stopping이 있으므로 실제 학습 epoch는 다를 수 있음

---

### 10. Optuna 하이퍼파라미터 최적화

#### 제공 코드
```python
# Optuna 최적화 없음
```

#### 프로젝트 코드
```python
# Optuna 하이퍼파라미터 최적화 지원
if USE_OPTUNA:
    best_params = optimize_hyperparameters(X, y, labels, feat_names, n_trials=N_TRIALS)
    # 최적화된 파라미터로 최종 학습
    train_and_eval(X, y, labels, feat_names, optuna_params=best_params)
```

**차이점:**
- 제공 코드: 하이퍼파라미터 최적화 없음
- 프로젝트: **Optuna를 통한 자동 하이퍼파라미터 최적화**

**성능 영향:**
- ✅ 프로젝트: 최적화된 하이퍼파라미터로 **더 나은 성능** 가능
- 💡 단, USE_OPTUNA=False이면 기본값 사용

---

### 11. Feature Importance 계산

#### 제공 코드
```python
# Feature Importance 계산 없음
```

#### 프로젝트 코드
```python
# Perturbation-Based Feature Importance 계산
fi_df = compute_feature_importance(
    model, X_va_sc, y_va_sc, X_te_sc, y_te_sc,
    scaler_y, feat_names, random_state=SEED
)
plot_feature_importance(fi_df, out_csv=..., out_png=...)
```

**차이점:**
- 제공 코드: Feature Importance 분석 없음
- 프로젝트: **Feature Importance 계산 및 시각화**

**성능 영향:**
- ✅ 프로젝트: 어떤 feature가 중요한지 분석 가능 (모델 해석성 향상)
- ⚠️ 성능 자체에는 직접적 영향 없음 (분석 도구)

---

### 12. 데이터 진단 및 로깅

#### 제공 코드
```python
# 기본적인 로깅만
print(f"[Data] Selected feature columns (order) -> {feat_names}")
```

#### 프로젝트 코드
```python
# 상세한 데이터 진단
print(f"\n🔬 vaccine_rate 데이터 분석:")
print(f"   - 범위: [{vax_data.min():.4f}, {vax_data.max():.4f}]")
print(f"   - 평균: {vax_data.mean():.4f}, 표준편차: {vax_data.std():.4f}")
print(f"   - 변동계수(CV): {vax_data.std()/vax_data.mean():.4f}")
print(f"   - 0인 값: {(vax_data == 0).sum()}개 / {len(vax_data)}개")
print(f"   - 상관계수 (vaccine_rate vs ili): {np.corrcoef(vax_data, y)[0,1]:.4f}")
```

**차이점:**
- 제공 코드: 최소한의 로깅
- 프로젝트: **상세한 데이터 진단 및 분석**

**성능 영향:**
- ⚠️ 성능에는 직접적 영향 없음 (디버깅/분석 도구)

---

## 🎯 성능 차이의 주요 원인 분석

### 1. 팬데믹 기간 데이터 처리 (가장 중요)

**제공 코드:**
- 팬데믹 기간 데이터를 그대로 사용
- 2020-W14 ~ 2022-W22 기간의 비정상적으로 낮은 ILI 값이 학습에 포함됨

**프로젝트:**
- 팬데믹 기간 데이터를 계절성 패턴으로 보간
- 정상 계절성 패턴 학습

**예상 성능 영향:**
- ❌ 제공 코드: 팬데믹 기간 이상치로 인한 **예측 정확도 저하**
- ✅ 프로젝트: 정상 패턴 학습으로 **더 안정적인 예측**

---

### 2. 손실 함수 차이

**제공 코드:**
- HuberLoss: 모든 시점 동일 가중치

**프로젝트:**
- Peak-weighted Loss: 유행기(peak)에 3배 가중치

**예상 성능 영향:**
- ⚠️ 제공 코드: 유행기 예측 정확도 낮을 수 있음
- ✅ 프로젝트: **유행기 예측 정확도 향상** (실제로 중요한 시점)

---

### 3. Feature Set 차이

**제공 코드:**
- 조건부 feature 선택 (USE_EXOG 모드에 따라)
- 일부 중요한 feature 누락 가능 (detection_rate, emergency_patients 등)

**프로젝트:**
- 모든 가능한 feature 강제 포함
- 더 풍부한 feature set

**예상 성능 영향:**
- ⚠️ 제공 코드: feature 부족으로 인한 **정보 손실**
- ✅ 프로젝트: 더 많은 정보로 **더 나은 패턴 학습** 가능
- ⚠️ 단, 노이즈 feature가 포함되면 오히려 성능 저하 가능

---

### 4. 데이터 양 및 품질

**제공 코드:**
- CSV 파일 기반 (약 436행, 주 단위 → 일 단위 변환 시 약 3,000행)
- 인위적으로 생성된 일 단위 데이터

**프로젝트:**
- PostgreSQL 기반 (4,983행, 연령대별 세분화)
- 원본 주 단위 데이터 유지

**예상 성능 영향:**
- ⚠️ 제공 코드: 인위적 데이터로 인한 **오버피팅 위험**
- ✅ 프로젝트: 실제 데이터로 **더 일반화된 모델**

---

### 5. 학습 시간 (Epochs)

**제공 코드:**
- EPOCHS = 100

**프로젝트:**
- EPOCHS = 200

**예상 성능 영향:**
- ✅ 프로젝트: 더 긴 학습으로 **더 나은 수렴** 가능

---

## 📊 종합 평가

### 성능 향상 요인 (프로젝트 우위)

1. ✅ **팬데믹 기간 데이터 처리**: 정상 패턴 학습
2. ✅ **Peak-weighted Loss**: 유행기 예측 정확도 향상
3. ✅ **더 풍부한 Feature Set**: 더 많은 정보 활용
4. ✅ **더 긴 학습 시간**: 더 나은 수렴
5. ✅ **Optuna 최적화**: 최적 하이퍼파라미터 탐색

### 성능 저하 요인 (제공 코드 우위)

1. ❌ **팬데믹 기간 이상치**: 학습 방해
2. ❌ **Feature 부족**: 정보 손실
3. ❌ **일반적인 손실 함수**: 유행기 예측 부족

---

## 💡 결론 및 권장사항

### 주요 성능 차이 원인

1. **팬데믹 기간 데이터 처리** (가장 중요)
   - 제공 코드: 이상치 포함 → 예측 정확도 저하
   - 프로젝트: 정상 패턴 보간 → 안정적인 예측

2. **손실 함수**
   - 제공 코드: 일반적인 HuberLoss
   - 프로젝트: Peak-weighted Loss (유행기 집중)

3. **Feature Set**
   - 제공 코드: 조건부 선택 (부족할 수 있음)
   - 프로젝트: 모든 feature 포함 (풍부한 정보)

### 프로젝트 코드의 장점

- ✅ 팬데믹 기간 데이터 특별 처리로 정상 패턴 학습
- ✅ Peak-weighted Loss로 유행기 예측 정확도 향상
- ✅ 더 풍부한 feature set
- ✅ Optuna 하이퍼파라미터 최적화
- ✅ 더 긴 학습 시간

### 개선 가능한 부분

1. ⚠️ **Feature 중복 체크**: week_sin/week_cos 중복 방지
2. ⚠️ **Feature 선택 로직**: 불필요한 feature 제거 옵션 추가
3. ⚠️ **일 단위 변환**: 필요 시 선택적 사용 옵션

---

## 📝 참고사항

- 모든 비교는 코드 분석 기반이며, 실제 성능은 데이터와 실행 환경에 따라 다를 수 있습니다.
- 프로젝트 코드는 PostgreSQL 기반으로 더 많은 데이터를 활용할 수 있습니다.
- 제공 코드는 CSV 기반으로 더 단순하고 빠른 실행이 가능합니다.
