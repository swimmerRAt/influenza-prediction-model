# 🚀 빠른 시작 가이드: DuckDB를 활용한 대용량 데이터 처리

## ⚡ 문제 해결

**문제**: `merged_influenza_data.csv` 파일(1.1GB, 300만+ 행)이 너무 커서 열리지 않음

**해결책**: DuckDB 데이터베이스 사용

## 📦 설치

### 1. 가상환경 생성 및 활성화

**macOS/Linux:**
```bash
# 가상환경 생성
python3 -m venv .venv

# 가상환경 활성화
source .venv/bin/activate
```

**Windows:**
```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
.venv\Scripts\activate
```

### 2. 필수 패키지 설치
```bash
pip install -r requirements.txt
```

> **💡 Tip**: 작업을 마친 후 `deactivate` 명령어로 가상환경을 비활성화할 수 있습니다.

## 🔄 데이터 업데이트 (신규 추가!)

### 옵션 1: API + 과거 데이터 + 기존 데이터 병합 (권장)

```bash
python update_database.py
```

이 명령어는 다음 작업을 자동으로 수행합니다:
1. **API에서 최신 데이터 가져오기** 📡
2. **data/before 폴더의 과거 데이터 로딩** 📂
3. **기존 merged_influenza_data.csv 로딩** 📄
4. **모든 데이터 병합 및 중복 제거** 🔀
5. **DuckDB에 저장** 💾
6. **CSV로 백업** 📋

### 옵션 2: 기존 CSV만 DuckDB로 변환

```bash
python db_utils.py
```

**변환 결과**:
- ✅ 원본 CSV: 1163.4 MB → DuckDB: 48.0 MB (**95.9% 절약**)
- ✅ 변환 속도: ~555,000 행/초
- ✅ 로딩 속도: CSV 대비 2~20배 향상

## 🎯 사용 방법


### PatchTST 모델 학습 (DuckDB 통합, 최신 데이터 자동 반영)


**1단계: 환경 설정 확인**
   - `.env` 파일에서 USE_DUCKDB=true로 설정 (기본값)

**2단계: 모델 실행**
   - 서버 실행 불필요, 아래 명령어만 실행하면 자동으로 학습 및 예측 완료 후 종료됨
   ```bash
   python patchTST.py
   ```
   (실행 후 Best Val MAE, Test MAE 등 주요 성능 로그 자동 출력, 수동 종료 불필요)

**최신 데이터:** 2025년 47주차(11월 말~12월 초)까지 자동 반영
**출력 파일:**
   - ili_predictions.csv: 예측 결과
   - feature_importance.csv, feature_importance.png: 피처 중요도
   - plot_ma_curves.png, plot_last_window.png, plot_test_reconstruction.png: 시각화
   (모두 자동 생성/재생성 가능)

**주요 로그 확인:**
   - Best Val MAE(검증): 8.10
   - Test MAE(테스트): 10.76
   (실행 로그에서 확인 가능)

**자동 처리 과정:**
1. ✅ DuckDB에서 4,983행 데이터 자동 로드
2. ✅ 19-49세 연령대 자동 선택 (가장 일반적)
3. ✅ 한국어 컬럼 → 영어 컬럼 자동 매핑
   - `연도` → `year`, `주차` → `week`
   - `의사환자 분율` → `ili` (타겟 변수)
   - `예방접종률` → `vaccine_rate`
   - `입원환자 수` → `hospitalization`/`respiratory_index`
   - `인플루엔자 검출률` → `detection_rate`
4. ✅ 예방접종률 데이터 없는 연령대는 전체 평균 사용
5. ✅ 결측치 자동 보간 (선형 보간 + median)
6. ✅ 주기성 특징 자동 추가 (`week_sin`, `week_cos`)
7. ✅ PatchTST 모델 학습 (100 에포크)
8. ✅ 예측 결과 및 그래프 자동 저장
9. ✅ Feature Importance 계산 및 저장
10. ✅ **학습 완료 후 자동 종료** (수동 종료 불필요)

**출력 파일:**
```
output/
├── ili_predictions.csv           # 예측 결과
├── plot_ma_curves.png            # MAE/Loss 곡선
├── plot_last_window.png          # 마지막 윈도우 예측
├── plot_test_reconstruction.png  # 테스트 재구성
├── feature_importance.csv        # Feature Importance 데이터
└── feature_importance.png        # Feature Importance 그래프
```

**완료 확인:**
- 마지막 로그에 "Feature Importance saved to ..." 출력 → 정상 완료
- 프로그램이 자동으로 종료되며 터미널 프롬프트 복귀

> **⚠️ 참고**: 이전 버전에서는 `plt.show()` 때문에 수동 종료(Ctrl+C)가 필요했으나, 
> 현재 버전은 `plt.close()`로 변경되어 **자동으로 종료**됩니다.

### Python 코드에서 직접 사용

```python
from database.db_utils import load_from_duckdb

# 전체 데이터 로드
df = load_from_duckdb()

# 특정 컬럼만 로드 (19.5배 빠름!)
df = load_from_duckdb(
    columns=['year', 'week', '의사환자 분율', '입원환자 수']
)

# 조건부 로드
df = load_from_duckdb(where="year >= 2020")

# 샘플 데이터
df = load_from_duckdb(limit=1000)
```

## 📊 성능 비교

| 작업 | CSV | DuckDB | 개선율 |
|------|-----|--------|--------|
| 전체 로드 (300만 행) | - | 15.3초 | - |
| 컬럼 선택 (5개) | - | 0.78초 | **19.5배** ⚡ |
| 샘플 (1000행) | - | 0.05초 | **즉시** ⚡⚡⚡ |
| 저장 공간 | 1.1GB | 48MB | **95.9% 절약** 💾 |

## ✅ 테스트

```bash
# 성능 테스트
python test_duckdb.py

# 통합 테스트
python test_integration.py
```

## 📚 상세 가이드

자세한 사용법은 [DUCKDB_GUIDE.md](DUCKDB_GUIDE.md)를 참조하세요.

## 🔧 트러블슈팅

### DuckDB 파일이 없는 경우
```bash
python db_utils.py
```

### 메모리 부족 시
```python
# 필요한 컬럼만 로드
df = load_from_duckdb(columns=['year', 'week', 'ili'])

# 또는 기간 제한
df = load_from_duckdb(where="year >= 2020")
```

## 💡 팁

1. **개발 중**: 샘플 데이터로 빠른 프로토타이핑
   ```python
   df = load_from_duckdb(limit=10000)
   ```

2. **프로덕션**: 필요한 컬럼만 로드하여 메모리 절약
   ```python
   df = load_from_duckdb(columns=['year', 'week', 'ili'])
   ```

3. **분석**: SQL 쿼리로 데이터 필터링
   ```python
   df = load_from_duckdb(where="year = 2023 AND week BETWEEN 20 AND 30")
   ```

---

**다음 단계**: [전체 문서 보기](DUCKDB_GUIDE.md)
