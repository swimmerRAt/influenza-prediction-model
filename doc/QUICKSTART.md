# 🚀 빠른 시작 가이드: DuckDB를 활용한 대용량 데이터 처리

## ⚡ 문제 해결

**문제**: `merged_influenza_data.csv` 파일(1.1GB, 300만+ 행)이 너무 커서 열리지 않음

**해결책**: DuckDB 데이터베이스 사용

## 📦 설치

1. **필수 패키지 설치**
```bash
pip install -r requirements.txt
```

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

### patchTST.py로 모델 학습

`.env` 파일 수정:
```env
USE_API=false
USE_DUCKDB=true
```

그 다음 모델 실행:
```bash
python patchTST.py
```

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
