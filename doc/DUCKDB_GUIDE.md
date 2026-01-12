# DuckDB를 활용한 대용량 시계열 데이터 관리 가이드

## 📌 개요

이 프로젝트는 **DuckDB**를 사용하여 대용량 인플루엔자 데이터(1.1GB CSV, 300만+ 행)를 효율적으로 관리합니다.

### 왜 DuckDB인가?

- ✅ **빠른 속도**: CSV 대비 2~20배 빠른 데이터 로딩
- ✅ **저장 공간 절약**: 95% 이상의 압축률 (1.1GB → 48MB)
- ✅ **시계열 데이터에 최적화**: OLAP 쿼리에 특화
- ✅ **설치 간편**: 별도 서버 없이 Python 라이브러리로 사용
- ✅ **SQL 지원**: 익숙한 SQL로 데이터 필터링 및 분석
- ✅ **Pandas 통합**: DataFrame과 완벽하게 호환

## 🚀 빠른 시작

### 1. DuckDB 설치

```bash
pip install duckdb
```

또는 `requirements.txt`에서 일괄 설치:

```bash
pip install -r requirements.txt
```

### 2. CSV를 DuckDB로 변환

```bash
python db_utils.py
```

**실행 결과:**
- 원본 CSV: 1163.4 MB
- DuckDB: 48.0 MB
- 압축률: **95.9% 절약**
- 변환 속도: **~555,000 행/초**

### 3. patchTST.py에서 사용

`.env` 파일에 다음을 추가:

```env
USE_API=false
USE_DUCKDB=true
```

이제 `patchTST.py`를 실행하면 자동으로 DuckDB에서 데이터를 로드합니다!

```bash
python patchTST.py
```

## 📊 성능 비교

| 작업 | CSV | DuckDB | 속도 향상 |
|------|-----|--------|----------|
| 전체 데이터 로드 (300만 행) | - | 15.3초 | - |
| 특정 컬럼만 로드 (5개 컬럼) | - | 0.78초 | **19.5배** |
| 샘플 데이터 (1000행) | - | 0.05초 | **즉시** |
| 100k 행 로드 | 0.54초 | 0.23초 | **2.3배** |

## 🔧 사용 방법

### 기본 사용법

```python
from database.db_utils import load_from_duckdb

# 전체 데이터 로드
df = load_from_duckdb()
print(df.shape)  # (3029039, 101)
```

### 특정 컬럼만 로드

```python
# 필요한 컬럼만 선택하여 메모리 절약
df = load_from_duckdb(
    columns=['year', 'week', '의사환자 분율', '입원환자 수', '예방접종률']
)
print(df.shape)  # (3029039, 5)
```

### 조건부 로드

```python
# 2020년 이후 데이터만
df = load_from_duckdb(where="year >= 2020")

# 특정 주차 범위
df = load_from_duckdb(where="week BETWEEN 20 AND 30")
```

### 정렬 및 제한

```python
# 최근 1000개 데이터
df = load_from_duckdb(
    limit=1000,
    order_by="year DESC, week DESC"
)
```

### 복합 쿼리

```python
# 2020년 이후, 특정 컬럼만, 최근 순으로 1000개
df = load_from_duckdb(
    columns=['year', 'week', '의사환자 분율'],
    where="year >= 2020",
    order_by="year DESC, week DESC",
    limit=1000
)
```

## 🛠️ 고급 사용법

### TimeSeriesDB 클래스 직접 사용

```python
from database.db_utils import TimeSeriesDB

# 컨텍스트 매니저로 사용 (자동 연결 관리)
with TimeSeriesDB("influenza_data.duckdb") as db:
    # 테이블 정보 확인
    db.get_table_info("influenza_data")
    
    # 데이터 로드
    df = db.load_data(
        table_name="influenza_data",
        where="year >= 2020",
        limit=10000
    )
    
    # Parquet으로 내보내기
    db.export_to_parquet(
        table_name="influenza_data",
        parquet_path="influenza_data.parquet"
    )
```

### CSV 파일 새로 추가

```python
from database.db_utils import TimeSeriesDB

db = TimeSeriesDB("influenza_data.duckdb")
db.connect()

# 새 CSV 임포트
db.import_csv_to_db(
    csv_path="new_data.csv",
    table_name="new_table"
)

# 데이터베이스 최적화
db.optimize_database()

db.close()
```

## 📁 파일 구조

```
influenza-prediction-model/
├── merged_influenza_data.csv    # 원본 CSV (1.1GB)
├── influenza_data.duckdb        # DuckDB 데이터베이스 (48MB)
├── db_utils.py                  # DuckDB 유틸리티
├── test_duckdb.py               # 성능 테스트
├── patchTST.py                  # 메인 모델 (자동으로 DuckDB 사용)
└── .env                         # 환경 설정
```

## ⚙️ 환경 변수

`.env` 파일에서 데이터 로딩 방식을 제어할 수 있습니다:

```env
# API에서 데이터 가져오기 vs 로컬 파일 사용
USE_API=false

# DuckDB 사용 vs CSV 직접 로드
USE_DUCKDB=true
```

### 데이터 로드 우선순위

1. `USE_API=true` → API에서 데이터 가져옴
2. `USE_API=false` + `USE_DUCKDB=true` → DuckDB 사용 (권장)
3. `USE_API=false` + `USE_DUCKDB=false` → CSV 직접 로드 (느림)

## 💡 팁과 권장사항

### 1. 대용량 파일은 항상 DuckDB 사용
- 100MB 이상의 CSV는 DuckDB로 변환 권장
- 변환은 한 번만 하면 됨

### 2. 필요한 컬럼만 로드
```python
# ❌ 비효율적
df = load_from_duckdb()  # 모든 컬럼 로드

# ✅ 효율적
df = load_from_duckdb(columns=['year', 'week', 'ili'])  # 필요한 컬럼만
```

### 3. 조건부 로드로 메모리 절약
```python
# 필요한 기간만 로드
df = load_from_duckdb(where="year >= 2020 AND year <= 2023")
```

### 4. 샘플링으로 빠른 프로토타이핑
```python
# 개발 중에는 샘플 데이터로 테스트
df_sample = load_from_duckdb(limit=10000)
```

### 5. Parquet으로 추가 최적화
```python
with TimeSeriesDB() as db:
    # Parquet은 컬럼 기반 포맷으로 분석 쿼리에 더 빠름
    db.export_to_parquet("influenza_data.parquet")
```

## 🔍 테스트 및 검증

성능 테스트 실행:

```bash
python test_duckdb.py
```

출력 예시:
```
[테스트 1] 전체 데이터 로드
✅ 로드 완료: (3029039, 101)
   소요 시간: 15.29초

[테스트 2] 특정 컬럼만 로드
✅ 로드 완료: (3029039, 5)
   소요 시간: 0.78초
   속도 향상: 19.5배 빠름
```

## 🐛 트러블슈팅

### DuckDB 파일이 없다는 오류
```
⚠️ DuckDB 파일이 없습니다: influenza_data.duckdb
```

**해결방법:**
```bash
python db_utils.py
```

### 컬럼명 오류 (공백 포함)
```
BinderException: Referenced column "의사환자" not found
```

**해결방법:** 이미 자동으로 처리됩니다. `db_utils.py`가 컬럼명을 따옴표로 감쌉니다.

### 메모리 부족
전체 데이터가 너무 크면 부분적으로 로드:
```python
# 연도별로 나누어 처리
for year in range(2017, 2026):
    df_year = load_from_duckdb(where=f"year = {year}")
    # 처리...
```

## 📚 참고 자료

- [DuckDB 공식 문서](https://duckdb.org/docs/)
- [DuckDB Python API](https://duckdb.org/docs/api/python/overview)
- [시계열 데이터베이스 비교](https://duckdb.org/why_duckdb)

## 🎯 요약

1. **설치**: `pip install duckdb`
2. **변환**: `python db_utils.py`
3. **사용**: `.env`에서 `USE_DUCKDB=true` 설정
4. **실행**: `python patchTST.py`

**결과**: 
- 🚀 10~20배 빠른 데이터 로딩
- 💾 95% 저장 공간 절약
- 🎨 기존 코드 수정 불필요
