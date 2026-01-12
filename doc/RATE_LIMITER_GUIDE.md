# 🛡️ Adaptive Rate Limiter 가이드

## 개요

서버 과부하를 자동으로 감지하고 API 요청 속도를 조절하는 **Adaptive Rate Limiter**가 추가되었습니다.

## 주요 기능

### 1. 🎯 자동 속도 조절 (Adaptive Throttling)

서버 응답 시간을 실시간으로 모니터링하여 요청 간격을 자동으로 조정합니다.

- **빠른 응답** (< 2.5초): 딜레이 감소 (최소 0.5초까지)
- **느린 응답** (> 5초): 딜레이 증가 (최대 30초까지)

### 2. 🔄 재시도 로직 (Exponential Backoff)

네트워크 오류나 타임아웃 발생 시 자동으로 재시도합니다.

- 최대 5회까지 재시도 (기본값)
- 재시도 간격은 지수적으로 증가 (1초 → 2초 → 4초 → 8초 → 16초)

### 3. 📊 서버 상태 모니터링

실시간으로 다음 지표를 추적합니다:

- **총 요청 수**: 전체 API 호출 횟수
- **에러 발생 수**: 실패한 요청 횟수
- **에러율**: 전체 요청 중 실패 비율
- **평균 응답 시간**: 서버 응답 속도
- **현재 딜레이**: 다음 요청까지 대기 시간
- **연속 에러**: 연속으로 실패한 횟수

### 4. ⚠️ 과부하 보호

연속 에러가 3회 이상 발생하면 사용자에게 확인을 요청합니다.

```
🚨 연속 3회 에러 발생!
계속 진행하시겠습니까? (y/n):
```

## 환경 변수 설정

`.env` 파일에서 다음 설정을 조정할 수 있습니다:

```bash
# 초기 대기 시간 (초) - 서버가 안정적일 때 기본 딜레이
RATE_LIMIT_INITIAL_DELAY=1.0

# 최대 대기 시간 (초) - 서버 과부하 시 최대 딜레이
RATE_LIMIT_MAX_DELAY=30.0

# 최소 대기 시간 (초) - 서버가 매우 빠를 때 최소 딜레이
RATE_LIMIT_MIN_DELAY=0.5

# 최대 재시도 횟수 - 실패 시 재시도 횟수
RATE_LIMIT_MAX_RETRIES=5
```

## 사용 예시

### 기본 사용

코드를 실행하면 자동으로 Rate Limiter가 활성화됩니다:

```python
# patchTST.py 실행 시 자동 적용
python patchTST.py
```

### 출력 예시

```
🛡️ Adaptive Rate Limiter 초기화
   초기 딜레이: 1.0초
   최대 딜레이: 30.0초
   최소 딜레이: 0.5초
   최대 재시도: 5회

====================================================
📥 [1/13] 데이터셋 로딩: ds_0101
====================================================
   ⏳ 서버 보호를 위해 1.00초 대기 중...
   ✅ 요청 성공 (응답 시간: 2.34초)
   ✅ ds_0101 로드 완료: (1500, 25)

====================================================
📥 [2/13] 데이터셋 로딩: ds_0102
====================================================
   ⏳ 서버 보호를 위해 1.00초 대기 중...
   🐢 느린 응답 감지 (6.78초) → 딜레이 증가: 1.00초 → 1.20초
   ✅ ds_0102 로드 완료: (1600, 25)

📊 [Rate Limiter 통계]
   총 요청: 13
   에러 발생: 0
   에러율: 0.0%
   평균 응답 시간: 3.45초
   현재 딜레이: 1.20초
   연속 에러: 0회
```

## 동작 원리

### 1. 응답 시간 기반 조정

```python
if 응답_시간 < 2.5초:
    딜레이 = 딜레이 × 0.9  # 10% 감소
elif 응답_시간 > 5초:
    딜레이 = 딜레이 × 1.2  # 20% 증가
```

### 2. 에러 발생 시 백오프

```python
if 에러_발생:
    딜레이 = 딜레이 × 2.0  # 2배 증가 (exponential backoff)
```

### 3. 재시도 간격 계산

```python
재시도_대기시간 = 현재_딜레이 × (2 ^ 재시도_횟수)
# 예: 1초 → 2초 → 4초 → 8초 → 16초
```

## 권장 설정

### 안정적인 서버 (빠른 응답)

```bash
RATE_LIMIT_INITIAL_DELAY=0.5
RATE_LIMIT_MAX_DELAY=10.0
RATE_LIMIT_MIN_DELAY=0.2
RATE_LIMIT_MAX_RETRIES=3
```

### 불안정한 서버 (느린 응답/에러 빈번)

```bash
RATE_LIMIT_INITIAL_DELAY=2.0
RATE_LIMIT_MAX_DELAY=60.0
RATE_LIMIT_MIN_DELAY=1.0
RATE_LIMIT_MAX_RETRIES=10
```

### 대용량 데이터 로드 (매우 느린 서버)

```bash
RATE_LIMIT_INITIAL_DELAY=5.0
RATE_LIMIT_MAX_DELAY=120.0
RATE_LIMIT_MIN_DELAY=2.0
RATE_LIMIT_MAX_RETRIES=15
```

## 문제 해결

### Q: 서버가 자주 타임아웃됩니다

**A:** 초기 딜레이를 늘리고, 타임아웃 시간을 늘리세요:

```bash
RATE_LIMIT_INITIAL_DELAY=3.0
RATE_LIMIT_MAX_DELAY=60.0
```

### Q: 데이터 로드가 너무 느립니다

**A:** 서버가 안정적이라면 딜레이를 줄이세요:

```bash
RATE_LIMIT_INITIAL_DELAY=0.3
RATE_LIMIT_MIN_DELAY=0.1
```

### Q: 연속 에러 메시지가 계속 나옵니다

**A:** 서버 상태를 확인하고, 딜레이를 크게 늘리세요:

```bash
RATE_LIMIT_INITIAL_DELAY=10.0
RATE_LIMIT_MAX_DELAY=180.0
```

## 통계 확인

프로그램 실행 중 언제든지 현재 통계를 확인할 수 있습니다:

```python
from patchTST import get_rate_limiter

limiter = get_rate_limiter()
limiter.print_stats()
```

## 이점

✅ **서버 보호**: 과도한 요청으로 인한 서버 다운 방지  
✅ **자동 복구**: 일시적 오류 시 자동 재시도  
✅ **최적 성능**: 서버 상태에 따라 최적 속도 자동 조절  
✅ **투명성**: 실시간 통계로 진행 상황 확인  
✅ **사용자 제어**: 연속 에러 시 사용자 개입 가능  

## 코드 예시

### 직접 사용하기

```python
from patchTST import get_rate_limiter

limiter = get_rate_limiter()

# 함수를 재시도 로직과 함께 실행
def my_api_call():
    # ... API 호출 코드
    return response

result = limiter.execute_with_retry(my_api_call)

# 통계 확인
limiter.print_stats()
```

---

**참고**: 이 기능은 `patchTST.py`의 `fetch_data_directly_from_gfid()` 함수에서 자동으로 적용됩니다.
