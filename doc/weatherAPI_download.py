"""
날씨 데이터 수집 모듈
- 한국 기상청 API (공공데이터포털)
- 과거 관측 데이터: ASOS(종관기상관측) 일자료 API
- 실시간 예보: 단기예보 API

사용법:
    한국 기상청 API:
    - https://www.data.go.kr 에서 API 키 발급
    - "기상청_지상(종관, ASOS) 일자료 조회서비스" 활용 신청 (과거 데이터)
    - "기상청_단기예보 조회서비스" 활용 신청 (실시간 예보)
"""

import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import List
import time


# API 키 설정
KMA_API_KEY = "d3bab1f86d0198e1bb79767953655e4a9d2c142230b731a3efb6a880be30d427"

# 기본 저장 경로 (weather_forecast/data)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")


class ASOSCollector:
    """기상청 ASOS(종관기상관측) API - 과거 관측 데이터 수집 (일자료)"""
    
    # ASOS 일자료 조회 서비스 URL
    ASOS_DAILY_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
    
    # 주요 관측소 코드 (stnId)
    STATION_IDS = {
        "서울": 108,
        "부산": 159,
        "대구": 143,
        "인천": 112,
        "광주": 156,
        "대전": 133,
        "울산": 152,
        "세종": 239,
        "수원": 119,
        "전주": 146,
        "청주": 131,
        "춘천": 101,
        "제주": 184,
        "강릉": 105,
        "목포": 165,
        "여수": 168,
        "포항": 138,
    }
    
    def __init__(self, api_key: str = KMA_API_KEY):
        """
        Args:
            api_key: 공공데이터포털 API 키
        """
        self.api_key = api_key
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    def get_daily_data(self, station: str, start_date: str, end_date: str, 
                       num_rows: int = 999) -> pd.DataFrame:
        """
        일별 관측 데이터 조회
        
        Args:
            station: 관측소명 (예: "서울", "부산")
            start_date: 시작 날짜 (YYYYMMDD 형식)
            end_date: 종료 날짜 (YYYYMMDD 형식) - 전일(D-1)까지 제공
            num_rows: 한 번에 가져올 데이터 수 (최대 999)
            
        Returns:
            관측 데이터 DataFrame
        """
        if station not in self.STATION_IDS:
            raise ValueError(f"지원하지 않는 관측소입니다: {station}. 지원 관측소: {list(self.STATION_IDS.keys())}")
        
        stn_id = self.STATION_IDS[station]
        
        params = {
            "serviceKey": self.api_key,
            "pageNo": 1,
            "numOfRows": num_rows,
            "dataType": "JSON",
            "dataCd": "ASOS",
            "dateCd": "DAY",  # 일자료
            "startDt": start_date,
            "endDt": end_date,
            "stnIds": stn_id,
        }
        
        response = requests.get(self.ASOS_DAILY_URL, params=params, headers=self.headers)
        response.raise_for_status()
        
        data = response.json()
        
        if data["response"]["header"]["resultCode"] != "00":
            raise Exception(f"API 오류: {data['response']['header']['resultMsg']}")
        
        items = data["response"]["body"]["items"]["item"]
        df = pd.DataFrame(items)
        
        # 컬럼명 한글화 (일자료 기준)
        column_mapping = {
            "tm": "날짜",
            "stnId": "관측소ID",
            "stnNm": "관측소명",
            "avgTa": "평균기온(℃)",
            "minTa": "최저기온(℃)",
            "minTaHrmt": "최저기온시각",
            "maxTa": "최고기온(℃)",
            "maxTaHrmt": "최고기온시각",
            "sumRn": "일강수량(mm)",
            "maxInsWs": "최대순간풍속(m/s)",
            "maxInsWsWd": "최대순간풍향(deg)",
            "maxInsWsHrmt": "최대순간풍속시각",
            "maxWs": "최대풍속(m/s)",
            "maxWsWd": "최대풍향(deg)",
            "maxWsHrmt": "최대풍속시각",
            "avgWs": "평균풍속(m/s)",
            "hr1MaxRn": "1시간최다강수량(mm)",
            "hr1MaxRnHrmt": "1시간최다강수량시각",
            "sumSsHr": "합계일조시간(hr)",
            "sumGsr": "합계일사량(MJ/m2)",
            "avgTd": "평균이슬점온도(℃)",
            "minRhm": "최저상대습도(%)",
            "minRhmHrmt": "최저상대습도시각",
            "avgRhm": "평균상대습도(%)",
            "avgPv": "평균증기압(hPa)",
            "avgPa": "평균현지기압(hPa)",
            "maxPs": "최고해면기압(hPa)",
            "maxPsHrmt": "최고해면기압시각",
            "minPs": "최저해면기압(hPa)",
            "minPsHrmt": "최저해면기압시각",
            "avgPs": "평균해면기압(hPa)",
            "ssDur": "가조시간(hr)",
            "sumDpthFhsc": "합계적설(cm)",
            "ddMefs": "일최심적설(cm)",
            "ddMefsHrmt": "일최심적설시각",
            "sumFogDur": "합계안개지속시간(hr)",
            "avgTca": "평균전운량(10분위)",
            "avgLmcsCa": "평균중하층운량(10분위)",
            "avgTs": "평균지면온도(℃)",
            "minTg": "최저초상온도(℃)",
            "avgCm5Te": "평균5cm지중온도(℃)",
            "avgCm10Te": "평균10cm지중온도(℃)",
            "avgCm20Te": "평균20cm지중온도(℃)",
            "avgCm30Te": "평균30cm지중온도(℃)",
            "avgM05Te": "0.5m지중온도(℃)",
            "avgM10Te": "1.0m지중온도(℃)",
            "avgM15Te": "1.5m지중온도(℃)",
            "avgM30Te": "3.0m지중온도(℃)",
            "avgM50Te": "5.0m지중온도(℃)",
            "sumLrgEv": "합계대형증발량(mm)",
            "sumSmlEv": "합계소형증발량(mm)",
            "n99Rn": "9-9강수(mm)",
        }
        
        df = df.rename(columns=column_mapping)
        
        return df
    
    def collect_historical_data(self, station: str, start_date: str, end_date: str,
                                output_dir: str = None) -> str:
        """
        과거 데이터를 수집
        
        Args:
            station: 관측소명
            start_date: 시작 날짜 (YYYYMMDD 형식)
            end_date: 종료 날짜 (YYYYMMDD 형식)
            output_dir: 저장 디렉토리
            
        Returns:
            저장된 파일 경로
        """
        if output_dir is None:
            output_dir = DEFAULT_DATA_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"수집 중: {station} ({start_date} ~ {end_date})...")
        
        df = self.get_daily_data(station, start_date, end_date)
        
        # 저장
        filename = f"weather_asos_{station}_{start_date}_{end_date}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        
        print(f"\n데이터 저장 완료: {filepath}")
        print(f"총 {len(df)}개의 관측 데이터 (일 단위)")
        
        return filepath
    
    def collect_by_year(self, station: str, start_year: int, end_year: int,
                        output_dir: str = None, interval_seconds: float = 0.5) -> List[str]:
        """
        연도별로 데이터를 수집하여 각각 별도의 CSV 파일로 저장
        
        Args:
            station: 관측소명
            start_year: 시작 연도 (예: 2017)
            end_year: 종료 연도 (예: 2026)
            output_dir: 저장 디렉토리
            interval_seconds: API 호출 간 대기 시간
            
        Returns:
            저장된 파일 경로 리스트
        """
        if output_dir is None:
            output_dir = DEFAULT_DATA_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        current_date = datetime.now()
        
        for year in range(start_year, end_year + 1):
            start_date = f"{year}0101"
            
            # 현재 연도인 경우 어제 날짜까지만
            if year == current_date.year:
                yesterday = current_date - timedelta(days=1)
                end_date = yesterday.strftime("%Y%m%d")
            else:
                end_date = f"{year}1231"
            
            # 미래 연도는 건너뛰기
            if year > current_date.year:
                continue
            
            print(f"수집 중: {station} {year}년 ({start_date} ~ {end_date})...")
            
            try:
                df = self.get_daily_data(station, start_date, end_date)
                
                # 연도별 파일로 저장
                filename = f"weather_asos_{station}_{year}.csv"
                filepath = os.path.join(output_dir, filename)
                df.to_csv(filepath, index=False, encoding="utf-8-sig")
                
                print(f"  → 저장 완료: {filename} ({len(df)}개)")
                saved_files.append(filepath)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                print(f"  → 오류 발생: {e}")
                continue
        
        print(f"\n총 {len(saved_files)}개 파일 저장 완료!")
        return saved_files


class KMACollector:
    """한국 기상청 단기예보 API - 실시간 예보 데이터"""
    
    SHORT_FORECAST_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0"
    
    CITY_COORDS = {
        "서울": (60, 127),
        "부산": (98, 76),
        "대구": (89, 90),
        "인천": (55, 124),
        "광주": (58, 74),
        "대전": (67, 100),
        "울산": (102, 84),
        "세종": (66, 103),
        "수원": (60, 121),
        "전주": (63, 89),
        "청주": (69, 107),
        "춘천": (73, 134),
        "제주": (52, 38),
    }
    
    def __init__(self, api_key: str = KMA_API_KEY):
        self.api_key = api_key
        
    def _get_base_datetime(self) -> tuple:
        """단기예보 발표 시간 계산 (02, 05, 08, 11, 14, 17, 20, 23시)"""
        now = datetime.now()
        base_times = ["0200", "0500", "0800", "1100", "1400", "1700", "2000", "2300"]
        
        current_time = now.strftime("%H%M")
        base_time = "2300"
        base_date = now
        
        for bt in base_times:
            if current_time >= bt:
                base_time = bt
            else:
                break
        
        if current_time < "0200":
            base_date = now - timedelta(days=1)
            base_time = "2300"
            
        return base_date.strftime("%Y%m%d"), base_time
    
    def get_short_forecast(self, city: str = "서울", num_rows: int = 1000) -> pd.DataFrame:
        """단기예보 조회 (3일 예보)"""
        if city not in self.CITY_COORDS:
            raise ValueError(f"지원하지 않는 도시: {city}")
        
        nx, ny = self.CITY_COORDS[city]
        base_date, base_time = self._get_base_datetime()
        
        params = {
            "serviceKey": self.api_key,
            "pageNo": 1,
            "numOfRows": num_rows,
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": base_time,
            "nx": nx,
            "ny": ny,
        }
        
        response = requests.get(f"{self.SHORT_FORECAST_URL}/getVilageFcst", params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data["response"]["header"]["resultCode"] != "00":
            raise Exception(f"API 오류: {data['response']['header']['resultMsg']}")
        
        items = data["response"]["body"]["items"]["item"]
        
        records = {}
        for item in items:
            key = f"{item['fcstDate']}_{item['fcstTime']}"
            if key not in records:
                records[key] = {
                    "forecast_date": item["fcstDate"],
                    "forecast_time": item["fcstTime"],
                }
            records[key][item["category"]] = item["fcstValue"]
        
        df = pd.DataFrame(list(records.values()))
        
        column_mapping = {
            "POP": "강수확률(%)",
            "PTY": "강수형태",
            "PCP": "1시간강수량(mm)",
            "REH": "습도(%)",
            "SNO": "1시간신적설(cm)",
            "SKY": "하늘상태",
            "TMP": "기온(℃)",
            "TMN": "일최저기온(℃)",
            "TMX": "일최고기온(℃)",
            "UUU": "동서바람성분(m/s)",
            "VVV": "남북바람성분(m/s)",
            "WAV": "파고(M)",
            "VEC": "풍향(deg)",
            "WSD": "풍속(m/s)",
        }
        
        df = df.rename(columns=column_mapping)
        return df


def main():
    """사용 예시"""
    
    print("=" * 60)
    print("한국 기상청 과거 날씨 데이터 수집 (ASOS 일자료)")
    print("=" * 60)
    
    # ASOS 수집기 초기화
    asos = ASOSCollector()
    
    # 2017년부터 현재까지 서울 데이터 수집 (연도별 파일 분리)
    print("\n[서울 과거 데이터 수집: 2017년 ~ 현재 (연도별 파일)]")
    
    try:
        filepaths = asos.collect_by_year(
            station="서울",
            start_year=2017,
            end_year=2026,
            interval_seconds=0.5  # API 호출 간 대기
        )
        
        print("\n[저장된 파일 목록]")
        for fp in filepaths:
            print(f"  - {os.path.basename(fp)}")
        
    except Exception as e:
        print(f"수집 실패: {e}")


if __name__ == "__main__":
    main()
