import apiClient from './config';
import axios from 'axios';
import { getAccessToken } from './auth';

/**
 * 절기를 날짜 범위로 변환하는 함수
 * @param {string} season - 절기 (예: '25/26')
 * @returns {Object} {from: string, to: string} - ISO 날짜 형식 (YYYY-MM-DD)
 */
const getDateRangeFromSeason = (season) => {
  // 절기 정의: XX/YY절기 = XX년 36주 ~ YY년 35주
  // 예: 25/26절기 = 2025년 36주 ~ 2026년 35주
  const [year1, year2] = season.split('/').map(y => parseInt('20' + y));
  
  // XX년 36주 시작일 계산 (대략 9월 첫째 주)
  const startDate = new Date(year1, 8, 1); // 9월 1일
  
  // YY년 35주 종료일 계산 (대략 8월 마지막 주)
  const endDate = new Date(year2, 7, 31); // 8월 31일
  
  return {
    from: startDate.toISOString().split('T')[0],
    to: endDate.toISOString().split('T')[0],
  };
};

/**
 * 자동수집 데이터중 특정 id의 최근 n건 데이터 조회
 * @param {string} dsid - 데이터셋 ID
 * @param {number} cnt - 조회할 데이터 건수
 * @returns {Promise} 최근 데이터
 */
export const getRecentETLData = async (dsid, cnt) => {
  try {
    const response = await apiClient.get(`/etl_data/id/${dsid}/recent/${cnt}`);
    return response.data;
  } catch (error) {
    console.error(`ETL 데이터 조회 실패 (dsid: ${dsid}, cnt: ${cnt}):`, error);
    throw error;
  }
};

/**
 * 자동수집 데이터중 특정 id의 특정 기간 데이터 조회
 * @param {string} dsid - 데이터셋 ID
 * @param {string} from - 시작 날짜 (YYYY-MM-DD 형식)
 * @param {string} to - 종료 날짜 (YYYY-MM-DD 형식)
 * @returns {Promise} 기간별 데이터
 */
export const getETLDataByDateRange = async (dsid, from, to) => {
  try {
    const apiUrl = `/etl_data/id/${dsid}/from/${from}/to/${to}`;
    
    console.log(`🔵 [날짜 범위 API] 요청 URL:`, apiUrl);
    console.log(`🔵 [날짜 범위 API] 요청 파라미터:`, { dsid, from, to });
    
    const response = await apiClient.get(apiUrl);
    
    console.log(`✅ [날짜 범위 API] 응답 성공:`, {
      status: response.status,
      statusText: response.statusText,
      dataType: typeof response.data,
      dataKeys: response.data ? Object.keys(response.data) : [],
      dataLength: Array.isArray(response.data) ? response.data.length : 'N/A',
    });
    
    if (response.data) {
      const rawData = response.data?.body?.data || response.data?.data || response.data;
      console.log(`📦 [날짜 범위 API] 실제 데이터:`, {
        데이터개수: Array.isArray(rawData) ? rawData.length : 'N/A',
        샘플데이터: Array.isArray(rawData) && rawData.length > 0 ? rawData[0] : null,
      });
    }
    
    return response.data;
  } catch (error) {
    console.error(`❌ [날짜 범위 API] 실패:`, {
      dsid,
      from,
      to,
      error: error.message,
      response: error.response?.data,
      status: error.response?.status,
      statusText: error.response?.statusText,
    });
    throw error;
  }
};

/**
 * 자동수집 데이터중 id별 총 데이터 수 조회
 * @returns {Promise} 통계 데이터
 */
export const getETLDataStatistics = async () => {
  try {
    const response = await apiClient.get('/etl_data/statistics');
    return response.data;
  } catch (error) {
    console.error('ETL 데이터 통계 조회 실패:', error);
    throw error;
  }
};

/**
 * 자동수집 데이터중 id별 특정 기간내 총 데이터 수 조회
 * @param {string} from - 시작 날짜 (YYYY-MM-DD 형식)
 * @param {string} to - 종료 날짜 (YYYY-MM-DD 형식)
 * @returns {Promise} 기간별 통계 데이터
 */
export const getETLDataStatisticsByDateRange = async (from, to) => {
  try {
    const response = await apiClient.get(`/etl_data/statistics/from/${from}/to/${to}`);
    return response.data;
  } catch (error) {
    console.error(`ETL 데이터 통계 조회 실패 (from: ${from}, to: ${to}):`, error);
    throw error;
  }
};

/**
 * 자동수집 데이터중 특정 id와 origin으로 데이터 조회
 * @param {string} dsid - 데이터셋 ID
 * @param {string} origin - origin 값
 * @returns {Promise} origin별 데이터
 */
export const getETLDataByOrigin = async (dsid, origin) => {
  try {
    // 전체 URL 사용: http://211.238.12.60:8084/data/api/v1/etl_data/id/{{dsid}}/origin/{{origin}}
    const fullUrl = `http://211.238.12.60:8084/data/api/v1/etl_data/id/${dsid}/origin/${origin}`;
    
    console.log(`🔵 [origin API] 요청 URL:`, fullUrl);
    console.log(`🔵 [origin API] 요청 파라미터:`, { dsid, origin });
    
    // 인증 토큰 가져오기
    let token = null;
    try {
      token = await getAccessToken();
    } catch (tokenError) {
      console.warn('토큰 가져오기 실패 (인증 없이 요청 진행):', tokenError.message);
    }
    
    // axios를 직접 사용하여 전체 URL로 요청
    const response = await axios.get(fullUrl, {
      headers: {
        'Content-Type': 'application/json',
        ...(token && { Authorization: `Bearer ${token}` }),
      },
      timeout: 30000,
    });
    
    console.log(`✅ [origin API] 응답 성공:`, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
      dataType: typeof response.data,
      dataKeys: response.data ? Object.keys(response.data) : [],
      dataLength: Array.isArray(response.data) ? response.data.length : 'N/A',
    });
    
    if (response.data) {
      const rawData = response.data?.body?.data || response.data?.data || response.data;
      console.log(`📦 [origin API] 실제 데이터:`, {
        데이터개수: Array.isArray(rawData) ? rawData.length : 'N/A',
        샘플데이터: Array.isArray(rawData) && rawData.length > 0 ? rawData[0] : null,
      });
    }
    
    return response.data;
  } catch (error) {
    console.error(`❌ [origin API] 실패:`, {
      origin,
      error: error.message,
      response: error.response?.data,
      status: error.response?.status,
      statusText: error.response?.statusText,
    });
    throw error;
  }
};

/**
 * 자동수집 데이터중 특정 id의 절기별 데이터 조회
 * 25/26절기는 origin별로 요청, 나머지는 날짜 범위로 요청
 * @param {string} dsid - 데이터셋 ID
 * @param {string} season - 절기 (예: '25/26')
 * @param {Array<string>} origins - origin 목록 (25/26절기인 경우 필수)
 * @returns {Promise} 절기별 데이터
 */
export const getETLDataBySeason = async (dsid, season, origins = null) => {
  try {
    // 25/26절기는 origin별로 요청
    if (season === '25/26' && origins && origins.length > 0) {
      console.log(`🔵 [${season}절기 API] origin별 요청 시작 (${origins.length}개)`);
      
      const allData = [];
      
      // 각 origin별로 요청
      for (const origin of origins) {
        try {
          const originData = await getETLDataByOrigin(dsid, origin);
          const originRawData = originData?.body?.data || originData?.data || originData;
          
          if (Array.isArray(originRawData)) {
            allData.push(...originRawData);
          } else if (originRawData) {
            allData.push(originRawData);
          }
        } catch (err) {
          console.warn(`⚠️ [${season}절기 API] origin ${origin} 요청 실패:`, err.message);
        }
      }
      
      console.log(`✅ [${season}절기 API] origin별 요청 완료: 총 ${allData.length}건`);
      
      return {
        body: { data: allData },
        data: allData,
      };
    } else {
      // 나머지 절기는 날짜 범위로 요청
      const dateRange = getDateRangeFromSeason(season);
      const apiUrl = `/etl_data/id/${dsid}/from/${dateRange.from}/to/${dateRange.to}`;
      
      console.log(`🔵 [${season}절기 API] 요청: ${apiUrl}`);
      console.log(`   날짜 범위: ${dateRange.from} ~ ${dateRange.to}`);
      
      const response = await apiClient.get(apiUrl);
      
      console.log(`✅ [${season}절기 API] 응답 성공:`, response.status);
      
      return response.data;
    }
  } catch (error) {
    console.error(`❌ [${season}절기 API] 실패:`, error.message);
    throw error;
  }
};



