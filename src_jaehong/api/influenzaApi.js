import apiClient from './config';

/**
 * 인플루엔자 의사환자 분율 (ILI) 데이터 가져오기
 * @param {string} season - 절기 (예: '25/26')
 * @param {string} week - 주차 (예: '37')
 * @returns {Promise} ILI 데이터
 */
export const getILIData = async (season, week) => {
  try {
    const response = await apiClient.get('/influenza/ili', {
      params: {
        season,
        week,
      },
    });
    return response.data;
  } catch (error) {
    console.error('ILI 데이터 로딩 실패:', error);
    throw error;
  }
};

/**
 * 급성호흡기감염증 환자 중 인플루엔자 환자 수 (ARI) 데이터 가져오기
 * @param {string} season - 절기
 * @param {string} week - 주차
 * @returns {Promise} ARI 데이터
 */
export const getARIData = async (season, week) => {
  try {
    const response = await apiClient.get('/influenza/ari', {
      params: { season, week },
    });
    return response.data;
  } catch (error) {
    console.error('ARI 데이터 로딩 실패:', error);
    throw error;
  }
};

/**
 * 중증급성호흡기감염증 환자 중 인플루엔자 환자 수 (SARI) 데이터 가져오기
 * @param {string} season - 절기
 * @param {string} week - 주차
 * @returns {Promise} SARI 데이터
 */
export const getSARIData = async (season, week) => {
  try {
    const response = await apiClient.get('/influenza/sari', {
      params: { season, week },
    });
    return response.data;
  } catch (error) {
    console.error('SARI 데이터 로딩 실패:', error);
    throw error;
  }
};

/**
 * 검사기관 인플루엔자 검출률 (I-RISS) 데이터 가져오기
 * @param {string} season - 절기
 * @param {string} week - 주차
 * @returns {Promise} I-RISS 데이터
 */
export const getIRISSData = async (season, week) => {
  try {
    const response = await apiClient.get('/influenza/iriss', {
      params: { season, week },
    });
    return response.data;
  } catch (error) {
    console.error('I-RISS 데이터 로딩 실패:', error);
    throw error;
  }
};

/**
 * 의원급 의료기관 인플루엔자 검출률 (K-RISS) 데이터 가져오기
 * @param {string} season - 절기
 * @param {string} week - 주차
 * @returns {Promise} K-RISS 데이터
 */
export const getKRISSData = async (season, week) => {
  try {
    const response = await apiClient.get('/influenza/kriss', {
      params: { season, week },
    });
    return response.data;
  } catch (error) {
    console.error('K-RISS 데이터 로딩 실패:', error);
    throw error;
  }
};

/**
 * 응급실 인플루엔자 환자 수 (NEDIS) 데이터 가져오기
 * @param {string} season - 절기
 * @param {string} week - 주차
 * @returns {Promise} NEDIS 데이터
 */
export const getNEDISData = async (season, week) => {
  try {
    const response = await apiClient.get('/influenza/nedis', {
      params: { season, week },
    });
    return response.data;
  } catch (error) {
    console.error('NEDIS 데이터 로딩 실패:', error);
    throw error;
  }
};

/**
 * 모든 인플루엔자 지표 데이터를 한번에 가져오기
 * @param {string} season - 절기
 * @param {string} week - 주차
 * @returns {Promise} 모든 인플루엔자 데이터
 */
export const getAllInfluenzaData = async (season, week) => {
  try {
    const response = await apiClient.get('/influenza/all', {
      params: { season, week },
    });
    return response.data;
  } catch (error) {
    console.error('인플루엔자 데이터 로딩 실패:', error);
    throw error;
  }
};

/**
 * 유행단계 데이터 가져오기
 * @returns {Promise} 유행단계 데이터
 */
export const getInfluenzaStage = async () => {
  try {
    const response = await apiClient.get('/influenza/stage');
    return response.data;
  } catch (error) {
    console.error('유행단계 데이터 로딩 실패:', error);
    throw error;
  }
};

/**
 * 주간 지표 요약 데이터 가져오기
 * @returns {Promise} 주간 지표 요약 데이터
 */
export const getWeeklySummary = async () => {
  try {
    const response = await apiClient.get('/influenza/weekly-summary');
    return response.data;
  } catch (error) {
    console.error('주간 지표 요약 데이터 로딩 실패:', error);
    throw error;
  }
};


