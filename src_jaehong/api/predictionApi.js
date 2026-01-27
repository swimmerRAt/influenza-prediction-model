import axios from 'axios';

// 예측 모델 Flask 서버 URL (환경 변수로 설정 가능)
const PREDICTION_API_URL = process.env.REACT_APP_PREDICTION_API_URL || 'http://210.117.143.172:6302';

/**
 * 인플루엔자 의사환자 분율 예측 데이터 가져오기
 * @param {Array<number>} inputData - 예측에 사용할 입력 데이터 (선택사항)
 * @param {number} steps - 예측할 스텝 수 (기본값: 3)
 * @returns {Promise} 예측 데이터
 */
export const getPrediction = async (inputData = null, steps = 3) => {
  try {
    const params = {};
    if (inputData && Array.isArray(inputData)) {
      params.input_data = JSON.stringify(inputData);
    }
    if (steps) {
      params.steps = steps;
    }
    
    const response = await axios.get(`${PREDICTION_API_URL}/predict`, {
      params,
      timeout: 30000,
    });
    return response.data;
  } catch (error) {
    console.error('예측 데이터 로딩 실패:', error);
    throw error;
  }
};

/**
 * 예측 모델의 특징 목록 가져오기
 * @returns {Promise} 특징 목록
 */
export const getFeatures = async () => {
  try {
    const response = await axios.get(`${PREDICTION_API_URL}/features`, {
      timeout: 30000,
    });
    return response.data;
  } catch (error) {
    console.error('특징 목록 로딩 실패:', error);
    throw error;
  }
};
