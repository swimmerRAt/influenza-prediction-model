import axios from 'axios';
import { getAccessToken } from './auth';

// 직접 URL 사용 (211.238.12.60:8084)
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://211.238.12.60:8084/data/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 요청 인터셉터 - 인증 토큰 자동 추가
apiClient.interceptors.request.use(
  async (config) => {
    try {
      const token = await getAccessToken();
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    } catch (error) {
      console.warn('토큰 가져오기 실패 (인증 없이 요청 진행):', error.message);
      // 토큰 가져오기 실패해도 요청은 진행 (서버에서 401 에러 처리)
      // 환경 변수가 설정되지 않은 경우에도 계속 진행
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 응답 인터셉터 (에러 처리 및 토큰 만료 처리)
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    console.error('API Error:', error);
    
    if (error.response) {
      // 401 Unauthorized - 토큰 만료 또는 무효
      if (error.response.status === 401) {
        console.warn('인증 토큰이 만료되었습니다. 새 토큰을 요청합니다.');
        // 토큰 제거하고 재시도
        const authModule = await import('./auth');
        if (authModule.clearToken) {
          authModule.clearToken();
        }
        
        try {
          // 새 토큰으로 재시도
          const token = await getAccessToken();
          if (token) {
            error.config.headers.Authorization = `Bearer ${token}`;
            return apiClient.request(error.config);
          }
        } catch (tokenError) {
          console.error('토큰 갱신 실패:', tokenError);
        }
      }
      
      // 서버가 응답했지만 에러 상태 코드
      console.error('Error Response:', error.response.data);
      console.error('Status:', error.response.status);
    } else if (error.request) {
      // 요청이 전송되었지만 응답을 받지 못함
      console.error('No Response:', error.request);
    } else {
      // 요청 설정 중 에러 발생
      console.error('Error:', error.message);
    }
    
    return Promise.reject(error);
  }
);

export default apiClient;

