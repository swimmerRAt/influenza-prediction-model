import axios from 'axios';

// Keycloak 인증 설정
const KEYCLOAK_SERVER_URL = process.env.REACT_APP_KEYCLOAK_SERVER_URL;
const KEYCLOAK_REALM = process.env.REACT_APP_KEYCLOAK_REALM;
const CLIENT_ID = process.env.REACT_APP_CLIENT_ID;
const CLIENT_SECRET = process.env.REACT_APP_CLIENT_SECRET;

// 환경 변수 확인
const isAuthConfigured = KEYCLOAK_SERVER_URL && 
                         KEYCLOAK_REALM && 
                         CLIENT_ID && 
                         CLIENT_SECRET &&
                         !KEYCLOAK_SERVER_URL.includes('{{') &&
                         !KEYCLOAK_REALM.includes('{{') &&
                         !CLIENT_ID.includes('{{') &&
                         !CLIENT_SECRET.includes('{{');

const TOKEN_KEY = 'auth_token';
const TOKEN_EXPIRY_KEY = 'auth_token_expiry';

/**
 * Keycloak에서 액세스 토큰 가져오기
 * @returns {Promise<string>} 액세스 토큰
 */
export const getAccessToken = async () => {
  // 환경 변수가 설정되지 않았으면 에러
  if (!isAuthConfigured) {
    console.warn('Keycloak 인증 설정이 완료되지 않았습니다. .env 파일을 확인하세요.');
    throw new Error('인증 설정이 완료되지 않았습니다. 환경 변수를 확인하세요.');
  }

  // 저장된 토큰이 있고 아직 유효한지 확인
  const storedToken = localStorage.getItem(TOKEN_KEY);
  const tokenExpiry = localStorage.getItem(TOKEN_EXPIRY_KEY);
  
  if (storedToken && tokenExpiry && new Date().getTime() < parseInt(tokenExpiry)) {
    return storedToken;
  }

  // 새 토큰 요청
  try {
    // 개발 환경에서는 프록시 사용, 프로덕션에서는 직접 호출
    const useProxy = process.env.NODE_ENV === 'development';
    const baseUrl = useProxy 
      ? '/keycloak-proxy' 
      : KEYCLOAK_SERVER_URL;
    
    const tokenUrl = `${baseUrl}/realms/${KEYCLOAK_REALM}/protocol/openid-connect/token`;
    
    const params = new URLSearchParams();
    params.append('client_id', CLIENT_ID);
    params.append('client_secret', CLIENT_SECRET);
    params.append('grant_type', 'client_credentials');

    const response = await axios.post(tokenUrl, params, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      withCredentials: false,
    });

    const { access_token, expires_in } = response.data;
    
    // 토큰 저장
    localStorage.setItem(TOKEN_KEY, access_token);
    
    // 만료 시간 저장 (expires_in은 초 단위이므로 밀리초로 변환)
    const expiryTime = new Date().getTime() + (expires_in * 1000) - 60000; // 1분 여유
    localStorage.setItem(TOKEN_EXPIRY_KEY, expiryTime.toString());

    return access_token;
  } catch (error) {
    console.error('토큰 발급 실패:', error);
    if (error.response) {
      console.error('응답 상태:', error.response.status);
      console.error('응답 데이터:', error.response.data);
    }
    throw new Error('인증 토큰을 가져올 수 없습니다.');
  }
};

/**
 * 저장된 토큰 제거
 */
export const clearToken = () => {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(TOKEN_EXPIRY_KEY);
};

/**
 * 토큰이 유효한지 확인
 * @returns {boolean}
 */
export const isTokenValid = () => {
  const token = localStorage.getItem(TOKEN_KEY);
  const expiry = localStorage.getItem(TOKEN_EXPIRY_KEY);
  
  if (!token || !expiry) {
    return false;
  }
  
  return new Date().getTime() < parseInt(expiry);
};

