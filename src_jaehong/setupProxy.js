const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  console.log('프록시 설정 로드됨');
  
  // Keycloak 인증 프록시
  app.use(
    '/keycloak-proxy',
    createProxyMiddleware({
      target: 'https://keycloak.211.238.12.60.nip.io:8100',
      changeOrigin: true,
      secure: false, // SSL 인증서 검증 비활성화 (개발 환경)
      pathRewrite: {
        '^/keycloak-proxy': '', // /keycloak-proxy를 제거
      },
      logLevel: 'debug',
      onProxyReq: (proxyReq, req, res) => {
        console.log('Keycloak 프록시 요청:', req.url, '->', proxyReq.path);
      },
      onError: (err, req, res) => {
        console.error('Keycloak 프록시 에러:', err.message);
      },
    })
  );

  // API 프록시
  app.use(
    '/api-proxy',
    createProxyMiddleware({
      target: 'http://211.238.12.60:8084',
      changeOrigin: true,
      pathRewrite: {
        '^/api-proxy': '', // /api-proxy를 제거
      },
      logLevel: 'debug',
      onProxyReq: (proxyReq, req, res) => {
        console.log('API 프록시 요청:', req.url, '->', proxyReq.path);
      },
      onError: (err, req, res) => {
        console.error('API 프록시 에러:', err.message);
      },
    })
  );
};

