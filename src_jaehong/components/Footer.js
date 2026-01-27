import React from 'react';
import { Box, Container, Typography } from '@mui/material';

const Footer = ({ isOpen = true }) => {
  return (
    <Box
      component="footer"
      sx={{
        backgroundColor: '#1a202c',
        borderTop: '1px solid rgba(255, 255, 255, 0.1)',
        mt: 'auto',
        py: 2,
        marginLeft: isOpen ? '240px' : '64px',
        transition: 'margin-left 0.3s ease',
      }}
    >
      <Container maxWidth="xl">
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: { xs: 'flex-start', sm: 'center' },
            flexDirection: { xs: 'column', sm: 'row' },
            gap: 1.5,
          }}
        >
          <Typography
            variant="caption"
            sx={{
              color: '#cbd5f5',
              lineHeight: 1.6,
            }}
          >
            1) 표본감시기관의 보고시점을 기준으로 취합 및 분석한 잠정통계로 변동 가능함<br />
            2) 동일주차별 비교를 위하여 22/23절기 53주를 제외한 그 외 절기의 53주는 52주와 동일(52주 중복)
          </Typography>
          <Typography variant="body2" sx={{ fontSize: '0.875rem', color: '#a0aec0' }}>
            최근 업데이트 일시: 2025-11-03
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default Footer;
