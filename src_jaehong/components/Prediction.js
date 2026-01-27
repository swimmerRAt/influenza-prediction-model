import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  Container,
  Paper,
  Typography,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { getPrediction } from '../api/predictionApi';
import { useInfluenzaData } from '../hooks/useInfluenzaData';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
);

const PRIMARY_COLOR = '#38bdf8';
const PRIMARY_FILL = 'rgba(56, 189, 248, 0.2)';
const PREDICTION_COLOR = '#ef4444';
const PREDICTION_FILL = 'rgba(239, 68, 68, 0.2)';

const Prediction = ({ isOpen = true }) => {
  const [predictionData, setPredictionData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // 환경 변수에서 DSID 가져오기
  const defaultDSID = process.env.REACT_APP_DSID || 'ds_0101';
  const defaultSeason = '24/25';
  const defaultWeek = '37';

  // 최근 데이터 가져오기 (12스텝)
  const { influenzaData, loading: dataLoading } = useInfluenzaData(
    defaultSeason,
    defaultWeek,
    defaultDSID
  );

  // 예측 데이터 가져오기
  useEffect(() => {
    const fetchPrediction = async () => {
      setLoading(true);
      setError(null);

      try {
        const data = await getPrediction();
        if (data.success) {
          setPredictionData(data);
        } else {
          setError(data.error || '예측 데이터를 가져오는데 실패했습니다.');
        }
      } catch (err) {
        console.error('예측 데이터 로딩 실패:', err);
        let errorMessage = '예측 데이터를 불러오는데 실패했습니다.';
        
        if (err.response) {
          errorMessage = `서버 오류 (${err.response.status}): ${err.response.data?.error || err.message || '알 수 없는 오류'}`;
        } else if (err.request) {
          errorMessage = '예측 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.';
        } else {
          errorMessage = err.message || '알 수 없는 오류가 발생했습니다.';
        }
        
        setError(errorMessage);
      } finally {
        setLoading(false);
      }
    };

    fetchPrediction();
  }, []);

  // 차트 데이터 생성 (최근 12스텝 + 예측 3스텝)
  const chartData = useMemo(() => {
    if (!influenzaData?.ili || !predictionData) {
      return null;
    }

    // 최근 12스텝 데이터 가져오기
    const iliWeeks = influenzaData.ili.weeks || [];
    const iliValues = influenzaData.ili.values || [];

    // 최근 12개만 선택
    const recentWeeks = iliWeeks.slice(-12);
    const recentValues = iliValues.slice(-12);

    // 예측 데이터
    const predictions = predictionData.predictions || [];
    const predictionLength = predictionData.prediction_length || 3;

    // 예측 주차 생성 (마지막 주차 다음부터)
    const lastWeek = recentWeeks[recentWeeks.length - 1];
    const lastWeekStr = lastWeek.toString().replace(/주/g, '').trim();
    const lastWeekNum = parseInt(lastWeekStr) || 0;
    
    const predictionWeeks = [];
    for (let i = 1; i <= predictionLength; i++) {
      let weekNum = lastWeekNum + i;
      // 53주를 넘어가면 다음 해 1주로 (53주는 보통 없지만 안전장치)
      if (weekNum > 53) {
        weekNum = weekNum - 53;
      }
      predictionWeeks.push(`${weekNum}주`);
    }

    // 전체 주차와 값 결합
    const allWeeks = [...recentWeeks, ...predictionWeeks];
    const allValues = [...recentValues, ...predictions];

    // 실제 데이터의 마지막 값 (예측 데이터와 연결점)
    const lastActualValue = recentValues[recentValues.length - 1];

    // 실제 데이터와 예측 데이터 구분을 위한 배열
    // 실제 데이터: 최근 데이터만 표시
    const actualData = [...recentValues, ...new Array(predictionLength).fill(null)];
    
    // 예측 데이터: 마지막 실제 값에서 시작하여 예측 값으로 자연스럽게 연결
    const predictedData = [
      ...new Array(recentValues.length - 1).fill(null), // 마지막 값 전까지는 null
      lastActualValue, // 마지막 실제 값 (연결점)
      ...predictions   // 예측 값들
    ];

    return {
      labels: allWeeks,
      datasets: [
        {
          label: '실제 의사환자 분율',
          data: actualData,
          borderColor: PRIMARY_COLOR,
          backgroundColor: PRIMARY_FILL,
          fill: true,
          tension: 0.35,
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: PRIMARY_COLOR,
          pointBorderColor: '#0f172a',
          pointBorderWidth: 1.5,
        },
        {
          label: 'AI 예측',
          data: predictedData,
          borderColor: PREDICTION_COLOR,
          backgroundColor: PREDICTION_FILL,
          fill: true,
          tension: 0.35,
          borderWidth: 2,
          borderDash: [5, 5],
          pointRadius: 4,
          pointBackgroundColor: PREDICTION_COLOR,
          pointBorderColor: '#0f172a',
          pointBorderWidth: 1.5,
        },
      ],
    };
  }, [influenzaData, predictionData]);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 15,
          font: {
            size: 11,
          },
          color: '#374151',
        },
      },
      tooltip: {
        backgroundColor: '#0f172a',
        titleColor: '#f8fafc',
        bodyColor: '#f8fafc',
        borderColor: 'rgba(148, 163, 184, 0.4)',
        borderWidth: 1,
        padding: 10,
        callbacks: {
          title: contexts => {
            if (!contexts?.length) return '';
            const label = contexts[0].label ?? '';
            return `< ${label} >`;
          },
          label: context => {
            const value = context.parsed.y;
            if (value == null) return '데이터 없음';
            return `${context.dataset.label}: ${value.toFixed(2)}`;
          },
        },
      },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: {
          color: '#6b7280',
          font: { size: 10 },
          maxRotation: 45,
          minRotation: 0,
        },
      },
      y: {
        grid: { color: 'rgba(148, 163, 184, 0.2)', borderDash: [4, 4] },
        ticks: { color: '#6b7280', font: { size: 10 } },
        title: {
          display: true,
          text: '인플루엔자 의사환자 분율(/1,000명 당)',
          color: '#6b7280',
          font: { size: 11 },
        },
      },
    },
    interaction: { intersect: false, mode: 'index' },
  };

  return (
    <Box
      sx={{
        backgroundColor: '#f8fafc',
        minHeight: '100vh',
        color: '#1f2937',
        py: 4,
        marginLeft: isOpen ? '240px' : '64px',
        marginTop: '60px',
        transition: 'margin-left 0.3s ease',
      }}
    >
      <Container maxWidth="xl">
        <Box
          sx={{
            borderRadius: 4,
            boxShadow: '0 40px 120px rgba(0, 0, 0, 0.1)',
            background: 'linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%)',
            border: '1px solid rgba(203, 213, 225, 0.2)',
            display: 'flex',
            overflow: 'hidden',
          }}
        >
          <Box sx={{ flex: 1, p: { xs: 3, md: 5 }, display: 'flex', flexDirection: 'column', gap: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 3 }}>
              <Typography
                variant="h5"
                sx={{
                  fontWeight: 700,
                  color: '#1f2937',
                  fontFamily: 'Pretendard',
                }}
              >
                AI 예측 - 인플루엔자 의사환자 분율
              </Typography>
            </Box>

            {/* 로딩 상태 */}
            {(loading || dataLoading) && (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', py: 4 }}>
                <CircularProgress sx={{ mr: 2 }} />
                <Typography variant="body1" sx={{ color: '#6b7280' }}>
                  데이터를 불러오는 중...
                </Typography>
              </Box>
            )}

            {/* 에러 상태 */}
            {error && (
              <Alert
                severity="error"
                sx={{ mb: 3 }}
                onClose={() => setError(null)}
              >
                {error}
                <Box sx={{ mt: 1, fontSize: '0.875rem', color: '#6b7280' }}>
                  예측 서버가 실행 중인지 확인하세요. (기본 URL: http://localhost:6302)
                </Box>
              </Alert>
            )}

            {/* 차트 */}
            {!loading && !dataLoading && chartData && (
              <Paper
                elevation={0}
                sx={{
                  p: 4,
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  borderRadius: 4,
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                }}
              >
                <Typography variant="h6" sx={{ fontWeight: 700, color: '#1f2937', mb: 3 }}>
                  인플루엔자 의사환자 분율 예측
                </Typography>
                <Box sx={{ height: 400, mt: 3 }}>
                  <Line data={chartData} options={chartOptions} />
                </Box>
                <Typography variant="caption" sx={{ color: 'rgba(148, 163, 184, 0.7)', display: 'block', mt: 2 }}>
                  최근 12주차의 실제 데이터와 AI 모델이 예측한 향후 3주차의 의사환자 분율을 표시합니다.
                </Typography>
                {predictionData && (
                  <Box sx={{ mt: 2, p: 2, backgroundColor: 'rgba(239, 246, 255, 0.8)', borderRadius: 2 }}>
                    <Typography variant="body2" sx={{ color: '#1e40af', fontWeight: 600, mb: 1 }}>
                      예측 정보
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#64748b' }}>
                      입력 길이: {predictionData.input_length}주차 | 예측 길이: {predictionData.prediction_length}주차 | 단위: {predictionData.unit}
                    </Typography>
                  </Box>
                )}
              </Paper>
            )}

            {/* 데이터 없음 */}
            {!loading && !dataLoading && !chartData && !error && (
              <Alert severity="info">
                데이터를 불러올 수 없습니다. 최근 데이터와 예측 데이터가 모두 필요합니다.
              </Alert>
            )}
          </Box>
        </Box>
      </Container>
    </Box>
  );
};

export default Prediction;
