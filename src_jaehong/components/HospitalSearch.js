import React, { useState, useEffect, useRef } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  Typography,
  IconButton,
  Box,
  Stack,
  TextField,
  Button,
  CircularProgress,
  Alert,
} from '@mui/material';
import { FiX } from 'react-icons/fi';

const HospitalSearch = ({ open, onClose }) => {
  const [searchKeyword, setSearchKeyword] = useState('');
  const [hospitals, setHospitals] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [map, setMap] = useState(null);
  const [markers, setMarkers] = useState([]);
  const [infoWindows, setInfoWindows] = useState([]);
  const mapContainerRef = useRef(null);
  const psRef = useRef(null); // Places 서비스 참조

  // 카카오맵 스크립트 로드 확인 및 초기화
  useEffect(() => {
    console.log('🗺️ 지도 초기화 useEffect 실행됨, open:', open);
    
    if (!open) return;

    // 카카오맵 스크립트 로드 함수
    const loadKakaoMapScript = () => {
      return new Promise((resolve, reject) => {
        // 이미 로드되어 있으면 바로 resolve
        if (window.kakao && window.kakao.maps && window.kakao.maps.services) {
          console.log('✅ 카카오맵 API 이미 로드됨');
          resolve();
          return;
        }

        // 스크립트가 이미 있는지 확인하고 로드 대기
        const existingScript = document.querySelector('script[src*="dapi.kakao.com"]');
        if (existingScript) {
          console.log('⏳ 카카오맵 스크립트 로딩 중...');
          
          // 이미 로드되었는지 확인
          const checkLoaded = setInterval(() => {
            if (window.kakao && window.kakao.maps && window.kakao.maps.services) {
              clearInterval(checkLoaded);
              console.log('✅ 카카오맵 스크립트 로드 완료');
              resolve();
            }
          }, 100);

          // 최대 10초 대기
          setTimeout(() => {
            clearInterval(checkLoaded);
            if (!window.kakao || !window.kakao.maps) {
              reject(new Error('카카오맵 스크립트 로드 시간 초과'));
            }
          }, 10000);

          existingScript.addEventListener('error', () => {
            clearInterval(checkLoaded);
            reject(new Error('카카오맵 스크립트 로드 실패'));
          });
          return;
        }

        // 스크립트 동적 로드
        console.log('📥 카카오맵 스크립트 동적 로드 시작');
        const script = document.createElement('script');
        script.src = 'https://dapi.kakao.com/v2/maps/sdk.js?appkey=a5e26726ce3b9dd59609c4494e21adec&libraries=services';
        script.async = true;
        script.onload = () => {
          console.log('✅ 카카오맵 스크립트 로드 완료');
          // 약간의 지연 후 resolve (API 초기화 시간 확보)
          setTimeout(() => {
            if (window.kakao && window.kakao.maps && window.kakao.maps.services) {
              resolve();
            } else {
              // API 키 오류 확인
              if (window.kakao && window.kakao.maps && window.kakao.maps.load) {
                console.error('❌ 카카오맵 API 초기화 실패 - API 키 또는 도메인 등록 확인 필요');
                reject(new Error('카카오맵 API 키가 유효하지 않거나 도메인이 등록되지 않았습니다.'));
              } else {
                reject(new Error('카카오맵 API 초기화 실패'));
              }
            }
          }, 500);
        };
        script.onerror = (error) => {
          console.error('❌ 카카오맵 스크립트 로드 실패:', error);
          reject(new Error('카카오맵 스크립트를 로드할 수 없습니다. 네트워크 연결을 확인해주세요.'));
        };
        document.head.appendChild(script);
      });
    };

    // 지도 초기화 함수
    const initializeMap = () => {
      if (!mapContainerRef.current) {
        console.log('⚠️ mapContainerRef.current가 없음');
        return;
      }

      if (window.kakao && window.kakao.maps && window.kakao.maps.services) {
        try {
          console.log('✅ 카카오맵 API 확인됨, 지도 생성 시작');
          // 지도 생성
          const mapOption = {
            center: new window.kakao.maps.LatLng(37.5665, 126.9780), // 서울 중심
            level: 5,
          };
          const newMap = new window.kakao.maps.Map(mapContainerRef.current, mapOption);
          console.log('✅ 지도 생성 완료');
          setMap(newMap);

          // Places 서비스 생성
          const ps = new window.kakao.maps.services.Places();
          psRef.current = ps;
          console.log('✅ Places 서비스 생성 완료');

          // 지도 위에 검색 컨트롤 추가
          const mapTypeControl = new window.kakao.maps.MapTypeControl();
          newMap.addControl(mapTypeControl, window.kakao.maps.ControlPosition.TOPRIGHT);

          // 줌 컨트롤 추가
          const zoomControl = new window.kakao.maps.ZoomControl();
          newMap.addControl(zoomControl, window.kakao.maps.ControlPosition.RIGHT);

          console.log('✅ 카카오맵 초기화 완료');
          setError(null);
        } catch (error) {
          console.error('❌ 카카오맵 초기화 오류:', error);
          const errorMsg = error.message || '알 수 없는 오류';
          
          // API 키 오류인 경우
          if (errorMsg.includes('Invalid') || errorMsg.includes('key') || errorMsg.includes('unauthorized')) {
            setError(
              '카카오맵 API 키 오류입니다.\n' +
              '카카오 개발자 콘솔에서 API 키와 도메인 설정을 확인해주세요.'
            );
          } else {
            setError('지도를 불러오는 중 오류가 발생했습니다: ' + errorMsg);
          }
        }
      } else {
        console.log('❌ 카카오맵 API를 찾을 수 없음');
        setError(
          '카카오맵 API를 불러올 수 없습니다.\n' +
          '카카오 개발자 콘솔에서 API 키와 도메인 설정을 확인해주세요.'
        );
      }
    };

    // 스크립트 로드 후 지도 초기화
    const init = async () => {
      try {
        await loadKakaoMapScript();
        // 다이얼로그가 완전히 렌더링된 후 지도 초기화
        setTimeout(initializeMap, 100);
      } catch (error) {
        console.error('❌ 카카오맵 스크립트 로드 실패:', error);
        const errorMessage = error.message || '알 수 없는 오류';
        
        // API 키 관련 오류인 경우 더 자세한 안내
        if (errorMessage.includes('API 키') || errorMessage.includes('도메인')) {
          setError(
            '카카오맵 API 키 설정이 필요합니다.\n' +
            '1. 카카오 개발자 콘솔(https://developers.kakao.com)에서 앱 키 확인\n' +
            '2. 플랫폼 설정에서 현재 도메인(localhost 등) 등록\n' +
            '3. JavaScript 키를 사용하여 API 호출'
          );
        } else {
          setError(`카카오맵을 불러올 수 없습니다: ${errorMessage}\n페이지를 새로고침해주세요.`);
        }
      }
    };

    init();
  }, [open]);

  // 기존 마커 제거
  const removeMarkers = () => {
    markers.forEach(marker => marker.setMap(null));
    infoWindows.forEach(infoWindow => infoWindow.close());
    setMarkers([]);
    setInfoWindows([]);
  };

  // 병원 검색 함수
  const searchHospitals = () => {
    console.log('🔍 검색 함수 호출됨');
    console.log('검색어:', searchKeyword);
    console.log('psRef.current:', psRef.current);
    console.log('map:', map);
    console.log('window.kakao:', window.kakao);
    
    if (!searchKeyword.trim()) {
      console.log('❌ 검색어가 비어있음');
      setError('검색어를 입력해주세요.');
      return;
    }

    if (!psRef.current || !map) {
      console.log('❌ 지도 또는 Places 서비스가 준비되지 않음');
      console.log('psRef.current:', psRef.current);
      console.log('map:', map);
      setError('지도가 아직 준비되지 않았습니다. 잠시 후 다시 시도해주세요.');
      return;
    }

    console.log('✅ 검색 시작');
    setLoading(true);
    setError(null);
    removeMarkers();

    // 검색 키워드에 "병원" 추가 (더 정확한 검색을 위해)
    const keyword = searchKeyword.trim().includes('병원') 
      ? searchKeyword.trim() 
      : `${searchKeyword.trim()} 병원`;

    // 키워드로 장소 검색
    console.log('검색 시작:', keyword);
    
    psRef.current.keywordSearch(keyword, (data, status, pagination) => {
      setLoading(false);
      console.log('검색 결과:', { status, dataLength: data?.length, data });

      if (status === window.kakao.maps.services.Status.OK) {
        // 병원만 필터링 (카테고리 코드: HP8 - 병원)
        const hospitalData = data.filter(
          place => {
            const isHospital = place.category_group_code === 'HP8' || 
                             (place.category_name && place.category_name.includes('병원')) ||
                             (place.place_name && place.place_name.includes('병원'));
            return isHospital;
          }
        );

        console.log('필터링된 병원 데이터:', hospitalData.length, hospitalData);

        if (hospitalData.length === 0) {
          // 병원 필터링 결과가 없으면 전체 결과 중 병원 관련 항목만 표시
          const allHospitalData = data.filter(
            place => place.place_name && (
              place.place_name.includes('병원') ||
              place.place_name.includes('의원') ||
              place.place_name.includes('클리닉') ||
              place.category_name?.includes('병원') ||
              place.category_name?.includes('의원')
            )
          );
          
          if (allHospitalData.length === 0) {
            setError('검색 결과가 없습니다. 다른 지역명으로 검색해보세요.');
            setHospitals([]);
            return;
          }
          
          setHospitals(allHospitalData);
          displayHospitalsOnMap(allHospitalData);
        } else {
          setHospitals(hospitalData);
          displayHospitalsOnMap(hospitalData);
        }
      } else if (status === window.kakao.maps.services.Status.ZERO_RESULT) {
        setError('검색 결과가 없습니다. 다른 지역명으로 검색해보세요.');
        setHospitals([]);
      } else if (status === window.kakao.maps.services.Status.ERROR) {
        setError('검색 중 오류가 발생했습니다. 다시 시도해주세요.');
        setHospitals([]);
      }
    });
  };

  // 지도에 병원 표시 함수
  const displayHospitalsOnMap = (hospitalData) => {
    if (!map || !hospitalData || hospitalData.length === 0) return;

    // 지도 중심 이동
    const bounds = new window.kakao.maps.LatLngBounds();
    const newMarkers = [];
    const newInfoWindows = [];

    hospitalData.forEach((place, index) => {
      const position = new window.kakao.maps.LatLng(place.y, place.x);
      bounds.extend(position);

      // 마커 생성
      const marker = new window.kakao.maps.Marker({
        position: position,
        map: map,
      });

      // 인포윈도우 생성
      const infoWindow = new window.kakao.maps.InfoWindow({
        content: `
          <div style="padding:10px;min-width:150px;">
            <div style="font-weight:bold;font-size:14px;margin-bottom:5px;">${place.place_name}</div>
            <div style="font-size:12px;color:#666;margin-bottom:3px;">${place.road_address_name || place.address_name}</div>
            ${place.phone ? `<div style="font-size:12px;color:#666;">${place.phone}</div>` : ''}
          </div>
        `,
      });

      // 마커 클릭 이벤트
      window.kakao.maps.event.addListener(marker, 'click', () => {
        // 다른 인포윈도우 닫기
        newInfoWindows.forEach(iw => iw.close());
        infoWindow.open(map, marker);
      });

      newMarkers.push(marker);
      newInfoWindows.push(infoWindow);
    });

    setMarkers(newMarkers);
    setInfoWindows(newInfoWindows);

    // 지도 범위 조정
    map.setBounds(bounds);
  };

  // 엔터 키로 검색
  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      searchHospitals();
    }
  };

  // 다이얼로그 닫기 시 초기화
  const handleClose = () => {
    removeMarkers();
    setSearchKeyword('');
    setHospitals([]);
    setError(null);
    setLoading(false);
    onClose();
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          backgroundColor: 'rgba(255, 255, 255, 0.98)',
          borderRadius: 3,
          border: '1px solid rgba(203, 213, 225, 0.5)',
          overflow: 'hidden',
        },
      }}
    >
      <DialogTitle
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          pr: 2.5,
          pl: 3,
          py: 2,
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          borderBottom: '1px solid rgba(203, 213, 225, 0.4)',
        }}
      >
        <Typography variant="subtitle1" sx={{ fontWeight: 700, color: '#1f2937' }}>
          근처 병원찾기
        </Typography>
        <IconButton onClick={handleClose} sx={{ color: '#6b7280' }}>
          <FiX size={18} />
        </IconButton>
      </DialogTitle>
      <DialogContent sx={{ backgroundColor: 'rgba(248, 250, 252, 0.95)', p: 3 }}>
        <Stack spacing={3}>
          {/* 검색 박스 */}
          <Box
            sx={{
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              borderRadius: 3,
              border: '1px solid rgba(203, 213, 225, 0.8)',
              p: 3,
            }}
          >
            <Typography variant="body2" sx={{ color: '#1f2937', fontWeight: 600, mb: 2 }}>
              지역을 입력하여 병원을 검색하세요
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <TextField
                fullWidth
                placeholder="지역명을 입력하세요 (예: 강남구, 서초구, 서울시 강남구)"
                value={searchKeyword}
                onChange={(e) => setSearchKeyword(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={loading}
                sx={{
                  '& .MuiOutlinedInput-root': {
                    backgroundColor: 'rgba(248, 250, 252, 0.9)',
                    '&:hover fieldset': {
                      borderColor: '#38bdf8',
                    },
                    '&.Mui-focused fieldset': {
                      borderColor: '#38bdf8',
                    },
                  },
                }}
              />
              <Button
                variant="contained"
                onClick={() => {
                  console.log('🔘 검색 버튼 클릭됨');
                  searchHospitals();
                }}
                disabled={loading || !searchKeyword.trim()}
                sx={{
                  px: 3,
                  py: 1.5,
                  backgroundColor: '#38bdf8',
                  color: 'white',
                  fontWeight: 600,
                  fontSize: '14px',
                  '&:hover': {
                    backgroundColor: '#0ea5e9',
                  },
                  '&:disabled': {
                    backgroundColor: 'rgba(148, 163, 184, 0.4)',
                  },
                  minWidth: 100,
                }}
              >
                {loading ? <CircularProgress size={20} color="inherit" /> : '검색'}
              </Button>
            </Box>
            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}
            {hospitals.length > 0 && (
              <Alert severity="success" sx={{ mt: 2 }}>
                {hospitals.length}개의 병원을 찾았습니다.
              </Alert>
            )}
          </Box>

          {/* 카카오맵 */}
          <Box
            sx={{
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              borderRadius: 3,
              border: '1px solid rgba(203, 213, 225, 0.8)',
              overflow: 'hidden',
            }}
          >
            <Typography 
              variant="subtitle2" 
              sx={{ 
                color: '#1f2937', 
                fontWeight: 600, 
                p: 2, 
                borderBottom: '1px solid rgba(203, 213, 225, 0.4)' 
              }}
            >
              {hospitals.length > 0 ? '검색된 병원 위치' : '병원 위치'}
            </Typography>
            
            <Box
              sx={{
                position: 'relative',
                width: '100%',
                height: '500px',
              }}
            >
              <Box
                ref={mapContainerRef}
                sx={{
                  width: '100%',
                  height: '100%',
                  backgroundColor: '#f8fafc',
                  position: 'relative',
                }}
              />
              {!map && (
                <Box
                  sx={{
                    textAlign: 'center',
                    color: '#6b7280',
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    zIndex: 1,
                  }}
                >
                  <CircularProgress sx={{ mb: 2 }} />
                  <Typography variant="body2">
                    지도를 불러오는 중입니다...
                  </Typography>
                </Box>
              )}
            </Box>
            
            <Typography 
              variant="caption" 
              sx={{ 
                display: 'block', 
                p: 2, 
                color: '#6b7280', 
                textAlign: 'center' 
              }}
            >
              {hospitals.length > 0 
                ? '마커를 클릭하면 병원 정보를 확인할 수 있습니다'
                : '지역명을 입력하고 검색 버튼을 클릭하여 병원을 찾아보세요'}
            </Typography>
          </Box>

          {/* 검색 결과 리스트 (선택사항) */}
          {hospitals.length > 0 && (
            <Box
              sx={{
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                borderRadius: 3,
                border: '1px solid rgba(203, 213, 225, 0.8)',
                p: 2,
                maxHeight: '300px',
                overflowY: 'auto',
              }}
            >
              <Typography variant="subtitle2" sx={{ color: '#1f2937', fontWeight: 600, mb: 2 }}>
                검색 결과 ({hospitals.length}개)
              </Typography>
              <Stack spacing={1}>
                {hospitals.map((hospital, index) => (
                  <Box
                    key={hospital.id || index}
                    onClick={() => {
                      if (map) {
                        const position = new window.kakao.maps.LatLng(hospital.y, hospital.x);
                        map.setCenter(position);
                        map.setLevel(3);
                        if (infoWindows[index]) {
                          infoWindows.forEach(iw => iw.close());
                          infoWindows[index].open(map, markers[index]);
                        }
                      }
                    }}
                    sx={{
                      p: 2,
                      borderRadius: 2,
                      border: '1px solid rgba(203, 213, 225, 0.4)',
                      cursor: 'pointer',
                      '&:hover': {
                        backgroundColor: 'rgba(56, 189, 248, 0.1)',
                        borderColor: '#38bdf8',
                      },
                    }}
                  >
                    <Typography variant="body2" sx={{ fontWeight: 600, color: '#1f2937', mb: 0.5 }}>
                      {hospital.place_name}
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#6b7280', display: 'block' }}>
                      {hospital.road_address_name || hospital.address_name}
                    </Typography>
                    {hospital.phone && (
                      <Typography variant="caption" sx={{ color: '#6b7280', display: 'block' }}>
                        {hospital.phone}
                      </Typography>
                    )}
                  </Box>
                ))}
              </Stack>
            </Box>
          )}
        </Stack>
      </DialogContent>
    </Dialog>
  );
};

export default HospitalSearch;

