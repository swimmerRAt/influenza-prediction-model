import { useState, useEffect } from 'react';
import { getETLDataByDateRange, getETLDataBySeason, getETLDataByOrigin } from '../api/etlDataApi';
import { getDateRangeFromSeason } from '../utils/dateUtils';
import { processETLData } from '../utils/dataProcessors';
import { loadHistoricalCSVData, convertCSVToETLFormat } from '../utils/csvDataLoader';
import { sortWeeksBySeason } from '../utils/seasonUtils';

// 기본 더미 데이터
const defaultIliWeeks = ['37주', '38주', '39주', '40주', '41주', '42주', '43주', '44주'];
const defaultIliValues = [10.5, 12.3, 14.8, 17.2, 19.5, 15.3, 18.7, 22.8];

const defaultAriWeeks = ['34주', '35주', '36주', '37주'];
const defaultAriValues = [18, 23, 28, 34];

const defaultSariWeeks = ['34주', '35주', '36주', '37주'];
const defaultSariValues = [8, 5, 4, 3];

const defaultIrissWeeks = ['37주', '38주', '39주', '40주', '41주', '42주'];
const defaultIrissValues = [2.4, 3.1, 4.2, 5.6, 6.9, 7.8];

const defaultKrissWeeks = ['40주', '41주', '42주', '43주'];
const defaultKrissValues = [3.5, 5.1, 6.8, 9.7];

const defaultNedisWeeks = ['40주', '41주', '42주', '43주'];
const defaultNedisValues = [456, 623, 892, 1231];

const defaultInfluenzaData = {
  ili: { weeks: defaultIliWeeks, values: defaultIliValues },
  ari: { weeks: defaultAriWeeks, values: defaultAriValues },
  sari: { weeks: defaultSariWeeks, values: defaultSariValues },
  iriss: { weeks: defaultIrissWeeks, values: defaultIrissValues },
  kriss: { weeks: defaultKrissWeeks, values: defaultKrissValues },
  nedis: { weeks: defaultNedisWeeks, values: defaultNedisValues },
};

/**
 * 인플루엔자 데이터를 가져오는 커스텀 훅
 * @param {string} selectedSeason - 선택된 절기 (예: '25/26')
 * @param {string} selectedWeek - 선택된 주차 (예: '37')
 * @param {string} dsid - 데이터셋 ID (기본값: 'ds_0101')
 * @returns {Object} {influenzaData, loading, error}
 */
export const useInfluenzaData = (selectedSeason, selectedWeek, dsid = 'ds_0101') => {
  const [influenzaData, setInfluenzaData] = useState(defaultInfluenzaData);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    console.log('🔄 [useInfluenzaData] useEffect 실행 - 절기:', selectedSeason, '주차:', selectedWeek);
    
    const fetchInfluenzaData = async () => {
      console.log(`🚀 [${selectedSeason}절기] 데이터 로드 시작`);
      
      setLoading(true);
      setError(null);

      try {
        // 25/26절기인지 확인 (최신 절기는 API에서만 가져옴)
        const isLatestSeason = selectedSeason === '25/26';
        
        // 1. CSV 데이터 로드 (25/26절기가 아닌 경우만)
        let csvETLData = [];
        
        if (!isLatestSeason) {
          // 25/26절기가 아닌 경우에만 CSV 데이터 로드
        const csvData = await loadHistoricalCSVData(dsid);
          csvETLData = convertCSVToETLFormat(csvData);
          
          // 해당 절기의 데이터만 필터링
          const [year1, year2] = selectedSeason.split('/').map(y => parseInt('20' + y));
          csvETLData = csvETLData.filter(item => {
              try {
                const parsedData = JSON.parse(item.parsedData || '[]');
                if (Array.isArray(parsedData) && parsedData.length > 0) {
                  const firstRow = parsedData[0];
                  const year = parseInt(firstRow['연도'] || firstRow['﻿연도'] || '0');
                  const week = parseInt(firstRow['주차'] || '0');
                
                // 절기 범위: XX년 36주 ~ YY년 35주
                if (year === year1 && week >= 36) return true;
                if (year === year2 && week <= 35) return true;
                return false;
                }
              } catch (e) {
              return false;
              }
            return false;
            });
          console.log(`📂 [${selectedSeason}절기] CSV 데이터 필터링 완료: ${csvETLData.length}건`);
        } else {
          // 25/26절기는 CSV 데이터 사용 안 함
          console.log(`📂 [${selectedSeason}절기] CSV 데이터 사용 안 함 (API만 사용)`);
        }
        
        // 2. API 데이터 가져오기 (25/26절기만)
        let apiRawData = [];
        
        if (isLatestSeason) {
          // 25/26절기만 origin별로 API 요청
          try {
            console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
            console.log(`📡 [${selectedSeason}절기] origin별 API 요청 시작`);
            console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
            
            // 먼저 날짜 범위로 origin 목록 가져오기
            const dateRange = getDateRangeFromSeason(selectedSeason, selectedWeek);
            console.log(`📅 [${selectedSeason}절기] 날짜 범위 API 요청:`, {
              dsid,
              from: '2025-09-01',
              to: dateRange.to,
            });
            
            const tempApiData = await getETLDataByDateRange(dsid, '2025-09-01', dateRange.to);
            const tempApiRawData = tempApiData?.body?.data || tempApiData?.data || tempApiData;
            
            console.log(`📦 [${selectedSeason}절기] 날짜 범위 API 응답:`, {
              전체응답: tempApiData,
              데이터개수: Array.isArray(tempApiRawData) ? tempApiRawData.length : 'N/A',
              샘플데이터: Array.isArray(tempApiRawData) && tempApiRawData.length > 0 ? tempApiRawData[0] : null,
            });
            
            // origin 목록 추출 (중복 제거)
            const origins = [];
            if (Array.isArray(tempApiRawData)) {
              tempApiRawData.forEach(item => {
                if (item.origin && !origins.includes(item.origin)) {
                  origins.push(item.origin);
                }
              });
            }
            
            console.log(`📋 [${selectedSeason}절기] 발견된 origin 목록:`, origins);
            console.log(`📋 [${selectedSeason}절기] origin 개수:`, origins.length);
            
            // 각 origin별로 요청
            for (let i = 0; i < origins.length; i++) {
              const origin = origins[i];
              try {
                console.log(`🔵 [${selectedSeason}절기] origin ${i + 1}/${origins.length} 요청:`, origin);
                
                const originData = await getETLDataByOrigin(dsid, origin);
                const originRawData = originData?.body?.data || originData?.data || originData;
                
                console.log(`✅ [${selectedSeason}절기] origin ${i + 1}/${origins.length} 응답:`, {
                  origin,
                  전체응답: originData,
                  데이터개수: Array.isArray(originRawData) ? originRawData.length : 'N/A',
                  샘플데이터: Array.isArray(originRawData) && originRawData.length > 0 ? originRawData[0] : null,
                });
                
                if (Array.isArray(originRawData)) {
                  apiRawData.push(...originRawData);
                } else if (originRawData) {
                  apiRawData.push(originRawData);
                }
              } catch (err) {
                console.error(`❌ [${selectedSeason}절기] origin ${i + 1}/${origins.length} 요청 실패:`, {
                  origin,
                  error: err.message,
                  response: err.response?.data,
                });
              }
            }
            
            console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
            console.log(`✅ [${selectedSeason}절기] origin별 API 요청 완료: 총 ${apiRawData.length}건`);
            console.log(`📊 [${selectedSeason}절기] 수집된 데이터 샘플:`, apiRawData.slice(0, 3));
            console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
        } catch (apiError) {
            console.error('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
            console.error(`❌ [${selectedSeason}절기] API 요청 실패:`, {
              error: apiError.message,
              response: apiError.response?.data,
              status: apiError.response?.status,
            });
            console.error('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
          apiRawData = [];
        }
        } else {
          // 25/26절기가 아니면 API 사용 안 함 (CSV만 사용)
          console.log(`📂 [${selectedSeason}절기] CSV 데이터만 사용 (API 사용 안 함)`);
        }
        
        // 3. 데이터 병합 (25/26절기는 API만, 나머지는 CSV만)
        let allRawData = [];
        
        if (isLatestSeason) {
          // 25/26절기는 API 데이터만 사용
          allRawData = Array.isArray(apiRawData) ? apiRawData : [];
          console.log(`📊 [${selectedSeason}절기] API 데이터만 사용: ${allRawData.length}건`);
        } else {
          // 다른 절기는 CSV 데이터만 사용
          allRawData = csvETLData;
          console.log(`📊 [${selectedSeason}절기] CSV 데이터만 사용: ${allRawData.length}건`);
        }
        
        if (allRawData && Array.isArray(allRawData)) {
          if (allRawData.length === 0) {
            // 빈 배열인 경우 - 기본 데이터 유지
          } else {
            // 데이터 처리
            const processedData = processETLData(allRawData);
            
          if (processedData && processedData.weeks && processedData.values) {
            // 주차를 절기별로 정렬 (36주부터 시작해서 다음 해 35주까지)
            const weeks = [...processedData.weeks].sort((a, b) => sortWeeksBySeason(a, b));
            
            // 모든 연령대의 평균값을 계산하여 ILI 데이터로 사용
            const allAgeGroups = Object.keys(processedData.values).filter(ageGroup => {
              const isSeason = /^\d{2}\/\d{2}$/.test(ageGroup);
              return !isSeason;
            });
            
            // 주차별로 그룹화된 데이터를 다시 매핑
            const weekValueMap = new Map();
            
            // 먼저 각 주차별로 모든 연령대의 평균값 계산
            processedData.weeks.forEach((week, index) => {
              const validValues = allAgeGroups
                .map(ageGroup => processedData.values[ageGroup]?.[index])
                .filter(val => val !== null && val !== undefined);
              
              if (validValues.length > 0) {
                const avgValue = validValues.reduce((sum, val) => sum + val, 0) / validValues.length;
                weekValueMap.set(week, avgValue);
              }
            });
            
            // 정렬된 주차 순서대로 ILI 값 매핑 (실제 데이터가 있는 주차만)
            const weekValuePairs = weeks
              .map(week => ({ week, value: weekValueMap.get(week) }))
              .filter(pair => pair.value !== null && pair.value !== undefined);
            
            const finalWeeks = weekValuePairs.map(pair => pair.week);
            const iliValues = weekValuePairs.map(pair => pair.value);
            
            // 연령대별 데이터 저장 (실제 데이터가 있는 주차만)
            const ageGroupData = {};
            allAgeGroups.forEach((ageGroup) => {
              const weekValueMapForAge = new Map();
              processedData.weeks.forEach((week, index) => {
                const value = processedData.values[ageGroup]?.[index];
                if (value !== null && value !== undefined) {
                  weekValueMapForAge.set(week, value);
                }
              });
              
              // 실제 데이터가 있는 주차만 필터링
              const ageWeekValuePairs = finalWeeks
                .map(week => ({ week, value: weekValueMapForAge.get(week) }))
                .filter(pair => pair.value !== null && pair.value !== undefined);
              
              ageGroupData[ageGroup] = {
                weeks: ageWeekValuePairs.map(pair => pair.week),
                values: ageWeekValuePairs.map(pair => pair.value),
              };
            });
            
            // 절기별 데이터 저장
            const seasonData = processedData.seasons || {};
            
            console.log(`✅ [${selectedSeason}절기] 데이터 처리 완료:`, {
              주차수: finalWeeks.length,
              주차목록: finalWeeks,
              값목록: iliValues,
              주차값쌍: finalWeeks.map((w, i) => ({ week: w, value: iliValues[i] })),
              절기수: Object.keys(seasonData).length,
            });
            
            setInfluenzaData({
              ili: { 
                weeks: finalWeeks, 
                values: iliValues, 
                ageGroups: ageGroupData, // 연령대별 데이터 추가
                seasons: seasonData, // 절기별 데이터 추가
              },
              // 다른 지표들은 기본값 유지 (추후 다른 DSID로 데이터 가져올 수 있음)
              ari: defaultInfluenzaData.ari,
              sari: defaultInfluenzaData.sari,
              iriss: defaultInfluenzaData.iriss,
              kriss: defaultInfluenzaData.kriss,
              nedis: defaultInfluenzaData.nedis,
            });
          }
          }
        }
      } catch (err) {
        // API 호출 실패 시 에러 로그
        console.error(`❌ [${selectedSeason}절기] 데이터 로드 실패:`, err.message);
        
        let errorMessage = '데이터를 불러오는데 실패했습니다. 기본 데이터를 표시합니다.';
        
        if (err.response) {
          // 서버 응답이 있는 경우
          if (err.response.status === 401) {
            errorMessage = '인증에 실패했습니다. 환경 변수를 확인하세요.';
          } else if (err.response.status === 404) {
            errorMessage = 'API 엔드포인트를 찾을 수 없습니다.';
          } else {
            errorMessage = `서버 오류 (${err.response.status}): ${err.response.data?.message || err.message || '알 수 없는 오류'}`;
          }
        } else if (err.request) {
          // 요청은 보냈지만 응답이 없는 경우 (CORS 등)
          if (err.message && (err.message.includes('CORS') || err.message.includes('Network Error'))) {
            errorMessage = 'CORS 오류: 개발 서버를 재시작하거나 백엔드에서 CORS 설정이 필요합니다. 기본 데이터를 표시합니다.';
          } else {
            errorMessage = '서버에 연결할 수 없습니다. 네트워크 연결을 확인하세요.';
          }
        } else if (err.message) {
          // 기타 에러
          if (err.message.includes('인증 설정')) {
            errorMessage = '인증 설정이 완료되지 않았습니다. .env 파일을 확인하세요.';
          } else {
            errorMessage = err.message;
          }
        }
        
        setError(errorMessage);
        // 기본값은 이미 useState 초기값으로 설정되어 있음
      } finally {
        setLoading(false);
      }
    };

    fetchInfluenzaData();
  }, [selectedSeason, selectedWeek, dsid]);

  return { influenzaData, loading, error };
};

