import { getSeasonFromWeek, extractYearFromWeek, sortWeeksBySeason } from './seasonUtils';

/**
 * ETL API 응답 데이터를 대시보드에서 사용할 형식으로 변환하는 함수
 * @param {Array} rawData - API에서 받은 원시 데이터 배열
 * @returns {Object|null} 처리된 데이터 또는 null
 */
export const processETLData = (rawData) => {
  try {
    if (!Array.isArray(rawData) || rawData.length === 0) {
      return null;
    }

    // 각 항목의 parsedData를 파싱하여 주차별로 그룹화
    const weekDataMap = new Map(); // 주차별 데이터 저장

    rawData.forEach((item, itemIndex) => {
      try {
        // parsedData는 JSON 문자열이므로 파싱 필요
        const parsedData = JSON.parse(item.parsedData || '[]');
        
        if (!Array.isArray(parsedData) || parsedData.length === 0) {
          return;
        }

        parsedData.forEach((row, rowIndex) => {
          // 주차 정보 추출 (다양한 형식 지원)
          let week = null;
          let year = null;
          
          // 형식 1: "수집 기간" 필드에 "2025년 32주" 형식
          const weekKey = row['수집 기간'] || row['﻿수집 기간'];
          if (weekKey) {
            const weekMatch = weekKey.match(/(\d+)주/);
            if (weekMatch) {
              week = weekMatch[1] + '주';
            }
            const yearMatch = weekKey.match(/(\d{4})년/);
            if (yearMatch) {
              year = yearMatch[1];
            }
          }
          
          // 형식 2: "주차" 필드에 직접 숫자 (예: "35", "43")
          if (!week && row['주차']) {
            const weekNum = parseInt(row['주차']);
            if (!isNaN(weekNum)) {
              week = weekNum + '주';
            }
          }
          
          // 연도 정보 추출 (주차 키에서 추출 실패 시 row에서 추출)
          if (!year) {
            year = extractYearFromWeek(weekKey, row) || row['연도'] || row['﻿연도'] || '2025';
          }
          
          if (!week) {
            return;
          }
          
          // 연도+주차 조합으로 키 생성 (같은 주차의 다른 연도 데이터를 구분하기 위해)
          const weekYearKey = `${year}_${week}`;
          
          // 주차별 데이터 초기화
          if (!weekDataMap.has(weekYearKey)) {
            weekDataMap.set(weekYearKey, {
              week,
              year,
              values: {},
              seasons: {},
            });
          }

          const weekData = weekDataMap.get(weekYearKey);
          
          // 연령대별 값 추출
          // 데이터 구조에 따라 두 가지 형식 처리:
          // 형식 1: 키가 연령대인 경우: { "65세 이상": "2.7", "0세": "3.4", ... } - "연령대" 필드 없음
          // 형식 2: 값 필드 + 연령대 필드: { "연령대": "65세이상", "의사환자 분율": "6.9", ... } - "연령대" 필드 있음
          
          // 형식 판별: "연령대" 필드가 있으면 형식 2, 없으면 형식 1
          const hasAgeGroupField = row.hasOwnProperty('연령대');
          
          if (hasAgeGroupField) {
            // 형식 2 처리: 값 필드 + 연령대 필드 조합
            let ageGroup = row['연령대'];
            // 절기 정보인지 확인 (예: "24/25절기", "17/18절기" 등)
            const isSeason = /^\d{2}\/\d{2}절기$/.test(ageGroup);
            
            // 연령대 키 정규화: "65세 이상" -> "65세이상"으로 통일 (절기가 아닌 경우만)
            if (!isSeason) {
              ageGroup = ageGroup.replace(/\s+/g, '');
            }
            
            // 값 필드 찾기 (의사환자 분율 우선, 입원환자 수는 제외)
            // ds_0101은 "의사환자 분율" 데이터이므로 "의사환자 분율" 필드만 사용
            const valueFieldsFound = [];
            let valueToUse = null;
            let valueFieldName = null;
            
            // 우선순위: "의사환자 분율" > 기타 숫자 필드
            // 절기별 데이터의 경우 "입원환자 수"도 허용 (절기별 데이터는 "입원환자 수" 필드에 있을 수 있음)
            const preferredField = '의사환자 분율';
            const excludedFields = isSeason 
              ? ['수집 기간', '주차', '연도', '﻿연도', '﻿수집 기간', '연령대'] 
              : ['입원환자 수', '수집 기간', '주차', '연도', '﻿연도', '﻿수집 기간', '연령대'];
            
            // 우선적으로 "의사환자 분율" 필드 찾기
            if (row[preferredField] !== undefined) {
              const value = parseFloat(row[preferredField]);
              if (!isNaN(value)) {
                valueToUse = value;
                valueFieldName = preferredField;
                valueFieldsFound.push({ key: preferredField, value, used: true });
              }
            }
            
            // "의사환자 분율"이 없으면 다른 숫자 필드 찾기 (단, 제외 필드는 제외)
            if (valueToUse === null) {
              Object.keys(row).forEach((key) => {
                // 제외 필드 확인 (정확한 매칭 + "연도" 포함 여부)
                if (excludedFields.includes(key) || key.includes('연도') || key.includes('주차') || key.includes('수집 기간')) {
                  return;
                }
                
                const value = parseFloat(row[key]);
                // 연도 범위가 아닌 값만 사용 (의사환자 분율은 보통 0-200 범위)
                if (!isNaN(value) && value >= 0 && value <= 1000) {
                  valueFieldsFound.push({ key, value, used: false });
                  if (valueToUse === null) {
                    valueToUse = value;
                    valueFieldName = key;
                    valueFieldsFound[valueFieldsFound.length - 1].used = true;
                  }
                }
              });
            }
            
            // 값이 있으면 저장
            if (valueToUse !== null) {
              // 절기 정보인 경우와 연령대 정보인 경우를 구분하여 저장
              if (isSeason) {
                // 절기별 데이터로 저장
                if (!weekData.seasons) {
                  weekData.seasons = {};
                }
                if (!weekData.seasons[ageGroup]) {
                  weekData.seasons[ageGroup] = [];
                }
                weekData.seasons[ageGroup].push(valueToUse);
              } else {
                // 연령대별 데이터로 저장
                if (!weekData.values[ageGroup]) {
                  weekData.values[ageGroup] = [];
                }
                weekData.values[ageGroup].push(valueToUse);
              }
            }
          } else {
            // 형식 1 처리: 키 자체가 연령대인 경우
            const ageGroupKeysFound = [];
            Object.keys(row).forEach((key) => {
              // 메타데이터 필드는 제외
              if (key === '수집 기간' || key === '주차' || key === '연도' || key === '﻿수집 기간' || key === '연령대') {
                return;
              }
              
              // 키가 연령대인 경우 (예: "65세 이상", "0세", "1-6세" 등)
              if (key.includes('세') || key === '0세' || key === '연령미상') {
                const value = parseFloat(row[key]);
                if (!isNaN(value)) {
                  // 연령대 키 정규화: "65세 이상" -> "65세이상"으로 통일
                  const normalizedKey = key.replace(/\s+/g, '');
                  ageGroupKeysFound.push({ original: key, normalized: normalizedKey, value });
                  if (!weekData.values[normalizedKey]) {
                    weekData.values[normalizedKey] = [];
                  }
                  weekData.values[normalizedKey].push(value);
                }
              }
            });
          }
        });
      } catch (parseError) {
        // 파싱 실패 시 해당 항목만 건너뜀
      }
    });

    // 주차별로 정렬하고 대시보드 형식으로 변환
    // weekDataMap의 키는 "연도_주차" 형식이므로, 주차만 추출하여 정렬
    const allWeeks = new Set();
    weekDataMap.forEach((weekData, weekYearKey) => {
      allWeeks.add(weekData.week);
    });
    
    // 절기별 주차 정렬 사용 (36주부터 시작해서 다음 해 35주까지)
    const sortedWeeks = Array.from(allWeeks).sort((a, b) => sortWeeksBySeason(a, b));

    if (sortedWeeks.length === 0) {
      return null;
    }

    // 대시보드에서 사용하는 형식으로 변환
    // 예: { ili: { weeks: [...], values: [...] } }
    // 실제 데이터 구조에 맞게 조정 필요
    const result = {
      weeks: sortedWeeks,
      values: {},
    };

    // 각 연령대별로 주차별 값 배열 생성
    // 모든 주차에 대해 동일한 연령대가 있는지 확인하고, 없는 주차는 null로 채움
    const allAgeGroups = new Set();
    weekDataMap.forEach((weekData) => {
      Object.keys(weekData.values || {}).forEach((ageGroup) => {
        allAgeGroups.add(ageGroup);
      });
    });
    
    // 실제 데이터가 있는 주차만 추출
    const weeksWithData = new Set();
    weekDataMap.forEach((weekData) => {
      // 해당 주차에 실제 값이 있는지 확인
      const hasData = Object.keys(weekData.values || {}).some(ageGroup => {
        const values = weekData.values[ageGroup];
        return values && values.length > 0 && values.some(v => v !== null && v !== undefined);
      });
      if (hasData) {
        weeksWithData.add(weekData.week);
      }
    });
    
    // 실제 데이터가 있는 주차만 정렬
    const finalWeeks = Array.from(weeksWithData).sort((a, b) => sortWeeksBySeason(a, b));

    // 각 연령대별로 주차별 값 배열 생성 (실제 데이터가 있는 주차만)
    allAgeGroups.forEach((ageGroup) => {
      result.values[ageGroup] = [];
      finalWeeks.forEach((week) => {
        // 같은 주차의 모든 연도 데이터를 수집
        const weekValues = [];
        weekDataMap.forEach((weekData, weekYearKey) => {
          if (weekData.week === week && weekData.values && weekData.values[ageGroup] && weekData.values[ageGroup].length > 0) {
            // 각 연도별로 평균값 계산
            const avgValue = weekData.values[ageGroup].reduce((sum, val) => sum + val, 0) / weekData.values[ageGroup].length;
            weekValues.push(avgValue);
          }
        });
        
        if (weekValues.length > 0) {
          // 모든 연도의 평균값을 다시 평균내어 해당 주차의 전체 평균 계산
          const overallAvg = weekValues.reduce((sum, val) => sum + val, 0) / weekValues.length;
          result.values[ageGroup].push(overallAvg);
        } else {
          // 해당 주차에 연령대 데이터가 없으면 null 추가
          result.values[ageGroup].push(null);
        }
      });
    });
    
    // 실제 데이터가 있는 주차만 사용
    result.weeks = finalWeeks;

    // 절기별 데이터 처리
    // 방법 1: 연령대 필드에 "24/25절기"가 있는 데이터 (기존 방식)
    const allSeasonsFromField = new Set();
    weekDataMap.forEach((weekData) => {
      if (weekData.seasons) {
        Object.keys(weekData.seasons).forEach((season) => {
          allSeasonsFromField.add(season);
        });
      }
    });

    // 방법 2: 주차 기반으로 절기 계산 (모든 주차 데이터를 절기별로 그룹화)
    const allSeasonsFromWeeks = new Set();
    const weekToSeasonMap = new Map(); // 주차 -> 절기 매핑
    
    // rawData를 다시 순회하여 각 주차의 연도 정보와 함께 절기 계산
    rawData.forEach((item, itemIndex) => {
      try {
        const parsedData = JSON.parse(item.parsedData || '[]');
        if (!Array.isArray(parsedData)) return;
        
        parsedData.forEach((row, rowIndex) => {
          const weekKey = row['수집 기간'] || row['﻿수집 기간'] || row['주차'];
          if (!weekKey) return;
          
          // 주차 추출
          let week = null;
          const weekMatch = weekKey.toString().match(/(\d+)주/);
          if (weekMatch) {
            week = weekMatch[1] + '주';
          } else if (row['주차']) {
            const weekNum = parseInt(row['주차']);
            if (!isNaN(weekNum)) {
              week = weekNum + '주';
            }
          }
          
          if (!week) return;
          
          // 연도 추출
          const year = extractYearFromWeek(weekKey, row) || row['연도'] || row['﻿연도'] || '2025';
          
          // 절기 계산
          const season = getSeasonFromWeek(week, year);
          if (season) {
            allSeasonsFromWeeks.add(season);
            // 같은 주차에 여러 연도 데이터가 있을 수 있으므로, 연도+주차 조합으로 키 생성
            const weekYearKey = `${year}_${week}`;
            if (!weekToSeasonMap.has(weekYearKey)) {
              weekToSeasonMap.set(weekYearKey, season);
            }
            // 주차만으로도 매핑 저장 (호환성) - 같은 주차에 여러 연도가 있으면 마지막 것으로 덮어씀
            weekToSeasonMap.set(week, season);
          }
        });
      } catch (error) {
        // 절기 계산 중 오류 발생 시 해당 항목만 건너뜀
      }
    });

    // 모든 절기 통합
    const allSeasons = new Set([...allSeasonsFromField, ...allSeasonsFromWeeks]);

    if (allSeasons.size > 0) {
      result.seasons = {};
      allSeasons.forEach((season) => {
        result.seasons[season] = {
          weeks: [],
          values: [],
        };
        
        // 방법 1: 연령대 필드에 절기 정보가 있는 경우
        weekDataMap.forEach((weekData, weekYearKey) => {
          const week = weekData.week;
          if (weekData.seasons && weekData.seasons[season] && weekData.seasons[season].length > 0) {
            const avgValue = weekData.seasons[season].reduce((sum, val) => sum + val, 0) / weekData.seasons[season].length;
            if (!result.seasons[season].weeks.includes(week)) {
              result.seasons[season].weeks.push(week);
              result.seasons[season].values.push(avgValue);
            }
          }
        });
        
        // 방법 2: 주차 기반으로 절기에 속하는 모든 주차 데이터 추가
        // weekToSeasonMap에서 해당 절기에 속하는 모든 연도+주차 조합 찾기
        const weekYearKeysForSeason = [];
        weekToSeasonMap.forEach((mappedSeason, weekKey) => {
          if (mappedSeason === season) {
            // weekKey가 "연도_주차" 형식이어야 함
            if (weekKey.includes('_')) {
              if (!weekYearKeysForSeason.includes(weekKey)) {
                weekYearKeysForSeason.push(weekKey);
              }
            }
          }
        });
          
          // weekDataMap에 있지만 weekYearKeysForSeason에 없는 경우 추가
        weekDataMap.forEach((weekData, weekYearKey) => {
            if (!weekYearKeysForSeason.includes(weekYearKey)) {
              const mappedSeason = weekToSeasonMap.get(weekYearKey);
            if (mappedSeason === season) {
                weekYearKeysForSeason.push(weekYearKey);
            }
          }
        });
        
        weekYearKeysForSeason.forEach((weekYearKey) => {
          // weekYearKey에서 주차 추출 (예: "2017_36주" -> "36주")
          const week = weekYearKey.includes('_') ? weekYearKey.split('_').slice(1).join('_') : weekYearKey;
          
          // 이미 추가된 주차는 제외 (같은 주차가 여러 연도에 있을 수 있으므로 중복 체크)
          if (result.seasons[season].weeks.includes(week)) {
            return;
          }
          
          // 해당 연도+주차의 모든 연령대 데이터 평균 계산
          const weekData = weekDataMap.get(weekYearKey);
          
          if (weekData && weekData.values) {
            const allAgeGroupValues = [];
            const ageGroupDetails = [];
            
            Object.keys(weekData.values).forEach((ageGroup) => {
              // 절기 형식이 아닌 연령대만 포함
              const isSeasonFormat = /^\d{2}\/\d{2}절기$/.test(ageGroup);
              if (!isSeasonFormat && weekData.values[ageGroup].length > 0) {
                // 각 연령대별로 평균 계산 (같은 주차에 여러 행이 있을 경우를 대비)
                const avgValue = weekData.values[ageGroup].reduce((sum, val) => sum + val, 0) / weekData.values[ageGroup].length;
                allAgeGroupValues.push(avgValue);
                ageGroupDetails.push({ ageGroup, avgValue, rawValues: weekData.values[ageGroup] });
              }
            });
            
            if (allAgeGroupValues.length > 0) {
              // 모든 연령대의 평균값을 다시 평균내어 해당 주차의 전체 평균 계산
              const overallAvg = allAgeGroupValues.reduce((sum, val) => sum + val, 0) / allAgeGroupValues.length;
              
              result.seasons[season].weeks.push(week);
              result.seasons[season].values.push(overallAvg);
            }
          }
        });
        
        // 주차 순서대로 정렬 (절기 기준: 36주부터 시작해서 다음 해 35주까지)
        const weekValuePairs = result.seasons[season].weeks.map((week, index) => ({
          week,
          value: result.seasons[season].values[index],
        })).sort((a, b) => sortWeeksBySeason(a.week, b.week));
        
        result.seasons[season].weeks = weekValuePairs.map(pair => pair.week);
        result.seasons[season].values = weekValuePairs.map(pair => pair.value);
      });
    }

    return result;
  } catch (error) {
    console.error('데이터 처리 실패:', error);
    return null;
  }
};

