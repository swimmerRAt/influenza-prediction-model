/**
 * 주차가 속한 절기를 계산하는 유틸리티 함수
 * @param {string} week - 주차 (예: "32주", "35주")
 * @param {string} year - 연도 (예: "2025")
 * @returns {string|null} 절기 (예: "24/25절기") 또는 null
 */
export const getSeasonFromWeek = (week, year) => {
  try {
    // 주차에서 숫자 추출
    const weekNum = parseInt(week.toString().replace(/주/g, '').trim());
    if (isNaN(weekNum) || weekNum < 1 || weekNum > 53) {
      return null;
    }
    
    // 연도 파싱
    const yearNum = parseInt(year);
    if (isNaN(yearNum)) {
      return null;
    }
    
    // 인플루엔자 절기는 9월부터 시작 (36주차부터 시작)
    // 절기 정의: XX/YY절기 = XX년 36주 ~ YY년 35주
    // 예: 25/26절기 = 2025년 36주 ~ 2026년 35주
    // 절기 계산 규칙:
    // - 36주차 이상: 현재 연도/다음 연도 절기 (예: 2025년 36주 -> 25/26절기)
    // - 35주차 이하: 이전 연도/현재 연도 절기 (예: 2026년 35주 -> 25/26절기)
    if (weekNum >= 36) {
      // 36주차 이상은 현재 연도/다음 연도 절기
      // 예: 2024년 36주차 이상 -> 24/25절기
      // 예: 2025년 36주차 이상 -> 25/26절기 (실제 데이터 기준으로 25/26절기에 포함)
      const seasonYear1 = yearNum % 100;
      const seasonYear2 = (yearNum % 100) + 1;
      return `${seasonYear1}/${seasonYear2}절기`;
    } else {
      // 35주차 이하는 이전 연도/현재 연도 절기
      // 예: 2025년 35주차 이하 -> 24/25절기
      const seasonYear1 = (yearNum % 100) - 1;
      const seasonYear2 = yearNum % 100;
      return `${seasonYear1}/${seasonYear2}절기`;
    }
  } catch (error) {
    console.warn('절기 계산 실패:', error, { week, year });
    return null;
  }
};

/**
 * 주차 문자열에서 연도 추출
 * @param {string} weekKey - 주차 키 (예: "2025년 32주", "32주")
 * @param {Object} row - 데이터 행 (연도 필드가 있을 수 있음)
 * @returns {string|null} 연도 (예: "2025") 또는 null
 */
export const extractYearFromWeek = (weekKey, row) => {
  try {
    // "2025년 32주" 형식에서 연도 추출
    const yearMatch = weekKey?.toString().match(/(\d{4})년/);
    if (yearMatch) {
      return yearMatch[1];
    }
    
    // row에서 연도 필드 확인
    if (row && row['연도']) {
      return row['연도'].toString();
    }
    
    // row에서 "﻿연도" 필드 확인 (BOM 문자 포함)
    if (row && row['﻿연도']) {
      return row['﻿연도'].toString();
    }
    
    return null;
  } catch (error) {
    console.warn('연도 추출 실패:', error, { weekKey, row });
    return null;
  }
};

/**
 * 절기별 주차 정렬 함수 (36주부터 시작해서 다음 해 35주까지)
 * @param {string} a - 주차 문자열 (예: "36주", "1주")
 * @param {string} b - 주차 문자열 (예: "36주", "1주")
 * @returns {number} 정렬 결과 (-1, 0, 1)
 */
export const sortWeeksBySeason = (a, b) => {
  const weekA = parseInt(a.toString().replace(/주/g, '').trim());
  const weekB = parseInt(b.toString().replace(/주/g, '').trim());
  
  if (isNaN(weekA) || isNaN(weekB)) {
    return a.toString().localeCompare(b.toString());
  }
  
  // 36주 이상은 그대로, 35주 이하는 +53으로 처리하여 36주 이후로 배치
  // 예: 36주=36, 37주=37, ..., 52주=52, 53주=53, 1주=54, 2주=55, ..., 35주=88
  const normalizedA = weekA >= 36 ? weekA : weekA + 53;
  const normalizedB = weekB >= 36 ? weekB : weekB + 53;
  
  return normalizedA - normalizedB;
};

