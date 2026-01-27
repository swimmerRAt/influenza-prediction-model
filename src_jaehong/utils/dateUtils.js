/**
 * 절기와 주차를 날짜 범위로 변환하는 유틸리티 함수
 * @param {string} season - 절기 (예: '25/26')
 * @param {string} week - 주차 (예: '37')
 * @returns {Object} {from: string, to: string} - ISO 날짜 형식 (YYYY-MM-DD)
 */
export const getDateRangeFromSeason = (season, week) => {
  // 예: '24/25' -> 2024년 9월부터 시작 (인플루엔자 시즌)
  const [year1, year2] = season.split('/').map(y => parseInt('20' + y));
  const weekNum = parseInt(week);
  
  // 인플루엔자 시즌은 보통 9월부터 시작 (37주차)
  const seasonStart = new Date(year1, 8, 1); // 9월 1일
  const daysToAdd = (weekNum - 37) * 7; // 주차에 따른 일수 추가
  const startDate = new Date(seasonStart);
  startDate.setDate(startDate.getDate() + daysToAdd);
  
  // 1주일 범위
  const endDate = new Date(startDate);
  endDate.setDate(endDate.getDate() + 6);
  
  // API가 collectedAt 필드로 검색하는 경우를 고려하여 매우 넓은 범위로 검색
  // 데이터 수집은 주차가 지난 후에 이루어지므로 충분한 여유를 둠
  const searchStartDate = new Date(year1, 0, 1); // 해당 연도 1월 1일부터
  const searchEndDate = new Date(year2, 11, 31); // 다음 연도 12월 31일까지
  
  return {
    from: searchStartDate.toISOString().split('T')[0],
    to: searchEndDate.toISOString().split('T')[0],
  };
};

