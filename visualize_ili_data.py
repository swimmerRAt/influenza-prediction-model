"""
PostgreSQL 데이터베이스의 ILI 데이터 시각화

이 스크립트는 PostgreSQL influenza 데이터베이스에서 
인플루엔자 유사 질환(ILI) 데이터를 로드하여 다양한 그래프로 시각화합니다.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path
import sys

# database 폴더의 db_utils 임포트
sys.path.append(str(Path(__file__).parent / 'database'))
from db_utils import load_from_postgres

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

def load_ili_data():
    """PostgreSQL에서 ILI 데이터 로드"""
    print("=" * 60)
    print("📊 PostgreSQL에서 ILI 데이터 로드 중...")
    print("=" * 60)
    
    df = load_from_postgres(table_name="influenza_data")
    
    print(f"\n✅ 데이터 로드 완료!")
    print(f"   - 총 행 수: {len(df):,}")
    print(f"   - 컬럼: {list(df.columns)}")
    print(f"   - 연도 범위: {df['year'].min():.0f} ~ {df['year'].max():.0f}")
    print(f"   - 고유 연령대: {df['age_group'].nunique()}개")
    
    return df

def plot_ili_overall_trend(df, save_path="plot_ili_overall_trend.png"):
    """전체 ILI 추세 그래프 (연령대별 평균)"""
    print(f"\n📈 전체 ILI 추세 그래프 생성 중...")
    
    # 연도/주차별 평균 계산
    df_avg = df.groupby(['year', 'week'], as_index=False)['ili'].mean()
    df_avg = df_avg.sort_values(['year', 'week'])
    
    # 결측치 제거
    df_avg = df_avg.dropna(subset=['ili'])
    
    # 시계열 인덱스 생성 (연도-주차)
    df_avg['time_label'] = df_avg['year'].astype(int).astype(str) + '-W' + df_avg['week'].astype(int).astype(str).str.zfill(2)
    
    plt.figure(figsize=(16, 6))
    plt.plot(df_avg.index, df_avg['ili'], linewidth=1.5, color='#2E86AB', alpha=0.8)
    plt.fill_between(df_avg.index, df_avg['ili'], alpha=0.3, color='#2E86AB')
    
    plt.title('인플루엔자 유사질환(ILI) 발생률 추세 (전체 연령대 평균)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('시점 (연도-주차)', fontsize=12)
    plt.ylabel('ILI 발생률 (인구 1,000명당)', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # x축 레이블 설정 (일부만 표시)
    n_points = len(df_avg)
    tick_indices = np.linspace(0, n_points-1, min(20, n_points), dtype=int)
    plt.xticks(tick_indices, df_avg.iloc[tick_indices]['time_label'], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 저장 완료: {save_path}")
    plt.close()

def plot_ili_by_age_group(df, save_path="plot_ili_by_age_group.png"):
    """연령대별 ILI 추세 비교"""
    print(f"\n📈 연령대별 ILI 추세 그래프 생성 중...")
    
    # 주요 연령대 선택
    target_ages = ['0-6세', '7-12세', '13-18세', '19-49세', '50-64세', '65세이상']
    df_filtered = df[df['age_group'].isin(target_ages)].copy()
    
    # 연도/주차별로 그룹화
    df_filtered = df_filtered.sort_values(['year', 'week'])
    df_filtered['time_idx'] = df_filtered.groupby(['year', 'week']).ngroup()
    
    plt.figure(figsize=(16, 8))
    
    colors = ['#E63946', '#F77F00', '#FCBF49', '#06A77D', '#2E86AB', '#8338EC']
    
    for i, age in enumerate(target_ages):
        df_age = df_filtered[df_filtered['age_group'] == age]
        if not df_age.empty:
            # 중복 제거 (같은 연도/주차에 여러 값이 있을 경우 평균)
            df_age_agg = df_age.groupby('time_idx', as_index=False).agg({
                'ili': 'mean',
                'year': 'first',
                'week': 'first'
            })
            df_age_agg = df_age_agg.dropna(subset=['ili'])
            
            if len(df_age_agg) > 0:
                plt.plot(df_age_agg['time_idx'], df_age_agg['ili'], 
                        label=age, linewidth=2, color=colors[i % len(colors)], alpha=0.8)
    
    plt.title('연령대별 ILI 발생률 추세 비교', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('시점 (연도-주차 순서)', fontsize=12)
    plt.ylabel('ILI 발생률 (인구 1,000명당)', fontsize=12)
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 저장 완료: {save_path}")
    plt.close()

def plot_ili_by_year(df, save_path="plot_ili_by_year.png"):
    """연도별 ILI 패턴 비교 (주차 기준)"""
    print(f"\n📈 연도별 ILI 패턴 그래프 생성 중...")
    
    # 연도/주차별 평균 계산
    df_avg = df.groupby(['year', 'week'], as_index=False)['ili'].mean()
    
    plt.figure(figsize=(14, 7))
    
    years = sorted(df_avg['year'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(years)))
    
    for i, year in enumerate(years):
        df_year = df_avg[df_avg['year'] == year].sort_values('week')
        df_year = df_year.dropna(subset=['ili'])
        
        if len(df_year) > 0:
            plt.plot(df_year['week'], df_year['ili'], 
                    label=f'{int(year)}년', linewidth=2, 
                    color=colors[i], alpha=0.7, marker='o', markersize=3)
    
    plt.title('연도별 ILI 발생률 패턴 (주차별)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('주차 (Week)', fontsize=12)
    plt.ylabel('ILI 발생률 (인구 1,000명당)', fontsize=12)
    plt.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(1, 53)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 저장 완료: {save_path}")
    plt.close()

def plot_ili_heatmap(df, save_path="plot_ili_heatmap.png"):
    """연도-주차 히트맵"""
    print(f"\n📈 ILI 히트맵 생성 중...")
    
    # 연도/주차별 평균 계산
    df_avg = df.groupby(['year', 'week'], as_index=False)['ili'].mean()
    
    # 피벗 테이블 생성
    pivot = df_avg.pivot(index='week', columns='year', values='ili')
    
    plt.figure(figsize=(14, 10))
    im = plt.imshow(pivot.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    plt.colorbar(im, label='ILI 발생률')
    plt.title('연도-주차별 ILI 발생률 히트맵', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('연도', fontsize=12)
    plt.ylabel('주차 (Week)', fontsize=12)
    
    # x축 레이블
    plt.xticks(range(len(pivot.columns)), [f'{int(y)}' for y in pivot.columns], rotation=45)
    # y축 레이블 (일부만)
    y_ticks = list(range(0, len(pivot.index), 4))
    plt.yticks(y_ticks, [int(pivot.index[i]) for i in y_ticks])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 저장 완료: {save_path}")
    plt.close()

def plot_ili_statistics(df, save_path="plot_ili_statistics.png"):
    """연령대별 ILI 통계 (박스플롯)"""
    print(f"\n📈 연령대별 ILI 통계 그래프 생성 중...")
    
    # 주요 연령대 선택
    target_ages = ['0-6세', '7-12세', '13-18세', '19-49세', '50-64세', '65세이상']
    df_filtered = df[df['age_group'].isin(target_ages)].copy()
    df_filtered = df_filtered.dropna(subset=['ili'])
    
    # 연령대 순서 정렬
    df_filtered['age_group'] = pd.Categorical(df_filtered['age_group'], 
                                              categories=target_ages, 
                                              ordered=True)
    df_filtered = df_filtered.sort_values('age_group')
    
    plt.figure(figsize=(12, 7))
    
    # 박스플롯
    box_data = [df_filtered[df_filtered['age_group'] == age]['ili'].values 
                for age in target_ages]
    
    bp = plt.boxplot(box_data, labels=target_ages, patch_artist=True,
                     notch=True, showmeans=True)
    
    # 색상 설정
    colors = ['#E63946', '#F77F00', '#FCBF49', '#06A77D', '#2E86AB', '#8338EC']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('연령대별 ILI 발생률 분포', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('연령대', fontsize=12)
    plt.ylabel('ILI 발생률 (인구 1,000명당)', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 저장 완료: {save_path}")
    plt.close()

def main():
    """메인 실행 함수"""
    print("\n" + "🎨 " * 30)
    print("ILI 데이터 시각화 시작!")
    print("🎨 " * 30 + "\n")
    
    # 데이터 로드
    df = load_ili_data()
    
    # 기본 데이터 확인
    print("\n" + "=" * 60)
    print("📋 데이터 미리보기")
    print("=" * 60)
    print(df.head(10))
    
    print("\n" + "=" * 60)
    print("📊 데이터 요약 통계")
    print("=" * 60)
    print(df[['year', 'week', 'ili']].describe())
    
    # 그래프 생성
    print("\n" + "=" * 60)
    print("🎨 그래프 생성 시작")
    print("=" * 60)
    
    plot_ili_overall_trend(df)
    plot_ili_by_age_group(df)
    plot_ili_by_year(df)
    plot_ili_heatmap(df)
    plot_ili_statistics(df)
    
    print("\n" + "=" * 60)
    print("✅ 모든 그래프 생성 완료!")
    print("=" * 60)
    print("\n생성된 파일:")
    print("  1. plot_ili_overall_trend.png - 전체 ILI 추세")
    print("  2. plot_ili_by_age_group.png - 연령대별 비교")
    print("  3. plot_ili_by_year.png - 연도별 패턴")
    print("  4. plot_ili_heatmap.png - 히트맵")
    print("  5. plot_ili_statistics.png - 통계 분포")
    print()

if __name__ == "__main__":
    main()
