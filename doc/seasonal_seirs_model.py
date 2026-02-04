"""
Seasonal SEIRS 모델을 이용한 인플루엔자 예측

SEIRS (Susceptible-Exposed-Infectious-Recovered-Susceptible) 모델:
- S: 감수성(Susceptible) - 감염될 수 있는 인구
- E: 잠복기(Exposed) - 감염되었지만 아직 전염력이 없는 인구
- I: 감염(Infectious) - 전염력을 가진 감염자
- R: 회복(Recovered) - 회복되어 면역을 가진 인구

계절성(Seasonality)을 고려하여 전파율(beta)이 시간에 따라 변화합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from pathlib import Path
import sys

# database 폴더의 db_utils 임포트
sys.path.append(str(Path(__file__).parent / 'database'))
from db_utils import load_from_postgres

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def seirs_model(y, t, beta0, a, sigma, gamma, omega, N):
    """
    Seasonal SEIRS 미분방정식 모델
    
    Parameters:
    -----------
    y : list
        [S, E, I, R] 각 구획의 인구수
    t : float
        시간 (주 단위)
    beta0 : float
        기본 전파율 (baseline transmission rate)
    a : float
        계절성 진폭 (seasonality amplitude, 0~1)
    sigma : float
        잠복기에서 감염기로의 전환율 (1/잠복기간)
    gamma : float
        회복률 (1/감염기간)
    omega : float
        면역 상실률 (1/면역지속기간)
    N : float
        총 인구수
    
    Returns:
    --------
    derivatives : list
        [dS/dt, dE/dt, dI/dt, dR/dt]
    """
    S, E, I, R = y
    
    # 계절성을 반영한 전파율 (52주 주기로 변동)
    beta_t = beta0 * (1 + a * np.cos(2 * np.pi * t / 52))
    
    # 미분방정식
    dSdt = omega * R - beta_t * S * I / N
    dEdt = beta_t * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I - omega * R
    
    return [dSdt, dEdt, dIdt, dRdt]


def simulate_seirs(params, t, N, initial_conditions):
    """
    SEIRS 모델 시뮬레이션
    
    Parameters:
    -----------
    params : dict
        모델 파라미터 (beta0, a, sigma, gamma, omega)
    t : array
        시간 배열 (주 단위)
    N : float
        총 인구수
    initial_conditions : list
        초기값 [S0, E0, I0, R0]
    
    Returns:
    --------
    I_per_1000 : array
        인구 1,000명당 감염자 수 (ILI에 해당)
    """
    sol = odeint(
        seirs_model, 
        initial_conditions, 
        t, 
        args=(params['beta0'], params['a'], params['sigma'], 
              params['gamma'], params['omega'], N)
    )
    
    S, E, I, R = sol.T
    I_per_1000 = I / N * 1000  # 1,000명당 감염자 수
    
    return I_per_1000


def load_real_data():
    """PostgreSQL에서 실제 ILI 데이터 로드 (팬데믹 기간 제외)"""
    print("=" * 60)
    print("📊 PostgreSQL에서 실제 ILI 데이터 로드 중...")
    print("=" * 60)
    
    df = load_from_postgres(table_name="influenza_data")
    
    # 팬데믹 기간 제외 (2020년 14주 ~ 2022년 22주)
    print("\n⚠️ 팬데믹 기간 데이터 제외 중 (2020년 14주 ~ 2022년 22주)...")
    before_count = len(df)
    
    pandemic_mask = (
        ((df['year'] == 2020) & (df['week'] >= 14)) |
        ((df['year'] == 2021)) |
        ((df['year'] == 2022) & (df['week'] <= 22))
    )
    
    df = df[~pandemic_mask].copy()
    after_count = len(df)
    removed_count = before_count - after_count
    
    print(f"   - 제거 전: {before_count:,}행")
    print(f"   - 제거 후: {after_count:,}행")
    print(f"   - 제거됨: {removed_count:,}행 ({removed_count/before_count*100:.1f}%)")
    
    # 연도/주차별 평균 계산
    df_avg = df.groupby(['year', 'week'], as_index=False)['ili'].mean()
    df_avg = df_avg.sort_values(['year', 'week'])
    df_avg = df_avg.dropna(subset=['ili'])
    
    print(f"\n✅ 데이터 로드 완료:")
    print(f"   - 데이터 포인트: {len(df_avg)}")
    print(f"   - ILI 범위: {df_avg['ili'].min():.2f} ~ {df_avg['ili'].max():.2f}")
    print(f"   - ILI 평균: {df_avg['ili'].mean():.2f}")
    
    return df_avg


def optimize_seirs_parameters(real_ili, N=1_000_000):
    """
    실제 ILI 데이터에 맞게 SEIRS 파라미터 최적화
    
    Parameters:
    -----------
    real_ili : array
        실제 ILI 데이터 (인구 1,000명당)
    N : float
        총 인구수
    
    Returns:
    --------
    best_params : dict
        최적화된 파라미터
    """
    print("\n" + "=" * 60)
    print("🔧 SEIRS 모델 파라미터 최적화 중...")
    print("=" * 60)
    
    n_weeks = len(real_ili)
    t = np.arange(n_weeks)
    
    # 손실 함수: 실제 데이터와 모델 예측의 MSE
    def loss_function(params_array):
        beta0, a, sigma, gamma, omega = params_array
        
        # 파라미터 제약 조건 체크
        if beta0 <= 0 or a < 0 or a > 1 or sigma <= 0 or gamma <= 0 or omega <= 0:
            return 1e10
        
        # 초기값 설정 (첫 ILI 값 기반)
        I0 = real_ili[0] * N / 1000
        E0 = I0 * 0.5  # 잠복기 인구는 감염자의 50%로 가정
        R0 = N * 0.1   # 초기 회복자 10%
        S0 = N - I0 - E0 - R0
        
        params = {
            'beta0': beta0,
            'a': a,
            'sigma': sigma,
            'gamma': gamma,
            'omega': omega
        }
        
        try:
            predicted_ili = simulate_seirs(params, t, N, [S0, E0, I0, R0])
            mse = np.mean((real_ili - predicted_ili) ** 2)
            return mse
        except:
            return 1e10
    
    # 초기 파라미터 추정
    # beta0: 기본 전파율 (0.3~0.7)
    # a: 계절성 진폭 (0~0.5)
    # sigma: 1/잠복기간 (잠복기 약 2일 = 0.3주 → sigma ≈ 3)
    # gamma: 1/감염기간 (감염기간 약 7일 = 1주 → gamma ≈ 1)
    # omega: 1/면역기간 (면역기간 약 180일 = 26주 → omega ≈ 0.04)
    initial_params = [0.5, 0.3, 3.0, 1.0, 0.04]
    
    # 파라미터 범위 설정
    bounds = [
        (0.1, 2.0),    # beta0
        (0.0, 0.8),    # a
        (0.5, 10.0),   # sigma
        (0.1, 5.0),    # gamma
        (0.01, 0.2)    # omega
    ]
    
    print("초기 파라미터:")
    print(f"  beta0 (전파율): {initial_params[0]:.3f}")
    print(f"  a (계절성): {initial_params[1]:.3f}")
    print(f"  sigma (1/잠복기): {initial_params[2]:.3f}")
    print(f"  gamma (1/감염기): {initial_params[3]:.3f}")
    print(f"  omega (1/면역기): {initial_params[4]:.3f}")
    
    print("\n최적화 진행 중...")
    result = minimize(
        loss_function,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500}
    )
    
    best_params = {
        'beta0': result.x[0],
        'a': result.x[1],
        'sigma': result.x[2],
        'gamma': result.x[3],
        'omega': result.x[4]
    }
    
    print("\n✅ 최적화 완료!")
    print(f"최종 파라미터:")
    print(f"  beta0 (전파율): {best_params['beta0']:.3f}")
    print(f"  a (계절성): {best_params['a']:.3f}")
    print(f"  sigma (1/잠복기): {best_params['sigma']:.3f} → 잠복기 약 {1/best_params['sigma']:.1f}주")
    print(f"  gamma (1/감염기): {best_params['gamma']:.3f} → 감염기 약 {1/best_params['gamma']:.1f}주")
    print(f"  omega (1/면역기): {best_params['omega']:.4f} → 면역기 약 {1/best_params['omega']:.1f}주")
    print(f"  최종 MSE: {result.fun:.4f}")
    
    return best_params


def evaluate_model(real_ili, predicted_ili, test_ratio=0.15):
    """
    모델 성능 평가 (최신 데이터만 사용)
    
    Parameters:
    -----------
    real_ili : array
        실제 ILI 데이터
    predicted_ili : array
        예측 ILI 데이터
    test_ratio : float
        평가에 사용할 최신 데이터 비율 (기본값: 0.15 = 15%)
    """
    # 최신 15% 데이터만 추출
    n_total = len(real_ili)
    n_test = int(n_total * test_ratio)
    
    # 최신 데이터 (마지막 15%)
    real_test = real_ili[-n_test:]
    pred_test = predicted_ili[-n_test:]
    
    # 평가 지표 계산 (MAE, MSE, RMSE)
    mae = np.mean(np.abs(real_test - pred_test))
    mse = np.mean((real_test - pred_test) ** 2)
    rmse = np.sqrt(mse)
    
    print("\n" + "=" * 60)
    print("🎯 최종 테스트 성능 평가 (최신 15% 데이터 기준)")
    print("=" * 60)
    print(f"평가 데이터 포인트: {n_test}/{n_total} ({test_ratio*100:.0f}%)")
    print(f"평가 기간: 최신 {n_test}주")
    print(f"\nMAE  (Mean Absolute Error):      {mae:.6f}")
    print(f"MSE  (Mean Squared Error):       {mse:.6f}")
    print(f"RMSE (Root Mean Squared Error):  {rmse:.6f}")
    print("=" * 60)
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'n_test': n_test, 'n_total': n_total}


def plot_results(df_avg, predicted_ili, metrics, save_path="seirs_model_results.png"):
    """결과 시각화"""
    print("\n📈 결과 그래프 생성 중...")
    
    # 시간 레이블 생성
    time_labels = df_avg['year'].astype(int).astype(str) + '-W' + \
                  df_avg['week'].astype(int).astype(str).str.zfill(2)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # 상단: 실제값 vs 예측값
    ax1 = axes[0]
    t = np.arange(len(df_avg))
    
    # 테스트 구간 (최신 15%) 표시
    n_test = metrics['n_test']
    test_start_idx = len(t) - n_test
    
    ax1.plot(t, df_avg['ili'].values, 'o-', label='실제 ILI 데이터', 
             linewidth=2, markersize=4, color='#2E86AB', alpha=0.7)
    ax1.plot(t, predicted_ili, '-', label='SEIRS 모델 예측', 
             linewidth=2.5, color='#E63946', alpha=0.8)
    
    # 테스트 구간 강조 (배경색)
    ax1.axvspan(test_start_idx, len(t)-1, alpha=0.15, color='yellow', 
                label=f'평가 구간 (최신 15%)')
    
    ax1.set_title('Seasonal SEIRS 모델: 실제 vs 예측', 
                  fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('시점 (연도-주차)', fontsize=12)
    ax1.set_ylabel('ILI 발생률 (인구 1,000명당)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # x축 레이블 (일부만)
    n_ticks = min(20, len(t))
    tick_indices = np.linspace(0, len(t)-1, n_ticks, dtype=int)
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels(time_labels.iloc[tick_indices], rotation=45, ha='right')
    
    # 성능 지표 텍스트 추가
    textstr = f"평가 구간: 최신 {metrics['n_test']}주 (15%)\nMAE: {metrics['MAE']:.4f}\nMSE: {metrics['MSE']:.4f}\nRMSE: {metrics['RMSE']:.4f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    # 하단: 오차 분석
    ax2 = axes[1]
    errors = df_avg['ili'].values - predicted_ili
    
    ax2.plot(t, errors, 'o-', linewidth=1.5, markersize=3, 
             color='#06A77D', alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.fill_between(t, errors, alpha=0.3, color='#06A77D')
    
    ax2.set_title('예측 오차 (실제값 - 예측값)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('시점 (연도-주차)', fontsize=12)
    ax2.set_ylabel('오차', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(time_labels.iloc[tick_indices], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 그래프 저장 완료: {save_path}")
    plt.show()


def save_predictions(df_avg, predicted_ili, save_path="seirs_predictions.csv"):
    """예측 결과 저장"""
    result_df = df_avg.copy()
    result_df['ili_predicted'] = predicted_ili
    result_df['error'] = result_df['ili'] - predicted_ili
    result_df['abs_error'] = np.abs(result_df['error'])
    
    result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 예측 결과 저장 완료: {save_path}")
    print(f"   컬럼: year, week, ili (실제), ili_predicted (예측), error, abs_error")


def main():
    """메인 실행 함수"""
    print("\n" + "🧮 " * 30)
    print("Seasonal SEIRS 모델 기반 인플루엔자 예측")
    print("🧮 " * 30 + "\n")
    
    # 1. 실제 데이터 로드
    df_avg = load_real_data()
    real_ili = df_avg['ili'].values
    
    # 2. 파라미터 최적화
    N = 1_000_000  # 총 인구수 (100만명)
    best_params = optimize_seirs_parameters(real_ili, N)
    
    # 3. 최적화된 파라미터로 예측
    print("\n" + "=" * 60)
    print("🎯 최적 파라미터로 SEIRS 모델 실행 중...")
    print("=" * 60)
    
    n_weeks = len(real_ili)
    t = np.arange(n_weeks)
    
    # 초기값 설정
    I0 = real_ili[0] * N / 1000
    E0 = I0 * 0.5
    R0 = N * 0.1
    S0 = N - I0 - E0 - R0
    
    predicted_ili = simulate_seirs(best_params, t, N, [S0, E0, I0, R0])
    
    # 4. 모델 평가
    metrics = evaluate_model(real_ili, predicted_ili)
    
    # 5. 결과 시각화
    plot_results(df_avg, predicted_ili, metrics)
    
    # 6. 결과 저장
    save_predictions(df_avg, predicted_ili)
    
    print("\n" + "=" * 60)
    print("✅ 모든 작업 완료!")
    print("=" * 60)
    print("\n생성된 파일:")
    print("  - seirs_model_results.png: 예측 결과 그래프")
    print("  - seirs_predictions.csv: 상세 예측 데이터")
    print("\n다음 단계:")
    print("  - patchTST 모델 결과와 비교 분석")
    print("  - 앙상블 모델 구축 고려")
    print()


if __name__ == "__main__":
    main()
