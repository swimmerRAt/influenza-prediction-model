"""
시각화 유틸리티 모듈
"""

from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

from src.config import config, BASE_DIR


def plot_last_window(dates_test: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                     age_name: str, save_path: Optional[str] = None,
                     show: bool = False) -> None:
    """
    마지막 테스트 윈도우 예측 결과 시각화
    
    Args:
        dates_test: 테스트 날짜 배열
        y_true: 실제값
        y_pred: 예측값
        age_name: 연령 그룹 이름
        save_path: 저장 경로
        show: 그래프 표시 여부
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(dates_test, y_true, 'b-o', label='True', markersize=4)
    ax.plot(dates_test, y_pred, 'r--s', label='Predicted', markersize=4)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('ILI Rate')
    ax.set_title(f'Last Window Forecast - {age_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 날짜 포맷
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {save_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_test_reconstruction(test_df: pd.DataFrame, dates: pd.Series,
                              age_name: str, pred_len: int,
                              save_path: Optional[str] = None) -> None:
    """
    테스트 구간 전체 재구성 플롯
    
    Args:
        test_df: 테스트 결과 DataFrame
        dates: 날짜 시리즈
        age_name: 연령 그룹 이름
        pred_len: 예측 길이
        save_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 실제값 (실선)
    ax.plot(dates, test_df['true'], 'b-', label='Actual', linewidth=1.5)
    
    # 예측값 (점선 + 마커)
    ax.plot(dates, test_df['pred'], 'r--o', label='Predicted',
            linewidth=1.2, markersize=3, alpha=0.8)
    
    # 신뢰 구간 (있는 경우)
    if 'pred_lower' in test_df and 'pred_upper' in test_df:
        ax.fill_between(dates, test_df['pred_lower'], test_df['pred_upper'],
                        color='red', alpha=0.2, label='95% CI')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('ILI Rate')
    ax.set_title(f'Test Period Reconstruction - {age_name} (Horizon={pred_len})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {save_path}")
    
    plt.close()


def plot_ma_curves(dates: np.ndarray, raw: np.ndarray, ma: np.ndarray,
                   age_name: str, window: int = 4,
                   save_path: Optional[str] = None) -> None:
    """
    원본 데이터 vs 이동평균 플롯
    
    Args:
        dates: 날짜 배열
        raw: 원본 데이터
        ma: 이동평균 데이터
        age_name: 연령 그룹 이름
        window: 이동평균 윈도우 크기
        save_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(dates, raw, 'c-', alpha=0.5, label='Original')
    ax.plot(dates, ma, 'b-', linewidth=1.5, label=f'{window}-Week MA')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('ILI Rate')
    ax.set_title(f'ILI Rate with Moving Average - {age_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {save_path}")
    
    plt.close()


def plot_feature_importance(df_fi: pd.DataFrame, top_n: int = 15,
                            save_path: Optional[str] = None) -> None:
    """
    Feature Importance 바 차트
    
    Args:
        df_fi: Feature importance DataFrame
        top_n: 상위 N개 표시
        save_path: 저장 경로
    """
    df_plot = df_fi.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#d73027' if v > 0 else '#4575b4' 
              for v in df_plot['importance_raw_val']]
    
    ax.barh(range(len(df_plot)), df_plot['importance_raw_val'],
            color=colors, alpha=0.8)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (ΔMSE)')
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # Legend
    legend_elements = [
        Patch(facecolor='#d73027', label='Positive (Higher MSE)'),
        Patch(facecolor='#4575b4', label='Negative (Lower MSE)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {save_path}")
    
    plt.close()


def plot_train_val_loss(train_losses: List[float], val_losses: List[float],
                         save_path: Optional[str] = None) -> None:
    """
    학습/검증 손실 곡선 플롯
    
    Args:
        train_losses: 학습 손실 리스트
        val_losses: 검증 손실 리스트
        save_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=1.5)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=1.5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {save_path}")
    
    plt.close()


def plot_horizon_errors(errors_by_horizon: np.ndarray, metric: str = 'MAE',
                        save_path: Optional[str] = None) -> None:
    """
    Horizon별 오차 플롯
    
    Args:
        errors_by_horizon: Horizon별 오차 배열
        metric: 메트릭 이름
        save_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    horizons = range(1, len(errors_by_horizon) + 1)
    ax.bar(horizons, errors_by_horizon, color='steelblue', alpha=0.8)
    ax.set_xlabel('Horizon (Weeks)')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} by Forecast Horizon')
    ax.set_xticks(horizons)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {save_path}")
    
    plt.close()


def plot_pandemic_periods(dates: np.ndarray, y: np.ndarray,
                          pandemic_start: str = "2020-02-01",
                          pandemic_end: str = "2023-05-11",
                          save_path: Optional[str] = None) -> None:
    """
    팬데믹 구간 표시 플롯
    
    Args:
        dates: 날짜 배열
        y: ILI rate 배열
        pandemic_start: 팬데믹 시작일
        pandemic_end: 팬데믹 종료일
        save_path: 저장 경로
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(dates, y, 'b-', linewidth=1.2)
    
    # 팬데믹 구간 음영
    p_start = pd.to_datetime(pandemic_start)
    p_end = pd.to_datetime(pandemic_end)
    ax.axvspan(p_start, p_end, alpha=0.2, color='red', label='COVID-19 Period')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('ILI Rate')
    ax.set_title('ILI Rate with COVID-19 Period Highlighted')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {save_path}")
    
    plt.close()


def save_all_results(results: dict, output_dir: str = None) -> None:
    """
    모든 결과를 저장
    
    Args:
        results: 결과 딕셔너리
        output_dir: 출력 디렉토리
    """
    import os
    
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, 'results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 메트릭 저장
    if 'metrics' in results:
        metrics_path = os.path.join(output_dir, 'metrics.csv')
        pd.DataFrame([results['metrics']]).to_csv(metrics_path, index=False)
        print(f"[Save] Metrics saved to: {metrics_path}")
    
    # Feature Importance 저장
    if 'feature_importance' in results:
        fi_path = os.path.join(output_dir, 'feature_importance.csv')
        results['feature_importance'].to_csv(fi_path, index=False)
        print(f"[Save] Feature importance saved to: {fi_path}")
    
    # 예측 결과 저장
    if 'predictions' in results:
        pred_path = os.path.join(output_dir, 'predictions.csv')
        results['predictions'].to_csv(pred_path, index=False)
        print(f"[Save] Predictions saved to: {pred_path}")
    
    print(f"[Save] All results saved to: {output_dir}")
