#!/usr/bin/env python
"""
PatchTST 인플루엔자 예측 모델 - 메인 진입점
=====================================

원본 patchTST.py를 그대로 사용하는 래퍼 스크립트입니다.

사용법:
    # 기본 실행 (PostgreSQL에서 데이터 로드)
    python main.py

    # 특정 연령 그룹 예측
    python main.py --age-group 65세이상
    
    # 하이퍼파라미터 최적화 스킵
    python main.py --skip-optuna
    
    # 원본 CSV에서 직접 로드 (data/before 디렉토리)
    python main.py --use-raw-data --age-group 19-49세
    
    # 아형별 검출률 예측
    python main.py --subtype A
"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# 원본 patchTST.py에서 필요한 모든 것을 import
from patchTST import (
    # 설정
    Config, DEVICE, SEED, 
    SEQ_LEN, PRED_LEN, PATCH_LEN, STRIDE,
    D_MODEL, N_HEADS, ENC_LAYERS, FF_DIM, DROPOUT, HEAD_HIDDEN,
    LR, WEIGHT_DECAY, PATIENCE, WARMUP_EPOCHS, EPOCHS, BATCH_SIZE,
    OUT_CSV, PLOT_LAST_WINDOW, PLOT_TEST_RECON,
    
    # 유틸리티
    set_seed,
    
    # 데이터 로딩
    load_data_from_postgres,
    load_weather_data_from_postgres,
    merge_weather_with_influenza,
    load_and_prepare,
    load_and_prepare_by_age,
    load_raw_data_by_age_group,
    prepare_subtype_data,
    AGE_GROUP_MAPPING,
    
    # 학습 및 평가
    train_and_eval,
)

# Optuna 관련 (있는 경우에만)
try:
    from patchTST import OPTUNA_AVAILABLE
    if OPTUNA_AVAILABLE:
        from patchTST import run_optuna_optimization
    else:
        run_optuna_optimization = None
except ImportError:
    OPTUNA_AVAILABLE = False
    run_optuna_optimization = None


def parse_args():
    """CLI 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='PatchTST 인플루엔자 예측 모델',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 데이터 소스 옵션
    parser.add_argument('--use-raw-data', action='store_true',
                       help='data/before 디렉토리에서 원본 CSV 직접 로드')
    parser.add_argument('--data-dir', type=str, default='data/before',
                       help='원본 CSV 디렉토리 경로 (--use-raw-data 시)')
    
    # 연령 그룹 옵션
    parser.add_argument('--age-group', type=str, default=None,
                       help=f'연령 그룹 (예: 19-49세, 65세이상). 가능한 값: {list(AGE_GROUP_MAPPING.keys())}')
    
    # 아형 옵션
    parser.add_argument('--subtype', type=str, default=None,
                       help='인플루엔자 아형 (A 또는 B). 지정 시 아형별 검출률 예측')
    
    # 학습 옵션
    parser.add_argument('--skip-optuna', action='store_true',
                       help='Optuna 하이퍼파라미터 최적화 스킵')
    parser.add_argument('--optuna-trials', type=int, default=50,
                       help='Optuna 시도 횟수 (기본: 50)')
    
    # 기타
    parser.add_argument('--seed', type=int, default=SEED,
                       help=f'랜덤 시드 (기본: {SEED})')
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    # 시드 설정
    set_seed(args.seed)
    
    # 시작 로그
    print(f"\n{'#'*60}")
    print(f"# PatchTST 인플루엔자 예측 모델")
    print(f"# Device: {DEVICE}")
    print(f"# Daily Mode: {Config.USE_DAILY_DATA}")
    if Config.USE_DAILY_DATA:
        print(f"# Seq Len: {Config.DAILY_SEQ_LEN}일 (일별)")
        print(f"# Pred Len: {Config.DAILY_PRED_LEN}일 (일별)")
    else:
        print(f"# Seq Len: {SEQ_LEN}주 (주별)")
        print(f"# Pred Len: {PRED_LEN}주 (주별)")
    print(f"{'#'*60}\n")
    
    # ===== 데이터 로드 =====
    X, y, labels, feat_names = None, None, None, None
    
    if args.subtype:
        # 아형별 검출률 예측 모드
        print(f"\n🔬 아형별 검출률 예측 모드: {args.subtype}")
        X, y, labels, feat_names = prepare_subtype_data(
            subtype=args.subtype,
            data_dir=args.data_dir
        )
    
    elif args.use_raw_data and args.age_group:
        # 원본 CSV에서 직접 로드 (PostgreSQL 우회)
        print(f"\n📂 원본 CSV 직접 로드 모드: {args.age_group}")
        X, y, labels, feat_names = load_and_prepare_by_age(
            age_group=args.age_group,
            data_dir=args.data_dir,
            use_exog=Config.USE_EXOG
        )
    
    else:
        # PostgreSQL에서 로드 (기본 모드)
        print(f"\n📊 PostgreSQL 데이터 로드 모드")
        df = load_data_from_postgres()
        
        # 날씨 데이터 병합 시도
        try:
            df_weather = load_weather_data_from_postgres()
            if df_weather is not None and not df_weather.empty:
                df = merge_weather_with_influenza(df, df_weather)
        except Exception as e:
            print(f"⚠️ 날씨 데이터 병합 실패: {e}")
        
        # 데이터 전처리
        X, y, labels, feat_names = load_and_prepare(
            df=df,
            use_exog=Config.USE_EXOG,
            age_group=args.age_group,
            subtype=args.subtype
        )
    
    # ===== Optuna 최적화 =====
    if Config.USE_OPTUNA and not args.skip_optuna and OPTUNA_AVAILABLE and run_optuna_optimization:
        print(f"\n🔍 Optuna 하이퍼파라미터 최적화 시작...")
        print(f"   - 시도 횟수: {args.optuna_trials}")
        
        try:
            best_params = run_optuna_optimization(
                X, y, labels, feat_names,
                n_trials=args.optuna_trials
            )
            print(f"\n✅ 최적 하이퍼파라미터:")
            for k, v in best_params.items():
                print(f"   - {k}: {v}")
        except Exception as e:
            print(f"⚠️ Optuna 최적화 실패: {e}")
            print("   기본 하이퍼파라미터로 진행합니다.")
    
    # ===== 학습 및 평가 =====
    print(f"\n🚀 모델 학습 시작...")
    train_and_eval(X, y, labels, feat_names)
    
    # 종료 로그
    print(f"\n{'#'*60}")
    print(f"# 학습 완료!")
    print(f"# 예측 결과: {OUT_CSV}")
    print(f"# 시각화: {PLOT_LAST_WINDOW}")
    print(f"{'#'*60}")


if __name__ == '__main__':
    main()
