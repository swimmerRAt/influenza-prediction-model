#!/usr/bin/env python
"""
일별 데이터 변환 테스트 스크립트
주차별 데이터를 일별 데이터로 변환하고 모델 학습 준비 확인
"""

import sys
from patchTST import Config, load_and_prepare_by_age

def main():
    print("\n" + "="*60)
    print("📋 Configuration (일별 데이터 변환 모드)")
    print("="*60)
    print(f'USE_DAILY_DATA:       {Config.USE_DAILY_DATA}')
    print(f'DAILY_INTERP_METHOD:  {Config.DAILY_INTERP_METHOD}')
    print(f'GAUSSIAN_STD:         {Config.GAUSSIAN_STD}')
    print(f'DAILY_SEQ_LEN:        {Config.DAILY_SEQ_LEN} (입력 시퀀스 = 16주 × 7)')
    print(f'DAILY_PRED_LEN:       {Config.DAILY_PRED_LEN} (예측 길이 = 4주 × 7)')
    
    print("\n" + "="*60)
    print("🔄 데이터 로드 및 변환 시작...")
    print("="*60)
    
    try:
        # 데이터 로드 (자동으로 일별 변환됨)
        X, y, labels, feat_names = load_and_prepare_by_age(
            age_group='19-49세', 
            use_exog='all'
        )
        
        print("\n" + "="*60)
        print("✅ 변환 완료!")
        print("="*60)
        
        print(f"\n📊 입력 피처 (X):")
        print(f"   - Shape: {X.shape}")
        print(f"   - 시계열 길이: {X.shape[0]} 타임스텝 (일)")
        print(f"   - 피처 수: {X.shape[1]}")
        print(f"   - 피처 목록: {feat_names}")
        
        print(f"\n🎯 타겟 변수 (y):")
        print(f"   - Shape: {y.shape}")
        print(f"   - 값의 범위: [{y.min():.2f}, {y.max():.2f}]")
        print(f"   - 평균: {y.mean():.2f} ± {y.std():.2f}")
        print(f"   - 중위수: {sorted(y)[len(y)//2]:.2f}")
        
        print(f"\n📈 데이터 확대 통계:")
        print(f"   - 원본 데이터: ~429주 (2017-2025)")
        print(f"   - 변환 후: {len(y)} 일")
        print(f"   - 확대 비율: {len(y)/429:.1f}배")
        print(f"   - 메모리 증가: ~{len(y)/429:.1f}배")
        
        print(f"\n🏷️  샘플 라벨 (처음 5개):")
        for i in range(min(5, len(labels))):
            print(f"   [{i}] {labels[i]}")
        
        print("\n" + "="*60)
        print("✅ 모든 테스트 통과!")
        print("="*60)
        print("\n💡 다음 단계:")
        print("   1. 모델 학습: python patchTST.py --age-group '19-49세'")
        print("   2. 예측: python patchTST.py --mode inference")
        print("   3. 주차별로 다시 변환할 경우 Config.USE_DAILY_DATA = False")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
