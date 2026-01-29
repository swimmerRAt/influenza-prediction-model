"""
ClimODE: Climate and Weather Forecasting With Physics-informed Neural ODEs
- 주간(Weekly) 기상 데이터 예측을 위한 통합 모듈
- 데이터 전처리, 모델, 학습, 평가가 하나의 파일에 통합됨
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
from glob import glob
import argparse
import warnings
from torchdiffeq import odeint
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

warnings.filterwarnings('ignore')


# ============================================
# 1. 상수 정의
# ============================================

# 인플루엔자 분석용 기상 컬럼 (한글 → 영어 매핑)
# 🔴 최고기온, 최저기온, 습도 3개만 사용
INFLUENZA_WEATHER_COLS_KR = [
    '최저기온(℃)',       # 야간 저온 → 호흡기 감염 증가
    '최고기온(℃)',       # 일교차 계산용
    '평균상대습도(%)',    # 낮을수록 전파력 증가
]

# 영어 컬럼명
INFLUENZA_WEATHER_COLS = [
    'min_temp',           # 최저기온
    'max_temp',           # 최고기온
    'avg_humidity',       # 평균상대습도
]

# 한글 → 영어 컬럼명 매핑
COLUMN_NAME_MAPPING = {
    '평균기온(℃)': 'avg_temp',
    '최저기온(℃)': 'min_temp',
    '최고기온(℃)': 'max_temp',
    '평균지면온도(℃)': 'avg_ground_temp',
    '평균5cm지중온도(℃)': 'avg_soil_temp_5cm',
    '평균10cm지중온도(℃)': 'avg_soil_temp_10cm',
    '평균20cm지중온도(℃)': 'avg_soil_temp_20cm',
    '평균30cm지중온도(℃)': 'avg_soil_temp_30cm',
    '평균상대습도(%)': 'avg_humidity',
    '최저상대습도(%)': 'min_humidity',
    '평균이슬점온도(℃)': 'avg_dew_point',
    '평균증기압(hPa)': 'avg_vapor_pressure',
    '일교차(℃)': 'temp_range',
}

# ODE 솔버 옵션
SOLVERS = ["dopri8", "dopri5", "bdf", "rk4", "midpoint", 'adams', 
           'explicit_adams', 'fixed_adams', "adaptive_heun", "euler"]


# ============================================
# 2. 데이터 전처리: 일별 → 주간 데이터 변환
# ============================================

def set_seed(seed: int = 42) -> None:
    """랜덤 시드 설정"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def load_weather_data(data_dir: str = None) -> pd.DataFrame:
    """
    data 폴더의 모든 CSV 파일을 읽어 하나의 DataFrame으로 합침
    
    Args:
        data_dir: 데이터 폴더 경로
        
    Returns:
        합쳐진 DataFrame
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    csv_files = sorted(glob(os.path.join(data_dir, "weather_asos_*.csv")))
    
    if not csv_files:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {data_dir}")
    
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['날짜'] = pd.to_datetime(combined_df['날짜'])
    combined_df = combined_df.sort_values('날짜').reset_index(drop=True)
    
    return combined_df


def create_weekly_data(df: pd.DataFrame, numeric_cols: list = None) -> pd.DataFrame:
    """
    일별 데이터를 7일씩 묶어서 주간 데이터로 변환
    
    Args:
        df: 일별 데이터 DataFrame
        numeric_cols: 집계할 숫자형 컬럼 리스트 (None이면 자동 선택)
        
    Returns:
        주간 데이터 DataFrame
    """
    df = df.copy()
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.sort_values('날짜').reset_index(drop=True)
    
    # 주차 번호 계산 (7일 단위)
    df['주차'] = df.index // 7
    
    # 숫자형 컬럼만 선택 (자동)
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # '관측소ID', '주차' 등 불필요한 컬럼 제외
        exclude_cols = ['관측소ID', '주차']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # 주간 집계 (평균)
    weekly_df = df.groupby('주차').agg({
        '날짜': 'first',
        **{col: 'mean' for col in numeric_cols if col in df.columns}
    }).reset_index()
    
    # 주 시작일과 종료일 추가
    weekly_df['주_시작일'] = df.groupby('주차')['날짜'].first().values
    weekly_df['주_종료일'] = df.groupby('주차')['날짜'].last().values
    weekly_df['일수'] = df.groupby('주차').size().values
    
    return weekly_df


def create_influenza_weather_data(data_dir: str = None, output_path: str = None) -> pd.DataFrame:
    """
    인플루엔자 데이터셋과 병합 가능한 주간 기상 데이터 생성
    - ISO 주차 기준 (year, week)
    - 컬럼명은 영어로 변환됨
    
    Args:
        data_dir: 일별 데이터 폴더 경로
        output_path: 저장할 파일 경로
        
    Returns:
        주간 기상 데이터 DataFrame
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # 일별 데이터 로드
    df = load_weather_data(data_dir)
    df['날짜'] = pd.to_datetime(df['날짜'])
    
    # ISO 연도, 주차 추출
    df['year'] = df['날짜'].dt.isocalendar().year
    df['week'] = df['날짜'].dt.isocalendar().week
    
    # 데이터에 존재하는 기상 컬럼만 선택
    available_cols_kr = [col for col in INFLUENZA_WEATHER_COLS_KR if col in df.columns]
    
    # 주간 평균 계산
    weekly_df = df.groupby(['year', 'week']).agg({
        **{col: 'mean' for col in available_cols_kr}
    }).reset_index()
    
    # 🔴 일교차 계산 제거 (3개 컬럼만 사용)
    # if '최고기온(℃)' in weekly_df.columns and '최저기온(℃)' in weekly_df.columns:
    #     weekly_df['일교차(℃)'] = weekly_df['최고기온(℃)'] - weekly_df['최저기온(℃)']
    
    # 결측치 처리
    weekly_df = weekly_df.ffill().bfill()
    
    # 소수점 2자리로 반올림
    numeric_cols = weekly_df.select_dtypes(include=[np.number]).columns
    weekly_df[numeric_cols] = weekly_df[numeric_cols].round(2)
    
    # 컬럼명을 영어로 변환
    weekly_df = weekly_df.rename(columns=COLUMN_NAME_MAPPING)
    
    # 저장
    if output_path is None:
        output_path = os.path.join(data_dir, "weather_for_influenza.csv")
    
    weekly_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"주간 기상 데이터 저장 완료: {output_path}")
    print(f"총 {len(weekly_df)}주의 데이터 ({weekly_df['year'].min()}년 ~ {weekly_df['year'].max()}년)")
    
    return weekly_df


# ============================================
# 3. ClimODE 모델 유틸리티
# ============================================

class ResidualBlock(nn.Module):
    """2D Residual Block"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 activation: str = "gelu", norm: bool = False, n_groups: int = 1):
        super().__init__()
        self.activation = nn.LeakyReLU(0.3)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(p=0.1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()
        
        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        h = self.activation(self.bn1(self.conv1(self.norm1(x))))
        h = self.activation(self.bn2(self.conv2(self.norm2(h))))
        h = self.drop(h)
        return h + self.shortcut(x)


class ResidualBlock1D(nn.Module):
    """1D Residual Block for sequential data"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.activation = nn.LeakyReLU(0.3)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(p=0.1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        h = self.activation(self.bn1(self.conv1(x)))
        h = self.activation(self.bn2(self.conv2(h)))
        h = self.drop(h)
        return h + self.shortcut(x)


class Climate_ResNet_1D(nn.Module):
    """1D ResNet for weekly weather data"""
    
    def __init__(self, num_channels, layers, hidden_sizes):
        super().__init__()
        layers_cnn = []
        
        for idx in range(len(layers)):
            if idx == 0:
                layers_cnn.append(self._make_layer(num_channels, hidden_sizes[idx], layers[idx]))
            else:
                layers_cnn.append(self._make_layer(hidden_sizes[idx-1], hidden_sizes[idx], layers[idx]))
        
        self.layer_cnn = nn.ModuleList(layers_cnn)

    def _make_layer(self, in_channels, out_channels, reps):
        layers = [ResidualBlock1D(in_channels, out_channels)]
        for _ in range(1, reps):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, data):
        dx_final = data.float()
        for layer in self.layer_cnn:
            dx_final = layer(dx_final)
        return dx_final


# ============================================
# 4. ClimODE 주간 예측 모델
# ============================================

class Optim_velocity_weekly(nn.Module):
    """주간 데이터를 위한 속도 최적화 모듈"""
    
    def __init__(self, num_years, num_features, seq_length):
        super().__init__()
        self.v = nn.Parameter(torch.randn(num_years, num_features, seq_length) * 0.01)
    
    def forward(self, data):
        # 시간에 따른 미분 (변화율)
        grad = torch.gradient(data, dim=2)[0]
        adv = self.v * grad
        return adv, self.v


class ClimODE_Weekly(nn.Module):
    """
    주간 기상 예측을 위한 ClimODE 모델
    Physics-informed Neural ODE 기반
    """
    
    def __init__(self, num_features: int, seq_length: int = 8, 
                 hidden_channels: list = None, method: str = 'euler',
                 use_uncertainty: bool = True):
        """
        Args:
            num_features: 입력 특성 수 (기상 변수)
            seq_length: 입력 시퀀스 길이 (과거 몇 주)
            hidden_channels: 히든 채널 리스트
            method: ODE 솔버 방법
            use_uncertainty: 불확실성 추정 사용 여부
        """
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [64, 128, 64]
        
        self.num_features = num_features
        self.seq_length = seq_length
        self.method = method
        self.use_uncertainty = use_uncertainty
        
        # 시간 임베딩 차원 포함한 입력 크기
        # 입력: [features, time_emb(4), velocity(features)]
        input_channels = num_features * 2 + 4
        
        # Velocity Field Network
        self.vel_net = Climate_ResNet_1D(
            num_channels=input_channels,
            layers=[3, 2, 2],
            hidden_sizes=[hidden_channels[0], hidden_channels[1], num_features]
        )
        
        # 불확실성 추정 네트워크
        if use_uncertainty:
            self.uncertainty_net = Climate_ResNet_1D(
                num_channels=num_features + 4,
                layers=[2, 2],
                hidden_sizes=[hidden_channels[0], num_features]
            )
        
        # 학습 가능한 초기 속도
        self.init_velocity = nn.Parameter(torch.zeros(1, num_features, 1))
        
        # 기타 파라미터
        self.gamma = nn.Parameter(torch.tensor([0.1]))

    def get_time_embedding(self, t, batch_size, seq_len):
        """
        시간 임베딩 생성 (계절성 반영)
        
        Args:
            t: 시간 텐서 (주차)
            batch_size: 배치 크기
            seq_len: 시퀀스 길이
        """
        # 연간 주기 (52주)
        t_week = t.view(-1, 1, 1).expand(batch_size, 1, seq_len)
        sin_yearly = torch.sin(2 * np.pi * t_week / 52)
        cos_yearly = torch.cos(2 * np.pi * t_week / 52)
        
        # 반년 주기 (26주)
        sin_half = torch.sin(2 * np.pi * t_week / 26)
        cos_half = torch.cos(2 * np.pi * t_week / 26)
        
        return torch.cat([sin_yearly, cos_yearly, sin_half, cos_half], dim=1)

    def pde(self, t, state):
        """
        ODE 시스템의 우변 정의 (물리 기반 미분방정식)
        
        ds/dt = v · ∇s + f(s, t)
        
        여기서:
        - s: 상태 (기상 변수들)
        - v: 속도장 (이류)
        - f: 학습된 비선형 함수
        """
        batch_size = state.shape[0]
        seq_len = state.shape[2]
        
        # 상태와 속도 분리
        s = state[:, :self.num_features, :]  # 현재 상태
        v = state[:, self.num_features:, :]  # 속도장
        
        # 공간 미분 (시퀀스 방향)
        ds_dt_spatial = torch.gradient(s, dim=2)[0]
        
        # 시간 임베딩
        t_emb = self.get_time_embedding(t, batch_size, seq_len).to(state.device)
        
        # 결합 표현
        combined = torch.cat([s, v, t_emb], dim=1)
        
        # 속도장 업데이트
        dv = self.vel_net(combined)
        
        # 이류 항: v · ∇s
        advection = v * ds_dt_spatial
        
        # 상태 변화
        ds = advection + self.gamma * dv
        
        return torch.cat([ds, dv], dim=1)

    def forward(self, x, future_steps: int = 1):
        """
        순전파
        
        Args:
            x: 입력 시퀀스 (batch, features, seq_length)
            future_steps: 예측할 미래 스텝 수
            
        Returns:
            mean: 예측 평균 (batch, features, future_steps)
            std: 예측 표준편차 (batch, features, future_steps) - use_uncertainty=True일 때
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 초기 속도 설정
        init_v = self.init_velocity.expand(batch_size, -1, x.shape[2]).to(device)
        
        # 초기 상태: [현재 상태, 속도]
        state = torch.cat([x, init_v], dim=1)
        
        # ODE 적분 시간
        t = torch.linspace(0, future_steps, steps=future_steps + 1).to(device)
        
        # ODE 적분
        pde_rhs = lambda t, state: self.pde(t, state)
        result = odeint(pde_rhs, state, t, method=self.method, atol=0.1, rtol=0.1)
        
        # 마지막 스텝들의 상태 추출
        predictions = result[1:, :, :self.num_features, -1]  # (future_steps, batch, features)
        predictions = predictions.permute(1, 2, 0)  # (batch, features, future_steps)
        
        if self.use_uncertainty:
            # 불확실성 추정
            t_emb = self.get_time_embedding(
                torch.tensor([future_steps]).float(), 
                batch_size, 
                self.seq_length
            ).to(device)
            
            uncertainty_input = torch.cat([x, t_emb], dim=1)
            log_std = self.uncertainty_net(uncertainty_input)
            std = F.softplus(log_std[:, :, -1:].expand(-1, -1, future_steps))
            
            return predictions, std
        
        return predictions, None


# ============================================
# 5. 학습 및 평가 함수
# ============================================

def prepare_weekly_data(data_path: str = None, seq_length: int = 8):
    """
    ClimODE 학습용 주간 데이터 준비
    
    Args:
        data_path: weather_for_influenza.csv 파일 경로
        seq_length: 입력 시퀀스 길이
        
    Returns:
        (X, y, feature_cols, scaler_params, df) 튜플
    """
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "data", "weather_for_influenza.csv"
        )
    
    df = pd.read_csv(data_path)
    
    # year, week 제외한 기상 변수
    feature_cols = [col for col in df.columns if col not in ['year', 'week']]
    
    # Min-Max 정규화
    data = df[feature_cols].values
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data_range = data_max - data_min
    data_range[data_range == 0] = 1
    data_normalized = (data - data_min) / data_range
    
    scaler_params = {'min': data_min, 'max': data_max, 'range': data_range}
    
    # 시퀀스 생성
    X, y = [], []
    for i in range(len(data_normalized) - seq_length):
        X.append(data_normalized[i:i+seq_length])
        y.append(data_normalized[i+seq_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # PyTorch 텐서 변환 (batch, features, seq_length)
    X = torch.FloatTensor(X).permute(0, 2, 1)
    y = torch.FloatTensor(y)
    
    return X, y, feature_cols, scaler_params, df


def nll_loss(mean, std, truth, var_coeff=0.001):
    """
    Negative Log-Likelihood 손실 함수
    
    Args:
        mean: 예측 평균
        std: 예측 표준편차
        truth: 실제값
        var_coeff: 분산 정규화 계수
    """
    if std is None:
        return F.mse_loss(mean.squeeze(-1), truth)
    
    normal_dist = torch.distributions.Normal(mean.squeeze(-1), std.squeeze(-1) + 1e-6)
    nll = -normal_dist.log_prob(truth)
    loss = nll.mean() + var_coeff * (std ** 2).mean()
    return loss


def train_climode(model, X, y, epochs: int = 100, lr: float = 0.001, 
                  batch_size: int = 32, device: str = 'cpu', verbose: bool = True):
    """
    ClimODE 모델 학습
    
    Args:
        model: ClimODE_Weekly 인스턴스
        X: 입력 데이터 (batch, features, seq_length)
        y: 타겟 데이터 (batch, features)
        epochs: 학습 에폭 수
        lr: 학습률
        batch_size: 배치 크기
        device: 연산 장치
        verbose: 학습 과정 출력 여부
        
    Returns:
        학습된 모델
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # 학습/검증 분할 (90% / 10%)
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx].to(device), X[split_idx:].to(device)
    y_train, y_val = y[:split_idx].to(device), y[split_idx:].to(device)
    
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print("=" * 60)
    print("ClimODE 학습 시작")
    print("=" * 60)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        var_coeff = 2 * scheduler.get_last_lr()[0] if epoch > 0 else 0.001
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            mean, std = model(batch_X, future_steps=1)
            loss = nll_loss(mean, std, batch_y, var_coeff)
            
            # L2 정규화
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + 0.001 * l2_norm
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # 검증
        model.eval()
        with torch.no_grad():
            val_mean, val_std = model(X_val, future_steps=1)
            val_loss = nll_loss(val_mean, val_std, y_val, var_coeff).item()
        
        scheduler.step()
        
        # 최적 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                  f"Train Loss: {train_loss/len(dataloader):.6f} | "
                  f"Val Loss: {val_loss:.6f}")
    
    # 최적 모델 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\n최적 검증 손실: {best_val_loss:.6f}")
    
    return model


def evaluate_climode(model, X, y, scaler_params, feature_cols, device='cpu'):
    """
    ClimODE 모델 평가
    
    Args:
        model: 학습된 ClimODE 모델
        X: 테스트 입력
        y: 테스트 타겟
        scaler_params: 스케일링 파라미터
        feature_cols: 특성 컬럼 리스트
        device: 연산 장치
        
    Returns:
        평가 결과 딕셔너리
    """
    model = model.to(device)
    model.eval()
    
    X = X.to(device)
    y = y.to(device)
    
    with torch.no_grad():
        mean, std = model(X, future_steps=1)
        mean = mean.squeeze(-1)
        
        # MSE, MAE 계산
        mse = F.mse_loss(mean, y).item()
        mae = F.l1_loss(mean, y).item()
        
        # 원래 스케일로 변환
        mean_np = mean.cpu().numpy()
        y_np = y.cpu().numpy()
        
        mean_original = mean_np * scaler_params['range'] + scaler_params['min']
        y_original = y_np * scaler_params['range'] + scaler_params['min']
        
        # RMSE (원래 단위)
        rmse_original = np.sqrt(np.mean((mean_original - y_original) ** 2, axis=0))
    
    results = {
        'mse': mse,
        'mae': mae,
        'rmse_per_feature': dict(zip(feature_cols, rmse_original)),
    }
    
    return results


def forecast_future_weeks(model, last_sequence, scaler_params, feature_cols, 
                          num_weeks: int = 4, last_year: int = None, 
                          last_week: int = None, device='cpu'):
    """
    미래 주간 기상 예측
    
    Args:
        model: 학습된 모델
        last_sequence: 마지막 시퀀스 데이터 (정규화된 상태)
        scaler_params: 스케일링 파라미터
        feature_cols: 특성 컬럼 리스트
        num_weeks: 예측할 주 수
        last_year: 마지막 데이터의 연도
        last_week: 마지막 데이터의 주차
        device: 연산 장치
        
    Returns:
        예측 결과 DataFrame
    """
    model = model.to(device)
    model.eval()
    
    predictions = []
    uncertainties = []
    current_seq = last_sequence.clone().to(device)
    
    with torch.no_grad():
        for _ in range(num_weeks):
            mean, std = model(current_seq, future_steps=1)
            pred = mean[:, :, 0]  # (1, features)
            predictions.append(pred.cpu().numpy()[0])
            
            if std is not None:
                uncertainties.append(std[:, :, 0].cpu().numpy()[0])
            
            # 시퀀스 업데이트
            pred_expanded = pred.unsqueeze(2)
            current_seq = torch.cat([current_seq[:, :, 1:], pred_expanded], dim=2)
    
    # 역정규화
    predictions = np.array(predictions)
    predictions_original = predictions * scaler_params['range'] + scaler_params['min']
    
    # DataFrame 생성
    result_df = pd.DataFrame(predictions_original, columns=feature_cols)
    
    # 불확실성 추가
    if uncertainties:
        uncertainties = np.array(uncertainties)
        uncertainties_original = uncertainties * scaler_params['range']
        for i, col in enumerate(feature_cols):
            result_df[f'{col}_std'] = uncertainties_original[:, i]
    
    # year, week 계산
    years, weeks = [], []
    current_year, current_week = last_year, last_week
    
    for _ in range(num_weeks):
        current_week += 1
        if current_week > 52:
            current_week = 1
            current_year += 1
        years.append(current_year)
        weeks.append(current_week)
    
    result_df.insert(0, 'year', years)
    result_df.insert(1, 'week', weeks)
    
    # 소수점 정리
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
    result_df[numeric_cols] = result_df[numeric_cols].round(2)
    
    return result_df


# ============================================
# 6. 메인 파이프라인
# ============================================

def run_climode_forecast(data_dir: str = None, output_path: str = None,
                         num_weeks: int = 4, epochs: int = 100,
                         seq_length: int = 8, solver: str = 'euler'):
    """
    전체 ClimODE 예측 파이프라인 실행
    
    Args:
        data_dir: 데이터 디렉토리 경로
        output_path: 예측 결과 저장 경로
        num_weeks: 예측할 주 수
        epochs: 학습 에폭 수
        seq_length: 입력 시퀀스 길이
        solver: ODE 솔버
        
    Returns:
        예측 결과 DataFrame
    """
    set_seed(42)
    
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    if output_path is None:
        output_path = os.path.join(data_dir, "climode_forecast.csv")
    
    data_path = os.path.join(data_dir, "weather_for_influenza.csv")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("ClimODE 기반 주간 기상 예측")
    print(f"Device: {device}")
    print("=" * 60)
    
    # 0. 데이터 전처리 (주간 데이터 파일이 없으면 자동 생성)
    if not os.path.exists(data_path):
        print("\n[0] 주간 기상 데이터 생성 (일별 → 주간 변환)...")
        try:
            create_influenza_weather_data(data_dir, data_path)
        except FileNotFoundError as e:
            print(f"\n❌ 오류: 원본 일별 데이터를 찾을 수 없습니다.")
            print(f"   data 폴더에 'weather_asos_*.csv' 파일이 있는지 확인하세요.")
            print(f"   경로: {data_dir}")
            raise e
    else:
        print(f"\n[0] 기존 주간 데이터 사용: {data_path}")
    
    # 1. 데이터 준비
    print("\n[1] 데이터 로드 및 전처리...")
    X, y, feature_cols, scaler_params, df = prepare_weekly_data(data_path, seq_length)
    print(f"  - 총 {len(df)}주의 데이터")
    print(f"  - 특성 수: {len(feature_cols)}")
    print(f"  - 학습 샘플 수: {len(X)}")
    
    last_year = int(df['year'].iloc[-1])
    last_week = int(df['week'].iloc[-1])
    print(f"  - 마지막 데이터: {last_year}년 {last_week}주차")
    
    # 2. 모델 생성
    print(f"\n[2] ClimODE 모델 생성 (ODE Solver: {solver})...")
    model = ClimODE_Weekly(
        num_features=len(feature_cols),
        seq_length=seq_length,
        hidden_channels=[64, 128, 64],
        method=solver,
        use_uncertainty=True
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 학습 파라미터 수: {total_params:,}")
    
    # 3. 모델 학습
    print(f"\n[3] 모델 학습 (에폭: {epochs})...")
    model = train_climode(model, X, y, epochs=epochs, lr=0.001, 
                          batch_size=32, device=device)
    
    # 4. 평가
    print("\n[4] 모델 평가...")
    split_idx = int(len(X) * 0.9)
    X_test, y_test = X[split_idx:], y[split_idx:]
    results = evaluate_climode(model, X_test, y_test, scaler_params, feature_cols, device)
    print(f"  - Test MSE: {results['mse']:.6f}")
    print(f"  - Test MAE: {results['mae']:.6f}")
    print("  - Feature-wise RMSE:")
    for feat, rmse in results['rmse_per_feature'].items():
        print(f"      {feat}: {rmse:.4f}")
    
    # 5. 미래 예측
    print(f"\n[5] 미래 {num_weeks}주 예측...")
    last_data = df[feature_cols].iloc[-seq_length:].values
    last_data_normalized = (last_data - scaler_params['min']) / scaler_params['range']
    last_sequence = torch.FloatTensor(last_data_normalized).T.unsqueeze(0)
    
    forecast_df = forecast_future_weeks(
        model, last_sequence, scaler_params, feature_cols,
        num_weeks=num_weeks, last_year=last_year, last_week=last_week,
        device=device
    )
    
    # 6. 결과 저장
    forecast_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[6] 예측 결과 저장: {output_path}")
    
    # 7. 모델 저장
    model_path = os.path.join(data_dir, "climode_weekly_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"    모델 저장: {model_path}")
    
    print("\n" + "=" * 60)
    print("예측 결과")
    print("=" * 60)
    # 불확실성 컬럼 제외하고 출력
    display_cols = ['year', 'week'] + feature_cols
    print(forecast_df[display_cols].to_string(index=False))
    
    return forecast_df, model


# ============================================
# 7. 메인 실행
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ClimODE 기반 주간 기상 예측\n'
                    '기본 실행: python ClimODE.py (전처리 → 학습 → 예측 자동 실행)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--weeks', type=int, default=4,
                        help='예측할 주 수 (기본: 4)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='학습 에폭 수 (기본: 100)')
    parser.add_argument('--seq_length', type=int, default=8,
                        help='입력 시퀀스 길이 (기본: 8주)')
    parser.add_argument('--solver', type=str, default='euler',
                        choices=SOLVERS, help='ODE 솔버')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='데이터 디렉토리 경로')
    
    args = parser.parse_args()
    
    # ==========================================
    # 전체 파이프라인 자동 실행
    # (전처리 → 학습 → 평가 → 예측)
    # ==========================================
    
    print("\n" + "=" * 60)
    print("🌤️  ClimODE 주간 기상 예측 시스템")
    print("=" * 60)
    print("""
    [자동 실행 파이프라인]
    1. 데이터 전처리 (일별 → 주간 변환)
    2. ClimODE 모델 학습
    3. 모델 평가
    4. 미래 기상 예측
    """)
    
    # 전체 파이프라인 실행 (전처리는 run_climode_forecast 내부에서 자동 처리)
    forecast_df, model = run_climode_forecast(
        data_dir=args.data_dir,
        num_weeks=args.weeks,
        epochs=args.epochs,
        seq_length=args.seq_length,
        solver=args.solver
    )
    
    # ==========================================
    # 원본 데이터와 예측 데이터 병합
    # ==========================================
    print("\n[7] 원본 데이터와 예측 데이터 병합...")
    
    data_dir = args.data_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data"
    )
    
    # 원본 주간 기상 데이터 로드
    original_data_path = os.path.join(data_dir, "weather_for_influenza.csv")
    original_df = pd.read_csv(original_data_path)
    
    # 예측 데이터에서 불확실성 컬럼 제거 (원본과 동일한 컬럼 구조로)
    forecast_cols = [col for col in forecast_df.columns if not col.endswith('_std')]
    forecast_clean = forecast_df[forecast_cols].copy()
    
    # 원본 데이터와 예측 데이터 병합 (세로로 연결)
    merged_df = pd.concat([original_df, forecast_clean], ignore_index=True)
    
    # 정렬 (year, week 기준)
    merged_df = merged_df.sort_values(['year', 'week']).reset_index(drop=True)
    
    # 병합된 데이터 저장
    merged_output_path = os.path.join(data_dir, "weather_forecast_data.csv")
    merged_df.to_csv(merged_output_path, index=False, encoding='utf-8-sig')
    
    print(f"  - 원본 데이터: {len(original_df)}주")
    print(f"  - 예측 데이터: {len(forecast_clean)}주")
    print(f"  - 병합된 데이터: {len(merged_df)}주")
    print(f"  - 저장 완료: {merged_output_path}")
    
    print("\n" + "=" * 60)
    print("✅ 완료!")
    print("=" * 60)
    print(f"""
📁 생성된 파일:
    - data/weather_for_influenza.csv  (원본 주간 기상 데이터)
    - data/climode_forecast.csv       (예측 결과)
    - data/weather_forecast_data.csv  (원본 + 예측 병합) ⭐ NEW
    - data/climode_weekly_model.pt    (학습된 모델)

🔧 옵션 변경:
    python ClimODE.py --weeks 8        # 8주 예측
    python ClimODE.py --epochs 200     # 200 에폭 학습
    python ClimODE.py --solver dopri5  # dopri5 솔버 사용

📊 특성 컬럼 (3개):
    min_temp (최저기온), max_temp (최고기온), avg_humidity (평균습도)
    """)
