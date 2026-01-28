import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import pandas as pd
import numpy as np
import os
from glob import glob


# ============================================
# 데이터 전처리: 일별 → 주간 데이터 변환
# ============================================

# 인플루엔자 분석용 기상 컬럼 (한글 → 영어 매핑)
INFLUENZA_WEATHER_COLS_KR = [
    '평균기온(℃)',       # 낮을수록 바이러스 생존 ↑
    '최저기온(℃)',       # 야간 저온 → 호흡기 감염 증가
    '최고기온(℃)',       # 일교차 계산용
    '평균지면온도(℃)',    # 냉기 체감 환경 반영
    '평균5cm지중온도(℃)', # 장기 계절성
    '평균10cm지중온도(℃)',
    '평균20cm지중온도(℃)',
    '평균30cm지중온도(℃)',
    '평균상대습도(%)',    # 낮을수록 전파력 증가
    '최저상대습도(%)',    # aerosol 전파 강화
    '평균이슬점온도(℃)',  # 절대습도 proxy (핵심)
    '평균증기압(hPa)',    # 절대습도 직접 지표
]

# 영어 컬럼명
INFLUENZA_WEATHER_COLS = [
    'avg_temp',           # 평균기온
    'min_temp',           # 최저기온
    'max_temp',           # 최고기온
    'avg_ground_temp',    # 평균지면온도
    'avg_soil_temp_5cm',  # 평균5cm지중온도
    'avg_soil_temp_10cm', # 평균10cm지중온도
    'avg_soil_temp_20cm', # 평균20cm지중온도
    'avg_soil_temp_30cm', # 평균30cm지중온도
    'avg_humidity',       # 평균상대습도
    'min_humidity',       # 최저상대습도
    'avg_dew_point',      # 평균이슬점온도
    'avg_vapor_pressure', # 평균증기압
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
        '날짜': 'first',  # 주의 시작일
        **{col: 'mean' for col in numeric_cols if col in df.columns}
    }).reset_index()
    
    # 주 시작일과 종료일 추가
    weekly_df['주_시작일'] = df.groupby('주차')['날짜'].first().values
    weekly_df['주_종료일'] = df.groupby('주차')['날짜'].last().values
    weekly_df['일수'] = df.groupby('주차').size().values
    
    # 마지막 주가 7일 미만이면 제거 (선택사항)
    # weekly_df = weekly_df[weekly_df['일수'] == 7]
    
    return weekly_df


def create_weekly_sequences(df: pd.DataFrame, feature_cols: list = None, 
                            seq_length: int = 4) -> tuple:
    """
    TCN 학습용 주간 시퀀스 데이터 생성
    
    Args:
        df: 일별 데이터 DataFrame
        feature_cols: 사용할 특성 컬럼 리스트
        seq_length: 입력 시퀀스 길이 (몇 주치 데이터를 입력으로 사용할지)
        
    Returns:
        (X, y) 튜플 - X: 입력 시퀀스, y: 예측 타겟
    """
    # 주간 데이터로 변환
    weekly_df = create_weekly_data(df)
    
    # 기본 특성 컬럼
    if feature_cols is None:
        feature_cols = [
            '평균기온(℃)', '최저기온(℃)', '최고기온(℃)',
            '평균상대습도(%)', '평균풍속(m/s)', '평균해면기압(hPa)'
        ]
    
    # 데이터에 존재하는 컬럼만 선택
    feature_cols = [col for col in feature_cols if col in weekly_df.columns]
    
    # 결측치 처리
    weekly_df[feature_cols] = weekly_df[feature_cols].ffill().bfill()
    
    # 시퀀스 생성
    data = weekly_df[feature_cols].values
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])  # 다음 주 예측
    
    X = np.array(X)
    y = np.array(y)
    
    # PyTorch 텐서로 변환 (batch, features, seq_length)
    X = torch.FloatTensor(X).permute(0, 2, 1)
    y = torch.FloatTensor(y)
    
    return X, y


def save_weekly_data(data_dir: str = None, output_path: str = None):
    """
    주간 데이터를 CSV 파일로 저장
    
    Args:
        data_dir: 일별 데이터 폴더 경로
        output_path: 저장할 파일 경로
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    if output_path is None:
        output_path = os.path.join(data_dir, "weather_weekly.csv")
    
    # 데이터 로드 및 변환
    df = load_weather_data(data_dir)
    weekly_df = create_weekly_data(df)
    
    # 저장
    weekly_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"주간 데이터 저장 완료: {output_path}")
    print(f"총 {len(weekly_df)}주의 데이터")
    
    return weekly_df


def create_influenza_weather_data(data_dir: str = None, output_path: str = None) -> pd.DataFrame:
    """
    인플루엔자 데이터셋과 병합 가능한 주간 기상 데이터 생성
    - ISO 주차 기준 (year, week)
    - 인플루엔자 분석에 필요한 기상 컬럼만 포함
    - 컬럼명은 영어로 변환됨
    
    Args:
        data_dir: 일별 데이터 폴더 경로
        output_path: 저장할 파일 경로
        
    Returns:
        주간 기상 데이터 DataFrame (year, week 기준, 영어 컬럼명)
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # 일별 데이터 로드
    df = load_weather_data(data_dir)
    df['날짜'] = pd.to_datetime(df['날짜'])
    
    # ISO 연도, 주차 추출 (인플루엔자 데이터와 동일한 기준)
    df['year'] = df['날짜'].dt.isocalendar().year
    df['week'] = df['날짜'].dt.isocalendar().week
    
    # 데이터에 존재하는 기상 컬럼만 선택 (한글 컬럼명 기준)
    available_cols_kr = [col for col in INFLUENZA_WEATHER_COLS_KR if col in df.columns]
    
    # 주간 평균 계산
    weekly_df = df.groupby(['year', 'week']).agg({
        **{col: 'mean' for col in available_cols_kr}
    }).reset_index()
    
    # 일교차 계산 (최고기온 - 최저기온)
    if '최고기온(℃)' in weekly_df.columns and '최저기온(℃)' in weekly_df.columns:
        weekly_df['일교차(℃)'] = weekly_df['최고기온(℃)'] - weekly_df['최저기온(℃)']
    
    # 컬럼 순서 정리
    col_order = ['year', 'week'] + available_cols_kr
    if '일교차(℃)' in weekly_df.columns:
        col_order.append('일교차(℃)')
    weekly_df = weekly_df[col_order]
    
    # 결측치 처리
    weekly_df = weekly_df.ffill().bfill()
    
    # 소수점 2자리로 반올림
    numeric_cols = weekly_df.select_dtypes(include=[np.number]).columns
    weekly_df[numeric_cols] = weekly_df[numeric_cols].round(2)
    
    # 컬럼명을 영어로 변환
    weekly_df = weekly_df.rename(columns=COLUMN_NAME_MAPPING)
    
    # 영어 컬럼명 리스트 생성
    available_cols_en = [COLUMN_NAME_MAPPING.get(col, col) for col in available_cols_kr]
    if '일교차(℃)' in col_order:
        available_cols_en.append('temp_range')
    
    # 저장
    if output_path is None:
        output_path = os.path.join(data_dir, "weather_for_influenza.csv")
    
    weekly_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"인플루엔자용 기상 데이터 저장 완료: {output_path}")
    print(f"총 {len(weekly_df)}주의 데이터 ({weekly_df['year'].min()}년 ~ {weekly_df['year'].max()}년)")
    print(f"\n포함된 기상 변수 (영어 컬럼명):")
    for col_en in available_cols_en:
        print(f"  - {col_en}")
    
    return weekly_df


def merge_with_influenza_data(influenza_df: pd.DataFrame, weather_df: pd.DataFrame = None,
                               data_dir: str = None) -> pd.DataFrame:
    """
    인플루엔자 데이터와 기상 데이터 병합
    
    Args:
        influenza_df: 인플루엔자 데이터 DataFrame (year, week 컬럼 필요)
        weather_df: 기상 데이터 DataFrame (없으면 자동 생성)
        data_dir: 기상 데이터 폴더 경로
        
    Returns:
        병합된 DataFrame (영어 컬럼명)
    """
    if weather_df is None:
        weather_df = create_influenza_weather_data(data_dir)
    
    # year, week 기준으로 병합
    merged_df = influenza_df.merge(
        weather_df, 
        on=['year', 'week'], 
        how='left'
    )
    
    print(f"병합 완료: {len(merged_df)}개 행")
    # 첫 번째 영어 컬럼명으로 매칭률 확인
    print(f"기상 데이터 매칭률: {merged_df[INFLUENZA_WEATHER_COLS[0]].notna().sum() / len(merged_df) * 100:.1f}%")
    
    return merged_df


# ============================================
# TCN 모델 정의
# ============================================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class WeatherForecastModel(nn.Module):
    """TCN 기반 기상 예측 모델"""
    
    def __init__(self, num_features: int, seq_length: int = 8, 
                 hidden_channels: list = None, dropout: float = 0.2):
        """
        Args:
            num_features: 입력 특성 수
            seq_length: 입력 시퀀스 길이 (과거 몇 주 데이터를 사용할지)
            hidden_channels: TCN 히든 채널 리스트
            dropout: 드롭아웃 비율
        """
        super(WeatherForecastModel, self).__init__()
        
        if hidden_channels is None:
            hidden_channels = [32, 64, 32]
        
        self.num_features = num_features
        self.seq_length = seq_length
        
        # TCN 인코더
        self.tcn = TemporalConvNet(
            num_inputs=num_features,
            num_channels=hidden_channels,
            kernel_size=3,
            dropout=dropout
        )
        
        # 출력 레이어
        self.fc = nn.Linear(hidden_channels[-1], num_features)
    
    def forward(self, x):
        """
        Args:
            x: (batch, features, seq_length) 형태의 입력
        Returns:
            (batch, features) 형태의 출력 (다음 주 예측값)
        """
        # TCN 통과
        tcn_out = self.tcn(x)  # (batch, hidden_channels[-1], seq_length)
        
        # 마지막 시점의 출력 사용
        out = tcn_out[:, :, -1]  # (batch, hidden_channels[-1])
        
        # FC 레이어로 특성 차원 복원
        out = self.fc(out)  # (batch, num_features)
        
        return out


# ============================================
# 데이터 준비 및 학습 함수
# ============================================

def prepare_forecast_data(data_path: str = None, seq_length: int = 8):
    """
    예측용 데이터 준비
    
    Args:
        data_path: weather_for_influenza.csv 파일 경로
        seq_length: 입력 시퀀스 길이
        
    Returns:
        (X, y, feature_cols, scaler_params) 튜플
    """
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "data", "weather_for_influenza.csv"
        )
    
    df = pd.read_csv(data_path)
    
    # 예측에 사용할 기상 변수 (year, week 제외)
    feature_cols = [col for col in df.columns if col not in ['year', 'week']]
    
    # 데이터 정규화 (Min-Max 스케일링)
    data = df[feature_cols].values
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data_range = data_max - data_min
    data_range[data_range == 0] = 1  # 0으로 나누기 방지
    data_normalized = (data - data_min) / data_range
    
    scaler_params = {'min': data_min, 'max': data_max, 'range': data_range}
    
    # 시퀀스 생성
    X, y = [], []
    for i in range(len(data_normalized) - seq_length):
        X.append(data_normalized[i:i+seq_length])
        y.append(data_normalized[i+seq_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # PyTorch 텐서로 변환 (batch, features, seq_length)
    X = torch.FloatTensor(X).permute(0, 2, 1)
    y = torch.FloatTensor(y)
    
    return X, y, feature_cols, scaler_params, df


def train_forecast_model(model, X, y, epochs: int = 100, lr: float = 0.001, 
                         batch_size: int = 32, verbose: bool = True):
    """
    모델 학습
    
    Args:
        model: WeatherForecastModel 인스턴스
        X: 입력 데이터 (batch, features, seq_length)
        y: 타겟 데이터 (batch, features)
        epochs: 학습 에폭 수
        lr: 학습률
        batch_size: 배치 크기
        verbose: 학습 과정 출력 여부
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 학습/검증 분할 (90% 학습, 10% 검증)
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 검증
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
        
        # 최적 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss/len(dataloader):.6f}, Val Loss: {val_loss:.6f}")
    
    # 최적 모델 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    if verbose:
        print(f"\n최적 검증 손실: {best_val_loss:.6f}")
    
    return model


def forecast_future_weeks(model, last_sequence, scaler_params, feature_cols, 
                          num_weeks: int = 4, last_year: int = None, last_week: int = None):
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
        
    Returns:
        예측 결과 DataFrame
    """
    model.eval()
    
    predictions = []
    current_seq = last_sequence.clone()  # (1, features, seq_length)
    
    with torch.no_grad():
        for _ in range(num_weeks):
            # 예측
            pred = model(current_seq)  # (1, features)
            predictions.append(pred.numpy()[0])
            
            # 시퀀스 업데이트 (새 예측을 추가하고 가장 오래된 데이터 제거)
            pred_expanded = pred.unsqueeze(2)  # (1, features, 1)
            current_seq = torch.cat([current_seq[:, :, 1:], pred_expanded], dim=2)
    
    # 역정규화
    predictions = np.array(predictions)
    predictions_original = predictions * scaler_params['range'] + scaler_params['min']
    
    # DataFrame 생성
    result_df = pd.DataFrame(predictions_original, columns=feature_cols)
    
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
    
    # 소수점 2자리로 반올림
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
    result_df[numeric_cols] = result_df[numeric_cols].round(2)
    
    return result_df


def run_weather_forecast(data_dir: str = None, output_path: str = None, 
                         num_weeks: int = 4, epochs: int = 100):
    """
    전체 기상 예측 파이프라인 실행
    
    Args:
        data_dir: 데이터 디렉토리 경로
        output_path: 예측 결과 저장 경로
        num_weeks: 예측할 주 수 (기본 4주 = 약 1개월)
        epochs: 학습 에폭 수
        
    Returns:
        예측 결과 DataFrame
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    if output_path is None:
        output_path = os.path.join(data_dir, "weather_forecast.csv")
    
    data_path = os.path.join(data_dir, "weather_for_influenza.csv")
    
    print("=" * 60)
    print("TCN 기반 기상 예측 시작")
    print("=" * 60)
    
    # 1. 데이터 준비
    print("\n[1] 데이터 로드 및 전처리...")
    seq_length = 8  # 과거 8주 데이터로 다음 주 예측
    X, y, feature_cols, scaler_params, df = prepare_forecast_data(data_path, seq_length)
    print(f"  - 총 {len(df)}주의 데이터")
    print(f"  - 특성 수: {len(feature_cols)}")
    print(f"  - 학습 샘플 수: {len(X)}")
    
    # 마지막 데이터의 year, week 저장
    last_year = int(df['year'].iloc[-1])
    last_week = int(df['week'].iloc[-1])
    print(f"  - 마지막 데이터: {last_year}년 {last_week}주차")
    
    # 2. 모델 생성
    print("\n[2] 모델 생성...")
    model = WeatherForecastModel(
        num_features=len(feature_cols),
        seq_length=seq_length,
        hidden_channels=[64, 128, 64],
        dropout=0.2
    )
    print(f"  - 입력 특성: {len(feature_cols)}")
    print(f"  - 시퀀스 길이: {seq_length}")
    
    # 3. 모델 학습
    print(f"\n[3] 모델 학습 (에폭: {epochs})...")
    model = train_forecast_model(model, X, y, epochs=epochs, lr=0.001, batch_size=32)
    
    # 4. 미래 예측
    print(f"\n[4] 미래 {num_weeks}주 예측...")
    
    # 마지막 seq_length 주의 데이터로 예측 시작
    last_data = df[feature_cols].iloc[-seq_length:].values
    last_data_normalized = (last_data - scaler_params['min']) / scaler_params['range']
    last_sequence = torch.FloatTensor(last_data_normalized).T.unsqueeze(0)  # (1, features, seq_length)
    
    forecast_df = forecast_future_weeks(
        model, last_sequence, scaler_params, feature_cols,
        num_weeks=num_weeks, last_year=last_year, last_week=last_week
    )
    
    # 5. 결과 저장
    forecast_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[5] 예측 결과 저장 완료: {output_path}")
    
    print("\n" + "=" * 60)
    print("예측 결과")
    print("=" * 60)
    print(forecast_df.to_string(index=False))
    
    return forecast_df


# ============================================
# 메인 실행
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TCN 기반 기상 예측')
    parser.add_argument('--mode', type=str, default='forecast', 
                        choices=['forecast', 'preprocess'],
                        help='실행 모드: forecast(예측), preprocess(전처리)')
    parser.add_argument('--weeks', type=int, default=4, 
                        help='예측할 주 수 (기본: 4주 = 약 1개월)')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='학습 에폭 수 (기본: 100)')
    
    args = parser.parse_args()
    
    if args.mode == 'preprocess':
        # 인플루엔자용 기상 데이터 전처리
        print("=" * 60)
        print("인플루엔자 분석용 주간 기상 데이터 생성")
        print("=" * 60)
        
        weather_df = create_influenza_weather_data()
        
        print("\n[데이터 미리보기]")
        print(weather_df.head(15).to_string())
        
    else:  # forecast 모드
        # TCN 기반 기상 예측 실행
        forecast_df = run_weather_forecast(
            num_weeks=args.weeks,
            epochs=args.epochs
        )
        
        print("\n" + "=" * 60)
        print("사용 방법")
        print("=" * 60)
        print(f"""
예측 결과 파일: weather_forecast/data/weather_forecast.csv

다른 옵션으로 실행:
    # 8주 예측
    python TCN.py --weeks 8
    
    # 200 에폭으로 학습
    python TCN.py --epochs 200
    
    # 전처리 모드 (weather_for_influenza.csv 생성)
    python TCN.py --mode preprocess

영어 컬럼명:
    - avg_temp: 평균기온(℃)
    - min_temp: 최저기온(℃)
    - max_temp: 최고기온(℃)
    - avg_ground_temp: 평균지면온도(℃)
    - avg_soil_temp_5cm~30cm: 평균지중온도(℃)
    - avg_humidity: 평균상대습도(%)
    - min_humidity: 최저상대습도(%)
    - avg_dew_point: 평균이슬점온도(℃)
    - avg_vapor_pressure: 평균증기압(hPa)
    - temp_range: 일교차(℃)
        """)
