from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def get_pandemic_mask(df: pd.DataFrame) -> pd.Series:
	"""팬데믹 기간(2020년 14주 ~ 2022년 22주) 마스크 반환."""
	if 'year' not in df.columns or 'week' not in df.columns:
		return pd.Series(False, index=df.index)
	return (
		((df['year'] == 2020) & (df['week'] >= 14)) |
		(df['year'] == 2021) |
		((df['year'] == 2022) & (df['week'] <= 22))
	)


def interpolate_pandemic_period(df: pd.DataFrame, target_cols: list = None) -> pd.DataFrame:
	"""
	팬데믹 기간(2020년 14주 ~ 2022년 22주) 데이터를 
	2017~2019년 주차별 평균 패턴으로 보간
	
	Parameters:
		df: 원본 DataFrame (year, week 컬럼 필수)
		target_cols: 보간할 컬럼 목록 (None이면 year, week 제외한 모든 수치형 컬럼)
	
	Returns:
		보간된 DataFrame
	"""
	df = df.copy()
	
	# year, week 컬럼 확인
	if 'year' not in df.columns or 'week' not in df.columns:
		print("⚠️  year, week 컬럼이 없어 팬데믹 보간 건너뜀")
		return df
	
	pandemic_mask = get_pandemic_mask(df)
	
	pandemic_count = pandemic_mask.sum()
	if pandemic_count == 0:
		print("ℹ️  팬데믹 기간 데이터 없음 - 보간 건너뜀")
		return df
	
	print(f"\n🦠 팬데믹 기간 보간 처리")
	print(f"   팬데믹 기간: 2020년 14주 ~ 2022년 22주 ({pandemic_count}건)")
	
	# 보간 대상 컬럼 결정 (year, week 제외한 수치형)
	if target_cols is None:
		numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
		target_cols = [c for c in numeric_cols if c not in ['year', 'week']]
	
	if len(target_cols) == 0:
		print("⚠️  보간할 수치형 컬럼이 없음")
		return df
	
	print(f"   보간 대상 컬럼: {target_cols}")
	
	# 1단계: 팬데믹 기간 데이터를 NaN으로 설정
	for col in target_cols:
		df.loc[pandemic_mask, col] = np.nan
	
	# 2단계: 2017~2019년 정상 계절성 패턴 계산
	pre_pandemic_mask = (df['year'] >= 2017) & (df['year'] <= 2019)
	
	weekly_patterns = {}
	for col in target_cols:
		df_pre = df[pre_pandemic_mask & df[col].notna()]
		if len(df_pre) > 0:
			weekly_patterns[col] = df_pre.groupby('week')[col].mean()
		else:
			weekly_patterns[col] = None
	
	# 3단계: 팬데믹 구간을 계절성 패턴으로 보간
	interpolated_counts = {col: 0 for col in target_cols}
	
	for idx in df[pandemic_mask].index:
		week_num = df.loc[idx, 'week']
		
		for col in target_cols:
			pattern = weekly_patterns.get(col)
			if pattern is not None:
				if week_num in pattern.index:
					df.loc[idx, col] = pattern[week_num]
				else:
					# 해당 주차 패턴이 없으면 전체 평균 사용
					df.loc[idx, col] = pattern.mean()
				interpolated_counts[col] += 1
	
	# 결과 출력
	for col, count in interpolated_counts.items():
		if count > 0:
			print(f"   ✅ {col}: {count}건 보간 완료")
	
	return df


def iso_week_start_date(year: int, week: int) -> pd.Timestamp:
	"""Return Monday of ISO week for a given year/week."""
	y = int(year)
	w = int(week)
	max_week = pd.Timestamp(y, 12, 28).isocalendar().week
	if w > max_week:
		w = max_week
	if w < 1:
		w = 1
	return pd.Timestamp.fromisocalendar(y, w, 1)


def build_daily_influenza_data(
	input_csv: Path,
	output_csv: Path,
	year_col: str = "year",
	week_col: str = "week",
) -> None:
	if not input_csv.exists():
		raise FileNotFoundError(f"Input file not found: {input_csv}")

	df = pd.read_csv(input_csv)
	if year_col not in df.columns or week_col not in df.columns:
		raise ValueError(f"Missing required columns: {year_col}, {week_col}")

	df = df.copy()
	df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
	df[week_col] = pd.to_numeric(df[week_col], errors="coerce")
	df = df.dropna(subset=[year_col, week_col])

	# 팬데믹 기간 보간 (주 단위 데이터에서 수행)
	df = interpolate_pandemic_period(df)

	df["week_start"] = [iso_week_start_date(y, w) for y, w in zip(df[year_col], df[week_col])]
	df = df.drop(columns=[year_col, week_col])

	# Aggregate duplicated week_start entries if any.
	num_cols = df.select_dtypes(include="number").columns.tolist()
	agg = {c: "mean" for c in num_cols}
	for c in df.columns:
		if c not in num_cols and c != "week_start":
			agg[c] = "first"
	df = df.groupby("week_start", as_index=False).agg(agg)

	df = df.set_index("week_start").sort_index()
	df_daily = df.resample("D").asfreq()

	# Linear interpolation for numeric columns.
	for c in num_cols:
		df_daily[c] = df_daily[c].interpolate(method="linear", limit_direction="both")

	# Round to 3 decimal places after interpolation.
	df_daily[num_cols] = df_daily[num_cols].round(3)

	# Forward/back fill for non-numeric columns.
	cat_cols = [c for c in df.columns if c not in num_cols]
	for c in cat_cols:
		df_daily[c] = df_daily[c].ffill().bfill()

	df_daily = df_daily.reset_index().rename(columns={"week_start": "date"})
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	df_daily.to_csv(output_csv, index=False, encoding="utf-8-sig")


def load_and_prepare_weather_data(data_dir: Path) -> pd.DataFrame:
	files = sorted(data_dir.glob("weather_asos_서울_*.csv"))
	if not files:
		raise FileNotFoundError(f"No weather files found in {data_dir}")

	frames = []
	for file_path in files:
		df = pd.read_csv(file_path)
		required_cols = [
			"날짜",
			"최저기온(℃)",
			"최고기온(℃)",
			"일강수량(mm)",
			"평균상대습도(%)",
		]
		missing = [c for c in required_cols if c not in df.columns]
		if missing:
			raise ValueError(f"Missing columns in {file_path.name}: {missing}")

		df = df[required_cols].copy()
		df["date"] = pd.to_datetime(df["날짜"], errors="coerce")
		df["min_temp"] = pd.to_numeric(df["최저기온(℃)"], errors="coerce")
		df["max_temp"] = pd.to_numeric(df["최고기온(℃)"], errors="coerce")
		df["daily_rain"] = pd.to_numeric(df["일강수량(mm)"], errors="coerce")
		df["avg_humidity"] = pd.to_numeric(df["평균상대습도(%)"], errors="coerce")
		df = df[["date", "min_temp", "max_temp", "daily_rain", "avg_humidity"]]
		frames.append(df)

	weather = pd.concat(frames, ignore_index=True)
	weather = weather.dropna(subset=["date"])
	weather = weather.groupby("date", as_index=False).mean(numeric_only=True)
	weather = weather.set_index("date").sort_index()

	weather_daily = weather.resample("D").asfreq()
	for c in ["min_temp", "max_temp", "daily_rain", "avg_humidity"]:
		weather_daily[c] = weather_daily[c].interpolate(method="linear", limit_direction="both")

	# Round to 3 decimal places after interpolation.
	weather_daily[["min_temp", "max_temp", "daily_rain", "avg_humidity"]] = (
		weather_daily[["min_temp", "max_temp", "daily_rain", "avg_humidity"]].round(3)
	)

	return weather_daily.reset_index()


def merge_influenza_with_weather(
	influenza_csv: Path,
	weather_df: pd.DataFrame,
	output_csv: Path,
) -> None:
	if not influenza_csv.exists():
		raise FileNotFoundError(f"Input file not found: {influenza_csv}")

	influenza = pd.read_csv(influenza_csv)
	if "date" in influenza.columns:
		influenza["date"] = pd.to_datetime(influenza["date"], errors="coerce")
	elif "날짜" in influenza.columns:
		influenza["date"] = pd.to_datetime(influenza["날짜"], errors="coerce")
	else:
		raise ValueError("Influenza data must include a 'date' or '날짜' column")

	weather_df = weather_df.copy()
	if "date" in weather_df.columns:
		weather_df["date"] = pd.to_datetime(weather_df["date"], errors="coerce")
	elif "날짜" in weather_df.columns:
		weather_df["date"] = pd.to_datetime(weather_df["날짜"], errors="coerce")
	else:
		raise ValueError("Weather data must include a 'date' or '날짜' column")

	merged = influenza.merge(weather_df, on="date", how="left")
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	merged.to_csv(output_csv, index=False, encoding="utf-8-sig")


def validate_weather_merge(
	data_dir: Path,
	merged_csv: Path,
	tolerance: float = 0.001,
) -> None:
	if not merged_csv.exists():
		raise FileNotFoundError(f"Merged file not found: {merged_csv}")

	weather_df = load_and_prepare_weather_data(data_dir)
	merged = pd.read_csv(merged_csv)
	if "date" not in merged.columns:
		raise ValueError("Merged data must include a 'date' column")

	merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
	weather_df["date"] = pd.to_datetime(weather_df["date"], errors="coerce")

	cols = ["min_temp", "max_temp", "daily_rain", "avg_humidity"]
	missing_cols = [c for c in cols if c not in merged.columns]
	if missing_cols:
		raise ValueError(f"Merged data missing weather columns: {missing_cols}")

	merged_cmp = merged[["date"] + cols].dropna(subset=["date"]).copy()
	weather_cmp = weather_df[["date"] + cols].dropna(subset=["date"]).copy()

	merged_cmp = merged_cmp.drop_duplicates(subset=["date"]).sort_values("date")
	weather_cmp = weather_cmp.drop_duplicates(subset=["date"]).sort_values("date")

	joined = merged_cmp.merge(weather_cmp, on="date", how="inner", suffixes=("_merged", "_raw"))
	if joined.empty:
		raise ValueError("No overlapping dates found for validation")

	print("\n=== Weather Merge Validation ===")
	print(f"Matched dates: {len(joined)}")
	for c in cols:
		col_m = f"{c}_merged"
		col_r = f"{c}_raw"
		delta = (joined[col_m] - joined[col_r]).abs()
		mismatch = (delta > tolerance).sum()
		max_diff = float(delta.max()) if len(delta) else 0.0
		print(f"{c}: mismatches={mismatch}, max_diff={max_diff:.6f}")


if __name__ == "__main__":
	workspace_dir = Path(__file__).resolve().parent
	base_influenza_path = workspace_dir / "influenza_data.csv"
	build_daily_influenza_data(
		input_csv=workspace_dir / "final_data.csv",
		output_csv=base_influenza_path,
	)

	weather_df = load_and_prepare_weather_data(workspace_dir / "data")
	merge_influenza_with_weather(
		influenza_csv=base_influenza_path,
		weather_df=weather_df,
		output_csv=workspace_dir / "influenza_data.csv",
	)

	validate_weather_merge(
		data_dir=workspace_dir / "data",
		merged_csv=workspace_dir / "influenza_data.csv",
	)
