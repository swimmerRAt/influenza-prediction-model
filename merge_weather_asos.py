#!/usr/bin/env python3
"""data/weather_asos_*.csv 파일을 병합해 data/weather_for_influenza.csv로 저장"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = DATA_DIR / "weather_for_influenza.csv"


def main() -> None:
    files = sorted(DATA_DIR.glob("weather_asos_*.csv"))
    if not files:
        raise FileNotFoundError(f"No weather_asos_*.csv found in {DATA_DIR}")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        df["_source_file"] = fp.name
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    # 중복 제거 (날짜/관측소 기준 우선)
    if {"날짜", "관측소ID"}.issubset(merged.columns):
        merged = merged.drop_duplicates(subset=["날짜", "관측소ID"], keep="last")
        merged = merged.sort_values(by=["날짜", "관측소ID"])
    elif "날짜" in merged.columns:
        merged = merged.drop_duplicates(subset=["날짜"], keep="last")
        merged = merged.sort_values(by=["날짜"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"✅ merged rows: {len(merged)}")
    print(f"✅ saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
