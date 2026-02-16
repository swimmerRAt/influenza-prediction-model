import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("final_data.csv")

# 2024-W41 ~ 2026-W03 범위 필터
mask = (
    (df["year"] > 2024) | ((df["year"] == 2024) & (df["week"] >= 41))
)
mask &= (
    (df["year"] < 2026) | ((df["year"] == 2026) & (df["week"] <= 3))
)

subset = df.loc[mask, ["year", "week", "ili"]].copy()
subset = subset.sort_values(["year", "week"])

x_labels = [f"{y}-W{int(w):02d}" for y, w in zip(subset["year"], subset["week"])]

plt.figure(figsize=(12, 5))
plt.plot(x_labels, subset["ili"], marker="o", linewidth=2)
plt.title("ILI from 2024-W41 to 2026-W03")
plt.xlabel("Year-Week")
plt.ylabel("ILI")
plt.xticks(rotation=45, ha="right")
plt.grid(True)
plt.tight_layout()
plt.savefig("ili_2024w41_2026w03.png", dpi=150)
plt.show()