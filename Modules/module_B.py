# SignalForge — Module B prototype
# AI-Driven Anomaly Detection in Component Burn-In & Screening
# Goal: use Value_0h + Value_24h to predict Value_168h

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("sih26170_synthetic_dataset.csv")

# 1. Basic inspection
print(df.shape)
print(df.head())
print(df["Condition"].value_counts())

# 2. Feature engineering
df["Early_Drift"] = df["Value_24h"] - df["Value_0h"]
df["Early_Drift_Rate"] = df["Early_Drift"] / 24.0

# 3. Visualize example trajectories
for _, row in df.sample(5, random_state=7).iterrows():
    plt.plot([0,24,96,168],
             [row.Value_0h,row.Value_24h,row.Value_96h,row.Value_168h],
             marker="o", label=row.Component_ID)
plt.xlabel("ESS time (hours)")
plt.ylabel("Parameter value")
plt.title("Sample component trajectories")
plt.legend()
plt.show()

# 4. Module B: early measurements -> 168h
X = df[["Value_0h", "Value_24h"]]
y = df["Value_168h"]

# IMPORTANT: split by LOT, not random rows, so a whole lot is unseen during testing.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=df["Lot_ID"]))

model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)
model.fit(X.iloc[train_idx], y.iloc[train_idx])

pred = model.predict(X.iloc[test_idx])
mae = mean_absolute_error(y.iloc[test_idx], pred)
r2 = r2_score(y.iloc[test_idx], pred)

print("Held-out-lot MAE:", round(mae, 3))
print("Held-out-lot R²:", round(r2, 3))

# 5. Prediction table for the test set
results = df.iloc[test_idx][
    ["Lot_ID","Component_ID","Value_0h","Value_24h","Value_168h","Condition"]
].copy()
results["Predicted_168h"] = pred
results["Absolute_Error"] = abs(results["Value_168h"] - results["Predicted_168h"])
print(results.sort_values("Absolute_Error", ascending=False).head(10))

# 6. Prototype safety-slope criterion
# The SIH statement requires a calculated safety slope. The supplied dataset
# does not provide an engineering-qualified slope, so this prototype calibrates
# a screening threshold from NORMAL components only.
normal_early_drift = df.loc[df["Condition"]=="normal", "Early_Drift"]
SAFETY_EARLY_DRIFT = normal_early_drift.quantile(0.95)
print("Prototype 95th-percentile normal early drift:", round(SAFETY_EARLY_DRIFT, 3))
print("Equivalent early drift rate per hour:", round(SAFETY_EARLY_DRIFT/24, 5))

# 7. Fit on all data for the demo predictor.
# This is for demonstration only; evaluation above remains the held-out-lot result.
demo_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)
demo_model.fit(X, y)
df["Predicted_168h"] = demo_model.predict(X)

# 8. Basic Module A dynamic lot-level anomaly score
# Robust z-score against each lot's median/MAD.
lot_median = df.groupby("Lot_ID")["Value_24h"].transform("median")
lot_mad = df.groupby("Lot_ID")["Value_24h"].transform(
    lambda s: np.median(np.abs(s - np.median(s)))
)
scale = (1.4826 * lot_mad).replace(0, np.nan)
df["Lot_Robust_Z"] = ((df["Value_24h"] - lot_median) / scale).abs().fillna(0)
df["Anomaly_Flag"] = df["Lot_Robust_Z"] >= 3.5

# 9. Module B drift-risk flag
df["Predicted_Early_to_168_Drift"] = df["Predicted_168h"] - df["Value_24h"]
df["Drift_Risk_Flag"] = (
    (df["Predicted_Early_to_168_Drift"] > SAFETY_EARLY_DRIFT)
    | (df["Early_Drift"] > SAFETY_EARLY_DRIFT)
)

# 10. Explainable risk engine
def risk(row):
    reasons = []
    if row["Anomaly_Flag"]:
        reasons.append("abnormal deviation from lot baseline")
    if row["Drift_Risk_Flag"]:
        reasons.append("projected/existing drift exceeds prototype safety criterion")
    if not reasons:
        return "NORMAL", "within lot baseline and drift criterion"
    if row["Anomaly_Flag"] and row["Drift_Risk_Flag"]:
        return "HIGH RISK", "; ".join(reasons)
    return "REVIEW", reasons[0]

df[["Risk","Reason"]] = df.apply(lambda r: pd.Series(risk(r)), axis=1)

# 11. Inspect a few interesting components
print(df[df["Condition"]!="normal"][
    ["Component_ID","Condition","Value_0h","Value_24h","Predicted_168h",
     "Value_168h","Early_Drift","Lot_Robust_Z","Risk","Reason"]
].head(20))
