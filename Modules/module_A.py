import pandas as pd
df = pd.read_csv("sih26170_synthetic_dataset.csv")

results = []

for lot_id, lot in df.groupby("Lot_ID"):
    lot = lot.copy()

    median = lot["Value_0h"].median()

    deviations = abs(lot["Value_0h"] - median)

    mad = deviations.median()

    scores = abs(lot["Value_0h"] - median) / (mad + 1e-9)

    lot["Median"] = median
    lot["MAD"] = mad
    lot["Anomaly_Score"] = scores

    results.append(lot)

    
results = pd.concat(results, ignore_index=True)
results["Anomaly_Flag"] = results["Anomaly_Score"] > 4

# Ground truth for Module A
results["Actual_Anomaly"] = results["Condition"] == "latent_anomaly"

# Calculate TP, FP, TN, FN
true_positive = ((results["Anomaly_Flag"] == True) & (results["Actual_Anomaly"] == True)).sum()

false_positive = ((results["Anomaly_Flag"] == True) & (results["Actual_Anomaly"] == False)).sum()

true_negative = ((results["Anomaly_Flag"] == False) & (results["Actual_Anomaly"] == False)).sum()

false_negative = ((results["Anomaly_Flag"] == False) & (results["Actual_Anomaly"] == True)).sum()


precision = true_positive / (true_positive + false_positive)

recall = true_positive / (true_positive + false_negative)