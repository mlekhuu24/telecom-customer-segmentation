import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "telecom_segmentation_model.pkl"

saved = joblib.load(MODEL_PATH)
scaler = saved["scaler"]
model = saved["model"]
features = saved["features"]

print("Telecom Customer Segmentation")
print("-" * 40)

tenure = float(input("Tenure in months: "))
monthly_charge = float(input("Monthly charge: "))
total_charge = float(input("Total charge: "))
data_usage = float(input("Data usage in GB: "))
call_minutes = float(input("Call minutes: "))
support_calls = float(input("Number of support calls: "))

sample = pd.DataFrame([{
    "tenure_months": tenure,
    "monthly_charge": monthly_charge,
    "total_charge": total_charge,
    "data_usage_gb": data_usage,
    "call_minutes": call_minutes,
    "support_calls": support_calls
}])[features]

scaled_sample = scaler.transform(sample)
segment = model.predict(scaled_sample)[0]

print(f"\nAssigned Customer Segment: {segment}")
print("Use the segment number to compare customer behavior and plan targeted offers.")
