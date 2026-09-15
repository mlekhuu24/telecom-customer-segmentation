import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "telecom_customers.csv"
MODEL_PATH = BASE_DIR / "telecom_segmentation_model.pkl"
CHART_PATH = BASE_DIR / "customer_segments.png"

df = pd.read_csv(DATA_PATH)

features = [
    "tenure_months", "monthly_charge", "total_charge",
    "data_usage_gb", "call_minutes", "support_calls"
]

X = df[features]

preprocessor = StandardScaler()
X_scaled = preprocessor.fit_transform(X)

model = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = model.fit_predict(X_scaled)

df["segment"] = clusters
score = silhouette_score(X_scaled, clusters)

print(f"Silhouette Score: {score:.2f}")
print("\nCustomer count by segment:")
print(df["segment"].value_counts().sort_index())

joblib.dump(
    {"scaler": preprocessor, "model": model, "features": features},
    MODEL_PATH
)
print(f"\nModel saved to: {MODEL_PATH}")

plt.figure(figsize=(8, 5))
plt.scatter(
    df["monthly_charge"],
    df["data_usage_gb"],
    c=df["segment"],
    alpha=0.7
)
plt.xlabel("Monthly Charge")
plt.ylabel("Data Usage (GB)")
plt.title("Telecom Customer Segments")
plt.tight_layout()
plt.savefig(CHART_PATH)
print(f"Chart saved to: {CHART_PATH}")

df.to_csv(BASE_DIR / "segmented_customers.csv", index=False)
print("Segmented data saved to: segmented_customers.csv")
