import pandas as pd
import numpy as np

# Load dataset
# Adjust the path if needed
csv_path = "../sales/Cashflow_Forecasting/cashflow_2024.csv"
df = pd.read_csv(csv_path, parse_dates=["Date"])
df.set_index("Date", inplace=True)

df.ffill(inplace=True)

# Create lag features (past cash flow as features)
for lag in range(1, 8):  # 7-day lag
    df[f"net_cash_flow_lag_{lag}"] = df["net_cash_flow"].shift(lag)

df.dropna(inplace=True)
if 'Activity' in df.columns:
    del df['Activity']

# Add time-based features
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month

# The features are all columns except 'net_cash_flow'
feature_columns = [
    f"net_cash_flow_lag_{i}" for i in range(1, 8)
] + ["day_of_week", "month"]

latest_features = df.iloc[-1][feature_columns].values

print("Latest feature vector (comma-separated, for web input):")
print(", ".join(str(float(x)) for x in latest_features)) 