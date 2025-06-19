# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../sales/Cashflow_Forecasting/cashflow_2024.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Fill missing values
# df.fillna(method="ffill", inplace=True)
df.ffill(inplace=True)

# Create lag features (past cash flow as features)
for lag in range(1, 8):  # 7-day lag
    df[f"net_cash_flow_lag_{lag}"] = df["net_cash_flow"].shift(lag)

# Drop missing values (due to lagging)
df.dropna(inplace=True)
del df['Activity']

# Check dataset
df.head()
# df.info()


# ### Step 3: Feature Engineering ###

# In[3]:


from sklearn.preprocessing import StandardScaler

# Add time-based features
df["day_of_week"] = df.index.dayofweek
df["month"] = df.index.month

# Explicitly specify the feature columns
feature_columns = [
    "net_cash_flow_lag_1",
    "net_cash_flow_lag_2",
    "net_cash_flow_lag_3",
    "net_cash_flow_lag_4",
    "net_cash_flow_lag_5",
    "net_cash_flow_lag_6",
    "net_cash_flow_lag_7",
    "day_of_week",
    "month"
]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[feature_columns])

# Convert back to DataFrame
df_scaled = pd.DataFrame(scaled_features, index=df.index, columns=feature_columns)
df_scaled["net_cash_flow"] = df["net_cash_flow"]  # Add target back for splitting

# Split into training and testing
train_size = int(len(df) * 0.8)
train, test = df_scaled.iloc[:train_size], df_scaled.iloc[train_size:]

X_train, y_train = train[feature_columns], train["net_cash_flow"]
X_test, y_test = test[feature_columns], test["net_cash_flow"]


# ### Step 4: Train an XGBoost Model ###

# In[4]:


import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# Train XGBoost model
model_xgb = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
model_xgb.fit(X_train, y_train)

# Make predictions
y_pred_xgb = model_xgb.predict(X_test)

# Evaluate model
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print(f"XGBoost MAE: {mae_xgb}")

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(test.index, y_test, label="Actual", color="blue")
plt.plot(test.index, y_pred_xgb, label="Predicted", color="red")
plt.legend()
plt.title("XGBoost Forecasting - Cash Flow")
plt.show()

# Save the model and scaler for deployment
import joblib

# Save the XGBoost model
joblib.dump(model_xgb, "model_xgb.joblib")

# Save the scaler
joblib.dump(scaler, "scaler.joblib")

print("Model and scaler saved successfully!")


# ### Step 5: Train an LSTM Model (Deep Learning) ###

# In[5]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Reshape for LSTM input
scaler_lstm = MinMaxScaler()
scaled_lstm = scaler_lstm.fit_transform(df)

# Create sequences for LSTM (past 7 days → next day prediction)
def create_sequences(data, n_steps=7):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps, :-1])  # All columns except target
        y.append(data[i+n_steps, -1])  # Target column
    return np.array(X), np.array(y)

X_lstm, y_lstm = create_sequences(scaled_lstm)
X_train_lstm, X_test_lstm = X_lstm[:train_size], X_lstm[train_size:]
y_train_lstm, y_test_lstm = y_lstm[:train_size], y_lstm[train_size:]

# Define LSTM model
model_lstm = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

# Compile and train
model_lstm.compile(optimizer="adam", loss="mse")
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=16, validation_data=(X_test_lstm, y_test_lstm))

# Predictions
y_pred_lstm = model_lstm.predict(X_test_lstm)

# Rescale predictions back to original values
y_pred_lstm = scaler_lstm.inverse_transform(np.concatenate((X_test_lstm[:, -1, :], y_pred_lstm), axis=1))[:, -1]

# Evaluate LSTM
mae_lstm = mean_absolute_error(y_test_lstm, y_pred_lstm)
print(f"LSTM MAE: {mae_lstm}")

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(test.index[-len(y_pred_lstm):], y_test_lstm, label="Actual", color="blue")
plt.plot(test.index[-len(y_pred_lstm):], y_pred_lstm, label="Predicted", color="red")
plt.legend()
plt.title("LSTM Forecasting - Cash Flow")
plt.show()


# ### Step 6: Deploy the Model ###

# In[12]:


from fastapi import FastAPI
import numpy as np

app = FastAPI()

@app.get("/predict_cash_flow/")
def predict_cash_flow(features: str):
    features_array = np.array([float(x) for x in features.split(",")]).reshape(1, -1)
    prediction = model_xgb.predict(features_array)[0]
    return {"predicted_cash_flow": float(prediction)}

