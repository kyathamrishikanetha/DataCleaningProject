# =========================================
# CAR PRICE PREDICTION - TRAIN MODEL
# =========================================

# =========================================
# IMPORT LIBRARIES
# =========================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================
# LOAD DATASET
# =========================================

print("Loading dataset...")

df = pd.read_csv(
    "/Users/rishika/Desktop/DataCleaningProject/car_price.csv",
    encoding='latin1'
)

print("Dataset Loaded Successfully!")

# =========================================
# USE SMALLER SAMPLE (FASTER TRAINING)
# =========================================

df = df.sample(50000, random_state=42)

# =========================================
# SHOW DATA
# =========================================

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nMISSING VALUES:")
print(df.isnull().sum())

# =========================================
# CLEAN DATA
# =========================================

print("\nCleaning data...")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Remove rows where target is missing
df.dropna(subset=['sellingprice'], inplace=True)

# Fill missing values
for column in df.columns:

    # Numeric columns
    if df[column].dtype in ['int64', 'float64']:

        df[column] = df[column].fillna(df[column].median())

    # String columns
    else:

        df[column] = df[column].fillna("Unknown")

print("Data Cleaning Completed!")

# =========================================
# ENCODE CATEGORICAL COLUMNS
# =========================================

print("\nEncoding categorical columns...")

label_encoders = {}

for column in df.select_dtypes(include=['object', 'string']).columns:

    le = LabelEncoder()

    df[column] = le.fit_transform(df[column].astype(str))

    label_encoders[column] = le

print("Encoding Completed!")

# =========================================
# FEATURES AND TARGET
# =========================================

X = df.drop('sellingprice', axis=1)

y = df['sellingprice']

# =========================================
# SPLIT DATA
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# TRAIN MODEL
# =========================================

print("\nTraining model...")

model = RandomForestRegressor(
    n_estimators=20,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model Trained Successfully!")

# =========================================
# SAVE MODEL
# =========================================

joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")

# =========================================
# SAVE LABEL ENCODERS
# =========================================

joblib.dump(label_encoders, "label_encoders.pkl")

print("Label encoders saved as label_encoders.pkl")

# =========================================
# MAKE PREDICTIONS
# =========================================

print("\nMaking predictions...")

predictions = model.predict(X_test)

# =========================================
# EVALUATE MODEL
# =========================================

mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("\n=================================")
print("MODEL PERFORMANCE")
print("=================================")

print(f"Mean Absolute Error: {mae:.2f}")

print(f"R2 Score: {r2:.2f}")

# =========================================
# SHOW SAMPLE PREDICTIONS
# =========================================

results = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': predictions
})

print("\nSAMPLE PREDICTIONS:")
print(results.head())

# =========================================
# SAVE PREDICTIONS CSV
# =========================================

results.to_csv("car_price_predictions.csv", index=False)

print("\nPrediction file saved successfully!")

print("\nTRAINING COMPLETED SUCCESSFULLY!")