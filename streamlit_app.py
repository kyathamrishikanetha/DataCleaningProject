import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("Car Price Prediction App")

# User Inputs
year = st.number_input("Year", 2000, 2025, 2018)

make = st.text_input("Make", "Toyota")

model_name = st.text_input("Model", "Camry")

trim = st.text_input("Trim", "SE")

body = st.text_input("Body", "Sedan")

transmission = st.text_input("Transmission", "automatic")

vin = st.text_input("VIN", "TESTVIN123")

state = st.text_input("State", "ca")

condition = st.number_input("Condition", 0.0, 5.0, 4.5)

odometer = st.number_input("Odometer", 0, 300000, 45000)

color = st.text_input("Color", "white")

interior = st.text_input("Interior", "black")

seller = st.text_input("Seller", "dealer")

mmr = st.number_input("MMR", 0, 100000, 18000)

saledate = st.text_input("Sale Date", "2024-01-01")

# Predict Button
if st.button("Predict Price"):

    new_car = pd.DataFrame([{
        'year': year,
        'make': make,
        'model': model_name,
        'trim': trim,
        'body': body,
        'transmission': transmission,
        'vin': vin,
        'state': state,
        'condition': condition,
        'odometer': odometer,
        'color': color,
        'interior': interior,
        'seller': seller,
        'mmr': mmr,
        'saledate': saledate
    }])

    # Encode categorical columns
    for column in new_car.columns:

        if column in label_encoders:

            le = label_encoders[column]

            new_car[column] = new_car[column].astype(str)

            new_car[column] = new_car[column].apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )

            new_car[column] = le.transform(new_car[column])

    # Prediction
    prediction = model.predict(new_car)

    st.success(f"Predicted Car Price: ${prediction[0]:,.2f}")