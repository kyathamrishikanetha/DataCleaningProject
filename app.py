# =========================================
# CAR PRICE PREDICTION WEB APP
# app.py
# =========================================

from flask import Flask, render_template, request
import pandas as pd
import joblib

# =========================================
# LOAD MODEL & ENCODERS
# =========================================

model = joblib.load("model.pkl")

label_encoders = joblib.load("label_encoders.pkl")

# =========================================
# CREATE FLASK APP
# =========================================

app = Flask(__name__)

# =========================================
# HOME PAGE
# =========================================

@app.route('/')

def home():
    return render_template("index.html")

# =========================================
# PREDICTION ROUTE
# =========================================

@app.route('/predict', methods=['POST'])

def predict():

    # Get form data
    year = int(request.form['year'])
    make = request.form['make']
    model_name = request.form['model']
    trim = request.form['trim']
    body = request.form['body']
    transmission = request.form['transmission']
    vin = request.form['vin']
    state = request.form['state']
    condition = float(request.form['condition'])
    odometer = float(request.form['odometer'])
    color = request.form['color']
    interior = request.form['interior']
    seller = request.form['seller']
    mmr = float(request.form['mmr'])
    saledate = request.form['saledate']

    # Create dataframe
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

            # Handle unknown values
            new_car[column] = new_car[column].apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )

            new_car[column] = le.transform(new_car[column])

    # Predict price
    prediction = model.predict(new_car)

    predicted_price = round(prediction[0], 2)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Car Price: ${predicted_price}"
    )

# =========================================
# RUN APP
# =========================================

if __name__ == "__main__":

    app.run(debug=True)