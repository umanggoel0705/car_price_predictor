import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import json

app = Flask(__name__)
model = pickle.load(open("model.pkl", 'rb'))
scaler = pickle.load(open("scaling.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    name = request.form.get("name")
    year = request.form.get("year")
    km_driven = request.form.get("km_driven")
    fuel = request.form.get("fuel")
    fuel_petrol = 0.0
    fuel_diesel = 0.0
    fuel_cng = 0.0
    fuel_lpg = 0.0
    if fuel == "diesel":
        fuel_diesel = 1.0
    elif fuel == "petrol":
        fuel_petrol = 1.0
    elif fuel == "cng":
        fuel_cng = 1.0
    elif fuel == "lpg":
        fuel_lpg = 1.0
    
    seller_type = request.form.get("seller_type")
    dealer = 0.0
    individual = 0.0
    trustmark_dealer = 0.0
    if seller_type == "dealer":
        dealer = 1.0
    elif seller_type == "individual":
        individual = 1.0
    elif seller_type == "trustmark_dealer":
        trustmark_dealer = 1.0
    
    transmission = request.form.get("transmission")
    manual = 0.0
    automatic = 0.0
    if transmission == "manual":
        manual = 1.0
    else:
        automatic = 1.0

    owner = request.form.get("owner")
    first = 0.0
    second = 0.0
    third = 0.0
    fourth = 0.0
    test_drive = 0.0
    if owner == "first":
        first = 1.0
    if owner == "second":
        second = 1.0
    if owner == "third":
        third = 1.0
    if owner == "fourth":
        fourth = 1.0
    if owner == "test_drive":
        test_drive = 1.0

    mileage = request.form.get("mileage")
    engine = request.form.get("engine")
    max_power = request.form.get("max_power")
    seats = request.form.get("seats")

    user_inp = [year, km_driven, fuel_diesel, fuel_petrol, fuel_cng, fuel_lpg, dealer, individual, trustmark_dealer, manual, automatic, first, second, third, fourth, test_drive, mileage, engine, max_power, seats]

    X_user = pd.DataFrame(np.array(user_inp).reshape(1,-1).tolist())
    X_user = scaler.transform(X_user)
    output = model.predict(X_user)[0]

    # print(output)
    return render_template("home.html", prediction="The predicted price is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)