
import streamlit as st
import joblib
import numpy as np

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Predictive Maintenance System")

rpm = st.number_input("Engine RPM")
oil_pressure = st.number_input("Lub Oil Pressure")
fuel_pressure = st.number_input("Fuel Pressure")
coolant_pressure = st.number_input("Coolant Pressure")
oil_temp = st.number_input("Lub Oil Temperature")
coolant_temp = st.number_input("Coolant Temperature")

if st.button("Predict"):

    data = np.array([
        rpm,
        oil_pressure,
        fuel_pressure,
        coolant_pressure,
        oil_temp,
        coolant_temp
    ]).reshape(1,-1)

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.error("Engine Requires Maintenance")
    else:
        st.success("Engine Operating Normally")
