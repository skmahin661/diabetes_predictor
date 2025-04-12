import streamlit as st
import pickle
import numpy as np

# Load saved model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🩺 Diabetes Prediction System")
st.markdown("Enter your health information to predict the risk of diabetes.")

# Input fields
preg = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

# Predict button
if st.button("Predict"):
    user_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    user_data_scaled = scaler.transform(user_data)
    result = model.predict(user_data_scaled)

    if result[0] == 1:
        st.error("⚠️ You are likely to have diabetes.")
    else:
        st.success("✅ You are unlikely to have diabetes.")
