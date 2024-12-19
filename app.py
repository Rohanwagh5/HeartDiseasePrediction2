import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('knn_model2.joblib')

# Set the page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤",
    layout="wide",
)

# Main page header
st.title("Heart Disease Prediction App")
st.markdown(
    """
    Predict the likelihood of heart disease with this interactive tool. 
    Please provide the required medical information below.
    """
)

# Sidebar for user input features
def user_input_features():
    st.sidebar.subheader("Patient Details")

    age = st.sidebar.number_input(
        "Age (Years)", min_value=29, max_value=77, value=50, step=1, help="Age of the patient."
    )
    sex = st.sidebar.radio(
        "Sex", [1, 0], index=0, format_func=lambda x: "Male" if x == 1 else "Female"
    )
    cp = st.sidebar.select_slider(
        "Chest Pain Type (0-3)", options=[0, 1, 2, 3], value=1, help="Type of chest pain experienced."
    )
    trestbps = st.sidebar.number_input(
        "Resting Blood Pressure (mm Hg)", min_value=94, max_value=200, value=120, step=1
    )
    chol = st.sidebar.number_input(
        "Serum Cholesterol (mg/dl)", min_value=126, max_value=564, value=250, step=1
    )
    fbs = st.sidebar.radio(
        "Fasting Blood Sugar > 120 mg/dl", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No"
    )
    restecg = st.sidebar.selectbox(
        "Resting ECG Results (0-2)", [0, 1, 2], help="ECG results classification."
    )
    thalach = st.sidebar.number_input(
        "Maximum Heart Rate Achieved", min_value=71, max_value=202, value=150, step=1
    )
    exang = st.sidebar.radio(
        "Exercise Induced Angina", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No"
    )
    oldpeak = st.sidebar.slider(
        "ST Depression Induced by Exercise", min_value=0.0, max_value=6.2, value=1.0, step=0.1
    )
    slope = st.sidebar.selectbox(
        "Slope of the Peak Exercise ST Segment", [0, 1, 2], help="Slope category."
    )
    ca = st.sidebar.select_slider(
        "Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4], value=0
    )
    thal = st.sidebar.selectbox(
        "Thalassemia Type", [0, 1, 2], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x]
    )

    # Collecting all the features into a dictionary
    features = np.array([
        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    ]).reshape(1, -1)
    return features

# Input features from the user
input_features = user_input_features()

# Main page Predict Button
st.subheader("Prediction")
st.write("Click the button below to predict the likelihood of heart disease based on the provided inputs.")

if st.button("Predict", help="Click to get the prediction based on the entered data."):
    prediction = model.predict(input_features)
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("The patient is *likely* to have heart disease.")
    else:
        st.success("The patient is *unlikely* to have heart disease.")

# Footer with additional information
st.markdown(
    """
---
*Model:* predicts the likelihood of heart disease using a KNN model
"""
)
