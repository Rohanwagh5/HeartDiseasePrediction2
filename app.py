import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('knn_model2.joblib')

# Set the page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
)

# Header
st.title("Heart Disease Prediction App")
st.markdown("""
Predict the likelihood of heart disease with this interactive tool. Use the sidebar to input the required medical information.
""")

# Sidebar for user input features
st.sidebar.header("Patient Details")
st.sidebar.write("Please provide the following details:")

# Define the feature input function
def user_input_features():
    age = st.sidebar.slider("Age", 29, 77, 50, help="Age of the patient.")
    sex = st.sidebar.radio("Sex", [1, 0], index=0, format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1, help="Type of chest pain experienced.")
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 94, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 126, 564, 250)
    fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
    restecg = st.sidebar.selectbox("Resting ECG Results (0-2)", [0, 1, 2], help="ECG results classification.")
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 71, 202, 150)
    exang = st.sidebar.radio("Exercise Induced Angina", [1, 0], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2], help="Slope category.")
    ca = st.sidebar.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 0)
    thal = st.sidebar.selectbox("Thalassemia Type", [0, 1, 2], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x])
    
    # Collecting all the features into a dictionary
    features = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    return features

# Main panel
st.header("Input Features")
st.write("Review your inputs on the sidebar before making a prediction.")

# Input features from the user
input_features = user_input_features()

# Add a predict button
if st.button("Predict", help="Click to get the prediction based on the entered data."):
    prediction = model.predict(input_features)
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("The patient is **likely** to have heart disease.")
    else:
        st.success("The patient is **unlikely** to have heart disease.")

# Footer with additional information
st.markdown("""
---
**Model:** predicts the likelihood of heart disease using an KNN model
""")
