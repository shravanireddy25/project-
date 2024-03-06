import streamlit as st
import pandas as pd
import pickle

# Load the pickled model
model = pickle.load(open(r"C:\Users\avani\sharvani\logistic_model.pkl", 'rb'))


# Function to make predictions
def predict_diagnosis(radius_mean, perimeter_mean, area_mean, symmetry_mean, compactness_mean, concave_points_mean):
    # Prepare the input data
    input_data = pd.DataFrame({
        'radius_mean': [radius_mean],
        'perimeter_mean': [perimeter_mean],
        'area_mean': [area_mean],
        'symmetry_mean': [symmetry_mean],
        'compactness_mean': [compactness_mean],
        'concave points_mean': [concave_points_mean]
    })
    # Make prediction
    prediction = model.predict(input_data)
    return prediction[0]


# Streamlit UI
def main():
    st.title("Breast Cancer Diagnosis Prediction")
    st.write("Enter the values for the features to predict the diagnosis:")

    # Input fields for user input
    radius_mean = st.sidebar.slider("Radius Mean", min_value=0.0, max_value=40.0, value=20.0)
    perimeter_mean = st.sidebar.slider("Perimeter Mean", min_value=0.0, max_value=300.0, value=150.0)
    area_mean = st.sidebar.slider("Area Mean", min_value=0.0, max_value=3000.0, value=1500.0)
    symmetry_mean = st.sidebar.slider("Symmetry Mean", min_value=0.0, max_value=2.0, value=1.0)
    compactness_mean = st.sidebar.slider("Compactness Mean", min_value=0.0, max_value=1.0, value=0.5)
    concave_points_mean = st.sidebar.slider("Concave Points Mean", min_value=0.0, max_value=1.0, value=0.5)

    # Predict the diagnosis
    if st.button("Predict"):
        prediction = predict_diagnosis(radius_mean, perimeter_mean, area_mean, symmetry_mean, compactness_mean,
                                       concave_points_mean)
        if prediction == 1:
            st.write("The diagnosis is malignant.")
        else:
            st.write("The diagnosis is benign.")


if __name__ == "__main__":
    main()
