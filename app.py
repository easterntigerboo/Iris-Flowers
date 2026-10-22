import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Iris Flower Predictor", page_icon="🌸", layout="centered")

@st.cache_resource
def load_model():
    with open("iris_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

st.title("🌸 Iris Flower Predictor")
st.write("Enter the flower measurements below to predict its Iris species.")

st.subheader("Flower measurements (cm)")
sepal_length = st.number_input("Sepal length", min_value=0.1, max_value=10.0, value=5.1, step=0.1)
sepal_width = st.number_input("Sepal width", min_value=0.1, max_value=10.0, value=3.5, step=0.1)
petal_length = st.number_input("Petal length", min_value=0.1, max_value=10.0, value=1.4, step=0.1)
petal_width = st.number_input("Petal width", min_value=0.1, max_value=10.0, value=0.2, step=0.1)

if st.button("Predict Iris Species", type="primary"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted species: **{prediction.capitalize()}**")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        st.subheader("Model confidence estimates")
        for species, probability in zip(model.classes_, probabilities):
            st.write(f"{species.capitalize()}: {probability:.1%}")
            st.progress(float(probability))

st.caption("Educational demo using a Random Forest classifier trained on the classic Iris dataset.")
