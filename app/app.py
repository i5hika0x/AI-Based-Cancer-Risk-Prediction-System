import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.predict import predict

# Page configuration
st.set_page_config(page_title="Cancer AI System", layout="centered")

st.title("Multi-Cancer Prediction System")
st.markdown("AI-powered cancer risk analysis using tumor features")
st.markdown("---")

# Cancer type selection
cancer_type = st.selectbox(
    "Select Cancer Type",
    ["Breast", "Lung", "Oesophagus"]
)

# Input mode
mode = st.radio("Select Input Mode:", ["Manual Input", "Upload CSV"])

# =========================
# MANUAL INPUT
# =========================
if mode == "Manual Input":

    st.subheader("Enter Tumor Details")

    col1, col2 = st.columns(2)

    with col1:
        radius = st.number_input("Mean Radius", 0.0, 30.0)
        texture = st.number_input("Mean Texture", 0.0, 40.0)

    with col2:
        area = st.number_input("Mean Area", 0.0, 2500.0)

    input_data = np.zeros(30)
    input_data[0] = radius
    input_data[1] = texture
    input_data[3] = area

    if st.button("Run Prediction"):

        pred, malignant, benign = predict(input_data)

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)
        col1.metric("Malignant Probability", f"{malignant*100:.2f}%")
        col2.metric("Benign Probability", f"{benign*100:.2f}%")

        # Graph
        st.subheader("Probability Chart")

        labels = ["Malignant", "Benign"]
        values = [malignant, benign]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylabel("Probability")
        ax.set_title("Cancer Prediction Confidence")

        st.pyplot(fig)

        # Diagnosis
        st.subheader("Diagnosis")

        st.write(f"Selected Cancer Type: {cancer_type}")

        if pred == 0:
            st.error(f"High Risk of {cancer_type} Cancer")
        else:
            st.success(f"Low Risk of {cancer_type} Cancer")

        st.info(
            "This prediction is based on tumor feature analysis using a trained machine learning model."
        )

        # Doctor search
        st.subheader("Find Specialists")

        search_query = f"{cancer_type} cancer specialist near me"
        google_maps_url = f"https://www.google.com/maps/search/{search_query.replace(' ', '+')}"

        st.markdown(f"[Search Specialists Near You]({google_maps_url})")

# =========================
# CSV INPUT
# =========================
elif mode == "Upload CSV":

    st.subheader("Upload CSV File with 30 Features")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(data)

        if st.button("Run Prediction on File"):

            row = data.iloc[0].values

            pred, malignant, benign = predict(row)

            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)
            col1.metric("Malignant Probability", f"{malignant*100:.2f}%")
            col2.metric("Benign Probability", f"{benign*100:.2f}%")

            # Graph
            st.subheader("Probability Chart")

            labels = ["Malignant", "Benign"]
            values = [malignant, benign]

            fig, ax = plt.subplots()
            ax.bar(labels, values)
            ax.set_ylabel("Probability")
            ax.set_title("Cancer Prediction Confidence")

            st.pyplot(fig)

            # Diagnosis
            st.subheader("Diagnosis")

            st.write(f"Selected Cancer Type: {cancer_type}")

            if pred == 0:
                st.error(f"High Risk of {cancer_type} Cancer")
            else:
                st.success(f"Low Risk of {cancer_type} Cancer")

            st.info(
                "This prediction is based on tumor feature analysis using a trained machine learning model."
            )

            # Doctor search
            st.subheader("Find Specialists")

            search_query = f"{cancer_type} cancer specialist near me"
            google_maps_url = f"https://www.google.com/maps/search/{search_query.replace(' ', '+')}"

            st.markdown(f"[Search Specialists Near You]({google_maps_url})")