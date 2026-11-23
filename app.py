
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌱",
    layout="wide"
)

# --------------------------------
# LOAD TRAINED MODELS
# --------------------------------

@st.cache_resource
def load_models():

    crop_model = joblib.load(
        "models/crop_model.pkl"
    )

    disease_model = tf.keras.models.load_model(
        "models/potato_disease_model.keras"
    )

    with open("models/disease_classes.json", "r") as file:
        classes = json.load(file)

    return crop_model, disease_model, classes


crop_model, disease_model, disease_classes = load_models()

# --------------------------------
# MAIN HEADER
# --------------------------------

st.title("🌱 Smart Agriculture AI System")

st.write(
    "An integrated Artificial Intelligence system "
    "for crop recommendation and potato leaf disease detection."
)

st.divider()

# --------------------------------
# SIDEBAR NAVIGATION
# --------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose an option",
    [
        "Home",
        "Crop Recommendation",
        "Potato Disease Detection"
    ]
)

# --------------------------------
# HOME PAGE
# --------------------------------

if page == "Home":

    st.header("Welcome to Smart Agriculture AI")

    st.write(
        "This application combines Machine Learning "
        "and Deep Learning to support agricultural decisions."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🌾 Crop Recommendation")

        st.write(
            "Enter soil nutrients and environmental "
            "conditions to receive a crop recommendation."
        )

    with col2:

        st.subheader("🥔 Potato Disease Detection")

        st.write(
            "Upload a potato leaf image to identify "
            "Early Blight, Late Blight, or Healthy leaves."
        )

    st.divider()

    st.success(
        "Both AI models are loaded and ready to use!"
    )

# --------------------------------
# CROP RECOMMENDATION PAGE
# --------------------------------

elif page == "Crop Recommendation":

    st.header("🌾 Crop Recommendation System")

    st.write(
        "Enter the soil and climate information below."
    )

    col1, col2 = st.columns(2)

    with col1:

        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            max_value=200.0,
            value=90.0
        )

        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            max_value=200.0,
            value=42.0
        )

        potassium = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            max_value=250.0,
            value=43.0
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=25.0
        )

    with col2:

        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0
        )

        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5
        )

        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=5000.0,
            value=200.0
        )

    if st.button("Recommend Crop", type="primary"):

        input_data = pd.DataFrame(
            [[
                nitrogen,
                phosphorus,
                potassium,
                temperature,
                humidity,
                ph,
                rainfall
            ]],
            columns=[
                "N",
                "P",
                "K",
                "temperature",
                "humidity",
                "ph",
                "rainfall"
            ]
        )

        prediction = crop_model.predict(input_data)

        st.success(
            f"Recommended Crop: {prediction[0].upper()}"
        )

        # Show prediction probabilities
        probabilities = crop_model.predict_proba(input_data)[0]

        results = pd.DataFrame({
            "Crop": crop_model.classes_,
            "Probability (%)": probabilities * 100
        })

        results = results.sort_values(
            by="Probability (%)",
            ascending=False
        )

        st.subheader("Top Crop Predictions")

        st.dataframe(
            results.head(5).style.format(
                {"Probability (%)": "{:.2f}%"}
            ),
            use_container_width=True
        )

# --------------------------------
# POTATO DISEASE DETECTION PAGE
# --------------------------------

elif page == "Potato Disease Detection":

    st.header("🥔 Potato Leaf Disease Detection")

    st.write(
        "Upload a clear image of a potato leaf."
    )

    uploaded_file = st.file_uploader(
        "Upload Potato Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Potato Leaf",
            width=350
        )

        if st.button("Detect Disease", type="primary"):

            # Resize image
            image = image.resize((224, 224))

            # Convert to array
            image_array = np.array(image)

            # Expand dimensions
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # Preprocess image
            image_array = preprocess_input(
                image_array.astype(np.float32)
            )

            # Model prediction
            predictions = disease_model.predict(
                image_array,
                verbose=0
            )[0]

            predicted_index = np.argmax(predictions)

            disease = disease_classes[predicted_index]

            confidence = predictions[predicted_index] * 100

            st.subheader("Prediction Result")

            st.success(
                f"Detected Class: {disease.replace('Potato___', '').replace('_', ' ').title()}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            # Treatment recommendations
            treatments = {

                "Potato___Early_blight": [
                    "Remove heavily affected leaves.",
                    "Avoid overhead irrigation where possible.",
                    "Maintain good plant spacing and airflow.",
                    "Use a locally registered fungicide if advised by an agricultural expert."
                ],

                "Potato___Late_blight": [
                    "Inspect nearby potato plants for symptoms.",
                    "Remove infected plant material safely.",
                    "Avoid working in wet foliage.",
                    "Seek immediate local agricultural advice about registered late blight control measures."
                ],

                "Potato___healthy": [
                    "No visible disease detected by the model.",
                    "Continue regular crop monitoring.",
                    "Maintain proper irrigation and balanced nutrition."
                ]
            }

            st.subheader("Recommended Actions")

            for treatment in treatments.get(
                disease,
                ["Consult an agricultural expert."]
            ):

                st.write("✅", treatment)

            # Display all class probabilities
            st.subheader("Prediction Probabilities")

            probability_df = pd.DataFrame({
                "Disease": [
                    c.replace("Potato___", "").replace("_", " ")
                    for c in disease_classes
                ],
                "Confidence (%)": predictions * 100
            })

            st.bar_chart(
                probability_df.set_index("Disease")
            )

            if confidence < 70:

                st.warning(
                    "Low prediction confidence. Please upload "
                    "a clearer leaf image and seek expert confirmation."
                )

st.divider()

st.caption(
    "Smart Agriculture AI | Crop Recommendation "
    "+ Potato Leaf Disease Detection"
)
