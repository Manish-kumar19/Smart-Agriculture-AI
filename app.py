
import streamlit as st
import tensorflow as tf
import numpy as np
import json

from PIL import Image

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Smart Agriculture AI",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 SMART AGRICULTURE AI SYSTEM")

st.markdown("""
### Crop Recommendation and Plant Disease Detection
**Using Artificial Intelligence and Deep Learning**
""")

# =====================================
# LOAD TRAINED MODELS
# =====================================

@st.cache_resource
def load_models():

    crop_model = tf.keras.models.load_model(
        "models/crop_model.keras"
    )

    disease_model = tf.keras.models.load_model(
        "models/disease_model.keras"
    )

    with open("models/crop_labels.json") as f:
        crop_labels = json.load(f)

    with open("models/disease_labels.json") as f:
        disease_labels = json.load(f)

    return (
        crop_model,
        disease_model,
        crop_labels,
        disease_labels
    )


try:
    crop_model, disease_model, crop_labels, disease_labels = (
        load_models()
    )

except Exception:
    st.error(
        "Models not found. First run: python train.py"
    )
    st.stop()


# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("🌾 SMART AGRICULTURE")

menu = st.sidebar.selectbox(
    "Choose Module",
    [
        "Home",
        "Crop Recommendation",
        "Plant Disease Detection"
    ]
)


# =====================================
# HOME PAGE
# =====================================

if menu == "Home":

    st.header("Welcome to Smart Agriculture AI")

    st.write("""
    This intelligent agriculture application
    provides two important features.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌱 Crop Recommendation")
        st.write("""
        Recommends the best crop based on:
        - Nitrogen
        - Phosphorus
        - Potassium
        - Temperature
        - Humidity
        - Soil pH
        - Rainfall
        """)

    with col2:
        st.subheader("🍃 Disease Detection")
        st.write("""
        Detects possible plant diseases
        from uploaded leaf images using CNN.
        """)

    st.success("AI Agriculture System Ready!")


# =====================================
# CROP RECOMMENDATION
# =====================================

elif menu == "Crop Recommendation":

    st.header("🌱 Crop Recommendation System")

    st.write(
        "Enter soil and weather information below."
    )

    col1, col2 = st.columns(2)

    with col1:

        N = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            value=90.0
        )

        P = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            value=42.0
        )

        K = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            value=43.0
        )

        temperature = st.number_input(
            "Temperature (°C)",
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
            value=200.0
        )

    if st.button("🌾 Recommend Crop"):

        input_data = np.array([[
            N, P, K,
            temperature,
            humidity,
            ph,
            rainfall
        ]], dtype=np.float32)

        prediction = crop_model.predict(
            input_data,
            verbose=0
        )

        index = np.argmax(prediction)

        crop = crop_labels[index]

        confidence = float(
            prediction[0][index] * 100
        )

        st.success(
            f"Recommended Crop: {crop.upper()}"
        )

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        st.info(
            "Validate crop suitability with local soil "
            "and agricultural recommendations."
        )


# =====================================
# PLANT DISEASE DETECTION
# =====================================

elif menu == "Plant Disease Detection":

    st.header("🍃 Plant Disease Detection")

    st.write(
        "Upload a plant leaf image to detect its category."
    )

    uploaded_file = st.file_uploader(
        "Upload Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Plant Leaf",
            width=350
        )

        if st.button("🔍 Detect Disease"):

            image = image.resize((224, 224))

            image_array = np.array(
                image,
                dtype=np.float32
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            prediction = disease_model.predict(
                image_array,
                verbose=0
            )

            index = np.argmax(prediction)

            disease = disease_labels[index]

            confidence = float(
                prediction[0][index] * 100
            )

            st.success(
                f"Predicted Class: {disease}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            if "healthy" in disease.lower():

                st.info(
                    "The leaf is classified as healthy."
                )

            else:

                st.warning(
                    "Possible disease detected. "
                    "Please confirm with an agricultural expert."
                )

# =====================================
# FOOTER
# =====================================

st.sidebar.markdown("---")

st.sidebar.caption(
    "Smart Agriculture AI | Deep Learning Project"
)
