import streamlit as st
import pandas as pd
import pickle

# ===============================
# Page Configuration
# ===============================

st.set_page_config(
    page_title="Food Delivery Prediction",
    page_icon="🍔",
    layout="centered"
)

# ===============================
# Load Model
# ===============================

model = pickle.load(open("model.pkl", "rb"))

# ===============================
# Title
# ===============================

st.title("🍔 Food Delivery Prediction")

st.markdown(
    "### Predict whether the delivery will be **On Time** or **Late**"
)

st.markdown("---")

# ===============================
# Sidebar
# ===============================

st.sidebar.title("📌 About Project")

st.sidebar.info("""
This project uses **Logistic Regression**
to predict whether a food order will be
delivered **On Time** or **Late**.

Developed using:

✅ Python

✅ Scikit-Learn

✅ Streamlit

✅ Pandas

✅ Machine Learning
""")

# ===============================
# Input Form
# ===============================

distance = st.slider(
    "🚚 Delivery Distance (KM)",
    1.0,
    15.0,
    5.0
)

weather = st.selectbox(
    "🌥️ Weather",
    ["Sunny", "Cloudy", "Rainy"]
)

traffic = st.selectbox(
    "🚦 Traffic",
    ["Low", "Medium", "High"]
)

rating = st.slider(
    "⭐ Rider Rating",
    1.0,
    5.0,
    4.0
)

prep = st.slider(
    "🍕 Preparation Time (Minutes)",
    10,
    35,
    15
)

vehicle = st.selectbox(
    "🛵 Vehicle",
    ["Bike", "Scooter"]
)

st.markdown("---")

# ===============================
# Encoding
# ===============================

weather_map = {
    "Cloudy": 0,
    "Rainy": 1,
    "Sunny": 2
}

traffic_map = {
    "High": 0,
    "Low": 1,
    "Medium": 2
}

vehicle_map = {
    "Bike": 0,
    "Scooter": 1
}

# ===============================
# Create Input DataFrame
# ===============================

input_data = pd.DataFrame({
    "Delivery_Distance": [distance],
    "Weather": [weather_map[weather]],
    "Traffic": [traffic_map[traffic]],
    "Rider_Rating": [rating],
    "Preparation_Time": [prep],
    "Vehicle": [vehicle_map[vehicle]]
})

# ===============================
# Prediction
# ===============================

if st.button("🔍 Predict Delivery Status"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    confidence = max(probability) * 100

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success("✅ Delivery will be On Time")

    else:

        st.error("❌ Delivery will be Late")

    st.metric(
        label="Prediction Confidence",
        value=f"{confidence:.2f}%"
    )

# ===============================
# Footer
# ===============================

st.markdown("---")

st.caption("Machine Learning Project | Logistic Regression | Streamlit")
