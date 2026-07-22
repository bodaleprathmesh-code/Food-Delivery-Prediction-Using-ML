# import streamlit as st
# import pickle
# import numpy as np

# # ==============================
# # Load Model
# # ==============================

# model = pickle.load(open("model.pkl", "rb"))

# # ==============================
# # Page Configuration
# # ==============================

# st.set_page_config(
#     page_title="Food Delivery Prediction",
#     page_icon="🍕",
#     layout="wide"
# )

# # ==============================
# # Sidebar
# # ==============================

# st.sidebar.title("🍕 Food Delivery Prediction")

# st.sidebar.write("""
# ### Machine Learning Project

# This project predicts whether
# the food delivery will be

# ✅ On Time

# or

# ❌ Late

# Model Used:
# - Logistic Regression
# """)

# # ==============================
# # Title
# # ==============================

# st.title("🍕 Food Delivery Prediction System")

# st.write("Fill all the details and click **Predict**.")

# st.markdown("---")

# # ==============================
# # Input Fields
# # ==============================

# col1, col2 = st.columns(2)

# with col1:

#     distance = st.number_input(
#         "Delivery Distance (KM)",
#         min_value=1.0,
#         max_value=50.0,
#         value=5.0
#     )

#     weather = st.selectbox(
#         "Weather",
#         ["Cloudy", "Rainy", "Sunny"]
#     )

#     rating = st.slider(
#         "Rider Rating",
#         1.0,
#         5.0,
#         4.0
#     )

# with col2:

#     traffic = st.selectbox(
#         "Traffic",
#         ["High", "Low", "Medium"]
#     )

#     prep_time = st.slider(
#         "Preparation Time (Minutes)",
#         5,
#         60,
#         20
#     )

#     vehicle = st.selectbox(
#         "Vehicle",
#         ["Bike", "Scooter"]
#     )

# # ==============================
# # Label Encoding Mapping
# # ==============================

# weather_map = {
#     "Cloudy": 0,
#     "Rainy": 1,
#     "Sunny": 2
# }

# traffic_map = {
#     "High": 0,
#     "Low": 1,
#     "Medium": 2
# }

# vehicle_map = {
#     "Bike": 0,
#     "Scooter": 1
# }

# weather = weather_map[weather]
# traffic = traffic_map[traffic]
# vehicle = vehicle_map[vehicle]

# # ==============================
# # Predict Button
# # ==============================

# if st.button("🚀 Predict Delivery Status"):

#     input_data = np.array([[
#         distance,
#         weather,
#         traffic,
#         rating,
#         prep_time,
#         vehicle
#     ]])

#     prediction = model.predict(input_data)

#     st.markdown("---")

#     st.subheader("Prediction Result")

#     if prediction[0] == 1:

#         st.success("✅ Delivery Status : ON TIME")

#         st.balloons()

#     else:

#         st.error("❌ Delivery Status : LATE")

# # ==============================
# # Footer
# # ==============================

# st.markdown("---")

# st.caption("Developed using Python | Scikit-Learn | Streamlit")






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