import io
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import requests
from fpdf import FPDF
from huggingface_hub import hf_hub_download
import tensorflow as tf
import os
import json
import time

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="🌾FARMDOC",
    layout="wide"
)

# ==========================
# BACKGROUND
# ==========================
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #1e1e2f 0%, #2e2e3f 100%) no-repeat center center fixed;
            background-size: cover;
        }
        .block-container {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border-radius: 20px;
            padding: 25px !important;
        }
        h1, h2, h3, h4, h5, h6, p, div, span {
            color: white !important;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #34a853 !important;
            color: white !important;
            font-size: 18px !important;
            border-radius: 12px !important;
            padding: 10px 25px !important;
        }
        video {
            transform: scaleX(-1) !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# ==========================
# LOAD MODEL
# ==========================
st.title("🌱 FarmDoc")
st.write("A simple tool to detect plant diseases and get easy-to-understand treatment advice using AI.")

model_path = hf_hub_download(
    repo_id="qwertymaninwork/Plant_Disease_Detection_System",
    filename="mobilenetv2_plant.h5"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

try:
    model = load_model()
    CLASS_NAMES = [
        'HEALTHY MILLET', 'HEALTHY POTATO', 'HEALTHY RICE', 'HEALTHY SUGARCANE',
        'HEALTHY TEA LEAF', 'HEALTHY TOMATO', 'HEALTHY WHEAT', 'MILLETS BLAST',
        'MILLETS RUST', 'POTATO EARLY BLIGHT', 'POTATO LATE BLIGHT',
        'RICE BACTERIAL BLIGHT', 'RICE BROWN SPOT', 'RICE LEAF SMUT',
        'SUGARCANE RED ROT', 'SUGARCANE RUST', 'SUGARCANE YELLOW',
        'TEA GRAY BLIGHT', 'TEA GREEN MIRID BUG', 'TEA HELOPELTIS',
        'TOMATO LEAF MOLD', 'TOMATO MOSAIC VIRUS', 'TOMATO SEPTORIA LEAF SPOT',
        'WHEAT BROWN RUST', 'WHEAT LOOSE SMUT', 'WHEAT YELLOW RUST'
    ]
except Exception as e:
    st.warning(f"⚠ Could not load model: {e}")
    model = None
    CLASS_NAMES = []

# ==========================
# SESSION STATE INIT (FIX ADDED)
# ==========================
if "report_text" not in st.session_state:
    st.session_state.report_text = ""

# ==========================
# SENSOR DATA
# ==========================
THINGSPEAK_CHANNEL_ID = "3152731"
READ_API_KEY = "8WGWK6AUAF74H6DJ"

def fetch_sensor_data():
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if "feeds" in data and len(data["feeds"]) > 0:
            latest = data["feeds"][0]
            return {
                "temperature": latest["field1"],
                "humidity": latest["field2"],
                "soil_moisture": latest["field3"],
                "timestamp": latest["created_at"]
            }
    except Exception:
        pass
    return {"temperature": None, "humidity": None, "soil_moisture": None, "timestamp": None}

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ["About", "AI Detection Panel"])

# ==========================
# ABOUT PAGE
# ==========================
if page == "About":
    st.header("About FarmDoc AI")
    st.markdown("""
    The FarmDoc AI helps farmers detect plant diseases using their phone’s camera or uploaded images.

    It also gives simple, clear advice on:
    - What the disease is  
    - How it affects the crop  
    - What actions to take  
    - How to prevent it in the future  

    It connects with your farm sensors (ESP32 + ThingSpeak) to include weather and soil data in your report.

    Take a photo → Let AI detect → Get your farm report.
    """)

# ==========================
# AI DETECTION PANEL
# ==========================
elif page == "AI Detection Panel":
    st.header("Step 1: Capture or Upload Plant Image")

    api_key = st.sidebar.text_input("🔐 Enter your OpenRouter API key (starts with sk-or-...)", type="password")

    uploaded_file = st.camera_input("📸 Take a photo of your crop leaf")
    if uploaded_file is None:
        uploaded_file = st.file_uploader("Or upload a leaf image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🪴 This is the captured image being analyzed", use_column_width=True)

        if model:
            img_resized = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            confidence = np.max(preds)
            predicted_class = CLASS_NAMES[np.argmax(preds)]
            st.session_state.predicted_class = predicted_class
            st.session_state.confidence = confidence

            st.success(f"🌿 The AI detected: {predicted_class} with {confidence*100:.2f}% confidence.")

    # ==========================
    # SENSOR DATA DISPLAY
    # ==========================
    st.header("🌡 Step 2: Check Live Farm Data")
    count = st_autorefresh(interval=5000, limit=None, key="sensor_refresh")

    sensor = fetch_sensor_data()
    if sensor["temperature"]:
        col1, col2, col3 = st.columns(3)
        col1.metric("🌡 Temperature", f"{sensor['temperature']} °C")
        col2.metric("💧 Humidity", f"{sensor['humidity']} %")
        col3.metric("🌱 Soil Moisture", f"{sensor['soil_moisture']} %")
        st.caption(f"Last updated: {sensor['timestamp']}")
    else:
        st.warning("Waiting for live data from your farm sensors...")

    # ==========================
    # AI REPORT GENERATION
    # ==========================
    st.header("Step 3: Get AI Farm Report")

    if st.button("🧾 Generate Farm Report"):
        if not api_key:
            st.error("Please enter your OpenRouter API key in the sidebar.")
        elif not uploaded_file:
            st.error("Please upload or take a photo first.")
        elif model is None:
            st.error("AI model not loaded.")
        else:
            with st.spinner("The AI is writing your report in simple farmer language..."):
                prompt = f"""
                You are a helpful agricultural assistant speaking to a farmer.
                Write a clear, short, and easy-to-understand farm report using simple words.
                Explain what disease was found: {st.session_state.predicted_class} (confidence {st.session_state.confidence*100:.2f}%).

                Farm conditions:
                - Temperature: {sensor['temperature']} °C
                - Humidity: {sensor['humidity']} %
                - Soil Moisture: {sensor['soil_moisture']} %
                """

                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                data = {
                    "model": "meta-llama/llama-3.1-8b-instruct",
                    "messages": [
                        {"role": "system", "content": "You are a friendly farm advisor speaking in simple words."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 600,
                    "temperature": 0.7
                }

                try:
                    response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                             headers=headers, json=data, timeout=60)
                    result = response.json()
                    full_text = result["choices"][0]["message"]["content"]

                    st.session_state.report_text = full_text  # ✅ stored permanently
                    st.success("✅ Report generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating report: {e}")

    # ==========================
    # SHOW REPORT (PERSISTS AFTER REFRESH)
    # ==========================
    if st.session_state.report_text:
        st.markdown("### 🌿 Your Farm Report")
        st.write(st.session_state.report_text)

        # PDF generation
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Easy Farm Report", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, st.session_state.report_text)

        if uploaded_file:
            temp_img_path = "temp_image.jpg"
            with open(temp_img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf.image(temp_img_path, x=10, y=None, w=100)

        pdf_bytes = pdf.output(dest='S').encode('latin-1')

        st.download_button(
            "📥 Download Simple Report (PDF)",
            data=pdf_bytes,
            file_name="farm_report.pdf",
            mime="application/pdf"
        )


# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("FarmDoc © 2025 — Helping Farmers Grow Smarter")
