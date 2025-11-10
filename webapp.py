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
st.set_page_config(page_title="🌾 FARMDOC AI", layout="wide")

# ==========================
# BACKGROUND & STYLE
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
# TITLE
# ==========================
st.title("🌱 Smart Farm Doctor AI")
st.write("AI-powered **plant species & disease detection** with live farm data and simple report generation.")

# ==========================
# LOAD LOCAL MODEL
# ==========================
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="qwertymaninwork/Plant_Disease_Detection_System",
        filename="mobilenetv2_plant.h5"
    )
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
    st.warning(f"⚠️ Model failed to load: {e}")
    model = None
    CLASS_NAMES = []

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
                "temperature": latest.get("field1", "N/A"),
                "humidity": latest.get("field2", "N/A"),
                "soil_moisture": latest.get("field3", "N/A"),
                "timestamp": latest["created_at"]
            }
    except Exception:
        pass
    return {"temperature": None, "humidity": None, "soil_moisture": None, "timestamp": None}

# ==========================
# SIDEBAR MENU
# ==========================
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ["About", "AI Detection Panel"])

# ==========================
# ABOUT PAGE
# ==========================
if page == "About":
    st.header("🌾 About Smart Farm Doctor")
    st.markdown("""
    **Smart Farm Doctor** combines multiple AIs:
    - 🌿 **PlantNet API** → Identifies plant species  
    - 🧠 **TensorFlow AI** → Detects crop diseases  
    - 🌡 **ThingSpeak IoT** → Reads real-time farm sensor data  
    - ✍️ **Llama 3.1 (OpenRouter)** → Writes simple farmer-friendly reports  

    📷 Take a photo → 🧠 AI analyzes → 📋 Get a smart, easy-to-read report.
    """)

# ==========================
# AI DETECTION PANEL
# ==========================
elif page == "AI Detection Panel":
    st.header("🧠 Step 1: Capture or Upload Plant Image")

    plantnet_key = st.sidebar.text_input("🔑 Enter PlantNet API Key", type="password")
    api_key = st.sidebar.text_input("🔐 Enter OpenRouter API Key (sk-or-...)", type="password")

    uploaded_file = st.camera_input("📸 Take a photo of your crop leaf")
    if uploaded_file is None:
        uploaded_file = st.file_uploader("Or upload a leaf image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🪴 Image being analyzed", use_column_width=True)

        # ===== STEP 1: PLANT IDENTIFICATION =====
        if plantnet_key:
            with st.spinner("🌿 Identifying plant species using PlantNet..."):
                try:
                    api_url = f"https://my-api.plantnet.org/v2/identify/all?api-key={plantnet_key}"
                    files = {'images': uploaded_file.getvalue()}
                    data = {'or
