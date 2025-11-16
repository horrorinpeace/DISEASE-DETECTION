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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# APP-WIDE STYLES
# ==========================
def set_background_and_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f1724 0%, #162033 45%, #20324a 100%) no-repeat center center fixed;
            background-size: cover;
            color: #e6eef8;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .block-container {
            background: rgba(10, 14, 20, 0.45);
            border-radius: 14px;
            padding: 28px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.45);
        }
        h1, h2, h3, h4, h5, h6, p, div, span, label { color: #eef6ff !important; }
        .card {
            background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 12px;
            box-shadow: 0 6px 18px rgba(2,6,23,0.6);
        }
        .caption { font-size: 12px; color: #d6e8ff; opacity: 0.8; }
        .stButton>button {
            background: linear-gradient(90deg,#2fb86f,#35c06f) !important;
            color: white !important;
            font-weight: 600;
            border-radius: 12px !important;
        }
        .stDownloadButton>button {
            background: rgba(255,255,255,0.06) !important;
            color: white !important;
            border-radius: 10px !important;
        }
        video { transform: scaleX(-1) !important; }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_and_styles()

# ==========================
# PAGE HEADER
# ==========================
header_col1, header_col2 = st.columns([0.9, 0.1])
with header_col1:
    st.markdown("<h1 style='margin:0;'>🌱 FarmDoc</h1>", unsafe_allow_html=True)
    st.markdown("<div class='caption'>Detect plant disease, view live farm sensor data, and get an easy-to-follow treatment report in multiple languages.</div>", unsafe_allow_html=True)

st.markdown("---")

# ==========================
# LOAD MODEL
# ==========================
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
except:
    st.warning("⚠ Could not load model.")
    model = None
    CLASS_NAMES = []

# ==========================
# SESSION STATE
# ==========================
if "report_text" not in st.session_state:
    st.session_state.report_text = ""

# ==========================
# SENSOR DATA
# ==========================
def fetch_sensor_data():
    url = f"https://api.thingspeak.com/channels/{3152731}/feeds.json?api_key=8WGWK6AUAF74H6DJ&results=1"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get("feeds"):
            latest = data["feeds"][0]
            return {
                "temperature": latest["field1"],
                "humidity": latest["field2"],
                "soil_moisture": latest["field3"],
                "timestamp": latest["created_at"]
            }
    except:
        pass
    return {"temperature": None, "humidity": None, "soil_moisture": None, "timestamp": None}

# ==========================
# MULTI-LANGUAGE OPTIONS
# ==========================
LANGUAGE_OPTIONS = {
    "English": "English",
    "हिन्दी (Hindi)": "Hindi",
    "বাংলা (Bengali)": "Bengali",
    "தமிழ் (Tamil)": "Tamil",
    "తెలుగు (Telugu)": "Telugu",
    "ಕನ್ನಡ (Kannada)": "Kannada",
    "മലയാളം (Malayalam)": "Malayalam",
    "मराठी (Marathi)": "Marathi",
    "ગુજરાતી (Gujarati)": "Gujarati",
    "ਪੰਜਾਬੀ (Punjabi)": "Punjabi",
    "ଓଡ଼ିଆ (Odia)": "Odia",
    "اردو (Urdu)": "Urdu"
}

# ==========================
# SIDEBAR MENU (Settings moved BELOW menu)
# ==========================
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ["About", "AI Detection Panel"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")

selected_language_display = st.sidebar.selectbox("Report language", list(LANGUAGE_OPTIONS.keys()))
selected_language = LANGUAGE_OPTIONS[selected_language_display]

api_key = st.sidebar.text_input("🔐 OpenRouter API key", type="password")

# ==========================
# ABOUT PAGE
# ==========================
if page == "About":
    st.header("About FarmDoc AI")
    st.markdown("""
    <div class="card">
        <strong>FarmDoc AI</strong> helps farmers detect plant diseases using a phone camera or uploaded images.
        It gives simple, clear advice on what the disease is, how it affects the crop, what actions to take,
        and how to prevent it — available in different languages.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### How it works")
    st.markdown("""
    1. Take a photo of the leaf or upload an image.  
    2. The AI model predicts the likely disease.  
    3. You generate a short report (in your chosen language).  
    4. Download the PDF report to share or print.
    """)

# ==========================
# AI DETECTION PANEL
# ==========================
elif page == "AI Detection Panel":

    st.header("Step 1 — Capture or Upload Plant Image")
    st.markdown("<div class='card'>Use your phone camera or upload a clear photo of the affected leaf.</div>", unsafe_allow_html=True)

    uploaded_file = st.camera_input("📸 Take a photo of your crop leaf")
    if uploaded_file is None:
        uploaded_file = st.file_uploader("Or upload a leaf image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🪴 This is the captured image being analyzed", width=300)

        if model:
            img = image.resize((224, 224))
            arr = tf.keras.preprocessing.image.img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            preds = model.predict(arr)
            confidence = np.max(preds)
            predicted_class = CLASS_NAMES[np.argmax(preds)]

            st.session_state.predicted_class = predicted_class
            st.session_state.confidence = confidence

            st.success(f"🌿 Detected: {predicted_class} — {confidence*100:.2f}%")

    # ==========================
    # SENSOR DATA
    # ==========================
    st.header("Step 2 — Live Farm Data")
    count = st_autorefresh(interval=5000, limit=None, key="sensor_refresh")

    sensor = fetch_sensor_data()

    if sensor["temperature"]:
        c1, c2, c3 = st.columns(3)
        c1.metric("🌡 Temperature", f"{sensor['temperature']} °C")
        c2.metric("💧 Humidity", f"{sensor['humidity']} %")
        c3.metric("🌱 Soil Moisture", f"{sensor['soil_moisture']} %")
        st.caption(f"Last updated: {sensor['timestamp']}")
    else:
        st.warning("Waiting for live data...")

    # ==========================
    # AI REPORT GENERATION (text update applied)
    # ==========================
    st.header("Step 3 — Get Farm Report")
    st.markdown("<div class='card'>The AI will write the report in the selected language.</div>", unsafe_allow_html=True)

    if st.button("🧾 Generate Farm Report"):
        if not api_key:
            st.error("Please enter your API key.")
        elif not uploaded_file:
            st.error("Please upload or capture an image.")
        elif model is None:
            st.error("Model not loaded.")
        else:
            with st.spinner("Writing report..."):
                prompt = f"""
                You are a helpful agricultural assistant.
                Write the report in, write it in a simple way for farmers to understand in {selected_language}.
                Use this format:
                - Disease Name:
                - What It Means:
                - What You Should Do:
                - Prevention Tips:

                Disease: {st.session_state.get('predicted_class')}
                Confidence: {st.session_state.get('confidence')*100:.2f}%

                Conditions:
                Temperature: {sensor['temperature']}
                Humidity: {sensor['humidity']}
                Soil Moisture: {sensor['soil_moisture']}
                """

                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                data = {
                    "model": "meta-llama/llama-3.1-8b-instruct",
                    "messages": [
                        {"role": "system", "content": "You give farm advice."},
                        {"role": "user", "content": prompt}
                    ]
                }

                try:
                    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
                    result = r.json()
                    st.session_state.report_text = result["choices"][0]["message"]["content"]
                    st.success("Report generated!")
                except:
                    st.error("Error generating report.")

    # ==========================
    # SHOW REPORT + PDF FIXES
    # ==========================
    if st.session_state.report_text:
        st.markdown("### 🌿 Your Farm Report")
        st.markdown(f"<div class='card'><pre style='white-space:pre-wrap'>{st.session_state.report_text}</pre></div>", unsafe_allow_html=True)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)

        pdf.cell(0, 10, "Farm Report", ln=True, align="C")   # <-- FIXED

        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, st.session_state.report_text)

        if uploaded_file:
            with open("temp.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf.image("temp.jpg", x=10, w=100)

        pdf_bytes = pdf.output(dest='S').encode("latin-1")

        st.download_button(
            "📥 Download Farm Report (PDF)",  # <-- FIXED
            pdf_bytes,
            "farm_report.pdf",
            "application/pdf"
        )

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("<div class='caption'>FarmDoc © 2025 — Helping Farmers Grow Smarter</div>", unsafe_allow_html=True)








