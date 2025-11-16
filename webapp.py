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
# APP-WIDE STYLES (UI ENHANCEMENTS)
# ==========================
def set_background_and_styles():
    st.markdown(
        """
        <style>
        /* Background gradient and main container */
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

        /* Headings & text */
        h1, h2, h3, h4, h5, h6, p, div, span, label {
            color: #eef6ff !important;
        }

        /* Sidebar tweaks */
        .css-1d391kg {  /* sidebar background container (may vary by streamlit versions) */
            background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
            border-radius: 10px;
            padding: 12px;
        }

        /* Buttons: modern green pill */
        .stButton>button {
            background: linear-gradient(90deg,#2fb86f,#35c06f) !important;
            color: white !important;
            font-weight: 600;
            border-radius: 12px !important;
            padding: 10px 22px !important;
            box-shadow: 0 6px 18px rgba(50,160,90,0.15) !important;
            border: none !important;
        }

        /* Secondary white buttons (download) */
        .stDownloadButton>button {
            background: rgba(255,255,255,0.06) !important;
            color: white !important;
            border-radius: 10px !important;
            padding: 8px 18px !important;
            border: 1px solid rgba(255,255,255,0.06) !important;
        }

        /* Card style containers */
        .card {
            background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 12px;
            box-shadow: 0 6px 18px rgba(2,6,23,0.6);
        }

        /* Mirror camera preview */
        video {
            transform: scaleX(-1) !important;
        }

        /* Smaller caption font */
        .caption {
            font-size: 12px;
            color: #d6e8ff;
            opacity: 0.8;
        }

        /* Metric small tweaks */
        .stMetric {
            color: #fff;
        }

        /* Make file_uploader button more visible */
        .stFileUploader>div>label>div {
            background: rgba(255,255,255,0.02) !important;
            border-radius: 8px !important;
            padding: 8px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_and_styles()

# ==========================
# PAGE HEADER (Nice Top Row)
# ==========================
header_col1, header_col2 = st.columns([0.9, 0.1])
with header_col1:
    st.markdown("<h1 style='margin:0;'>🌱 FarmDoc</h1>", unsafe_allow_html=True)
    st.markdown("<div class='caption'>Detect plant disease, view live farm sensor data, and get an easy-to-follow treatment report in multiple Indian languages.</div>", unsafe_allow_html=True)
with header_col2:
    st.write("")  # reserved for future icon / small status

st.markdown("---")

# ==========================
# LOAD MODEL (unchanged)
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
except Exception as e:
    st.warning(f"⚠ Could not load model: {e}")
    model = None
    CLASS_NAMES = []

# ==========================
# SESSION STATE INIT (unchanged)
# ==========================
if "report_text" not in st.session_state:
    st.session_state.report_text = ""

# ==========================
# SENSOR DATA (unchanged)
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
# MULTI-LANGUAGE SETUP (All common Indian languages)
# We will pass the selected language into the AI prompt so the response comes in that language.
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
# SIDEBAR (with language selector)
# ==========================
st.sidebar.title("Menu")
st.sidebar.markdown("### Settings")

# Language selector
selected_language_display = st.sidebar.selectbox("Report language (choose one)", list(LANGUAGE_OPTIONS.keys()), index=0)
selected_language = LANGUAGE_OPTIONS[selected_language_display]

# API key input (kept in sidebar as before)
api_key = st.sidebar.text_input("🔐 OpenRouter API key (sk-or-...)", type="password")

st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", ["About", "AI Detection Panel"])

# ==========================
# ABOUT PAGE
# ==========================
if page == "About":
    st.header("About FarmDoc AI")
    st.markdown("""
    <div class="card">
    <strong>FarmDoc AI</strong> helps farmers detect plant diseases using a phone camera or uploaded images.
    It gives simple, clear advice on what the disease is, how it affects the crop, what actions to take,
    and how to prevent it — now available in many Indian languages.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### How it works")
    st.markdown("""
    1. Take a photo of the leaf or upload an image.  
    2. The AI model predicts the likely disease.  
    3. You generate a short report (in your chosen language) with easy steps and prevention tips.  
    4. Download the simple PDF report to share or print.
    """)

# ==========================
# AI DETECTION PANEL
# ==========================
elif page == "AI Detection Panel":
    st.header("Step 1 — Capture or Upload Plant Image")
    st.markdown("<div class='card'>Use your phone camera or upload a clear photo of the affected leaf. Try to fill the frame with the leaf.</div>", unsafe_allow_html=True)

    # Camera/upload (unchanged but image width reduced per your earlier request)
    uploaded_file = st.camera_input("📸 Take a photo of your crop leaf")
    if uploaded_file is None:
        uploaded_file = st.file_uploader("Or upload a leaf image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        # show smaller preview (width=300) to keep UI compact — requested earlier
        st.image(image, caption="🪴 This is the captured image being analyzed", width=300)

        if model:
            img_resized = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            confidence = np.max(preds)
            predicted_class = CLASS_NAMES[np.argmax(preds)]
            st.session_state.predicted_class = predicted_class
            st.session_state.confidence = confidence

            st.success(f"🌿 Detected: {predicted_class} — {confidence*100:.2f}%")

    # ==========================
    # SENSOR DATA DISPLAY (kept the same functionality, nicer cards)
    # ==========================
    st.header("Step 2 — Live Farm Data")
    st.markdown("<div class='card'>Your farm sensors (ESP32 → ThingSpeak) update automatically.</div>", unsafe_allow_html=True)

    # autorefresh (same)
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
    # AI REPORT GENERATION (only change: ask AI to respond in selected language)
    # ==========================
    st.header("Step 3 — Get an Easy Farm Report")
    st.markdown("<div class='card'>The AI will write the report in simple words for a farmer, in the language you selected in the sidebar.</div>", unsafe_allow_html=True)

    if st.button("🧾 Generate Farm Report"):
        if not api_key:
            st.error("Please enter your OpenRouter API key in the sidebar.")
        elif not uploaded_file:
            st.error("Please upload or take a photo first.")
        elif model is None:
            st.error("AI model not loaded.")
        else:
            with st.spinner("Writing the report in simple language..."):
                # Compose a prompt that requests the language explicitly.
                prompt = f"""
                You are a helpful agricultural assistant speaking to a farmer.
                Write a clear, short, and easy-to-understand farm report using simple words (no technical terms).
                Respond in {selected_language}.
                Use this exact format:
                - Disease Name: (name)
                - What It Means: simple explanation
                - What You Should Do: 2-3 easy steps for treatment
                - Prevention Tips: short and clear advice for next time

                Observed disease (English label): {st.session_state.get('predicted_class', 'Unknown')}
                Confidence: {st.session_state.get('confidence', 0)*100:.2f}%

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
                    response.raise_for_status()
                    result = response.json()
                    full_text = result["choices"][0]["message"]["content"]

                    # store and persist the report exactly as returned by the AI
                    st.session_state.report_text = full_text
                    st.success("✅ Report generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating report: {e}")

    # ==========================
    # SHOW REPORT (PERSISTS)
    # ==========================
    if st.session_state.report_text:
        st.markdown("### 🌿 Your Farm Report")
        # show the report in a pleasant card
        st.markdown(f"<div class='card'><pre style='white-space:pre-wrap; font-family:inherit; font-size:15px'>{st.session_state.report_text}</pre></div>", unsafe_allow_html=True)

        # PDF generation (unchanged functionality)
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
            # keep PDF image width at 100 as before
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
st.markdown("<div class='caption'>FarmDoc © 2025 — Helping Farmers Grow Smarter</div>", unsafe_allow_html=True)
