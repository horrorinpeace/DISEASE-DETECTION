import io
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import requests
from fpdf import FPDF
from huggingface_hub import hf_hub_download
import tensorflow as tf
import os
import json

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="🌾 Smart Farm Doctor",
    layout="wide"
)

# ==========================
# BACKGROUND
# ==========================
def set_background(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{url}") no-repeat center center fixed;
            background-size: cover;
        }}
        .block-container {{
            background-color: rgba(0, 0, 0, 0.55) !important;
            border-radius: 20px;
            padding: 25px !important;
        }}
        h1, h2, h3, h4, h5, h6, p, div, span {{
            color: white !important;
            font-family: 'Segoe UI', sans-serif;
        }}
        .css-1q8dd3e p {{
            font-size: 18px !important;
        }}
        .stButton>button {{
            background-color: #34a853 !important;
            color: white !important;
            font-size: 18px !important;
            border-radius: 12px !important;
            padding: 10px 25px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("https://images.unsplash.com/photo-1524592094714-0f0654e20314?auto=format&fit=crop&w=1200&q=80")

# ==========================
# LOAD MODEL
# ==========================
st.title("🌱 Smart Farm Doctor")
st.write("A simple tool to **detect plant diseases** and get **easy-to-understand treatment advice** using AI.")

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
    st.warning(f"⚠️ Could not load model: {e}")
    model = None
    CLASS_NAMES = []

# ==========================
# SENSOR DATA FROM THINGSPEAK
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
# NAVIGATION
# ==========================
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ["About", "AI Detection Panel"])

# ==========================
# ABOUT
# ==========================
if page == "About":
    st.header("🌾 About Smart Farm Doctor")
    st.markdown("""
    **Smart Farm Doctor** helps farmers detect plant diseases using their phone’s camera or uploaded images.

    It also gives **simple, clear advice** on:
    - What the disease is  
    - How it affects the crop  
    - What actions to take  
    - How to prevent it in the future  

    It connects with your farm sensors (ESP32 + ThingSpeak) to include weather and soil data in your report.

    📷 Just take a photo → 🧠 Let AI detect → 📋 Get your easy farm report.
    """)

# ==========================
# DETECTION PANEL
# ==========================
elif page == "AI Detection Panel":
    st.header("🧠 Step 1: Capture or Upload Plant Image")

    api_key = st.sidebar.text_input("🔐 Enter your OpenRouter API key (starts with sk-or-...)", type="password")

    uploaded_file = st.camera_input("📸 Take a photo of your crop leaf")
    if uploaded_file is None:
        uploaded_file = st.file_uploader("Or upload a leaf image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🪴 This is the image being analyzed", use_column_width=True)

        if model:
            img_resized = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            confidence = np.max(preds)
            predicted_class = CLASS_NAMES[np.argmax(preds)]

            st.success(f"🌿 The AI detected: **{predicted_class}** with {confidence*100:.2f}% confidence.")

    # ==========================
    # SENSOR DATA
    # ==========================
    st.header("🌡 Step 2: Check Live Farm Data")
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=10000, key="sensor_refresh")
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
    # AI REPORT (Simplified Language)
    # ==========================
    st.header("📋 Step 3: Get Simple AI Farm Report")

    if "report_text" not in st.session_state:
        st.session_state.report_text = ""
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False

    if uploaded_file and model:
        st.session_state.predicted_class = predicted_class
        st.session_state.confidence = confidence

    if st.button("🧾 Generate Easy Farm Report") and not st.session_state.is_generating:
        if not api_key:
            st.error("Please enter your OpenRouter API key in the sidebar.")
        elif not uploaded_file:
            st.error("Please upload or take a photo first.")
        elif model is None:
            st.error("AI model not loaded.")
        else:
            st.session_state.is_generating = True
            st.session_state.report_text = ""
            st.info("🧠 The AI is writing your report in simple farmer language...")

            prompt = f"""
            You are a helpful agricultural assistant speaking to a farmer.
            Write a clear, short, and easy-to-understand farm report using simple words (no technical terms).
            Explain what disease was found: {st.session_state.predicted_class} (confidence {st.session_state.confidence*100:.2f}%)
            and how it affects the plant.

            Use this format:
            - **Disease Name:** (name)
            - **What It Means:** simple explanation
            - **What You Should Do:** 2-3 easy steps for treatment
            - **Prevention Tips:** short and clear advice for next time

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
                "max_tokens": 500,
                "temperature": 0.7,
                "stream": True
            }

            report_placeholder = st.empty()
            full_text = ""

            try:
                with requests.post("https://openrouter.ai/api/v1/chat/completions",
                                   headers=headers, json=data, stream=True, timeout=90) as response:
                    if response.status_code != 200:
                        st.error(f"OpenRouter API Error: {response.status_code}")
                    else:
                        for line in response.iter_lines():
                            if line:
                                try:
                                    decoded = line.decode("utf-8")
                                    if decoded.startswith("data: "):
                                        payload = json.loads(decoded[6:])
                                        delta = payload.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                        full_text += delta
                                        report_placeholder.markdown(full_text + "▌")
                                except Exception:
                                    continue

                        report_placeholder.markdown("### 🌿 Your Farm Report\n" + full_text)
                        st.session_state.report_text = full_text
                        st.session_state.is_generating = False
                        st.success("✅ Report ready! Scroll down to download it.")

            except Exception as e:
                st.error(f"❌ Error generating report: {e}")
                st.session_state.is_generating = False

    # ==========================
    # DOWNLOAD SECTION
    # ==========================
    if st.session_state.report_text and not st.session_state.is_generating:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Easy Farm Report", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, st.session_state.report_text)

        temp_img_path = "temp_image.jpg"
        if uploaded_file:
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
st.markdown("🌾 **Smart Farm Doctor © 2025** — Helping Farmers Grow Smarter 🌿")
