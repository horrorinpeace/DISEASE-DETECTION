import io
import threading
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import requests
from fpdf import FPDF
from huggingface_hub import hf_hub_download
from flask import Flask, request, jsonify
import tensorflow as tf
import os

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="AI Detection & Lab Report Generator",
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
            background-color: transparent !important;
        }}
        body, p, div, span, h1, h2, h3, h4, h5, h6 {{
            color: white !important;
            text-shadow:
                -1px -1px 0 #000,
                1px -1px 0 #000,
                -1px 1px 0 #000;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("https://images.unsplash.com/photo-1503264116251-35a269479413?auto=format&fit=crop&w=1200&q=80")

# ==========================
# LOAD MODEL
# ==========================
model_path = hf_hub_download(
    repo_id="qwertymaninwork/Plant_Disease_Detection_System",
    filename="mobilenetv2_plant.h5"
)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    return model

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
# THINGSPEAK CONFIG
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
    except Exception as e:
        print("ThingSpeak fetch error:", e)
    return {"temperature": None, "humidity": None, "soil_moisture": None, "timestamp": None}

# ==========================
# SIDEBAR NAV
# ==========================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About", "Detection Panel"])

# ==========================
# ABOUT PAGE
# ==========================
if page == "About":
    st.title("🌱 AI Detection & Lab Report Webapp")
    st.markdown("""
    This application:
    - Accepts real sensor data from your ESP32 via ThingSpeak  
    - Uses an AI model to detect plant diseases  
    - Generates GPT-based lab reports automatically  
    """)

# ==========================
# DETECTION PANEL
# ==========================
elif page == "Detection Panel":
    st.title("🔬 Detection Panel")

    # API Key entry box
    st.sidebar.subheader("🔐 OpenRouter API Key")
    api_key = st.sidebar.text_input("Enter your OpenRouter API key (starts with sk-or-...)", type="password")

    uploaded_file = st.camera_input("Capture an image")
    if uploaded_file is None:
        uploaded_file = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Captured / Uploaded Image")

        if model:
            img_resized = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            confidence = np.max(preds)
            predicted_class = CLASS_NAMES[np.argmax(preds)]

            df_results = pd.DataFrame({
                "Disease": CLASS_NAMES,
                "Probability": preds[0]
            }).sort_values(by="Probability", ascending=False)

            st.success(f"✅ Prediction: **{predicted_class}** ({confidence*100:.2f}% confidence)")
            st.table(df_results)
        else:
            st.error("No model loaded. Please ensure the model file is available.")

    st.subheader("📡 Live Sensor Data")
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=10000, key="sensor_refresh")
    sensor = fetch_sensor_data()

    if sensor["temperature"] is not None:
        st.success("✅ Latest Sensor Readings:")
        col1, col2, col3 = st.columns(3)
        col1.metric("🌡 Temperature (°C)", f"{sensor['temperature']}")
        col2.metric("💧 Humidity (%)", f"{sensor['humidity']}")
        col3.metric("🌱 Soil Moisture (%)", f"{sensor['soil_moisture']}")
        st.caption(f"⏱ Last updated: {sensor['timestamp']}")
    else:
        st.warning("Waiting for ESP32 data from ThingSpeak...")

    # ==========================
        # ==========================
    # ==========================
    # 🧠 LAB REPORT GENERATION (OpenRouter) — STREAMING FIXED VERSION
    # ==========================

# ==========================
# 🧠 LAB REPORT GENERATION (OpenRouter) — PERMANENT FIX
# ==========================
st.subheader("🧾 Generate AI Lab Report")

# Initialize state variable to persist AI output
if "ai_report_text" not in st.session_state:
    st.session_state.ai_report_text = ""

if st.button("Generate Lab Report"):
    if not api_key:
        st.error("Please enter your OpenRouter API key in the sidebar.")
    elif uploaded_file is None:
        st.error("Please upload or capture an image first.")
    elif model is None:
        st.error("AI model not loaded.")
    else:
        st.session_state.ai_report_text = ""  # reset output
        st.info("🧠 Generating AI-based lab report using OpenRouter...")

        prompt = f"""
        You are an agricultural scientist.
        Create a concise lab report with recommendations based on:
        - Detected disease: {predicted_class} (confidence {confidence*100:.2f}%)
        - Temperature: {sensor['temperature']} °C
        - Humidity: {sensor['humidity']} %
        - Soil moisture: {sensor['soil_moisture']} %
        Include sections: Diagnosis, Observations, Recommended Actions, Preventive Measures.
        """

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }

        payload = {
            "model": "meta-llama/llama-3.1-8b-instruct",
            "messages": [
                {"role": "system", "content": "You are an expert agricultural scientist writing detailed reports."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "stream": True
        }

        try:
            with requests.post("https://openrouter.ai/api/v1/chat/completions",
                               headers=headers, json=payload, stream=True, timeout=120) as response:
                if response.status_code != 200:
                    st.error(f"OpenRouter API Error: {response.status_code}")
                    st.text_area("API Response", response.text, height=200)
                else:
                    st.markdown("### 🧾 AI Lab Report")
                    report_area = st.empty()
                    full_text = ""

                    for line in response.iter_lines(decode_unicode=True):
                        if line and line.startswith("data: "):
                            content = line[len("data: "):].strip()
                            if content == "[DONE]":
                                break
                            try:
                                data = json.loads(content)
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                token = delta.get("content", "")
                                if token:
                                    full_text += token
                                    report_area.markdown(full_text)
                            except Exception:
                                continue

                    if not full_text.strip():
                        st.warning("⚠️ No content received. Try again later.")
                    else:
                        st.session_state.ai_report_text = full_text  # Save for persistence

        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Please try again.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ==========================
# 🧾 Display Persisted AI Report
# ==========================
if st.session_state.ai_report_text:
    st.markdown("### 🧾 AI Lab Report (Saved)")
    st.markdown(st.session_state.ai_report_text)

    # PDF download option
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI Lab Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, st.session_state.ai_report_text)

    temp_img_path = "temp_image.jpg"
    if uploaded_file:
        with open(temp_img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf.image(temp_img_path, x=10, y=None, w=100)

    pdf_bytes = pdf.output(dest='S').encode('latin-1')

    st.download_button(
        "📥 Download PDF",
        data=pdf_bytes,
        file_name="lab_report.pdf",
        mime="application/pdf"
    )




# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("© 2025 AI Detection Lab — Built with ❤️ using Streamlit.")



