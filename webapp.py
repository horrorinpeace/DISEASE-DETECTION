import io
import threading
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import requests
from fpdf import FPDF
import openai
from huggingface_hub import hf_hub_download
from flask import Flask, request, jsonify
import tensorflow as tf

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
                -1px 1px 0 #000,
                1px 1px 0 #000;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("https://images.unsplash.com/photo-1503264116251-35a269479413?auto=format&fit=crop&w=1200&q=80")

# ==========================
# OPENAI CONFIG
# ==========================
openai.api_key = st.secrets.get("OPENAI_API_KEY", "sk-yourkeyhere")

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
    CLASS_NAMES = ['HEALTHY MILLET', 'HEALTHY POTATO', 'HEALTHY RICE', 'HEALTHY SUGARCANE',
                  'HEALTHY TEA LEAF', 'HEALTHY TOMATO', 'HEALTHY WHEAT', 'MILLETS BLAST',
                  'MILLETS RUST', 'POTATO EARLY BLIGHT', 'POTATO LATE BLIGHT',
                  'RICE BACTERIAL BLIGHT', 'RICE BROWN SPOT', 'RICE LEAF SMUT',
                  'SUGARCANE RED ROT', 'SUGARCANE RUST', 'SUGARCANE YELLOW',
                  'TEA GRAY BLIGHT', 'TEA GREEN MIRID BUG', 'TEA HELOPELTIS',
                  'TOMATO LEAF MOLD', 'TOMATO MOSAIC VIRUS', 'TOMATO SEPTORIA LEAF SPOT',
                  'WHEAT BROWN RUST', 'WHEAT LOOSE SMUT', 'WHEAT YELLOW RUST']
except Exception as e:
    st.warning(f"⚠️ Could not load model: {e}")
    model = None
    CLASS_NAMES = []

# ==========================
# THINGSPEAK CONFIG
# ==========================
THINGSPEAK_CHANNEL_ID = "3152731"   # ← Replace with your ThingSpeak Channel ID
READ_API_KEY = "8WGWK6AUAF74H6DJ"          # ← Replace with your ThingSpeak Read API Key

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

    uploaded_file = st.camera_input("Capture an image")
    if uploaded_file is None:
        uploaded_file = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])

    # ==========================
    # MODEL PREDICTION
    # ==========================
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Captured / Uploaded Image",)

        if model:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

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

    # ==========================
    # SENSOR DATA (TEXT-ONLY)
    # ==========================
    st.subheader("📡 Live Sensor Data ")
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
    # LAB REPORT GENERATION
    # ==========================
    st.subheader("🧾 Generate Lab Report")
    if st.button("Generate Lab Report"):
        if uploaded_file is None:
            st.error("Please upload or capture an image first.")
        elif model is None:
            st.error("AI model  not loaded.")
        else:
            st.info("Generating lab report via GPT...")

            prompt = f"""
            Create a concise lab report using:
            Detection Results: {df_results.to_dict(orient='records')}
            Sensor Data: {sensor}
            Format as structured tables under headings: 
            Sample Analysis, Sensor Data, Observations, Conclusion.
            """

            try:
                import requests
                import os

                API_URL = "https://h2ogpte.genai.h2o.ai/v1/chat/completions"
                API_KEY = os.getenv("H2O_API_KEY", "your_api_key_here")

                def h2ogpte_chat(prompt: str, model: str = "h2ogpt-7b-chat") -> str:
                payload = {
                "model": model,
                "messages": [
                 {"role": "system", "content": "You are a scientific report writer."},
                {"role": "user",   "content": prompt}
                ],
                "max_tokens": 800,
                 "temperature": 0.7
                 }
                 headers = {
                 "Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"
                 }
                 resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
                 resp.raise_for_status()
                 result = resp.json()
                 # depending on API shape, adapt:
                 return result["choices"][0]["message"]["content"]

                # Usage inside your Streamlit logic:
                prompt = f"""Create a concise lab report using:
                Detection Results: {df_results.to_dict(orient='records')}
                Sensor Data: {sensor}
                Format as structured tables under headings:
                Sample Analysis, Sensor Data, Observations, Conclusion."""
try:
    report_text = h2ogpte_chat(prompt)
    except Exception as e:
    st.error(f"H2O GPT Error: {e}")
    report_text = "Could not generate report."

                 st.markdown(report_text)

            # PDF generation
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "AI Lab Report", ln=True, align="C")
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 8, report_text)

            # Add image safely to PDF
            temp_img_path = "temp_image.jpg"
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


st.markdown("---")
st.markdown("© 2025 AI Detection Lab — Built with ❤️ using Streamlit.")





































