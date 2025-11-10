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
                    data = {'organs': ['leaf']}

                    response = requests.post(api_url, files=files, data=data, timeout=60)
                    response.raise_for_status()
                    result = response.json()

                    if result.get("results"):
                        top_result = result["results"][0]
                        species_name = top_result["species"]["scientificNameWithoutAuthor"]
                        common_name = ", ".join(top_result["species"].get("commonNames", [])) or "Unknown"
                        confidence = round(top_result["score"] * 100, 2)

                        st.session_state["species"] = species_name
                        st.session_state["species_confidence"] = confidence

                        st.success(f"🌿 **Plant Identified:** {species_name} ({common_name}) — {confidence}% confidence")
                    else:
                        st.warning("❌ No plant species identified. Try another image.")
                except Exception as e:
                    st.error(f"🚫 PlantNet API error: {e}")

        # ===== STEP 2: DISEASE DETECTION =====
        if model and "species" in st.session_state:
            with st.spinner("🔬 Detecting possible disease using local AI..."):
                img_resized = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                preds = model.predict(img_array)

                disease_index = np.argmax(preds)
                confidence_disease = np.max(preds)
                predicted_disease = CLASS_NAMES[disease_index]

                st.session_state["disease"] = predicted_disease
                st.session_state["disease_confidence"] = confidence_disease

                st.success(f"🧬 Disease Detected: **{predicted_disease}** ({confidence_disease*100:.2f}% confidence)")

    # ===== STEP 3: LIVE FARM DATA =====
    st.header("🌡 Step 2: Live Farm Conditions")
    st_autorefresh(interval=5000, limit=None, key="refresh")
    sensor = fetch_sensor_data()
    if sensor["temperature"]:
        col1, col2, col3 = st.columns(3)
        col1.metric("🌡 Temperature", f"{sensor['temperature']} °C")
        col2.metric("💧 Humidity", f"{sensor['humidity']} %")
        col3.metric("🌱 Soil Moisture", f"{sensor['soil_moisture']} %")
        st.caption(f"Last updated: {sensor['timestamp']}")
    else:
        st.warning("Waiting for live sensor data...")

    # ===== STEP 4: AI FARM REPORT =====
    st.header("📋 Step 3: Generate Farm Report")

    if st.button("🧾 Generate Easy Farm Report"):
        if not api_key:
            st.error("❗ Please enter your OpenRouter API key first.")
        elif "species" not in st.session_state or "disease" not in st.session_state:
            st.error("⚠️ Run plant and disease detection first.")
        else:
            with st.spinner("🧠 Generating your farm report..."):
                prompt = f"""
                You are a friendly agricultural assistant. Write a short and easy-to-understand farm report for a farmer.

                - Plant Detected: {st.session_state['species']} ({st.session_state['species_confidence']}% confidence)
                - Detected Disease: {st.session_state['disease']} ({st.session_state['disease_confidence']*100:.2f}% confidence)
                - Temperature: {sensor['temperature']} °C
                - Humidity: {sensor['humidity']} %
                - Soil Moisture: {sensor['soil_moisture']} %

                Format:
                - **Plant Name**
                - **Detected Condition**
                - **What You Should Do** (3 simple steps)
                - **Prevention Tips**
                """

                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                data = {
                    "model": "meta-llama/llama-3.1-8b-instruct",
                    "messages": [
                        {"role": "system", "content": "You are a kind farm advisor using simple words and farmer-friendly tone."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.7
                }

                try:
                    response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                             headers=headers, json=data, timeout=60)
                    response.raise_for_status()
                    result = response.json()
                    report = result["choices"][0]["message"]["content"]

                    st.session_state["report"] = report
                    st.success("✅ Report generated successfully!")
                    st.markdown("### 🌿 Your Easy Farm Report\n" + report)
                except Exception as e:
                    st.error(f"❌ Error generating report: {e}")

    # ===== STEP 5: PDF DOWNLOAD =====
    if "report" in st.session_state:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Easy Farm Report", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, st.session_state["report"])

        temp_img_path = "temp_leaf.jpg"
        with open(temp_img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf.image(temp_img_path, x=10, y=None, w=100)

        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        st.download_button(
            "📥 Download Report (PDF)",
            data=pdf_bytes,
            file_name="farm_report.pdf",
            mime="application/pdf"
        )

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("🌾 **Smart Farm Doctor © 2025** — Empowering Farmers with AI 🌿")
