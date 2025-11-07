import io
import threading
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
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
    repo_id="qwertymaninwork/Plant_Disease_Detection_System",  # your repo name
    filename="mobilenetv2_plant.h5"
    )
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    return model

try:
    model = load_model()
    CLASS_NAMES = ['HEALTHY RICE',
                'RICE BACTERIAL BLIGHT',
                'RICE BROWN SPOT',
                'RICE LEAF SMUT',
                'HEALTHY WHEAT',
                'WHEAT LOOSE SMUT',
                'WHEAT YELLOW RUST',
                'WHEAT BROWN RUST',
                'HEALTHY MILLET',
                'MILLET RUST',
                'MILLET BLAST',
                'HEALTHY SUGARCANE',
                'SUGARCANE YELLOW',
                'SUGARCANE RED ROT',
                'SUGARCANE RUST',
                'HEALTHY TEA LEAF',
                'TEA GREEN MIRID BUG',
                'TEA GRAY BLIGHT',
                'TEA HELOPELITIS',
                'HEALTHY POTATO',
                'POTATO EARLY BLIGHT',
                'POTATO LATE BLIGHT',
                'HEALTHY TOMATO',
                'TOMATO LEAF MOLD',
                'TOMATO MOSAIC VIRUS',
                'TOMATO SEPTORIA LEAF SPOT',
                'HEALTHY RICE',
                'HEALTHY SUGARCANE',
                'HEALTHY TEA LEAF',
                'HEALTHY WHEAT',
            ]
except Exception as e:
    st.warning(f"⚠️ Could not load model: {e}")
    model = None
    CLASS_NAMES = []

# ==========================
# SHARED SENSOR DATA
# ==========================
shared_data = {"temperature": None, "humidity": None, "soil_moisture": None}

# ==========================
# FLASK BACKEND (for ESP32)
# ==========================
flask_app = Flask(__name__)

@flask_app.route('/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    shared_data.update(data)
    return jsonify({"status": "received"}), 200

def run_flask():
    flask_app.run(host="0.0.0.0", port=8600, debug=False, use_reloader=False)

threading.Thread(target=run_flask, daemon=True).start()

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
    - Accepts real sensor data directly from an ESP32 device  
    - Uses your trained AI model for plant disease detection  
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

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Captured / Uploaded Image")

        # ==========================
        # MODEL PREDICTION
        # ==========================
        if model:
            # Preprocess
            img_resized = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Predict
            preds = model.predict(img_array)
            confidence = np.max(preds)
            predicted_class = CLASS_NAMES[np.argmax(preds)]
            st.write("### Debug Info")
            st.write("CLASS_NAMES length:", len(CLASS_NAMES))
            st.write("CLASS_NAMES:", CLASS_NAMES)
            st.write("Predictions shape:", preds.shape)
            st.write("Predictions array:", preds)

            df_results = pd.DataFrame({
                "Disease": CLASS_NAMES,
                "Probability": preds[0]
            }).sort_values(by="Probability", ascending=False)
            st.success(f"✅ Prediction: **{predicted_class}** ({confidence*100:.2f}% confidence)")
            st.table(df_results)
        else:
            st.error("No model loaded. Please ensure 'model.h5' exists in your project folder.")

    # ==========================
    # SENSOR DATA
    # ==========================
    st.subheader("📡 Live Sensor Data from ESP32")
    if all(v is not None for v in shared_data.values()):
        df_sensor = pd.DataFrame([shared_data])
        st.table(df_sensor)
    else:
        st.warning("Waiting for data from ESP32...")

    # ==========================
    # LAB REPORT GENERATION
    # ==========================
    if st.button("Generate Lab Report"):
        if uploaded_file is None:
            st.error("Please upload or capture an image first.")
        elif model is None:
            st.error("AI model not loaded. Please check model file.")
        else:
            st.info("Generating lab report via GPT...")

            prompt = f"""
            Create a concise lab report using:
            Detection Results: {df_results.to_dict(orient='records')}
            Sensor Data: {shared_data}
            Format as structured tables under headings: Sample Analysis, Sensor Data, Observations, Conclusion.
            """

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a scientific report writer."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800
                )
                report_text = response.choices[0].message.content
            except Exception as e:
                st.error(f"GPT Error: {e}")
                report_text = "Could not generate report."

            st.markdown(report_text)

            # PDF generation
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "AI Lab Report", ln=True, align="C")
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 8, report_text)

            # Add image
            uploaded_file.seek(0)
            pdf.image(uploaded_file, x=10, y=None, w=100)
            pdf_bytes = io.BytesIO()
            pdf.output(pdf_bytes)
            pdf_bytes.seek(0)
            st.download_button("📥 Download PDF", pdf_bytes, "lab_report.pdf")

st.markdown("---")
st.markdown("© 2025 AI Detection Lab — Built with ❤️ using Streamlit.")










