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
from types import MethodType
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==========================
# TRANSLATION FALLBACK (Google Translate unofficial endpoint)
# ==========================
LANG_CODE_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Odia": "or",
    "Urdu": "ur",
}

def translate_with_google(text, target_language_name):
    """Translate English text to target_language using Google Translate unofficial endpoint."""
    code = LANG_CODE_MAP.get(target_language_name, "en")
    if code == "en":
        return text  # no need to translate to English

    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": "en",   # source language (we assume fallback happens only when text is English)
            "tl": code,   # target language code
            "dt": "t",
            "q": text
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        # data[0] is a list of [translated_chunk, original_chunk, ...]
        translated = "".join(chunk[0] for chunk in data[0] if chunk[0])
        return translated if translated else text
    except Exception:
        # if anything fails, just return original text so app never breaks
        return text

def is_mostly_english(text: str) -> bool:
    """Heuristic: checks if text is mostly ASCII letters (English-like)."""
    if not text:
        return False
    total_letters = sum(c.isalpha() for c in text)
    if total_letters == 0:
        return False
    ascii_letters = sum(c.isascii() and c.isalpha() for c in text)
    return (ascii_letters / total_letters) > 0.85

# ==========================
# ROBUST: disable augmentation inside model (recursive + bound method)
# ==========================
def disable_augmentation_layers(model):
    disabled = []

    def identity_call(self, inputs, training=False):
        return inputs

    def walk(layer):
        lname = getattr(layer, "name", "").lower()
        cname = layer._class.name_.lower()  # FIXED

        if ("augmentation" in lname) or any(
            k in cname for k in ["random", "flip", "rotate", "rotation", "zoom", "contrast", "crop"]
        ):
            try:
                layer.call = MethodType(identity_call, layer)
                layer.trainable = False
                disabled.append(layer.name)
            except Exception:
                pass

        if hasattr(layer, "layers") and layer.layers:
            for sub in layer.layers:
                walk(sub)

    walk(model)
    return model

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
    filename="fix.h5"
)

@st.cache_resource
def load_model():
    m = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    m = disable_augmentation_layers(m)
    return m

try:
    model = load_model()
    CLASS_NAMES = [
        'CURRY POWDERY MILDEW', 'HEALTHY MILLET', 'HEALTHY POTATO', 'HEALTHY RICE',
        'HEALTHY SUGARCANE', 'HEALTHY TEA LEAF', 'HEALTHY TOMATO', 'HEALTHY WHEAT',
        'MILLETS BLAST', 'MILLETS RUST', 'POTATO EARLY BLIGHT', 'POTATO LATE BLIGHT',
        'RICE BACTERIAL BLIGHT', 'RICE BROWN SPOT', 'RICE LEAF SMUT', 'SUGARCANE RED ROT',
        'SUGARCANE RUST', 'SUGARCANE YELLOW', 'TEA GRAY BLIGHT', 'TEA GREEN MIRID BUG',
        'TEA HELOPELTIS', 'TOMATO LEAF MOLD', 'TOMATO MOSAIC VIRUS',
        'TOMATO SEPTORIA LEAF SPOT', 'WHEAT BROWN RUST', 'WHEAT LOOSE SMUT',
        'WHEAT YELLOW RUST'
    ]
except Exception as e:
    st.error(f"Model failed to load: {e}")
    model = None
    CLASS_NAMES = []

# SESSION STATE INIT
if "report_text" not in st.session_state:
    st.session_state.report_text = ""

if "selected_language" not in st.session_state:
    st.session_state.selected_language = "English"

if "auto_refresh_on" not in st.session_state:
    st.session_state.auto_refresh_on = True

# ==========================
# SENSOR DATA
# ==========================
# SENSOR DATA
# ==========================
def fetch_sensor_data():
    url = "https://api.thingspeak.com/channels/3152731/feeds.json?api_key=8WGWK6AUAF74H6DJ&results=1"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get("feeds"):
            latest = data["feeds"][0]
            return {
                "temperature": latest["field1"],
                "humidity": latest["field2"],
                "soil_moisture": latest["field3"],
                "air_quality": latest["field4"],    # 🔥 AIR QUALITY ADDED
                "timestamp": latest["created_at"]
            }
    except:
        pass
    return {
        "temperature": None,
        "humidity": None,
        "soil_moisture": None,
        "air_quality": None,
        "timestamp": None
    }

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
# SIDEBAR MENU
# ==========================
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ["About", "AI Detection Panel"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")

selected_language_display = st.sidebar.selectbox("Report language", list(LANGUAGE_OPTIONS.keys()))

# language persistence
st.session_state.selected_language = LANGUAGE_OPTIONS[selected_language_display]

api_key = st.sidebar.text_input("🔐 Groq API key", type="password")

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

            predicted_class = CLASS_NAMES[np.argmax(preds)]

            st.session_state.predicted_class = predicted_class

            st.success(f"🌿 Detected: {predicted_class}")

    # ==========================
    # SENSOR DATA VIEW (ONLY AQ ADDED)
# ==========================
    sensor = fetch_sensor_data()

    if sensor["temperature"] is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("🌡 Temperature", f"{sensor['temperature']} °C")
        c2.metric("💧 Humidity", f"{sensor['humidity']} %")
        c3.metric("🌱 Soil Moisture", f"{sensor['soil_moisture']} %")

        # 🔥 AIR QUALITY DISPLAY ADDED — NO OTHER CHANGES MADE
        st.metric("🫁 Air Quality (PPM)", f"{sensor['air_quality']} ppm")

        st.caption(f"Last updated: {sensor['timestamp']}")
    else:
        st.warning("Waiting for live data...")

    # ==========================
    # AI REPORT GENERATION
    # ==========================
    st.header("Step 3 — Get Farm Report")
    st.markdown("<div class='card'>The AI will write the report in the selected language.</div>", unsafe_allow_html=True)

    if st.button("🧾 Generate Farm Report"):
        st.session_state.report_text = ""

        if not api_key:
            st.error("Please enter your Groq API key.")
        elif not uploaded_file:
            st.error("Please upload or capture an image.")
        elif model is None:
            st.error("Model not loaded.")
        elif "predicted_class" not in st.session_state:
            st.error("No disease prediction available.")
        else:
            # pause auto-refresh while generating to avoid interruption
            st.session_state.auto_refresh_on = False

            try:
                with st.spinner("Writing report..."):

                    prompt = f"""
                    You are a helpful agricultural assistant for farmers.

                    Write a VERY detailed, step-by-step farm advisory report in a simple way for farmers to understand in {st.session_state.selected_language}.

                    STRICT RULES (must follow all):
                    - Never skip information or stay vague.
                    - Always give specific names of fungicides/pesticides (generic name + 1–2 common brand examples if possible).
                    - ALWAYS give exact dose in:
                      • ml or g per litre of water
                      • ml or g per 15 L knapsack sprayer
                      • total quantity per acre (or per hectare) and approximate water volume.
                    - Clearly mention:
                      • how many times to spray
                      • gap between sprays (in days)
                      • waiting period before harvest, if needed.
                    - Clearly list ALL tools and materials needed:
                      • type of sprayer
                      • nozzle type
                      • measuring cup/spoon
                      • protective clothing (gloves, mask, etc.)
                      • any other tools.
                    - Use only safe, commonly used agricultural practices. Do NOT suggest anything illegal or extremely dangerous.
                    - If you are not fully sure of an exact product name, give a best-practice generic recommendation (for example: “systemic fungicide from triazole group such as …”) instead of writing “depends” or “consult expert”.
                    - Fill EVERY section completely. Do NOT leave any bullet empty.

                    Use THIS EXACT FORMAT and fill each point with detailed, practical, farmer-friendly instructions:

                    - Disease Name:
                    - What It Means:
                    - Cause:
                    - Name of spray to be used & Amount to be sprayed:
                    - Tools and Materials Needed (with quantities):
                    - Step By Step Process For Treatment (with exact measurements and timing):
                    - How many times to spray & gap between sprays:
                    - Safety Precautions for Farmers:
                    - Prevention Tips:


                    Disease: {st.session_state.get('predicted_class')}
                    
                    Conditions:
                    Temperature: {sensor['temperature']}
                    Humidity: {sensor['humidity']}
                    Soil Moisture: {sensor['soil_moisture']}
                    """

                    url = "https://api.groq.com/openai/v1/chat/completions"

                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    }

                    data = {
                        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                        "messages": [
                            {"role": "system", "content": "You give farm advice."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.6,
                        "max_completion_tokens": 800,
                        "top_p": 1,
                        "stream": False
                    }

                    r = requests.post(url, headers=headers, json=data, timeout=40)
                    r.raise_for_status()
                    result = r.json()

                    # raw text from Llama
                    raw_text = result["choices"][0]["message"]["content"]
                    st.session_state.report_text = raw_text

                    # ==========================
                    # TRANSLATION FALLBACK LOGIC
                    # ==========================
                    selected_lang_name = st.session_state.selected_language
                    if selected_lang_name != "English" and is_mostly_english(raw_text):
                        translated = translate_with_google(raw_text, selected_lang_name)
                        if translated:
                            st.session_state.report_text = translated

                    st.success("Report generated!")

            except requests.exceptions.HTTPError as http_err:
                st.error(f"HTTP Error: {http_err}")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                # resume auto-refresh after we finish
                st.session_state.auto_refresh_on = True

# ==========================
# SHOW REPORT (TXT + DOCX)
# ==========================
if st.session_state.report_text:
    st.markdown("### 🌿 Your Farm Report")
    st.markdown(
        f"<div class='card'><pre style='white-space:pre-wrap'>{st.session_state.report_text}</pre></div>",
        unsafe_allow_html=True
    )

    txt_bytes = st.session_state.report_text.encode("utf-8")
    st.download_button(
        "📥 Download Farm Report (TXT)",
        data=txt_bytes,
        file_name="farm_report.txt",
        mime="text/plain; charset=utf-8"
    )

    try:
        from docx import Document
        from docx.shared import Pt
        from io import BytesIO

        doc = Document()
        doc.add_heading("Farm Report", level=1)

        for line in st.session_state.report_text.splitlines():
            p = doc.add_paragraph(line)
            for run in p.runs:
                run.font.size = Pt(12)

        if 'uploaded_file' in locals() and uploaded_file:
            img_bytes = uploaded_file.getbuffer()
            image_stream = BytesIO(img_bytes)
            doc.add_page_break()
            doc.add_paragraph("Attached leaf image:")
            doc.add_picture(image_stream, width=Pt(300))

        f = BytesIO()
        doc.save(f)
        f.seek(0)

        st.download_button(
            "📥 Download Farm Report (DOCX)",
            data=f.read(),
            file_name="farm_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    except:
        st.warning("DOCX export not available. Please install python-docx in requirements.txt.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("<div class='caption'>FarmDoc © 2025 — Helping Farmers Grow Smarter</div>", unsafe_allow_html=True)
