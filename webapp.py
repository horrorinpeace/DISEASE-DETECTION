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
    st.markdown("<div class='caption'>Detect plant disease, view live farm sensor data, and generate a farmer-friendly treatment report.</div>", unsafe_allow_html=True)
with header_col2:
    st.write("")

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
    CLASS_NAMES = [...]
except Exception as e:
    st.warning(f"⚠ Could not load model: {e}")
    model = None
    CLASS_NAMES = []

# ==========================
# SESSION STATE INIT
# ==========================
if "report_text" not in st.session_state:
    st.session_state.report_text = ""

# ==========================
# SENSOR DATA
# ==========================
def fetch_sensor_data():
    ...
    return {...}

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
# SIDEBAR MENU (settings moved down)
# ==========================
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ["About", "AI Detection Panel"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")

selected_language_display = st.sidebar.selectbox("Report language", list(LANGUAGE_OPTIONS.keys()), index=0)
selected_language = LANGUAGE_OPTIONS[selected_language_display]

api_key = st.sidebar.text_input("🔐 OpenRouter API key", type="password")

# ==========================
# ABOUT PAGE
# ==========================
if page == "About":
    st.header("About FarmDoc AI")
    st.markdown("""
    <div class="card">
        FarmDoc AI helps farmers detect plant diseases from photos and provides simple,
        easy-to-follow guidance for treatment and prevention.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🌐 About FarmDoc (Multiple Languages)")
    st.markdown("""
    **Hindi:** फॉर्मडॉक किसानों को पत्तों की तस्वीर से रोग पहचानने और आसान भाषा में उपचार बताने में मदद करता है।  
    **Bengali:** ফার্মডক পাতা দেখে রোগ চিহ্নিত করে সহজ ভাষায় করণীয় জানায়।  
    **Tamil:** ஃபார்ம்‌డாக் இலைப் புகைப்படம் மூலம் நோயை கண்டறிந்து எளிய ஆலோசனைகளை வழங்கும்.  
    **Telugu:** ఫార్మ్‌డాక్ ఆకుల ఫోటో ద్వారా వ్యాధులను గుర్తించి సులభమైన సూచనలు ఇస్తుంది.  
    **Kannada:** ಫಾರ್ಮ್‌ಡಾಕ್ ಎಲೆಗಳ ಚಿತ್ರದಿಂದ ರೋಗ ಗುರುತುಹಾಕಿ ಸರಳ ಸಲಹೆಗಳನ್ನು ನೀಡುತ್ತದೆ.  
    **Malayalam:** ഫാംഡോക് ഇലയുടെ ചിത്രം കൊണ്ട് രോഗം തിരിച്ചറിഞ്ഞ് ലളിതമായി ഉപദേശം നൽകുന്നു.  
    **Marathi:** फार्मडॉक पानांच्या फोटोंवरून रोग ओळखून सोपे उपाय सांगते।  
    **Gujarati:** ફાર્મડોક પાંદડાની તસ્વીરથી રોગ ઓળખે છે અને સરળ સલાહ આપે છે.  
    **Punjabi:** ਫਾਰਮਡੌਕ ਪੱਤੇ ਦੀ ਤਸਵੀਰ ਤੋਂ ਰੋਗ ਪਛਾਣ ਕੇ ਸੌਖੇ ਉਪਾਅ ਦਿੰਦਾ ਹੈ।  
    **Odia:** ଫାର୍ମଡକ୍ ପତ୍ର ଫଟୋରୁ ରୋଗ ଚିହ୍ନଟ କରି ସହଜ ସୁପରିଶ ଦେଇଥାଏ।  
    **Urdu:** فارم ڈاک پتے کی تصویر سے بیماری شناخت کر کے آسان مشورہ دیتا ہے۔  
    """)

# ==========================
# AI DETECTION PANEL
# ==========================
elif page == "AI Detection Panel":
    st.header("Step 1 — Capture or Upload Plant Image")
    ...

    # Detection block unchanged except UI

    st.header("Step 2 — Live Farm Data")
    ...

    # ==========================
    # AI REPORT GENERATION
    # ==========================
    st.header("Step 3 — Get Farm Report")  # <-- UPDATED TEXT

    if st.button("🧾 Generate Farm Report"):
        ...
        prompt = f"""
            ...
            Respond in {selected_language}.
            Use this exact format:
            - Disease Name:
            - What It Means:
            - What You Should Do:
            - Prevention Tips:
        """
        ...

    # ==========================
    # SHOW REPORT
    # ==========================
    if st.session_state.report_text:
        ...

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)

        pdf.cell(0, 10, "Farm Report", ln=True, align="C")   # <-- UPDATED PDF TITLE

        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 8, st.session_state.report_text)
        ...

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("<div class='caption'>FarmDoc © 2025 — Helping Farmers Grow Smarter</div>", unsafe_allow_html=True)
