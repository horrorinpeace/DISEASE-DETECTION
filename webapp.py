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
    code = LANG_CODE_MAP.get(target_language_name, "en")
    if code == "en":
        return text
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {"client": "gtx", "sl": "en", "tl": code, "dt": "t", "q": text}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        translated = "".join(chunk[0] for chunk in data[0] if chunk[0])
        return translated if translated else text
    except Exception:
        return text

def is_mostly_english(text: str) -> bool:
    if not text: return False
    total_letters = sum(c.isalpha() for c in text)
    if total_letters == 0: return False
    ascii_letters = sum(c.isascii() and c.isalpha() for c in text)
    return (ascii_letters / total_letters) > 0.85

# ==========================
# ROBUST: disable augmentation inside model
# ==========================
def disable_augmentation_layers(model):
    disabled=[]
    def identity_call(self, inputs, training=False): return inputs
    def walk(layer):
        lname = getattr(layer, "name", "").lower()
        cname = layer.__class__.__name__.lower()
        if ("augmentation" in lname) or any(k in cname for k in ["random","flip","rotate","rotation","zoom","contrast","crop"]):
            try:
                layer.call = MethodType(identity_call, layer)
                layer.trainable=False
                disabled.append(layer.name)
            except: pass
        if hasattr(layer,"layers") and layer.layers:
            for sub in layer.layers: walk(sub)
    walk(model)
    return model

# ==========================
# PAGE CONFIG + STYLES
# ==========================
st.set_page_config(page_title="🌾FARMDOC", layout="wide", initial_sidebar_state="expanded")

def set_background_and_styles():
    st.markdown("""
    <style>
    .stApp {background: linear-gradient(135deg,#0f1724,#162033 45%,#20324a);}
    .block-container {background:rgba(10,14,20,.45);border-radius:14px;padding:28px;}
    .card {background:rgba(255,255,255,0.02);padding:14px;border-radius:12px;}
    h1,h2,h3,h4,h5,p,span,label{color:#eef6ff!important;}
    .stButton>button{background:#35c06f!important;color:white!important;border-radius:12px!important;}
    .stDownloadButton>button{background:rgba(255,255,255,.06)!important;color:white!important;}
    </style>""",unsafe_allow_html=True)
set_background_and_styles()

# ==========================
# HEADER
# ==========================
header_col1, header_col2 = st.columns([0.9,0.1])
with header_col1:
    st.markdown("<h1>🌱 FarmDoc</h1>", unsafe_allow_html=True)
    st.markdown("<div class='caption'>Detect disease + view live sensor data.</div>", unsafe_allow_html=True)
st.markdown("---")

# ==========================
# LOAD MODEL
# ==========================
model_path = hf_hub_download(repo_id="qwertymaninwork/Plant_Disease_Detection_System",filename="fix.h5")

@st.cache_resource
def load_model():
    m=tf.keras.models.load_model(model_path,compile=False,safe_mode=False)
    return disable_augmentation_layers(m)

try:
    model=load_model()
    CLASS_NAMES=['CURRY POWDERY MILDEW','HEALTHY MILLET','HEALTHY POTATO','HEALTHY RICE','HEALTHY SUGARCANE','HEALTHY TEA LEAF','HEALTHY TOMATO','HEALTHY WHEAT','MILLETS BLAST','MILLETS RUST','POTATO EARLY BLIGHT','POTATO LATE BLIGHT','RICE BACTERIAL BLIGHT','RICE BROWN SPOT','RICE LEAF SMUT','SUGARCANE RED ROT','SUGARCANE RUST','SUGARCANE YELLOW','TEA GRAY BLIGHT','TEA GREEN MIRID BUG','TEA HELOPELTIS','TOMATO LEAF MOLD','TOMATO MOSAIC VIRUS','TOMATO SEPTORIA LEAF SPOT','WHEAT BROWN RUST','WHEAT LOOSE SMUT','WHEAT YELLOW RUST']
except: model=None; CLASS_NAMES=[]

# ==========================
# SESSION STATE
# ==========================
if "report_text" not in st.session_state: st.session_state.report_text=""
if "selected_language" not in st.session_state: st.session_state.selected_language="English"
if "auto_refresh_on" not in st.session_state: st.session_state.auto_refresh_on=True

# ==========================
# UPDATED SENSOR FETCH — (AIR QUALITY ADDED)
# ==========================
def fetch_sensor_data():
    url="https://api.thingspeak.com/channels/3152731/feeds.json?api_key=8WGWK6AUAF74H6DJ&results=1"
    try:
        data=requests.get(url,timeout=5).json()
        if data.get("feeds"):
            latest=data["feeds"][0]
            return {
                "temperature": latest.get("field1"),
                "humidity": latest.get("field2"),
                "soil_moisture": latest.get("field3"),
                "air_quality": latest.get("field4"),      # 🔥 ADDED
                "timestamp": latest.get("created_at")
            }
    except: pass
    return {"temperature":None,"humidity":None,"soil_moisture":None,"air_quality":None,"timestamp":None}


# ==========================
# UI + SIDEBAR
# ==========================
LANGUAGE_OPTIONS={"English":"English","हिन्दी (Hindi)":"Hindi","বাংলা (Bengali)":"Bengali","தமிழ் (Tamil)":"Tamil","తెలుగు (Telugu)":"Telugu","ಕನ್ನಡ (Kannada)":"Kannada","മലയാളം (Malayalam)":"Malayalam","मराठी (Marathi)":"Marathi","ગુજરાતી (Gujarati)":"Gujarati","ਪੰਜਾਬੀ (Punjabi)":"Punjabi","ଓଡ଼ିଆ (Odia)":"Odia","اردو (Urdu)":"Urdu"}
st.sidebar.title("Menu")
page=st.sidebar.radio("Go to",["About","AI Detection Panel"])
st.sidebar.markdown("---")
st.session_state.selected_language = LANGUAGE_OPTIONS[ st.sidebar.selectbox("Report language", list(LANGUAGE_OPTIONS.keys())) ]
api_key = st.sidebar.text_input("🔐 Groq API key", type="password")

# ==========================
# ABOUT PAGE
# ==========================
if page=="About":
    st.header("About FarmDoc AI")
    st.markdown("<div class='card'>Helps detect diseases + generate reports.</div>",unsafe_allow_html=True)
    st.markdown("1. Upload image\n2. AI detects disease\n3. Live sensor + PDF report")

# ==========================
# AI DETECTION PAGE
# ==========================
elif page=="AI Detection Panel":
    st.header("Step 1 — Upload Leaf")
    uploaded_file=st.camera_input("📷 Take photo") or st.file_uploader("Upload image",type=["jpg","png","jpeg"])

    if uploaded_file and model:
        img=Image.open(uploaded_file).convert("RGB"); st.image(img,width=300)
        arr=np.expand_dims(tf.keras.preprocessing.image.img_to_array(img.resize((224,224))),0)
        pred=CLASS_NAMES[np.argmax(model.predict(arr))]
        st.session_state.predicted_class=pred; st.success(f"Detected: {pred}")

    # ==========================
    # 🔥 SENSOR SECTION WITH AIR QUALITY
    # ==========================
    st.header("Step 2 — Live Farm Data")

    if st.session_state.auto_refresh_on: st_autorefresh(interval=5000,key="sens")
    sensor=fetch_sensor_data()

    if sensor["temperature"] is not None:
        c1,c2,c3,c4=st.columns(4)
        c1.metric("🌡 Temperature",f"{sensor['temperature']} °C")
        c2.metric("💧 Humidity",f"{sensor['humidity']} %")
        c3.metric("🌱 Soil Moisture",f"{sensor['soil_moisture']} %")
        c4.metric("☁️ Air Quality (PPM)",f"{sensor['air_quality']} ppm")  # 🔥 ADDED
        st.caption(f"Last Updated: {sensor['timestamp']}")
    else:
        st.warning("Waiting for live data...")

    # (Report generator left untouched — no changes)
