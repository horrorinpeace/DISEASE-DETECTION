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
        params = {
            "client": "gtx",
            "sl": "en",
            "tl": code,
            "dt": "t",
            "q": text
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        translated = "".join(chunk[0] for chunk in data[0] if chunk[0])
        return translated if translated else text
    except Exception:
        return text


def is_mostly_english(text: str) -> bool:
    if not text:
        return False
    total_letters = sum(c.isalpha() for c in text)
    if total_letters == 0:
        return False
    ascii_letters = sum(c.isascii() and c.isalpha() for c in text)
    return (ascii_letters / total_letters) > 0.85


# ==========================
# ROBUST: disable augmentation inside model
# ==========================
def disable_augmentation_layers(model):
    disabled = []

    def identity_call(self, inputs, training=False):
        return inputs

    def walk(layer):
        lname = getattr(layer, "name", "").lower()
        cname = layer._class.name_.lower()
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
            font-family: 'Segoe UI', Tahoma, Geneva, Verd-serif;
        }
        .block-container {
            background: rgba(10, 14, 20, 0.45);
            border-radius: 14px;
            padding: 28px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.45);
        }
        h1,h2,h3,h4,h5,h6,p,div,span,label{color:#eef6ff!important;}
        .card {
            background: linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.01));
            border-radius:12px;padding:14px;margin-bottom:12px;
            box-shadow:0 6px 18px rgba(2,6,23,0.6);
        }
        .caption { font-size:12px;color:#d6e8ff;opacity:.8 }
        .stButton>button {
            background: linear-gradient(90deg,#2fb86f,#35c06f)!important;color:white!important;
            font-weight:600;border-radius:12px!important;
        }
        .stDownloadButton>button{
            background:rgba(255,255,255,0.06)!important;color:white!important;border-radius:10px!important;
        }
        video{ transform:scaleX(-1)!important;}
        </style>""",unsafe_allow_html=True)

set_background_and_styles()


# ==========================
# PAGE HEADER
# ==========================
header_col1, header_col2 = st.columns([0.9,0.1])
with header_col1:
    st.markdown("<h1 style='margin:0;'>🌱 FarmDoc</h1>",unsafe_allow_html=True)
    st.markdown("<div class='caption'>Detect plant disease, view live farm sensor data, and get an easy-to-follow treatment report in multiple languages.</div>",unsafe_allow_html=True)

st.markdown("---")


# ==========================
# ==========================
# ==========================
# LOAD MODEL FROM GITHUB + SAFE AUGMENTATION DISABLE
# ==========================
import os, urllib.request, tensorflow as tf
import streamlit as st

MODEL_URL = "https://raw.githubusercontent.com/horrorinpeace/DISEASE-DETECTION/main/fix.h5"
MODEL_PATH = "fix.h5"

# ---- NEW FIXED VERSION ----
def disable_augmentation_layers(model):
    for layer in model.layers:
        lname = layer.__class__.__name__.lower()
        if "augment" in lname or "preprocess" in lname or "random" in lname:
            layer.trainable = False
    return model

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    m = tf.keras.models.load_model(MODEL_PATH, compile=False)
    m = disable_augmentation_layers(m)   # now safe and compatible
    return m

try:
    model = load_model()
    CLASS_NAMES = [
        'CURRY POWDERY MILDEW','HEALTHY MILLET','HEALTHY POTATO','HEALTHY RICE',
        'HEALTHY SUGARCANE','HEALTHY TEA LEAF','HEALTHY TOMATO','HEALTHY WHEAT',
        'MILLETS BLAST','MILLETS RUST','POTATO EARLY BLIGHT','POTATO LATE BLIGHT',
        'RICE BACTERIAL BLIGHT','RICE BROWN SPOT','RICE LEAF SMUT','SUGARCANE RED ROT',
        'SUGARCANE RUST','SUGARCANE YELLOW','TEA GRAY BLIGHT','TEA GREEN MIRID BUG',
        'TEA HELOPELTIS','TOMATO LEAF MOLD','TOMATO MOSAIC VIRUS',
        'TOMATO SEPTORIA LEAF SPOT','WHEAT BROWN RUST','WHEAT LOOSE SMUT','WHEAT YELLOW RUST'
    ]
except Exception as e:
    st.error(f"Model failed to load: {e}")
    model=None
    CLASS_NAMES=[]

# ==========================
# SENSOR DATA (PRESSURE ADDED FIELD 6)
# ==========================
def fetch_sensor_data():
    url = "https://api.thingspeak.com/channels/3152731/feeds.json?api_key=8WGWK6AUAF74H6DJ&results=1"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get("feeds"):
            latest=data["feeds"][0]
            return {
                "temperature": latest.get("field1"),
                "humidity": latest.get("field2"),
                "soil_moisture": latest.get("field3"),
                "air_quality": latest.get("field4"),
                "light_intensity": latest.get("field5"),
                "pressure": latest.get("field6"),     # ← ADDED FIELD 6
                "timestamp": latest.get("created_at")
            }
    except:
        pass
    return {"temperature":None,"humidity":None,"soil_moisture":None,"air_quality":None,"light_intensity":None,"pressure":None,"timestamp":None}


# ==========================
LANGUAGE_OPTIONS={
  "English":"English","हिन्दी (Hindi)":"Hindi","বাংলা (Bengali)":"Bengali","தமிழ் (Tamil)":"Tamil",
  "తెలుగు (Telugu)":"Telugu","ಕನ್ನಡ (Kannada)":"Kannada","മലയാളം (Malayalam)":"Malayalam",
  "मराठी (Marathi)":"Marathi","ગુજરાતી (Gujarati)":"Gujarati","ਪੰਜਾਬੀ (Punjabi)":"Punjabi",
  "ଓଡ଼ିଆ (Odia)":"Odia","اردو (Urdu)":"Urdu"
}


# ==========================
# SIDEBAR MENU
# ==========================
st.sidebar.title("Menu")
page=st.sidebar.radio("Go to",["About","AI Detection Panel"])
st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
selected_language_display=st.sidebar.selectbox("Report language",list(LANGUAGE_OPTIONS.keys()))
st.session_state.selected_language=LANGUAGE_OPTIONS[selected_language_display]
api_key=st.sidebar.text_input("🔐 Groq API key",type="password")


# ==========================
# ABOUT
# ==========================
if page=="About":
    st.header("About FarmDoc AI")
    st.markdown("""
    <div class="card">
        <strong>FarmDoc AI</strong> helps farmers detect plant diseases using a phone camera or uploaded images.
        It gives simple, clear advice on what the disease is, how it affects the crop, what actions to take,
        and how to prevent it — available in different languages.
    </div>""",unsafe_allow_html=True)

    st.markdown("### How it works")
    st.markdown("""
    1. Take a photo of the leaf or upload an image.
    2. The AI recommends the most likely disease.
    3. You generate the report.
    4. Download it and share.
    """)


# ==========================
# AI PANEL
# ==========================
elif page=="AI Detection Panel":

    st.header("Step 1 — Capture or Upload Plant Image")
    st.markdown("<div class='card'>Use your phone camera or upload a clear photo.</div>",unsafe_allow_html=True)

    uploaded_file=st.camera_input("📸 Take a photo of your crop leaf")
    if uploaded_file is None:
        uploaded_file=st.file_uploader("Or upload a leaf image",type=["png","jpg","jpeg"])

    if uploaded_file:
        image=Image.open(uploaded_file).convert("RGB")
        st.image(image,caption="🪴 This is the captured image being analyzed",width=300)

        if model:
            img=image.resize((224,224))
            arr=tf.keras.preprocessing.image.img_to_array(img)
            arr=np.expand_dims(arr,0)
            preds=model.predict(arr)
            predicted_class=CLASS_NAMES[np.argmax(preds)]
            st.session_state.predicted_class=predicted_class
            st.success(f"🌿 Detected: {predicted_class}")


    # ==========================
    # LIVE FARM DATA (PRESSURE INCLUDED)
    # ==========================
    st.header("Step 2 — Live Farm Data")
    if st.session_state.auto_refresh_on:
        st_autorefresh(interval=5000,limit=None,key="sensor_refresh")

    sensor=fetch_sensor_data()

    c1,c2,c3,c4,c5,c6 = st.columns(6)     # 6 COLUMNS NOW — NO OTHER CHANGE

    if sensor["temperature"]!=None:  c1.metric("🌡 Temperature",f"{sensor['temperature']} °C")
    if sensor["humidity"]!=None:     c2.metric("💧 Humidity",f"{sensor['humidity']} %")
    if sensor["soil_moisture"]!=None:c3.metric("🌱 Soil Moisture",f"{sensor['soil_moisture']} %")
    if sensor["air_quality"]!=None:  c4.metric("🫁 Air Quality",f"{sensor['air_quality']} AQI")
    if sensor["light_intensity"]!=None:c5.metric("💡 Light Intensity",f"{sensor['light_intensity']} lx")

    if sensor["pressure"]!=None:     c6.metric("🌬 Pressure",f"{sensor['pressure']} hPa")   # FIELD 6 DISPLAY HERE

    st.caption(f"Last updated: {sensor['timestamp']}")


    # ==========================
    # REPORT SECTION (UNCHANGED)
    # ==========================
    st.header("Step 3 — Get Farm Report")
    st.markdown("<div class='card'>The AI will write the report in selected language.</div>",unsafe_allow_html=True)

    if st.button("🧾 Generate Farm Report"):
        st.session_state.report_text=""
        if not api_key:st.error("Enter Groq API key.")
        elif not uploaded_file:st.error("Upload or capture an image.")
        elif model is None:st.error("Model not loaded.")
        elif "predicted_class"not in st.session_state:st.error("No disease prediction.")
        else:
            st.session_state.auto_refresh_on=False
            try:
                with st.spinner("Writing report..."):
                    prompt=f"""
                    You are a helpful agricultural assistant.
                    Write a detailed step-by-step farm advisory report in {st.session_state.selected_language}:
                     Disease: {st.session_state.get('predicted_class')}
                     Temperature: {sensor['temperature']}
                     Humidity: {sensor['humidity']}
                     Soil Moisture: {sensor['soil_moisture']}
                     Air Quality: {sensor['air_quality']}
                     Light Intensity: {sensor['light_intensity']}
                     Pressure: {sensor['pressure']}
                    """

                    url="https://api.groq.com/openai/v1/chat/completions"
                    headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"}
                    data={
                        "model":"meta-llama/llama-4-scout-17b-16e-instruct",
                        "messages":[{"role":"system","content":"You give farm advice."},{"role":"user","content":prompt}],
                        "temperature":0.6,"max_completion_tokens":800,"top_p":1,"stream":False
                    }

                    r=requests.post(url,headers=headers,json=data,timeout=40)
                    raw_text=r.json()["choices"][0]["message"]["content"]
                    st.session_state.report_text=raw_text

                    if st.session_state.selected_language!="English" and is_mostly_english(raw_text):
                        translated=translate_with_google(raw_text,st.session_state.selected_language)
                        if translated:st.session_state.report_text=translated

                    st.success("Report generated!")
            except Exception as e: st.error(f"Error: {e}")
            finally:st.session_state.auto_refresh_on=True


# ==========================
# REPORT DISPLAY + DOWNLOAD
# ==========================
if st.session_state.report_text:
    st.markdown("### 🌿 Your Farm Report")
    st.markdown(f"<div class='card'><pre style='white-space:pre-wrap'>{st.session_state.report_text}</pre></div>",unsafe_allow_html=True)

    txt_bytes=st.session_state.report_text.encode("utf-8")
    st.download_button("📥 Download Farm Report (TXT)",data=txt_bytes,file_name="farm_report.txt",mime="text/plain;charset=utf-8")

    try:
        from docx import Document
        from docx.shared import Pt
        from io import BytesIO

        doc=Document()
        doc.add_heading("Farm Report",level=1)

        for line in st.session_state.report_text.splitlines():
            p=doc.add_paragraph(line)
            for run in p.runs:run.font.size=Pt(12)

        if 'uploaded_file'in locals()and uploaded_file:
            img_bytes=uploaded_file.getbuffer()
            doc.add_page_break()
            doc.add_paragraph("Attached leaf image:")
            doc.add_picture(BytesIO(img_bytes),width=Pt(300))

        f=BytesIO();doc.save(f);f.seek(0)
        st.download_button("📥 Download Farm Report (DOCX)",data=f.read(),file_name="farm_report.docx",mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    except:
        st.warning("DOCX export not available.")


# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("<div class='caption'>FarmDoc © 2025 — Helping Farmers Grow Smarter</div>",unsafe_allow_html=True)


