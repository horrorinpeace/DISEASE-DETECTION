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
import urllib.request

# ==========================
# TRANSLATION FALLBACK
# ==========================
LANG_CODE_MAP = {
    "English": "en","Hindi": "hi","Bengali": "bn","Tamil": "ta","Telugu": "te",
    "Kannada": "kn","Malayalam": "ml","Marathi": "mr","Gujarati": "gu",
    "Punjabi": "pa","Odia": "or","Urdu": "ur",
}

def translate_with_google(text, target_language_name):
    code = LANG_CODE_MAP.get(target_language_name, "en")
    if code=="en": return text
    try:
        url="https://translate.googleapis.com/translate_a/single"
        params={"client":"gtx","sl":"en","tl":code,"dt":"t","q":text}
        r=requests.get(url,params=params,timeout=15); r.raise_for_status()
        data=r.json()
        translated="".join(chunk[0] for chunk in data[0] if chunk[0])
        return translated if translated else text
    except: return text

def is_mostly_english(text:str)->bool:
    if not text: return False
    total=sum(c.isalpha() for c in text)
    if total==0: return False
    ascii_count=sum(c.isascii() and c.isalpha() for c in text)
    return (ascii_count/total)>0.85


# ==========================
# DISABLE AUGMENTATION
# ==========================
def disable_augmentation_layers(model):
    disabled=[]
    def identity_call(self,inputs,training=False): return inputs
    def walk(layer):
        lname=getattr(layer,"name","").lower()
        cname=layer.__class__.__name__.lower()
        if "augmentation" in lname or any(k in cname for k in["random","flip","rotate","rotation","zoom","contrast","crop","augment","preprocess"]):
            try:
                layer.call=MethodType(identity_call,layer)
                layer.trainable=False; disabled.append(layer.name)
            except: pass
        if hasattr(layer,"layers"):
            for l in layer.layers: walk(l)
    walk(model); return model


# ==========================
# PAGE THEME UI — ENHANCED
# ==========================
st.set_page_config(page_title="🌱 FarmDoc AI",layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>

/* Entire App BG */
.stApp{
    background:linear-gradient(120deg,#0e1523,#15263a,#1f3a54);
    color:#ECF5FF;
    font-family:'Segoe UI',sans-serif;
}

/* Main container card */
.block-container{
    padding-top:20px !important;
    border-radius:18px;
    margin-top:10px;
}

/* Section Cards */
.card{
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:16px;
    padding:18px;
    margin-bottom:15px;
    box-shadow:0 0 20px rgba(0,0,0,.35);
    transition:.2s;
}
.card:hover{
    transform:scale(1.01);
    box-shadow:0 0 28px rgba(0,0,0,.45);
}

/* Headings */
h1,h2,h3,h4,h5{
    font-weight:650;
    color:#E9F6FF !important;
}

/* Buttons */
.stButton>button{
    background:linear-gradient(90deg,#28c76f,#32d67c);
    color:white !important;
    border-radius:12px;
    padding:12px 22px;
    font-size:16px;
    font-weight:600;
    border:none;
}
.stDownloadButton>button{
    background:#ffffff10;
    border:1px solid #ffffff22;
    padding:10px 18px;
    border-radius:10px;
}

/* Metric Box UI */
[data-testid="stMetricValue"]{
    font-size:22px;
    font-weight:700;
}

/* Sidebar */
.sidebar .sidebar-content {padding-top:20px;}
</style>
""",unsafe_allow_html=True)


# ==========================
# PREVENT SESSION CRASH
# ==========================
if "report_text" not in st.session_state: st.session_state.report_text=""
if "auto_refresh_on" not in st.session_state: st.session_state.auto_refresh_on=True


# ==========================
# HEADER
# ==========================
st.markdown("""
<div style="
text-align:center;
padding:25px;
border-radius:20px;
background:linear-gradient(120deg,#17304c,#234b6d);
border:1px solid #ffffff18;
box-shadow:0 0 25px rgba(0,0,0,.4);
">
<h1 style="font-size:45px;margin-bottom:10px;">🌾 <b>FarmDoc AI</b></h1>
<span style="font-size:18px;opacity:.85;">Detect diseases • Get AI advisory • Live farm monitoring</span>
</div>
<br>
""",unsafe_allow_html=True)


# ==========================
# LOAD MODEL
# ==========================
MODEL_URL="https://raw.githubusercontent.com/horrorinpeace/DISEASE-DETECTION/main/fix.h5"
MODEL_PATH="fix.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models",exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL,MODEL_PATH)
    m=tf.keras.models.load_model(MODEL_PATH,compile=False)
    return disable_augmentation_layers(m)

try:
    model=load_model()
    CLASS_NAMES=[
        'CURRY POWDERY MILDEW','HEALTHY MILLET','HEALTHY POTATO','HEALTHY RICE','HEALTHY SUGARCANE','HEALTHY TEA LEAF',
        'HEALTHY TOMATO','HEALTHY WHEAT','MILLETS BLAST','MILLETS RUST','POTATO EARLY BLIGHT','POTATO LATE BLIGHT',
        'RICE BACTERIAL BLIGHT','RICE BROWN SPOT','RICE LEAF SMUT','SUGARCANE RED ROT','SUGARCANE RUST',
        'SUGARCANE YELLOW','TEA GRAY BLIGHT','TEA GREEN MIRID BUG','TEA HELOPELTIS','TOMATO LEAF MOLD',
        'TOMATO MOSAIC VIRUS','TOMATO SEPTORIA LEAF SPOT','WHEAT BROWN RUST','WHEAT LOOSE SMUT','WHEAT YELLOW RUST'
    ]
except Exception as e:
    st.error(f"Model failed to load: {e}")
    model=None; CLASS_NAMES=[]


# ==========================
# READ KEY
# ==========================
READ_KEY = "SO5QAU5RBCQ15WKD"


def fetch_sensor_data():
    url=f"https://api.thingspeak.com/channels/3152731/feeds.json?api_key={READ_KEY}&results=40"
    try:
        res=requests.get(url,timeout=5).json()
        feeds=res.get("feeds",[])
        def latest(field):
            for row in reversed(feeds):
                if row.get(field) not in (None,""): return row.get(field)
            return "None"
        timestamp = feeds[-1].get("created_at","—") if feeds else "—"
        return {
            "temperature":latest("field1"),
            "humidity":latest("field2"),
            "soil_moisture":latest("field3"),
            "air_quality":latest("field4"),
            "light_intensity":latest("field5"),
            "pressure":latest("field6"),
            "soil_temperature":latest("field7"),
            "timestamp":timestamp
        }
    except:
        return {"temperature":"None","humidity":"None","soil_moisture":"None",
                "air_quality":"None","light_intensity":"None","pressure":"None",
                "soil_temperature":"None","timestamp":"—"}


# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("⚙ Settings")
page=st.sidebar.radio("📍 Navigation",["About","AI Detection Panel"])

LANGUAGE_OPTIONS={
    "English":"English","हिन्दी (Hindi)":"Hindi","বাংলা (Bengali)":"Bengali","தமிழ் (Tamil)":"Tamil",
    "తెలుగు (Telugu)":"Telugu","ಕನ್ನಡ (Kannada)":"Kannada","മലയാളം (Malayalam)":"Malayalam",
    "मराठी (Marathi)":"Marathi","ગુજરાતી (Gujarati)":"Gujarati","ਪੰਜਾਬੀ (Punjabi)":"Punjabi",
    "ଓଡ଼ିଆ (Odia)":"Odia","اردو (Urdu)":"Urdu"
}
selected_language_display=st.sidebar.selectbox("🌐 Report Language",list(LANGUAGE_OPTIONS.keys()))
st.session_state.selected_language=LANGUAGE_OPTIONS[selected_language_display]

api_key=st.sidebar.text_input("🔐 Enter Groq API Key",type="password")


# ==========================
# ABOUT PAGE
# ==========================
if page=="About":
    st.markdown("""<h2>📘 About FarmDoc AI</h2>""",unsafe_allow_html=True)
    st.markdown("""
<div class="card">
FarmDoc AI is a farmer-friendly AI system that detects plant diseases using a leaf image.
It also shows live farm sensor values and generates a complete treatment plan in any language.
</div>
""",unsafe_allow_html=True)


    st.markdown("""<h3>How it Works</h3>""")
    st.info("""
1. Capture or upload plant leaf photo  
2. AI identifies disease using deep learning  
3. You get a full treatment & prevention report  
4. Download and share with farmers  
""")


# ==========================
# AI DETECTION PANEL
# ==========================
elif page=="AI Detection Panel":

    st.markdown("""<h2>📸 Step 1 — Capture / Upload Plant Image</h2>""",unsafe_allow_html=True)
    st.markdown("""<div class='card'>Upload a clear leaf photo with good lighting for best accuracy.</div>""",unsafe_allow_html=True)

    uploaded_file=st.camera_input("Take a picture using camera")
    if uploaded_file is None:
        uploaded_file=st.file_uploader("or Upload Image",type=["png","jpg","jpeg"])

    if uploaded_file:
        image=Image.open(uploaded_file).convert("RGB")
        st.image(image,caption="Uploaded Leaf",width=350)

        if model:
            img=image.resize((224,224))
            arr=tf.keras.preprocessing.image.img_to_array(img)
            arr=np.expand_dims(arr,0)
            pred=model.predict(arr)
            st.session_state.predicted_class=CLASS_NAMES[np.argmax(pred)]
            st.success(f"🌿 Detected: {st.session_state.predicted_class}")


    st.markdown("""<h2>📊 Step 2 — Live Farm Sensor Data</h2>""",unsafe_allow_html=True)

    if st.session_state.auto_refresh_on:
        st_autorefresh(interval=5000,limit=None,key="sensor_refresh")

    data=fetch_sensor_data()

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("🌡 Temp",f"{data['temperature']} °C")
    c2.metric("💧 Humidity",f"{data['humidity']} %")
    c3.metric("🌱 Soil Moist",f"{data['soil_moisture']} %")
    c4.metric("🫁 Air Quality",f"{data['air_quality']} AQI")
    c5.metric("💡 Light",f"{data['light_intensity']} lx")
    c6.metric("🌬 Pressure",f"{data['pressure']} hPa")
    c7.metric("🌡 Soil Temp",f"{data['soil_temperature']} °C")

    st.caption(f"Last updated: {data['timestamp']}")


    # ==========================
    # REPORT GENERATION
    # ==========================
    st.markdown("""<h2>🧾 Step 3 — Generate AI Report</h2>""",unsafe_allow_html=True)
    st.markdown("""<div class='card'>FarmDoc will prepare a complete advisory report.</div>""",unsafe_allow_html=True)

    if st.button("Generate Full Treatment Report"):
        st.session_state.report_text=""
        if not api_key: st.error("Enter API Key.")
        elif not uploaded_file: st.error("Upload an image.")
        elif model is None: st.error("Model not loaded.")
        elif "predicted_class" not in st.session_state: st.error("No prediction detected.")
        else:
            st.session_state.auto_refresh_on=False
            try:
                with st.spinner("⏳ AI preparing report..."):
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
                      • sprayer type + nozzle type
                      • measuring tools
                      • gloves, goggles, face mask, etc.
                    - Fill every bullet completely.

                    Live Sensor Conditions:
                    Temperature = {data['temperature']} °C
                    Humidity = {data['humidity']} %
                    Soil Moisture = {data['soil_moisture']} %
                    Soil Temperature = {data['soil_temperature']} °C
                    Air Quality Index = {data['air_quality']}
                    Light Intensity = {data['light_intensity']} lx
                    Pressure = {data['pressure']} hPa

                    Format Output Strictly:

                    - Disease Name:
                    - What It Means:
                    - Cause:
                    - Name of spray to be used & Amount to be sprayed:
                    - Tools and Materialy Step Process For Treatment (with exact measurements and timing):
                    - How many times to spray & gap between sprays:
                    - Safety Precautions for Farmers:
                    - Prevention Tips Needed (with quantities):
                    """

                    url="https://api.groq.com/openai/v1/chat/completions"
                    headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"}
                    payload={
                        "model":"meta-llama/llama-4-scout-17b-16e-instruct",
                        "messages":[{"role":"system","content":"You give farm advice."},{"role":"user","content":prompt}],
                        "temperature":0.6,"max_completion_tokens":800,"top_p":1,"stream":False
                    }

                    r=requests.post(url,headers=headers,json=payload,timeout=40)
                    output=r.json()["choices"][0]["message"]["content"]

                    st.session_state.report_text = (
                        translate_with_google(output,st.session_state.selected_language)
                        if st.session_state.selected_language!="English" and is_mostly_english(output)
                        else output
                    )
                st.success("Report Ready!")

            finally: st.session_state.auto_refresh_on=True

# ==========================
# DISPLAY & DOWNLOAD REPORT
# ==========================
if st.session_state.report_text:
    st.markdown("""<h2>📥 Farm Advisory Report</h2>""",unsafe_allow_html=True)
    st.markdown(f"<div class='card'><pre style='white-space:pre-wrap'>{st.session_state.report_text}</pre></div>",unsafe_allow_html=True)

    st.download_button("💾 DOWNLOAD TXT",st.session_state.report_text.encode(),file_name="farm_report.txt")

    try:
        from docx import Document
        from docx.shared import Pt
        from io import BytesIO

        d=Document(); d.add_heading("Farm Report",level=1)
        for line in st.session_state.report_text.splitlines():
            p=d.add_paragraph(line)
            for run in p.runs: run.font.size=Pt(12)

        if 'uploaded_file' in locals():
            d.add_page_break(); d.add_paragraph("Attached Leaf Image:")
            d.add_picture(BytesIO(uploaded_file.getbuffer()),width=Pt(300))

        buf=BytesIO(); d.save(buf); buf.seek(0)
        st.download_button("📄 DOWNLOAD DOCX",buf.read(),"farm_report.docx")
    except:
        st.warning("DOCX export unavailable on this device.")

st.markdown("<br><center>🌱 FarmDoc AI © 2025 — Helping Farmers Grow Smarter</center><br>",unsafe_allow_html=True)
