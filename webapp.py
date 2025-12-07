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

# =====================================================================
# LANGUAGE MAP
# =====================================================================
LANG_CODE_MAP={
    "English":"en","Hindi":"hi","Bengali":"bn","Tamil":"ta","Telugu":"te",
    "Kannada":"kn","Malayalam":"ml","Marathi":"mr","Gujarati":"gu",
    "Punjabi":"pa","Odia":"or","Urdu":"ur"
}

def translate_with_google(text,target_lang):
    code=LANG_CODE_MAP.get(target_lang,"en")
    if code=="en":return text
    try:
        r=requests.get("https://translate.googleapis.com/translate_a/single",
                       params={"client":"gtx","sl":"en","tl":code,"dt":"t","q":text},
                       timeout=8).json()
        return "".join(i[0] for i in r[0])
    except:return text

def is_mostly_english(t):
    total=sum(c.isalpha() for c in t)
    asc=sum(c.isascii() and c.isalpha() for c in t)
    return asc/total>0.85 if total else False

# =====================================================================
# REMOVE RANDOM LAYERS IF ANY
# =====================================================================
def disable_augmentation_layers(model):
    def identity(self,x,training=False):return x
    def scan(layer):
        cname=layer.__class__.__name__.lower()
        if any(k in cname for k in["random","flip","rotate","zoom","augment"]):
            layer.call=identity;layer.trainable=False
        if hasattr(layer,"layers"):
            for s in layer.layers:scan(s)
    scan(model);return model

# =====================================================================
# UI THEME (Professional Minimal CropTech UI)
# =====================================================================
st.set_page_config(page_title="FarmDoc AI",layout="wide")

st.markdown("""
<style>

html, body, .stApp {
    background-color:#f6f7f9;
    font-family:'Inter',sans-serif;
    color:#111;
}

/* === Header === */
.header-box {
    background:#114225;
    padding:38px 10px;
    border-radius:10px;
    color:white;
    margin-bottom:30px;
}
.header-title {font-size:38px;font-weight:700;margin-bottom:6px;}
.header-sub {opacity:.88;font-size:16px;}

/* === Card Sections === */
.block {
    background:white;
    border-radius:10px;
    padding:24px;
    border:1px solid #dcdcdc;
    margin-bottom:22px;
    box-shadow:0 2px 10px rgba(0,0,0,0.04);
}

/* === Buttons === */
.stButton>button {
    background:#18764b!important;
    color:white!important;
    padding:10px 26px;
    font-weight:600;
    border-radius:6px;
    border:none;
    font-size:15px;
}
.stButton>button:hover {background:#0d5936!important;}

/* === Sensor Tiles === */
.metric-tile {
    background:white;
    border-radius:8px;
    padding:14px;
    border:1px solid #e0e0e0;
    text-align:center;
    box-shadow:0 1px 6px rgba(0,0,0,0.05);
}
.metric-label {font-size:13px;font-weight:600;color:#333;}
.metric-value {font-size:20px;font-weight:700;color:#195c3d;margin-top:3px;}

/* Sidebar */
[data-testid="stSidebar"] {
    background:white;
    border-right:1px solid #ccc;
    padding-top:18px;
}
.sidebar-title {font-size:18px;font-weight:650;margin-bottom:6px;}

</style>
""",unsafe_allow_html=True)


# =====================================================================
# SESSION INIT
# =====================================================================
if "report_text" not in st.session_state: st.session_state.report_text=""
if "auto_refresh_on" not in st.session_state: st.session_state.auto_refresh_on=True

# =====================================================================
# HEADER
# =====================================================================
st.markdown("""
<div class="header-box">
    <div class="header-title">FarmDoc AI</div>
    <div class="header-sub">Plant disease recognition • Live farm telemetry • Action-ready treatment reports</div>
</div>
""",unsafe_allow_html=True)

# =====================================================================
# MODEL LOAD
# =====================================================================
MODEL_URL="https://raw.githubusercontent.com/horrorinpeace/DISEASE-DETECTION/main/fix.h5"
MODEL_PATH="fix.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL,MODEL_PATH)
    m=tf.keras.models.load_model(MODEL_PATH,compile=False)
    return disable_augmentation_layers(m)

try:
    model=load_model()
    CLASS_NAMES=[ 'CURRY POWDERY MILDEW','HEALTHY MILLET','HEALTHY POTATO','HEALTHY RICE','HEALTHY SUGARCANE','HEALTHY TEA LEAF',
        'HEALTHY TOMATO','HEALTHY WHEAT','MILLETS BLAST','MILLETS RUST','POTATO EARLY BLIGHT','POTATO LATE BLIGHT',
        'RICE BACTERIAL BLIGHT','RICE BROWN SPOT','RICE LEAF SMUT','SUGARCANE RED ROT','SUGARCANE RUST',
        'SUGARCANE YELLOW','TEA GRAY BLIGHT','TEA GREEN MIRID BUG','TEA HELOPELTIS','TOMATO LEAF MOLD',
        'TOMATO MOSAIC VIRUS','TOMATO SEPTORIA LEAF SPOT','WHEAT BROWN RUST','WHEAT LOOSE SMUT','WHEAT YELLOW RUST' ]
except:model=None;CLASS_NAMES=[]

# =====================================================================
# SENSOR FETCH
# =====================================================================
READ_KEY="SO5QAU5RBCQ15WKD"
def fetch():
    try:
        r=requests.get(f"https://api.thingspeak.com/channels/3152731/feeds.json?api_key={READ_KEY}&results=50",timeout=5).json()
        feeds=r.get("feeds",[])
        def last(x):return next((f[x] for f in feeds[::-1] if f.get(x)),None)
        return {"temperature":last("field1"),"humidity":last("field2"),"soil_moisture":last("field3"),
        "air_quality":last("field4"),"light_intensity":last("field5"),"pressure":last("field6"),
        "soil_temperature":last("field7"),"timestamp":feeds[-1]["created_at"] if feeds else "—"}
    except:return{v:"—"for v in["temperature","humidity","soil_moisture","air_quality","light_intensity","pressure","soil_temperature","timestamp"]}

# =====================================================================
# SIDEBAR
# =====================================================================
st.sidebar.markdown("<div class='sidebar-title'>Navigation</div>",unsafe_allow_html=True)
page=st.sidebar.radio("",["About","AI Detection Panel"])

lang_map={"English":"English","हिन्दी (Hindi)":"Hindi","বাংলা (Bengali)":"Bengali","தமிழ் (Tamil)":"Tamil",
"తెలుగు (Telugu)":"Telugu","ಕನ್ನಡ (Kannada)":"Kannada","മലയാളം (Malayalam)":"Malayalam",
"मराठी (Marathi)":"Marathi","ગુજરાતી (Gujarati)":"Gujarati","ਪੰਜਾਬੀ (Punjabi)":"Punjabi",
"ଓଡ଼ିଆ (Odia)":"Odia","اردو (Urdu)":"Urdu"}

selection=st.sidebar.selectbox("Report Language",list(lang_map.keys()))
st.session_state.selected_language=lang_map[selection]

api_key=st.sidebar.text_input("Groq API Key",type="password")

# =====================================================================
# ABOUT PAGE
# =====================================================================
if page=="About":
    st.markdown("<h3>About</h3>",unsafe_allow_html=True)
    st.markdown("""
    <div class='block'>
    FarmDoc AI helps farmers detect crop diseases using a leaf scan, monitors
    farm conditions in real-time and produces accurate dosage-based treatment guidance.
    </div>
    """,unsafe_allow_html=True)

# =====================================================================
# AI PANEL
# =====================================================================
elif page=="AI Detection Panel":

    st.markdown("## Step 1: Upload Leaf Image")
    st.markdown("<div class='block'>Upload/scan the leaf clearly for best detection accuracy.</div>",unsafe_allow_html=True)

    upl=st.camera_input("Capture Image") or st.file_uploader("Or Upload from device",type=["png","jpg","jpeg"])

    if upl:
        img=Image.open(upl).convert("RGB")
        st.image(img,width=350)

        if model:
            r_img=img.resize((224,224))
            arr=tf.keras.preprocessing.image.img_to_array(r_img)[None]
            pred=model.predict(arr)
            st.session_state.predicted_class=CLASS_NAMES[np.argmax(pred)]
            st.success("Disease Detected: "+st.session_state.predicted_class)

    st.markdown("## Step 2: Live Field Telemetry")

    if st.session_state.auto_refresh_on:
        st_autorefresh(interval=5500,key="refresh")

    data=fetch()
    c=st.columns(7)
    labels=["Temp °C","Humidity %","Soil Moisture","Air Quality","Light (lx)","Pressure hPa","Soil Temp °C"]
    values=[data['temperature'],data['humidity'],data['soil_moisture'],data['air_quality'],
            data['light_intensity'],data['pressure'],data['soil_temperature']]

    for col,lbl,val in zip(c,labels,values):
        col.markdown(f"<div class='metric-tile'><div class='metric-label'>{lbl}</div><div class='metric-value'>{val}</div></div>",
                     unsafe_allow_html=True)

    st.caption("Last updated: "+str(data['timestamp']))

    st.markdown("## Step 3: Generate Advisory Report")

    if st.button("Generate Report"):
        if not api_key:st.error("Enter API Key")
        elif not upl:st.error("Upload an image first")
        elif "predicted_class" not in st.session_state:st.error("No disease detected")
        else:
            st.session_state.auto_refresh_on=False

            with st.spinner("Generating report..."):
                prompt=f"""
Write professional farming treatment advice in {st.session_state.selected_language}.
Include:

- Disease Name
- What It Means
- Cause
- Spray Recommendation + Dosage
- Step-by-step Treatment Method
- Spray Frequency + Interval
- Safety Precautions
- Preventive Measures

Use live farm conditions:
Temp={data['temperature']}, Humidity={data['humidity']}, Soil Moisture={data['soil_moisture']}
Soil Temp={data['soil_temperature']}, AQI={data['air_quality']}, Light={data['light_intensity']}, Pressure={data['pressure']}
"""

                r=requests.post("https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
                    json={"model":"meta-llama/llama-4-scout-17b-16e-instruct",
                    "messages":[{"role":"system","content":"You are a farming advisor."},
                                {"role":"user","content":prompt}],
                    "temperature":0.6,"max_completion_tokens":800})

                output=r.json()["choices"][0]["message"]["content"]
                st.session_state.report_text=translate_with_google(output,st.session_state.selected_language)\
                                            if st.session_state.selected_language!="English" and is_mostly_english(output) else output

            st.success("Report Generated")

# =====================================================================
# SHOW REPORT
# =====================================================================
if st.session_state.report_text:
    st.markdown("### Advisory Report")
    st.markdown(f"<div class='block'><pre style='white-space:pre-wrap'>{st.session_state.report_text}</pre></div>",
                unsafe_allow_html=True)

    st.download_button("Download as Text",st.session_state.report_text.encode(),"farm_report.txt")
