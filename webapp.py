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

# ================================================================
# LANGUAGE MAP
# ================================================================
LANG_CODE_MAP = {
    "English": "en","Hindi":"hi","Bengali":"bn","Tamil":"ta","Telugu":"te",
    "Kannada":"kn","Malayalam":"ml","Marathi":"mr","Gujarati":"gu",
    "Punjabi":"pa","Odia":"or","Urdu":"ur",
}

def translate_with_google(text,target_lang):
    code=LANG_CODE_MAP.get(target_lang,"en")
    if code=="en":return text
    try:
        r=requests.get("https://translate.googleapis.com/translate_a/single",
        params={"client":"gtx","sl":"en","tl":code,"dt":"t","q":text},timeout=8).json()
        return "".join(i[0] for i in r[0])
    except:return text

def is_mostly_english(t):
    total=sum(c.isalpha() for c in t)
    asc=sum(c.isascii() and c.isalpha() for c in t)
    return asc/total>0.85 if total else False

# ================================================================
# REMOVE RANDOM LAYERS IF PRESENT
# ================================================================
def disable_augmentation_layers(model):
    def identity(self,x,training=False):return x
    def scan(l):
        cname=l.__class__.__name__.lower()
        if any(i in cname for i in["random","flip","rotate","zoom","augment"]):
            l.call=identity;l.trainable=False
        if hasattr(l,"layers"):
            for s in l.layers:scan(s)
    scan(model);return model


# ================================================================
# UI THEME (CLEAN, PROFESSIONAL)
# ================================================================
st.set_page_config(page_title="FarmDoc AI",layout="wide")

st.markdown("""
<style>
body,html, .stApp {background:#f1f3f5; font-family:'Inter',sans-serif;}

/* HEADER BAR */
#hdr{
    background:#0d3d2c;
    padding:30px 10px;
    border-radius:10px;
    text-align:center;
    color:white;
    margin-bottom:25px;
}
#hdr h1{margin:0;font-size:40px;font-weight:700;}
#hdr p{margin-top:6px;font-size:15px;opacity:.9;}

/* WHITE SECTIONS */
.card{
    background:white;
    border-radius:12px;
    padding:22px;
    border:1px solid #e5e5e5;
    box-shadow:0px 4px 14px rgba(0,0,0,0.05);
    margin:14px 0;
}

/* BUTTON STYLE */
.stButton>button{
    background:#1f8c58 !important;
    color:white !important;
    border:none;
    padding:10px 22px;
    border-radius:7px;
    font-weight:600;
}
.stButton>button:hover{background:#13633d !important;}

/* METRIC CARDS */
.metric{
    background:white;
    border-radius:10px;
    padding:14px;
    text-align:center;
    box-shadow:0 3px 11px rgba(0,0,0,0.05);
    border:1px solid #e5e5e5;
    transition:.2s;
}
.metric:hover{transform:scale(1.03);}
.metric-label{font-size:13px;font-weight:600;color:#333;}
.metric-value{font-size:21px;font-weight:700;color:#187a51;margin-top:3px;}

/* SIDEBAR */
section[data-testid="stSidebar"]{
    background:white;
    border-right:1px solid #ccc;
    padding-top:15px;
}
.sidebar-h{font-size:19px;font-weight:650;margin-bottom:10px;}

</style>
""",unsafe_allow_html=True)

# ================================================================
if "report_text" not in st.session_state:st.session_state.report_text=""
if "auto_refresh_on" not in st.session_state:st.session_state.auto_refresh_on=True

# ================================================================
# HEADER
# ================================================================
st.markdown("""
<div id='hdr'>
<h1>FarmDoc AI</h1>
<p>Leaf disease recognition • Live field conditions • Automated treatment guidance</p>
</div>
""",unsafe_allow_html=True)


# ================================================================
# LOAD MODEL
# ================================================================
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
        'TOMATO MOSAIC VIRUS','TOMATO SEPTORIA LEAF SPOT','WHEAT BROWN RUST','WHEAT LOOSE SMUT','WHEAT YELLOW RUST']
except:model=None;CLASS_NAMES=[]


# ================================================================
# SENSOR FETCH
# ================================================================
READ_KEY="SO5QAU5RBCQ15WKD"
def fetch():
    try:
        r=requests.get(f"https://api.thingspeak.com/channels/3152731/feeds.json?api_key={READ_KEY}&results=50",timeout=5).json()
        f=r.get("feeds",[])
        def get(k):return next((x[k]for x in f[::-1] if x.get(k)),None)
        return {"temperature":get("field1"),"humidity":get("field2"),"soil_moisture":get("field3"),
            "air_quality":get("field4"),"light_intensity":get("field5"),"pressure":get("field6"),
            "soil_temperature":get("field7"),"timestamp":f[-1]["created_at"]if f else"—"}
    except:return{a:"—"for a in["temperature","humidity","soil_moisture","air_quality","light_intensity","pressure","soil_temperature","timestamp"]}

# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.markdown("<div class='sidebar-h'>Menu</div>",unsafe_allow_html=True)
page=st.sidebar.radio("",["About","AI Panel"])

lang_map={"English":"English","हिन्दी (Hindi)":"Hindi","বাংলা (Bengali)":"Bengali","தமிழ் (Tamil)":"Tamil",
"తెలుగు (Telugu)":"Telugu","ಕನ್ನಡ (Kannada)":"Kannada","മലയാളം (Malayalam)":"Malayalam",
"मराठी (Marathi)":"Marathi","ગુજરાતી (Gujarati)":"Gujarati","ਪੰਜਾਬੀ (Punjabi)":"Punjabi",
"ଓଡ଼ିଆ (Odia)":"Odia","اردو (Urdu)":"Urdu"}

l=st.sidebar.selectbox("Report Language",list(lang_map.keys()))
st.session_state.selected_language=lang_map[l]

api_key=st.sidebar.text_input("Groq API Key",type="password")


# ================================================================
# PAGE — ABOUT
# ================================================================
if page=="About":
    st.markdown("<h2>About FarmDoc</h2>",unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
    FarmDoc AI helps farmers diagnose plant diseases from leaf images, view live field
    sensor readings and get a complete advisory report for treatment, dosage and prevention.
    </div>
    """,unsafe_allow_html=True)


# ================================================================
# PAGE — AI PANEL
# ================================================================
elif page=="AI Panel":

    st.markdown("<h2>Step 1: Upload Plant Leaf Image</h2>",unsafe_allow_html=True)
    st.markdown("<div class='card'>Upload / capture a leaf image with clear visibility.</div>",unsafe_allow_html=True)

    f=st.camera_input("Capture Photo") or st.file_uploader("Or Upload",type=["png","jpg","jpeg"])

    if f:
        img=Image.open(f).convert("RGB")
        st.image(img,width=320)

        if model:
            x=img.resize((224,224))
            y=tf.keras.preprocessing.image.img_to_array(x)[None]
            pred=model.predict(y)
            st.session_state.predicted_class=CLASS_NAMES[np.argmax(pred)]
            st.success("Disease Detected: "+st.session_state.predicted_class)

    st.markdown("<h2>Step 2: Live Field Sensors</h2>",unsafe_allow_html=True)
    if st.session_state.auto_refresh_on:st_autorefresh(interval=5500,key="refresh")

    d=fetch()
    cols=st.columns(7)
    for c,lbl,val in zip(cols,
    ["Temp(°C)","Humidity(%)","Soil Moisture","Air Quality","Light(lx)","Pressure(hPa)","Soil Temp"],
    [d['temperature'],d['humidity'],d['soil_moisture'],d['air_quality'],d['light_intensity'],d['pressure'],d['soil_temperature']]):
        c.markdown(f"""
        <div class='metric'>
            <div class='metric-label'>{lbl}</div>
            <div class='metric-value'>{val}</div>
        </div>""",unsafe_allow_html=True)

    st.write(f"Last Update: {d['timestamp']}")

    st.markdown("<h2>Step 3: Generate Report</h2>",unsafe_allow_html=True)

    if st.button("Generate Treatment Report"):
        if not api_key:st.error("Enter API Key")
        elif not f:st.error("Upload image first")
        elif "predicted_class" not in st.session_state:st.error("No disease detected")
        else:
            st.session_state.auto_refresh_on=False

            with st.spinner("Generating advisory..."):
                prompt=f"""
Write a detailed farming advisory in {st.session_state.selected_language}.
Use this structure strictly:

- Disease Name:
- What It Means:
- Cause:
- Recommended Spray + Dosage (per litre, per knapsack, per acre):
- Step-by-step Application:
- Spray Frequency and Interval:
- Safety Precautions:
- Preventive Measures:

Live Conditions:
Temp={d['temperature']} Humidity={d['humidity']} Soil Moisture={d['soil_moisture']}
Soil Temp={d['soil_temperature']} AQI={d['air_quality']} Light={d['light_intensity']} Pressure={d['pressure']}
"""

                r=requests.post("https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
                    json={"model":"meta-llama/llama-4-scout-17b-16e-instruct",
                          "messages":[{"role":"system","content":"You are a farming expert."},
                                      {"role":"user","content":prompt}],
                          "temperature":0.6,"max_completion_tokens":800})
                
                out=r.json()["choices"][0]["message"]["content"]
                st.session_state.report_text=translate_with_google(out,st.session_state.selected_language)\
                                            if st.session_state.selected_language!="English" and is_mostly_english(out) else out
            st.success("Report Ready.")

# ================================================================
# SHOW REPORT
# ================================================================
if st.session_state.report_text:
    st.markdown("<h3>Farm Advisory Report</h3>",unsafe_allow_html=True)
    st.markdown(f"<div class='card'><pre style='white-space:pre-wrap'>{st.session_state.report_text}</pre></div>",unsafe_allow_html=True)

    st.download_button("Download TXT",st.session_state.report_text.encode(),"farm_report.txt")
