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
        'HEALTHY TE
