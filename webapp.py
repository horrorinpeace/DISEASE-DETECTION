# webapp.py
import streamlit as st
from PIL import Image
import pandas as pd
import io
from fpdf import FPDF
import openai
import threading
from flask import Flask, request, jsonify

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
        /* Remove white background from the main container */
        .block-container {{
            background-color: transparent !important;
        }}
        /* Text styling: white text with black outline */
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
    flask_app.run(host="0.0.0.0", port=8502, debug=False, use_reloader=False)

# Run Flask in background thread
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
    - Processes images via AI  
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
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        st.image(image, caption="Captured / Uploaded Image", use_container_width=True)

        # Fake detection results
        df_results = pd.DataFrame({
            "Feature": ["Leaf Spot", "Discoloration", "Pest Damage"],
            "Probability": [0.75, 0.20, 0.05]
        })
        st.table(df_results)

    st.subheader("📡 Live Sensor Data from ESP32")
    if all(v is not None for v in shared_data.values()):
        df_sensor = pd.DataFrame([shared_data])
        st.table(df_sensor)
    else:
        st.warning("Waiting for data from ESP32...")

    if st.button("Generate Lab Report"):
        if uploaded_file is None:
            st.error("Please upload or capture an image first.")
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

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "AI Lab Report", ln=True, align="C")
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 8, report_text)
            uploaded_file.seek(0)
            pdf.image(uploaded_file, x=10, y=None, w=100)
            pdf_bytes = io.BytesIO()
            pdf.output(pdf_bytes)
            pdf_bytes.seek(0)
            st.download_button("📥 Download PDF", pdf_bytes, "lab_report.pdf")

st.markdown("---")
st.markdown("© 2025 AI Detection Lab — Built with ❤️ using Streamlit.")
