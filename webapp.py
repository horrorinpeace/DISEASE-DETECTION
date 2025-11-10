import streamlit as st
import time
import requests

THINGSPEAK_CHANNEL_ID = "3152731"
READ_API_KEY = "8WGWK6AUAF74H6DJ"

def fetch_sensor_data():
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if "feeds" in data and len(data["feeds"]) > 0:
            latest = data["feeds"][0]
            return {
                "temperature": latest["field1"],
                "humidity": latest["field2"],
                "soil_moisture": latest["field3"],
                "timestamp": latest["created_at"]
            }
    except Exception:
        pass
    return {"temperature": None, "humidity": None, "soil_moisture": None, "timestamp": None}


st.title("🌱 Live Sensor Dashboard")

placeholder = st.empty()

while True:
    sensor_data = fetch_sensor_data()
    with placeholder.container():
        st.metric("🌡️ Temperature (°C)", sensor_data["temperature"])
        st.metric("💧 Humidity (%)", sensor_data["humidity"])
        st.metric("🌿 Soil Moisture (%)", sensor_data["soil_moisture"])
        st.caption(f"Last updated: {sensor_data['timestamp']}")
    time.sleep(20)  # 🔁 update every 20 seconds
