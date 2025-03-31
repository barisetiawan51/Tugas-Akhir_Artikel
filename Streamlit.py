import streamlit as st
import pandas as pd
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# Hugging Face repository
HF_REPO_ID = "barisetiawan51/stacking-model"
MODEL_FILENAME = "stacking_model.pkl"
SCALER_FILENAME = "scaler.pkl"

# Unduh model dari Hugging Face jika belum ada
model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
scaler_path = hf_hub_download(repo_id=HF_REPO_ID, filename=SCALER_FILENAME)

# Memuat model stacking dan scaler
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"Gagal memuat model atau scaler: {e}")
    st.stop()

# Daftar fitur yang digunakan dalam model
fitur = ['gender', 'age_years', 'bmi', 'tekanan_denyut_nadi', 'tekanan_arteri_ratarata', 'sys_dsys_ratio', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

def prediksi_resiko(data):
    data_scaled = scaler.transform(data)
    prediksi = model.predict(data_scaled)
    return prediksi

def app():
    st.title("Deteksi Risiko Kardiovaskular")

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Jenis Kelamin", [0, 1], format_func=lambda x: "Pria" if x == 0 else "Wanita")
    with col2:
        age_years = st.number_input("Usia (tahun)", min_value=0, step=1)
    with col3:
        bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    with col1:
        tekanan_denyut_nadi = st.number_input("Tekanan Denyut Nadi", min_value=0, step=1)
    with col2:
        tekanan_arteri_ratarata = st.number_input("Tekanan Arteri Rata-rata", min_value=0.0, step=0.1)
    with col3:
        sys_dsys_ratio = st.number_input("Rasio Tekanan Sistolik-Diastolik", min_value=0.0, step=0.01)
    with col1:
        cholesterol = st.selectbox("Kolesterol", [1, 2, 3], format_func=lambda x: "Normal" if x == 1 else "Di atas Normal" if x == 2 else "Sangat Di atas Normal")
    with col2:
        gluc = st.selectbox("Glukosa", [1, 2, 3], format_func=lambda x: "Normal" if x == 1 else "Di atas Normal" if x == 2 else "Sangat Di atas Normal")
    with col3:
        smoke = st.selectbox("Merokok", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
    with col1:
        alco = st.selectbox("Konsumsi Alkohol", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
    with col2:
        active = st.selectbox("Aktivitas Fisik", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
    
    data_input = pd.DataFrame({
        'gender': [gender],
        'age_years': [age_years],
        'bmi': [bmi],
        'tekanan_denyut_nadi': [tekanan_denyut_nadi],
        'tekanan_arteri_ratarata': [tekanan_arteri_ratarata],
        'sys_dsys_ratio': [sys_dsys_ratio],
        'cholesterol': [cholesterol],
        'gluc': [gluc],
        'smoke': [smoke],
        'alco': [alco],
        'active': [active]
    })

    if st.button("Prediksi Risiko Kardiovaskular"):
        prediksi = prediksi_resiko(data_input)
        st.write("Prediksi:", "Berisiko Terkena" if prediksi[0] == 1 else "Tidak Berisiko Terkena")

if __name__ == "__main__":
    app()