import streamlit as st
import pandas as pd
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# Unduh model dari Hugging Face
model_path = hf_hub_download(repo_id="barisetiawan51/stacking-model-100", filename="stacking_model_100_compressed.pkl")
scaler_path = hf_hub_download(repo_id="barisetiawan51/stacking-model-100", filename="scaler.pkl")

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

def jelaskan_prediksi(data):
    penjelasan = []
    bmi = data['bmi'].iloc[0]
    tekanan_denyut_nadi = data['tekanan_denyut_nadi'].iloc[0]
    tekanan_arteri_ratarata = data['tekanan_arteri_ratarata'].iloc[0]
    sys_dsys_ratio = data['sys_dsys_ratio'].iloc[0]
    cholesterol = data['cholesterol'].iloc[0]
    gluc = data['gluc'].iloc[0]
    smoke = data['smoke'].iloc[0]
    alco = data['alco'].iloc[0]
    active = data['active'].iloc[0]

    if tekanan_denyut_nadi > 40:
        penjelasan.append("Denyut nadi tinggi")
    if tekanan_arteri_ratarata > 100:
        penjelasan.append("Tekanan arteri rata-rata tinggi")
    if sys_dsys_ratio > 1.5:
        penjelasan.append("Rasio tekanan sistolik dan diastolik tidak normal")
    if cholesterol == 2:
        penjelasan.append("Kolesterol di atas normal")
    elif cholesterol == 3:
        penjelasan.append("Kolesterol sangat di atas normal")
    if gluc > 1:
        penjelasan.append("Tingkat glukosa tinggi")
    if smoke == 1:
        penjelasan.append("Kebiasaan merokok")
    if alco == 1:
        penjelasan.append("Konsumsi alkohol")
    if active == 0:
        penjelasan.append("Kurang aktivitas fisik")
    if 30 <= bmi < 35:
        penjelasan.append(f"BMI = {bmi:.1f}, Obesitas Kelas I: Peningkatan risiko penyakit kardiovaskular.")
    elif 35 <= bmi < 40:
        penjelasan.append(f"BMI = {bmi:.1f}, Obesitas Kelas II: Risiko tinggi terkena hipertensi, diabetes, dan dislipidemia.")
    elif bmi >= 40:
        penjelasan.append(f"BMI = {bmi:.1f}, Obesitas Kelas III: Risiko penyakit kardiovaskular sangat tinggi.")
    
    return penjelasan

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
        penjelasan = jelaskan_prediksi(data_input)
        st.write("Alasan: ")
        for alasan in penjelasan:
            st.write(f"- {alasan}")

if __name__ == "__main__":
    app()
