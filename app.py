import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Konfigurasi halaman
st.set_page_config(page_title="Aplikasi Deteksi Cacat", layout="centered")
st.title("Inspeksi Dunia")

# Memuat model (cache agar tidak dimuat berulang kali)
@st.cache_resource
def load_model():
    # Ganti "yolov8n.pt" dengan "best.pt" jika model custom Anda sudah siap
    return YOLO("yolov8n.pt")

model = load_model()

# Membuat tab untuk pilihan input gambar
tab1, tab2 = st.tabs(["Unggah Gambar", "Ambil Foto (Kamera)"])

image_data = None

# Tab 1: Upload File
with tab1:
    uploaded_file = st.file_uploader("Unggah gambar untuk dianalisis...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_data = uploaded_file

# Tab 2: Kamera
with tab2:
    camera_file = st.camera_input("Ambil gambar menggunakan kamera...")
    if camera_file is not None:
        image_data = camera_file

# Proses jika ada gambar yang dimasukkan (dari upload maupun kamera)
if image_data is not None:
    # Membaca dan menampilkan gambar
    image = Image.open(image_data)
    st.image(image, caption="Gambar Input", width=500)
    
    if st.button("Jalankan Prediksi"):
        with st.spinner("Mendeteksi objek..."):
            # Konversi gambar untuk inferensi YOLO
            img_array = np.array(image)
            results = model(img_array)
            
            # Mendapatkan gambar hasil plot bounding box
            res_plotted = results[0].plot()
            result_image = Image.fromarray(res_plotted)
            
            # Menampilkan gambar hasil prediksi
            st.image(result_image, caption="Hasil Deteksi", width=500)
            
            # Menampilkan rangkuman hasil
            boxes = results[0].boxes
            st.success(f"Ditemukan {len(boxes)} area yang terdeteksi.")
