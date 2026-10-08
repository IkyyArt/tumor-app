# Klasifikasi tumor otak dari citra MRI

Prototipe pembelajaran, bukan alat diagnosis medis.

Aplikasi web Streamlit: daftar dan masuk, unggah gambar MRI, prediksi beserta probabilitas,
heatmap Grad-CAM, perbandingan model, dan riwayat prediksi.

## Isi repo
- `streamlit_app.py`: kode aplikasi
- `requirements.txt`: daftar library
- `config.json`: nama kelas dan ukuran input
- `hasil_evaluasi.csv`: hasil evaluasi tiap model (isi sendiri)
- file model `.keras` atau `.h5`: taruh di root repo atau folder `models/`

## Yang perlu kamu sesuaikan
1. Taruh file model hasil training di repo.
2. Samakan `class_names` di `config.json` dengan urutan kelas saat training.
3. Isi `hasil_evaluasi.csv` dengan akurasi, precision, recall, dan f1 tiap model.
