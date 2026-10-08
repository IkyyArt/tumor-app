"""Klasifikasi tumor otak dari citra MRI (prototipe pembelajaran).

Fitur: daftar/login (SQLite), unggah gambar MRI, prediksi + probabilitas,
heatmap Grad-CAM, perbandingan model, dan riwayat prediksi.

File model (.keras / .h5) cukup ditaruh di root repo atau di folder models/.
Pengaturan kelas dan ukuran input ada di config.json.
"""
import glob
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

DB_PATH = "app_data.db"
CONFIG_PATH = "config.json"
EVAL_PATH = "hasil_evaluasi.csv"
DISCLAIMER = "Prototipe pembelajaran, bukan alat diagnosis medis."
MAX_FILES = 50
DEFAULT_CONFIG = {
    "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
    "img_size": 224,
    "rescale_255": True,
}


# ---------------------------------------------------------------- konfigurasi
def load_config():
    cfg = dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except (OSError, ValueError):
        pass
    return cfg


# ------------------------------------------------------------------- database
def db_run(query, params=(), fetch=False):
    with closing(sqlite3.connect(DB_PATH)) as con:
        con.row_factory = sqlite3.Row
        cur = con.execute(query, params)
        rows = cur.fetchall() if fetch else None
        con.commit()
        return rows


def init_db():
    db_run(
        """CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL)"""
    )
    db_run(
        """CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            model_name TEXT NOT NULL,
            filename TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL)"""
    )


# ----------------------------------------------------------------------- auth
def hash_password(password, salt_hex=None):
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return salt.hex(), digest.hex()


def register_user(username, password):
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9_]{3,30}", username):
        return False, "Username 3-30 karakter: huruf, angka, atau garis bawah."
    if len(password) < 6:
        return False, "Password minimal 6 karakter."
    if db_run("SELECT 1 FROM users WHERE username = ?", (username,), fetch=True):
        return False, "Username itu sudah dipakai. Coba yang lain."
    salt, digest = hash_password(password)
    db_run(
        "INSERT INTO users VALUES (?, ?, ?, ?)",
        (username, salt, digest, datetime.now().isoformat(timespec="seconds")),
    )
    return True, "Akun dibuat. Silakan masuk."


def verify_user(username, password):
    rows = db_run(
        "SELECT salt, password_hash FROM users WHERE username = ?",
        (username.strip().lower(),),
        fetch=True,
    )
    if not rows:
        return False
    _, digest = hash_password(password, rows[0]["salt"])
    return hmac.compare_digest(digest, rows[0]["password_hash"])


# ---------------------------------------------------------------------- model
def find_models():
    paths = []
    for pattern in ("models/*.keras", "models/*.h5", "*.keras", "*.h5"):
        paths += glob.glob(pattern)
    return sorted(set(paths))


@st.cache_resource(show_spinner="Memuat model...", max_entries=1)
def load_model(path):
    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=False)


def input_spec(model, cfg):
    """Ukuran (tinggi, lebar, kanal) dibaca dari model; config.json sebagai cadangan."""
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        h, w, c = shape[1], shape[2], shape[3]
        if h and w:
            return int(h), int(w), int(c or 3)
    except Exception:
        pass
    s = int(cfg["img_size"])
    return s, s, 3


def preprocess(img, spec, rescale):
    h, w, c = spec
    im = img.convert("L" if c == 1 else "RGB").resize((w, h))
    arr = np.asarray(im, dtype="float32")
    if c == 1:
        arr = arr[..., None]
    if rescale:
        arr = arr / 255.0
    return im.convert("RGB"), arr[None, ...]


def softmax(v):
    e = np.exp(v - np.max(v))
    return e / e.sum()


def predict_probs(model, x):
    probs = np.asarray(model.predict(x, verbose=0)[0], dtype="float64").ravel()
    if probs.size == 1:
        probs = np.array([1 - probs[0], probs[0]])
    elif probs.min() < 0 or abs(probs.sum() - 1) > 1e-3:
        probs = softmax(probs)
    return probs


# ------------------------------------------------------------------- Grad-CAM
def _find_feature_layer(model):
    """Cari layer terakhir yang menghasilkan feature map 4D (atau sub-model yang memuatnya)."""
    import tensorflow as tf

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            inner = _find_feature_layer(layer)
            if inner is not None:
                return layer, inner
        else:
            try:
                if len(layer.output.shape) == 4:
                    return layer, None
            except Exception:
                continue
    return None


def gradcam_heatmap(model, x, class_idx):
    import tensorflow as tf

    found = _find_feature_layer(model)
    if found is None:
        return None
    layer, inner = found
    x = tf.convert_to_tensor(x)

    def class_score(preds):
        if preds.shape[-1] == 1:
            return preds[:, 0] if class_idx == 1 else 1 - preds[:, 0]
        return preds[:, class_idx]

    def forward_by_layers(feat_model=None):
        """Jalankan layer satu per satu (untuk model Sequential atau base bersarang)."""
        h, conv_out = x, None
        for lyr in model.layers:
            if isinstance(lyr, tf.keras.layers.InputLayer):
                continue
            if feat_model is not None and lyr is layer:
                conv_out, h = feat_model(h)
            else:
                h = lyr(h)
                if feat_model is None and lyr is layer:
                    conv_out = h
        return conv_out, h

    with tf.GradientTape() as tape:
        if inner is not None:
            feat_model = tf.keras.Model(layer.inputs, [inner[0].output, layer.output])
            conv_out, preds = forward_by_layers(feat_model)
        else:
            try:
                grad_model = tf.keras.Model(model.inputs, [layer.output, model.output])
                conv_out, preds = grad_model(x)
            except Exception:
                conv_out, preds = forward_by_layers()
        score = class_score(preds)
    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * pooled, axis=-1)).numpy()
    return cam / (cam.max() + 1e-8)


def overlay_heatmap(base_img, cam, alpha=0.4):
    from matplotlib import colormaps

    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(base_img.size, Image.BILINEAR)
    colored = colormaps["jet"](np.asarray(cam_img) / 255.0)[..., :3]
    base = np.asarray(base_img.convert("RGB"), dtype="float32") / 255.0
    out = (1 - alpha) * base + alpha * colored
    return Image.fromarray(np.uint8(np.clip(out, 0, 1) * 255))


def analyze_image(model, img, cfg):
    spec = input_spec(model, cfg)
    shown, x = preprocess(img, spec, cfg["rescale_255"])
    probs = predict_probs(model, x)
    cam_img = None
    try:
        cam = gradcam_heatmap(model, x, int(np.argmax(probs)))
        if cam is not None:
            cam_img = overlay_heatmap(shown, cam)
    except Exception:
        cam_img = None
    return {"preprocessed": shown, "probs": probs, "gradcam": cam_img}


# ------------------------------------------------------------------ halaman
def page_auth():
    st.title("Klasifikasi tumor otak")
    st.caption(DISCLAIMER)
    tab_login, tab_daftar = st.tabs(["Masuk", "Daftar"])
    with tab_login:
        with st.form("form_masuk"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Masuk"):
                if verify_user(u, p):
                    st.session_state["user"] = u.strip().lower()
                    st.rerun()
                else:
                    st.error("Username atau password salah.")
    with tab_daftar:
        with st.form("form_daftar"):
            u = st.text_input("Username baru")
            p = st.text_input("Password baru", type="password")
            p2 = st.text_input("Ulangi password", type="password")
            if st.form_submit_button("Buat akun"):
                if p != p2:
                    st.error("Password dan pengulangannya tidak sama.")
                else:
                    ok, msg = register_user(u, p)
                    (st.success if ok else st.error)(msg)


def inject_css():
    st.markdown(
        """
        <style>
        .hero {background:#fff;border-radius:22px;padding:28px 34px;margin-bottom:18px;
               box-shadow:0 6px 24px rgba(15,60,80,.08);}
        .hero .kicker {font-size:12px;letter-spacing:.18em;font-weight:700;color:#0f5c6e;}
        .hero h1 {margin:4px 0 6px 0;font-size:2.6rem;}
        .hero p {margin:0;color:#4a5b66;}
        .note {background:#fff;border-left:5px solid #0f5c6e;border-radius:12px;
               padding:14px 18px;margin-bottom:18px;box-shadow:0 4px 16px rgba(15,60,80,.06);}
        .section-kicker {font-size:12px;letter-spacing:.18em;font-weight:700;color:#0f5c6e;margin-top:10px;}
        .modelbox {background:#e3f5ea;color:#1b6b3a;border-radius:12px;padding:14px 16px;font-weight:600;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero():
    st.markdown(
        """
        <div class="hero">
          <div class="kicker">MEDICAL IMAGE CLASSIFICATION &bull; RESEARCH PROTOTYPE</div>
          <h1>Brain Tumor MRI Classifier</h1>
          <p><b>Perbandingan CNN (MobileNetV2, ResNet50V2, InceptionV3) untuk klasifikasi tumor otak</b></p>
          <p>Multi-image research prototype: glioma, meningioma, no tumor, pituitary</p>
        </div>
        <div class="note"><b>Prototipe penelitian.</b> Output adalah respons klasifikasi model untuk
        demonstrasi akademik, bukan diagnosis klinis. Gunakan citra MRI otak yang mirip dengan data
        training (Brain Tumor MRI Dataset).</div>
        """,
        unsafe_allow_html=True,
    )


def page_predict(cfg):
    hero()
    models = find_models()
    model = None
    choice = None
    with st.sidebar:
        st.markdown("**Model aktif**")
        if models:
            choice = st.selectbox("Model", models, format_func=os.path.basename, label_visibility="collapsed")
            st.markdown(f'<div class="modelbox">{os.path.basename(choice)}</div>', unsafe_allow_html=True)
            try:
                model = load_model(choice)
                h, w, c = input_spec(model, cfg)
                st.caption(f"Input {h}x{w}x{c} | Output {', '.join(cfg['class_names'])}")
            except Exception as e:
                st.error(f"Model gagal dimuat: {e}")
        else:
            st.warning("Belum ada file model (.keras / .h5) di repo. Hanya pratinjau gambar yang aktif.")
        st.markdown("**Input prototype**")
        st.caption("JPG / JPEG / PNG, multiple image files")
        with st.expander("Advanced probability inspection"):
            low_conf = st.slider("Batas keyakinan rendah", 0.30, 0.95, 0.60, 0.05)

    classes = cfg["class_names"]
    if model is not None:
        n_out = model.output_shape[-1]
        expected = 2 if n_out == 1 else n_out
        if expected != len(classes):
            st.error(
                f"Model punya {expected} kelas, tetapi config.json berisi {len(classes)} nama kelas. "
                "Samakan jumlah dan urutannya dengan data latihmu."
            )
            return

    st.markdown('<div class="section-kicker">MULTI-IMAGE INPUT</div>', unsafe_allow_html=True)
    st.subheader("Unggah gambar MRI")
    st.caption(f"Maksimal {MAX_FILES} file sekali analisis.")
    files = st.file_uploader(
        "Pilih JPG / JPEG / PNG",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    if len(files) > MAX_FILES:
        st.warning(f"Maksimal {MAX_FILES} gambar. Yang diproses {MAX_FILES} pertama.")
        files = files[:MAX_FILES]
    if not files:
        return

    if st.button("Jalankan analisis", type="primary"):
        results = {}
        bar = st.progress(0.0, text="Memproses...")
        for n, f in enumerate(files, 1):
            try:
                img = Image.open(f)
                img.load()
            except Exception:
                results[f.name] = {"error": True}
                continue
            if model is None:
                shown, _ = preprocess(img, (int(cfg["img_size"]),) * 2 + (3,), cfg["rescale_255"])
                results[f.name] = {"original": img.convert("RGB"), "preprocessed": shown,
                                   "probs": None, "gradcam": None}
            else:
                res = analyze_image(model, img, cfg)
                res["original"] = img.convert("RGB")
                results[f.name] = res
                top = int(np.argmax(res["probs"]))
                db_run(
                    "INSERT INTO history (username, model_name, filename, predicted_class, confidence, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (st.session_state["user"], os.path.basename(choice), f.name, classes[top],
                     float(res["probs"][top]), datetime.now().isoformat(timespec="seconds")),
                )
            bar.progress(n / len(files), text=f"Memproses {n}/{len(files)}")
        bar.empty()
        st.session_state["results"] = results

    results = st.session_state.get("results")
    if not results:
        return
    results = {k: v for k, v in results.items() if not v.get("error")}
    if not results:
        st.error("Tidak ada file yang bisa dibaca sebagai gambar.")
        return

    st.markdown('<div class="section-kicker">HASIL</div>', unsafe_allow_html=True)
    st.subheader("Ringkasan analisis")
    if next(iter(results.values()))["probs"] is not None:
        rows = []
        for name, r in results.items():
            top = int(np.argmax(r["probs"]))
            rows.append({"File": name, "Prediksi": classes[top],
                         "Keyakinan": f"{r['probs'][top] * 100:.1f}%",
                         "Catatan": "keyakinan rendah" if r["probs"][top] < low_conf else ""})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    pilih = st.selectbox("Pilih image untuk inspeksi detail", list(results.keys()))
    r = results[pilih]
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Original**")
    c1.image(r["original"], width="stretch")
    c1.caption("Uploaded RGB image")
    c2.markdown("**Pra-proses**")
    c2.image(r["preprocessed"], width="stretch")
    c2.caption("Resize ke ukuran input model")
    c3.markdown("**Overlay Grad-CAM**")
    if r["gradcam"] is not None:
        c3.image(r["gradcam"], width="stretch")
        c3.caption("Area yang paling memengaruhi prediksi")
    else:
        c3.caption("Grad-CAM tidak tersedia.")
    if r["probs"] is not None:
        probs = r["probs"]
        top = int(np.argmax(probs))
        st.markdown(f"Hasil prediksi: **{classes[top]}** ({probs[top] * 100:.1f}%)")
        if probs[top] < low_conf:
            st.caption("Keyakinan model rendah, hasil ini kurang bisa dipercaya.")
        for i in np.argsort(probs)[::-1]:
            st.progress(float(probs[i]), text=f"{classes[i]}  {probs[i] * 100:.1f}%")


def page_compare():
    st.title("Perbandingan model")
    st.caption(DISCLAIMER)
    metrics = ["akurasi", "precision", "recall", "f1"]
    try:
        df = pd.read_csv(EVAL_PATH)
    except Exception:
        st.info("File hasil_evaluasi.csv belum ada di repo.")
        return
    for m in metrics:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    metrics = [m for m in metrics if m in df.columns]
    df = df.dropna(subset=metrics, how="all")
    if df.empty or not metrics:
        st.info(
            "Belum ada hasil evaluasi. Isi hasil_evaluasi.csv dengan angka tiap model "
            "(akurasi, precision, recall, f1), lalu perbarui file di GitHub."
        )
        return
    if df[metrics].max().max() > 1:
        df[metrics] = df[metrics] / 100
    st.dataframe(
        df.style.format({m: "{:.2%}" for m in metrics}, na_rep="-"),
        width="stretch",
        hide_index=True,
    )
    if "akurasi" in df.columns and df["akurasi"].notna().any():
        best = df.loc[df["akurasi"].idxmax()]
        st.success(f"Akurasi tertinggi: {best['model']} ({best['akurasi']:.2%})")
    st.bar_chart(df.set_index("model")[metrics])


def page_history():
    st.title("Riwayat")
    rows = db_run(
        "SELECT created_at, model_name, filename, predicted_class, confidence "
        "FROM history WHERE username = ? ORDER BY id DESC LIMIT 200",
        (st.session_state["user"],),
        fetch=True,
    )
    if not rows:
        st.info("Belum ada prediksi. Coba unggah gambar di halaman Prediksi.")
        return
    df = pd.DataFrame([dict(r) for r in rows])
    df.columns = ["Waktu", "Model", "File", "Hasil", "Keyakinan"]
    df["Keyakinan"] = (df["Keyakinan"] * 100).round(1).astype(str) + "%"
    st.dataframe(df, width="stretch", hide_index=True)
    if st.button("Hapus riwayat saya"):
        db_run("DELETE FROM history WHERE username = ?", (st.session_state["user"],))
        st.rerun()


def main():
    st.set_page_config(page_title="Klasifikasi tumor otak", layout="wide")
    init_db()
    inject_css()
    if "user" not in st.session_state:
        page_auth()
        return
    cfg = load_config()
    with st.sidebar:
        st.subheader("Brain Tumor MRI")
        st.caption("MobileNetV2 / ResNet50V2 / InceptionV3")
        page = st.radio("Menu", ["Prediksi", "Perbandingan model", "Riwayat"])
        st.divider()
        st.caption(f"Masuk sebagai {st.session_state['user']}")
        if st.button("Keluar"):
            del st.session_state["user"]
            st.rerun()
        st.caption("Data akun dan riwayat disimpan sementara di server dan bisa hilang saat aplikasi restart.")
    if page == "Prediksi":
        page_predict(cfg)
    elif page == "Perbandingan model":
        page_compare()
    else:
        page_history()


if __name__ == "__main__":
    main()
