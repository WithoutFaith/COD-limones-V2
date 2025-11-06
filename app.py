import os, io, tempfile, re, requests
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# --- CONFIGURACIÓN DE LA APP ---
st.set_page_config(page_title="Detector de plagas en limones COD", page_icon="🪲", layout="centered")
st.title("🪲 Detector de plagas en limones COD")
st.caption("Sube una imagen para analizarla con tu modelo YOLO local (sin API).")

# --- RUTAS DE PESOS ---
DEFAULT_WEIGHTS = "weights/best.pt"
WEIGHTS_PATH = os.getenv("YOLO_WEIGHTS", DEFAULT_WEIGHTS)

# =======================
# DESCARGA ROBUSTA + VALIDACIÓN (.pt)
# =======================
def _gdrive_extract_id(url: str):
    m = re.search(r"/d/([a-zA-Z0-9_-]{10,})/", url)
    if m: return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]{10,})", url)
    return m.group(1) if m else None

def _get_confirm_token(resp):
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            return v
    return None

def _save_stream(resp, dest):
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

def _looks_like_html(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(2048).lower()
        return (b"<html" in head) or (b"<!doctype html" in head)
    except:
        return True

def _size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0

def _download_from_gdrive(url: str, dest: str):
    file_id = _gdrive_extract_id(url)
    if not file_id:
        raise RuntimeError("No se pudo extraer el ID de Google Drive.")
    session = requests.Session()
    base = "https://drive.google.com/uc?export=download"
    r = session.get(base, params={"id": file_id}, stream=True)
    token = _get_confirm_token(r)
    if token:
        r = session.get(base, params={"id": file_id, "confirm": token}, stream=True)
    r.raise_for_status()
    _save_stream(r, dest)

def maybe_download_weights(weights_path: str):
    url = os.getenv("WEIGHTS_URL")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    # ¿ya existe y parece válido?
    if os.path.exists(weights_path):
        sz = _size_mb(weights_path)
        if (not _looks_like_html(weights_path)) and (sz > 50):
            st.info(f"✅ Pesos detectados: {weights_path} ({sz:.1f} MB)")
            return
        else:
            try: os.remove(weights_path)
            except: pass

    if not url:
        return

    try:
        if "drive.google.com" in url:
            _download_from_gdrive(url, weights_path)
            if _looks_like_html(weights_path) or _size_mb(weights_path) < 50:
                try:
                    import gdown
                    os.remove(weights_path)
                    gdown.download(url=url, output=weights_path, quiet=False, fuzzy=True)
                except Exception as ge:
                    raise RuntimeError(f"Fallo Drive; prueba HuggingFace o Dropbox. gdown: {ge}")
        else:
            with requests.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                _save_stream(r, weights_path)

        sz = _size_mb(weights_path)
        if (not os.path.exists(weights_path)) or _looks_like_html(weights_path) or (sz < 50):
            raise RuntimeError("La descarga no parece un .pt válido (muy pequeño o HTML).")
        st.info(f"✅ Pesos descargados: {weights_path} ({sz:.1f} MB)")
    except Exception as e:
        st.error(f"❌ No se pudo descargar los pesos desde WEIGHTS_URL.\nDetalle: {e}")
        st.stop()

# --- LLAMAR LA DESCARGA DESPUÉS DE DEFINIR LA FUNCIÓN ---
maybe_download_weights(WEIGHTS_PATH)

# --- CHEQUEO OPCIONAL DE OPENCV ---
try:
    import cv2, numpy as np
    st.write(f"OpenCV OK: {cv2.__version__} · NumPy: {np.__version__}")
except Exception as e:
    st.error(f"❌ Error importando OpenCV/NumPy: {e}")
    st.stop()

# --- IMPORTAR YOLO ---
from ultralytics import YOLO

@st.cache_resource
def load_model(path: str):
    return YOLO(path)

model = load_model(WEIGHTS_PATH)
