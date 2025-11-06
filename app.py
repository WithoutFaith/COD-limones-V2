import os, io, tempfile, re, requests
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# --- CONFIGURACIÓN DE LA APP ---
st.set_page_config(page_title="Detector de plagas en limones COD", page_icon="🪲", layout="centered")
st.title("🪲 Detector de plagas en limones COD")
st.caption("Sube una imagen para analizarla")

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
            return  # no mostrar mensajes
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
    except Exception as e:
        st.error(f"❌ No se pudo descargar los pesos desde WEIGHTS_URL.\nDetalle: {e}")
        st.stop()

# --- LLAMAR LA DESCARGA DESPUÉS DE DEFINIR LA FUNCIÓN ---
maybe_download_weights(WEIGHTS_PATH)

# --- CHEQUEO SILENCIOSO DE OPENCV ---
try:
    import cv2, numpy as np
except Exception as e:
    st.error(f"❌ Error importando OpenCV/NumPy: {e}")
    st.stop()

# --- IMPORTAR YOLO ---
from ultralytics import YOLO

@st.cache_resource
def load_model(path: str):
    return YOLO(path)

model = load_model(WEIGHTS_PATH)

# ==== INTERFAZ ====
uploaded = st.file_uploader("Sube una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])

conf = st.slider("Confianza mínima", 0.0, 1.0, 0.5, 0.05)
iou  = st.slider("IoU (overlap) máx.", 0.0, 1.0, 0.5, 0.05)

st.caption(
    "💡 **Confianza**: 0.5–0.7 equilibrio · 0.8–0.9 muy estricto  \n"
    "💡 **IoU**: cuánto se permiten solapar las cajas tras NMS."
)

# ==== UTILIDADES ====

# Normalizador de nombres (para que "Hoja_Verde", "verde-hoja", etc. cuenten como "verde")
def normalize(label: str) -> str:
    return (
        label.strip()
             .lower()
             .replace("_", "")
             .replace("-", "")
             .replace("hoja", "")
             .replace(" hojas", "")
             .strip()
    )

# Diagnósticos y colores por clase (usamos claves NORMALIZADAS)
DIAGNOSIS = {
    "negra": {
        "color": "red",
        "msg": "⚠️ Se detectaron **hojas negras** — posible fumagina/daño por plagas. Revisa focos, retira hojas muy afectadas y evalúa tratamiento fungicida."
    },
    "blanca": {
        "color": "orange",
        "msg": "⚠️ Se detectaron **hojas blancas (melaza)** — probable presencia de insectos (pulgones/mosca blanca). Limpia, monitorea y considera control biológico o químico."
    },
    "verde": {
        "color": "lime",
        "msg": "✅ Solo se detectaron **hojas verdes** — el cultivo parece sano en esta imagen."
    }
}

def pick_color(label: str) -> str:
    return DIAGNOSIS.get(normalize(label), {}).get("color", "gray")

def to_json(results, class_names):
    if not results:
        return {"image": {}, "predictions": []}
    r = results[0]
    h, w = r.orig_shape
    preds = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss  = r.boxes.cls.cpu().numpy().astype(int)
        for (x0, y0, x1, y1), cf, ci in zip(xyxy, confs, clss):
            ww, hh = float(x1 - x0), float(y1 - y0)
            xc, yc = float(x0 + ww/2), float(y0 + hh/2)
            preds.append({
                "x": xc, "y": yc, "width": ww, "height": hh,
                "class": class_names.get(ci, str(ci)),  # nombre original
                "confidence": float(cf)
            })
    return {"image": {"width": w, "height": h}, "predictions": preds}

def draw_boxes(pil_img, preds):
    draw = ImageDraw.Draw(pil_img)
    try: font = ImageFont.load_default()
    except: font = None
    W, H = pil_img.size
    iw = preds.get("image", {}).get("width", W)
    ih = preds.get("image", {}).get("height", H)
    sx, sy = W / float(iw), H / float(ih)

    for p in preds.get("predictions", []):
        x, y, ww, hh = p["x"], p["y"], p["width"], p["height"]
        x0, y0 = int((x - ww/2) * sx), int((y - hh/2) * sy)
        x1, y1 = int((x + ww/2) * sx), int((y + hh/2) * sy)
        color = pick_color(p["class"])
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f"{p['class']} {p['confidence']:.2f}"
        pad = 3
        if font:
            tw = draw.textlength(label, font=font)
            draw.rectangle([x0, y0 - 14, x0 + tw + 2*pad, y0], fill="black")
            draw.text((x0 + pad, y0 - 12), label, fill="white", font=font)
        else:
            draw.text((x0, max(0, y0 - 12)), label, fill="white")
    return pil_img

# ==== INFERENCIA ====
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    max_w = 1600
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)))

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name, "JPEG", quality=95)
        temp_path = tmp.name

    with st.spinner("🔍 Detectando..."):
        results = model.predict(source=temp_path, conf=conf, iou=iou, verbose=False)

    preds = to_json(results, model.names)

    # ---- Conteo por clase (normalizado) y diagnóstico resumido ----
    from collections import Counter
    counts = Counter(normalize(p["class"]) for p in preds.get("predictions", []))

    if counts:
        c_negra  = int(counts.get("negra", 0))
        c_blanca = int(counts.get("blanca", 0))
        c_verde  = int(counts.get("verde", 0))

        col1, col2, col3 = st.columns(3)
        col1.metric("Negra",  c_negra)
        col2.metric("Blanca", c_blanca)
        col3.metric("Verde",  c_verde)

        # Prioridad del diagnóstico: negra > blanca > verde
        if c_negra > 0:
            st.error(DIAGNOSIS["negra"]["msg"])
        elif c_blanca > 0:
            st.warning(DIAGNOSIS["blanca"]["msg"])
        elif c_verde > 0:
            st.success(DIAGNOSIS["verde"]["msg"])
    else:
        st.info("No se detectaron hojas en la imagen.")

    st.subheader("📊 Predicciones (JSON)")
    st.json(preds)

    vis = draw_boxes(img.copy(), preds)
    st.image(vis, caption="Detecciones", use_container_width=True)

    buf = io.BytesIO()
    vis.save(buf, "PNG")
    buf.seek(0)
    st.download_button("⬇️ Descargar imagen con detecciones", buf, "detecciones.png", "image/png")
else:
    st.info("Sube una imagen para ejecutar la detección.")
