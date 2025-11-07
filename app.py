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

# --- IMAGEN DE REFERENCIA (Pulgón Negro) ---
# Puedes usar una URL directa o definirla en tu archivo de Secrets
APHID_IMAGE_URL = st.secrets.get("APHID_IMAGE_URL", "") or os.getenv("APHID_IMAGE_URL", "")

# Si quieres fijarla manualmente:
if not APHID_IMAGE_URL:
    APHID_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/4/4b/Toxoptera_citricida02.jpg"

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


# Canonicaliza el nombre detectado a { "negras", "blanca", "verdes" }
def canonical(label: str) -> str:
    s = (
        label.strip().lower()
        .replace("_", "").replace("-", "")
        .replace("hoja", "").replace("hojas", "")
    )
    # Mapeo flexible por subcadenas
    if "negra" in s:
        return "negras"
    if "blanc" in s:
        return "blanca"
    if "verde" in s:
        return "verdes"
    return s  # por si llegara otra clase

# Diagnósticos y colores por clase (claves CANÓNICAS)
DIAGNOSIS = {
    "negras": {
        "color": "red",
        "msg": "⚠️ Se detectaron hojas negras — posible **fumagina** o daño por plagas. Revisa focos, retira hojas muy afectadas y evalúa tratamiento fungicida."
    },
    "blanca": {
        "color": "orange",
        "msg": "⚠️ Se detectaron hojas blancas (melaza) — probable presencia de insectos (pulgones/mosca blanca). Limpia, monitorea y considera control biológico o químico."
    },
    "verdes": {
        "color": "lime",
        "msg": "✅ Solo se detectaron hojas verdes — el cultivo parece **sano** en esta imagen."
    }
}

def pick_color(label: str) -> str:
    return DIAGNOSIS.get(canonical(label), {}).get("color", "gray")

def render_black_aphid_card():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
      .card {font-family:'Poppins',system-ui; background:#0f172a; border:1px solid #1f2937;
             border-radius:16px; padding:18px 20px; color:#e5e7eb;}
      .card h3 {margin:0 0 10px 0; font-weight:700; font-size:1.1rem}
      .tag {display:inline-block; padding:4px 10px; border-radius:999px; font-size:.82rem;
            background:#111827; color:#facc15; border:1px solid #374151; margin-bottom:8px;}
      table {width:100%; border-collapse:separate; border-spacing:0 8px; font-size:.95rem}
      td:nth-child(1){color:#9ca3af; width:34%;}
      td{vertical-align:top; padding:6px 8px; background:#0b1220; border-radius:10px; border:1px solid #1f2937;}
      .emph{font-weight:600; color:#f9fafb}
      .strong{font-weight:700;}
    </style>
    <div class="card">
      <span class="tag">🍋 Pulgón Negro de los Cítricos <span class="emph">(Toxoptera citricida)</span> — Adulto áptero</span>
      <table>
        <tr><td><span class="strong">Identificación</span></td><td>Pequeño (~1.5–2.5 mm). Cuerpo <span class="emph">globoso</span>, de color <span class="emph">negro brillante</span> o castaño muy oscuro. La forma áptera es <span class="emph">sin alas</span>.</td></tr>
        <tr><td><span class="strong">Ubicación</span></td><td>Colonias densas en el <span class="emph">envés</span> de hojas jóvenes y <span class="emph">brotes tiernos</span>.</td></tr>
        <tr><td><span class="strong">Temporadas (Perú)</span></td><td>Actividad alta en <span class="emph">Primavera</span> (Sep–Dic) y <span class="emph">Verano</span> (Dic–Mar).</td></tr>
        <tr><td><span class="strong">Daño Directo</span></td><td><span class="emph">Succiona la savia</span>, debilitando la planta. Hojas pueden <span class="emph">enrollarse/deformarse</span>.</td></tr>
        <tr><td><span class="strong">Daño Indirecto</span></td><td>Produce <span class="emph">melaza</span> que favorece el hongo <span class="emph">fumagina</span> (capa negra de hollín).</td></tr>
        <tr><td><span class="strong">Mayor Riesgo</span></td><td>Vector eficiente del <span class="emph">Virus de la Tristeza de los Cítricos (VTC)</span>.</td></tr>
      </table>
    </div>
    """, unsafe_allow_html=True)



## ==== INFERENCIA ====
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

    from collections import Counter
    counts = Counter(canonical(p["class"]) for p in preds.get("predictions", []))
    c_negras  = int(counts.get("negras", 0))
    c_blanca  = int(counts.get("blanca", 0))
    c_verdes  = int(counts.get("verdes", 0))

    col1, col2, col3 = st.columns(3)
    col1.metric("Negras",  c_negras)
    col2.metric("Blanca",  c_blanca)
    col3.metric("Verdes",  c_verdes)

    vis = draw_boxes(img.copy(), preds)
    st.image(vis, caption="Detecciones", use_container_width=True)

    # Diagnóstico principal
    if c_negras > 0:
        st.error(DIAGNOSIS["negras"]["msg"])
    elif c_blanca > 0:
        st.warning(DIAGNOSIS["blanca"]["msg"])
    elif c_verdes > 0:
        st.success(DIAGNOSIS["verdes"]["msg"])
    else:
        st.info("No se detectaron hojas en la imagen.")

    # Si se detectan hojas negras, mostrar ficha técnica + foto
    if c_negras > 0:
        left, right = st.columns([2, 1])
        with left:
            render_black_aphid_card()
        with right:
            if APHID_IMAGE_URL:
                st.image(APHID_IMAGE_URL, caption="Pulgón negro (referencia)", use_container_width=True)
            else:
                st.caption("ℹ️ Agrega APHID_IMAGE_URL en Secrets/.env para mostrar una foto de referencia.")

    # JSON opcional colapsable
    with st.expander("📊 Ver JSON de predicciones"):
        st.json(preds)

    buf = io.BytesIO()
    vis.save(buf, "PNG")
    buf.seek(0)
    st.download_button("⬇️ Descargar imagen con detecciones", buf, "detecciones.png", "image/png")
else:
    st.info("Sube una imagen para ejecutar la detección.")
