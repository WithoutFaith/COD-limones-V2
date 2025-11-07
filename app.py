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
APHID_IMAGE_URL = st.secrets.get("APHID_IMAGE_URL", "") or os.getenv("APHID_IMAGE_URL", "")
if not APHID_IMAGE_URL:
    APHID_IMAGE_URL = "https://inaturalist-open-data.s3.amazonaws.com/photos/4728889/medium.jpg"
# --- IMAGEN DE REFERENCIA 2 (Pulgón Pardo/Negro del Naranjo) ---
APHID2_IMAGE_URL = st.secrets.get("APHID2_IMAGE_URL", "") or os.getenv("APHID2_IMAGE_URL", "")
if not APHID2_IMAGE_URL:
    APHID2_IMAGE_URL = "http://gipcitricos.ivia.es/wp-content/uploads/2010/07/Foto-15.jpg"
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

    if os.path.exists(weights_path):
        sz = _size_mb(weights_path)
        if (not _looks_like_html(weights_path)) and (sz > 50):
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
    except Exception as e:
        st.error(f"❌ No se pudo descargar los pesos desde WEIGHTS_URL.\nDetalle: {e}")
        st.stop()

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
st.caption("💡 **Confianza**: 0.5–0.7 equilibrio · 0.8–0.9 muy estricto  \n"
           "💡 **IoU**: cuánto se permiten solapar las cajas tras NMS.")

# ==== UTILIDADES (deben ir ANTES de la inferencia) ====

# Canonicaliza el nombre detectado a { "negras", "blanca", "verdes" }
def canonical(label: str) -> str:
    s = (
        label.strip().lower()
        .replace("_", "").replace("-", "")
        .replace("hoja", "").replace("hojas", "")
    )
    if "negra" in s:   # negras, negra, hoja_negra...
        return "negras"
    if "blanc" in s:   # blanca, blancas
        return "blanca"
    if "verde" in s:   # verde, verdes
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

def to_json(results, class_names):
    """Convierte los resultados de YOLO en un JSON simple {image:{w,h}, predictions:[...]}."""
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
                "class": class_names.get(ci, str(ci)),
                "confidence": float(cf)
            })
    return {"image": {"width": w, "height": h}, "predictions": preds}

def draw_boxes(pil_img, preds):
    """Dibuja cajas con color según clase canónica."""
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

def render_black_aphid_card():
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
      .card {font-family:'Poppins',system-ui; background:#0f172a; border:1px solid #1f2937;
             border-radius:16px; padding:18px 20px; color:#e5e7eb;}
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

def render_brownblack_aphid_card():
    st.markdown("""
    <style>
      .card {font-family:'Poppins',system-ui; background:#0f172a; border:1px solid #1f2937;
             border-radius:16px; padding:18px 20px; color:#e5e7eb;}
      .tag {display:inline-block; padding:4px 10px; border-radius:999px; font-size:.82rem;
            background:#111827; color:#f59e0b; border:1px solid #374151; margin-bottom:8px;}
      table {width:100%; border-collapse:separate; border-spacing:0 8px; font-size:.95rem}
      td:nth-child(1){color:#9ca3af; width:34%;}
      td{vertical-align:top; padding:6px 8px; background:#0b1220; border-radius:10px; border:1px solid #1f2937;}
      .emph{font-weight:600; color:#f9fafb}
      .strong{font-weight:700;}
    </style>
    <div class="card">
      <span class="tag">🍊 Pulgón Pardo/Negro del Naranjo <span class="emph">(Toxoptera aurantii)</span> — Adulto áptero</span>
      <table>
        <tr><td><span class="strong">Identificación</span></td><td>Tamaño similar (1.5–2 mm). Cuerpo <span class="emph">ovoidal</span>, de color <span class="emph">pardo oscuro a negro</span> y brillante. Se distingue de <span class="emph">T. citricida</span> porque las hembras ápteras tienen <span class="emph">antenas con bandas blancas y negras</span>.</td></tr>
        <tr><td><span class="strong">Ubicación</span></td><td>Colonias muy densas en el <span class="emph">envés de hojas tiernas</span>, <span class="emph">yemas florales</span> y <span class="emph">frutos recién cuajados</span>.</td></tr>
        <tr><td><span class="strong">Temporadas (Perú)</span></td><td>Activo todo el año en climas tropicales; picos en <span class="emph">Primavera</span> (Sep–Dic) y <span class="emph">Verano</span> (Dic–Mar), coincidiendo con la <span class="emph">brotación y floración</span>.</td></tr>
        <tr><td><span class="strong">Daño Directo</span></td><td><span class="emph">Succiona savia</span>, causando que las hojas se <span class="emph">deformen ligeramente</span> y se <span class="emph">endurezcan</span>. En flores y frutos jóvenes puede provocar <span class="emph">caída</span> o desarrollo dificultoso.</td></tr>
        <tr><td><span class="strong">Daño Indirecto</span></td><td>Produce <span class="emph">melaza</span> que es rápidamente cubierta por <span class="emph">fumagina</span> (negrilla), limitando la fotosíntesis.</td></tr>
        <tr><td><span class="strong">Importancia</span></td><td>Especie <span class="emph">polífaga</span> (cacao, café, mango, etc.). Vector de virus (como el <span class="emph">VTC</span>), aunque suele considerarse <span class="emph">menos eficiente</span> que <span class="emph">T. citricida</span> como vector del VTC.</td></tr>
      </table>
    </div>
    """, unsafe_allow_html=True)
   

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

   # Si se detectan hojas negras, mostrar dos fichas técnicas + dos fotos
if c_negras > 0:
    left, right = st.columns([2, 1])

    with left:
        # 1) T. citricida
        render_black_aphid_card()
        st.divider()
        # 2) T. aurantii
        render_brownblack_aphid_card()

    with right:
        # Imagen 1
        if APHID_IMAGE_URL:
            st.image(APHID_IMAGE_URL, caption="Pulgón negro (T. citricida) — referencia", use_container_width=True)
        # Imagen 2
        if APHID2_IMAGE_URL:
            st.image(APHID2_IMAGE_URL, caption="Pulgón pardo/negro (T. aurantii) — referencia", use_container_width=True)
        if not (APHID_IMAGE_URL or APHID2_IMAGE_URL):
            st.caption("ℹ️ Agrega APHID_IMAGE_URL y/o APHID2_IMAGE_URL en Secrets/.env para mostrar fotos de referencia.")



