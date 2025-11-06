import os, io, tempfile, requests
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# --- YOLO (Ultralytics) ---
from ultralytics import YOLO

st.set_page_config(page_title="Detector de plagas en limones COD", page_icon="🪲", layout="centered")
st.title("🪲 Detector de plagas en limones COD")
st.caption("Sube una imagen para analizarla con tu modelo YOLO local (sin API).")

# --- Rutas de pesos ---
DEFAULT_WEIGHTS = "weights/best.pt"
WEIGHTS_PATH = os.getenv("YOLO_WEIGHTS", DEFAULT_WEIGHTS)

# --- Función para descargar pesos si no existen ---
def maybe_download_weights(weights_path: str):
    url = os.getenv("WEIGHTS_URL")
    if os.path.exists(weights_path) or not url:
        return
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    try:
        if "drive.google.com" in url:
            import gdown
            # Maneja enlaces tipo /file/d/<ID>/view o uc?export=download&id=<ID>
            gdown.download(url=url, output=weights_path, quiet=False, fuzzy=True)
        else:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(weights_path, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
    except Exception as e:
        st.error(f"❌ No se pudo descargar los pesos desde WEIGHTS_URL.\nDetalle: {e}")
        st.stop()

maybe_download_weights(WEIGHTS_PATH)

if not os.path.exists(WEIGHTS_PATH):
    st.error(f"No se encontró el archivo de pesos: {WEIGHTS_PATH}")
    st.stop()

@st.cache_resource
def load_model(path: str):
    return YOLO(path)

model = load_model(WEIGHTS_PATH)

# --- Interfaz Streamlit ---
uploaded = st.file_uploader("Sube una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])
conf = st.slider("Confianza mínima", 0.0, 1.0, 0.5, 0.05)
iou = st.slider("IoU (overlap) máx.", 0.0, 1.0, 0.5, 0.05)

st.caption(
    "💡 **Confianza**: 0.5–0.7 equilibrio · 0.8–0.9 muy estricto  \n"
    "💡 **IoU**: cuánto se permiten solapar las cajas tras NMS."
)

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
                "class": class_names.get(ci, str(ci)),
                "confidence": float(cf)
            })
    return {"image": {"width": w, "height": h}, "predictions": preds}

def draw_boxes(pil_img, preds):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.load_default()
    except:
        font = None
    W, H = pil_img.size
    iw = preds.get("image", {}).get("width", W)
    ih = preds.get("image", {}).get("height", H)
    sx, sy = W / float(iw), H / float(ih)

    for p in preds.get("predictions", []):
        x, y, ww, hh = p["x"], p["y"], p["width"], p["height"]
        x0, y0 = int((x - ww/2) * sx), int((y - hh/2) * sy)
        x1, y1 = int((x + ww/2) * sx), int((y + hh/2) * sy)
        draw.rectangle([x0, y0, x1, y1], outline="lime", width=3)
        label = f"{p['class']} {p['confidence']:.2f}"
        pad = 3
        if font:
            tw = draw.textlength(label, font=font)
            draw.rectangle([x0, y0 - 14, x0 + tw + 2*pad, y0], fill="black")
            draw.text((x0 + pad, y0 - 12), label, fill="white", font=font)
        else:
            draw.text((x0, max(0, y0 - 12)), label, fill="white")
    return pil_img

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    max_w = 1600
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)))

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name, "JPEG", quality=95)
        pt = tmp.name

    with st.spinner("🔍 Detectando..."):
        results = model.predict(source=pt, conf=conf, iou=iou, verbose=False)

    preds = to_json(results, model.names)
    st.subheader("📊 Predicciones (formato JSON)")
    st.json(preds)

    vis = draw_boxes(img.copy(), preds)
    st.image(vis, caption="Detecciones", use_container_width=True)

    buf = io.BytesIO()
    vis.save(buf, "PNG")
    buf.seek(0)
    st.download_button("⬇️ Descargar imagen con detecciones", buf, "detecciones.png", "image/png")
else:
    st.info("Sube una imagen para ejecutar la detección.")
