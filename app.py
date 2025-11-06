# --- Descarga robusta + validación del .pt ---
import re, requests, os, streamlit as st

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
        if (not _looks_like_html(weights_path)) and (sz > 50):  # esperamos ~85 MB
            st.info(f"✅ Pesos detectados: {weights_path} ({sz:.1f} MB)")
            return
        else:
            # borrar archivo corrupto/HTML o demasiado pequeño
            try: os.remove(weights_path)
            except: pass

    if not url:
        return

    try:
        if "drive.google.com" in url:
            # 1er intento: requests manejando token
            _download_from_gdrive(url, weights_path)

            # si parece HTML o quedó muy chico, reintenta con gdown si está disponible
            if _looks_like_html(weights_path) or _size_mb(weights_path) < 50:
                try:
                    import gdown
                    os.remove(weights_path)
                    gdown.download(url=url, output=weights_path, quiet=False, fuzzy=True)
                except Exception as ge:
                    raise RuntimeError(f"Fallo Drive; prueba HuggingFace o Dropbox. Detalle gdown: {ge}")
        else:
            # URL directa (HuggingFace raw, S3, Dropbox ?dl=1)
            with requests.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                _save_stream(r, weights_path)

        # validar resultado final
        sz = _size_mb(weights_path)
        if (not os.path.exists(weights_path)) or _looks_like_html(weights_path) or (sz < 50):
            raise RuntimeError("La descarga no parece un .pt válido (muy pequeño o HTML).")
        st.info(f"✅ Pesos descargados: {weights_path} ({sz:.1f} MB)")
    except Exception as e:
        st.error(f"❌ No se pudo descargar los pesos desde WEIGHTS_URL.\nDetalle: {e}")
        st.stop()

maybe_download_weights(WEIGHTS_PATH)
