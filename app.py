import streamlit as st
import tempfile
import os
import zipfile
import requests
import io
import gc
import re
import time
import numpy as np
from PIL import Image
from deepface import DeepFace
from dotenv import load_dotenv

# ==========================
# Setup
# ==========================
load_dotenv()
# Streamlit Cloud stores secrets in st.secrets; local dev uses .env
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    API_KEY = os.getenv("GOOGLE_API_KEY", "")

st.set_page_config(
    page_title="FindMe – Face Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading ArcFace model…")
def load_model():
    return DeepFace.build_model("ArcFace")


load_model()

# ==========================
# Sidebar settings
# ==========================
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        "Match sensitivity (cosine distance)",
        min_value=0.20,
        max_value=0.60,
        value=0.40,
        step=0.01,
        help=(
            "Lower = stricter (fewer false positives, may miss some).\n"
            "Higher = looser (catches more, but may include wrong faces).\n"
            "Recommended: 0.35 – 0.45"
        ),
    )
    scan_subfolders = st.checkbox("Scan subfolders recursively", value=True)
    st.divider()
    st.markdown(
        "**How it works**\n"
        "1. Upload your selfie\n"
        "2. Paste a Drive folder link shared as *Anyone with link: Viewer*\n"
        "3. FindMe scans every image using ArcFace + RetinaFace + MTCNN\n"
        "4. Download all matching photos as a ZIP"
    )


# ==========================
# Google Drive helpers
# ==========================

def extract_folder_id(link):
    patterns = [
        r"drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]{25,})",
        r"/([a-zA-Z0-9_-]{25,})",
    ]
    for pattern in patterns:
        m = re.search(pattern, link)
        if m:
            return m.group(1)
    return None


def get_all_images(folder_id, recursive=True, _depth=0):
    """List all image files from a Drive folder with pagination + optional recursion."""
    if _depth > 6:
        return []

    results = []
    page_token = None

    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "nextPageToken,files(id,name,mimeType,thumbnailLink)",
            "key": API_KEY,
            "pageSize": 1000,
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            r = requests.get(
                "https://www.googleapis.com/drive/v3/files",
                params=params,
                timeout=30,
            )
        except requests.RequestException as exc:
            st.warning(f"Network error while listing folder: {exc}")
            break

        if r.status_code == 403:
            st.error(
                "❌ Access denied (403). Make sure the folder is shared as "
                "**Anyone with link: Viewer** and the API key is enabled."
            )
            break
        if r.status_code != 200:
            err_msg = r.json().get("error", {}).get("message", r.text)
            st.error(f"❌ Drive API error {r.status_code}: {err_msg}")
            break

        data = r.json()
        for f in data.get("files", []):
            mime = f.get("mimeType", "")
            if mime.startswith("image/"):
                results.append(f)
            elif recursive and mime == "application/vnd.google-apps.folder":
                results.extend(get_all_images(f["id"], recursive, _depth + 1))

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return results


# ==========================
# Image fetching helpers
# ==========================

def _is_valid_image(data):
    if not data or len(data) < 8:
        return False
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        return True
    except Exception:
        return False


def _fetch_url(url, timeout=30, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.content
        except requests.RequestException:
            pass
        if attempt < retries - 1:
            time.sleep(1.5 ** attempt)
    return None


def get_image_for_check(file):
    """
    Returns (bytes, is_full_resolution).
    Strategy: thumbnail (fast) → API download → direct public URL.
    """
    thumb_url = file.get("thumbnailLink")
    if thumb_url:
        if "=s" in thumb_url:
            thumb_url = thumb_url.split("=s")[0] + "=s512"
        data = _fetch_url(thumb_url, timeout=15)
        if data and _is_valid_image(data):
            return data, False

    fid = file["id"]
    data = _fetch_url(
        f"https://www.googleapis.com/drive/v3/files/{fid}?alt=media&key={API_KEY}"
    )
    if data and _is_valid_image(data):
        return data, True

    data = _fetch_url(
        f"https://drive.google.com/uc?export=download&id={fid}&confirm=1"
    )
    if data and _is_valid_image(data):
        return data, True

    return None, False


def get_full_image(file_id):
    data = _fetch_url(
        f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={API_KEY}"
    )
    if data and _is_valid_image(data):
        return data
    data = _fetch_url(
        f"https://drive.google.com/uc?export=download&id={file_id}&confirm=1"
    )
    return data if data and _is_valid_image(data) else None


# ==========================
# Face verification
# ==========================

_BACKENDS = ["opencv", "retinaface", "mtcnn"]


def verify_face(selfie_path, candidate_bytes, threshold):
    """
    All-backend ArcFace verification.
    Tries opencv → retinaface → mtcnn, exits early on a confirmed match.
    Returns (is_match, best_distance).
    """
    try:
        img_np = np.array(Image.open(io.BytesIO(candidate_bytes)).convert("RGB"))
    except Exception:
        return False, 1.0

    best_dist = 1.0
    for backend in _BACKENDS:
        try:
            r = DeepFace.verify(
                img1_path=selfie_path,
                img2_path=img_np,
                model_name="ArcFace",
                distance_metric="cosine",
                detector_backend=backend,
                enforce_detection=False,
            )
            dist = r["distance"]
            if dist < best_dist:
                best_dist = dist
            if dist < threshold:
                return True, dist   # confirmed — no need to try more backends
        except Exception:
            continue

    return False, best_dist


# ==========================
# Main UI
# ==========================

st.title("🔍 FindMe – Find Your Face in Google Drive")
st.markdown(
    "Upload **your selfie** and paste a **Google Drive folder link** "
    "(*Anyone with link: Viewer*). FindMe scans every image and finds the ones containing your face."
)

col_left, col_right = st.columns([1, 2])

with col_left:
    selfie_file = st.file_uploader(
        "📸 Upload your selfie",
        type=["jpg", "jpeg", "png", "webp"],
        help="Clear, front-facing photo works best.",
    )
    if selfie_file:
        st.image(selfie_file, caption="Your selfie", width=220)

with col_right:
    drive_link = st.text_input(
        "🔗 Google Drive folder link",
        placeholder="https://drive.google.com/drive/folders/...",
    )
    if drive_link.strip():
        fid_preview = extract_folder_id(drive_link.strip())
        if fid_preview:
            st.success(f"Folder ID detected: `{fid_preview}`")
        else:
            st.error(
                "Couldn't detect a folder ID. "
                "Expected: `https://drive.google.com/drive/folders/<id>`"
            )

run = st.button("🚀 Find My Photos", type="primary", use_container_width=True)

if run:
    if not selfie_file:
        st.error("Please upload a selfie.")
        st.stop()
    if not drive_link.strip():
        st.error("Please paste a Drive folder link.")
        st.stop()
    if not API_KEY:
        st.error("GOOGLE_API_KEY is missing. Add it to your `.env` file.")
        st.stop()

    folder_id = extract_folder_id(drive_link.strip())
    if not folder_id:
        st.error("Invalid Drive link. Could not extract folder ID.")
        st.stop()

    selfie_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            selfie_path = tmp.name
            Image.open(selfie_file).convert("RGB").save(selfie_path, "JPEG")

        # --- Fetch file list ---
        with st.spinner("📂 Fetching image list from Google Drive…"):
            files = get_all_images(folder_id, recursive=scan_subfolders)

        if not files:
            st.warning(
                "No images found. Check that the folder is shared correctly "
                "and contains image files."
            )
            st.stop()

        st.info(
            f"Found **{len(files)}** image(s)"
            + (" across all subfolders" if scan_subfolders else "")
            + ". Scanning for your face…"
        )

        # --- Scanning loop ---
        matched = []
        failed  = []

        progress_bar = st.progress(0.0)
        status_text  = st.empty()
        live_matches = st.container()

        for i, file in enumerate(files):
            name = file.get("name", f"image_{i + 1}")
            status_text.text(f"Scanning {i + 1} / {len(files)}: {name}")

            check_bytes, already_full = get_image_for_check(file)
            if not check_bytes:
                failed.append(name)
                progress_bar.progress((i + 1) / len(files))
                continue

            is_match, dist = verify_face(selfie_path, check_bytes, threshold)

            # Thumbnail failed → retry on full-resolution image
            full_bytes = None
            if not is_match and not already_full:
                full_bytes = get_full_image(file["id"])
                if full_bytes:
                    is_match2, dist2 = verify_face(selfie_path, full_bytes, threshold)
                    if dist2 < dist:
                        dist = dist2
                    if is_match2:
                        is_match = True

            if is_match:
                if already_full:
                    full_bytes = check_bytes
                elif full_bytes is None:
                    full_bytes = get_full_image(file["id"]) or check_bytes

                matched.append((name, file["id"], full_bytes, dist))
                with live_matches:
                    st.image(
                        full_bytes,
                        caption=f"✅ {name}  (dist={dist:.3f})",
                        width=300,
                    )

            progress_bar.progress((i + 1) / len(files))
            if (i + 1) % 10 == 0:
                gc.collect()

        status_text.text(f"Scan complete — {len(files)} images checked.")
        progress_bar.progress(1.0)

        # --- Results ---
        st.divider()
        if matched:
            st.success(f"🎉 Found **{len(matched)}** matching photo(s)!")

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for idx, (name, fid, img_bytes, _dist) in enumerate(matched):
                    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
                    zf.writestr(f"match_{idx + 1:03d}_{safe_name}", img_bytes)

            st.download_button(
                label="📦 Download all matches as ZIP",
                data=zip_buf.getvalue(),
                file_name="findme_matches.zip",
                mime="application/zip",
                type="primary",
            )

            with st.expander("📋 Match details"):
                for idx, (name, fid, _bytes, dist) in enumerate(matched):
                    st.markdown(
                        f"`{idx + 1:03d}` **{name}** — distance `{dist:.4f}` — "
                        f"[Open in Drive](https://drive.google.com/file/d/{fid}/view)"
                    )

        else:
            st.warning(
                "😕 No matching photos found.\n\n"
                "**Tips:**\n"
                "- Try increasing the sensitivity threshold in the sidebar.\n"
                "- Use a clear, well-lit, front-facing selfie.\n"
                "- Make sure the folder contains images where your face is visible."
            )

        if failed:
            with st.expander(f"⚠️ {len(failed)} file(s) could not be fetched"):
                for name in failed:
                    st.text(name)

    finally:
        if selfie_path and os.path.exists(selfie_path):
            try:
                os.unlink(selfie_path)
            except OSError:
                pass
