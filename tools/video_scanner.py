"""
Сервис сканирования ценников: загрузка видео → детекция YOLO → QR-сканирование → результаты.
Запуск: streamlit run tools/video_scanner.py
"""
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "best.onnx"
IMGSZ      = 1280


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


_qr_detector = cv2.QRCodeDetector()


def decode_qr(bgr_crop: np.ndarray) -> str:
    if bgr_crop.size == 0:
        return ""

    h, w = bgr_crop.shape[:2]
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)

    candidates = [
        gray,
        cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC),
        cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray),
    ]

    for candidate in candidates:
        data, _, _ = _qr_detector.detectAndDecode(candidate)
        if data:
            return data

    return ""


def detect_color(bgr_crop: np.ndarray) -> str:
    if bgr_crop.size == 0:
        return ""

    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    mask = (s > 40) & (v > 40) & (v < 240)
    hues = h[mask]

    if hues.size < 50:
        return ""

    red    = int(((hues <= 10) | (hues >= 160)).sum())
    orange = int(((hues > 10) & (hues <= 25)).sum())
    yellow = int(((hues > 25) & (hues <= 70)).sum())
    blue   = int(((hues > 70) & (hues < 160)).sum())

    return max(("red", red), ("orange", orange), ("yellow", yellow), ("blue", blue), key=lambda x: x[1])[0]


def process_video(video_path: str, step_ms: int, conf: float, filename: str = ""):
    model = load_model()

    cap         = cv2.VideoCapture(video_path)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 20
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_ms = int(frame_count / fps * 1000)
    cap.release()

    timestamps    = list(range(0, duration_ms, step_ms))
    progress_bar  = st.progress(0.0, text="Обработка…")
    frame_display = st.empty()
    rows = []

    for i, ts in enumerate(timestamps):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, ts)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        preds = model.predict(frame, imgsz=IMGSZ, conf=conf, verbose=False)[0]

        progress_bar.progress(
            (i + 1) / len(timestamps),
            text=f"Кадр {i + 1}/{len(timestamps)}  ({ts // 1000} с)",
        )

        if len(preds.boxes) == 0:
            continue

        vis    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w   = frame.shape[:2]

        for box in preds.boxes:
            x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            conf_val = float(box.conf[0])

            crop     = frame[y1:y2, x1:x2]
            qr_data  = decode_qr(crop) if crop.size > 0 else ""
            color_name = detect_color(crop) if crop.size > 0 else ""

            # DEBUG: сохраняем первые 3 кропа для ручной проверки
            debug_dir = ROOT / "data" / "debug_crops"
            debug_dir.mkdir(exist_ok=True)
            debug_saved = list(debug_dir.glob("*.jpg"))
            if len(debug_saved) < 3 and crop.size > 0:
                cv2.imwrite(str(debug_dir / f"crop_{ts}_{x1}_{y1}.jpg"), crop)

            color = (0, 220, 80) if qr_data else (255, 140, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 5)
            label = qr_data[:30] if qr_data else f"{conf_val:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            rows.append({
                "filename":                filename,
                "product_name":            "",
                "price_default":           "",
                "price_card":              "",
                "price_discount":          "",
                "barcode":                 "",
                "discount_amount":         "",
                "id_sku":                  "",
                "print_datetime":          "",
                "code":                    "",
                "additional_info":         "",
                "color":                   color_name,
                "special_symbols":         "",
                "frame_timestamp":         ts,
                "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
                "qr_code_barcode":         qr_data,
                "price1_qr":               "",
                "price2_qr":               "",
                "price3_qr":               "",
                "price4_qr":               "",
                "wholesale_level_1_count": "",
                "wholesale_level_1_price": "",
                "wholesale_level_2_count": "",
                "wholesale_level_2_price": "",
                "action_price_qr":         "",
                "action_code_qr":          "",
            })

        frame_display.image(vis, caption=f"t = {ts // 1000} с", width=480)

    progress_bar.empty()
    return rows


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Сканер ценников", layout="wide")
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    header [data-testid="stToolbar"] {display: none;}
    [data-testid="stDeployButton"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Сканер ценников")

uploaded = st.file_uploader("Загрузи видео (.mp4 или .mov)", type=["mp4", "mov"])

col1, col2 = st.columns(2)
step_ms = col1.slider("Шаг между кадрами, мс", 500, 5000, 2000, 500)
conf    = col2.slider("Порог уверенности", 0.10, 0.90, 0.70, 0.05)

if uploaded is not None:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    if st.button("Запустить сканирование"):
        rows = process_video(tmp_path, step_ms, conf, filename=uploaded.name)

        if not rows:
            st.warning("Ценники не найдены.")
        else:
            df       = pd.DataFrame(rows)
            qr_found = df["qr_code_barcode"].astype(bool).sum()
            st.success(f"Найдено боксов: {len(df)}  |  QR считано: {qr_found}")

            st.dataframe(
                df[["filename", "frame_timestamp", "qr_code_barcode",
                    "x_min", "y_min", "x_max", "y_max"]],
                use_container_width=True,
                hide_index=True,
            )

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Скачать CSV",
                data=csv_bytes,
                file_name="scan_results.csv",
                mime="text/csv",
            )
