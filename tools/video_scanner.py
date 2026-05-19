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
from boxmot.trackers.bytetrack.bytetrack import ByteTrack

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


def frame_sharpness(frame: np.ndarray) -> float:
    small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def crop_sharpness(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def process_video(video_path: str, sharpness_threshold: float, conf: float, filename: str = ""):
    model   = load_model()
    cap     = cv2.VideoCapture(video_path)
    fps     = cap.get(cv2.CAP_PROP_FPS) or 20
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker    = ByteTrack(track_buffer=60, frame_rate=int(fps))
    track_data = {}   # track_id -> лучший кадр + накопленные данные

    progress_bar  = st.progress(0.0, text="Обработка…")
    frame_display = st.empty()
    stats         = st.empty()
    processed = 0
    skipped   = 0

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        ts = int(frame_idx / fps * 1000)

        progress_bar.progress(
            (frame_idx + 1) / total_frames,
            text=f"Кадр {frame_idx + 1}/{total_frames}  |  резких: {processed}  |  пропущено: {skipped}  |  треков: {len(track_data)}",
        )

        if frame_sharpness(frame) < sharpness_threshold:
            skipped += 1
            continue

        processed += 1
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        preds = model.predict(frame, imgsz=IMGSZ, conf=conf, verbose=False)[0]

        if len(preds.boxes) > 0:
            dets = np.array([
                [*[round(v) for v in box.xyxy[0].tolist()], float(box.conf[0]), 0]
                for box in preds.boxes
            ], dtype=np.float32)
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        tracks = tracker.update(dets, frame)

        if len(tracks) == 0:
            continue

        vis  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        for t in tracks:
            x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
            track_id = int(t[4])
            conf_val = float(t[5])

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop  = frame[y1:y2, x1:x2]
            sharp = crop_sharpness(crop)

            prev = track_data.get(track_id, {})

            # обновляем лучший кроп если текущий резче
            if sharp > prev.get("sharpness", -1):
                track_data[track_id] = {
                    **prev,
                    "crop":      crop.copy(),
                    "sharpness": sharp,
                    "ts":        ts,
                    "bbox":      (x1, y1, x2, y2),
                    "conf":      conf_val,
                }

            # пробуем считать QR на каждом кадре пока не получится
            if not track_data[track_id].get("qr_data"):
                qr = decode_qr(crop)
                if qr:
                    track_data[track_id]["qr_data"] = qr

            has_qr = bool(track_data[track_id].get("qr_data"))
            color  = (0, 220, 80) if has_qr else (255, 140, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 5)
            cv2.putText(vis, f"#{track_id}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame_display.image(vis, caption=f"t = {ts // 1000} с", width=480)

    cap.release()
    progress_bar.empty()
    stats.caption(
        f"Всего кадров: {total_frames}  |  резких: {processed}  |  "
        f"пропущено: {skipped}  |  уникальных ценников: {len(track_data)}"
    )

    # собираем финальный CSV — один ценник = одна строка
    rows = []
    for data in track_data.values():
        crop       = data["crop"]
        x1, y1, x2, y2 = data["bbox"]
        color_name = detect_color(crop)
        qr_data    = data.get("qr_data", "")

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
            "frame_timestamp":         data["ts"],
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
sharpness_threshold = col1.slider("Порог резкости", 100, 2000, 500, 50,
                                   help="Смазанные кадры ниже порога пропускаются. Выше = строже фильтр.")
conf = col2.slider("Порог уверенности", 0.10, 0.90, 0.70, 0.05)

if uploaded is not None:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    if st.button("Запустить сканирование"):
        rows = process_video(tmp_path, sharpness_threshold, conf, filename=uploaded.name)

        if not rows:
            st.warning("Ценники не найдены.")
        else:
            df       = pd.DataFrame(rows)
            qr_found = df["qr_code_barcode"].astype(bool).sum()
            st.success(f"Уникальных ценников: {len(df)}  |  QR считано: {qr_found}")

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
