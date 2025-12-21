import os
from pathlib import Path
import cv2
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN

# simple helper module for the Streamlit app

MODEL_NAME = "Facenet"
DETECTOR = MTCNN()
THRESHOLD = 0.4

BASE_DIR = Path(__file__).parent
MATCH_DIR = BASE_DIR / "matches"
OTHER_DIR = BASE_DIR / "others"
MATCH_DIR.mkdir(parents=True, exist_ok=True)
OTHER_DIR.mkdir(parents=True, exist_ok=True)


def detect_faces_boxes(rgb_image):
    detections = DETECTOR.detect_faces(rgb_image)
    boxes = []
    for d in detections:
        x, y, w, h = d.get("box", (0, 0, 0, 0))
        x = max(0, x)
        y = max(0, y)
        w = max(0, w)
        h = max(0, h)
        boxes.append((x, y, w, h))
    return boxes


def crop_with_margin(rgb, box, margin=0.2):
    x, y, w, h = box
    img_h, img_w = rgb.shape[:2]
    dx = int(w * margin)
    dy = int(h * margin)
    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(img_w, x + w + dx)
    y2 = min(img_h, y + h + dy)
    return rgb[y1:y2, x1:x2]


def get_embedding_from_crop(rgb_crop, model_name=MODEL_NAME):
    rep = DeepFace.represent(img_path=rgb_crop, model_name=model_name, enforce_detection=False)
    if isinstance(rep, list) and len(rep) > 0:
        r = rep[0]
        if isinstance(r, dict) and "embedding" in r:
            emb = np.array(r["embedding"])
        elif isinstance(r, dict) and "embeddings" in r:
            emb = np.array(r["embeddings"])
        elif isinstance(r, (list, np.ndarray)):
            emb = np.array(r)
        else:
            emb = np.array(r)
    elif isinstance(rep, dict) and "embedding" in rep:
        emb = np.array(rep["embedding"])
    else:
        emb = np.array(rep)
    return emb.reshape(-1)


def cosine_distance(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    cos_sim = np.dot(a, b) / denom
    return 1.0 - float(cos_sim)


def _ensure_rgb_array(img):
    # Accept path, BGR numpy (from cv2), or RGB numpy (from PIL->np.array)
    if isinstance(img, (str, Path)):
        bgr = cv2.imread(str(img))
        if bgr is None:
            raise FileNotFoundError(f"Image not found: {img}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if isinstance(img, np.ndarray):
        # if it's BGR (common when loaded with cv2) vs RGB (PIL)
        if img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3:
            # Heuristic: assume if mean of first channel > last channel drastically might be BGR; skip robust detection
            # We assume caller passes RGB (Streamlit's Image->np.array gives RGB). If it's BGR, results still ok for detection but colors may be off.
            return img.copy()
    raise ValueError("Unsupported image format. Provide a numpy RGB array or a file path.")


def segregate_group_image(target_image, group_image, threshold=THRESHOLD, visualize=True):
    """
    Accepts either numpy RGB arrays or file paths.
    Returns a summary dict with keys: group_image, n_faces, matches, others, visualization (RGB numpy array).
    Also saves face crops into matches/others folders under this module.
    """
    target_rgb = _ensure_rgb_array(target_image)
    group_rgb = _ensure_rgb_array(group_image)

    # get target face box(es), pick largest if multiple
    t_boxes = detect_faces_boxes(target_rgb)
    if len(t_boxes) == 0:
        target_crop = target_rgb
    else:
        largest = max(t_boxes, key=lambda b: b[2] * b[3])
        target_crop = crop_with_margin(target_rgb, largest, margin=0.2)

    target_emb = get_embedding_from_crop(target_crop)

    g_boxes = detect_faces_boxes(group_rgb)

    summary = {"group_image": None, "n_faces": len(g_boxes), "matches": [], "others": []}

    vis_bgr = cv2.cvtColor(group_rgb, cv2.COLOR_RGB2BGR)

    for idx, box in enumerate(g_boxes, start=1):
        face_crop_rgb = crop_with_margin(group_rgb, box, margin=0.2)
        try:
            emb = get_embedding_from_crop(face_crop_rgb)
        except Exception:
            emb = None

        label = "other"
        dist = None
        if emb is not None:
            dist = cosine_distance(target_emb, emb)
            if dist <= threshold:
                label = "match"

        if label == "match":
            out_path = MATCH_DIR / f"{Path('group') .stem}_face{idx}_d{dist:.3f}.jpg"
            summary["matches"].append({"face_index": idx, "box": box, "distance": dist, "path": str(out_path)})
        else:
            out_path = OTHER_DIR / f"{Path('group') .stem}_face{idx}_d{dist if dist is not None else 'na'}.jpg"
            summary["others"].append({"face_index": idx, "box": box, "distance": dist, "path": str(out_path)})

        # save crop (convert to BGR for cv2)
        face_crop_bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), face_crop_bgr)

        # draw on visualization image
        x, y, w, h = box
        color = (0, 255, 0) if label == "match" else (0, 0, 255)
        cv2.rectangle(vis_bgr, (x, y), (x + w, y + h), color=color, thickness=2)
        text = f"{label} ({dist:.3f})" if dist is not None else label
        cv2.putText(vis_bgr, text, (x, max(10, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    if visualize:
        summary["visualization"] = vis_rgb
    else:
        summary["visualization"] = None

    return summary