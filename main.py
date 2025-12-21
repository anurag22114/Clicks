"""
segregate_by_embedding.py

Requirements:
  pip install deepface mtcnn opencv-python
"""

import os
from pathlib import Path
import cv2
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN

# --- CONFIG ---
MODEL_NAME = "Facenet"          # DeepFace model for embeddings (VGG-Face, Facenet, ArcFace, etc.)
DETECTOR = MTCNN()              # face detector
THRESHOLD = 0.4                 # cosine distance threshold (smaller -> stricter)
MATCH_DIR = Path("matches")
OTHER_DIR = Path("others")
VISUALIZE = True                # save a visualization image with boxes and labels
# ----------------


def detect_faces_boxes(rgb_image):
    """Return list of bounding boxes (x, y, w, h) for faces in an RGB image using MTCNN."""
    detections = DETECTOR.detect_faces(rgb_image)
    boxes = []
    for d in detections:
        x, y, w, h = d["box"]
        # mtcnn can give negative coords; clamp
        x = max(0, x)
        y = max(0, y)
        w = max(0, w)
        h = max(0, h)
        boxes.append((x, y, w, h))
    return boxes


def crop_with_margin(rgb, box, margin=0.2):
    """Crop bounding box with some margin (relative). Returns RGB crop."""
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
    """
    Use DeepFace.represent to get embedding for a given RGB crop (numpy array).
    enforce_detection=False to avoid errors if crop is small — but ensure detector did a good job.
    Returns 1D numpy array embedding.
    """
    # DeepFace.represent accepts a numpy array as input for img_path
    rep = DeepFace.represent(img_path=rgb_crop, model_name=model_name, enforce_detection=False)
    # DeepFace.represent may return a list containing a dict or a plain list/ndarray depending on version.
    # Handle common return formats.
    if isinstance(rep, list) and len(rep) > 0:
        r = rep[0]
        if isinstance(r, dict) and "embedding" in r:
            emb = np.array(r["embedding"])
        elif isinstance(r, dict) and "embeddings" in r:
            emb = np.array(r["embeddings"])
        elif isinstance(r, (list, np.ndarray)):
            emb = np.array(r)
        else:
            # fallback: try to convert whole rep to array
            emb = np.array(r)
    elif isinstance(rep, dict) and "embedding" in rep:
        emb = np.array(rep["embedding"])
    else:
        emb = np.array(rep)
    # ensure 1D
    emb = emb.reshape(-1)
    return emb


def cosine_distance(a, b):
    """Return cosine distance (1 - cosine_similarity) between 1D arrays a and b."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    cos_sim = np.dot(a, b) / denom
    return 1.0 - float(cos_sim)


def ensure_dirs():
    MATCH_DIR.mkdir(parents=True, exist_ok=True)
    OTHER_DIR.mkdir(parents=True, exist_ok=True)


def segregate_group_image(target_image_path, group_image_path, threshold=THRESHOLD, visualize=VISUALIZE):
    """
    Compare target face to each detected face in group image and save crops to MATCH_DIR/OTHER_DIR.
    Also returns summary dict.
    """
    ensure_dirs()

    # load images (OpenCV loads BGR)
    target_bgr = cv2.imread(str(target_image_path))
    group_bgr = cv2.imread(str(group_image_path))
    if target_bgr is None:
        raise FileNotFoundError(f"Target image not found or cannot be read: {target_image_path}")
    if group_bgr is None:
        raise FileNotFoundError(f"Group image not found or cannot be read: {group_image_path}")

    # convert to RGB (DeepFace expects RGB arrays)
    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    group_rgb = cv2.cvtColor(group_bgr, cv2.COLOR_BGR2RGB)

    # get target face box(es), pick largest (in case target image has multiple faces)
    t_boxes = detect_faces_boxes(target_rgb)
    if len(t_boxes) == 0:
        # fallback: use full image as target crop
        target_crop = target_rgb
    else:
        # pick largest area
        largest = max(t_boxes, key=lambda b: b[2] * b[3])
        target_crop = crop_with_margin(target_rgb, largest, margin=0.2)

    # get embedding for target
    print("Computing embedding for target image...")
    target_emb = get_embedding_from_crop(target_crop)

    # detect faces in group image
    g_boxes = detect_faces_boxes(group_rgb)
    print(f"Found {len(g_boxes)} face(s) in group image.")

    summary = {"group_image": str(group_image_path), "n_faces": len(g_boxes), "matches": [], "others": []}
    vis_img = group_bgr.copy()  # BGR for drawing

    for idx, box in enumerate(g_boxes, start=1):
        face_crop_rgb = crop_with_margin(group_rgb, box, margin=0.2)
        try:
            emb = get_embedding_from_crop(face_crop_rgb)
        except Exception as e:
            print(f"Warning: embedding failed for face {idx}: {e}")
            emb = None

        label = "other"
        dist = None
        if emb is not None:
            dist = cosine_distance(target_emb, emb)
            if dist <= threshold:
                label = "match"

        # save crop
        if label == "match":
            out_path = MATCH_DIR / f"{Path(group_image_path).stem}_face{idx}_d{dist:.3f}.jpg"
            summary["matches"].append({"face_index": idx, "box": box, "distance": dist, "path": str(out_path)})
        else:
            out_path = OTHER_DIR / f"{Path(group_image_path).stem}_face{idx}_d{dist if dist is not None else 'na'}.jpg"
            summary["others"].append({"face_index": idx, "box": box, "distance": dist, "path": str(out_path)})

        # save image (BGR conversion)
        face_crop_bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), face_crop_bgr)

        # visualization: draw rect + label
        if visualize:
            x, y, w, h = box
            color = (0, 255, 0) if label == "match" else (0, 0, 255)
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color=color, thickness=2)
            text = f"{label} ({dist:.3f})" if dist is not None else label
            cv2.putText(vis_img, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # save visualization
    if visualize:
        vis_out = Path(f"{Path(group_image_path).stem}_vis.jpg")
        cv2.imwrite(str(vis_out), vis_img)
        summary["visualization"] = str(vis_out)

    return summary


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Segregate faces in group images by comparing embeddings to a target person image.")
    p.add_argument("target", help="Path to target person's image (single person preferred)")
    p.add_argument("group", help="Path to group image (or a folder containing group images)")
    p.add_argument("--threshold", type=float, default=THRESHOLD, help="Cosine distance threshold (default 0.4)")
    p.add_argument("--no-vis", action="store_true", help="Disable visualization image")
    args = p.parse_args()

    # if group is a folder, process all images inside
    group_path = Path(args.group)
    if group_path.is_dir():
        results = []
        for img_file in sorted(group_path.iterdir()):
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue
            print(f"\nProcessing {img_file} ...")
            res = segregate_group_image(args.target, img_file, threshold=args.threshold, visualize=not args.no_vis)
            results.append(res)
        print("\nDone. Summary:")
        for r in results:
            print(r)
    else:
        res = segregate_group_image(args.target, group_path, threshold=args.threshold, visualize=not args.no_vis)
        print("\nResult:")
        print(res)
