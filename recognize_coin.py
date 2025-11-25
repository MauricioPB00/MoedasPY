import sys
import json
import os
import cv2
import numpy as np
from PIL import Image
import torch
import clip

from ultralytics.utils import LOGGER
import logging
LOGGER.setLevel(logging.ERROR)

from ultralytics import YOLO


def respond(obj):
    print(json.dumps(obj, ensure_ascii=False))


# ============================
# UTILITÁRIOS DE EMBEDDINGS
# ============================
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_embeddings():
    path = os.path.join(os.path.dirname(__file__), "moedas_embeddings.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_best_match(crop_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    img = Image.open(crop_path).convert("RGB")
    img_preprocessed = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(img_preprocessed)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        emb = emb.cpu().numpy().flatten()

    db = load_embeddings()
    if db is None:
        return None, None

    best = None
    best_score = -1

    for entry in db:
        if entry.get("embedding_obverse"):
            score_obv = cosine_similarity(emb, np.array(entry["embedding_obverse"]))
            if score_obv > best_score:
                best_score = score_obv
                best = {"id": entry["id"], "side": "obverse"}

        if entry.get("embedding_reverse"):
            score_rev = cosine_similarity(emb, np.array(entry["embedding_reverse"]))
            if score_rev > best_score:
                best_score = score_rev
                best = {"id": entry["id"], "side": "reverse"}

    return best, best_score


# ============================
# MAIN YOLO + MATCHING
# ============================
def main():
    if len(sys.argv) < 2:
        respond({"error": "No image path received"})
        return

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        respond({"error": "Image file does not exist", "path": image_path})
        return

    image = cv2.imread(image_path)
    if image is None:
        respond({"error": "Unable to read image file"})
        return

    saved_original = os.path.join(os.path.dirname(__file__), "temp_original.jpg")
    cv2.imwrite(saved_original, image)

    model_path = r"C:\Users\Usuario\.pyenv\runs\detect\train26\weights\best.pt"
    if not os.path.exists(model_path):
        respond({"error": "YOLO model not found", "expected_path": model_path})
        return

    try:
        model = YOLO(model_path)
    except Exception as e:
        respond({"error": "Failed to load YOLO model", "details": str(e)})
        return

    try:
        results = model(image, verbose=False)
        result = results[0]
    except Exception as e:
        respond({"error": "YOLO inference failed", "details": str(e)})
        return

    annotated = result.plot()
    detected_output_path = os.path.join(os.path.dirname(__file__), "temp_detected.jpg")
    cv2.imwrite(detected_output_path, annotated)

    if len(result.boxes) == 0:
        respond({
            "detected": False,
            "message": "Nenhuma moeda encontrada",
            "saved_original": saved_original,
            "output_detected": detected_output_path
        })
        return

    best_box = max(
        result.boxes,
        key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
    )

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    cropped = image[y1:y2, x1:x2]

    temp_crop_path = os.path.join(os.path.dirname(__file__), "temp_crop.jpg")
    cv2.imwrite(temp_crop_path, cropped)

    match, score = find_best_match(temp_crop_path)

    respond({
        "detected": True,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "saved_original": saved_original,
        "output_detected": detected_output_path,
        "cropped_path": temp_crop_path,
        "best_match": match,
        "similarity": score
    })


if __name__ == "__main__":
    main()
