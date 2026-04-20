"""
sketch shape classifier
──────────────────────────────────────────────────────────────
마스크 이미지의 외형(shape)을 분류하고 inpainting용 prompt를 반환합니다.

우선순위:
  1. CLIP zero-shot 분류 (transformers 설치 시 자동 사용)
  2. models/sketch_classifier.pt 가 있으면 YOLO 분류 모델 사용
  3. OpenCV 컨투어 분석 (항상 fallback)
"""

import cv2
import math
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "sketch_classifier.pt"

# ──────────────────────────────────────────────────────────────
# Shape → inpainting subject 매핑
# ──────────────────────────────────────────────────────────────
SHAPE_TO_PROMPT: dict[str, str] = {
    # ── 자연/날씨 ──
    "heart":        "a heart",
    "star":         "a five-pointed star",
    "cloud":        "a cloud",
    "moon":         "a crescent moon",
    "rainbow":      "a rainbow arc",
    "lightning":    "a lightning bolt",
    "fire":         "a flame fire",
    "flower":       "a daisy flower",
    "leaf":         "a leaf",
    "butterfly":    "a butterfly",
    # ── 패션/소품 ──
    "sunglasses":   "sunglasses",
    "crown":        "a crown",
    "hat":          "a party hat",
    "bow":          "a ribbon bow",
    "diamond":      "a diamond gem",
    "cat_ears":     "cat ears",
    "rabbit_ears":  "bunny ears",
    "mustache":      "a mustache",
    "whiskers":     "cat whiskers",
    # ── 기타 ──
    "arrow":        "an arrow",
    "music_note":   "a musical note",
    "speech_bubble":"a speech bubble",
    "bomb":         "a cartoon bomb",
    "unknown":      "a decorative accessory",
}

# CLIP 후보 텍스트 — 각 shape의 손그림 특징을 설명
_CLIP_CANDIDATES = {
    # ── 자연/날씨 ──
    "heart":        "a hand-drawn filled heart shape, love symbol, sketch",
    "star":         "a hand-drawn five-pointed star shape, sketch",
    "cloud":        "a hand-drawn fluffy cloud shape with bumpy top, sketch",
    "moon":         "a hand-drawn crescent moon shape, sketch",
    "rainbow":      "a hand-drawn rainbow arc shape with curves, sketch",
    "lightning":    "a hand-drawn lightning bolt zigzag shape, sketch",
    "fire":         "a hand-drawn flame fire shape with pointed top, sketch",
    "flower":       "a hand-drawn flower with petals around center, sketch",
    "leaf":         "a hand-drawn leaf shape with pointed tip, sketch",
    "butterfly":    "a hand-drawn butterfly shape with two wings, sketch",
    # ── 패션/소품 ──
    "sunglasses":   "a hand-drawn sunglasses shape with two round lenses, sketch",
    "crown":        "a hand-drawn crown shape with pointed tips on top, sketch",
    "hat":          "a hand-drawn party hat or top hat shape, sketch",
    "bow":          "a hand-drawn ribbon bow shape with two rounded side loops and a center knot, sketch",
    "diamond":      "a hand-drawn diamond or rhombus gem shape, a single closed polygon, sketch",
    "cat_ears":     "a hand-drawn cat ears sketch, either a closed simple shape with two pointed triangular ears connected together, or a wearable cat ears accessory above a person's head with the bottom open like a headband silhouette, sketch",
    "rabbit_ears":  "a hand-drawn bunny ears sketch, either a closed simple shape with two long upright ears connected together, or a wearable bunny ears accessory above a person's head with the bottom open like a headband silhouette, sketch",
    "mustache":     "a hand-drawn mustache shape with two symmetrical curved parts spreading left and right from the center, sketch",
    "whiskers":     "a hand-drawn cat whiskers shape, several thin horizontal lines spreading left and right from the center, sketch",
    # ── 기타 ──
    "arrow":        "a hand-drawn arrow shape with one shaft and one pointed arrowhead showing direction, sketch",
    "music_note":   "a hand-drawn musical note shape with a round note head and a thin vertical stem, sketch",
    "speech_bubble":"a hand-drawn speech bubble or chat balloon shape, sketch",
    "bomb":         "a hand-drawn round bomb shape with a fuse on top, sketch",
}

# CLIP 모델 캐싱 (최초 1회 로드)
_clip_model = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        print("  CLIP 모델 로딩 중... (최초 1회)")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        print("  CLIP 로딩 완료.")
    return _clip_model, _clip_processor


def classify(mask_path) -> tuple[str, str, float | None]:
    """
    마스크 이미지에서 shape을 분류한다.

    Returns:
        (shape_name, inpaint_prompt, clip_top1_confidence)
    """
    mask_path = Path(mask_path)

    ear_shape = _classify_ear_accessory(mask_path)
    if ear_shape is not None:
        print(f"  귀 액세서리 규칙 기반 분류: {ear_shape}")
        prompt = SHAPE_TO_PROMPT.get(ear_shape, SHAPE_TO_PROMPT["unknown"])
        return ear_shape, prompt, 1.0

    # 우선순위 1: CLIP zero-shot
    try:
        import transformers  # noqa: F401
        print("  CLIP zero-shot으로 분류 시도...")
        shape, confidence = _classify_with_clip(mask_path)
    except ImportError:
        # 우선순위 2: YOLO
        if MODEL_PATH.exists():
            print("  YOLO 모델로 분류 시도...")
            shape = _classify_with_yolo(mask_path)
        else:
            print("  OpenCV 컨투어 분석으로 분류 시도...")
            shape = _classify_with_opencv(mask_path)
        confidence = None

    prompt = SHAPE_TO_PROMPT.get(shape, SHAPE_TO_PROMPT["unknown"])
    return shape, prompt, confidence


def _classify_ear_accessory(mask_path: Path) -> str | None:
    """분리된 두 개의 귀 실루엣을 규칙 기반으로 먼저 분류한다."""
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    height, width = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= 150]
    if len(contours) != 2:
        return None

    contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])
    features = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        rect_area = max(1, w * h)
        fill_ratio = area / rect_area
        epsilon = 0.04 * cv2.arcLength(contour, True)
        vertices = len(cv2.approxPolyDP(contour, epsilon, True))
        features.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "aspect": w / h if h > 0 else 999,
            "fill_ratio": fill_ratio,
            "vertices": vertices,
        })

    left, right = features
    gap = right["x"] - (left["x"] + left["w"])
    avg_height = (left["h"] + right["h"]) / 2
    avg_width = (left["w"] + right["w"]) / 2
    avg_aspect = (left["aspect"] + right["aspect"]) / 2
    avg_fill = (left["fill_ratio"] + right["fill_ratio"]) / 2
    avg_vertices = (left["vertices"] + right["vertices"]) / 2

    if gap < -10 or gap > avg_height * 0.9:
        return None
    if left["y"] > height * 0.35 or right["y"] > height * 0.35:
        return None
    if left["h"] < height * 0.18 or right["h"] < height * 0.18:
        return None
    if abs(left["h"] - right["h"]) / max(left["h"], right["h"]) > 0.35:
        return None
    if abs(left["area"] - right["area"]) / max(left["area"], right["area"]) > 0.45:
        return None
    if left["aspect"] > 1.0 or right["aspect"] > 1.0:
        return None

    # 토끼 귀: 더 길고 가늘며, 곡선이 많아 꼭짓점 수가 비교적 많다.
    if avg_aspect < 0.68 and avg_height / max(avg_width, 1) > 1.6 and avg_vertices >= 5:
        return "rabbit_ears"

    # 고양이 귀: 상대적으로 짧고 삼각형에 가까워 채움 비율이 낮고 꼭짓점 수가 적다.
    if avg_aspect < 0.95 and avg_fill < 0.62:
        return "cat_ears"

    return None


# ──────────────────────────────────────────────────────────────
# CLIP zero-shot 분류 (transformers 설치 시 자동 활성화)
# ──────────────────────────────────────────────────────────────
def _classify_with_clip(mask_path: Path) -> tuple[str, float]:
    import torch
    from PIL import Image

    model, processor = _load_clip()

    # 마스크를 RGB로 변환 (흰색 shape, 검정 배경 → 반전하여 검정 sketch 느낌)
    with Image.open(mask_path) as img:
        img_rgb = img.convert("RGB")

    labels = list(_CLIP_CANDIDATES.keys())
    texts  = list(_CLIP_CANDIDATES.values())

    inputs = processor(
        text=texts,
        images=img_rgb,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits  = outputs.logits_per_image  # (1, num_classes)
        probs   = logits.softmax(dim=1)[0]

    best_idx   = probs.argmax().item()
    best_label = labels[best_idx]
    best_prob  = probs[best_idx].item()

    # 신뢰도 출력
    top3 = sorted(zip(labels, probs.tolist()), key=lambda x: -x[1])[:3]
    print(f"  CLIP Top-3: " + ", ".join(f"{l}({p:.2f})" for l, p in top3))

    # 신뢰도 낮으면 OpenCV fallback
    if best_prob < 0.25:
        print(f"  신뢰도 낮음({best_prob:.2f}) → OpenCV fallback")
        return _classify_with_opencv(mask_path), best_prob

    return best_label, best_prob


# ──────────────────────────────────────────────────────────────
# YOLO 분류 모델 (학습 후 models/sketch_classifier.pt 배치 시 활성화)
# ──────────────────────────────────────────────────────────────
def _classify_with_yolo(mask_path: Path) -> str:
    try:
        from ultralytics import YOLO # type: ignore
        model = YOLO(str(MODEL_PATH))
        results = model(str(mask_path), verbose=False)
        return results[0].names[results[0].probs.top1]
    except Exception as e:
        print(f"  [YOLO] 분류 실패: {e} → OpenCV fallback")
        return _classify_with_opencv(mask_path)


# ──────────────────────────────────────────────────────────────
# OpenCV 컨투어 기반 shape 분류 (즉시 동작, 학습 불필요)
# ──────────────────────────────────────────────────────────────
def _classify_with_opencv(mask_path: Path) -> str:
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "unknown"

    # 이진화
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"

    # 가장 큰 컨투어 선택
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0 or area < 100:
        return "unknown"

    # ── 기본 지표 ──
    # 원형도: 1.0 = 완전한 원
    circularity = 4 * math.pi * area / (perimeter ** 2)

    # 볼록 껍질(convex hull) 대비 채움 비율
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # 다각형 근사 꼭짓점 수
    epsilon = 0.04 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)

    # 유의미한 오목부(convexity defect) 수
    num_defects = _count_significant_defects(contour, threshold=8)

    # 종횡비
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 1.0

    # ── 분류 규칙 ──
    if circularity > 0.82:
        return "circle"

    if vertices == 3:
        return "triangle"

    if vertices == 4:
        return "square" if 0.85 <= aspect_ratio <= 1.15 else "rectangle"

    # 별: 볼록 결함 4개 이상, solidity 낮음
    if num_defects >= 4 and solidity < 0.65:
        return "star"

    # 하트: 볼록 결함 정확히 1~2개, 위쪽 두 돌출부 특징
    if num_defects in (1, 2) and 0.55 < solidity < 0.90 and aspect_ratio < 1.2:
        return "heart"

    # 화살표: 세로로 길거나 가로로 긴 비대칭 다각형
    if vertices in (5, 6, 7) and (aspect_ratio > 1.6 or aspect_ratio < 0.6):
        return "arrow"

    # 오각형 이상의 둥그스름한 형태
    if vertices >= 5 and circularity > 0.65:
        return "circle"

    return "unknown"


def _count_significant_defects(contour, threshold: float = 8) -> int:
    """컨투어의 유의미한 오목부 수를 반환한다."""
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if len(hull_idx) <= 3:
        return 0
    defects = cv2.convexityDefects(contour, hull_idx)
    if defects is None:
        return 0
    return sum(1 for d in defects if d[0][3] / 256 > threshold)
