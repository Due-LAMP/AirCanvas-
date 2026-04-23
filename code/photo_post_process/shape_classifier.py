"""
sketch shape classifier
──────────────────────────────────────────────────────────────
마스크 이미지의 외형(shape)을 분류하고 inpainting용 prompt를 반환합니다.

우선순위:
  1. CLIP zero-shot 분류 (transformers 설치 시 자동 사용)
  2. models/sketch_classifier.pt 가 있으면 YOLO 분류 모델 사용
  3. OpenCV 컨투어 분석 (항상 fallback)
"""

import os
import cv2
import math
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "sketch_classifier.pt"
CLIP_MODEL_ID = os.getenv("AIRCANVAS_CLIP_MODEL", "openai/clip-vit-base-patch32")
# CLIP_MODEL_ID = os.getenv("AIRCANVAS_CLIP_MODEL", "openai/clip-vit-large-patch14")
CLIP_FALLBACK_MODEL_ID = "openai/clip-vit-base-patch32"
LOW_CONFIDENCE_FILLED_RETRY_THRESHOLD = 0.25
LOW_CONFIDENCE_GENERIC_PROMPT_THRESHOLD = 0.3

# ──────────────────────────────────────────────────────────────
# Shape → inpainting subject 매핑
# ──────────────────────────────────────────────────────────────
SHAPE_TO_PROMPT: dict[str, str] = {
    # ── 자연/날씨 ──
    "heart":        "heart",
    "star":         "five-pointed star",
    "cloud":        "cloud",
    "moon":         "crescent moon",
    # "rainbow":      "rainbow arc",
    "lightning":    "lightning bolt",
    "fire":         "flame fire",
    "flower":       "daisy flower",
    # "leaf":         "leaf",
    # "butterfly":    "butterfly",
    "tree":         "tree",

    # ── 패션/소품 ──
    "sunglasses":   "sunglasses",
    "crown":        "crown",
    "hat":          "party hat",
    "bow":          "ribbon bow",
    # "diamond":      "diamond gem",
    "cat_ears":     "cat ears",
    "rabbit_ears":  "bunny ears",
    "mustache":      "mustache",
    "whiskers":     "cat whiskers",
    # ── 기타 ──
    # "arrow":        "arrow",
    "music_note":   "musical note",
    "speech_bubble":"speech bubble",
    # "bomb":         "cartoon bomb",
    "unknown":      "decorative accessory",
}

# CLIP 후보 텍스트 — 각 shape의 손그림 특징을 설명
_CLIP_CANDIDATES = {
    # ── 자연/날씨 ──
    "heart":        "a filled heart with two rounded lobes and one bottom point",
    "star":         "a five-pointed star with five sharp tips and radial symmetry",
    "cloud":        "a soft rounded cloud with a bumpy top and no sharp tips",
    "moon":         "a moon, thin crescent moon arc open on one side",
    # "rainbow":      "a rainbow, smooth wide rainbow arc with nested curved bands",
    "lightning":    "a lightning bolt, single sharp lightning bolt with a zigzag body and no left-right symmetry",
    "fire":         "a flame shape with a pointed top and curved sides",
    "flower":       "a flower with a center and several rounded petals around it",
    # "leaf":         "a hand-drawn leaf shape with pointed tip, sketch",
    # "butterfly":    "a butterfly with two left-right wings and a narrow center body",
    "tree":         "a tree with a narrow trunk and a rounded leafy canopy",
    "fruit":        "a fruit, round shape with a small stem or leaf such as orange or apple",
    # ── 패션/소품 ──
    "sunglasses":   "a sunglasses, wide horizontal glasses accessory with two left-right lenses joined by a thin center bridge",
    "crown":        "a crown with a flat bottom band and three or more spikes on top, symmetric",
    "hat":          "a hat, cone party hat with one top point and a wide flat bottom, a single centered hat silhouette",
    "bow":          "a ribbon bow with two side loops and a small center knot",
    # "diamond":      "a hand-drawn diamond or rhombus gem shape, a single closed polygon, sketch",
    "cat_ears":     "two separate pointed triangular cat ear shapes at the top, left and right, like a cat ear headband, not a rounded speech bubble",
    "rabbit_ears":  "two separate long narrow upright bunny ear shapes at the top, left and right, like a headband",
    "mustache":     "a mustache with two mirrored curved halves and a split center, wide and horizontal",
    "whiskers":     "thin whisker lines extending left and right from the center, not a closed shape",
    # ── 기타 ──
    # "arrow":        "a hand-drawn arrow shape with one shaft and one pointed arrowhead showing direction, sketch",
    "music_note":   "a musical note with one round note head and one thin vertical stem",
    "speech_bubble":"one single rounded closed speech bubble with a small tail, not two separate pointed ear shapes",
    # "bomb":         "a hand-drawn round bomb shape with a fuse on top, sketch",
}

# CLIP 모델 캐싱 (최초 1회 로드)
_clip_model = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        import logging
        import transformers as _tr
        _tr.logging.set_verbosity_error()
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from transformers import CLIPProcessor, CLIPModel
        model_id = CLIP_MODEL_ID
        print(f"  CLIP 모델 로딩 중... (최초 1회, {model_id})")
        try:
            _clip_model = CLIPModel.from_pretrained(model_id)
            _clip_processor = CLIPProcessor.from_pretrained(model_id)
        except Exception as clip_error:
            if model_id == CLIP_FALLBACK_MODEL_ID:
                raise
            print(f"  CLIP 로딩 실패: {clip_error}")
            print(f"  fallback 모델로 재시도: {CLIP_FALLBACK_MODEL_ID}")
            _clip_model = CLIPModel.from_pretrained(CLIP_FALLBACK_MODEL_ID)
            _clip_processor = CLIPProcessor.from_pretrained(CLIP_FALLBACK_MODEL_ID)
        _clip_model.eval()
        print("  CLIP 로딩 완료.")
    return _clip_model, _clip_processor


def classify(mask_path, debug_label: str | None = None) -> tuple[str, str, float | None]:
    """
    마스크 이미지에서 shape을 분류한다.

    Returns:
        (shape_name, inpaint_prompt, clip_top1_confidence)
    """
    mask_path = Path(mask_path)

    ear_shape = _classify_ear_accessory(mask_path)
    if ear_shape is not None:
        prefix = f"[{debug_label}] " if debug_label else ""
        print(f"  {prefix}귀 액세서리 규칙 기반 분류: {ear_shape}")
        prompt = SHAPE_TO_PROMPT.get(ear_shape, SHAPE_TO_PROMPT["unknown"])
        return ear_shape, prompt, 1.0

    # 우선순위 1: CLIP zero-shot
    try:
        import transformers  # noqa: F401
        prefix = f"[{debug_label}] " if debug_label else ""
        print(f"  {prefix}CLIP zero-shot으로 분류 시도...")
        shape, confidence = _classify_with_clip(mask_path, debug_label=debug_label)
    except ImportError:
        # 우선순위 2: YOLO
        if MODEL_PATH.exists():
            print("  YOLO 모델로 분류 시도...")
            shape = _classify_with_yolo(mask_path)
            confidence = None
        else:
            prefix = f"[{debug_label}] " if debug_label else ""
            print(f"  {prefix}분류 모델 없음 → generic accessory로 처리")
            shape = "unknown"
            confidence = 0.0

    prompt = SHAPE_TO_PROMPT.get(shape, SHAPE_TO_PROMPT["unknown"])
    return _apply_low_confidence_generic_fallback(shape, prompt, confidence, debug_label=debug_label)


def _build_contour_feature(contour) -> dict[str, float]:
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    rect_area = max(1, w * h)
    fill_ratio = area / rect_area
    epsilon = 0.04 * cv2.arcLength(contour, True)
    vertices = len(cv2.approxPolyDP(contour, epsilon, True))
    return {
        "x": float(x),
        "y": float(y),
        "w": float(w),
        "h": float(h),
        "area": float(area),
        "aspect": float(w / h) if h > 0 else 999.0,
        "fill_ratio": float(fill_ratio),
        "vertices": float(vertices),
    }


def _classify_paired_ear_features(features: list[dict[str, float]], height: int, width: int) -> str | None:
    if len(features) != 2:
        return None

    left, right = sorted(features, key=lambda feature: feature["x"])
    gap = right["x"] - (left["x"] + left["w"])
    avg_height = (left["h"] + right["h"]) / 2
    avg_width = (left["w"] + right["w"]) / 2
    avg_aspect = (left["aspect"] + right["aspect"]) / 2
    avg_fill = (left["fill_ratio"] + right["fill_ratio"]) / 2
    avg_vertices = (left["vertices"] + right["vertices"]) / 2
    pair_span = (right["x"] + right["w"]) - left["x"]

    if gap < -min(left["w"], right["w"]) * 0.25:
        return None
    if gap > max(avg_width * 2.5, width * 0.3):
        return None
    if pair_span < width * 0.16:
        return None
    if left["y"] > height * 0.6 or right["y"] > height * 0.6:
        return None
    if left["h"] < height * 0.12 or right["h"] < height * 0.12:
        return None
    if abs(left["h"] - right["h"]) / max(left["h"], right["h"]) > 0.5:
        return None
    if abs(left["area"] - right["area"]) / max(left["area"], right["area"]) > 0.8:
        return None
    if avg_aspect > 1.35:
        return None

    if avg_aspect < 0.72 and avg_height / max(avg_width, 1.0) > 1.45:
        return "rabbit_ears"

    if avg_fill < 0.8 or avg_vertices <= 8:
        return "cat_ears"

    return "cat_ears"


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
    features = [_build_contour_feature(contour) for contour in contours]
    return _classify_paired_ear_features(features, height, width)


def _prepare_clip_mask_rgb(mask_gray: np.ndarray) -> np.ndarray:
    """CLIP 입력용으로 마스크를 타이트 crop + margin + square pad 한다."""
    if mask_gray.ndim == 3:
        mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    nonzero = cv2.findNonZero(binary)
    if nonzero is None:
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    x, y, w, h = cv2.boundingRect(nonzero)
    margin = max(12, int(max(w, h) * 0.18))
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(binary.shape[1], x + w + margin)
    y2 = min(binary.shape[0], y + h + margin)

    cropped = binary[y1:y2, x1:x2]
    crop_h, crop_w = cropped.shape[:2]
    side = max(crop_h, crop_w)
    square = np.zeros((side, side), dtype=np.uint8)
    offset_y = (side - crop_h) // 2
    offset_x = (side - crop_w) // 2
    square[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w] = cropped

    if side < 224:
        square = cv2.resize(square, (224, 224), interpolation=cv2.INTER_NEAREST)

    return cv2.cvtColor(square, cv2.COLOR_GRAY2RGB)


def _run_clip_probs(image_rgb: np.ndarray, labels: list[str], texts: list[str], model, processor):
    import torch
    from PIL import Image

    inputs = processor(
        text=texts,
        images=Image.fromarray(image_rgb),
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.logits_per_image.softmax(dim=1)[0]


def _print_clip_top3(labels: list[str], probs, debug_label: str | None = None, tag: str = "CLIP Top-3"):
    prefix = f"[{debug_label}] " if debug_label else ""
    top3 = sorted(zip(labels, probs.tolist()), key=lambda x: -x[1])[:3]
    print(f"  {prefix}{tag}: " + ", ".join(f"{label}({score:.2f})" for label, score in top3))
    return top3


# ──────────────────────────────────────────────────────────────
# CLIP zero-shot 분류 (transformers 설치 시 자동 활성화)
# ──────────────────────────────────────────────────────────────
def _classify_with_clip(mask_path: Path, debug_label: str | None = None) -> tuple[str, float]:
    model, processor = _load_clip()

    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        return "unknown", 0.0
    raw_rgb = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB)
    crop_rgb = _prepare_clip_mask_rgb(mask_gray)

    labels = list(_CLIP_CANDIDATES.keys())
    texts  = list(_CLIP_CANDIDATES.values())

    raw_probs = _run_clip_probs(raw_rgb, labels, texts, model, processor)

    best_idx   = raw_probs.argmax().item()
    best_label = labels[best_idx]
    best_prob  = raw_probs[best_idx].item()

    _print_clip_top3(labels, raw_probs, debug_label=debug_label, tag="CLIP Top-3")

    if best_prob >= 0.25:
        return best_label, best_prob

    crop_probs = _run_clip_probs(crop_rgb, labels, texts, model, processor)
    crop_idx = crop_probs.argmax().item()
    crop_label = labels[crop_idx]
    crop_prob = crop_probs[crop_idx].item()
    _print_clip_top3(labels, crop_probs, debug_label=debug_label, tag="CLIP crop Top-3")

    prefix = f"[{debug_label}] " if debug_label else ""
    if crop_prob >= 0.25 and crop_prob - best_prob >= 0.15:
        print(f"  {prefix}crop 결과 채택({crop_label}, {crop_prob:.2f})")
        return crop_label, crop_prob

    print(f"  {prefix}신뢰도 낮음({best_prob:.2f}) → generic accessory fallback")
    return "unknown", best_prob




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


# ──────────────────────────────────────────────────────────────
# 디스크 I/O 없이 numpy array 에서 직접 분류 (고속 경로)
# ──────────────────────────────────────────────────────────────

def _classify_ear_accessory_from_array(img: np.ndarray) -> str | None:
    """_classify_ear_accessory 의 array 버전."""
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    height, width = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= 150]
    features = [_build_contour_feature(contour) for contour in contours]
    return _classify_paired_ear_features(features, height, width)


def _classify_with_opencv_from_array(img: np.ndarray) -> str:
    """_classify_with_opencv 의 array 버전."""
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0 or area < 100:
        return "unknown"
    circularity = 4 * math.pi * area / (perimeter ** 2)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    epsilon = 0.04 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    num_defects = _count_significant_defects(contour, threshold=8)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 1.0
    if circularity > 0.82: return "circle"
    if vertices == 3: return "triangle"
    if vertices == 4: return "square" if 0.85 <= aspect_ratio <= 1.15 else "rectangle"
    if num_defects >= 4 and solidity < 0.65: return "star"
    if num_defects in (1, 2) and 0.55 < solidity < 0.90 and aspect_ratio < 1.2: return "heart"
    if vertices in (5, 6, 7) and (aspect_ratio > 1.6 or aspect_ratio < 0.6): return "arrow"
    if vertices >= 5 and circularity > 0.65: return "circle"
    return "unknown"


def _classify_with_clip_from_array(img: np.ndarray, debug_label: str | None = None) -> tuple[str, float]:
    """_classify_with_clip 의 array 버전 — PIL.Image.fromarray 로 파일 I/O 생략."""
    model, processor = _load_clip()
    raw_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img
    crop_rgb = _prepare_clip_mask_rgb(img)
    labels = list(_CLIP_CANDIDATES.keys())
    texts  = list(_CLIP_CANDIDATES.values())
    raw_probs = _run_clip_probs(raw_rgb, labels, texts, model, processor)
    best_idx   = raw_probs.argmax().item()
    best_label = labels[best_idx]
    best_prob  = raw_probs[best_idx].item()
    _print_clip_top3(labels, raw_probs, debug_label=debug_label, tag="CLIP Top-3")
    if best_prob >= 0.25:
        return best_label, best_prob

    crop_probs = _run_clip_probs(crop_rgb, labels, texts, model, processor)
    crop_idx = crop_probs.argmax().item()
    crop_label = labels[crop_idx]
    crop_prob = crop_probs[crop_idx].item()
    _print_clip_top3(labels, crop_probs, debug_label=debug_label, tag="CLIP crop Top-3")

    prefix = f"[{debug_label}] " if debug_label else ""
    if crop_prob >= 0.25 and crop_prob - best_prob >= 0.15:
        print(f"  {prefix}crop 결과 채택({crop_label}, {crop_prob:.2f})")
        return crop_label, crop_prob

    print(f"  {prefix}신뢰도 낮음({best_prob:.2f}) → generic accessory fallback")
    return "unknown", best_prob


def _classify_from_array_once(mask_gray: np.ndarray, debug_label: str | None = None) -> tuple[str, str, float | None]:
    """단일 마스크 배열에 대해 한 번만 shape 분류를 수행한다."""
    ear_shape = _classify_ear_accessory_from_array(mask_gray)
    if ear_shape is not None:
        prefix = f"[{debug_label}] " if debug_label else ""
        print(f"  {prefix}귀 액세서리 규칙 기반 분류: {ear_shape}")
        prompt = SHAPE_TO_PROMPT.get(ear_shape, SHAPE_TO_PROMPT["unknown"])
        return ear_shape, prompt, 1.0

    try:
        import transformers  # noqa: F401
        prefix = f"[{debug_label}] " if debug_label else ""
        print(f"  {prefix}CLIP zero-shot으로 분류 시도...")
        shape, confidence = _classify_with_clip_from_array(mask_gray, debug_label=debug_label)
    except ImportError:
        if MODEL_PATH.exists():
            # YOLO 는 파일 경로가 필요 → 임시 파일 최소 사용
            import tempfile
            print("  YOLO 모델로 분류 시도...")
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
                tmp_path = tf.name
            try:
                cv2.imwrite(tmp_path, mask_gray)
                shape = _classify_with_yolo(Path(tmp_path))
            finally:
                try:
                    import os as _os
                    _os.unlink(tmp_path)
                except Exception:
                    pass
            confidence = None
        else:
            prefix = f"[{debug_label}] " if debug_label else ""
            print(f"  {prefix}분류 모델 없음 → generic accessory로 처리")
            shape = "unknown"
            confidence = 0.0

    prompt = SHAPE_TO_PROMPT.get(shape, SHAPE_TO_PROMPT["unknown"])
    return shape, prompt, confidence


def _apply_low_confidence_generic_fallback(
    shape: str,
    prompt: str,
    confidence: float | None,
    debug_label: str | None = None,
    threshold: float = LOW_CONFIDENCE_GENERIC_PROMPT_THRESHOLD,
) -> tuple[str, str, float | None]:
    if confidence is None or confidence >= threshold:
        return shape, prompt, confidence

    prefix = f"[{debug_label}] " if debug_label else ""
    print(f"  {prefix}최종 신뢰도 낮음({confidence:.2f}) → unknown/generic accessory 사용")
    return "unknown", SHAPE_TO_PROMPT["unknown"], confidence


def classify_from_array(
    mask_gray: np.ndarray,
    debug_label: str | None = None,
    filled_mask_gray: np.ndarray | None = None,
    low_confidence_threshold: float = LOW_CONFIDENCE_FILLED_RETRY_THRESHOLD,
) -> tuple[str, str, float | None]:
    """
    numpy array(그레이스케일)에서 직접 shape을 분류한다. 디스크 I/O 없음.

    라인 마스크를 우선 사용하고, 신뢰도가 낮을 때만 filled mask로 재분류한다.

    Returns:
        (shape_name, inpaint_prompt, clip_top1_confidence)
    """
    shape, prompt, confidence = _classify_from_array_once(mask_gray, debug_label=debug_label)

    if filled_mask_gray is None or confidence is None or confidence >= low_confidence_threshold:
        return shape, prompt, confidence

    prefix = f"[{debug_label}] " if debug_label else ""
    print(f"  {prefix}라인 분류 신뢰도 낮음({confidence:.2f}) → filled mask 재분류")

    filled_label = f"{debug_label}:filled" if debug_label else "filled"
    filled_shape, filled_prompt, filled_confidence = _classify_from_array_once(
        filled_mask_gray,
        debug_label=filled_label,
    )

    if filled_confidence is not None and filled_confidence >= low_confidence_threshold and filled_confidence > confidence:
        print(f"  {prefix}filled 결과 채택({filled_shape}, {filled_confidence:.2f})")
        return _apply_low_confidence_generic_fallback(
            filled_shape,
            filled_prompt,
            filled_confidence,
            debug_label=debug_label,
        )

    print(f"  {prefix}라인 결과 유지({shape}, {confidence:.2f})")
    return _apply_low_confidence_generic_fallback(shape, prompt, confidence, debug_label=debug_label)
