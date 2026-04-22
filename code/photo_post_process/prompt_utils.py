from collections import Counter
from pathlib import Path

import colorsys
import numpy as np
from PIL import Image, ImageChops, ImageFilter


ColorWeights = list[tuple[str, float]]
MIN_COLOR_WEIGHT = 0.1
MAX_COLOR_COUNT = 5

ACCESSORY_SHAPES = {"cat_ears", "rabbit_ears", "crown", "hat", "bow", "sunglasses"}
ICON_SHAPES = {"heart", "star", "cloud", "moon", "rainbow", "lightning", "fire", "diamond", "arrow", "music_note", "speech_bubble", "bomb"}
NATURE_SHAPES = {"flower", "leaf", "butterfly"}

PROMPT_COLOR_NAME_MAP = {
    "light yellow": "pale yellow",
    "light green": "mint green",
    "light blue": "sky blue",
    "light pink": "soft pink",
    "light purple": "lavender",
    "dark blue": "navy blue",
    "dark green": "forest green",
    "dark red": "deep red",
    "light gray": "silver gray",
    "gray": "neutral gray",
    "cyan": "aqua blue",
}


def _map_color_name_for_prompt(name: str) -> str:
    """Stability prompt에서 더 잘 해석될 색상 표현으로 변환한다."""
    return PROMPT_COLOR_NAME_MAP.get(name, name)


# def _describe_color_emphasis(color_weights: ColorWeights) -> str:
#     """가중치 목록을 모델이 해석하기 쉬운 자연어 색상 설명으로 바꾼다."""
#     if not color_weights:
#         return ""

#     descriptions: list[str] = []
#     for index, (name, weight) in enumerate(color_weights):
#         prompt_name = _map_color_name_for_prompt(name)
#         if index == 0 or weight >= 0.75:
#             qualifier = "primarily"
#         elif weight >= 0.4:
#             qualifier = "mostly"
#         elif weight >= 0.2:
#             qualifier = "with secondary"
#         else:
#             qualifier = "with small"

#         if qualifier in {"primarily", "mostly"}:
#             descriptions.append(f"{qualifier} {prompt_name}")
#         elif qualifier == "with secondary":
#             descriptions.append(f"secondary {prompt_name} accents")
#         else:
#             descriptions.append(f"small {prompt_name} details")

#     return ", ".join(descriptions)


# def _build_shape_guidance(shape: str, subject: str) -> str:
#     """모양별 추가 제약을 반환한다."""
#     if shape == "rabbit_ears":
#         return (
#             "Create exactly one clean pair of bunny ears as a headband accessory above the head. "
#             "Make the two ears upright, symmetrical, and clearly readable. Do not create extra ears or a new character."
#         )
#     if shape == "cat_ears":
#         return (
#             "Create exactly one clean pair of cat ears as a headband accessory above the head. "
#             "Make the ears pointed, symmetrical, and clearly readable. Do not create extra ears or a new character."
#         )
#     if shape == "sunglasses":
#         return (
#             "Create exactly one clean pair of sunglasses worn naturally on the face. "
#             "Keep them symmetrical, readable, and properly aligned without changing the face."
#         )
#     if shape == "crown":
#         return (
#             "Create exactly one clean crown accessory placed naturally on top of the head. "
#             "Keep it centered, symmetrical, and clearly readable without creating a new character."
#         )
#     if shape == "hat":
#         return (
#             "Create exactly one clean hat accessory worn naturally on the head. "
#             "Keep the silhouette simple, stable, and clearly readable without changing the person."
#         )
#     if shape == "bow":
#         return (
#             "Create exactly one clean ribbon bow accessory with two balanced loops and a clear center knot. "
#             "Keep it symmetrical and neatly attached to the person."
#         )
#     if shape == "mustache":
#         return (
#             "Create exactly one clean mustache with two balanced sides and a readable center split. "
#             "Keep it neat, symmetrical, and attached naturally without changing the face shape."
#         )
#     if shape == "whiskers":
#         return (
#             "Create clean cat whiskers as a simple facial accessory with thin balanced lines on both sides. "
#             "Do not create fur, a cat face, or a new character."
#         )
#     if shape in ACCESSORY_SHAPES:
#         return (
#             f"Create {subject} as a wearable accessory placed naturally on the person. "
#             "Keep the shape clean and do not create extra accessories or a new character."
#         )
#     if shape in ICON_SHAPES:
#         return (
#             f"Create only one clean {subject} with a simple, readable, well-defined silhouette inside the masked area. "
#             "Do not distort the shape and do not add extra decorative objects or characters."
#         )
#     if shape in NATURE_SHAPES:
#         return (
#             f"Create one clean {subject} with a clear silhouette and natural proportions inside the masked area. "
#             "Keep the shape readable and do not add unrelated objects or characters."
#         )
#     return (
#         f"Create only {subject} inside the masked area with a clean, readable silhouette. "
#         "Do not add unrelated objects or characters."
#     )


def build_inpaint_prompt(shape: str, subject_prompt: str, color_weights: ColorWeights) -> str:
    """shape 분류 결과와 스케치 색상을 바탕으로 인페인팅 프롬프트를 생성."""
    subject = subject_prompt.strip() if subject_prompt else shape.replace("_", " ")
    subject = subject.rstrip(".")
    color_clause = ""
    if color_weights:
        # color_emphasis = _describe_color_emphasis(color_weights)
        # color_clause = (
        #     "Use the original sketch colors, "
        #     f"{color_emphasis}. Weighted color hint: {format_color_weights_for_prompt(color_weights)}."
        # )
        color_clause = (
            f"{format_color_weights_for_prompt(color_weights)}"
        )
    # shape_guidance = _build_shape_guidance(shape, subject)
    return (
        f"Add a {color_clause} {subject} naturally in the masked area.\n"
        # f"Match the photo's lighting, angle, and scale.\n{shape_guidance}\n{color_clause}\n"
        f"Match the photo's lighting, angle, and scale.\n"
        f"Keep the person's face, hair, body, and clothes unchanged.\n"
        f"Do not generate person, face, or body parts in the masked area.\n"
        f"Focus on the {subject} and make it look like a natural part of the photo."
    )


def format_color_weights_for_prompt(color_weights: ColorWeights) -> str:
    """프롬프트에 넣기 위한 색상 가중치 문자열을 반환한다."""
    return " and ".join(f"({_map_color_name_for_prompt(name)}:{weight:.1f})" for name, weight in color_weights)


def format_color_weights_for_display(color_weights: ColorWeights) -> str:
    """CLI 출력용 색상 가중치 문자열을 반환한다."""
    if not color_weights:
        return ""
    return ", ".join(f"({name}:{weight:.1f})" for name, weight in color_weights)


def _rgb_to_color_name(rgb: tuple[int, int, int]) -> str:
    """RGB 값을 사람이 읽기 쉬운 간단한 색상명으로 변환한다."""
    red, green, blue = [channel / 255 for channel in rgb]
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    hue *= 360

    if value < 0.16:
        return "black"
    if saturation < 0.12:
        if value > 0.9:
            return "white"
        if value > 0.65:
            return "light gray"
        return "gray"

    prefix = ""
    if value > 0.82 and saturation < 0.45:
        prefix = "light "
    elif value < 0.4:
        prefix = "dark "

    if hue < 15 or hue >= 345:
        base = "red"
    elif hue < 40:
        base = "orange"
    elif hue < 70:
        base = "yellow"
    elif hue < 160:
        base = "green"
    elif hue < 200:
        base = "cyan"
    elif hue < 255:
        base = "blue"
    elif hue < 300:
        base = "purple"
    else:
        base = "pink"

    return f"{prefix}{base}".strip()


def _build_raw_color_weights(color_counts: list[tuple[str, int]]) -> ColorWeights:
    """최빈 색 대비 상대 강조값(raw emphasis)을 계산한다."""
    if not color_counts:
        return []

    max_count = color_counts[0][1]
    if max_count <= 0:
        return []

    raw_weights = [(name, count / max_count) for name, count in color_counts]
    filtered = [(name, weight) for name, weight in raw_weights if weight >= MIN_COLOR_WEIGHT]
    if not filtered and raw_weights:
        filtered = [raw_weights[0]]
    return filtered


def _collect_color_weights(image: Image.Image, region_mask: Image.Image, max_colors: int = MAX_COLOR_COUNT) -> ColorWeights:
    image_np = np.asarray(image, dtype=np.uint8)
    mask_np = np.asarray(region_mask, dtype=np.uint8)

    ys, xs = np.nonzero(mask_np)
    if len(xs) == 0:
        return []

    region_colors: list[tuple[int, int, int]] = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        red, green, blue = image_np[y, x].tolist()
        hue, saturation, value = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
        if saturation < 0.18 or value < 0.18:
            continue
        quantized = tuple(min(255, (channel // 32) * 32 + 16) for channel in (red, green, blue))
        region_colors.append(quantized)

    if not region_colors:
        return []

    named_colors = Counter(_rgb_to_color_name(rgb) for rgb in region_colors)
    most_common = named_colors.most_common(max_colors)
    if not most_common:
        return []

    return _build_raw_color_weights(most_common)


def detect_sketch_color_weights_from_images(image: Image.Image, mask: Image.Image) -> ColorWeights:
    """이미지와 마스크 이미지 객체에서 스케치 경계선 색상을 추정한다."""
    image = image.convert("RGB")
    mask = mask.convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)

    binary_mask = mask.point(lambda value: 255 if value > 127 else 0)
    eroded_mask = binary_mask.filter(ImageFilter.MinFilter(size=3))
    edge_mask = ImageChops.subtract(binary_mask, eroded_mask)

    return _collect_color_weights(image, edge_mask)


def detect_sketch_color_hint_from_images(image: Image.Image, mask: Image.Image) -> str:
    """이미지와 마스크 이미지 객체에서 스케치 색상 요약 문자열을 반환한다."""
    return format_color_weights_for_display(detect_sketch_color_weights_from_images(image, mask))


def detect_sketch_color_weights(image_path: Path, mask_path: Path) -> ColorWeights:
    """원본 사진의 마스크 경계선 픽셀에서 스케치 색상 가중치를 추정한다."""
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        return detect_sketch_color_weights_from_images(image, mask)


def detect_sketch_color_hint(image_path: Path, mask_path: Path) -> str:
    """원본 사진의 마스크 경계선 픽셀에서 스케치 색상 요약 문자열을 반환한다."""
    return format_color_weights_for_display(detect_sketch_color_weights(image_path, mask_path))


def detect_sketch_color_weights_from_arrays(image_rgb: np.ndarray, mask_gray: np.ndarray) -> ColorWeights:
    """RGB ndarray와 grayscale mask ndarray에서 스케치 색상 가중치를 추정한다."""
    image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
    mask = Image.fromarray(mask_gray.astype(np.uint8), mode="L")
    return detect_sketch_color_weights_from_images(image, mask)


def detect_sketch_color_hint_from_arrays(image_rgb: np.ndarray, mask_gray: np.ndarray) -> str:
    """RGB ndarray와 grayscale mask ndarray에서 스케치 색상 요약 문자열을 반환한다."""
    return format_color_weights_for_display(detect_sketch_color_weights_from_arrays(image_rgb, mask_gray))