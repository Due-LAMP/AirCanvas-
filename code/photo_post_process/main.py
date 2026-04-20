import io
import math
import os
import colorsys
from collections import Counter
from pathlib import Path
from PIL import Image, ImageChops, ImageFilter
from carvekit.api.high import HiInterface
from shape_classifier import classify as classify_shape

# ──────────────────────────────────────────────────────────────
# 디렉토리 경로
# ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
BG_DIR    = BASE_DIR / "background"
OUT_DIR   = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────
# carvekit 배경 제거기 (최초 실행 시 모델 자동 다운로드)
# ──────────────────────────────────────────────────────────────
_bg_remover = None

BG_REMOVER_OBJECT_TYPE = os.getenv("CARVEKIT_OBJECT_TYPE", "object")
BG_MASK_EXPAND_PX = max(0, int(os.getenv("BG_MASK_EXPAND_PX", "4")))

def _get_bg_remover():
    global _bg_remover
    if _bg_remover is None:
        _bg_remover = HiInterface(
            object_type=BG_REMOVER_OBJECT_TYPE,
            batch_size_seg=1,
            batch_size_matting=1,
            device="cpu",
            seg_mask_size=640,
            matting_mask_size=2048,
        )
    return _bg_remover

# ──────────────────────────────────────────────────────────────
# 스타일 목록 — Stability style_preset 선택용
# ──────────────────────────────────────────────────────────────
STYLE_PRESET_OPTIONS = [
    ("3d-model", "3D Model (3D 입체)"),
    ("analog-film", "Analog Film (아날로그 필름)"),
    ("anime", "Anime (애니메이션)"),
    ("cinematic", "Cinematic (영화)"),
    ("comic-book", "Comic Book (만화책)"),
    ("digital-art", "Digital Art (디지털 일러스트)"),
    ("enhance", "Enhance (선명 강화)"),
    ("fantasy-art", "Fantasy Art (판타지 일러스트)"),
    ("isometric", "Isometric (등각 투시도)"),
    ("line-art", "Line Art (선화)"),
    ("low-poly", "Low Poly (단순 면 3D 그래픽)"),
    ("modeling-compound", "Modeling Compound (점토 조형물)"),
    ("neon-punk", "Neon Punk (네온 사이버펑크)"),
    ("origami", "Origami (종이접기)"),
    ("photographic", "Photographic (실사 사진)"),
    ("pixel-art", "Pixel Art (픽셀 아트)"),
    ("tile-texture", "Tile Texture (타일 질감)"),
]

STYLES = {
    str(index): {"name": display_name, "preset": preset}
    for index, (preset, display_name) in enumerate(STYLE_PRESET_OPTIONS, start=1)
}


def build_inpaint_prompt(shape: str, subject_prompt: str, color_hint: str) -> str:
    """shape 분류 결과와 스케치 색상을 바탕으로 인페인팅 프롬프트를 생성."""
    subject = subject_prompt.strip() if subject_prompt else shape.replace("_", " ")
    subject = subject.rstrip(".")
    color_clause = ""
    if color_hint:
        color_clause = f" Use the original sketch colors, especially {color_hint}."
    return (
        f"Add {subject} naturally in the masked area. "
        f"Match the person's pose, lighting, angle, and scale.{color_clause}."
        "Keep the person's face, hair, body, and clothes unchanged." \
        f"Do not generate person, face, or body parts in the masked area. Focus on the {subject} and make it look like a natural part of the photo."
    )


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


def detect_sketch_color_hint(image_path: Path, mask_path: Path) -> str:
    """원본 사진의 마스크 경계선 픽셀에서 스케치 색상을 추정한다."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    with Image.open(mask_path) as mask:
        mask = mask.convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)

    binary_mask = mask.point(lambda value: 255 if value > 127 else 0)
    eroded_mask = binary_mask.filter(ImageFilter.MinFilter(size=3))
    edge_mask = ImageChops.subtract(binary_mask, eroded_mask)

    def collect_colors(region_mask: Image.Image) -> list[str]:
        region_colors: list[tuple[int, int, int]] = []
        for rgb, mask_value in zip(image.getdata(), region_mask.getdata()):
            if mask_value == 0:
                continue
            red, green, blue = rgb
            hue, saturation, value = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
            if saturation < 0.18 or value < 0.18:
                continue
            quantized = tuple(min(255, (channel // 32) * 32 + 16) for channel in rgb)
            region_colors.append(quantized)

        if not region_colors:
            return []

        named_colors = Counter(_rgb_to_color_name(rgb) for rgb in region_colors)
        ordered_names = [name for name, _count in named_colors.most_common(3)]
        unique_names: list[str] = []
        for name in ordered_names:
            if name not in unique_names:
                unique_names.append(name)
        return unique_names[:2]

    color_names = collect_colors(edge_mask)

    if not color_names:
        return ""
    if len(color_names) == 1:
        return color_names[0]
    return f"{color_names[0]} and {color_names[1]}"


def load_backgrounds() -> dict:
    """background/ 디렉토리 이미지를 자동 탐색. '0'번은 항상 'Original'."""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(p for p in BG_DIR.iterdir() if p.suffix.lower() in exts)
    menu = {"0": {"name": "Original", "path": None}}
    for i, p in enumerate(files):
        menu[str(i + 1)] = {"name": p.stem, "path": p}
    return menu


def find_input_pair() -> tuple:
    """input/ 디렉토리에서 원본 이미지와 마스크 파일을 반환."""
    exts = {".jpg", ".jpeg", ".png"}
    images = [
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in exts and "_mask" not in p.stem
    ]
    if not images:
        raise FileNotFoundError(f"input/ 디렉토리에 이미지가 없습니다: {INPUT_DIR}")

    if len(images) == 1:
        img_path = images[0]
    else:
        menu = {str(i + 1): {"name": p.name, "path": p} for i, p in enumerate(images)}
        key = select_from_menu("입력 이미지 선택", menu)
        img_path = menu[key]["path"]

    mask_candidates = [INPUT_DIR / f"{img_path.stem}_mask.png"]

    stem_parts = img_path.stem.split("_")
    if stem_parts and stem_parts[-1].isdigit():
        mask_candidates.append(INPUT_DIR / f"mask_{stem_parts[-1]}.png")

    mask_path = next((candidate for candidate in mask_candidates if candidate.exists()), None)
    if mask_path is None:
        tried = ", ".join(str(path.name) for path in mask_candidates)
        raise FileNotFoundError(f"마스크 파일을 찾을 수 없습니다. 확인한 이름: {tried}")
    return img_path, mask_path


def print_menu(title: str, options: dict) -> None:
    print(f"\n{'─'*44}")
    print(f"  {title}")
    print(f"{'─'*44}")
    for key, val in options.items():
        label = val["name"] if isinstance(val, dict) else val
        print(f"  {key:>2}. {label}")
    print(f"{'─'*44}")


def select_from_menu(title: str, options: dict, default_choice: str | None = None) -> str:
    print_menu(title, options)
    prompt = "번호를 입력하세요: "
    if default_choice is not None:
        prompt = f"번호를 입력하세요 [기본값: {default_choice}]: "

    while True:
        choice = input(prompt).strip()
        if not choice and default_choice in options:
            return default_choice
        if choice in options:
            return choice
        print(f"  ※ 올바른 번호를 입력하세요 ({', '.join(options.keys())})")


def resize_to_api_limit(path, max_pixels: int = 1024 * 1024) -> bytes:
    """이미지를 API 최대 픽셀 수(1,048,576) 이하로 리사이즈하여 bytes 반환."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if w * h > max_pixels:
            scale = math.sqrt(max_pixels / (w * h))
            new_w = (int(w * scale) // 64) * 64
            new_h = (int(h * scale) // 64) * 64
            img = img.resize((new_w, new_h), Image.LANCZOS)
            print(f"  리사이즈: {w}x{h} → {new_w}x{new_h}  ({Path(path).name})")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def resize_mask_to_match(mask_path, image_bytes: bytes) -> bytes:
    """마스크를 원본 이미지와 동일한 크기로 맞춘 뒤 bytes 반환."""
    with Image.open(io.BytesIO(image_bytes)) as ref:
        ref_size = ref.size
    with Image.open(mask_path) as mask:
        mask = mask.convert("L")  # 흰색=채울 영역, 검정=유지 영역
        if mask.size != ref_size:
            mask = mask.resize(ref_size, Image.LANCZOS)
            print(f"  마스크 리사이즈: {mask.size} → {ref_size}")
        buf = io.BytesIO()
        mask.save(buf, format="PNG")
        return buf.getvalue()


def step0_detect_sketch(mask_path) -> tuple:
    """[Step 0] 마스크 shape 분류 (CLIP 기반, YOLO/OpenCV 폴백)."""
    print("[Step 0] 스케치 모양 분석 중...")
    shape, prompt, confidence = classify_shape(mask_path)
    print(f"  감지된 모양: {shape}")
    if confidence is not None:
        print(f"  Top-1 confidence: {confidence:.2f}")

    # CLIP 신뢰도가 낮을 때만 사용자 확인 단계 진행
    if confidence is not None and confidence <= 0.4:
        print(f"  사용 가능한 모양: {', '.join(__import__('shape_classifier').SHAPE_TO_PROMPT.keys())}")
        correction = input("  수정할 모양을 입력하세요 (맞으면 Enter 스킵): ").strip().lower()
        if correction:
            from shape_classifier import SHAPE_TO_PROMPT
            if correction in SHAPE_TO_PROMPT:
                shape = correction
                prompt = SHAPE_TO_PROMPT[shape]
                print(f"  수정됨: {shape}")
            else:
                shape = correction
                prompt = correction
                print(f"  커스텀 입력: {prompt}")

    return shape, prompt


def step2_remove_background(image_bytes: bytes) -> bytes:
    """인물 이미지를 carvekit으로 배경 제거한다."""
    print(f"[배경 교체] 배경 제거 중 (로컬 carvekit, object_type={BG_REMOVER_OBJECT_TYPE})...")
    with Image.open(io.BytesIO(image_bytes)).convert("RGB") as img:
        result = _get_bg_remover()([img])[0]  # RGBA 반환

    if BG_MASK_EXPAND_PX > 0:
        r, g, b, alpha = result.split()
        filter_size = BG_MASK_EXPAND_PX * 2 + 1
        alpha = alpha.filter(ImageFilter.MaxFilter(size=filter_size))
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=2))
        result = Image.merge("RGBA", (r, g, b, alpha))

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    print("  완료.")
    return buf.getvalue()


def step3_composite(person_bytes: bytes, bg_path: Path, ref_size: tuple) -> bytes:
    """배경 이미지 위에 인물(투명 PNG)을 합성."""
    print(f"[배경 교체] 배경 합성 중... ({bg_path.name})")
    with Image.open(bg_path) as bg:
        bg = bg.convert("RGBA").resize(ref_size, Image.LANCZOS)
    with Image.open(io.BytesIO(person_bytes)) as person:
        person = person.convert("RGBA").resize(ref_size, Image.LANCZOS)

    # 알파 채널 가장자리 페더링 — 경계를 부드럽게 블렌딩
    r, g, b, alpha = person.split()
    alpha_feathered = alpha.filter(ImageFilter.GaussianBlur(radius=3))
    person = Image.merge("RGBA", (r, g, b, alpha_feathered))

    bg.paste(person, (0, 0), mask=person)
    buf = io.BytesIO()
    bg.convert("RGB").save(buf, format="JPEG", quality=95)
    print("  완료.")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════════
BACKGROUNDS = load_backgrounds()
input_image_path, mask_image_path = find_input_pair()

# ── 모드 선택 ──
mode_key = select_from_menu(
    "모드 선택",
    {
        "1": {"name": "스티커 검색  (stickers/ 폴더 사용, 무료)"},
        "2": {"name": "AI 생성      (Stability AI inpainting, 유료)"},
    },
    default_choice="2",
)
use_retrieval = (mode_key == "1")

# ── 스타일 선택 (생성 모드만) ──
if use_retrieval:
    selected_style = {"name": "검색 모드", "preset": None}
    style_key = "R"
else:
    style_key = select_from_menu("스타일 선택", STYLES)
    selected_style = STYLES[style_key]

# ── 배경 선택 ──
bg_key        = select_from_menu("배경 선택", BACKGROUNDS)
selected_bg   = BACKGROUNDS[bg_key]
use_custom_bg = (bg_key != "0")

# ── Shape 분류 → 프롬프트 ──
sketch_shape, sketch_subject_prompt = step0_detect_sketch(mask_image_path)
sketch_color_hint = detect_sketch_color_hint(input_image_path, mask_image_path)
combined_prompt = build_inpaint_prompt(sketch_shape, sketch_subject_prompt, sketch_color_hint)
print(combined_prompt)

bg_label          = selected_bg["name"]
style_suffix      = selected_style["preset"] or "retrieval"
output_filename   = f"{input_image_path.stem}_{bg_label}_style-{style_suffix}.jpeg"
output_image_path = OUT_DIR / output_filename

print(f"\n{'='*40}")
print(f"  입력 이미지\t: {input_image_path.name}")
print(f"  마스크 이미지\t: {mask_image_path.name}")
print(f"  모드\t\t: {'검색' if use_retrieval else 'AI 생성'}")
print(f"  스케치 모양\t: {sketch_shape}")
print(f"  색상 힌트\t: {sketch_color_hint or '자동 추출 실패'}")
print(f"  스타일\t: {selected_style['name']}")
print(f"  프롬프트\t: {combined_prompt}")
print(f"  배경\t\t: {bg_label}")
print(f"  출력\t\t: output/{output_filename}\n")

# ── 이미지 / 마스크 준비 ──
image_bytes = resize_to_api_limit(input_image_path)
mask_bytes  = resize_mask_to_match(mask_image_path, image_bytes)

with Image.open(io.BytesIO(image_bytes)) as _ref:
    ref_size = _ref.size

# ──────────────────────────────────────────────────────────────
# Step 1: 배경 교체를 먼저 수행
#   - 배경 선택 시: 원본 인물에서 배경 제거 후 새 배경 합성
#   - Original 선택 시: 원본 이미지를 그대로 사용
#   생성된 스티커/인페인팅 결과가 배경 제거 단계에서 사라지는 것을 막기 위함
# ──────────────────────────────────────────────────────────────
base_image_bytes = image_bytes
if use_custom_bg:
    person_bytes = step2_remove_background(image_bytes)
    base_image_bytes = step3_composite(person_bytes, selected_bg["path"], ref_size)

# ──────────────────────────────────────────────────────────────
# Step 2: 모드에 따라 분기
#   - 검색 모드: stickers/ 폴더에서 PNG 가져와 합성
#   - 생성 모드: Stability API로 인페인팅
# ──────────────────────────────────────────────────────────────
if use_retrieval:
    from aircanvas_retrieval import step1_retrieve_sticker
    inpainted_bytes = step1_retrieve_sticker(base_image_bytes, mask_image_path, sketch_shape)
else:
    from aircanvas_inpainting import step1_inpaint
    inpainted_bytes = step1_inpaint(
        base_image_bytes,
        mask_bytes,
        combined_prompt,
        selected_style["preset"],
    )

# ──────────────────────────────────────────────────────────────
# Step 3: 결과 저장
# ──────────────────────────────────────────────────────────────
with Image.open(io.BytesIO(inpainted_bytes)) as img:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    final_bytes = buf.getvalue()

# ── 결과 저장 ──
with open(output_image_path, "wb") as f:
    f.write(final_bytes)
print(f"\n이미지 생성 성공! → output/{output_filename} 저장됨")
