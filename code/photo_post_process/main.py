import io
import math
from pathlib import Path
from PIL import Image, ImageFilter
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

def _get_bg_remover():
    global _bg_remover
    if _bg_remover is None:
        _bg_remover = HiInterface(
            object_type="hairs-like",
            batch_size_seg=1,
            batch_size_matting=1,
            device="cpu",
            seg_mask_size=640,
            matting_mask_size=2048,
        )
    return _bg_remover

# ──────────────────────────────────────────────────────────────
# 스타일 목록 — 마스크 영역에 채울 스타일별 프롬프트
# ──────────────────────────────────────────────────────────────
STYLES = {
    "1": {
        "name": "사실적 (Photorealistic)",
        "prompt": "A photorealistic object seamlessly blended with the surrounding scene, high quality, 8k, natural lighting",
    },
    "2": {
        "name": "수채화 (Watercolor)",
        "prompt": "Watercolor painting style, soft brushstrokes, pastel tones, artistic, seamlessly blended with the scene",
    },
    "3": {
        "name": "애니메이션 (Anime)",
        "prompt": "Anime style illustration, vibrant colors, clean lines, cel-shading, seamlessly blended with the scene",
    },
    "4": {
        "name": "유화 (Oil Painting)",
        "prompt": "Oil painting style, rich textures, impasto technique, classical art, seamlessly blended with the scene",
    },
    "5": {
        "name": "미니멀 (Minimalist)",
        "prompt": "Minimalist design, clean lines, flat colors, modern aesthetic, seamlessly blended with the scene",
    },
}


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

    mask_path = INPUT_DIR / f"{img_path.stem}_mask.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"마스크 파일을 찾을 수 없습니다: {mask_path}")
    return img_path, mask_path


def print_menu(title: str, options: dict) -> None:
    print(f"\n{'─'*44}")
    print(f"  {title}")
    print(f"{'─'*44}")
    for key, val in options.items():
        label = val["name"] if isinstance(val, dict) else val
        print(f"  {key:>2}. {label}")
    print(f"{'─'*44}")


def select_from_menu(title: str, options: dict) -> str:
    print_menu(title, options)
    while True:
        choice = input("번호를 입력하세요: ").strip()
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
    shape, prompt = classify_shape(mask_path)
    print(f"  감지된 모양: {shape}")

    # 분류 결과 확인 단계 — 오분류 시 직접 수정 가능
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
    """[Step 2] carvekit으로 로컬 배경 제거 (API 토큰 불필요)."""
    print("[Step 2] 배경 제거 중 (로컬 carvekit)...")
    with Image.open(io.BytesIO(image_bytes)).convert("RGB") as img:
        result = _get_bg_remover()([img])[0]  # RGBA 반환
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    print("  완료.")
    return buf.getvalue()


def step3_composite(person_bytes: bytes, bg_path: Path, ref_size: tuple) -> bytes:
    """[Step 3] 배경 이미지 위에 인물(투명 PNG)을 합성."""
    print(f"[Step 3] 배경 합성 중... ({bg_path.name})")
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
mode_key = select_from_menu("모드 선택", {
    "1": {"name": "스티커 검색  (stickers/ 폴더 사용, 토큰 무료)"},
    "2": {"name": "AI 생성      (Stability API inpainting, 토큰 소모)"},
})
use_retrieval = (mode_key == "1")

# ── 스타일 선택 (생성 모드만) ──
if use_retrieval:
    selected_style = {"name": "검색 모드", "prompt": ""}
    style_key = "R"
else:
    style_key = select_from_menu("스타일 선택", STYLES)
    selected_style = STYLES[style_key]

# ── 배경 선택 ──
bg_key        = select_from_menu("배경 선택", BACKGROUNDS)
selected_bg   = BACKGROUNDS[bg_key]
use_custom_bg = (bg_key != "0")

# ── Shape 분류 → 프롬프트 ──
sketch_shape, sketch_prompt = step0_detect_sketch(mask_image_path)
combined_prompt = f"[{sketch_shape.upper()}] {sketch_prompt}. Style: {selected_style['prompt']}"

bg_label          = selected_bg["name"]
output_filename   = f"{input_image_path.stem}_{bg_label}_style{style_key}.jpeg"
output_image_path = OUT_DIR / output_filename

print(f"\n  입력       : {input_image_path.name}")
print(f"  마스크     : {mask_image_path.name}")
print(f"  모드       : {'검색' if use_retrieval else 'AI 생성'}")
print(f"  스케치 모양: {sketch_shape}")
print(f"  스타일     : {selected_style['name']}")
print(f"  배경       : {bg_label}")
print(f"  출력       : output/{output_filename}\n")

# ── 이미지 / 마스크 준비 ──
image_bytes = resize_to_api_limit(input_image_path)
mask_bytes  = resize_mask_to_match(mask_image_path, image_bytes)

with Image.open(io.BytesIO(image_bytes)) as _ref:
    ref_size = _ref.size

# ──────────────────────────────────────────────────────────────
# Step 1: 모드에 따라 분기
#   - 검색 모드: stickers/ 폴더에서 PNG 가져와 합성
#   - 생성 모드: Stability API로 인페인팅
# ──────────────────────────────────────────────────────────────
if use_retrieval:
    from aircanvas_retrieval import step1_retrieve_sticker
    inpainted_bytes = step1_retrieve_sticker(image_bytes, mask_image_path, sketch_shape)
else:
    from aircanvas_inpainting import step1_inpaint
    inpainted_bytes = step1_inpaint(image_bytes, mask_bytes, combined_prompt)

# ──────────────────────────────────────────────────────────────
# Step 2~3: 배경 교체 (공통)
# ──────────────────────────────────────────────────────────────
if use_custom_bg:
    person_bytes = step2_remove_background(inpainted_bytes)
    final_bytes  = step3_composite(person_bytes, selected_bg["path"], ref_size)
else:
    with Image.open(io.BytesIO(inpainted_bytes)) as img:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=95)
        final_bytes = buf.getvalue()

# ── 결과 저장 ──
with open(output_image_path, "wb") as f:
    f.write(final_bytes)
print(f"\n이미지 생성 성공! → output/{output_filename} 저장됨")
