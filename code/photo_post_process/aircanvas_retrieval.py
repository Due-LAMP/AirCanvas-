import io
import numpy as np
from pathlib import Path
from PIL import Image

BASE_DIR    = Path(__file__).parent
STICKER_DIR = BASE_DIR / "stickers"


def step1_retrieve_sticker(image_bytes: bytes, mask_path, shape: str) -> bytes:
    """[Step 1 - 검색 모드] stickers/ 폴더에서 shape에 맞는 스티커를 찾아 마스크 위치에 합성."""
    print(f"\n[Step 1] 스티커 검색 및 합성 중... (shape: {shape})")

    # stickers/ 에서 shape명 매칭 파일 탐색 (heart.png, heart.webp 등)
    exts = [".png", ".webp", ".jpg", ".jpeg"]
    sticker_path = None
    for ext in exts:
        candidate = STICKER_DIR / f"{shape}{ext}"
        if candidate.exists():
            sticker_path = candidate
            break

    if sticker_path is None:
        raise FileNotFoundError(
            f"stickers/{shape}.png 를 찾을 수 없습니다. "
            f"stickers/ 폴더에 {shape}.png 파일을 추가하세요."
        )
    print(f"  스티커 파일: {sticker_path.name}")

    # 마스크에서 붙일 위치(bounding box) 계산
    with Image.open(mask_path) as mask_img:
        mask_gray = mask_img.convert("L")

    mask_arr = np.array(mask_gray)
    rows = np.any(mask_arr > 127, axis=1)
    cols = np.any(mask_arr > 127, axis=0)
    if not rows.any():
        raise ValueError("마스크에 흰색 영역이 없습니다.")
    y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    bbox_w, bbox_h = x1 - x0, y1 - y0

    # 원본 이미지 위에 스티커 합성
    with Image.open(io.BytesIO(image_bytes)).convert("RGBA") as base:
        # 마스크 크기가 이미 base와 맞춰져 있어야 함
        with Image.open(sticker_path).convert("RGBA") as sticker:
            sticker_resized = sticker.resize((bbox_w, bbox_h), Image.LANCZOS)

        result = base.copy()
        result.paste(sticker_resized, (x0, y0), mask=sticker_resized)

        buf = io.BytesIO()
        result.convert("RGB").save(buf, format="PNG")
        print(f"  합성 위치: ({x0},{y0}) ~ ({x1},{y1})  크기: {bbox_w}x{bbox_h}")
        print("  완료.")
        return buf.getvalue()

