import cv2
import numpy as np
import config


def _load_cells_from_mask(mask_path, min_area=3000):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    _, binary = cv2.threshold(m, 50, 255, cv2.THRESH_BINARY_INV)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cells = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area:
            cells.append((x, y, x + w, y + h))
    cells.sort(key=lambda c: (c[1], c[0]))
    return cells if cells else None


# ── 이미지 로드 ───────────────────────────────────────────────────
bg_raw        = cv2.imread(config.BG_IMAGE_PATH)
bg_result_raw = cv2.imread(config.BG_RESULT_IMAGE_PATH)
intro_raw     = cv2.imread(config.INTRO_IMAGE_PATH)
frame_raw     = cv2.imread(config.FRAME_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

if bg_raw is None:
    print(f"[경고] 배경 이미지 없음: {config.BG_IMAGE_PATH} → 단색 배경 사용")
if bg_result_raw is None:
    print(f"[경고] 결과 배경 이미지 없음: {config.BG_RESULT_IMAGE_PATH}")
if intro_raw is None:
    print(f"[경고] 인트로 이미지 없음: {config.INTRO_IMAGE_PATH}")
if frame_raw is None:
    print(f"[경고] 프레임 이미지 없음: {config.FRAME_IMAGE_PATH}")
elif frame_raw.shape[2] == 3:
    bgra = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2BGRA)
    white_mask = np.all(frame_raw >= 240, axis=2)
    bgra[white_mask, 3] = 0
    frame_raw = bgra
    print("[프레임] 알파채널 없음 → 흰색 투명 처리")

# ── 테마 그리드 ───────────────────────────────────────────────────
source_theme_img = cv2.imread(config.SOURCE_THEME_PATH)
if source_theme_img is None:
    print(f"[경고] source_theme.png 로드 실패: {config.SOURCE_THEME_PATH}")

_theme_cells = _load_cells_from_mask(config.SOURCE_THEME_MASK_PATH)
SOURCE_THEME_CELLS = _theme_cells if _theme_cells else config.SOURCE_THEME_CELLS_FALLBACK
print(f"[테마 그리드] {'마스크' if _theme_cells else '폴백'} 기준 {len(SOURCE_THEME_CELLS)}개 셀")

# ── 배경 그리드 ───────────────────────────────────────────────────
source_bg_img = cv2.imread(config.SOURCE_BG_PATH)
if source_bg_img is None:
    print(f"[경고] source_background.png 로드 실패: {config.SOURCE_BG_PATH}")
else:
    h, w = source_bg_img.shape[:2]
    print(f"[배경 그리드] {w}x{h}, {len(config.SOURCE_BG_CELLS)}개 칸")
