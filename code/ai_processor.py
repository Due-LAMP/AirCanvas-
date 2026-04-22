import sys
import os
import cv2
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import config
import assets

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'photo_post_process'))

try:
    from shape_classifier import classify_from_array as _classify_shape_from_array
    SHAPE_CLASSIFIER_AVAILABLE = True
except Exception as _e:
    print(f'[경고] shape_classifier 로드 실패: {_e}', flush=True)
    SHAPE_CLASSIFIER_AVAILABLE = False

try:
    from aircanvas_inpainting import step1_inpaint as _step1_inpaint
    INPAINTING_AVAILABLE = True
except Exception as _e:
    print(f'[경고] aircanvas_inpainting 로드 실패: {_e}', flush=True)
    INPAINTING_AVAILABLE = False

try:
    from prompt_utils import (
        build_inpaint_prompt as _build_inpaint_prompt,
        detect_sketch_color_weights_from_arrays as _detect_sketch_color_weights_from_arrays,
    )
    _PROMPT_UTILS_AVAILABLE = True
except Exception as _e:
    print(f'[경고] prompt_utils 로드 실패: {_e}', flush=True)
    _PROMPT_UTILS_AVAILABLE = False

try:
    import mediapipe as _mp_module
    _SELFIE_SEG_AVAILABLE = True
except Exception as _e:
    print(f'[경고] mediapipe 로드 실패: {_e}', flush=True)
    _SELFIE_SEG_AVAILABLE = False

_selfie_seg_local = threading.local()


def _get_selfie_seg():
    if not _SELFIE_SEG_AVAILABLE:
        return None
    if not hasattr(_selfie_seg_local, 'instance'):
        _selfie_seg_local.instance = _mp_module.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    return _selfie_seg_local.instance


def fill_mask_interior(mask_bin):
    h, w = mask_bin.shape[:2]
    close_k = np.ones((3, 3), np.uint8)
    closed  = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, close_k, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        filled = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        return cv2.bitwise_or(filled, mask_bin)
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:h+1, 1:w+1] = closed
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    outside  = flood[1:h+1, 1:w+1]
    interior = cv2.bitwise_not(outside)
    return cv2.bitwise_or(interior, mask_bin)


def pixelart_inpaint_one(img_bgr, mask_gray, reference_bgr=None, style_preset='pixel-art'):
    import io as _io
    from PIL import Image as _PILImage

    h, w = img_bgr.shape[:2]
    _, mask_bin = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(mask_bin) == 0:
        return img_bgr

    mask_filled = fill_mask_interior(mask_bin)

    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buf_img  = _io.BytesIO()
    _PILImage.fromarray(img_rgb).save(buf_img, format='JPEG', quality=92)
    image_bytes = buf_img.getvalue()

    buf_mask = _io.BytesIO()
    _PILImage.fromarray(mask_filled).save(buf_mask, format='PNG')
    mask_bytes = buf_mask.getvalue()

    shape_name    = 'unknown'
    subject_prompt = 'a decorative accessory'

    if SHAPE_CLASSIFIER_AVAILABLE:
        shape_name, subject_prompt, _conf = _classify_shape_from_array(mask_gray)

    if _PROMPT_UTILS_AVAILABLE:
        color_weights = []
        if reference_bgr is not None:
            reference_rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
            color_weights = _detect_sketch_color_weights_from_arrays(reference_rgb, mask_gray)
        prompt = _build_inpaint_prompt(shape_name, subject_prompt, color_weights)
    else:
        style_label = style_preset.replace('-', ' ')
        prompt = (
            f"Add {subject_prompt} naturally in the masked area.\n"
            f"Match the photo's lighting, angle, and scale.\n"
            f"Keep the person's face, hair, body, and clothes unchanged.\n"
            f"Do not generate person, face, or body parts in the masked area.\n"
            f"Focus on the {subject_prompt} and make it look like a natural part of the photo."
        )

    print(f'[AI] 형태 감지: {shape_name} / 스타일: {style_preset} / 프롬프트: {prompt[:60]}...', flush=True)

    result_bytes = _step1_inpaint(image_bytes, mask_bytes, prompt, style_preset=style_preset)

    arr = np.frombuffer(result_bytes, dtype=np.uint8)
    result_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if result_bgr is None:
        raise RuntimeError('인페인팅 응답 이미지 디코딩 실패')
    return cv2.resize(result_bgr, (w, h))


def remove_bg_composite(img_bgr, bg_bgr):
    seg = _get_selfie_seg()
    if seg is None:
        print('[배경] MediaPipe 없음 → 배경 교체 스킵', flush=True)
        return img_bgr

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    print('[배경] 인물 배경 제거 중 (MediaPipe)...', flush=True)
    result = seg.process(img_rgb)
    mask = (result.segmentation_mask > 0.6).astype(np.uint8) * 255

    # 가장자리 페더링
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask_3 = mask[:, :, np.newaxis].astype(np.float32) / 255.0

    bg = cv2.resize(bg_bgr, (w, h)).astype(np.float32)
    fg = img_bgr.astype(np.float32)
    composite = (fg * mask_3 + bg * (1.0 - mask_3)).astype(np.uint8)
    print('  완료.', flush=True)
    return composite


def _process_one_photo(args):
    """단일 사진 처리 (배경 교체 + 인페인팅) - ThreadPoolExecutor 워커용."""
    i, clean, drawn, mask, bg_img, style_preset = args
    # Step 1: 배경 합성 (MediaPipe - thread-local 인스턴스 사용)
    if bg_img is not None:
        print(f'[AI] 사진 {i+1}/4 배경 교체 중 (MediaPipe)...', flush=True)
        base = remove_bg_composite(clean, bg_img)
    else:
        base = clean.copy()
    # Step 2: 인페인팅 (Stability API HTTP 요청)
    has_drawing = (cv2.countNonZero(mask) > 0)
    if has_drawing:
        print(f'[AI] 사진 {i+1}/4 인페인팅 처리 중...', flush=True)
        base = pixelart_inpaint_one(base, mask, reference_bgr=drawn,
                                    style_preset=style_preset)
    return i, base


def build_ai_4cut(clean_photos, drawn_photos, masks, session_dir, bucket, theme_name, bg_img):
    try:
        if not config.STABILITY_API_KEY:
            raise ValueError('STABILITY_API_KEY 환경변수가 설정되지 않았습니다.')

        style_preset = theme_name if theme_name else 'pixel-art'
        print(f'[AI] 테마={style_preset}, 배경={"선택됨" if bg_img is not None else "원본"}', flush=True)

        n = min(4, len(clean_photos), len(drawn_photos), len(masks))

        # bg_img 는 모든 사진에 동일하게 사용 → 1회만 리사이즈
        if bg_img is not None and n > 0:
            ref_h, ref_w = clean_photos[0].shape[:2]
            bg_img = cv2.resize(bg_img, (ref_w, ref_h))

        task_args = [
            (i, clean_photos[i], drawn_photos[i], masks[i], bg_img, style_preset)
            for i in range(n)
        ]

        ai_photos = [None] * n
        print(f'[AI] 사진 {n}장 병렬 처리 시작 (최대 {n}개 스레드)...', flush=True)
        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = {executor.submit(_process_one_photo, args): args[0] for args in task_args}
            for future in as_completed(futures):
                idx, result = future.result()
                ai_photos[idx] = result
                print(f'[AI] 사진 {idx+1}/4 완료', flush=True)

        if assets.frame_raw is not None:
            fh, fw = assets.frame_raw.shape[:2]
            collage = np.ones((fh, fw, 3), dtype=np.uint8) * 255
            for i, (sx, sy, sw, sh) in enumerate(config.PHOTO_SLOTS):
                if i < len(ai_photos):
                    ph_s, pw_s = ai_photos[i].shape[:2]
                    pw = config.SAVE_PHOTO_W
                    ph = int(ph_s * (pw / pw_s))
                    resized = cv2.resize(ai_photos[i], (pw, ph))
                    ox  = sx + (sw - pw) // 2
                    oy  = sy + (sh - ph) // 2
                    oy2 = min(oy + ph, fh)
                    ox2 = min(ox + pw, fw)
                    collage[oy:oy2, ox:ox2] = resized[:oy2-oy, :ox2-ox]
            if assets.frame_raw.shape[2] == 4:
                alpha   = assets.frame_raw[:, :, 3:4] / 255.0
                collage = (assets.frame_raw[:, :, :3] * alpha + collage * (1 - alpha)).astype(np.uint8)
            else:
                collage = assets.frame_raw.copy()
        else:
            slot_h, slot_w = ai_photos[0].shape[:2]
            collage = np.vstack([cv2.resize(p, (slot_w, slot_h)) for p in ai_photos])

        ai_path = os.path.join(session_dir, 'ai_4cut.jpg')
        cv2.imwrite(ai_path, collage)

        bucket['img']   = collage
        bucket['error'] = None
        print(f'[AI] 완료 → {ai_path}', flush=True)

    except Exception as e:
        import traceback
        bucket['img']   = None
        bucket['error'] = str(e)
        print(f'[AI] 오류: {e}', flush=True)
        traceback.print_exc()
