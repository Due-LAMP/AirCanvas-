import sys
import os
import cv2
import numpy as np
import tempfile
import threading
import config
import assets

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'photo_post_process'))

try:
    from shape_classifier import classify as _classify_shape
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
    from carvekit.api.high import HiInterface as _HiInterface
    _BG_REMOVER_AVAILABLE = True
except Exception as _e:
    print(f'[경고] carvekit 로드 실패 (배경 교체 비활성화): {_e}', flush=True)
    _BG_REMOVER_AVAILABLE = False

_bg_remover_instance = None
_bg_remover_lock = threading.Lock()


def get_bg_remover():
    global _bg_remover_instance
    if _bg_remover_instance is None and _BG_REMOVER_AVAILABLE:
        with _bg_remover_lock:
            if _bg_remover_instance is None:
                _bg_remover_instance = _HiInterface(
                    object_type='object', batch_size_seg=1, batch_size_matting=1,
                    device='cpu', seg_mask_size=640, matting_mask_size=2048,
                )
    return _bg_remover_instance


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
    _PILImage.fromarray(img_rgb).save(buf_img, format='PNG')
    image_bytes = buf_img.getvalue()

    buf_mask = _io.BytesIO()
    _PILImage.fromarray(mask_filled).save(buf_mask, format='PNG')
    mask_bytes = buf_mask.getvalue()

    shape_name    = 'unknown'
    subject_prompt = 'a decorative accessory'

    if SHAPE_CLASSIFIER_AVAILABLE:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
            tmp_path = tf.name
        try:
            cv2.imwrite(tmp_path, mask_gray)
            shape_name, subject_prompt, _conf = _classify_shape(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

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
    import io as _io
    from PIL import Image as _PILImage, ImageFilter as _ImageFilter

    remover = get_bg_remover()
    if remover is None:
        print('[배경] carvekit 없음 → 배경 교체 스킵', flush=True)
        return img_bgr

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = _PILImage.fromarray(img_rgb)

    print('[배경] 인물 배경 제거 중 (carvekit)...', flush=True)
    rgba = remover([pil_img])[0]

    r, g, b, alpha = rgba.split()
    alpha = alpha.filter(_ImageFilter.GaussianBlur(radius=2))
    rgba = _PILImage.merge('RGBA', (r, g, b, alpha))

    bg_rgb = cv2.cvtColor(cv2.resize(bg_bgr, (w, h)), cv2.COLOR_BGR2RGB)
    bg_pil = _PILImage.fromarray(bg_rgb).convert('RGBA')

    bg_pil.paste(rgba, (0, 0), mask=rgba)
    result = np.array(bg_pil.convert('RGB'))
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def process_shot_inpaint(photo_clean, drawn_photo, mask, theme_name, result_bucket, index):
    """배경 교체 없는 경우 촬영 직후 인페인팅만 백그라운드로 처리.
    carvekit(CPU-heavy)은 포함하지 않으므로 촬영 중 프레임 드랍이 없음."""
    print(f'[AI] {index+1}번 인페인팅 시작 (네트워크)...', flush=True)
    try:
        style_preset = theme_name if theme_name else 'pixel-art'
        has_drawing  = (cv2.countNonZero(mask) > 0)
        out = pixelart_inpaint_one(photo_clean, mask, reference_bgr=drawn_photo,
                                   style_preset=style_preset) if has_drawing else photo_clean.copy()
        result_bucket[index] = out
        print(f'[AI] {index+1}번 인페인팅 완료', flush=True)
    except Exception as e:
        result_bucket[index] = photo_clean
        print(f'[AI] {index+1}번 인페인팅 오류 (원본 사용): {e}', flush=True)


def build_ai_4cut(clean_photos, drawn_photos, masks, session_dir, bucket, theme_name, bg_img,
                  preprocess_bucket=None, preprocess_threads=None):
    try:
        if not config.STABILITY_API_KEY:
            raise ValueError('STABILITY_API_KEY 환경변수가 설정되지 않았습니다.')

        style_preset = theme_name if theme_name else 'pixel-art'
        print(f'[AI] 테마={style_preset}, 배경={"선택됨" if bg_img is not None else "원본"}', flush=True)

        # 인페인팅 사전처리 스레드가 아직 실행 중이면 완료 대기
        if preprocess_threads:
            for t in preprocess_threads:
                t.join()

        ai_photos = []
        for i, (clean, drawn, mask) in enumerate(zip(clean_photos[:4], drawn_photos[:4], masks[:4])):
            if bg_img is None and preprocess_bucket and i in preprocess_bucket:
                # 배경 없음: 인페인팅 사전처리 결과 사용
                ai_photos.append(preprocess_bucket[i])
            else:
                # 배경 있음: carvekit(마지막에) → 인페인팅 순서로 처리
                print(f'[AI] 사진 {i+1}/4 처리 중 (carvekit + 인페인팅)...', flush=True)
                base = clean.copy()
                if bg_img is not None:
                    base = remove_bg_composite(clean, bg_img)
                has_drawing = (cv2.countNonZero(mask) > 0)
                ai_photos.append(pixelart_inpaint_one(base, mask, reference_bgr=drawn,
                                                      style_preset=style_preset) if has_drawing else base)

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
