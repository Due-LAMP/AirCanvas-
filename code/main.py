import sys
import os
import argparse
import cv2
import mediapipe as mp
import numpy as np
import time
import signal
import threading
from datetime import datetime

import config
import assets
import ui
import ai_processor
import saver

os.makedirs(config.SAVE_DIR, exist_ok=True)
for _f in (config.VID_TMP, config.VID_PLAY):
    try:
        if os.path.exists(_f):
            os.remove(_f)
    except Exception:
        pass

# ── CLIP 백그라운드 프리로드 ──────────────────────────────────────
def _preload_clip():
    try:
        from shape_classifier import _load_clip
        _load_clip()
    except Exception as _e:
        print(f"[CLIP 프리로드] 실패 (무시): {_e}", flush=True)

threading.Thread(target=_preload_clip, daemon=True).start()

# ── Ctrl+C 안전 종료 ──────────────────────────────────────────────
_exit_requested = False


def _sigint_handler(sig, frame):
    global _exit_requested
    _exit_requested = True


signal.signal(signal.SIGINT, _sigint_handler)


def _decode_email_input_key(key: int):
    """OpenCV 키 코드를 이메일 입력용 제어/문자로 정규화한다."""
    if key < 0:
        return None, None

    enter_codes = {10, 13, 3}
    escape_codes = {27}
    backspace_codes = {8, 127}

    if key in enter_codes:
        return 'enter', None
    if key in escape_codes:
        return 'escape', None
    if key in backspace_codes:
        return 'backspace', None

    key_ascii = key & 0xFF
    if key_ascii in enter_codes:
        return 'enter', None
    if key_ascii in escape_codes:
        return 'escape', None
    if key_ascii in backspace_codes:
        return 'backspace', None
    if 32 <= key_ascii <= 126:
        return 'text', chr(key_ascii)
    return None, None


# ── MediaPipe 설정 ────────────────────────────────────────────────
BaseOptions              = mp.tasks.BaseOptions
GestureRecognizer        = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode              = mp.tasks.vision.RunningMode

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_mp_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(_BASE_DIR, 'models/gesture_recognizer.task')),
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=RunningMode.VIDEO,
)


# ── 카메라 초기화 ─────────────────────────────────────────────────
def _init_camera():
    if sys.platform == 'darwin':
        backend_candidates = [
            ('AVFoundation', getattr(cv2, 'CAP_AVFOUNDATION', None)),
            ('Default', None),
        ]
    else:
        backend_candidates = [
            ('V4L2', getattr(cv2, 'CAP_V4L2', None)),
            ('Default', None),
        ]

    for backend_name, backend in backend_candidates:
        for idx in range(10):
            cap = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue

            if sys.platform != 'darwin':
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, frame = cap.read()
            if ret and frame is not None and frame.ndim == 3:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"[카메라] backend={backend_name}, index={idx} 연결됨 ({w}x{h} @ {fps:.0f}fps)")
                return cap
            cap.release()
    return None


def run(screen_record: bool = False):
    video = _init_camera()
    if video is None:
        print("[오류] 카메라를 찾을 수 없습니다.")
        sys.exit(1)

    print("=" * 55)
    print("  인생네컷 포토부스 + QR + Email + Theme/BG Select")
    print("=" * 55)
    print(" ######## 제스처 안내 ########")
    print(" [인트로]")
    print("  open           : 테마 & 배경 선택 시작")
    print(" [선택 화면]")
    print("  thumb up       : 테마 & 배경 선택")
    print(" [촬영 화면]")
    print("  hand (기본)    : 팔레트 터치로 색 변경 / 커서")
    print("  fist 0.2s 홀드 : DEFAULT -> PAINT / PAINT ↔ ERASE 전환")
    print("  thumb down     : PAINT 초기화")
    print("  open           : DEFAULT 모드로 전환")
    print("  peace          : 촬영 시작")
    print(" [결과 화면]")
    print("  thumb up       : 이메일 전송")
    print("  open 3s 홀드   : 리뷰/결과/이메일 화면에서 초기화")
    print("  ESC / q        : 종료")
    print("=" * 55)

    cv2.namedWindow('PhotoBooth', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('PhotoBooth', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    state           = config.STATE_INTRO
    photos          = []
    countdown_start = None
    flash_start     = None
    last_gesture    = None

    draw_canvas    = None
    prev_x, prev_y = None, None
    color_idx      = 0
    drawing_color  = config.PEN_COLORS[color_idx]
    line_thickness = 5

    draw_mode        = config.DRAW_DEFAULT
    draw_delay_start = None
    fist_start       = None
    fist_toggled     = False
    reset_start      = None
    thumbdown_start  = None
    thumbdown_fired  = False
    victory_start    = None
    victory_fired    = False

    countdown_cooldown_until   = 0.0
    selection_hold_start       = None
    selection_cooldown_until   = 0.0

    selected_theme_name = None
    selected_bg_img     = None
    theme_hovered_cell  = -1
    bg_hovered_cell     = -1

    out_writer     = None
    screen_writer  = None
    review_cap     = None
    review_start   = None
    result_collage = None
    ai_collage     = None
    ai_bucket      = {}
    qr_img         = None
    session_dir    = None
    photos_clean   = []
    draw_masks     = []

    _bg_resized = None

    with GestureRecognizer.create_from_options(_mp_options) as recognizer:
        while True:
            ret, frame = video.read()
            if not ret or _exit_requested:
                break

            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            target_w = fh * 4 // 3
            x0 = (fw - target_w) // 2
            frame = frame[:, x0:x0 + target_w]
            cam_h, cam_w = frame.shape[:2]
            now         = time.time()
            frame_clean = frame.copy()

            if draw_canvas is None:
                draw_canvas = np.full((cam_h, cam_w, 3), 255, dtype=np.uint8)

            if _bg_resized is None:
                if assets.bg_raw is not None:
                    _bg_resized = assets.bg_raw.copy()
                else:
                    total_w = config.FRAME_X + (assets.frame_raw.shape[1] if assets.frame_raw is not None else 280) + 20
                    total_h = max(config.CAM_Y + config.CAM_H, config.FRAME_Y + (assets.frame_raw.shape[0] if assets.frame_raw is not None else 580)) + 20
                    _bg_resized = np.full((total_h, total_w, 3), config.BG_COLOR, dtype=np.uint8)

            if out_writer is None and state not in (
                config.STATE_INTRO, config.STATE_REVIEW, config.STATE_RESULT,
                config.STATE_EMAIL_INPUT, config.STATE_SELECT_THEME, config.STATE_SELECT_BG
            ):
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                writer = cv2.VideoWriter(config.VID_TMP, fourcc, config.FPS, (cam_w, cam_h))
                if writer.isOpened():
                    out_writer = writer
                    print("[녹화 시작]")
                else:
                    writer.release()
                    out_writer = False

            # ── 손 인식
            gesture = None
            result  = None
            if state != config.STATE_REVIEW:
                img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result   = recognizer.recognize_for_video(mp_image, int(time.time() * 1000))

            if result is None or not result.hand_landmarks:
                prev_x, prev_y   = None, None
                draw_delay_start = None
                fist_start       = None
                fist_toggled     = False
                thumbdown_start  = None
                thumbdown_fired  = False
                victory_start    = None
                victory_fired    = False
                selection_hold_start = None
            else:
                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    raw     = result.gestures[i][0].category_name if result.gestures else 'None'
                    gesture = config.GESTURE_MAP.get(raw, None)
                    ix      = int(hand_landmarks[8].x * cam_w)
                    iy      = int(hand_landmarks[8].y * cam_h)

                    if gesture == 'open':
                        draw_mode        = config.DRAW_DEFAULT
                        prev_x, prev_y   = None, None
                        draw_delay_start = None

                    elif gesture == 'fist':
                        if last_gesture != 'fist':
                            fist_start   = now
                            fist_toggled = False
                        if fist_start is not None and not fist_toggled and now - fist_start >= config.HOLD_FIST:
                            if draw_mode in (config.DRAW_DEFAULT, config.DRAW_ERASE):
                                draw_mode = config.DRAW_PAINTING
                            else:
                                draw_mode = config.DRAW_ERASE
                            fist_toggled     = True
                            draw_delay_start = None
                        prev_x, prev_y = None, None

                    elif gesture == 'thumbdown':
                        if last_gesture != 'thumbdown':
                            thumbdown_start = now
                            thumbdown_fired = False
                        if thumbdown_start is not None and not thumbdown_fired and now - thumbdown_start >= config.HOLD_CLEAR:
                            draw_canvas     = np.full((cam_h, cam_w, 3), 255, dtype=np.uint8)
                            thumbdown_fired = True
                        prev_x, prev_y   = None, None
                        draw_delay_start = None

                    elif gesture == 'peace':
                        if last_gesture != 'peace' or victory_start is None:
                            victory_start = now
                            victory_fired = False
                        if (victory_start is not None and not victory_fired
                                and now - victory_start >= config.HOLD_PHOTO
                                and now >= countdown_cooldown_until
                                and state == config.STATE_WAITING):
                            state           = config.STATE_COUNTDOWN
                            countdown_start = now
                            victory_fired   = True
                        prev_x, prev_y   = None, None
                        draw_delay_start = None

                    elif gesture == 'thumbup':
                        if (state == config.STATE_RESULT and last_gesture != 'thumbup'
                                and session_dir is not None and saver.email_status is None):
                            saver.email_input_text = ''
                            state = config.STATE_EMAIL_INPUT
                        prev_x, prev_y   = None, None
                        draw_delay_start = None

                    else:
                        if state != config.STATE_COUNTDOWN:
                            hit = ui.palette_hit(ix, iy, cam_h)
                            if hit >= 0:
                                color_idx        = hit
                                drawing_color    = config.PEN_COLORS[color_idx]
                                prev_x, prev_y   = None, None
                                draw_delay_start = None
                            elif draw_mode == config.DRAW_PAINTING:
                                if draw_delay_start is None:
                                    draw_delay_start = now
                                if now - draw_delay_start >= 0.5:
                                    if prev_x is not None:
                                        cv2.line(draw_canvas, (prev_x, prev_y), (ix, iy),
                                                 drawing_color, line_thickness)
                                    prev_x, prev_y = ix, iy
                            elif draw_mode == config.DRAW_ERASE:
                                if draw_delay_start is None:
                                    draw_delay_start = now
                                if now - draw_delay_start >= 0.5:
                                    cv2.circle(draw_canvas, (ix, iy), line_thickness * 4 + 1, config.WHITE, -1)
                                prev_x, prev_y = None, None
                            else:
                                prev_x, prev_y   = None, None
                                draw_delay_start = None
                        else:
                            prev_x, prev_y = None, None

                    if gesture != 'fist':
                        fist_start, fist_toggled = None, False
                    if gesture != 'thumbdown':
                        thumbdown_start, thumbdown_fired = None, False
                    if gesture != 'peace':
                        victory_start, victory_fired = None, False

                    if draw_mode == config.DRAW_PAINTING:
                        ui.draw_pencil_icon(frame, ix, iy, drawing_color)
                    elif draw_mode == config.DRAW_ERASE:
                        ui.draw_eraser_icon(frame, ix, iy)
                    else:
                        ui.draw_cursor_icon(frame, ix, iy)

            # ── open → 인트로에서 테마 선택으로
            if gesture == 'open' and last_gesture != 'open' and state == config.STATE_INTRO:
                state = config.STATE_SELECT_THEME
                theme_hovered_cell = -1
                print("→ 테마 선택")

            # ── 테마 선택 제스처
            if state == config.STATE_SELECT_THEME and result is not None and result.hand_landmarks:
                hand = result.hand_landmarks[0]
                fx = int(hand[8].x * cam_w)
                fy = int(hand[8].y * cam_h)

                if assets.source_theme_img is not None:
                    hit = ui.cell_hit(fx, fy, cam_w, cam_h, assets.SOURCE_THEME_CELLS)
                    if hit >= 0:
                        theme_hovered_cell = hit

                if gesture == 'thumbup':
                    if selection_hold_start is None:
                        selection_hold_start = now
                    elif now - selection_hold_start >= config.HOLD_SELECT and theme_hovered_cell >= 0:
                        selected_theme_name = config.SOURCE_THEME_NAMES[theme_hovered_cell]
                        print(f"[테마 선택] 셀 {theme_hovered_cell} '{selected_theme_name}'")
                        state = config.STATE_SELECT_BG
                        bg_hovered_cell = -1
                        selection_hold_start = None
                        selection_cooldown_until = now + config.THEME_TO_BG_COOLDOWN
                        countdown_cooldown_until = now + 1.5
                        victory_start = None
                        victory_fired = False
                else:
                    selection_hold_start = None

            # ── 배경 선택 제스처
            if state == config.STATE_SELECT_BG and result is not None and result.hand_landmarks:
                hand = result.hand_landmarks[0]
                fx = int(hand[8].x * cam_w)
                fy = int(hand[8].y * cam_h)

                if assets.source_bg_img is not None:
                    hit = ui.cell_hit(fx, fy, cam_w, cam_h, config.SOURCE_BG_CELLS)
                    if hit >= 0:
                        bg_hovered_cell = hit

                if gesture == 'thumbup' and now >= selection_cooldown_until:
                    if selection_hold_start is None:
                        selection_hold_start = now
                    elif now - selection_hold_start >= config.HOLD_SELECT and bg_hovered_cell >= 0:
                        name = config.SOURCE_BG_NAMES[bg_hovered_cell]
                        if name == 'none':
                            selected_bg_img = None
                            print(f"[배경 선택] 셀 {bg_hovered_cell} 'none' → 배경 교체 없음 (inpainting만)")
                        else:
                            bg_file = os.path.join(config.SOURCE_IMAGE_DIR, config.SOURCE_BG_FILES[bg_hovered_cell])
                            _loaded = cv2.imread(bg_file)
                            if _loaded is not None:
                                selected_bg_img = _loaded
                                print(f"[배경 선택] 셀 {bg_hovered_cell} '{name}'  {bg_file}  크기={_loaded.shape[1]}x{_loaded.shape[0]}")
                            else:
                                selected_bg_img = None
                                print(f"[배경 선택] 파일 로드 실패: {bg_file}")
                        state = config.STATE_WAITING
                        selection_hold_start = None
                        countdown_cooldown_until = now + 1.5
                        victory_start = None
                        victory_fired = False
                else:
                    selection_hold_start = None

            # ── open 3초 홀드 → 리뷰/결과/이메일 초기화
            _open_held = (gesture == 'open' or last_gesture == 'open') and gesture != 'thumbup'
            if _open_held and state in (config.STATE_REVIEW, config.STATE_RESULT, config.STATE_EMAIL_INPUT):
                if reset_start is None:
                    reset_start = now
                elif now - reset_start >= config.HOLD_RESET:
                    photos         = []
                    photos_clean   = []
                    draw_masks     = []
                    result_collage = None
                    ai_collage     = None
                    ai_bucket      = {}
                    qr_img         = None
                    review_start   = None
                    saver.email_status = None
                    if review_cap:
                        review_cap.release()
                        review_cap = None
                    out_writer  = None
                    draw_canvas = np.full((cam_h, cam_w, 3), 255, dtype=np.uint8)
                    for _p in (config.VID_TMP, config.VID_PLAY):
                        try:
                            if os.path.exists(_p):
                                os.remove(_p)
                        except Exception:
                            pass
                    reset_start = None
                    state       = config.STATE_INTRO
                    print("초기화 완료")
            else:
                if gesture != 'open' and last_gesture != 'open':
                    reset_start = None

            last_gesture = gesture

            # ── 상태 전환
            if state == config.STATE_COUNTDOWN:
                if now - countdown_start >= config.COUNTDOWN_SEC:
                    gray_m  = cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
                    _, msk  = cv2.threshold(gray_m, 253, 255, cv2.THRESH_BINARY_INV)
                    msk_inv = cv2.bitwise_not(msk)
                    shot    = cv2.add(cv2.bitwise_and(frame_clean, frame_clean, mask=msk_inv),
                                      cv2.bitwise_and(draw_canvas, draw_canvas, mask=msk))
                    shot_index = len(photos)
                    photos.append(shot.copy())
                    photos_clean.append(frame_clean.copy())
                    draw_masks.append(msk.copy())
                    draw_canvas = np.full((cam_h, cam_w, 3), 255, dtype=np.uint8)
                    state       = config.STATE_FLASH
                    flash_start = now
                    print(f"[{len(photos)}/{config.TOTAL_SHOTS}] 촬영!")

                    if len(photos) >= config.TOTAL_SHOTS:
                        session_dir = os.path.join(config.SAVE_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))

                        def _save_and_start_ai(_photos, _session_dir, _masks,
                                               _theme, _bg, _bucket_ref):
                            saver.save_final(_photos, _session_dir, masks=_masks)
                            _rc = cv2.imread(os.path.join(_session_dir, "4cut.jpg"))
                            _bucket_ref['result_collage'] = _rc
                            _sn  = os.path.basename(_session_dir)
                            _url = f"http://{saver.LOCAL_IP}:{config.HTTP_PORT}/{_sn}/index.html"
                            _bucket_ref['qr_url'] = _url
                            if ai_processor.INPAINTING_AVAILABLE:
                                ai_processor.build_ai_4cut(
                                    list(_photos), list(_photos), list(_masks),
                                    _session_dir, _bucket_ref, _theme, _bg,
                                )

                        ai_bucket  = {}
                        ai_collage = None
                        saver.email_status = None
                        threading.Thread(
                            target=_save_and_start_ai,
                            args=(list(photos), session_dir, list(draw_masks),
                                  selected_theme_name, selected_bg_img, ai_bucket),
                            daemon=True,
                        ).start()

            elif state == config.STATE_FLASH:
                if now - flash_start >= config.FLASH_SEC:
                    if len(photos) >= config.TOTAL_SHOTS:
                        if out_writer and out_writer is not False:
                            _writer_to_release = out_writer
                            out_writer = None
                            print("[녹화 종료]")
                            saver.release_and_save(_writer_to_release)
                        state        = config.STATE_REVIEW
                        review_start = now
                    else:
                        state = config.STATE_WAITING

            elif state == config.STATE_REVIEW:
                if review_cap is None and os.path.exists(config.VID_PLAY):
                    review_cap = cv2.VideoCapture(config.VID_PLAY)
                # 저장 완료 시 result_collage / qr 갱신
                if result_collage is None and ai_bucket.get('result_collage') is not None:
                    result_collage = ai_bucket['result_collage']
                    _qr_url = ai_bucket.get('qr_url', '')
                    if _qr_url:
                        qr_img = saver.make_qr_cv(_qr_url)
                        print(f"[QR] {_qr_url}")
                ai_done = ai_bucket.get('img') is not None or ai_bucket.get('error') is not None
                if ai_done:
                    state = config.STATE_RESULT

            if state in (config.STATE_RESULT, config.STATE_EMAIL_INPUT):
                if review_cap is None and os.path.exists(config.VID_PLAY):
                    review_cap = cv2.VideoCapture(config.VID_PLAY)

            # ── 그리기 합성
            if state not in (config.STATE_REVIEW, config.STATE_RESULT, config.STATE_EMAIL_INPUT,
                             config.STATE_SELECT_THEME, config.STATE_SELECT_BG):
                gray_m  = cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
                _, msk  = cv2.threshold(gray_m, 253, 255, cv2.THRESH_BINARY_INV)
                msk_inv = cv2.bitwise_not(msk)
                frame   = cv2.add(cv2.bitwise_and(frame, frame, mask=msk_inv),
                                  cv2.bitwise_and(draw_canvas, draw_canvas, mask=msk))

            if out_writer and out_writer is not False and state not in (
                config.STATE_REVIEW, config.STATE_RESULT, config.STATE_EMAIL_INPUT,
                config.STATE_SELECT_THEME, config.STATE_SELECT_BG
            ):
                out_writer.write(frame)

            # ── 캔버스 합성
            canvas = _bg_resized.copy()

            if state == config.STATE_INTRO:
                if assets.intro_raw is not None:
                    canvas = cv2.resize(assets.intro_raw, (canvas.shape[1], canvas.shape[0]))
                else:
                    cv2.putText(canvas, "Open Palm to Start", (100, canvas.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, config.BLACK, 2, cv2.LINE_AA)

            elif state == config.STATE_SELECT_THEME:
                finger_x, finger_y = -1, -1
                if result is not None and result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    finger_x = int(hand[8].x * canvas.shape[1])
                    finger_y = int(hand[8].y * canvas.shape[0])
                theme_hovered_cell = ui.draw_theme_grid(canvas, theme_hovered_cell, finger_x, finger_y)
                if selection_hold_start is not None and finger_x >= 0:
                    prog  = min((now - selection_hold_start) / config.HOLD_SELECT, 1.0)
                    angle = int(-360 * prog)
                    cv2.ellipse(canvas, (finger_x, finger_y), (22, 22), -90, 0, angle, (0, 220, 255), 3)

            elif state == config.STATE_SELECT_BG:
                finger_x, finger_y = -1, -1
                if result is not None and result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    finger_x = int(hand[8].x * canvas.shape[1])
                    finger_y = int(hand[8].y * canvas.shape[0])
                bg_hovered_cell = ui.draw_bg_grid(canvas, bg_hovered_cell, finger_x, finger_y)
                if selection_hold_start is not None and finger_x >= 0:
                    prog  = min((now - selection_hold_start) / config.HOLD_SELECT, 1.0)
                    angle = int(-360 * prog)
                    cv2.ellipse(canvas, (finger_x, finger_y), (22, 22), -90, 0, angle, (0, 220, 255), 3)

            elif state in (config.STATE_RESULT, config.STATE_EMAIL_INPUT):
                if ai_collage is None and ai_bucket.get('img') is not None:
                    ai_collage = ai_bucket['img']
                    print("[AI] 픽셀아트 4cut 완료 — 화면 반영")
                if ai_collage is None and ai_bucket.get('error'):
                    print(f"[AI] 오류: {ai_bucket['error']}")
                    ai_bucket = {}

                if assets.bg_result_raw is not None:
                    canvas = assets.bg_result_raw.copy()

                cy0       = 20
                PAD       = 14
                LABEL_H   = 20
                cw_total  = canvas.shape[1]
                content_h = canvas.shape[0] - cy0 - 20
                left_w    = cw_total * 6 // 10 - PAD
                half_w    = left_w
                lx        = PAD

                TOP_H = content_h // 4
                BOT_H = content_h - TOP_H - PAD

                top_sz      = TOP_H - LABEL_H - 8
                replay_h    = top_sz
                replay_w    = replay_h * 4 // 3
                top_total_w = replay_w + PAD + top_sz
                top_start_x = lx + (half_w - top_total_w) // 2
                replay_x = top_start_x
                replay_y = cy0 + 4
                if review_cap:
                    ret_v, vframe = review_cap.read()
                    if not ret_v:
                        review_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_v, vframe = review_cap.read()
                    if ret_v and vframe is not None:
                        canvas[replay_y:replay_y + replay_h, replay_x:replay_x + replay_w] = \
                            cv2.resize(vframe, (replay_w, replay_h))
                    cv2.rectangle(canvas, (replay_x, replay_y),
                                  (replay_x + replay_w, replay_y + replay_h), config.GRAY, 1)
                (rt, _), _ = cv2.getTextSize("Replay", cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                cv2.putText(canvas, "Replay",
                            (replay_x + (replay_w - rt) // 2, replay_y + replay_h + LABEL_H - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.BLACK, 1, cv2.LINE_AA)

                qr_x = top_start_x + replay_w + PAD
                qr_y = cy0 + 4
                if qr_img is not None:
                    qr_resized = cv2.resize(qr_img, (top_sz, top_sz))
                    canvas[qr_y:qr_y + top_sz, qr_x:qr_x + top_sz] = qr_resized
                    cv2.rectangle(canvas, (qr_x, qr_y), (qr_x + top_sz, qr_y + top_sz), config.GRAY, 1)
                    (qt, _), _ = cv2.getTextSize("QR scan to save", cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                    cv2.putText(canvas, "QR scan to save",
                                (qr_x + (top_sz - qt) // 2, qr_y + top_sz + LABEL_H - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.BLACK, 1, cv2.LINE_AA)

                live_y = cy0 + TOP_H + PAD
                live_h = BOT_H - PAD - LABEL_H
                live_w = live_h * 4 // 3
                if live_w > half_w:
                    live_w = half_w
                    live_h = live_w * 3 // 4
                live_x = lx + (half_w - live_w) // 2
                canvas[live_y:live_y + live_h, live_x:live_x + live_w] = cv2.resize(frame, (live_w, live_h))

                if reset_start is not None:
                    progress = min((now - reset_start) / config.HOLD_RESET, 1.0)
                    bar_y = live_y + live_h + 1
                    cv2.rectangle(canvas, (live_x, bar_y), (live_x + live_w, bar_y + 7), (180, 180, 180), -1)
                    cv2.rectangle(canvas, (live_x, bar_y), (live_x + int(live_w * progress), bar_y + 7), config.WHITE, -1)

                if saver.email_status == 'sending':
                    email_msg = "Sending email..."
                elif saver.email_status == 'sent':
                    email_msg = "Email sent!"
                elif saver.email_status == 'error':
                    email_msg = "Email failed"
                else:
                    email_msg = "Thumbs up to send email"
                (et, _), _ = cv2.getTextSize(email_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                cv2.putText(canvas, email_msg,
                            (live_x + (live_w - et) // 2, live_y + live_h + LABEL_H),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.BLACK, 1, cv2.LINE_AA)

                rx      = cw_total * 6 // 10 + PAD
                right_w = cw_total - rx - PAD
                if result_collage is not None:
                    img_gap    = 10
                    n_imgs     = 2 if ai_collage is not None else 1
                    max_each_h = content_h - LABEL_H - PAD * 2
                    each_h     = min(max_each_h, result_collage.shape[0])
                    each_w     = int(each_h * result_collage.shape[1] / result_collage.shape[0])
                    total_w    = each_w * n_imgs + img_gap * (n_imgs - 1)
                    img_x0     = rx + max(0, (right_w - total_w) // 2)
                    img_y      = cy0 + PAD + (max_each_h - each_h) // 2

                    canvas[img_y:img_y + each_h, img_x0:img_x0 + each_w] = \
                        cv2.resize(result_collage, (each_w, each_h))
                    (ot, _), _ = cv2.getTextSize("Original", cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                    cv2.putText(canvas, "Original",
                                (img_x0 + (each_w - ot) // 2, img_y + each_h + LABEL_H),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.BLACK, 1, cv2.LINE_AA)

                    if ai_collage is not None:
                        ax = img_x0 + each_w + img_gap
                        canvas[img_y:img_y + each_h, ax:ax + each_w] = \
                            cv2.resize(ai_collage, (each_w, each_h))
                        (pt, _), _ = cv2.getTextSize("AI Generated", cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                        cv2.putText(canvas, "AI Generated",
                                    (ax + (each_w - pt) // 2, img_y + each_h + LABEL_H),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.BLACK, 1, cv2.LINE_AA)

            elif state == config.STATE_REVIEW:
                ui.render_frame(canvas, photos)
                if review_cap:
                    ret_v, vframe = review_cap.read()
                    if not ret_v:
                        review_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_v, vframe = review_cap.read()
                    if ret_v and vframe is not None:
                        canvas[config.CAM_Y:config.CAM_Y + config.CAM_H,
                               config.CAM_X:config.CAM_X + config.CAM_W] = cv2.resize(vframe, (config.CAM_W, config.CAM_H))
                        cv2.putText(canvas, "REPLAY", (config.CAM_X + 10, config.CAM_Y + 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.BLACK, 3, cv2.LINE_AA)
                        cv2.putText(canvas, "REPLAY", (config.CAM_X + 10, config.CAM_Y + 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.WHITE, 1, cv2.LINE_AA)
                        # AI 생성 중 자막
                        if int(time.time() * 2) % 2 == 0:
                            dots = "." * (int(time.time() * 1.5) % 4)
                            _ai_text = f"GENERATING{dots}"
                            (gw, _), _ = cv2.getTextSize(_ai_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 3)
                            _ai_tx = config.CAM_X + (config.CAM_W - gw) // 2
                            _ai_ty = config.CAM_Y + config.CAM_H - 18
                            cv2.putText(canvas, _ai_text, (_ai_tx, _ai_ty),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.BLACK, 3, cv2.LINE_AA)
                            cv2.putText(canvas, _ai_text, (_ai_tx, _ai_ty),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.WHITE, 1, cv2.LINE_AA)


            else:
                ui.render_frame(canvas, photos)
                canvas[config.CAM_Y:config.CAM_Y + config.CAM_H,
                       config.CAM_X:config.CAM_X + config.CAM_W] = cv2.resize(frame, (config.CAM_W, config.CAM_H))
                ui.draw_color_palette(canvas[config.CAM_Y:config.CAM_Y + config.CAM_H,
                                             config.CAM_X:config.CAM_X + config.CAM_W], color_idx)

                if state == config.STATE_COUNTDOWN:
                    num_show = max(1, config.COUNTDOWN_SEC - int(now - countdown_start))
                    cx = config.CAM_X + config.CAM_W // 2
                    cy = config.CAM_Y + config.CAM_H // 2
                    ov = canvas.copy()
                    cv2.circle(ov, (cx, cy), 100, config.BLACK, -1)
                    cv2.addWeighted(ov, 0.5, canvas, 0.5, 0, canvas)
                    tw = cv2.getTextSize(str(num_show), cv2.FONT_HERSHEY_DUPLEX, 5.0, 10)[0][0]
                    cv2.putText(canvas, str(num_show), (cx - tw // 2, cy + 35),
                                cv2.FONT_HERSHEY_DUPLEX, 5.0, config.WHITE, 10, cv2.LINE_AA)

                elif state == config.STATE_FLASH:
                    ratio = 1.0 - (now - flash_start) / config.FLASH_SEC
                    wh    = np.full_like(canvas, 255)
                    cv2.addWeighted(wh, ratio * 0.9, canvas, 1 - ratio * 0.9, 0, canvas)

                ui.draw_info_panel(canvas, gesture, result, draw_mode)

            if state == config.STATE_EMAIL_INPUT:
                overlay = canvas.copy()
                bx = canvas.shape[1] // 2 - 280
                by = canvas.shape[0] // 2 - 60
                cv2.rectangle(overlay, (bx, by), (bx + 560, by + 120), (30, 30, 30), -1)
                cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)
                cv2.rectangle(canvas, (bx, by), (bx + 560, by + 120), config.WHITE, 2)
                cv2.putText(canvas, "Enter email address:", (bx + 16, by + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.WHITE, 1, cv2.LINE_AA)
                cursor = '|' if int(time.time() * 2) % 2 == 0 else ' '
                cv2.putText(canvas, saver.email_input_text + cursor, (bx + 16, by + 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 220, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, "Enter: Send   ESC: Cancel", (bx + 16, by + 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.GRAY, 1, cv2.LINE_AA)

            if screen_record:
                if screen_writer is None:
                    sh, sw = canvas.shape[:2]
                    screen_writer = cv2.VideoWriter(
                        os.path.join(config.SAVE_DIR, 'screen_recording.mp4'),
                        cv2.VideoWriter_fourcc(*'mp4v'), 30, (sw, sh),
                    )
                screen_writer.write(canvas)

            cv2.imshow('PhotoBooth', canvas)
            key = cv2.waitKeyEx(1)

            if state == config.STATE_EMAIL_INPUT:
                key_type, key_value = _decode_email_input_key(key)
                if key_type == 'enter':
                    if saver.email_input_text.strip():
                        attachments = [os.path.join(session_dir, "4cut.jpg")]
                        ai_path = os.path.join(session_dir, "ai_4cut.jpg")
                        if os.path.exists(ai_path):
                            attachments.append(ai_path)
                        saver.email_status = 'sending'
                        threading.Thread(target=saver.send_email_async,
                                         args=(attachments, saver.email_input_text.strip()),
                                         daemon=True).start()
                    state = config.STATE_RESULT
                elif key_type == 'escape':
                    state = config.STATE_RESULT
                elif key_type == 'backspace':
                    saver.email_input_text = saver.email_input_text[:-1]
                elif key_type == 'text' and key_value is not None:
                    saver.email_input_text += key_value
            elif key & 0xFF in (27, ord('q')):
                break

    # ── 종료 정리 ─────────────────────────────────────────────────
    if out_writer and out_writer is not False:
        out_writer.release()
    if review_cap:
        review_cap.release()
    if screen_writer:
        screen_writer.release()
        print(f"[녹화] 저장 완료 → {os.path.join(config.SAVE_DIR, 'screen_recording.mp4')}")
    if video:
        video.release()
    cv2.destroyAllWindows()
    print("종료.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen-record', action='store_true', help='화면 전체를 screen_recording.mp4로 저장')
    args = parser.parse_args()
    run(screen_record=args.screen_record)