import cv2
import mediapipe as mp
import os
import numpy as np
import time
import signal
import shutil
from datetime import datetime

# ══════════════════════════════════════════════════════════════════
#  Ctrl+C 안전 종료
# ══════════════════════════════════════════════════════════════════
_exit_requested = False

def _sigint_handler(sig, frame):
    global _exit_requested
    _exit_requested = True

signal.signal(signal.SIGINT, _sigint_handler)


# ══════════════════════════════════════════════════════════════════
#  상수 / 레이아웃
# ══════════════════════════════════════════════════════════════════
TOTAL_SHOTS   = 4
COUNTDOWN_SEC = 3
FLASH_SEC     = 0.5
FPS           = 30

# ─── 드로우 모드 ──────────────────────────────────────────────
DRAW_DEFAULT  = 'default'
DRAW_PAINTING = 'painting'
DRAW_ERASE    = 'erase'

# ─── 홀드 타이밍 ──────────────────────────────────────────────
HOLD_FIST    = 0.2   # fist 홀드 → painting ↔ erase 토글
HOLD_CLEAR   = 0.5   # thumbdown 홀드 → 캔버스 전체 삭제
HOLD_PHOTO   = 0.2   # victory 홀드 → 촬영

# ─── 색상 팔레트 (카메라 내) ──────────────────────────────────
PALETTE_CX      = 32
PALETTE_RADIUS  = 18
PALETTE_SPACING = 46

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── 배경 / 네컷 프레임 PNG ────────────────────────────────────
BG_IMAGE_PATH    = os.path.join(_BASE_DIR, 'image/background.png')
FRAME_IMAGE_PATH = os.path.join(_BASE_DIR, 'image/4cut_frame2.png')

# ─── 카메라 영역 ───────────────────────────────────────────────
CAM_X  = 80
CAM_W, CAM_H = 560, 420
CAM_Y  = (600 - CAM_H) // 2

# ─── 정보 패널 (카메라 상단) ───────────────────────────────────
INFO_X = CAM_X
INFO_Y = CAM_Y - 28
INFO_W = CAM_W

# ─── 네컷 프레임 캔버스 배치 위치 ─────────────────────────────
FRAME_X, FRAME_Y = 720, 10

# ─── 사진 슬롯 (4cut_frame.png 내부 상대 좌표) ────────────────
PHOTO_W, PHOTO_H = 155, 116

PHOTO_SLOTS = [
    (16,  13),
    (16, 135),
    (16, 257),
    (16, 380),
]

# ─── 컬러 ──────────────────────────────────────────────────────
WHITE    = (255, 255, 255)
BLACK    = (0,   0,   0)
GRAY     = (180, 180, 180)
BG_COLOR = (245, 235, 255)

PEN_COLORS = [
    (0,   0,   255),
    (0,   165, 255),
    (0,   220, 255),
    (60,  200,  60),
    (255, 120,   0),
    (200,  60, 180),
    (255, 255, 255),
]

# ─── 저장 경로 ─────────────────────────────────────────────────
SAVE_DIR  = os.path.join(_BASE_DIR, 'photobooth_output')
_VID_TMP  = os.path.join(SAVE_DIR, '_rec_tmp.avi')
_VID_PLAY = os.path.join(SAVE_DIR, '_rec_play.avi')
os.makedirs(SAVE_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  MediaPipe 설정
# ══════════════════════════════════════════════════════════════════
BaseOptions              = mp.tasks.BaseOptions
GestureRecognizer        = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode              = mp.tasks.vision.RunningMode

_mp_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(_BASE_DIR, 'gesture_recognizer.task')),
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=RunningMode.VIDEO,
)

_GESTURE_MAP = {
    'Closed_Fist': 'fist',
    'Victory':     'peace',
    'Open_Palm':   'open',
    'Thumb_Down':  'thumbdown',
}


# ══════════════════════════════════════════════════════════════════
#  카메라 초기화
# ══════════════════════════════════════════════════════════════════
def _init_camera():
    for idx in range(20):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.ndim == 3:
                print(f"[카메라] /dev/video{idx} 연결됨")
                return cap
        cap.release()
    return None

video = _init_camera()
if video is None:
    print("[오류] 카메라를 찾을 수 없습니다.")
    exit(1)


# ══════════════════════════════════════════════════════════════════
#  배경 / 프레임 이미지 로드
# ══════════════════════════════════════════════════════════════════
_bg_raw    = cv2.imread(BG_IMAGE_PATH)
_frame_raw = cv2.imread(FRAME_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

if _bg_raw is None:
    print(f"[경고] 배경 이미지 없음: {BG_IMAGE_PATH} → 단색 배경 사용")
if _frame_raw is None:
    print(f"[경고] 프레임 이미지 없음: {FRAME_IMAGE_PATH}")
elif _frame_raw.shape[2] == 3:
    bgra = cv2.cvtColor(_frame_raw, cv2.COLOR_BGR2BGRA)
    white_mask = np.all(_frame_raw >= 240, axis=2)
    bgra[white_mask, 3] = 0
    _frame_raw = bgra
    print("[프레임] 알파채널 없음 → 흰색 투명 처리")

_bg_resized = None


# ══════════════════════════════════════════════════════════════════
#  색상 팔레트 (카메라 프레임 위)
# ══════════════════════════════════════════════════════════════════
def _palette_positions(h):
    n       = len(PEN_COLORS)
    total   = n * PALETTE_SPACING
    start_y = (h - total) // 2 + PALETTE_SPACING // 2
    return [(PALETTE_CX, start_y + i * PALETTE_SPACING) for i in range(n)]


def _draw_color_palette(frame, color_idx):
    h         = frame.shape[0]
    positions = _palette_positions(h)

    for idx, ((cx, cy), color) in enumerate(zip(positions, PEN_COLORS)):
        is_sel = (idx == color_idx)
        r = PALETTE_RADIUS if is_sel else PALETTE_RADIUS - 4
        cv2.circle(frame, (cx, cy), r, color, -1)
        if is_sel:
            cv2.circle(frame, (cx, cy), r + 3, WHITE, 2)


def _palette_hit(ix, iy, h):
    for idx, (cx, cy) in enumerate(_palette_positions(h)):
        if (ix - cx) ** 2 + (iy - cy) ** 2 <= (PALETTE_RADIUS + 6) ** 2:
            return idx
    return -1


# ══════════════════════════════════════════════════════════════════
#  커서 아이콘
# ══════════════════════════════════════════════════════════════════
def _draw_pencil_icon(frame, ix, iy, color_bgr):
    length, tip_len, width = 36, 10, 8
    rad  = np.deg2rad(-45)
    dx   = int(np.cos(rad) * length)
    dy   = int(np.sin(rad) * length)
    bx, by = ix - dx, iy - dy
    perp = rad + np.pi / 2
    pw   = int(width / 2)
    pdx  = int(np.cos(perp) * pw)
    pdy  = int(np.sin(perp) * pw)
    body = np.array([
        [bx + pdx, by + pdy],
        [bx - pdx, by - pdy],
        [ix - pdx - int(np.cos(rad) * tip_len), iy - pdy - int(np.sin(rad) * tip_len)],
        [ix + pdx - int(np.cos(rad) * tip_len), iy + pdy - int(np.sin(rad) * tip_len)],
    ], dtype=np.int32)
    tip_bx = ix - int(np.cos(rad) * tip_len)
    tip_by = iy - int(np.sin(rad) * tip_len)
    tip = np.array([[tip_bx + pdx, tip_by + pdy], [tip_bx - pdx, tip_by - pdy], [ix, iy]], dtype=np.int32)
    cv2.fillPoly(frame, [body], color_bgr)
    cv2.polylines(frame, [body], True, WHITE, 1, cv2.LINE_AA)
    cv2.fillPoly(frame, [tip], (100, 190, 255))
    cv2.polylines(frame, [tip], True, WHITE, 1, cv2.LINE_AA)
    cv2.circle(frame, (ix, iy), 2, (50, 50, 50), -1)


def _draw_eraser_icon(frame, ix, iy):
    w2, h2 = 18, 12
    ox, oy = ix + 5, iy - h2 - 5
    cv2.rectangle(frame, (ox, oy), (ox + w2 * 2, oy + h2 * 2), (220, 220, 255), -1)
    cv2.rectangle(frame, (ox, oy), (ox + w2 * 2, oy + h2 * 2), WHITE, 2)
    sy = oy + int(h2 * 1.4)
    cv2.rectangle(frame, (ox, sy), (ox + w2 * 2, oy + h2 * 2), (130, 100, 255), -1)
    cv2.line(frame, (ox, sy), (ox + w2 * 2, sy), WHITE, 1)
    cv2.circle(frame, (ix, iy), 3, (130, 100, 255), -1)


def _draw_cursor_icon(frame, ix, iy):
    s   = 28
    tri = np.array([[ix, iy], [ix, iy + s], [ix + s * 2 // 3, iy + s * 2 // 3]], dtype=np.int32)
    cv2.fillPoly(frame, [tri], WHITE)
    cv2.polylines(frame, [tri], True, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.circle(frame, (ix, iy), 3, (0, 200, 255), -1)


# ══════════════════════════════════════════════════════════════════
#  프레임 + 사진 캔버스에 그리기
# ══════════════════════════════════════════════════════════════════
def _render_frame(canvas, photos):
    if _frame_raw is None:
        return

    fh, fw = _frame_raw.shape[:2]
    avail_w = canvas.shape[1] - FRAME_X
    avail_h = canvas.shape[0] - FRAME_Y
    scale   = min(avail_w / fw, avail_h / fh)
    nw, nh  = int(fw * scale), int(fh * scale)
    frame_disp = cv2.resize(_frame_raw, (nw, nh))

    for i, (sx, sy) in enumerate(PHOTO_SLOTS):
        if i < len(photos):
            ph_s, pw_s = photos[i].shape[:2]
            ps = min(PHOTO_W / pw_s, PHOTO_H / ph_s)
            pw, ph = int(pw_s * ps), int(ph_s * ps)
            resized = cv2.resize(photos[i], (pw, ph))
            ax  = FRAME_X + int(sx * scale) + (int(PHOTO_W * scale) - pw) // 2
            ay  = FRAME_Y + int(sy * scale) + (int(PHOTO_H * scale) - ph) // 2
            ay2 = min(ay + ph, canvas.shape[0])
            ax2 = min(ax + pw, canvas.shape[1])
            canvas[ay:ay2, ax:ax2] = resized[:ay2 - ay, :ax2 - ax]

    roi = canvas[FRAME_Y:FRAME_Y + nh, FRAME_X:FRAME_X + nw]
    if frame_disp.shape[2] == 4:
        alpha = frame_disp[:, :, 3:4] / 255.0
        roi[:] = (frame_disp[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)
    else:
        roi[:] = frame_disp


# ══════════════════════════════════════════════════════════════════
#  정보 패널
# ══════════════════════════════════════════════════════════════════
_GESTURE_LABEL = {
    'peace':    'PEACE',
    'fist':     'FIST',
    'open':     'OPEN',
    'thumbdown':'THUMB DOWN',
}
_MODE_LABEL = {
    DRAW_DEFAULT:  '',
    DRAW_PAINTING: 'PAINT',
    DRAW_ERASE:    'ERASE',
}

def _draw_info_panel(canvas, gesture, result, draw_mode):
    x, y, w = INFO_X, INFO_Y, INFO_W
    cy = y + 14

    parts = []
    if result is not None and result.hand_landmarks:
        g = _GESTURE_LABEL.get(gesture, 'DRAW')
        parts.append(g)
    m = _MODE_LABEL.get(draw_mode, '')
    if m:
        parts.append(f'[{m}]')

    text = '  '.join(parts)
    if text:
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        cv2.putText(canvas, text, (x + w // 2 - tw // 2, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 200, 255), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════
#  콜라주 저장
# ══════════════════════════════════════════════════════════════════
def save_final(photos, session_dir):
    os.makedirs(session_dir, exist_ok=True)
    for i, photo in enumerate(photos):
        cv2.imwrite(os.path.join(session_dir, f"shot_{i+1}.jpg"), photo)
        print(f"  저장: shot_{i+1}.jpg")

    if _frame_raw is not None:
        fh, fw = _frame_raw.shape[:2]
        collage = np.ones((fh, fw, 3), dtype=np.uint8) * 255

        for i, (sx, sy) in enumerate(PHOTO_SLOTS):
            if i < len(photos):
                ph_s, pw_s = photos[i].shape[:2]
                s  = min(PHOTO_W / pw_s, PHOTO_H / ph_s)
                pw, ph = int(pw_s * s), int(ph_s * s)
                resized = cv2.resize(photos[i], (pw, ph))
                ox  = sx + (PHOTO_W - pw) // 2
                oy  = sy + (PHOTO_H - ph) // 2
                oy2 = min(oy + ph, fh)
                ox2 = min(ox + pw, fw)
                collage[oy:oy2, ox:ox2] = resized[:oy2 - oy, :ox2 - ox]

        if _frame_raw.shape[2] == 4:
            alpha   = _frame_raw[:, :, 3:4] / 255.0
            collage = (_frame_raw[:, :, :3] * alpha + collage * (1 - alpha)).astype(np.uint8)
        else:
            collage = _frame_raw.copy()

        cv2.imwrite(os.path.join(session_dir, "4cut.jpg"), collage)

    print(f"✓ 저장 완료 → {session_dir}")


# ══════════════════════════════════════════════════════════════════
#  상태 머신
# ══════════════════════════════════════════════════════════════════
STATE_WAITING   = 'waiting'
STATE_COUNTDOWN = 'countdown'
STATE_FLASH     = 'flash'
STATE_REVIEW    = 'review'


# ══════════════════════════════════════════════════════════════════
#  메인 루프
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("  인생네컷 포토부스  v6")
print("=" * 55)
print("  peace          : 촬영")
print("  hand (기본)    : 팔레트 터치로 색 변경 / 커서")
print("  fist 0.2s 홀드 : PAINT ↔ ERASE 전환")
print("  thumb down     : DEFAULT 모드로 복귀")
print("  open           : 캔버스 지우기 (리뷰 시 초기화)")
print("  ESC / q        : 종료")
print("=" * 55)

cv2.namedWindow('PhotoBooth', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('PhotoBooth', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

state           = STATE_WAITING
photos          = []
countdown_start = None
flash_start     = None
last_gesture    = None

draw_canvas    = None
prev_x, prev_y = None, None
color_idx      = 0
drawing_color  = PEN_COLORS[color_idx]
line_thickness = 5

draw_mode           = DRAW_DEFAULT
fist_start          = None
fist_toggled        = False
thumbdown_start     = None
thumbdown_fired     = False
victory_start       = None
victory_fired       = False

out_writer  = None
review_cap  = None

with GestureRecognizer.create_from_options(_mp_options) as recognizer:
    frame_index = 0

    while True:
        ret, frame = video.read()
        if not ret or _exit_requested:
            break

        frame        = cv2.flip(frame, 1)
        cam_h, cam_w = frame.shape[:2]
        now          = time.time()
        frame_clean  = frame.copy()

        if draw_canvas is None:
            draw_canvas = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

        if _bg_resized is None:
            if _bg_raw is not None:
                _bg_resized = _bg_raw.copy()
            else:
                total_w = FRAME_X + (_frame_raw.shape[1] if _frame_raw is not None else 280) + 20
                total_h = max(CAM_Y + CAM_H, FRAME_Y + (_frame_raw.shape[0] if _frame_raw is not None else 580)) + 20
                _bg_resized = np.full((total_h, total_w, 3), BG_COLOR, dtype=np.uint8)

        if out_writer is None and state != STATE_REVIEW:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(_VID_TMP, fourcc, FPS, (cam_w, cam_h))
            if writer.isOpened():
                out_writer = writer
                print("[녹화 시작]")
            else:
                writer.release()
                out_writer = False

        # ── 손 인식
        gesture = None
        result  = None
        if state != STATE_REVIEW:
            img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result   = recognizer.recognize_for_video(mp_image, int(frame_index * 1000 / FPS))
        frame_index += 1

        if result is None or not result.hand_landmarks:
            prev_x, prev_y  = None, None
            fist_start      = None
            fist_toggled    = False
            thumbdown_start = None
            thumbdown_fired = False
            victory_start   = None
            victory_fired   = False
        else:
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                raw     = result.gestures[i][0].category_name if result.gestures else 'None'
                gesture = _GESTURE_MAP.get(raw, None)
                ix      = int(hand_landmarks[8].x * cam_w)
                iy      = int(hand_landmarks[8].y * cam_h)

                if gesture == 'open':
                    # 즉시 DEFAULT 모드 전환
                    draw_mode      = DRAW_DEFAULT
                    prev_x, prev_y = None, None

                elif gesture == 'fist':
                    if last_gesture != 'fist':
                        fist_start   = now
                        fist_toggled = False
                    if fist_start is not None and not fist_toggled and now - fist_start >= HOLD_FIST:
                        if draw_mode in (DRAW_DEFAULT, DRAW_ERASE):
                            draw_mode = DRAW_PAINTING
                        else:
                            draw_mode = DRAW_ERASE
                        fist_toggled = True
                    prev_x, prev_y = None, None

                elif gesture == 'thumbdown':
                    if last_gesture != 'thumbdown':
                        thumbdown_start = now
                        thumbdown_fired = False
                    if thumbdown_start is not None and not thumbdown_fired and now - thumbdown_start >= HOLD_CLEAR:
                        draw_canvas     = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
                        thumbdown_fired = True
                    prev_x, prev_y = None, None

                elif gesture == 'peace':
                    if last_gesture != 'peace' or victory_start is None:
                        victory_start = now
                        victory_fired = False
                    if (victory_start is not None and not victory_fired
                            and now - victory_start >= HOLD_PHOTO
                            and state == STATE_WAITING):
                        state           = STATE_COUNTDOWN
                        countdown_start = now
                        victory_fired   = True
                    prev_x, prev_y = None, None

                else:
                    if state != STATE_COUNTDOWN:
                        hit = _palette_hit(ix, iy, cam_h)
                        if hit >= 0:
                            color_idx      = hit
                            drawing_color  = PEN_COLORS[color_idx]
                            prev_x, prev_y = None, None
                        elif draw_mode == DRAW_PAINTING:
                            if prev_x is not None:
                                cv2.line(draw_canvas, (prev_x, prev_y), (ix, iy),
                                         drawing_color, line_thickness)
                            prev_x, prev_y = ix, iy
                        elif draw_mode == DRAW_ERASE:
                            cv2.circle(draw_canvas, (ix, iy), line_thickness * 4, BLACK, -1)
                            prev_x, prev_y = None, None
                        else:
                            prev_x, prev_y = None, None
                    else:
                        prev_x, prev_y = None, None

                # 홀드 타이머 리셋
                if gesture != 'fist':
                    fist_start, fist_toggled = None, False
                if gesture != 'thumbdown':
                    thumbdown_start, thumbdown_fired = None, False
                if gesture != 'peace':
                    victory_start, victory_fired = None, False

                # 커서 아이콘
                if draw_mode == DRAW_PAINTING:
                    _draw_pencil_icon(frame, ix, iy, drawing_color)
                elif draw_mode == DRAW_ERASE:
                    _draw_eraser_icon(frame, ix, iy)
                else:
                    _draw_cursor_icon(frame, ix, iy)

        # 팔레트 표시 (리뷰 제외)
        if state != STATE_REVIEW:
            _draw_color_palette(frame, color_idx)

        # ── open → 리뷰 초기화
        if gesture == 'open' and last_gesture != 'open' and state == STATE_REVIEW:
            photos     = []
            if review_cap:
                review_cap.release()
                review_cap = None
            out_writer  = None
            draw_canvas = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
            state       = STATE_WAITING
            print("초기화 완료")

        last_gesture = gesture

        # ── 상태 전환
        if state == STATE_COUNTDOWN:
            if now - countdown_start >= COUNTDOWN_SEC:
                gray_m  = cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
                _, msk  = cv2.threshold(gray_m, 1, 255, cv2.THRESH_BINARY)
                msk_inv = cv2.bitwise_not(msk)
                shot    = cv2.add(cv2.bitwise_and(frame_clean, frame_clean, mask=msk_inv),
                                  cv2.bitwise_and(draw_canvas, draw_canvas, mask=msk))
                photos.append(shot.copy())
                draw_canvas = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
                state       = STATE_FLASH
                flash_start = now
                print(f"[{len(photos)}/{TOTAL_SHOTS}] 촬영!")
                if len(photos) >= TOTAL_SHOTS:
                    session_dir = os.path.join(SAVE_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
                    save_final(photos, session_dir)

        elif state == STATE_FLASH:
            if now - flash_start >= FLASH_SEC:
                if len(photos) >= TOTAL_SHOTS:
                    if out_writer and out_writer is not False:
                        out_writer.release()
                        out_writer = None
                    if os.path.exists(_VID_TMP):
                        shutil.copy2(_VID_TMP, _VID_PLAY)
                        review_cap = cv2.VideoCapture(_VID_PLAY)
                    state = STATE_REVIEW
                else:
                    state = STATE_WAITING

        # ── 녹화
        if out_writer and out_writer is not False and state != STATE_REVIEW:
            out_writer.write(frame_clean)

        # ── 그리기 합성
        if state != STATE_REVIEW:
            gray_m  = cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
            _, msk  = cv2.threshold(gray_m, 1, 255, cv2.THRESH_BINARY)
            msk_inv = cv2.bitwise_not(msk)
            frame   = cv2.add(cv2.bitwise_and(frame, frame, mask=msk_inv),
                              cv2.bitwise_and(draw_canvas, draw_canvas, mask=msk))

        # ── 캔버스 합성
        canvas = _bg_resized.copy()
        _render_frame(canvas, photos)

        if state == STATE_REVIEW:
            if review_cap:
                ret_v, vframe = review_cap.read()
                if not ret_v:
                    review_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_v, vframe = review_cap.read()
                if ret_v and vframe is not None:
                    canvas[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W] = cv2.resize(vframe, (CAM_W, CAM_H))
                    cv2.putText(canvas, "REPLAY", (CAM_X + 10, CAM_Y + 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 3, cv2.LINE_AA)
                    cv2.putText(canvas, "REPLAY", (CAM_X + 10, CAM_Y + 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

        else:
            canvas[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W] = cv2.resize(frame, (CAM_W, CAM_H))

            if state == STATE_COUNTDOWN:
                num_show = max(1, COUNTDOWN_SEC - int(now - countdown_start))
                cx = CAM_X + CAM_W // 2
                cy = CAM_Y + CAM_H // 2
                ov = canvas.copy()
                cv2.circle(ov, (cx, cy), 100, BLACK, -1)
                cv2.addWeighted(ov, 0.5, canvas, 0.5, 0, canvas)
                tw = cv2.getTextSize(str(num_show), cv2.FONT_HERSHEY_DUPLEX, 5.0, 10)[0][0]
                cv2.putText(canvas, str(num_show), (cx - tw // 2, cy + 35),
                            cv2.FONT_HERSHEY_DUPLEX, 5.0, WHITE, 10, cv2.LINE_AA)

            elif state == STATE_FLASH:
                ratio = 1.0 - (now - flash_start) / FLASH_SEC
                wh    = np.full_like(canvas, 255)
                cv2.addWeighted(wh, ratio * 0.9, canvas, 1 - ratio * 0.9, 0, canvas)

            _draw_info_panel(canvas, gesture, result, draw_mode)

        cv2.imshow('PhotoBooth', canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break


# ══════════════════════════════════════════════════════════════════
#  종료 정리
# ══════════════════════════════════════════════════════════════════
if out_writer and out_writer is not False:
    out_writer.release()
if review_cap:
    review_cap.release()
if video:
    video.release()
cv2.destroyAllWindows()
print("종료.")
