import sys
import cv2
import mediapipe as mp
import os
import numpy as np
import time
import signal
import shutil
import socket
import threading
import http.server
import socketserver
from datetime import datetime

import qrcode
from PIL import Image

# Gmail API
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gmail_api'))
from send_message import gmail_send_message_with_attachment

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
HOLD_FIST    = 0.2
HOLD_CLEAR   = 0.5
HOLD_PHOTO   = 0.2

# ─── 색상 팔레트 (카메라 내) ──────────────────────────────────
PALETTE_CX      = 32
PALETTE_RADIUS  = 18
PALETTE_SPACING = 46

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── 배경 / 네컷 프레임 PNG ────────────────────────────────────
BG_IMAGE_PATH        = os.path.join(_BASE_DIR, 'image/background_line.png')
BG_RESULT_IMAGE_PATH = os.path.join(_BASE_DIR, 'image/background.png')
FRAME_IMAGE_PATH     = os.path.join(_BASE_DIR, 'image/4cut_frame.png')

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

# ─── 사진 슬롯 (4cut_frame.png 원본 좌표: x, y, w, h) ─────────
PHOTO_SLOTS = [
    (3,  38, 420, 217),
    (3, 320, 420, 223),
    (3, 628, 420, 186),
    (3, 908, 420, 200),
]

# ─── 화면 표시용 사진 크기 ────────────────────────────────────
DISPLAY_PHOTO_W = 160
DISPLAY_PHOTO_H = 120

# ─── 저장용 사진 너비 (높이는 비율에 따라 자동 계산) ───────────
SAVE_PHOTO_W = 340

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

# ─── HTTP 서버 포트 ────────────────────────────────────────────
HTTP_PORT = 8080

# ─── 결과 페이지 미니 영상 / QR 크기 ──────────────────────────
MINI_W, MINI_H = 200, 150   # 좌상단 소형 리플레이
QR_SIZE        = 150         # QR 코드 크기 (정사각형)


# ══════════════════════════════════════════════════════════════════
#  로컬 IP / HTTP 서버
# ══════════════════════════════════════════════════════════════════
def _get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.254.254.254', 1))
        return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        s.close()

LOCAL_IP = _get_local_ip()
print(f"[서버] 로컬 IP: {LOCAL_IP}:{HTTP_PORT}")


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SAVE_DIR, **kwargs)

    def log_message(self, *args):
        pass


class _ReuseServer(socketserver.TCPServer):
    allow_reuse_address = True


def _start_http_server():
    with _ReuseServer(('', HTTP_PORT), _QuietHandler) as httpd:
        httpd.serve_forever()


threading.Thread(target=_start_http_server, daemon=True).start()
print(f"[서버] http://{LOCAL_IP}:{HTTP_PORT} 에서 사진 서비스 중")


# ══════════════════════════════════════════════════════════════════
#  QR 코드 생성 (OpenCV 이미지로 반환)
# ══════════════════════════════════════════════════════════════════
def _make_qr_cv(url, size=QR_SIZE):
    qr = qrcode.QRCode(box_size=4, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    pil_img = qr.make_image(fill_color='black', back_color='white').convert('RGB')
    pil_img = pil_img.resize((size, size), Image.NEAREST)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


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
    'Pointing_Up': 'cursor',
    'Thumb_Up':    'thumbup',
}


# ══════════════════════════════════════════════════════════════════
#  카메라 초기화
# ══════════════════════════════════════════════════════════════════
def _init_camera():
    for idx in range(20):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        if ret and frame is not None and frame.ndim == 3:
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"[카메라] /dev/video{idx} 연결됨 ({w}x{h} @ {fps:.0f}fps MJPG)")
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
_bg_raw        = cv2.imread(BG_IMAGE_PATH)
_bg_result_raw = cv2.imread(BG_RESULT_IMAGE_PATH)
_frame_raw     = cv2.imread(FRAME_IMAGE_PATH, cv2.IMREAD_UNCHANGED)

if _bg_raw is None:
    print(f"[경고] 배경 이미지 없음: {BG_IMAGE_PATH} → 단색 배경 사용")
if _bg_result_raw is None:
    print(f"[경고] 결과 배경 이미지 없음: {BG_RESULT_IMAGE_PATH}")
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

    for i, (sx, sy, sw, sh) in enumerate(PHOTO_SLOTS):
        if i < len(photos):
            ph_s, pw_s = photos[i].shape[:2]
            ps = min(DISPLAY_PHOTO_W / pw_s, DISPLAY_PHOTO_H / ph_s)
            pw, ph = int(pw_s * ps), int(ph_s * ps)
            resized = cv2.resize(photos[i], (pw, ph))
            ax  = FRAME_X + int(sx * scale) + (int(sw * scale) - pw) // 2
            ay  = FRAME_Y + int(sy * scale) + (int(sh * scale) - ph) // 2
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
    'cursor':    'Pointing_Up',
    'peace':     'Victory',
    'fist':      'Closed_Fist',
    'open':      'Open_Palm',
    'thumbdown': 'Thumb_Down',
    'thumbup':   'Thumb_Up',
}
_MODE_LABEL = {
    DRAW_DEFAULT:  'DEFAULT',
    DRAW_PAINTING: 'PAINT',
    DRAW_ERASE:    'ERASE',
}

def _draw_info_panel(canvas, gesture, result, draw_mode):
    x, y, w = INFO_X, INFO_Y, INFO_W
    cy = y + 14

    if result is not None and result.hand_landmarks:
        raw = result.gestures[0][0].category_name if result.gestures else 'None'
        g = _GESTURE_LABEL.get(gesture, raw)
        m = 'SHOOT' if gesture == 'peace' else 'RESET' if gesture == 'thumbdown' else _MODE_LABEL.get(draw_mode, 'DEFAULT')
        text = f'{g} [{m}]'
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        cv2.putText(canvas, text, (x + w // 2 - tw // 2, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLACK, 1, cv2.LINE_AA)


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

        for i, (sx, sy, sw, sh) in enumerate(PHOTO_SLOTS):
            if i < len(photos):
                ph_s, pw_s = photos[i].shape[:2]
                pw = SAVE_PHOTO_W
                ph = int(ph_s * (pw / pw_s))
                resized = cv2.resize(photos[i], (pw, ph))
                ox  = sx + (sw - pw) // 2
                oy  = sy + (sh - ph) // 2
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
STATE_RESULT    = 'result'

REVIEW_SEC = 5


# ══════════════════════════════════════════════════════════════════
#  이메일 전송 (백그라운드 스레드)
# ══════════════════════════════════════════════════════════════════
email_status = None   # None | 'sending' | 'sent' | 'error'

def _send_email_async(attachment_path):
    global email_status
    print(f"[이메일] 전송 중: {attachment_path}")
    result = gmail_send_message_with_attachment(attachment_path)
    if result:
        email_status = 'sent'
        print("[이메일] 전송 완료!")
    else:
        email_status = 'error'
        print("[이메일] 전송 실패")


# ══════════════════════════════════════════════════════════════════
#  메인 루프
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("  인생네컷 포토부스  v6 + QR + Email")
print("=" * 55)
print("  peace          : 촬영")
print("  hand (기본)    : 팔레트 터치로 색 변경 / 커서")
print("  fist 0.2s 홀드 : PAINT ↔ ERASE 전환")
print("  thumb down     : DEFAULT 모드로 복귀")
print("  open           : 캔버스 지우기 (리뷰/결과 시 초기화)")
print("  thumb up       : [결과 화면] 이메일 전송")
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

draw_mode       = DRAW_DEFAULT
fist_start      = None
fist_toggled    = False
thumbdown_start = None
thumbdown_fired = False
victory_start   = None
victory_fired   = False

out_writer     = None
review_cap     = None
review_start   = None
result_collage = None
qr_img         = None
session_dir    = None

with GestureRecognizer.create_from_options(_mp_options) as recognizer:
    while True:
        ret, frame = video.read()
        if not ret or _exit_requested:
            break

        frame = cv2.flip(frame, 1)
        # 16:9 → 4:3 센터 크롭 (1280x720 → 960x720)
        fh, fw = frame.shape[:2]
        target_w = fh * 4 // 3
        x0 = (fw - target_w) // 2
        frame = frame[:, x0:x0 + target_w]
        cam_h, cam_w = frame.shape[:2]
        now         = time.time()
        frame_clean = frame.copy()

        if draw_canvas is None:
            draw_canvas = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

        if _bg_resized is None:
            if _bg_raw is not None:
                _bg_resized = _bg_raw.copy()
            else:
                total_w = FRAME_X + (_frame_raw.shape[1] if _frame_raw is not None else 280) + 20
                total_h = max(CAM_Y + CAM_H, FRAME_Y + (_frame_raw.shape[0] if _frame_raw is not None else 580)) + 20
                _bg_resized = np.full((total_h, total_w, 3), BG_COLOR, dtype=np.uint8)

        if out_writer is None and state not in (STATE_REVIEW, STATE_RESULT):
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
            result   = recognizer.recognize_for_video(mp_image, int(time.time() * 1000))

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

                elif gesture == 'thumbup':
                    # 결과 화면에서 엄지 → 이메일 전송
                    if (state == STATE_RESULT and last_gesture != 'thumbup'
                            and session_dir is not None and email_status is None):
                        attachment = os.path.join(session_dir, "4cut.jpg")
                        email_status = 'sending'
                        threading.Thread(target=_send_email_async, args=(attachment,), daemon=True).start()
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
                            cv2.circle(draw_canvas, (ix, iy), line_thickness * 4 + 1, BLACK, -1)
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

        # 팔레트 표시 (리뷰/결과 제외)
        if state not in (STATE_REVIEW, STATE_RESULT):
            _draw_color_palette(frame, color_idx)

        # ── open → 리뷰/결과 초기화
        if gesture == 'open' and last_gesture != 'open' and state in (STATE_REVIEW, STATE_RESULT):
            photos         = []
            result_collage = None
            qr_img         = None
            review_start   = None
            email_status   = None
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
                    result_collage = cv2.imread(os.path.join(session_dir, "4cut.jpg"))
                    session_name   = os.path.basename(session_dir)
                    qr_url         = f"http://{LOCAL_IP}:{HTTP_PORT}/{session_name}/4cut.jpg"
                    qr_img         = _make_qr_cv(qr_url)
                    email_status   = None
                    print(f"[QR] {qr_url}")

        elif state == STATE_FLASH:
            if now - flash_start >= FLASH_SEC:
                if len(photos) >= TOTAL_SHOTS:
                    if out_writer and out_writer is not False:
                        out_writer.release()
                        out_writer = None
                    if os.path.exists(_VID_TMP):
                        shutil.copy2(_VID_TMP, _VID_PLAY)
                        review_cap = cv2.VideoCapture(_VID_PLAY)
                    state        = STATE_REVIEW
                    review_start = now
                else:
                    state = STATE_WAITING

        elif state == STATE_REVIEW:
            if review_start is not None and now - review_start >= REVIEW_SEC:
                state = STATE_RESULT

        # ── 녹화
        if out_writer and out_writer is not False and state not in (STATE_REVIEW, STATE_RESULT):
            out_writer.write(frame_clean)

        # ── 그리기 합성
        if state not in (STATE_REVIEW, STATE_RESULT):
            gray_m  = cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
            _, msk  = cv2.threshold(gray_m, 1, 255, cv2.THRESH_BINARY)
            msk_inv = cv2.bitwise_not(msk)
            frame   = cv2.add(cv2.bitwise_and(frame, frame, mask=msk_inv),
                              cv2.bitwise_and(draw_canvas, draw_canvas, mask=msk))

        # ── 캔버스 합성
        canvas = _bg_resized.copy()

        if state == STATE_RESULT:
            # ── 결과 페이지: background.png + 중앙 콜라주 + 좌상단 소형 리플레이 + 우상단 QR
            if _bg_result_raw is not None:
                canvas = _bg_result_raw.copy()

            # 중앙 콜라주
            if result_collage is not None:
                ch, cw   = result_collage.shape[:2]
                disp_h   = canvas.shape[0] - 40
                disp_w   = int(cw * disp_h / ch)
                cx0      = (canvas.shape[1] - disp_w) // 2
                cy0      = 20
                canvas[cy0:cy0 + disp_h, cx0:cx0 + disp_w] = cv2.resize(result_collage, (disp_w, disp_h))

            # 좌상단 소형 리플레이
            if review_cap:
                ret_v, vframe = review_cap.read()
                if not ret_v:
                    review_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_v, vframe = review_cap.read()
                if ret_v and vframe is not None:
                    canvas[10:10 + MINI_H, 10:10 + MINI_W] = cv2.resize(vframe, (MINI_W, MINI_H))
                cv2.rectangle(canvas, (10, 10), (10 + MINI_W, 10 + MINI_H), WHITE, 1)

            # 우상단 QR 코드
            if qr_img is not None:
                qx = 10 + MINI_W + 15
                qy = 10
                canvas[qy:qy + QR_SIZE, qx:qx + QR_SIZE] = qr_img
                cv2.rectangle(canvas, (qx, qy), (qx + QR_SIZE, qy + QR_SIZE), GRAY, 1)
                cv2.putText(canvas, "QR scan to save", (qx, qy + QR_SIZE + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1, cv2.LINE_AA)

            # 이메일 상태 표시
            if email_status == 'sending':
                msg      = "Sending email..."
                msg_col  = (0, 165, 255)
            elif email_status == 'sent':
                msg      = "Email sent!"
                msg_col  = (0, 200, 80)
            elif email_status == 'error':
                msg      = "Email failed"
                msg_col  = (0, 0, 220)
            else:
                msg      = "Thumb Up to send email"
                msg_col  = GRAY

            tw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            ex = (canvas.shape[1] - tw) // 2
            ey = canvas.shape[0] - 14
            cv2.putText(canvas, msg, (ex, ey),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, msg_col, 1, cv2.LINE_AA)

        elif state == STATE_REVIEW:
            _render_frame(canvas, photos)
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
            _render_frame(canvas, photos)
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
