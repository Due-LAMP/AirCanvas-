import sys
import cv2
import mediapipe as mp
import os
import numpy as np
import time
import signal
import socket
import threading
import tempfile
import http.server
import socketserver
from datetime import datetime

import qrcode
from PIL import Image

# legacy/ 의 한 단계 위(code/)를 루트로 사용
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Gmail API
sys.path.insert(0, os.path.join(_CODE_DIR, 'gmail_api'))
from send_message import gmail_send_message_with_attachment

# photo_post_process (shape_classifier + aircanvas_inpainting)
sys.path.insert(0, os.path.join(_CODE_DIR, 'photo_post_process'))
try:
    from shape_classifier import classify as _classify_shape
    _SHAPE_CLASSIFIER_AVAILABLE = True
except Exception as _e:
    print(f'[경고] shape_classifier 로드 실패: {_e}')
    _SHAPE_CLASSIFIER_AVAILABLE = False

try:
    from aircanvas_inpainting import step1_inpaint as _step1_inpaint
    _INPAINTING_AVAILABLE = True
except Exception as _e:
    print(f'[경고] aircanvas_inpainting 로드 실패: {_e}')
    _INPAINTING_AVAILABLE = False

try:
    from carvekit.api.high import HiInterface as _HiInterface
    _BG_REMOVER_AVAILABLE = True
except Exception as _e:
    print(f'[경고] carvekit 로드 실패 (배경 교체 비활성화): {_e}')
    _BG_REMOVER_AVAILABLE = False

_bg_remover_instance = None
def _get_bg_remover():
    global _bg_remover_instance
    if _bg_remover_instance is None and _BG_REMOVER_AVAILABLE:
        _bg_remover_instance = _HiInterface(
            object_type='object', batch_size_seg=1, batch_size_matting=1,
            device='cpu', seg_mask_size=640, matting_mask_size=2048,
        )
    return _bg_remover_instance

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

# ─── Stability AI ────────────────────────────────────────────────
STABILITY_API_KEY = os.environ.get('STABILITY_API_KEY', '')

# ─── 드로우 모드 ──────────────────────────────────────────────
DRAW_DEFAULT  = 'default'
DRAW_PAINTING = 'painting'
DRAW_ERASE    = 'erase'

# ─── 홀드 타이밍 ──────────────────────────────────────────────
HOLD_FIST    = 0.2
HOLD_CLEAR   = 0.5
HOLD_PHOTO   = 0.2
HOLD_RESET   = 3.0

# ─── 테마 / 배경 선택 그리드 ───────────────────────────────────
SOURCE_THEME_PATH      = os.path.join(_CODE_DIR, 'image/source_theme.png')
SOURCE_THEME_MASK_PATH = os.path.join(_CODE_DIR, 'image/source_theme_mask.png')

def _load_cells_from_mask(mask_path, min_area=3000):
    _m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if _m is None:
        return None
    _, _bin = cv2.threshold(_m, 50, 255, cv2.THRESH_BINARY_INV)
    _n, _, _stats, _ = cv2.connectedComponentsWithStats(_bin, connectivity=8)
    _cells = []
    for _i in range(1, _n):
        _x, _y, _w, _h, _a = _stats[_i]
        if _a >= min_area:
            _cells.append((_x, _y, _x + _w, _y + _h))
    _cells.sort(key=lambda c: (c[1], c[0]))
    return _cells

_theme_cells_from_mask = _load_cells_from_mask(SOURCE_THEME_MASK_PATH)
SOURCE_THEME_CELLS = _theme_cells_from_mask if _theme_cells_from_mask else [
    (86,  190, 318, 346),   # 0: analog
    (396, 190, 627, 346),   # 1: origami
    (705, 190, 937, 346),   # 2: pixel art
    (86,  398, 318, 554),   # 3: neon punk
    (396, 398, 627, 554),   # 4: 3D model
    (705, 398, 937, 554),   # 5: photographic
]
SOURCE_THEME_NAMES   = ['analog-film', 'origami', 'pixel-art', 'neon-punk', '3d-model', 'photographic']
SOURCE_BG_PATH       = os.path.join(_CODE_DIR, 'image/source_background.png')
SOURCE_BG_MASK_PATH  = os.path.join(_CODE_DIR, 'image/source_background_mask.png')
SOURCE_BG_COLS       = 4   # 가로 셀 수
SOURCE_BG_ROWS       = 2   # 세로 행 수
# 마스크에서 추출한 8개 칸의 실제 픽셀 좌표 (x1,y1,x2,y2) in 1024×600 원본 기준
SOURCE_BG_CELLS = [
    (47,  182, 253, 337),   # 0: white
    (290, 182, 496, 337),   # 1: skyblue
    (533, 182, 739, 337),   # 2: lightpink
    (775, 182, 982, 337),   # 3: lightgreen
    (47,  391, 253, 546),   # 4: beach
    (290, 391, 496, 546),   # 5: space
    (533, 391, 739, 546),   # 6: zombie
    (775, 391, 982, 546),   # 7: chimchakman
]
SOURCE_BG_NAMES = [
    'white', 'skyblue', 'lightpink', 'lightgreen',
    'beach', 'space', 'zombie', 'chimchakman',
]
SOURCE_IMAGE_DIR  = os.path.join(_CODE_DIR, 'image/source')
SOURCE_BG_FILES   = [
    'White.jpg', 'Skyblue.jpg', 'Lightpink.jpg', 'Green.jpg',
    'Beach.jpg', 'Space.jpg',   'Zombie.jpg',     'Chimchakman.jpg',
]

# ─── 색상 팔레트 (카메라 내) ──────────────────────────────────
PALETTE_CX      = 32
PALETTE_RADIUS  = 18
PALETTE_SPACING = 46

_BASE_DIR = _CODE_DIR

# ─── 배경 / 네컷 프레임 PNG ────────────────────────────────────
BG_IMAGE_PATH        = os.path.join(_BASE_DIR, 'image/background_line.png')
BG_RESULT_IMAGE_PATH = os.path.join(_BASE_DIR, 'image/background.png')
FRAME_IMAGE_PATH     = os.path.join(_BASE_DIR, 'image/4cut_frame.png')
INTRO_IMAGE_PATH     = os.path.join(_BASE_DIR, 'image/page_1.png')

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
    (0,    0,   0),
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
    base_options=BaseOptions(model_asset_path=os.path.join(_BASE_DIR, 'models/gesture_recognizer.task')),
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
_intro_raw     = cv2.imread(INTRO_IMAGE_PATH)

if _bg_raw is None:
    print(f"[경고] 배경 이미지 없음: {BG_IMAGE_PATH} → 단색 배경 사용")
if _bg_result_raw is None:
    print(f"[경고] 결과 배경 이미지 없음: {BG_RESULT_IMAGE_PATH}")
if _intro_raw is None:
    print(f"[경고] 인트로 이미지 없음: {INTRO_IMAGE_PATH}")
if _frame_raw is None:
    print(f"[경고] 프레임 이미지 없음: {FRAME_IMAGE_PATH}")
elif _frame_raw.shape[2] == 3:
    bgra = cv2.cvtColor(_frame_raw, cv2.COLOR_BGR2BGRA)
    white_mask = np.all(_frame_raw >= 240, axis=2)
    bgra[white_mask, 3] = 0
    _frame_raw = bgra
    print("[프레임] 알파채널 없음 → 흰색 투명 처리")

# ── source_theme.png / source_background.png 로드 ──────────────────────
_source_theme_img = cv2.imread(SOURCE_THEME_PATH)
if _source_theme_img is None:
    print(f"[경고] source_theme.png 로드 실패: {SOURCE_THEME_PATH}")
else:
    print(f"[테마 그리드] {_source_theme_img.shape[1]}x{_source_theme_img.shape[0]}, {len(SOURCE_THEME_CELLS)}개 칸")

_source_bg_img = cv2.imread(SOURCE_BG_PATH)
if _source_bg_img is None:
    print(f"[경고] source_background.png 로드 실패: {SOURCE_BG_PATH}")
else:
    _sbh, _sbw = _source_bg_img.shape[:2]
    print(f"[배경 그리드] {_sbw}x{_sbh}, {len(SOURCE_BG_CELLS)}개 칸 (마스크 기반)")

selected_theme_name = None   # 선택된 테마 이름
selected_bg_img     = None   # 선택된 배경 BGR ndarray
theme_hovered_cell  = -1     # 테마 호버 셀 (0~5)
bg_hovered_cell     = -1     # 배경 호버 셀 (0~7)
select_peace_start        = None   # 선택 화면 Victory 홀드 시작 시각
select_peace_cooldown_until = 0.0  # 테마→배경 전환 후 브이 무시 구간 종료 시각
HOLD_SELECT         = 0.8    # 선택 확정에 필요한 Victory 홀드 시간 (초)
THEME_TO_BG_COOLDOWN = 1.5   # 테마 확정 후 배경 선택에서 브이 무시 시간 (초)

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
def save_final(photos, session_dir, masks=None):
    os.makedirs(session_dir, exist_ok=True)
    for i, photo in enumerate(photos):
        cv2.imwrite(os.path.join(session_dir, f"shot_{i+1}.jpg"), photo)
        print(f"  저장: shot_{i+1}.jpg")
        if masks is not None and i < len(masks):
            mask_path = os.path.join(session_dir, f"shot_{i+1}_mask.png")
            cv2.imwrite(mask_path, masks[i])
            print(f"  저장: shot_{i+1}_mask.png")

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
#  Pixel Art AI 인페인팅 (main.py 방식 — shape 감지 → pixel-art preset)
# ══════════════════════════════════════════════════════════════════
def _fill_mask_interior(mask_bin):
    """닫힌 윤곽선 내부를 채운다."""
    h, w = mask_bin.shape[:2]
    close_k = np.ones((3, 3), np.uint8)
    closed  = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, close_k, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        filled = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        return cv2.bitwise_or(filled, mask_bin)
    # fallback flood-fill
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:h+1, 1:w+1] = closed
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    outside  = flood[1:h+1, 1:w+1]
    interior = cv2.bitwise_not(outside)
    return cv2.bitwise_or(interior, mask_bin)


def _pixelart_inpaint_one(img_bgr, mask_gray, style_preset='pixel-art'):
    """
    마스크 영역을 shape 감지 후 지정 스타일로 인페인팅.
    style_preset: Stability AI style_preset 문자열
    """
    import io as _io
    from PIL import Image as _PILImage

    h, w = img_bgr.shape[:2]
    _, mask_bin = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(mask_bin) == 0:
        return img_bgr

    mask_filled = _fill_mask_interior(mask_bin)

    # BGR → PNG bytes
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buf_img  = _io.BytesIO()
    _PILImage.fromarray(img_rgb).save(buf_img, format='PNG')
    image_bytes = buf_img.getvalue()

    # 마스크 → PNG bytes
    buf_mask = _io.BytesIO()
    _PILImage.fromarray(mask_filled).save(buf_mask, format='PNG')
    mask_bytes = buf_mask.getvalue()

    # shape 감지 → 프롬프트
    style_label = style_preset.replace('-', ' ')
    prompt = (
        f'{style_label} style object, vivid colors, '
        'seamlessly blended with surrounding photo'
    )
    if _SHAPE_CLASSIFIER_AVAILABLE:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
            tmp_path = tf.name
        try:
            cv2.imwrite(tmp_path, mask_gray)
            shape_name, _, _conf = _classify_shape(tmp_path)
            if shape_name != 'unknown':
                subject = shape_name.replace('_', ' ')
                prompt = (
                    f'Add {subject} in {style_label} style, '
                    'vivid colors, seamlessly blended with surrounding photo'
                )
            print(f'[AI] 형태 감지: {shape_name} / 스타일: {style_preset}')
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    result_bytes = _step1_inpaint(image_bytes, mask_bytes, prompt, style_preset=style_preset)

    arr = np.frombuffer(result_bytes, dtype=np.uint8)
    result_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if result_bgr is None:
        raise RuntimeError('인페인팅 응답 이미지 디코딩 실패')
    return cv2.resize(result_bgr, (w, h))


def _remove_bg_composite(img_bgr, bg_bgr):
    """
    carvekit으로 인물 배경 제거 후 bg_bgr 위에 합성. BGR ndarray 반환.
    carvekit 없으면 원본 반환.
    """
    import io as _io
    from PIL import Image as _PILImage, ImageFilter as _ImageFilter

    remover = _get_bg_remover()
    if remover is None:
        print('[배경] carvekit 없음 → 배경 교체 스킵')
        return img_bgr

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = _PILImage.fromarray(img_rgb)

    print('[배경] 인물 배경 제거 중 (carvekit)...')
    rgba = remover([pil_img])[0]  # RGBA

    # 알파 페더링
    r, g, b, alpha = rgba.split()
    alpha = alpha.filter(_ImageFilter.GaussianBlur(radius=2))
    rgba = _PILImage.merge('RGBA', (r, g, b, alpha))

    # bg_bgr → PIL RGBA
    bg_rgb = cv2.cvtColor(cv2.resize(bg_bgr, (w, h)), cv2.COLOR_BGR2RGB)
    bg_pil = _PILImage.fromarray(bg_rgb).convert('RGBA')

    bg_pil.paste(rgba, (0, 0), mask=rgba)
    result = np.array(bg_pil.convert('RGB'))
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def _build_ai_4cut(clean_photos, masks, session_dir, bucket, theme_name, bg_img):
    """
    백그라운드 스레드:
    1) bg_img 선택 시 → carvekit 배경 제거 후 합성
    2) theme_name(style_preset) 으로 마스크 영역 인페인팅
    → ai_4cut.jpg 저장 + bucket['img'] 에 BGR ndarray 저장
    """
    try:
        if not STABILITY_API_KEY:
            raise ValueError('STABILITY_API_KEY 환경변수가 설정되지 않았습니다.')

        style_preset = theme_name if theme_name else 'pixel-art'
        print(f'[AI] 테마={style_preset}, 배경={"선택됨" if bg_img is not None else "원본"}')

        ai_photos = []
        for i, (clean, mask) in enumerate(zip(clean_photos[:4], masks[:4])):
            print(f'[AI] 사진 {i+1}/4 처리 중...')

            # Step 1: 배경 교체 (bg_img 선택 시)
            base = clean.copy()
            if bg_img is not None:
                base = _remove_bg_composite(clean, bg_img)

            # Step 2: 마스크 영역 인페인팅 (드로잉 있을 때만)
            has_drawing = (cv2.countNonZero(mask) > 0)
            if has_drawing:
                out = _pixelart_inpaint_one(base, mask, style_preset=style_preset)
            else:
                out = base
            ai_photos.append(out)

        # save_final 과 동일한 4cut 콜라주 구성
        if _frame_raw is not None:
            fh, fw = _frame_raw.shape[:2]
            collage = np.ones((fh, fw, 3), dtype=np.uint8) * 255
            for i, (sx, sy, sw, sh) in enumerate(PHOTO_SLOTS):
                if i < len(ai_photos):
                    ph_s, pw_s = ai_photos[i].shape[:2]
                    pw = SAVE_PHOTO_W
                    ph = int(ph_s * (pw / pw_s))
                    resized = cv2.resize(ai_photos[i], (pw, ph))
                    ox  = sx + (sw - pw) // 2
                    oy  = sy + (sh - ph) // 2
                    oy2 = min(oy + ph, fh)
                    ox2 = min(ox + pw, fw)
                    collage[oy:oy2, ox:ox2] = resized[:oy2-oy, :ox2-ox]
            if _frame_raw.shape[2] == 4:
                alpha   = _frame_raw[:, :, 3:4] / 255.0
                collage = (_frame_raw[:, :, :3] * alpha + collage * (1 - alpha)).astype(np.uint8)
            else:
                collage = _frame_raw.copy()
        else:
            # 프레임 없으면 그냥 세로 스택
            slot_h, slot_w = ai_photos[0].shape[:2]
            collage = np.vstack([cv2.resize(p, (slot_w, slot_h)) for p in ai_photos])

        # 저장
        ai_path = os.path.join(session_dir, 'ai_4cut.jpg')
        cv2.imwrite(ai_path, collage)

        bucket['img']   = collage
        bucket['error'] = None
        print(f'[AI] 완료 → {ai_path}')

    except Exception as e:
        import traceback
        bucket['img']   = None
        bucket['error'] = str(e)
        print(f'[AI] 오류: {e}')
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════
#  상태 머신
# ══════════════════════════════════════════════════════════════════
STATE_INTRO       = 'intro'
STATE_SELECT_THEME = 'select_theme'
STATE_SELECT_BG   = 'select_bg'
STATE_WAITING     = 'waiting'
STATE_COUNTDOWN   = 'countdown'
STATE_FLASH       = 'flash'
STATE_REVIEW      = 'review'
STATE_RESULT      = 'result'
STATE_EMAIL_INPUT = 'email_input'

REVIEW_SEC = 10


# ══════════════════════════════════════════════════════════════════
#  이메일 전송 (백그라운드 스레드)
# ══════════════════════════════════════════════════════════════════
email_status     = None   # None | 'sending' | 'sent' | 'error'
email_input_text = ''

def _release_and_save(writer):
    """녹화 종료: writer 닫고 tmp → play 로 원자적 이동."""
    writer.release()
    try:
        if os.path.exists(_VID_PLAY):
            os.remove(_VID_PLAY)
        if os.path.exists(_VID_TMP):
            os.rename(_VID_TMP, _VID_PLAY)
            print("[녹화 저장 완료]")
    except Exception as e:
        print(f"[녹화 저장 실패] {e}")


def _send_email_async(attachment_path, recipient):
    global email_status
    print(f"[이메일] 전송 중 → {recipient}")
    result = gmail_send_message_with_attachment(attachment_path, recipient=recipient)
    if result:
        email_status = 'sent'
        print("[이메일] 전송 완료!")
    else:
        email_status = 'error'
        print("[이메일] 전송 실패")


# ══════════════════════════════════════════════════════════════════
#  선택 그리드 UI 함수
# ══════════════════════════════════════════════════════════════════
def _cell_hit(finger_x: int, finger_y: int, canvas_w: int, canvas_h: int, cells: list) -> int:
    """손가락 좌표가 cells 중 어느 셀에 있는지 반환. 없으면 -1."""
    src_w, src_h = 1024, 600
    sx = finger_x * src_w / canvas_w
    sy = finger_y * src_h / canvas_h
    for i, (x1, y1, x2, y2) in enumerate(cells):
        if x1 <= sx <= x2 and y1 <= sy <= y2:
            return i
    return -1


def _draw_selection_grid(canvas, src_img, cells, hovered_cell: int,
                         finger_x: int = -1, finger_y: int = -1) -> int:
    """src_img를 전체 화면에 표시하고 cells 기준으로 호버 테두리를 그린다."""
    ch, cw = canvas.shape[:2]
    if src_img is None:
        cv2.putText(canvas, "image not found", (50, ch // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
        return hovered_cell

    canvas[:] = cv2.resize(src_img, (cw, ch))

    hover = hovered_cell
    if finger_x >= 0 and finger_y >= 0:
        hit = _cell_hit(finger_x, finger_y, cw, ch, cells)
        if hit >= 0:
            hover = hit

    if 0 <= hover < len(cells):
        sx1, sy1, sx2, sy2 = cells[hover]
        x1 = int(sx1 * cw / 1024)
        y1 = int(sy1 * ch / 600)
        x2 = int(sx2 * cw / 1024)
        y2 = int(sy2 * ch / 600)
        cv2.rectangle(canvas, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (0, 0, 0), 6)
        cv2.rectangle(canvas, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (0, 220, 255), 3)

    if finger_x >= 0 and finger_y >= 0:
        cv2.circle(canvas, (finger_x, finger_y), 10, (0, 0, 0), 4)
        cv2.circle(canvas, (finger_x, finger_y), 10, (0, 220, 255), 2)
        cv2.circle(canvas, (finger_x, finger_y), 4, (0, 220, 255), -1)

    return hover


def _draw_theme_grid(canvas, hovered_cell, finger_x=-1, finger_y=-1):
    return _draw_selection_grid(canvas, _source_theme_img, SOURCE_THEME_CELLS,
                                hovered_cell, finger_x, finger_y)


def _draw_bg_grid(canvas, hovered_cell, finger_x=-1, finger_y=-1):
    return _draw_selection_grid(canvas, _source_bg_img, SOURCE_BG_CELLS,
                                hovered_cell, finger_x, finger_y)


# ══════════════════════════════════════════════════════════════════
#  메인 루프
# ══════════════════════════════════════════════════════════════════
print("=" * 55)
print("  인생네컷 포토부스  v9 + QR + Email + Theme/BG Select")
print("=" * 55)
print("  open           : 인트로 → 테마 선택 시작")
print("  peace          : 촬영 / 선택 확정 (홀드)")
print("  hand (기본)    : 팔레트 터치로 색 변경 / 커서")
print("  fist 0.2s 홀드 : PAINT ↔ ERASE 전환")
print("  thumb down     : DEFAULT 모드로 복귀")
print("  open 3s 홀드   : 리뷰/결과/이메일 화면에서 초기화")
print("  thumb up       : [결과 화면] 이메일 전송")
print("  ESC / q        : 종료")
print("=" * 55)

cv2.namedWindow('PhotoBooth', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('PhotoBooth', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

state           = STATE_INTRO
photos          = []
countdown_start = None
flash_start     = None
last_gesture    = None

draw_canvas    = None
prev_x, prev_y = None, None
color_idx      = 0
drawing_color  = PEN_COLORS[color_idx]
line_thickness = 5

draw_mode        = DRAW_DEFAULT
draw_delay_start = None   # 그리기/지우기 0.5초 딜레이
fist_start      = None
fist_toggled    = False
reset_start     = None
thumbdown_start = None
thumbdown_fired = False
victory_start   = None
victory_fired   = False

countdown_cooldown_until = 0.0   # 배경 선택 후 카운트다운 방지용

out_writer     = None
review_cap     = None
review_start   = None
result_collage = None
ai_collage     = None
ai_bucket      = {}    # {'img': ..., 'error': ...}
ai_thread_main = None
qr_img         = None
session_dir    = None
photos_clean   = []
draw_masks     = []

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

        if out_writer is None and state not in (STATE_INTRO, STATE_REVIEW, STATE_RESULT, STATE_EMAIL_INPUT, STATE_SELECT_THEME, STATE_SELECT_BG):
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
            prev_x, prev_y   = None, None
            draw_delay_start = None
            fist_start       = None
            fist_toggled    = False
            thumbdown_start = None
            thumbdown_fired = False
            victory_start   = None
            victory_fired   = False
            select_peace_start = None
        else:
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                raw     = result.gestures[i][0].category_name if result.gestures else 'None'
                gesture = _GESTURE_MAP.get(raw, None)
                ix      = int(hand_landmarks[8].x * cam_w)
                iy      = int(hand_landmarks[8].y * cam_h)

                if gesture == 'open':
                    draw_mode        = DRAW_DEFAULT
                    prev_x, prev_y   = None, None
                    draw_delay_start = None

                elif gesture == 'fist':
                    if last_gesture != 'fist':
                        fist_start   = now
                        fist_toggled = False
                    if fist_start is not None and not fist_toggled and now - fist_start >= HOLD_FIST:
                        if draw_mode in (DRAW_DEFAULT, DRAW_ERASE):
                            draw_mode = DRAW_PAINTING
                        else:
                            draw_mode = DRAW_ERASE
                        fist_toggled     = True
                        draw_delay_start = None
                    prev_x, prev_y = None, None

                elif gesture == 'thumbdown':
                    if last_gesture != 'thumbdown':
                        thumbdown_start = now
                        thumbdown_fired = False
                    if thumbdown_start is not None and not thumbdown_fired and now - thumbdown_start >= HOLD_CLEAR:
                        draw_canvas     = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
                        thumbdown_fired = True
                    prev_x, prev_y   = None, None
                    draw_delay_start = None

                elif gesture == 'peace':
                    if last_gesture != 'peace' or victory_start is None:
                        victory_start = now
                        victory_fired = False
                    if (victory_start is not None and not victory_fired
                            and now - victory_start >= HOLD_PHOTO
                            and now >= countdown_cooldown_until
                            and state == STATE_WAITING):
                        state           = STATE_COUNTDOWN
                        countdown_start = now
                        victory_fired   = True
                    prev_x, prev_y   = None, None
                    draw_delay_start = None

                elif gesture == 'thumbup':
                    if (state == STATE_RESULT and last_gesture != 'thumbup'
                            and session_dir is not None and email_status is None):
                        email_input_text = ''
                        state = STATE_EMAIL_INPUT
                    prev_x, prev_y   = None, None
                    draw_delay_start = None

                else:
                    if state != STATE_COUNTDOWN:
                        hit = _palette_hit(ix, iy, cam_h)
                        if hit >= 0:
                            color_idx        = hit
                            drawing_color    = PEN_COLORS[color_idx]
                            prev_x, prev_y   = None, None
                            draw_delay_start = None
                        elif draw_mode == DRAW_PAINTING:
                            if draw_delay_start is None:
                                draw_delay_start = now
                            if now - draw_delay_start >= 0.5:
                                if prev_x is not None:
                                    cv2.line(draw_canvas, (prev_x, prev_y), (ix, iy),
                                             drawing_color, line_thickness)
                                prev_x, prev_y = ix, iy
                        elif draw_mode == DRAW_ERASE:
                            if draw_delay_start is None:
                                draw_delay_start = now
                            if now - draw_delay_start >= 0.5:
                                cv2.circle(draw_canvas, (ix, iy), line_thickness * 4 + 1, BLACK, -1)
                            prev_x, prev_y = None, None
                        else:
                            prev_x, prev_y   = None, None
                            draw_delay_start = None
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


        # ── open → 인트로에서 테마 선택으로
        if gesture == 'open' and last_gesture != 'open' and state == STATE_INTRO:
            state = STATE_SELECT_THEME
            theme_hovered_cell = -1
            print("→ 테마 선택")

        # ── 테마 선택 제스처
        if state == STATE_SELECT_THEME and result is not None and result.hand_landmarks:
            hand = result.hand_landmarks[0]
            fx = int(hand[8].x * cam_w)
            fy = int(hand[8].y * cam_h)

            if _source_theme_img is not None:
                hit = _cell_hit(fx, fy, cam_w, cam_h, SOURCE_THEME_CELLS)
                if hit >= 0:
                    theme_hovered_cell = hit

            if gesture == 'peace':
                if select_peace_start is None:
                    select_peace_start = now
                elif now - select_peace_start >= HOLD_SELECT and theme_hovered_cell >= 0:
                    selected_theme_name = SOURCE_THEME_NAMES[theme_hovered_cell]
                    print(f"[테마 선택] 셀 {theme_hovered_cell} '{selected_theme_name}'")
                    state = STATE_SELECT_BG
                    bg_hovered_cell = -1
                    select_peace_start = None
                    select_peace_cooldown_until = now + THEME_TO_BG_COOLDOWN
                    countdown_cooldown_until = now + 1.5
                    victory_start = None
                    victory_fired = False
            else:
                select_peace_start = None

        # ── 배경 선택 제스처 (Pointing_Up 커서 + Victory 확정)
        if state == STATE_SELECT_BG and result is not None and result.hand_landmarks:
            hand = result.hand_landmarks[0]
            fx = int(hand[8].x * cam_w)
            fy = int(hand[8].y * cam_h)

            if _source_bg_img is not None:
                hit = _cell_hit(fx, fy, cam_w, cam_h, SOURCE_BG_CELLS)
                if hit >= 0:
                    bg_hovered_cell = hit

            # Victory → 홀드 후 선택 확정 (테마 전환 직후 쿨다운 중엔 무시)
            if gesture == 'peace' and now >= select_peace_cooldown_until:
                if select_peace_start is None:
                    select_peace_start = now
                elif now - select_peace_start >= HOLD_SELECT and bg_hovered_cell >= 0:
                    name = SOURCE_BG_NAMES[bg_hovered_cell]
                    bg_file = os.path.join(SOURCE_IMAGE_DIR, SOURCE_BG_FILES[bg_hovered_cell])
                    _loaded = cv2.imread(bg_file)
                    if _loaded is not None:
                        selected_bg_img = _loaded
                        print(f"[배경 선택] 셀 {bg_hovered_cell} '{name}'  {bg_file}  크기={_loaded.shape[1]}x{_loaded.shape[0]}")
                    else:
                        selected_bg_img = None
                        print(f"[배경 선택] 파일 로드 실패: {bg_file}")
                    state = STATE_WAITING
                    select_peace_start = None
                    countdown_cooldown_until = now + 1.5
                    victory_start = None
                    victory_fired = False
            else:
                select_peace_start = None

        # ── open 3초 홀드 → 리뷰/결과/이메일 초기화
        # gesture 또는 last_gesture 중 하나라도 open이면 타이머 유지 (1프레임 깜빡임 방지)
        _open_held = (gesture == 'open' or last_gesture == 'open') and gesture != 'thumbup'
        if _open_held and state in (STATE_REVIEW, STATE_RESULT, STATE_EMAIL_INPUT):
            if reset_start is None:
                reset_start = now
            elif now - reset_start >= HOLD_RESET:
                photos         = []
                photos_clean   = []
                draw_masks     = []
                result_collage = None
                ai_collage     = None
                ai_bucket      = {}
                ai_thread_main = None
                qr_img         = None
                review_start   = None
                email_status   = None
                if review_cap:
                    review_cap.release()
                    review_cap = None
                out_writer  = None
                draw_canvas = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
                for _p in (_VID_TMP, _VID_PLAY):
                    try:
                        if os.path.exists(_p):
                            os.remove(_p)
                    except Exception:
                        pass
                reset_start = None
                state       = STATE_INTRO
                print("초기화 완료")
        else:
            if gesture != 'open' and last_gesture != 'open':
                reset_start = None

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
                photos_clean.append(frame_clean.copy())
                draw_masks.append(msk.copy())
                draw_canvas = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
                state       = STATE_FLASH
                flash_start = now
                print(f"[{len(photos)}/{TOTAL_SHOTS}] 촬영!")
                if len(photos) >= TOTAL_SHOTS:
                    session_dir = os.path.join(SAVE_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
                    save_final(photos, session_dir, masks=draw_masks)
                    result_collage = cv2.imread(os.path.join(session_dir, "4cut.jpg"))
                    session_name   = os.path.basename(session_dir)
                    qr_url         = f"http://{LOCAL_IP}:{HTTP_PORT}/{session_name}/4cut.jpg"
                    qr_img         = _make_qr_cv(qr_url)
                    email_status   = None
                    print(f"[QR] {qr_url}")
                    # ── AI 4cut 자동 생성 (백그라운드)
                    if _INPAINTING_AVAILABLE:
                        ai_bucket = {}
                        ai_collage = None
                        ai_thread_main = threading.Thread(
                            target=_build_ai_4cut,
                            args=(list(photos_clean), list(draw_masks), session_dir,
                                  ai_bucket, selected_theme_name, selected_bg_img),
                            daemon=True,
                        )
                        ai_thread_main.start()
                        print(f'[AI] 생성 시작... 테마={selected_theme_name}, 배경={"있음" if selected_bg_img is not None else "없음"}')

        elif state == STATE_FLASH:
            if now - flash_start >= FLASH_SEC:
                if len(photos) >= TOTAL_SHOTS:
                    if out_writer and out_writer is not False:
                        _writer_to_release = out_writer
                        out_writer = None
                        print("[녹화 종료]")
                        threading.Thread(target=_release_and_save, args=(_writer_to_release,), daemon=True).start()
                    state        = STATE_REVIEW
                    review_start = now
                else:
                    state = STATE_WAITING

        elif state == STATE_REVIEW:
            if review_cap is None and os.path.exists(_VID_PLAY):
                review_cap = cv2.VideoCapture(_VID_PLAY)
            ai_done = ai_bucket.get('img') is not None or ai_bucket.get('error') is not None
            if ai_done:
                state = STATE_RESULT

        # STATE_RESULT/EMAIL에서도 복사가 늦게 끝난 경우 열기
        if state in (STATE_RESULT, STATE_EMAIL_INPUT):
            if review_cap is None and os.path.exists(_VID_PLAY):
                review_cap = cv2.VideoCapture(_VID_PLAY)

        # ── 그리기 합성
        if state not in (STATE_REVIEW, STATE_RESULT, STATE_EMAIL_INPUT, STATE_SELECT_THEME, STATE_SELECT_BG):
            gray_m  = cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
            _, msk  = cv2.threshold(gray_m, 1, 255, cv2.THRESH_BINARY)
            msk_inv = cv2.bitwise_not(msk)
            frame   = cv2.add(cv2.bitwise_and(frame, frame, mask=msk_inv),
                              cv2.bitwise_and(draw_canvas, draw_canvas, mask=msk))

        # ── 녹화 (그리기 합성 후 → 그림 포함, 팔레트/UI 제외)
        if out_writer and out_writer is not False and state not in (STATE_REVIEW, STATE_RESULT, STATE_EMAIL_INPUT, STATE_SELECT_THEME, STATE_SELECT_BG):
            out_writer.write(frame)

        # ── 캔버스 합성
        canvas = _bg_resized.copy()

        if state == STATE_INTRO:
            if _intro_raw is not None:
                canvas = cv2.resize(_intro_raw, (canvas.shape[1], canvas.shape[0]))
            else:
                cv2.putText(canvas, "Open Palm to Start", (100, canvas.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, BLACK, 2, cv2.LINE_AA)

        elif state == STATE_SELECT_THEME:
            finger_x, finger_y = -1, -1
            if result is not None and result.hand_landmarks:
                hand = result.hand_landmarks[0]
                finger_x = int(hand[8].x * canvas.shape[1])
                finger_y = int(hand[8].y * canvas.shape[0])
            theme_hovered_cell = _draw_theme_grid(canvas, theme_hovered_cell, finger_x, finger_y)
            if select_peace_start is not None and finger_x >= 0:
                prog = min((now - select_peace_start) / HOLD_SELECT, 1.0)
                angle = int(-360 * prog)
                cv2.ellipse(canvas, (finger_x, finger_y), (22, 22), -90, 0, angle, (0, 220, 255), 3)

        elif state == STATE_SELECT_BG:
            finger_x, finger_y = -1, -1
            if result is not None and result.hand_landmarks:
                hand = result.hand_landmarks[0]
                finger_x = int(hand[8].x * canvas.shape[1])
                finger_y = int(hand[8].y * canvas.shape[0])
            bg_hovered_cell = _draw_bg_grid(canvas, bg_hovered_cell, finger_x, finger_y)
            if select_peace_start is not None and finger_x >= 0:
                prog = min((now - select_peace_start) / HOLD_SELECT, 1.0)
                angle = int(-360 * prog)
                cv2.ellipse(canvas, (finger_x, finger_y), (22, 22), -90, 0, angle, (0, 220, 255), 3)

        elif state in (STATE_RESULT, STATE_EMAIL_INPUT):
            # ── AI 결과 수신 체크
            if ai_collage is None and ai_bucket.get('img') is not None:
                ai_collage = ai_bucket['img']
                print("[AI] 픽셀아트 4cut 완료 — 화면 반영")
            if ai_collage is None and ai_bucket.get('error'):
                print(f"[AI] 오류: {ai_bucket['error']}")
                ai_bucket = {}  # 재출력 방지

            # ── 결과 페이지: background.png + 콜라주(원본 + AI) + 좌상단 리플레이 + 우상단 QR
            if _bg_result_raw is not None:
                canvas = _bg_result_raw.copy()

            cy0       = 20
            PAD       = 14
            LABEL_H   = 20
            cw_total  = canvas.shape[1]
            content_h = canvas.shape[0] - cy0 - 20
            left_w    = cw_total * 6 // 10 - PAD   # 60% 좌측
            half_w    = left_w
            lx        = PAD

            # ────────────────────────────────
            # 좌측 절반: 위(1) = 리플레이 + QR  /  아래(3) = 라이브
            # ────────────────────────────────
            TOP_H = content_h // 4
            BOT_H = content_h - TOP_H - PAD

            # 리플레이 + QR: TOP_H 안에 라벨까지 들어오도록 크기 제한
            top_sz   = TOP_H - LABEL_H - 8   # 위아래 여백 포함
            replay_h = top_sz
            replay_w = replay_h * 4 // 3
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
                              (replay_x + replay_w, replay_y + replay_h), GRAY, 1)
            (rt, _), _ = cv2.getTextSize("Replay", cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(canvas, "Replay",
                        (replay_x + (replay_w - rt) // 2, replay_y + replay_h + LABEL_H - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, BLACK, 1, cv2.LINE_AA)

            # QR (리플레이 오른쪽, 세로 정렬)
            qr_x = top_start_x + replay_w + PAD
            qr_y = cy0 + 4
            if qr_img is not None:
                qr_resized = cv2.resize(qr_img, (top_sz, top_sz))
                canvas[qr_y:qr_y + top_sz, qr_x:qr_x + top_sz] = qr_resized
                cv2.rectangle(canvas, (qr_x, qr_y), (qr_x + top_sz, qr_y + top_sz), GRAY, 1)
                (qt, _), _ = cv2.getTextSize("QR scan to save", cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                cv2.putText(canvas, "QR scan to save",
                            (qr_x + (top_sz - qt) // 2, qr_y + top_sz + LABEL_H - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, BLACK, 1, cv2.LINE_AA)

            # 라이브 카메라 (좌측 하단, 높이 기준으로 크기 결정 → 자연스럽게 중앙 정렬)
            live_y = cy0 + TOP_H + PAD
            live_h = BOT_H - PAD - LABEL_H   # PAD 만큼 줄여서 오른쪽 라벨과 높이 맞춤
            live_w = live_h * 4 // 3
            if live_w > half_w:
                live_w = half_w
                live_h = live_w * 3 // 4
            live_x = lx + (half_w - live_w) // 2
            canvas[live_y:live_y + live_h, live_x:live_x + live_w] = cv2.resize(frame, (live_w, live_h))

            # 보자기 홀드 진행 바
            if reset_start is not None:
                progress = min((now - reset_start) / HOLD_RESET, 1.0)
                bar_y = live_y + live_h + 1
                cv2.rectangle(canvas, (live_x, bar_y), (live_x + live_w, bar_y + 7), (180, 180, 180), -1)
                cv2.rectangle(canvas, (live_x, bar_y), (live_x + int(live_w * progress), bar_y + 7), WHITE, -1)

            if email_status == 'sending':
                email_msg = "Sending email..."
            elif email_status == 'sent':
                email_msg = "Email sent!"
            elif email_status == 'error':
                email_msg = "Email failed"
            else:
                email_msg = "Thumbs up to send email"
            (et, _), _ = cv2.getTextSize(email_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(canvas, email_msg,
                        (live_x + (live_w - et) // 2, live_y + live_h + LABEL_H),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, BLACK, 1, cv2.LINE_AA)

            # ────────────────────────────────
            # 우측 절반: 이미지 두 장 (자연 비율 유지, 가로 중앙 정렬)
            # ────────────────────────────────
            rx = cw_total * 6 // 10 + PAD
            right_w = cw_total - rx - PAD
            if result_collage is not None:
                img_gap  = 10
                n_imgs   = 2 if ai_collage is not None else 1
                max_each_h = content_h - LABEL_H - PAD * 2
                each_h   = min(max_each_h, result_collage.shape[0])
                each_w   = int(each_h * result_collage.shape[1] / result_collage.shape[0])
                total_w  = each_w * n_imgs + img_gap * (n_imgs - 1)
                img_x0   = rx + max(0, (right_w - total_w) // 2)
                img_y    = cy0 + PAD + (max_each_h - each_h) // 2

                # 원본
                canvas[img_y:img_y + each_h, img_x0:img_x0 + each_w] = \
                    cv2.resize(result_collage, (each_w, each_h))
                (ot, _), _ = cv2.getTextSize("Original", cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                cv2.putText(canvas, "Original",
                            (img_x0 + (each_w - ot) // 2, img_y + each_h + LABEL_H),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, BLACK, 1, cv2.LINE_AA)

                # AI 픽셀아트
                if ai_collage is not None:
                    ax = img_x0 + each_w + img_gap
                    canvas[img_y:img_y + each_h, ax:ax + each_w] = \
                        cv2.resize(ai_collage, (each_w, each_h))
                    (pt, _), _ = cv2.getTextSize("AI Generated", cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                    cv2.putText(canvas, "AI Generated",
                                (ax + (each_w - pt) // 2, img_y + each_h + LABEL_H),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, BLACK, 1, cv2.LINE_AA)


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

            # AI 생성 중 깜빡이는 텍스트 (REPLAY 와 동일 스타일)
            if ai_thread_main is not None and ai_thread_main.is_alive():
                if int(time.time() * 2) % 2 == 0:
                    dots = "." * (int(time.time() * 1.5) % 4)
                    gen_text = f"GENERATING{dots}"
                    (gw, _), _ = cv2.getTextSize(gen_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 3)
                    gx = CAM_X + (CAM_W - gw) // 2
                    gy = CAM_Y + CAM_H - 18
                    cv2.putText(canvas, gen_text, (gx, gy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 3, cv2.LINE_AA)
                    cv2.putText(canvas, gen_text, (gx, gy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

        else:
            _render_frame(canvas, photos)
            canvas[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W] = cv2.resize(frame, (CAM_W, CAM_H))
            _draw_color_palette(canvas[CAM_Y:CAM_Y + CAM_H, CAM_X:CAM_X + CAM_W], color_idx)

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

        if state == STATE_EMAIL_INPUT:
            overlay = canvas.copy()
            bx = canvas.shape[1] // 2 - 280
            by = canvas.shape[0] // 2 - 60
            cv2.rectangle(overlay, (bx, by), (bx + 560, by + 120), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)
            cv2.rectangle(canvas, (bx, by), (bx + 560, by + 120), WHITE, 2)
            cv2.putText(canvas, "Enter email address:", (bx + 16, by + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
            cursor = '|' if int(time.time() * 2) % 2 == 0 else ' '
            cv2.putText(canvas, email_input_text + cursor, (bx + 16, by + 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Enter: Send   ESC: Cancel", (bx + 16, by + 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1, cv2.LINE_AA)

        cv2.imshow('PhotoBooth', canvas)
        key = cv2.waitKey(1)

        if state == STATE_EMAIL_INPUT:
            if key in (13, 10):   # Enter
                if email_input_text.strip():
                    attachments = [os.path.join(session_dir, "4cut.jpg")]
                    ai_path = os.path.join(session_dir, "ai_4cut.jpg")
                    if os.path.exists(ai_path):
                        attachments.append(ai_path)
                    email_status = 'sending'
                    threading.Thread(target=_send_email_async,
                                     args=(attachments, email_input_text.strip()),
                                     daemon=True).start()
                state = STATE_RESULT
            elif key == 27:   # ESC
                state = STATE_RESULT
            elif key in (8, 127):   # Backspace
                email_input_text = email_input_text[:-1]
            elif key >= 0:
                key_ascii = key & 0xFF
                if 32 <= key_ascii <= 126:  # printable ASCII (@=64, etc.)
                    email_input_text += chr(key_ascii)
        elif key & 0xFF in (27, ord('q')):
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
