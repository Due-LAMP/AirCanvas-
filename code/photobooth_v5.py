import cv2
import mediapipe as mp
import os
import numpy as np
import time
import signal
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
#  상수 / 레이아웃  (1280 x 720 기준)
# ══════════════════════════════════════════════════════════════════
TOTAL_SHOTS   = 4
COUNTDOWN_SEC = 3
FLASH_SEC     = 0.5
GESTURE_HOLD  = 0.8
FPS           = 30

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── 배경 PNG ──────────────────────────────────────────────────
BG_IMAGE_PATH = os.path.join(_BASE_DIR, 'background/background.png')

# ─── 카메라 영역 ───────────────────────────────────────────────
CAM_X, CAM_Y = 20,  20
CAM_W, CAM_H = 840, 680

# ─── 네컷 사진 슬롯 (배경 PNG 틀 좌표에 맞게 조정) ────────────
PHOTO_W, PHOTO_H = 360, 155

PHOTO_SLOTS = [
    (880,  20),   # 1번 사진 (x, y)
    (880, 195),   # 2번 사진
    (880, 370),   # 3번 사진
    (880, 545),   # 4번 사진
]

# ─── 컬러 ──────────────────────────────────────────────────────
WHITE    = (255, 255, 255)
BLACK    = (0,   0,   0)
GRAY     = (180, 180, 180)
ACCENT   = (100,  50, 255)
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
SAVE_DIR = os.path.join(_BASE_DIR, 'photobooth_output')
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
#  배경 이미지 로드
# ══════════════════════════════════════════════════════════════════
_bg_raw     = cv2.imread(BG_IMAGE_PATH)
if _bg_raw is None:
    print(f"[경고] 배경 이미지 없음: {BG_IMAGE_PATH} → 단색 배경 사용")
_bg_resized = None


# ══════════════════════════════════════════════════════════════════
#  사진 저장
# ══════════════════════════════════════════════════════════════════
def save_final(photos, session_dir):
    os.makedirs(session_dir, exist_ok=True)
    for i, photo in enumerate(photos):
        cv2.imwrite(os.path.join(session_dir, f"shot_{i+1}.jpg"), photo)
        print(f"  저장: shot_{i+1}.jpg")
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
print("=" * 50)
print("  인생네컷 포토부스")
print("=" * 50)
print("  peace (0.8s) : 촬영")
print("  손 (기본)    : 그리기")
print("  fist         : 펜 색상 변경")
print("  open         : 캔버스 지우기 / 리뷰 시 초기화")
print("  ESC / q      : 종료")
print("=" * 50)

cv2.namedWindow('PhotoBooth', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('PhotoBooth', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 상태 변수
state           = STATE_WAITING
photos          = []
countdown_start = None
flash_start     = None
last_gesture    = None
gesture_start   = None

# 그리기
draw_canvas    = None
prev_x, prev_y = None, None
color_idx      = 0
drawing_color  = PEN_COLORS[color_idx]
line_thickness = 5

with GestureRecognizer.create_from_options(_mp_options) as recognizer:
    frame_index = 0

    while True:
        ret, frame = video.read()
        if not ret or _exit_requested:
            break

        frame        = cv2.flip(frame, 1)
        cam_h, cam_w = frame.shape[:2]
        now          = time.time()

        # ── 첫 프레임 초기화
        if draw_canvas is None:
            draw_canvas = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

        if _bg_resized is None:
            if _bg_raw is not None:
                _bg_resized = _bg_raw.copy()
            else:
                total_w = max(CAM_X + CAM_W, max(sx + PHOTO_W for sx, sy in PHOTO_SLOTS)) + 20
                total_h = max(CAM_Y + CAM_H, max(sy + PHOTO_H for sx, sy in PHOTO_SLOTS)) + 20
                _bg_resized = np.full((total_h, total_w, 3), BG_COLOR, dtype=np.uint8)

        # ── 손 인식
        gesture = None
        result  = None
        if state != STATE_REVIEW:
            img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result   = recognizer.recognize_for_video(mp_image, int(frame_index * 1000 / FPS))
        frame_index += 1

        if result is None or not result.hand_landmarks:
            prev_x, prev_y = None, None
        else:
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                raw     = result.gestures[i][0].category_name if result.gestures else 'None'
                gesture = _GESTURE_MAP.get(raw, None)
                ix      = int(hand_landmarks[8].x * cam_w)
                iy      = int(hand_landmarks[8].y * cam_h)

                if gesture == 'fist':
                    if last_gesture != 'fist':
                        color_idx     = (color_idx + 1) % len(PEN_COLORS)
                        drawing_color = PEN_COLORS[color_idx]
                    prev_x, prev_y = None, None

                elif gesture == 'open':
                    draw_canvas    = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
                    prev_x, prev_y = None, None

                elif gesture == 'peace':
                    prev_x, prev_y = None, None

                else:
                    if state != STATE_COUNTDOWN:
                        if prev_x is not None:
                            cv2.line(draw_canvas, (prev_x, prev_y), (ix, iy),
                                     drawing_color, line_thickness)
                        prev_x, prev_y = ix, iy
                    else:
                        prev_x, prev_y = None, None

                # 커서
                cv2.circle(frame, (ix, iy), line_thickness + 4, drawing_color, 2)
                cv2.circle(frame, (ix, iy), 3, drawing_color, -1)

        # ── peace 유지 → 촬영 트리거
        if gesture == 'peace':
            if last_gesture != 'peace':
                gesture_start = now
            elif now - gesture_start >= GESTURE_HOLD and state == STATE_WAITING:
                state           = STATE_COUNTDOWN
                countdown_start = now
                gesture_start   = now
        else:
            gesture_start = None

        # ── open → 초기화
        if gesture == 'open' and last_gesture != 'open':
            if state == STATE_REVIEW:
                photos      = []
                draw_canvas = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
                state       = STATE_WAITING
                print("초기화 완료")
            elif state == STATE_WAITING:
                draw_canvas = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

        last_gesture = gesture

        # ── 상태 전환
        if state == STATE_COUNTDOWN:
            if now - countdown_start >= COUNTDOWN_SEC:
                gray_m  = cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
                _, msk  = cv2.threshold(gray_m, 1, 255, cv2.THRESH_BINARY)
                msk_inv = cv2.bitwise_not(msk)
                shot    = cv2.add(cv2.bitwise_and(frame, frame, mask=msk_inv),
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
                state = STATE_REVIEW if len(photos) >= TOTAL_SHOTS else STATE_WAITING

        # ── 그리기 캔버스 합성
        gray_m  = cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
        _, msk  = cv2.threshold(gray_m, 1, 255, cv2.THRESH_BINARY)
        msk_inv = cv2.bitwise_not(msk)
        frame   = cv2.add(cv2.bitwise_and(frame, frame, mask=msk_inv),
                          cv2.bitwise_and(draw_canvas, draw_canvas, mask=msk))

        # ══════════════════════════════════════════════════════════
        #  캔버스 합성: 배경 + 카메라 + 사진
        # ══════════════════════════════════════════════════════════
        canvas = _bg_resized.copy()
        total_w = canvas.shape[1]
        total_h = canvas.shape[0]

        # 카메라 영상
        canvas[CAM_Y:CAM_Y+CAM_H, CAM_X:CAM_X+CAM_W] = cv2.resize(frame, (CAM_W, CAM_H))

        # 촬영된 사진 순서대로 슬롯에 붙이기
        for i, (sx, sy) in enumerate(PHOTO_SLOTS):
            if i < len(photos):
                canvas[sy:sy+PHOTO_H, sx:sx+PHOTO_W] = cv2.resize(photos[i], (PHOTO_W, PHOTO_H))

        # ── 카운트다운
        if state == STATE_COUNTDOWN:
            num_show = max(1, COUNTDOWN_SEC - int(now - countdown_start))
            cx = CAM_X + CAM_W // 2
            cy = CAM_Y + CAM_H // 2
            ov = canvas.copy()
            cv2.circle(ov, (cx, cy), 100, BLACK, -1)
            cv2.addWeighted(ov, 0.5, canvas, 0.5, 0, canvas)
            tw = cv2.getTextSize(str(num_show), cv2.FONT_HERSHEY_DUPLEX, 5.0, 10)[0][0]
            cv2.putText(canvas, str(num_show), (cx - tw//2, cy + 35),
                        cv2.FONT_HERSHEY_DUPLEX, 5.0, WHITE, 10, cv2.LINE_AA)

        # ── 플래시
        elif state == STATE_FLASH:
            ratio = 1.0 - (now - flash_start) / FLASH_SEC
            wh    = np.full_like(canvas, 255)
            cv2.addWeighted(wh, ratio * 0.9, canvas, 1 - ratio * 0.9, 0, canvas)

        # ── 리뷰 안내
        elif state == STATE_REVIEW:
            hint = "Open palm : New Session  |  ESC : Quit"
            hw   = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
            cv2.putText(canvas, hint, (total_w//2 - hw//2, total_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRAY, 1, cv2.LINE_AA)

        # ── 대기 안내
        elif state == STATE_WAITING:
            if gesture == 'peace' and gesture_start is not None:
                ratio = min((now - gesture_start) / GESTURE_HOLD, 1.0)
                bx1   = CAM_X + 10
                bx2   = CAM_X + CAM_W - 10
                by    = CAM_Y + CAM_H - 10
                cv2.rectangle(canvas, (bx1, by-8), (bx2, by+8), (60, 60, 60), -1)
                cv2.rectangle(canvas, (bx1, by-8),
                              (bx1 + int((bx2-bx1)*ratio), by+8), ACCENT, -1)
                cv2.rectangle(canvas, (bx1, by-8), (bx2, by+8), WHITE, 1)
            else:
                hint = "PEACE:shoot  HAND:draw  FIST:color  OPEN:clear"
                hw   = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
                cv2.putText(canvas, hint,
                            (CAM_X + CAM_W//2 - hw//2, CAM_Y + CAM_H - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1, cv2.LINE_AA)

        cv2.imshow('PhotoBooth', canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break


# ══════════════════════════════════════════════════════════════════
#  종료 정리
# ══════════════════════════════════════════════════════════════════
if video:
    video.release()
cv2.destroyAllWindows()
print("종료.")