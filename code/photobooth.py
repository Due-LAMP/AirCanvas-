import cv2
import mediapipe as mp
import os
import numpy as np
import time
from datetime import datetime

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

# ── Mediapipe 설정 ──────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=RunningMode.VIDEO,
)

# ── 상수 ────────────────────────────────────────────────────────
TOTAL_SHOTS    = 4          # 총 촬영 장수
COUNTDOWN_SEC  = 3          # 카운트다운 초
FLASH_SEC      = 0.5        # 촬영 후 플래시 효과 시간
STRIP_W        = 300        # 오른쪽 스트립 폭
SLOT_MARGIN    = 12         # 슬롯 간격
BORDER_COLOR   = (200, 180, 255)   # 라벤더
FLASH_COLOR    = (255, 255, 255)
BG_COLOR       = (245, 235, 255)   # 연보라 배경
TEXT_COLOR     = (80,  20, 120)
ACCENT_COLOR   = (180, 100, 255)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'photobooth_output')
os.makedirs(SAVE_DIR, exist_ok=True)


# ── 손가락 인식 ─────────────────────────────────────────────────
def get_finger_status(landmarks):
    fingers = []
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for tip, pip_j in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        fingers.append(1 if landmarks[tip].y < landmarks[pip_j].y else 0)
    return fingers


def recognize_gesture(fingers_status):
    if fingers_status == [0, 0, 0, 0, 0]:
        return 'fist'
    elif fingers_status == [0, 1, 0, 0, 0]:
        return 'point'
    elif fingers_status == [1, 1, 1, 1, 1]:
        return 'open'
    elif fingers_status == [0, 1, 1, 0, 0]:
        return 'peace'
    return None


# ── 카메라 초기화 ────────────────────────────────────────────────
picam2 = None
video  = None
use_picamera = False

for idx in range(0, 20):
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None and len(test_frame.shape) == 3:
            print(f"USB 카메라 발견: /dev/video{idx}")
            video = cap
            break
    cap.release()

if video is None and HAS_PICAMERA2:
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        use_picamera = True
        print("Picamera2 (CSI 카메라) 사용")
    except Exception as e:
        print(f"Picamera2 실패: {e}")
        picam2 = None

if not use_picamera and video is None:
    print("카메라를 찾을 수 없습니다.")
    exit(1)


# ── 유틸리티 ─────────────────────────────────────────────────────
def put_kr_text(img, text, pos, size=1.0, color=TEXT_COLOR, thickness=2):
    """간단 영문/숫자 텍스트 렌더링 (한글은 사각형으로 대체)"""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, size, color, thickness, cv2.LINE_AA)


def draw_rounded_rect(img, pt1, pt2, color, radius=15, thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius
    if thickness == -1:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
        cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y1), color, thickness)
        cv2.rectangle(img, (x1+r, y2), (x2-r, y2), color, thickness)
        cv2.rectangle(img, (x1, y1+r), (x1, y2-r), color, thickness)
        cv2.rectangle(img, (x2, y1+r), (x2, y2-r), color, thickness)
        for cx, cy, a, b in [(x1+r, y1+r, 180, 270), (x2-r, y1+r, 270, 360),
                              (x1+r, y2-r, 90, 180), (x2-r, y2-r, 0, 90)]:
            cv2.ellipse(img, (cx, cy), (r, r), 0, a, b, color, thickness)


def make_strip(photos, strip_w, cam_h):
    """4장 사진을 세로 스트립으로 합성"""
    n = TOTAL_SHOTS
    margin = SLOT_MARGIN
    slot_h = (cam_h - margin * (n + 1)) // n
    slot_w = strip_w - margin * 2

    strip = np.full((cam_h, strip_w, 3), BG_COLOR, dtype=np.uint8)

    for i in range(n):
        y0 = margin + i * (slot_h + margin)
        x0 = margin
        if i < len(photos) and photos[i] is not None:
            thumb = cv2.resize(photos[i], (slot_w, slot_h))
            strip[y0:y0+slot_h, x0:x0+slot_w] = thumb
            draw_rounded_rect(strip, (x0-2, y0-2), (x0+slot_w+2, y0+slot_h+2),
                               BORDER_COLOR, radius=8, thickness=3)
        else:
            draw_rounded_rect(strip, (x0, y0), (x0+slot_w, y0+slot_h),
                               (220, 210, 240), radius=8, thickness=-1)
            num_text = str(i + 1)
            tw, th = cv2.getTextSize(num_text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 2)[0]
            cv2.putText(strip, num_text,
                        (x0 + slot_w//2 - tw//2, y0 + slot_h//2 + th//2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (200, 180, 230), 2, cv2.LINE_AA)
    return strip


def make_final_collage(photos):
    """인생네컷 스타일 세로 스트립 고해상도 합성"""
    if not photos:
        return None
    ph, pw = photos[0].shape[:2]

    # 레이아웃 설정
    margin     = 30
    top_pad    = 80   # 상단 타이틀 공간
    bottom_pad = 60   # 하단 날짜 공간
    col_w = pw + margin * 2
    col_h = top_pad + (ph + margin) * TOTAL_SHOTS + margin + bottom_pad

    collage = np.full((col_h, col_w, 3), BG_COLOR, dtype=np.uint8)

    # 타이틀
    title = "4-CUT PHOTO"
    tw = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0][0]
    cv2.putText(collage, title,
                (col_w // 2 - tw // 2, 55),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, ACCENT_COLOR, 2, cv2.LINE_AA)

    # 사진 4장 배치
    for i, photo in enumerate(photos):
        y0 = top_pad + margin + i * (ph + margin)
        x0 = margin
        collage[y0:y0+ph, x0:x0+pw] = photo
        # 라벤더 테두리
        draw_rounded_rect(collage, (x0-4, y0-4), (x0+pw+4, y0+ph+4),
                          BORDER_COLOR, radius=10, thickness=4)

    # 하단 날짜
    date_str = datetime.now().strftime("%Y.%m.%d")
    dw = cv2.getTextSize(date_str, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)[0][0]
    cv2.putText(collage, date_str,
                (col_w // 2 - dw // 2, col_h - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, TEXT_COLOR, 1, cv2.LINE_AA)
    return collage


def save_final(photos, cam_h, strip_w):
    """개별 사진 4장 + 인생네컷 합성 이미지 저장"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(SAVE_DIR, ts)
    os.makedirs(session_dir, exist_ok=True)

    # 개별 사진 저장
    for i, photo in enumerate(photos):
        path = os.path.join(session_dir, f"shot_{i+1}.jpg")
        cv2.imwrite(path, photo)
        print(f"  저장: {path}")

    # 인생네컷 합성 이미지 저장
    collage = make_final_collage(photos)
    if collage is not None:
        collage_path = os.path.join(session_dir, "4cut.jpg")
        cv2.imwrite(collage_path, collage)
        print(f"  합성 저장: {collage_path}")

    print(f"✓ 저장 완료 → {session_dir}")
    return session_dir


# ── 상태 머신 ────────────────────────────────────────────────────
STATE_WAITING    = 'waiting'    # 다음 촬영 대기
STATE_COUNTDOWN  = 'countdown'  # 카운트다운 진행 중
STATE_FLASH      = 'flash'      # 촬영 직후 플래시
STATE_DONE       = 'done'       # 4장 완료

state          = STATE_WAITING
photos         = []             # 촬영된 사진 리스트
countdown_start = None
flash_start    = None
last_gesture   = None
gesture_start  = None
GESTURE_HOLD   = 0.8            # peace 제스처 유지 시간(초)

fps = 30

# ── 그리기 캔버스 ──────────────────────────────────────────────
draw_canvas  = None
prev_x, prev_y = None, None
drawing_color  = (0, 0, 255)
line_thickness = 3

print("=" * 50)
print("  인생네컷 포토부스")
print("=" * 50)
print("  ✌ peace  : 사진 촬영 (0.8초 유지)")
print("  ☝ point  : 그리기")
print("  ✋ open   : 그림 지우기")
print("  ✊ fist   : 처음부터 다시")
print("  ESC      : 종료")
print("=" * 50)

cv2.namedWindow('PhotoBooth', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('PhotoBooth', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with HandLandmarker.create_from_options(options) as landmarker:
    frame_index = 0

    while True:
        # ── 프레임 읽기
        if use_picamera:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = video.read()
            if not ret:
                break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.time()

        # 캔버스 초기화 (첫 프레임)
        if draw_canvas is None:
            draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # ── 손 인식
        img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        ts_ms    = int(frame_index * 1000 / fps)
        frame_index += 1
        result = landmarker.detect_for_video(mp_image, ts_ms)

        gesture = None
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                fingers_status = get_finger_status(hand_landmarks)
                gesture = recognize_gesture(fingers_status)

                # 검지 끝(landmark 8) 좌표
                ix = int(hand_landmarks[8].x * w)
                iy = int(hand_landmarks[8].y * h)

                # 그리기 제스처 처리
                if gesture in ('point', 'standby'):
                    if prev_x is not None and prev_y is not None:
                        cv2.line(draw_canvas, (prev_x, prev_y), (ix, iy), drawing_color, line_thickness)
                    prev_x, prev_y = ix, iy
                elif gesture == 'open':
                    draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    prev_x, prev_y = None, None
                else:
                    prev_x, prev_y = None, None

                # 검지 위치 원 표시
                cv2.circle(frame, (ix, iy), 8, (0, 255, 255), -1)

                # 랜드마크 그리기
                for connection in HAND_CONNECTIONS:
                    s = hand_landmarks[connection[0]]
                    e = hand_landmarks[connection[1]]
                    cv2.line(frame,
                             (int(s.x*w), int(s.y*h)),
                             (int(e.x*w), int(e.y*h)),
                             (180, 255, 180), 2)
                for lm in hand_landmarks:
                    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (255,180,180), -1)

        # ── 제스처 유지 타이머 (peace → 촬영 트리거)
        if gesture == 'peace':
            if last_gesture != 'peace':
                gesture_start = now
            elif now - gesture_start >= GESTURE_HOLD:
                if state == STATE_WAITING:
                    state = STATE_COUNTDOWN
                    countdown_start = now
                gesture_start = now  # 재촉발 방지
        else:
            gesture_start = None

        if gesture != 'point' and gesture != 'standby' and not (gesture == 'peace'):
            prev_x, prev_y = None, None

        if gesture == 'fist' and state in (STATE_WAITING, STATE_DONE):
            photos = []
            draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)
            state = STATE_WAITING
            print("초기화 완료")

        last_gesture = gesture

        # ── 상태 처리
        if state == STATE_COUNTDOWN:
            elapsed  = now - countdown_start
            remaining = COUNTDOWN_SEC - elapsed
            if remaining <= 0:
                # 그리기 캔버스를 프레임에 합성 후 촬영
                mask = cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                canvas_fg = cv2.bitwise_and(draw_canvas, draw_canvas, mask=mask)
                shot_frame = cv2.add(frame_bg, canvas_fg)
                photos.append(shot_frame.copy())
                state = STATE_FLASH
                flash_start = now
                draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)  # 촬영 후 캔버스 초기화
                print(f"[{len(photos)}/{TOTAL_SHOTS}] 촬영!")
                if len(photos) >= TOTAL_SHOTS:
                    save_final(photos, h, STRIP_W)

        elif state == STATE_FLASH:
            if now - flash_start >= FLASH_SEC:
                state = STATE_DONE if len(photos) >= TOTAL_SHOTS else STATE_WAITING

        # ── 그리기 캔버스를 카메라 프레임에 합성
        mask = cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(draw_canvas, draw_canvas, mask=mask)
        frame = cv2.add(frame_bg, canvas_fg)

        # ── UI 합성
        # 오른쪽 스트립
        strip = make_strip(photos, STRIP_W, h)

        # 캔버스: cam + strip
        canvas = np.full((h, w + STRIP_W, 3), BG_COLOR, dtype=np.uint8)
        canvas[:, :w] = frame
        canvas[:, w:] = strip

        # ── 오버레이: 카운트다운
        if state == STATE_COUNTDOWN:
            elapsed   = now - countdown_start
            remaining = int(COUNTDOWN_SEC - elapsed) + 1
            remaining = max(1, remaining)
            # 반투명 원
            overlay = canvas.copy()
            cx, cy = w // 2, h // 2
            cv2.circle(overlay, (cx, cy), 90, ACCENT_COLOR, -1)
            cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
            tw = cv2.getTextSize(str(remaining), cv2.FONT_HERSHEY_DUPLEX, 5, 8)[0][0]
            cv2.putText(canvas, str(remaining),
                        (cx - tw//2, cy + 35),
                        cv2.FONT_HERSHEY_DUPLEX, 5, (255,255,255), 8, cv2.LINE_AA)

        # ── 오버레이: 플래시
        elif state == STATE_FLASH:
            ratio = 1.0 - (now - flash_start) / FLASH_SEC
            overlay = np.full_like(canvas, 255)
            cv2.addWeighted(overlay, ratio * 0.8, canvas, 1 - ratio * 0.8, 0, canvas)

        # ── 오버레이: 완료 화면
        elif state == STATE_DONE:
            msg1 = "DONE!  Fist to Retry"
            tw = cv2.getTextSize(msg1, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)[0][0]
            draw_rounded_rect(canvas,
                              (w//2 - tw//2 - 20, h - 55),
                              (w//2 + tw//2 + 20, h - 15),
                              (60, 20, 100), radius=10, thickness=-1)
            cv2.putText(canvas, msg1,
                        (w//2 - tw//2, h - 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 230, 255), 2, cv2.LINE_AA)

        # ── 오버레이: 대기 안내
        elif state == STATE_WAITING:
            shot_num = len(photos) + 1
            msg = f"Shot {shot_num}/{TOTAL_SHOTS}  -  Peace:Capture  Point:Draw  Open:Clear"
            if gesture == 'peace' and gesture_start:
                held = now - gesture_start
                bar_w = int(min(held / GESTURE_HOLD, 1.0) * (w - 40))
                cv2.rectangle(canvas, (20, h-20), (w-20, h-10), (200,180,255), -1)
                cv2.rectangle(canvas, (20, h-20), (20+bar_w, h-10), ACCENT_COLOR, -1)
            tw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0][0]
            cv2.putText(canvas, msg,
                        (w//2 - tw//2, 38),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

        # ── 제스처 표시
        if gesture:
            icons = {'peace': 'PEACE', 'fist': 'FIST', 'open': 'OPEN (Clear)', 'point': 'POINT (Draw)'}
            label = icons.get(gesture, gesture.upper())
            cv2.putText(canvas, label, (10, h - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, ACCENT_COLOR, 2, cv2.LINE_AA)

        cv2.imshow('PhotoBooth', canvas)
        if cv2.waitKey(1) == 27:
            break

if use_picamera and picam2:
    picam2.stop()
if video:
    video.release()
cv2.destroyAllWindows()
