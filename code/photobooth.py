import cv2
import mediapipe as mp
import os
import numpy as np
import time
import logging
import signal
from datetime import datetime

# ── Ctrl+C 안전 종료 ──────────────────────────────────────────
_exit_requested = False
def _sigint_handler(sig, frame):
    global _exit_requested
    _exit_requested = True
signal.signal(signal.SIGINT, _sigint_handler)

# ── 로그 설정 ────────────────────────────────────────────────────
_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'photobooth_debug.log')
_file_handler = logging.FileHandler(_log_path, mode='w', encoding='utf-8')
_file_handler.setLevel(logging.DEBUG)
_stream_handler = logging.StreamHandler()
_stream_handler.setLevel(logging.WARNING)   # 터미널엔 WARNING 이상만
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[_file_handler, _stream_handler]
)
log = logging.getLogger('photobooth')


# ── Mediapipe 설정 ──────────────────────────────────────────────
BaseOptions          = mp.tasks.BaseOptions
GestureRecognizer    = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode          = mp.tasks.vision.RunningMode
HAND_CONNECTIONS     = mp.solutions.hands.HAND_CONNECTIONS

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gesture_recognizer.task')

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=RunningMode.VIDEO,
)

# MediaPipe 제스처명 → 내부 제스처명 매핑
# Closed_Fist → fist, Victory → peace, Open_Palm → open, 나머지 → None(그리기)
_GESTURE_MAP = {
    'Closed_Fist': 'fist',
    'Victory':     'peace',
    'Open_Palm':   'open',
}

# ── 상수 ────────────────────────────────────────────────────────
TOTAL_SHOTS   = 4          # 총 촬영 장수
COUNTDOWN_SEC = 3          # 카운트다운 초
FLASH_SEC     = 0.5        # 촬영 후 플래시 효과 시간
STRIP_W       = 200        # 오른쪽 스트립 폭

# ── 테마 컬러 (BGR) ─────────────────────────────────────────────
DARK         = (35,  25,  50)    # 짙은 보라 (HUD, 오버레이)
ACCENT       = (100,  50, 255)   # 핫핑크/바이올렛
PINK         = (180, 120, 255)   # 소프트 핑크
GOLD         = (50,  190, 255)   # 골드/옐로우
WHITE        = (255, 250, 255)   # 화이트
STRIP_BG     = (255, 252, 255)   # 스트립 배경
BG_COLOR     = (245, 235, 255)   # fallback
BORDER_COLOR = (200, 150, 255)   # 테두리
TEXT_COLOR   = (40,   20,  80)   # 다크 텍스트
ACCENT_COLOR = ACCENT            # 하위 호환
FLASH_COLOR  = (255, 255, 255)

# ── 펜 색상 팔레트 ──────────────────────────────────────────────
PEN_COLORS = [
    (0,   0,   255),   # 빨강
    (0,   165, 255),   # 주황
    (0,   220, 255),   # 노랑
    (60,  200,  60),   # 초록
    (255, 120,   0),   # 파랑
    (200,  60, 180),   # 보라
    (255, 255, 255),   # 화이트
]

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'photobooth_output')
os.makedirs(SAVE_DIR, exist_ok=True)



# ── 카메라 초기화 ────────────────────────────────────────────────
video  = None
for idx in range(0, 20):
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None and len(test_frame.shape) == 3:
            print(f"USB 카메라 발견: /dev/video{idx}")
            video = cap
            break
    cap.release()
if video is None:
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
    """세련된 4컷 스트립 패널"""
    n       = TOTAL_SHOTS
    title_h = 56
    foot_h  = 44
    margin  = 14
    slot_w  = strip_w - margin * 2
    avail_h = cam_h - title_h - foot_h - margin * (n + 1)
    slot_h  = avail_h // n

    strip = np.full((cam_h, strip_w, 3), STRIP_BG, dtype=np.uint8)

    # 미세 격자 패턴
    for yy in range(0, cam_h, 18):
        cv2.line(strip, (0, yy), (strip_w, yy), (238, 230, 248), 1)
    for xx in range(0, strip_w, 18):
        cv2.line(strip, (xx, 0), (xx, cam_h), (238, 230, 248), 1)

    # 상단 타이틀 바
    ov = strip.copy()
    cv2.rectangle(ov, (0, 0), (strip_w, title_h), DARK, -1)
    cv2.addWeighted(ov, 0.88, strip, 0.12, 0, strip)
    title = "4-CUT"
    tw_t = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.15, 2)[0][0]
    cv2.putText(strip, title, (strip_w//2 - tw_t//2, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.15, ACCENT, 2, cv2.LINE_AA)
    # 타이틀 바 하단 구분선
    cv2.line(strip, (0, title_h), (strip_w, title_h), PINK, 2)

    # 사진 슬롯
    for i in range(n):
        y0 = title_h + margin + i * (slot_h + margin)
        x0 = margin

        if i < len(photos) and photos[i] is not None:
            # 폴라로이드 스타일: 흰 테두리 + 그림자
            pad = 5
            shadow_off = 3
            cv2.rectangle(strip,
                          (x0-pad+shadow_off, y0-pad+shadow_off),
                          (x0+slot_w+pad+shadow_off, y0+slot_h+pad+shadow_off),
                          (210, 200, 225), -1)
            cv2.rectangle(strip, (x0-pad, y0-pad),
                          (x0+slot_w+pad, y0+slot_h+pad), WHITE, -1)
            draw_rounded_rect(strip, (x0-pad-1, y0-pad-1),
                              (x0+slot_w+pad+1, y0+slot_h+pad+1),
                              PINK, radius=4, thickness=2)
            thumb = cv2.resize(photos[i], (slot_w, slot_h))
            strip[y0:y0+slot_h, x0:x0+slot_w] = thumb
            # 완료 배지
            bx, by = x0+slot_w+pad-3, y0+slot_h+pad-3
            cv2.circle(strip, (bx, by), 13, ACCENT, -1)
            cv2.circle(strip, (bx, by), 13, WHITE, 1)
            cv2.putText(strip, str(i+1), (bx-5, by+5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        else:
            # 빈 슬롯: 점선 테두리 + 번호 원
            draw_rounded_rect(strip, (x0, y0), (x0+slot_w, y0+slot_h),
                              (242, 235, 252), radius=8, thickness=-1)
            for d in range(0, slot_w + slot_h, 14):
                if d < slot_w:
                    cv2.circle(strip, (x0+d, y0), 1, PINK, -1)
                    cv2.circle(strip, (x0+d, y0+slot_h), 1, PINK, -1)
                if d < slot_h:
                    cv2.circle(strip, (x0, y0+d), 1, PINK, -1)
                    cv2.circle(strip, (x0+slot_w, y0+d), 1, PINK, -1)
            cxs, cys = x0 + slot_w//2, y0 + slot_h//2
            cv2.circle(strip, (cxs, cys), 24, PINK, 2)
            num_t = str(i+1)
            ntw = cv2.getTextSize(num_t, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0][0]
            cv2.putText(strip, num_t, (cxs - ntw//2, cys+8),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, PINK, 2, cv2.LINE_AA)

    # 하단: 진행 도트 (●●○○)
    taken  = len(photos)
    dot_y  = cam_h - foot_h//2
    dot_x0 = strip_w//2 - n*12
    for i in range(n):
        cx_d = dot_x0 + i*22 + 8
        if i < taken:
            cv2.circle(strip, (cx_d, dot_y), 8, ACCENT, -1)
        else:
            cv2.circle(strip, (cx_d, dot_y), 8, (210, 195, 230), -1)
            cv2.circle(strip, (cx_d, dot_y), 8, PINK, 1)

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


def make_inpaint_mask(draw_canvas):
    """그린 경계선 안쪽을 채운 inpainting 마스크 반환 (흰색=마스크 영역)"""
    gray = cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 선 두께가 얇아도 윤곽이 닫히도록 약간 팽창
    kernel = np.ones((line_thickness * 2 + 1, line_thickness * 2 + 1), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # 외곽 윤곽선을 찾아 내부를 흰색으로 채움
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask


def save_final(photos, cam_h, strip_w, masks=None, session_dir=None):
    """개별 사진 4장 + 인생네컷 합성 이미지 + inpainting 마스크 저장"""
    if session_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(SAVE_DIR, ts)
    os.makedirs(session_dir, exist_ok=True)

    # 개별 사진 저장
    for i, photo in enumerate(photos):
        path = os.path.join(session_dir, f"shot_{i+1}.jpg")
        cv2.imwrite(path, photo)
        print(f"  저장: {path}")

        # inpainting 마스크 저장 (경계선 안쪽이 흰색=마스크)
        if masks and i < len(masks) and masks[i] is not None:
            mask_path = os.path.join(session_dir, f"shot_{i+1}_mask.png")
            cv2.imwrite(mask_path, masks[i])
            print(f"  마스크 저장: {mask_path}")

    # 인생네컷 합성 이미지 저장
    collage = make_final_collage(photos)
    if collage is not None:
        collage_path = os.path.join(session_dir, "4cut.jpg")
        cv2.imwrite(collage_path, collage)
        print(f"  합성 저장: {collage_path}")

    print(f"✓ 저장 완료 → {session_dir}")
    return session_dir


def _make_video_writer(path, frame_size, fps):
    """AVI/MJPG VideoWriter 생성 (라즈베리파이 호환)"""
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
    if writer.isOpened():
        return writer
    writer.release()
    return None


def _avi_to_mp4(src_avi, dst_mp4):
    """ffmpeg으로 AVI → MP4 재인코딩. 성공 시 True, 실패 시 False."""
    import subprocess, shutil as _sh
    if not _sh.which('ffmpeg'):
        return False
    ret = subprocess.run(
        ['ffmpeg', '-y', '-i', src_avi,
         '-vcodec', 'libx264', '-preset', 'fast',
         '-crf', '23', '-movflags', '+faststart',
         dst_mp4],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return ret.returncode == 0


# ── 상태 머신 ────────────────────────────────────────────────────
STATE_WAITING    = 'waiting'    # 다음 촬영 대기
STATE_COUNTDOWN  = 'countdown'  # 카운트다운 진행 중
STATE_FLASH      = 'flash'      # 촬영 직후 플래시
STATE_DONE       = 'done'       # 4장 완료 (레거시 호환)
STATE_REVIEW     = 'review'     # 최종 사진 리뷰 화면

state          = STATE_WAITING
photos         = []             # 촬영된 사진 리스트
photo_masks    = []             # 각 촬영에 대응하는 inpainting 마스크
review_collage = None           # 리뷰 화면에 표시할 최종 콜라주
countdown_start = None
flash_start    = None
last_gesture   = None
gesture_start  = None
GESTURE_HOLD   = 0.8            # peace 제스처 유지 시간(초)

fps = 30

# ── 그리기 캔버스 ──────────────────────────────────────────────
draw_canvas    = None
prev_x, prev_y = None, None
color_idx      = 0
drawing_color  = PEN_COLORS[color_idx]
line_thickness = 5
last_standby   = False

print("=" * 50)
print("  인생네컷 포토부스")
print("=" * 50)
print("  peace (0.8s)  : 촬영")
print("  hand (default): 그리기 (landmark 8 트래킹)")
print("  fist          : 펜 색상 변경")
print("  open          : 그림 지우기 / 완료 시 전체 리셋")
print("  ESC           : 종료")
print("=" * 50)

cv2.namedWindow('PhotoBooth', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('PhotoBooth', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ── 녹화 상태 변수 ──────────────────────────────────────────────
_out_writer   = None   # cv2.VideoWriter
_session_dir  = None   # save_final 이 생성한 세션 폴더
_review_cap   = None   # 리뷰 화면에서 재생할 VideoCapture

with GestureRecognizer.create_from_options(options) as recognizer:
    frame_index = 0

    while True:

        # ── 프레임 읽기
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.time()

        # 캔버스 초기화 (첫 프레임)
        if draw_canvas is None:
            draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # ── 손 인식 (4-cut 리뷰 화면에서는 inference 생략)
        gesture = None
        if state in (STATE_REVIEW, STATE_DONE):
            prev_x, prev_y = None, None
        else:
            img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            ts_ms    = int(frame_index * 1000 / fps)
            frame_index += 1
            result = recognizer.recognize_for_video(mp_image, ts_ms)

        if state not in (STATE_REVIEW, STATE_DONE) and not result.hand_landmarks:
            if prev_x is not None:
                log.warning('hand NOT detected → draw interrupted (prev reset)')
            prev_x, prev_y = None, None
        if state not in (STATE_REVIEW, STATE_DONE) and result.hand_landmarks:
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                # GestureRecognizer 결과에서 제스처 읽기
                raw_gesture = result.gestures[i][0].category_name if result.gestures else 'None'
                gesture = _GESTURE_MAP.get(raw_gesture, None)  # 매핑 없으면 None(=그리기)

                # 제스처 변경 시 로그
                if gesture != last_gesture:
                    log.info(f'gesture: {last_gesture} → {gesture}  (raw={raw_gesture})')

                # landmark 8 (검지 끝) 좌표
                ix = int(hand_landmarks[8].x * w)
                iy = int(hand_landmarks[8].y * h)

                # 특수 제스처 처리
                if gesture == 'fist':
                    # 첫 진입 시 펜 색상 한 단계 전환
                    if last_gesture != 'fist':
                        color_idx = (color_idx + 1) % len(PEN_COLORS)
                        drawing_color = PEN_COLORS[color_idx]
                        log.info(f'color changed → {drawing_color}')
                    prev_x, prev_y = None, None
                elif gesture == 'open':
                    log.info('canvas cleared')
                    draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    prev_x, prev_y = None, None
                elif gesture == 'peace':
                    # 촬영 트리거 - 그리기 일시 중단
                    prev_x, prev_y = None, None
                else:
                    # 카운트다운 중에는 그리기 비활성화
                    if state == STATE_COUNTDOWN:
                        prev_x, prev_y = None, None
                    else:
                        # landmark 8 트래킹으로 항상 그리기
                        if prev_x is not None and prev_y is not None:
                            cv2.line(draw_canvas, (prev_x, prev_y), (ix, iy), drawing_color, line_thickness)
                            log.debug(f'draw ({prev_x},{prev_y})→({ix},{iy})')
                        else:
                            log.debug(f'draw START at ({ix},{iy})')
                        prev_x, prev_y = ix, iy

                # 그리기 커서
                if gesture in ('fist', 'open', 'peace'):
                    cv2.circle(frame, (ix, iy), 11, (200, 200, 200), 1)
                else:
                    cv2.circle(frame, (ix, iy), line_thickness + 5, drawing_color, 2)
                    cv2.circle(frame, (ix, iy), 3, drawing_color, -1)

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

        last_standby = (gesture == 'fist')

        if gesture == 'open' and state in (STATE_WAITING, STATE_DONE, STATE_REVIEW):
            if state in (STATE_DONE, STATE_REVIEW) or last_gesture != 'open':
                if state in (STATE_DONE, STATE_REVIEW):
                    photos = []
                    photo_masks = []
                    review_collage = None
                    draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    state = STATE_WAITING
                    # 리뷰 재생 캡처 해제
                    if _review_cap:
                        _review_cap.release()
                        _review_cap = None
                        _vid_play_r = os.path.join(SAVE_DIR, '_recording_play.avi')
                        if os.path.exists(_vid_play_r):
                            os.remove(_vid_play_r)
                    # 현재 녹화 저장 후 새 녹화 시작
                    if _out_writer and _out_writer is not False:
                        _out_writer.release()
                        import shutil as _sh_reset
                        _vid_tmp_r = os.path.join(SAVE_DIR, '_recording_tmp.avi')
                        if os.path.exists(_vid_tmp_r) and _session_dir:
                            os.makedirs(_session_dir, exist_ok=True)
                            _vid_mp4_r = os.path.join(_session_dir, 'recording.mp4')
                            _vid_avi_r = os.path.join(_session_dir, 'recording.avi')
                            if _avi_to_mp4(_vid_tmp_r, _vid_mp4_r):
                                os.remove(_vid_tmp_r)
                            else:
                                _sh_reset.move(_vid_tmp_r, _vid_avi_r)
                        _out_writer = None   # 다음 프레임에서 새 writer 생성
                        _session_dir = None
                    print("초기화 완료")
                else:
                    draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)

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
                # 경계선 안쪽 채움 마스크 생성 (캔버스 클리어 전)
                photo_masks.append(make_inpaint_mask(draw_canvas))
                state = STATE_FLASH
                flash_start = now
                draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)  # 촬영 후 캔버스 초기화
                print(f"[{len(photos)}/{TOTAL_SHOTS}] 촬영!")
                if len(photos) >= TOTAL_SHOTS:
                    ts_s = datetime.now().strftime("%Y%m%d_%H%M%S")
                    _session_dir = os.path.join(SAVE_DIR, ts_s)
                    save_final(photos, h, STRIP_W, masks=photo_masks, session_dir=_session_dir)

        elif state == STATE_FLASH:
            if now - flash_start >= FLASH_SEC:
                if len(photos) >= TOTAL_SHOTS:
                    review_collage = make_final_collage(photos)
                    state = STATE_REVIEW
                    # ── 현재까지 녹화본을 복사해 재생용으로 열기 (writer는 계속 유지)
                    _vid_tmp_rv = os.path.join(SAVE_DIR, '_recording_tmp.avi')
                    _vid_play   = os.path.join(SAVE_DIR, '_recording_play.avi')
                    if os.path.exists(_vid_tmp_rv):
                        import shutil as _sh_cp
                        _sh_cp.copy2(_vid_tmp_rv, _vid_play)
                        _review_cap = cv2.VideoCapture(_vid_play)
                else:
                    state = STATE_WAITING

        # ── 그리기 캔버스를 카메라 프레임에 합성
        mask = cv2.cvtColor(draw_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(draw_canvas, draw_canvas, mask=mask)
        frame = cv2.add(frame_bg, canvas_fg)

        # ── UI 합성 ─────────────────────────────────────────────
        strip  = make_strip(photos, STRIP_W, h)
        canvas = np.full((h, w + STRIP_W, 3), BG_COLOR, dtype=np.uint8)
        canvas[:, :w] = frame
        canvas[:, w:] = strip

        # ── 상단 HUD 바
        hud_h = 52
        hud_ov = canvas.copy()
        cv2.rectangle(hud_ov, (0, 0), (w, hud_h), DARK, -1)
        cv2.addWeighted(hud_ov, 0.75, canvas, 0.25, 0, canvas)
        cv2.line(canvas, (0, hud_h), (w, hud_h), PINK, 1)
        # 촬영 진행 도트
        for i in range(TOTAL_SHOTS):
            cx_h = 28 + i * 28
            cy_h = hud_h // 2
            if i < len(photos):
                cv2.circle(canvas, (cx_h, cy_h), 10, ACCENT, -1)
                cv2.circle(canvas, (cx_h, cy_h), 10, WHITE, 1)
            else:
                cv2.circle(canvas, (cx_h, cy_h), 10, (70, 55, 95), -1)
                cv2.circle(canvas, (cx_h, cy_h), 10, (120, 90, 155), 1)
        # Shot 텍스트
        shot_lbl = f"Shot {min(len(photos)+1,TOTAL_SHOTS)}/{TOTAL_SHOTS}"
        cv2.putText(canvas, shot_lbl, (TOTAL_SHOTS*28+40, hud_h//2+6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 155, 210), 1, cv2.LINE_AA)
        # 펜 색상 표시
        cv2.circle(canvas, (w - 38, hud_h//2), 13, drawing_color, -1)
        cv2.circle(canvas, (w - 38, hud_h//2), 13, WHITE, 2)
        cv2.putText(canvas, "pen", (w-75, hud_h//2+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 140, 190), 1, cv2.LINE_AA)

        # ── 오버레이: 카운트다운
        if state == STATE_COUNTDOWN:
            elapsed_cd  = now - countdown_start
            num_show    = max(1, int(COUNTDOWN_SEC - elapsed_cd) + 1)
            progress_cd = 1.0 - (elapsed_cd % 1.0)
            cx, cy = w // 2, h // 2
            # 반투명 다크 원
            ov_cd = canvas.copy()
            cv2.circle(ov_cd, (cx, cy), 115, DARK, -1)
            cv2.addWeighted(ov_cd, 0.78, canvas, 0.22, 0, canvas)
            # 링 트랙
            cv2.circle(canvas, (cx, cy), 92, (80, 60, 110), 8)
            # 진행 호
            arc_angle = int(360 * progress_cd)
            cv2.ellipse(canvas, (cx, cy), (92, 92), -90, 0, arc_angle, ACCENT, 8)
            # 외곽 빛 링
            cv2.circle(canvas, (cx, cy), 100, PINK, 2)
            # 숫자
            tw_cd = cv2.getTextSize(str(num_show), cv2.FONT_HERSHEY_DUPLEX, 4.5, 8)[0][0]
            cv2.putText(canvas, str(num_show),
                        (cx - tw_cd//2, cy + 32),
                        cv2.FONT_HERSHEY_DUPLEX, 4.5, WHITE, 8, cv2.LINE_AA)

        # ── 오버레이: 플래시
        elif state == STATE_FLASH:
            ratio = 1.0 - (now - flash_start) / FLASH_SEC
            wh_ov = np.full_like(canvas, 255)
            cv2.addWeighted(wh_ov, ratio * 0.85, canvas, 1 - ratio * 0.85, 0, canvas)

        # ── 오버레이: 완료 화면 (레거시 - STATE_REVIEW로 대체됨)
        elif state == STATE_DONE:
            ov_done = canvas.copy()
            cv2.rectangle(ov_done, (0, h//2-80), (w, h//2+80), DARK, -1)
            cv2.addWeighted(ov_done, 0.82, canvas, 0.18, 0, canvas)
            draw_rounded_rect(canvas, (w//2-160, h//2-70), (w//2+160, h//2+70),
                              DARK, radius=20, thickness=-1)
            draw_rounded_rect(canvas, (w//2-160, h//2-70), (w//2+160, h//2+70),
                              ACCENT, radius=20, thickness=2)
            msg_d = "SAVED!"
            tw_d = cv2.getTextSize(msg_d, cv2.FONT_HERSHEY_DUPLEX, 2.4, 5)[0][0]
            cv2.putText(canvas, msg_d, (w//2 - tw_d//2, h//2),
                        cv2.FONT_HERSHEY_DUPLEX, 2.4, GOLD, 5, cv2.LINE_AA)
            sub_d = "Open palm : New Session  |  ESC : Quit"
            sw_d  = cv2.getTextSize(sub_d, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
            cv2.putText(canvas, sub_d, (w//2 - sw_d//2, h//2 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, PINK, 1, cv2.LINE_AA)

        # ── 오버레이: 리뷰 화면 (4장 완료 후 최종 사진 표시)
        elif state == STATE_REVIEW:
            if review_collage is not None:
                # 전체 캔버스를 다크로
                canvas[:] = (30, 20, 45)

                # 영역 분할: 콜라주 40% / 영상 60%
                total_canvas_w = w + STRIP_W
                half_w = int(total_canvas_w * 0.4)
                outer  = 8    # 화면 바깥쪽 여백
                gap    = 16   # 콜라주↔영상 사이 간격 (절반씩 적용)

                # ── 콜라주 (왼쪽)
                avail_h = h - 56 - 52
                avail_w = half_w - outer - gap // 2
                col_h_orig, col_w_orig = review_collage.shape[:2]
                scale = min(avail_h / col_h_orig, avail_w / col_w_orig)
                new_h = int(col_h_orig * scale)
                new_w = int(col_w_orig * scale)
                col_resized = cv2.resize(review_collage, (new_w, new_h))
                y_off = 56 + (avail_h - new_h) // 2
                x_off = outer + (avail_w - new_w) // 2
                canvas[y_off:y_off+new_h, x_off:x_off+new_w] = col_resized
                draw_rounded_rect(canvas, (x_off-4, y_off-4),
                                  (x_off+new_w+4, y_off+new_h+4),
                                  ACCENT, radius=8, thickness=2)

                # ── 영상 (오른쪽)
                if _review_cap and _review_cap.isOpened():
                    ret_v, vframe = _review_cap.read()
                    if not ret_v:  # 루프
                        _review_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_v, vframe = _review_cap.read()
                    if ret_v and vframe is not None:
                        vid_avail_w = (w + STRIP_W) - half_w - outer - gap // 2
                        vh, vw = vframe.shape[:2]
                        v_scale = min(avail_h / vh, vid_avail_w / vw)
                        v_new_h = int(vh * v_scale)
                        v_new_w = int(vw * v_scale)
                        vframe_r = cv2.resize(vframe, (v_new_w, v_new_h))
                        v_y = 56 + (avail_h - v_new_h) // 2
                        v_x = half_w + gap // 2 + (vid_avail_w - v_new_w) // 2
                        canvas[v_y:v_y+v_new_h, v_x:v_x+v_new_w] = vframe_r
                        draw_rounded_rect(canvas, (v_x-4, v_y-4),
                                          (v_x+v_new_w+4, v_y+v_new_h+4),
                                          PINK, radius=8, thickness=2)
                        cv2.putText(canvas, 'REPLAY', (v_x + 6, v_y + 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GOLD, 2, cv2.LINE_AA)
                else:
                    # 영상 없음 안내
                    msg = 'No video'
                    mw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0][0]
                    canvas[h//2-12:h//2+12, half_w:w+STRIP_W] = (50, 40, 70)
                    cv2.putText(canvas, msg, (half_w + ((w+STRIP_W-half_w)-mw)//2, h//2+8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120,100,160), 1, cv2.LINE_AA)

                # 상단 타이틀 바
                total_w = w + STRIP_W
                ov_rt = canvas.copy()
                cv2.rectangle(ov_rt, (0, 0), (total_w, 52), DARK, -1)
                cv2.addWeighted(ov_rt, 0.85, canvas, 0.15, 0, canvas)
                cv2.line(canvas, (0, 52), (total_w, 52), PINK, 1)
                title_r = "YOUR 4-CUT"
                trw = cv2.getTextSize(title_r, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0][0]
                cv2.putText(canvas, title_r, (total_w//2 - trw//2, 38),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, GOLD, 2, cv2.LINE_AA)

                # 하단 안내 바
                ov_rb = canvas.copy()
                cv2.rectangle(ov_rb, (0, h-52), (total_w, h), DARK, -1)
                cv2.addWeighted(ov_rb, 0.80, canvas, 0.20, 0, canvas)
                cv2.line(canvas, (0, h-52), (total_w, h-52), PINK, 1)
                hint_r = "Open (palm) : New Session  |  ESC : Quit"
                hrw = cv2.getTextSize(hint_r, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
                cv2.putText(canvas, hint_r, (total_w//2 - hrw//2, h-18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)

                # 반짝이 파티클 (상단 타이틀 영역)
                rng_rv = np.random.default_rng(int(now * 4) % 999)
                for _ in range(10):
                    sx = int(rng_rv.integers(10, total_w-10))
                    sy = int(rng_rv.integers(4, 48))
                    sr = int(rng_rv.integers(2, 5))
                    sc = [ACCENT, GOLD, PINK, WHITE][int(rng_rv.integers(0, 4))]
                    cv2.circle(canvas, (sx, sy), sr, sc, -1)

        # ── 오버레이: 대기 안내 (하단 HUD)
        elif state == STATE_WAITING:
            hud2_y = h - 58
            ov_w = canvas.copy()
            cv2.rectangle(ov_w, (0, hud2_y), (w, h), DARK, -1)
            cv2.addWeighted(ov_w, 0.72, canvas, 0.28, 0, canvas)
            cv2.line(canvas, (0, hud2_y), (w, hud2_y), PINK, 1)

            if gesture == 'peace' and gesture_start is not None:
                held  = now - gesture_start
                ratio = min(held / GESTURE_HOLD, 1.0)
                bx1, bx2 = 20, w - 20
                by = h - 18
                cv2.rectangle(canvas, (bx1, by-7), (bx2, by+7), (70, 55, 95), -1)
                fill_x = bx1 + int((bx2-bx1)*ratio)
                cv2.rectangle(canvas, (bx1, by-7), (fill_x, by+7), ACCENT, -1)
                cv2.rectangle(canvas, (bx1, by-7), (bx2, by+7), PINK, 1)
                gm = "Hold peace to capture..."
                cv2.putText(canvas, gm, (bx1, h-28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
            else:
                hints = [("PEACE","capture"),("HAND","draw"),("FIST","color"),("OPEN","clear")]
                seg_w = w // len(hints)
                for hi, (gn, gd) in enumerate(hints):
                    cx_g = hi * seg_w + seg_w // 2
                    active = gesture and gesture.upper() == gn
                    col_g  = ACCENT if active else (150, 130, 180)
                    nw_g   = cv2.getTextSize(gn, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)[0][0]
                    cv2.putText(canvas, gn, (cx_g - nw_g//2, h-34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, col_g, 2, cv2.LINE_AA)
                    dw_g = cv2.getTextSize(gd, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
                    cv2.putText(canvas, gd, (cx_g - dw_g//2, h-14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 95, 140), 1, cv2.LINE_AA)

        # ── 구분선 (카메라 / 스트립 경계) - 리뷰 화면에서는 숨김
        if state != STATE_REVIEW:
            cv2.line(canvas, (w, 0), (w, h), PINK, 2)

        # ── 녹화: VideoWriter 초기화 (첫 프레임에서 크기 확정 후 생성)
        if _out_writer is None:
            _canvas_h, _canvas_w = canvas.shape[:2]
            _vid_tmp = os.path.join(SAVE_DIR, '_recording_tmp.avi')
            _out_writer = _make_video_writer(_vid_tmp, (_canvas_w, _canvas_h), fps)
            if _out_writer:
                print(f"[녹화 시작] {_vid_tmp}")
            else:
                print("[경고] VideoWriter 초기화 실패 — 녹화 비활성화")
                _out_writer = False  # 재시도 방지
        if _out_writer:
            _out_writer.write(canvas)

        cv2.imshow('PhotoBooth', canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or _exit_requested:  # ESC / q / Ctrl+C
            break

# ── 녹화 종료 및 저장 ─────────────────────────────────────────
# _review_cap 해제 (혹시 남아 있을 경우)
if _review_cap:
    _review_cap.release()
    # 재생용 임시 복사본 삭제
    _vid_play_cleanup = os.path.join(SAVE_DIR, '_recording_play.avi')
    if os.path.exists(_vid_play_cleanup):
        os.remove(_vid_play_cleanup)

import shutil as _shutil
_vid_tmp = os.path.join(SAVE_DIR, '_recording_tmp.avi')

# 녹화 종료 및 세션 폴더에 저장
if _out_writer and _out_writer is not False:
    _out_writer.release()
    _ts_end  = datetime.now().strftime("%Y%m%d_%H%M%S")
    if _session_dir is None:
        _session_dir = os.path.join(SAVE_DIR, _ts_end)
    os.makedirs(_session_dir, exist_ok=True)
    if os.path.exists(_vid_tmp):
        _vid_mp4 = os.path.join(_session_dir, 'recording.mp4')
        _vid_avi = os.path.join(_session_dir, 'recording.avi')
        print("[녹화 변환 중...]")
        if _avi_to_mp4(_vid_tmp, _vid_mp4):
            os.remove(_vid_tmp)
            print(f"[녹화 저장] {_vid_mp4}  (ffmpeg 재인코딩)")
        else:
            _shutil.move(_vid_tmp, _vid_avi)
            print(f"[녹화 저장] {_vid_avi}  (AVI, ffmpeg 없음)")

if video:
    video.release()
cv2.destroyAllWindows()
