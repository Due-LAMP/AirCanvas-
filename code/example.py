import cv2
import mediapipe as mp
import os
import numpy as np

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

# Mediapipe 새 tasks API 설정
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

# 모델 경로 설정
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=RunningMode.VIDEO,
)


def get_finger_status(landmarks):
    """
    손가락이 펴져 있는지 접혀 있는지 확인하는 함수
    """
    fingers = []

    # 엄지: 랜드마크 4가 랜드마크 3의 왼쪽에 있으면 펼쳐진 상태
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # 나머지 손가락: 각 손가락의 팁 (8, 12, 16, 20)이 PIP (6, 10, 14, 18) 위에 있으면 펼쳐진 상태
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]
    for tip, pip_j in zip(tips, pip_joints):
        if landmarks[tip].y < landmarks[pip_j].y:
            fingers.append(1)
        else:
            fingers.append(0)

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
    elif fingers_status == [1, 1, 0, 0, 0]:
        return 'standby'
    return None


# 카메라 초기화: Picamera2(CSI) 우선, 실패 시 USB 웹캠 사용
picam2 = None
video = None
use_picamera = False

# 먼저 USB 웹캠 탐색 (V4L2 Video Capture 장치만)
for idx in range(0, 20):
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None and len(test_frame.shape) == 3:
            print(f"USB 카메라 발견: /dev/video{idx}")
            video = cap
            break
    cap.release()

# USB 웹캠 못 찾으면 Picamera2(CSI 카메라) 시도
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
    print("카메라를 찾을 수 없습니다. 카메라를 연결해주세요.")
    exit(1)

fps = 30

print("Webcam is running... Press 'ESC' to exit.")
print("검지(point): 그리기 | 주먹(fist): 초기화 | 보자기(open): 그리기 중지")

# 그리기용 캔버스 및 이전 좌표
canvas = None
prev_x, prev_y = None, None
drawing_color = (0, 0, 255)  # 빨간색
line_thickness = 3

# 전체화면 설정
cv2.namedWindow('Hand Gesture', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Hand Gesture', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with HandLandmarker.create_from_options(options) as landmarker:
    frame_index = 0
    while True:
        if use_picamera:
            frame = picam2.capture_array()
            # Picamera2 RGB888 -> BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = video.read()
            if not ret:
                break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # 캔버스 초기화 (첫 프레임)
        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # mediapipe Image 생성
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        timestamp_ms = int(frame_index * 1000 / fps)
        frame_index += 1

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                fingers_status = get_finger_status(hand_landmarks)
                gesture = recognize_gesture(fingers_status)

                # 검지 끝(landmark 8) 좌표
                ix, iy = int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h)

                if gesture in ('point', 'standby'):
                    # 검지로 그리기
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (ix, iy), drawing_color, line_thickness)
                    prev_x, prev_y = ix, iy
                elif gesture == 'open':
                    # 보자기: 캔버스 초기화
                    canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    prev_x, prev_y = None, None
                else:
                    # 다른 제스처일 때 그리기 중지
                    prev_x, prev_y = None, None

                # 검지 위치에 원 표시
                cv2.circle(frame, (ix, iy), 8, (0, 255, 255), -1)

                # 손 랜드마크와 연결선 그리기
                for connection in HAND_CONNECTIONS:
                    start = hand_landmarks[connection[0]]
                    end = hand_landmarks[connection[1]]
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                # 제스처 텍스트 표시
                if gesture:
                    cv2.putText(frame, gesture, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            prev_x, prev_y = None, None

        # 캔버스를 프레임에 합성
        mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        frame = cv2.add(frame_bg, canvas_fg)

        cv2.imshow('Hand Gesture', frame)
        if cv2.waitKey(1) == 27:  # ESC 키로 종료
            break

if use_picamera and picam2:
    picam2.stop()
if video:
    video.release()
cv2.destroyAllWindows()