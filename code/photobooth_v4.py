#!/usr/bin/env python3
"""
photobooth_v4.py – Lo-Fi Brown Theme
따뜻한 베이지/갈색 + 빈티지 필름 느낌
"""

import os, sys, time, signal
import numpy as np

import cv2
import mediapipe as mp
from datetime import datetime

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms'
os.environ.pop('QT_PLUGIN_PATH', None)

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                              QHBoxLayout, QSizePolicy)
from PyQt5.QtCore    import Qt, QTimer, QPointF, QRectF, QRect
from PyQt5.QtGui     import (QPainter, QColor, QLinearGradient, QRadialGradient,
                              QPen, QBrush, QPainterPath, QFont,
                              QImage, QPixmap)

# ── Ctrl+C ─────────────────────────────────────────────────────
_exit_requested = False
def _sigint_handler(sig, frame):
    global _exit_requested
    _exit_requested = True
signal.signal(signal.SIGINT, _sigint_handler)

# ── MediaPipe ──────────────────────────────────────────────────
BaseOptions              = mp.tasks.BaseOptions
GestureRecognizer        = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode              = mp.tasks.vision.RunningMode
HAND_CONNECTIONS         = mp.solutions.hands.HAND_CONNECTIONS

_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gesture_recognizer.task')
_GESTURE_MAP = {'Closed_Fist': 'fist', 'Victory': 'peace', 'Open_Palm': 'open'}

# ── 상수 ───────────────────────────────────────────────────────
TOTAL_SHOTS   = 4
COUNTDOWN_SEC = 3
FLASH_SEC     = 0.6
GESTURE_HOLD  = 0.8
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'photobooth_output')
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Lo-Fi Brown 테마 컬러 ──────────────────────────────────────
# 배경 계열
C_BG0     = QColor( 28,  22,  16)   # 가장 어두운 갈색
C_BG1     = QColor( 40,  32,  22)   # 패널 배경
C_BG2     = QColor( 55,  44,  30)   # 카드 배경
C_BG3     = QColor( 72,  58,  40)   # 밝은 패널

# 포인트 컬러
C_AMBER   = QColor(220, 160,  50)   # 앰버 (메인 액센트)
C_AMBER2  = QColor(180, 120,  30)   # 어두운 앰버
C_CREAM   = QColor(240, 225, 190)   # 크림
C_TAN     = QColor(200, 170, 120)   # 탄
C_RUST    = QColor(180,  85,  40)   # 러스트 레드

# 텍스트
C_WHITE   = QColor(245, 235, 210)   # 따뜻한 화이트
C_GRAY1   = QColor(160, 140, 110)   # 보조 텍스트
C_GRAY2   = QColor( 70,  58,  40)   # 비활성

# 필름 그레인 시드 고정용
_GRAIN_FRAMES = 0

# 펜 팔레트 (따뜻한 톤)
PEN_COLORS_QC = [
    QColor(220,  70,  50),   # 버밀리온 레드
    QColor(220, 140,  30),   # 앰버 오렌지
    QColor(200, 190,  60),   # 머스타드 옐로우
    QColor( 80, 160,  80),   # 세이지 그린
    QColor( 80, 130, 180),   # 스틸 블루
    QColor(160,  90, 140),   # 더스티 모브
    QColor(240, 225, 190),   # 크림 화이트
]
PEN_COLORS_CV = [
    ( 50,  70, 220), ( 30, 140, 220), ( 60, 190, 200),
    ( 80, 160,  80), (180, 130,  80), (140,  90, 160),
    (190, 225, 240),
]

STATE_WAITING   = 'waiting'
STATE_COUNTDOWN = 'countdown'
STATE_FLASH     = 'flash'
STATE_REVIEW    = 'review'


# ── 유틸 ───────────────────────────────────────────────────────
def _cv_to_pixmap(img: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qi = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qi.copy())


def _apply_lofi_lut(img: np.ndarray) -> np.ndarray:
    """따뜻한 Lo-Fi 색보정: 그린 채널 살짝 줄이고 레드/블루 틴트"""
    result = img.astype(np.float32)
    result[:, :, 0] *= 0.88   # B 약간 감소
    result[:, :, 1] *= 0.93   # G 약간 감소
    result[:, :, 2] = result[:, :, 2] * 1.04 + 8   # R 살짝 부스트
    # 전체 따뜻한 베이지 틴트 (화이트밸런스 warm shift)
    result[:, :, 0] = result[:, :, 0] * 0.95 + 5
    result[:, :, 2] = result[:, :, 2] * 0.98 + 12
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_grain(img: np.ndarray, strength: float = 8.0) -> np.ndarray:
    """필름 그레인 효과"""
    rng = np.random.default_rng(int(time.time() * 15) % 9999)
    grain = rng.normal(0, strength, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + grain, 0, 255).astype(np.uint8)


def _make_final_collage(photos):
    if not photos:
        return None
    ph, pw = photos[0].shape[:2]
    margin = 28; top_pad = 88; bottom_pad = 56
    col_w = pw + margin * 2
    col_h = top_pad + (ph + margin) * TOTAL_SHOTS + margin + bottom_pad
    # 따뜻한 크림 배경
    collage = np.full((col_h, col_w, 3), (210, 195, 165), dtype=np.uint8)
    # 상단 어두운 바
    collage[:top_pad, :] = (45, 36, 25)
    title = "4 - CUT"
    tw = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.3, 2)[0][0]
    cv2.putText(collage, title, (col_w//2 - tw//2, 60),
                cv2.FONT_HERSHEY_DUPLEX, 1.3, (210, 155, 50), 2, cv2.LINE_AA)
    for i, photo in enumerate(photos):
        # Lo-Fi 보정 적용
        lofi_photo = _apply_lofi_lut(photo)
        y0 = top_pad + margin + i * (ph + margin)
        collage[y0:y0+ph, margin:margin+pw] = lofi_photo
        # 갈색 테두리
        cv2.rectangle(collage, (margin-4, y0-4), (margin+pw+4, y0+ph+4), (130, 100, 60), 2)
    date_str = datetime.now().strftime("%Y . %m . %d")
    dw = cv2.getTextSize(date_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
    cv2.putText(collage, date_str, (col_w//2 - dw//2, col_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (110, 90, 60), 1, cv2.LINE_AA)
    return collage


def _make_inpaint_mask(canvas, thickness):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    kernel = np.ones((thickness * 2 + 1, thickness * 2 + 1), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask


def _save_final(photos, masks=None, session_dir=None):
    if session_dir is None:
        session_dir = os.path.join(SAVE_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(session_dir, exist_ok=True)
    for i, photo in enumerate(photos):
        cv2.imwrite(os.path.join(session_dir, f"shot_{i+1}.jpg"), photo)
        if masks and i < len(masks) and masks[i] is not None:
            cv2.imwrite(os.path.join(session_dir, f"shot_{i+1}_mask.png"), masks[i])
    collage = _make_final_collage(photos)
    if collage is not None:
        cv2.imwrite(os.path.join(session_dir, "4cut.jpg"), collage)
    print(f"✓ 저장 완료 → {session_dir}")
    return session_dir


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _fill_warm(p: QPainter, rect: QRectF, radius: float = 8):
    """따뜻한 갈색 글래스 패널"""
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    p.fillPath(path, QBrush(QColor(50, 40, 28, 200)))
    p.setPen(QPen(QColor(180, 145, 80, 80), 1))
    p.setBrush(Qt.NoBrush)
    p.drawPath(path)


def _draw_scratchy_line(p: QPainter, x1, y1, x2, y2):
    """빈티지 긁힌 선 효과"""
    p.setPen(QPen(QColor(200, 165, 90, 60), 1))
    p.drawLine(x1, y1, x2, y2)
    p.setPen(QPen(QColor(200, 165, 90, 25), 2))
    p.drawLine(x1 + 1, y1, x2 + 1, y2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  StripPanel – Lo-Fi Brown 스타일
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class StripPanel(QWidget):
    STRIP_W = 230

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(self.STRIP_W)
        self.photos: list = []

    def setPhotos(self, photos):
        self.photos = photos
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()
        n = TOTAL_SHOTS

        # ── 배경: 따뜻한 다크 브라운 그라디언트
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor(38, 30, 20))
        grad.setColorAt(0.5, QColor(30, 24, 15))
        grad.setColorAt(1.0, QColor(22, 17, 10))
        p.fillRect(0, 0, w, h, QBrush(grad))

        # 필름 스프로킷 홀 (좌측 세로)
        hole_color = QColor(18, 14, 9)
        border_color = QColor(80, 65, 40)
        for yy in range(20, h - 20, 32):
            p.setPen(QPen(border_color, 1))
            p.setBrush(QBrush(hole_color))
            p.drawRoundedRect(QRectF(4, yy, 10, 16), 3, 3)

        # 우측도 동일
        for yy in range(20, h - 20, 32):
            p.setPen(QPen(border_color, 1))
            p.setBrush(QBrush(hole_color))
            p.drawRoundedRect(QRectF(w - 14, yy, 10, 16), 3, 3)

        # 필름 테두리 세로선
        p.setPen(QPen(QColor(80, 65, 40), 1))
        p.drawLine(18, 0, 18, h)
        p.drawLine(w - 19, 0, w - 19, h)

        title_h = 72
        foot_h  = 52
        margin  = 14
        inner_x = 20  # 스프로킷 홀 안쪽 시작
        inner_w = w - 40
        slot_w  = inner_w - margin * 2 + 20
        avail_h = h - title_h - foot_h - margin * (n + 1)
        slot_h  = max(avail_h // n, 40)

        x0 = inner_x + margin - 8

        # ── 타이틀 영역
        title_rect = QRectF(18, 0, w - 36, title_h)
        tr_path = QPainterPath()
        tr_path.addRect(title_rect)
        tgrad = QLinearGradient(0, 0, 0, title_h)
        tgrad.setColorAt(0.0, QColor(55, 42, 25))
        tgrad.setColorAt(1.0, QColor(38, 30, 18))
        p.fillPath(tr_path, QBrush(tgrad))

        # 앰버 구분선
        _draw_scratchy_line(p, 18, title_h, w - 18, title_h)

        # 타이틀 텍스트
        p.setFont(QFont("Georgia", 15, QFont.Bold))
        p.setPen(C_AMBER)
        p.drawText(QRectF(18, 4, w - 36, title_h - 20), Qt.AlignCenter, "4-CUT")
        p.setFont(QFont("Courier", 7))
        p.setPen(QColor(160, 130, 75, 180))
        p.drawText(QRectF(18, title_h - 22, w - 36, 18),
                   Qt.AlignCenter, "lo-fi  photobooth")

        # ── 사진 슬롯
        for i in range(n):
            y0 = title_h + margin + i * (slot_h + margin)
            slot_rect = QRectF(x0, y0, slot_w, slot_h)

            if i < len(self.photos) and self.photos[i] is not None:
                # 따뜻한 그림자
                shadow_path = QPainterPath()
                shadow_path.addRoundedRect(slot_rect.translated(3, 3), 3, 3)
                p.fillPath(shadow_path, QBrush(QColor(0, 0, 0, 90)))

                # 크림 테두리 (폴라로이드)
                pad = 5
                pol_rect = slot_rect.adjusted(-pad, -pad, pad, pad + 12)
                pol_path = QPainterPath()
                pol_path.addRoundedRect(pol_rect, 3, 3)
                p.fillPath(pol_path, QBrush(QColor(215, 200, 170)))
                p.setPen(QPen(QColor(160, 135, 90), 1))
                p.drawPath(pol_path)

                # 이미지 클립 + Lo-Fi 느낌 (약간 비네트)
                clip_path = QPainterPath()
                clip_path.addRoundedRect(slot_rect, 2, 2)
                p.setClipPath(clip_path)
                lofi_pix = _cv_to_pixmap(_apply_lofi_lut(self.photos[i]))
                p.drawPixmap(slot_rect.toRect(), lofi_pix)
                p.setClipping(False)

                # 폴라로이드 하단 여백에 번호
                p.setFont(QFont("Courier", 8))
                p.setPen(QColor(100, 80, 50))
                note_rect = QRectF(pol_rect.x(), slot_rect.bottom() + 1,
                                   pol_rect.width(), 14)
                p.drawText(note_rect, Qt.AlignCenter, f"frame  0{i+1}")

            else:
                # 빈 슬롯: 크림 배경 + 긁힌 테두리
                empty_path = QPainterPath()
                empty_path.addRoundedRect(slot_rect, 3, 3)
                p.fillPath(empty_path, QBrush(QColor(50, 40, 26)))

                # 대시 테두리
                pen_dash = QPen(QColor(150, 120, 70, 120), 1.2, Qt.DashLine)
                pen_dash.setDashPattern([4, 5])
                p.setPen(pen_dash)
                p.setBrush(Qt.NoBrush)
                ip = QPainterPath()
                ip.addRoundedRect(slot_rect.adjusted(3, 3, -3, -3), 2, 2)
                p.drawPath(ip)

                # 중앙 번호
                p.setFont(QFont("Georgia", 16, QFont.Bold))
                p.setPen(QColor(140, 110, 60, 150))
                p.drawText(slot_rect, Qt.AlignCenter, f"0{i+1}")

        # ── 하단 필름 진행 표시
        taken   = len(self.photos)
        dot_y   = h - foot_h // 2
        seg_w_bar = (w - 60) // n
        for i in range(n):
            bx1 = 30 + i * seg_w_bar + 2
            br  = QRectF(bx1, dot_y - 5, seg_w_bar - 4, 10)
            bp  = QPainterPath()
            bp.addRoundedRect(br, 5, 5)
            if i < taken:
                grad_seg = QLinearGradient(bx1, 0, bx1 + seg_w_bar, 0)
                grad_seg.setColorAt(0.0, C_AMBER2)
                grad_seg.setColorAt(1.0, C_AMBER)
                p.fillPath(bp, QBrush(grad_seg))
            else:
                p.fillPath(bp, QBrush(QColor(45, 36, 22)))
                p.setPen(QPen(QColor(100, 80, 45, 80), 1))
                p.drawPath(bp)

        p.setFont(QFont("Courier", 7))
        p.setPen(QColor(130, 105, 60, 160))
        p.drawText(QRectF(0, dot_y + 8, w, 14), Qt.AlignCenter,
                   f"{taken} / {n}  exposed")

        p.end()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CameraView – Lo-Fi Brown 스타일
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CameraView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frame_pix:        QPixmap | None = None
        self.state:            str            = STATE_WAITING
        self.taken:            int            = 0
        self.countdown_start:  float | None   = None
        self.flash_start:      float | None   = None
        self.gesture:          str | None     = None
        self.gesture_start:    float | None   = None
        self.drawing_color:    QColor         = PEN_COLORS_QC[0]
        self.review_pix:       QPixmap | None = None
        self.review_video_pix: QPixmap | None = None
        self._tick = 0

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()
        self._tick += 1

        if self.state == STATE_REVIEW:
            self._draw_review(p, w, h)
            p.end()
            return

        # 카메라 프레임
        if self.frame_pix:
            p.drawPixmap(0, 0, w, h, self.frame_pix)
        else:
            p.fillRect(0, 0, w, h, QBrush(C_BG0))

        # Lo-Fi 비네트 (모서리 강하게)
        self._draw_vignette(p, w, h)

        # 필름 그레인 오버레이
        self._draw_grain_overlay(p, w, h)

        self._draw_top_hud(p, w, h)

        if self.state == STATE_COUNTDOWN and self.countdown_start:
            self._draw_countdown(p, w, h)
        elif self.state == STATE_FLASH and self.flash_start:
            self._draw_flash(p, w, h)
        elif self.state == STATE_WAITING:
            self._draw_bottom_hud(p, w, h)

        p.end()

    def _draw_vignette(self, p: QPainter, w: int, h: int):
        grad = QRadialGradient(w / 2, h / 2, max(w, h) * 0.68)
        grad.setColorAt(0.45, QColor(0, 0, 0,   0))
        grad.setColorAt(1.0,  QColor(12, 8, 4, 160))
        p.fillRect(0, 0, w, h, QBrush(grad))

    def _draw_grain_overlay(self, p: QPainter, w: int, h: int):
        """가벼운 필름 노이즈 수평선 스트라이프"""
        rng = np.random.default_rng(self._tick % 60)
        p.setPen(QPen(QColor(220, 180, 100, 10), 1))
        for _ in range(8):
            yy = int(rng.integers(0, h))
            p.drawLine(0, yy, w, yy)

    # ── 상단 HUD ─────────────────────────────────
    def _draw_top_hud(self, p: QPainter, w: int, h: int):
        hud_h = 54

        # 따뜻한 다크 바
        grad = QLinearGradient(0, 0, 0, hud_h)
        grad.setColorAt(0.0, QColor(28, 22, 13, 230))
        grad.setColorAt(1.0, QColor(40, 32, 18, 200))
        p.fillRect(0, 0, w, hud_h, QBrush(grad))
        _draw_scratchy_line(p, 0, hud_h, w, hud_h)

        cy = hud_h // 2

        # 촬영 진행 – 원형 도트 (아날로그 느낌)
        for i in range(TOTAL_SHOTS):
            cx_dot = 24 + i * 32
            if i < self.taken:
                p.setBrush(QBrush(C_AMBER))
                p.setPen(QPen(C_AMBER2, 1))
                p.drawEllipse(QPointF(cx_dot, cy), 10, 10)
                # 체크 느낌 크로스
                p.setPen(QPen(C_BG0, 2))
                p.drawLine(int(cx_dot - 5), cy, int(cx_dot + 5), cy)
            else:
                p.setBrush(QBrush(QColor(55, 44, 26)))
                p.setPen(QPen(QColor(120, 95, 55), 1))
                p.drawEllipse(QPointF(cx_dot, cy), 10, 10)
                p.setFont(QFont("Courier", 7))
                p.setPen(QColor(120, 95, 55))
                p.drawText(QRectF(cx_dot - 10, cy - 8, 20, 16),
                           Qt.AlignCenter, f"0{i+1}")

        # 중앙 타이틀
        p.setFont(QFont("Georgia", 11, QFont.Bold))
        p.setPen(C_CREAM)
        p.drawText(QRectF(0, 0, w, hud_h), Qt.AlignCenter,
                   f"take  {min(self.taken + 1, TOTAL_SHOTS)}  of  {TOTAL_SHOTS}")

        # 펜 색상 (오른쪽)
        px, py_pen = w - 32, cy
        p.setBrush(QBrush(C_BG1))
        p.setPen(QPen(QColor(160, 130, 70), 1))
        p.drawEllipse(QPointF(px, py_pen), 17, 17)
        p.setBrush(QBrush(self.drawing_color))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(px, py_pen), 11, 11)
        p.setFont(QFont("Courier", 6))
        p.setPen(C_GRAY1)
        p.drawText(QRectF(w - 74, py_pen + 14, 44, 12), Qt.AlignCenter, "ink")

    # ── 카운트다운 ────────────────────────────────
    def _draw_countdown(self, p: QPainter, w: int, h: int):
        elapsed  = time.time() - self.countdown_start
        num_show = max(1, int(COUNTDOWN_SEC - elapsed) + 1)
        progress = 1.0 - (elapsed % 1.0)
        cx, cy   = w // 2, h // 2
        R = 88

        # 어두운 원형 배경
        rad_bg = QRadialGradient(cx, cy, 140)
        rad_bg.setColorAt(0.0, QColor(20, 15, 8, 230))
        rad_bg.setColorAt(1.0, QColor(20, 15, 8,   0))
        p.setBrush(QBrush(rad_bg))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), 140, 140)

        # 트랙 링 (오래된 느낌)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(90, 72, 40, 80), 10, Qt.SolidLine, Qt.RoundCap))
        p.drawEllipse(QPointF(cx, cy), R, R)

        # 진행 호 (앰버)
        p.setPen(QPen(C_AMBER, 10, Qt.SolidLine, Qt.RoundCap))
        arc_rect = QRect(cx - R, cy - R, R * 2, R * 2)
        p.drawArc(arc_rect, 90 * 16, -int(360 * progress) * 16)

        # 외곽 링 (러스트)
        p.setPen(QPen(QColor(180, 85, 40, 60), 2))
        p.drawEllipse(QPointF(cx, cy), R + 10, R + 10)

        # 숫자 (세리프 폰트)
        p.setFont(QFont("Georgia", 70, QFont.Bold))
        # 텍스트 그림자
        p.setPen(QColor(0, 0, 0, 80))
        p.drawText(QRectF(cx - 65, cy - 58, 132, 120).translated(3, 3),
                   Qt.AlignCenter, str(num_show))
        p.setPen(C_CREAM)
        p.drawText(QRectF(cx - 65, cy - 58, 132, 120),
                   Qt.AlignCenter, str(num_show))

        # 하단 라벨
        p.setFont(QFont("Courier", 8))
        p.setPen(QColor(200, 160, 80, 200))
        p.drawText(QRectF(cx - 80, cy + R + 16, 160, 18),
                   Qt.AlignCenter, "developing film...")

    # ── 플래시 ────────────────────────────────────
    def _draw_flash(self, p: QPainter, w: int, h: int):
        ratio = max(0.0, 1.0 - (time.time() - self.flash_start) / FLASH_SEC)
        # 따뜻한 앰버/오렌지 플래시
        p.fillRect(0, 0, w, h,
                   QBrush(QColor(255, 220, 140, int(ratio * 220))))

    # ── 하단 제스처 HUD ───────────────────────────
    def _draw_bottom_hud(self, p: QPainter, w: int, h: int):
        hud2_h = 58
        hud2_y = h - hud2_h

        grad = QLinearGradient(0, hud2_y, 0, h)
        grad.setColorAt(0.0, QColor(25, 19, 11, 200))
        grad.setColorAt(1.0, QColor(35, 27, 16, 220))
        p.fillRect(0, hud2_y, w, hud2_h, QBrush(grad))
        _draw_scratchy_line(p, 0, hud2_y, w, hud2_y)

        now = time.time()

        if self.gesture == 'peace' and self.gesture_start is not None:
            held  = now - self.gesture_start
            ratio = min(held / GESTURE_HOLD, 1.0)
            bx1, bx2 = 20, w - 20
            by = h - 18

            # 트랙 (낡은 느낌)
            track = QRectF(bx1, by - 6, bx2 - bx1, 12)
            tp = QPainterPath()
            tp.addRoundedRect(track, 6, 6)
            p.fillPath(tp, QBrush(QColor(40, 32, 18)))
            p.setPen(QPen(QColor(140, 110, 55, 60), 1))
            p.drawPath(tp)

            if ratio > 0:
                fill = QRectF(bx1, by - 6, (bx2 - bx1) * ratio, 12)
                fp = QPainterPath()
                fp.addRoundedRect(fill, 6, 6)
                grad_fill = QLinearGradient(bx1, 0, bx2, 0)
                grad_fill.setColorAt(0.0, C_AMBER2)
                grad_fill.setColorAt(1.0, C_AMBER)
                p.fillPath(fp, QBrush(grad_fill))

            p.setFont(QFont("Georgia", 9, QFont.Bold))
            p.setPen(C_CREAM)
            p.drawText(QRectF(bx1, hud2_y + 8, bx2 - bx1, 20),
                       Qt.AlignCenter, "✌  hold  to  snap...")
        else:
            hints = [("✌", "peace", "snap"), ("☝", "hand", "draw"),
                     ("✊", "fist", "ink"), ("🖐", "open", "clear")]
            seg_w = w // len(hints)
            for hi, (icon, raw, label) in enumerate(hints):
                active = self.gesture is not None and self.gesture == raw

                # 버튼 배경
                btn_rect = QRectF(hi * seg_w + 6, hud2_y + 6,
                                  seg_w - 12, hud2_h - 12)
                btn_p = QPainterPath()
                btn_p.addRoundedRect(btn_rect, 6, 6)

                if active:
                    p.fillPath(btn_p, QBrush(QColor(100, 75, 35, 160)))
                    p.setPen(QPen(C_AMBER, 1))
                    p.drawPath(btn_p)
                else:
                    p.fillPath(btn_p, QBrush(QColor(38, 30, 17, 130)))
                    p.setPen(QPen(QColor(100, 80, 45, 55), 1))
                    p.drawPath(btn_p)

                col_main = C_AMBER if active else C_TAN
                col_sub  = QColor(200, 160, 80, 180) if active else QColor(100, 80, 50)

                p.setFont(QFont("Georgia", 9, QFont.Bold if active else QFont.Normal))
                p.setPen(col_main)
                p.drawText(QRectF(hi * seg_w, hud2_y + 8, seg_w, 22),
                           Qt.AlignCenter, f"{icon}  {raw}")
                p.setFont(QFont("Courier", 7))
                p.setPen(col_sub)
                p.drawText(QRectF(hi * seg_w, hud2_y + 34, seg_w, 16),
                           Qt.AlignCenter, label)

    # ── 리뷰 화면 ─────────────────────────────────
    def _draw_review(self, p: QPainter, w: int, h: int):
        # 따뜻한 다크 배경
        grad_bg = QLinearGradient(0, 0, 0, h)
        grad_bg.setColorAt(0.0, QColor(32, 25, 14))
        grad_bg.setColorAt(1.0, QColor(20, 15,  8))
        p.fillRect(0, 0, w, h, QBrush(grad_bg))

        # 필름 스트라이프 텍스처 (수평 미세선)
        p.setPen(QPen(QColor(80, 60, 30, 18), 1))
        for yy in range(0, h, 5):
            p.drawLine(0, yy, w, yy)

        title_h = 62
        foot_h  = 54

        # ── 타이틀 바
        tgrad = QLinearGradient(0, 0, 0, title_h)
        tgrad.setColorAt(0.0, QColor(48, 36, 18, 240))
        tgrad.setColorAt(1.0, QColor(32, 25, 12, 240))
        p.fillRect(0, 0, w, title_h, QBrush(tgrad))
        _draw_scratchy_line(p, 0, title_h, w, title_h)

        p.setFont(QFont("Georgia", 17, QFont.Bold))
        p.setPen(QColor(0, 0, 0, 60))
        p.drawText(QRectF(2, 2, w, title_h), Qt.AlignCenter, "your  4-cut")
        p.setPen(C_AMBER)
        p.drawText(QRectF(0, 0, w, title_h), Qt.AlignCenter, "your  4-cut")

        # 날짜/시간 (우측)
        p.setFont(QFont("Courier", 8))
        p.setPen(QColor(160, 130, 70, 160))
        date_str = datetime.now().strftime("%Y . %m . %d")
        p.drawText(QRectF(w - 190, 0, 178, title_h),
                   Qt.AlignVCenter | Qt.AlignRight, date_str)

        # ── 하단 안내 바
        fgrad = QLinearGradient(0, h - foot_h, 0, h)
        fgrad.setColorAt(0.0, QColor(38, 30, 14, 230))
        fgrad.setColorAt(1.0, QColor(25, 19,  9, 230))
        p.fillRect(0, h - foot_h, w, foot_h, QBrush(fgrad))
        _draw_scratchy_line(p, 0, h - foot_h, w, h - foot_h)
        p.setFont(QFont("Courier", 8))
        p.setPen(C_GRAY1)
        p.drawText(QRectF(0, h - foot_h, w, foot_h),
                   Qt.AlignCenter, "open palm  ·  new session          esc  ·  quit")

        avail_y = title_h + 20
        avail_h = h - title_h - foot_h - 40
        half_w  = int(w * 0.42)
        gap     = 30

        # ── 콜라주 (왼쪽)
        if self.review_pix:
            col_rect = self._fit_rect(self.review_pix, 18, avail_y, half_w - 18, avail_h)

            # 따뜻한 그림자
            p.fillRect(col_rect.translated(6, 6).toRect(),
                       QBrush(QColor(0, 0, 0, 100)))

            # 크림 폴라로이드 테두리
            pad = 10
            pol = col_rect.adjusted(-pad, -pad, pad, pad + 20)
            pol_path = QPainterPath()
            pol_path.addRoundedRect(pol, 3, 3)
            p.fillPath(pol_path, QBrush(QColor(215, 200, 165)))
            p.setPen(QPen(QColor(160, 135, 85), 1))
            p.drawPath(pol_path)

            # 이미지
            p.drawPixmap(col_rect.toRect(), self.review_pix)

            # 앰버 얇은 테두리
            p.setPen(QPen(C_AMBER2, 1.5))
            p.setBrush(Qt.NoBrush)
            bp = QPainterPath()
            bp.addRect(col_rect)
            p.drawPath(bp)

            # 하단 폴라로이드 텍스트
            p.setFont(QFont("Courier", 8))
            p.setPen(QColor(90, 72, 45))
            p.drawText(QRectF(pol.x(), col_rect.bottom() + 4,
                               pol.width(), 18),
                       Qt.AlignCenter, "saved  ✓")

        # ── 영상 (오른쪽)
        if self.review_video_pix:
            vid_x = half_w + gap
            vid_w_avail = w - vid_x - 18
            vid_rect = self._fit_rect(self.review_video_pix, vid_x, avail_y,
                                      vid_w_avail, avail_h)

            # 따뜻한 그림자
            p.fillRect(vid_rect.translated(6, 6).toRect(),
                       QBrush(QColor(0, 0, 0, 100)))

            # 이미지 클립
            clip2 = QPainterPath()
            clip2.addRoundedRect(vid_rect, 2, 2)
            p.setClipPath(clip2)
            p.drawPixmap(vid_rect.toRect(), self.review_video_pix)
            p.setClipping(False)

            # 필름 스트라이프 오버레이 (얇은 선)
            p.setPen(QPen(QColor(220, 170, 80, 12), 1))
            for yy in range(int(vid_rect.top()), int(vid_rect.bottom()), 4):
                p.drawLine(int(vid_rect.left()), yy, int(vid_rect.right()), yy)

            # 러스트 테두리
            p.setPen(QPen(C_RUST, 1.5))
            p.setBrush(Qt.NoBrush)
            vp = QPainterPath()
            vp.addRoundedRect(vid_rect, 2, 2)
            p.drawPath(vp)

            # REPLAY 뱃지
            badge_w, badge_h = 76, 20
            badge = QRectF(vid_rect.x() + 8, vid_rect.y() + 8, badge_w, badge_h)
            bpath = QPainterPath()
            bpath.addRoundedRect(badge, 2, 2)
            p.fillPath(bpath, QBrush(QColor(30, 22, 10, 190)))
            p.setPen(QPen(C_AMBER2, 1))
            p.drawPath(bpath)
            p.setFont(QFont("Courier", 7, QFont.Bold))
            p.setPen(C_AMBER)
            p.drawText(badge, Qt.AlignCenter, "▶  replay")
        else:
            vid_x = half_w + gap
            vid_w_avail = w - vid_x - 18
            placeholder = QRectF(vid_x, avail_y, vid_w_avail, avail_h)
            _fill_warm(p, placeholder)
            p.setFont(QFont("Courier", 9))
            p.setPen(QColor(120, 95, 55, 120))
            p.drawText(placeholder, Qt.AlignCenter, "no film")

        # 중앙 세로 구분선 (손으로 긁힌 느낌)
        sep_x = half_w + gap // 2
        p.setPen(QPen(QColor(150, 120, 60, 35), 1, Qt.DashLine))
        p.drawLine(sep_x, avail_y, sep_x, avail_y + avail_h)

    @staticmethod
    def _fit_rect(pix: QPixmap, x0: int, y0: int, avail_w: int, avail_h: int) -> QRectF:
        pw, ph = pix.width(), pix.height()
        scale  = min(avail_w / pw, avail_h / ph)
        nw, nh = int(pw * scale), int(ph * scale)
        xoff   = x0 + (avail_w - nw) // 2
        yoff   = y0 + (avail_h - nh) // 2
        return QRectF(xoff, yoff, nw, nh)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PhotoboothWindow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PhotoboothWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("4-CUT PHOTOBOOTH  v4  lo-fi")
        self.setStyleSheet("background: rgb(22, 17, 9);")

        self.video = None
        for idx in range(20):
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                ret, tf = cap.read()
                if ret and tf is not None and len(tf.shape) == 3:
                    self.video = cap
                    print(f"카메라: /dev/video{idx}")
                    break
            cap.release()
        if self.video is None:
            print("카메라를 찾을 수 없습니다.")
            sys.exit(1)

        opts = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=RunningMode.VIDEO,
        )
        self.recognizer = GestureRecognizer.create_from_options(opts)

        self.state            = STATE_WAITING
        self.photos:    list  = []
        self.photo_masks: list = []
        self.draw_canvas      = None
        self.prev_x           = None
        self.prev_y           = None
        self.color_idx        = 0
        self.draw_color_cv    = PEN_COLORS_CV[0]
        self.draw_color_qc    = PEN_COLORS_QC[0]
        self.line_thickness   = 5
        self.last_gesture     = None
        self.gesture_start    = None
        self.countdown_start  = None
        self.flash_start      = None
        self.frame_index      = 0
        self.fps              = 30
        self._review_cap      = None
        self._session_dir     = None
        self._vid_tmp         = os.path.join(SAVE_DIR, '_recording_tmp_v4.avi')
        self._out_writer      = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.cam_view = CameraView()
        self.strip    = StripPanel()
        layout.addWidget(self.cam_view)
        layout.addWidget(self.strip)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_frame)
        self.timer.start(1000 // self.fps)

        self.showFullScreen()
        print("=" * 50)
        print("  4-CUT PHOTOBOOTH  v4  (Lo-Fi Brown)")
        print("=" * 50)
        print("  peace (0.8s) : 촬영")
        print("  hand         : 그리기")
        print("  fist         : 잉크 색상 변경")
        print("  open         : 지우기 / 리셋")
        print("  ESC / Q      : 종료")
        print("=" * 50)

    def _on_frame(self):
        if _exit_requested:
            self._cleanup()
            QApplication.quit()
            return

        ret, frame = self.video.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        now   = time.time()

        if self.draw_canvas is None:
            self.draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        gesture = None
        if self.state not in (STATE_REVIEW,):
            # Lo-Fi 색보정 적용 (뷰에만, 원본 저장은 따로)
            frame_display = _apply_lofi_lut(frame)

            img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            ts_ms    = int(self.frame_index * 1000 / self.fps)
            self.frame_index += 1
            result = self.recognizer.recognize_for_video(mp_image, ts_ms)

            if result.hand_landmarks:
                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    raw_g   = result.gestures[i][0].category_name if result.gestures else 'None'
                    gesture = _GESTURE_MAP.get(raw_g, None)
                    ix = int(hand_landmarks[8].x * w)
                    iy = int(hand_landmarks[8].y * h)

                    if gesture == 'fist':
                        if self.last_gesture != 'fist':
                            self.color_idx     = (self.color_idx + 1) % len(PEN_COLORS_CV)
                            self.draw_color_cv = PEN_COLORS_CV[self.color_idx]
                            self.draw_color_qc = PEN_COLORS_QC[self.color_idx]
                        self.prev_x = self.prev_y = None
                    elif gesture == 'open':
                        self.draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                        self.prev_x = self.prev_y = None
                    elif gesture == 'peace':
                        self.prev_x = self.prev_y = None
                    else:
                        if self.state != STATE_COUNTDOWN:
                            if self.prev_x is not None and self.prev_y is not None:
                                cv2.line(self.draw_canvas,
                                         (self.prev_x, self.prev_y), (ix, iy),
                                         self.draw_color_cv, self.line_thickness)
                            self.prev_x, self.prev_y = ix, iy
                        else:
                            self.prev_x = self.prev_y = None

                    # 랜드마크 (갈색 계열)
                    for conn in HAND_CONNECTIONS:
                        s = hand_landmarks[conn[0]]; e = hand_landmarks[conn[1]]
                        cv2.line(frame_display,
                                 (int(s.x*w), int(s.y*h)),
                                 (int(e.x*w), int(e.y*h)), (60, 130, 180), 1)
                    for lm in hand_landmarks:
                        cv2.circle(frame_display,
                                   (int(lm.x*w), int(lm.y*h)), 3, (50, 160, 210), -1)

                    if gesture in ('fist', 'open', 'peace'):
                        cv2.circle(frame_display, (ix, iy), 14, (160, 130, 60), 1)
                        cv2.circle(frame_display, (ix, iy),  4, (160, 130, 60), -1)
                    else:
                        cv2.circle(frame_display, (ix, iy),
                                   self.line_thickness + 6, self.draw_color_cv, 2)
                        cv2.circle(frame_display, (ix, iy), 3, self.draw_color_cv, -1)
            else:
                self.prev_x = self.prev_y = None
        else:
            frame_display = frame

        if gesture == 'peace':
            if self.last_gesture != 'peace':
                self.gesture_start = now
            elif now - self.gesture_start >= GESTURE_HOLD:
                if self.state == STATE_WAITING:
                    self.state = STATE_COUNTDOWN
                    self.countdown_start = now
                self.gesture_start = now
        else:
            self.gesture_start = None

        if gesture == 'open' and self.state == STATE_REVIEW:
            if self.last_gesture != 'open':
                self._reset_session(h, w)

        self.last_gesture = gesture

        # ── 상태 처리
        if self.state == STATE_COUNTDOWN:
            if now - self.countdown_start >= COUNTDOWN_SEC:
                mask = cv2.cvtColor(self.draw_canvas, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                mask_inv  = cv2.bitwise_not(mask)
                frame_bg  = cv2.bitwise_and(frame, frame, mask=mask_inv)
                canvas_fg = cv2.bitwise_and(self.draw_canvas, self.draw_canvas, mask=mask)
                shot = cv2.add(frame_bg, canvas_fg)
                self.photos.append(shot.copy())
                self.photo_masks.append(
                    _make_inpaint_mask(self.draw_canvas, self.line_thickness))
                self.draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                self.state = STATE_FLASH
                self.flash_start = now
                print(f"[{len(self.photos)}/{TOTAL_SHOTS}] 촬영!")
                if len(self.photos) >= TOTAL_SHOTS:
                    self._session_dir = os.path.join(
                        SAVE_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
                    _save_final(self.photos, masks=self.photo_masks,
                                session_dir=self._session_dir)

        elif self.state == STATE_FLASH:
            if now - self.flash_start >= FLASH_SEC:
                if len(self.photos) >= TOTAL_SHOTS:
                    collage = _make_final_collage(self.photos)
                    self.cam_view.review_pix = (
                        _cv_to_pixmap(collage) if collage is not None else None)
                    self.state = STATE_REVIEW
                    _vid_play = os.path.join(SAVE_DIR, '_recording_play_v4.avi')
                    if self._out_writer and self._out_writer is not False:
                        self._out_writer.release()
                        self._out_writer = None
                    if os.path.exists(self._vid_tmp):
                        import shutil as _sh
                        _sh.copy2(self._vid_tmp, _vid_play)
                        self._review_cap = cv2.VideoCapture(_vid_play)
                else:
                    self.state = STATE_WAITING

        # ── 그리기 합성 (display frame 기준)
        if self.state not in (STATE_REVIEW,):
            mask = cv2.cvtColor(self.draw_canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            mask_inv  = cv2.bitwise_not(mask)
            frame_bg  = cv2.bitwise_and(frame_display, frame_display, mask=mask_inv)
            canvas_fg = cv2.bitwise_and(self.draw_canvas, self.draw_canvas, mask=mask)
            frame_display = cv2.add(frame_bg, canvas_fg)

        # ── 녹화 (원본 frame 기준)
        if self._out_writer is None and self.state not in (STATE_REVIEW,):
            _vh, _vw = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(self._vid_tmp, fourcc, self.fps, (_vw, _vh))
            if writer.isOpened():
                self._out_writer = writer
                print(f"[녹화 시작] {self._vid_tmp}")
            else:
                writer.release()
                self._out_writer = False
        if self._out_writer and self._out_writer is not False and self.state not in (STATE_REVIEW,):
            self._out_writer.write(frame)

        # ── 리뷰 영상 업데이트
        if self.state == STATE_REVIEW and self._review_cap:
            ret_v, vframe = self._review_cap.read()
            if not ret_v:
                self._review_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_v, vframe = self._review_cap.read()
            if ret_v and vframe is not None:
                # 리뷰 영상도 Lo-Fi 보정 적용
                self.cam_view.review_video_pix = _cv_to_pixmap(
                    _apply_lofi_lut(vframe))

        # ── 위젯 업데이트
        self.cam_view.frame_pix       = _cv_to_pixmap(frame_display)
        self.cam_view.state           = self.state
        self.cam_view.taken           = len(self.photos)
        self.cam_view.countdown_start = self.countdown_start
        self.cam_view.flash_start     = self.flash_start
        self.cam_view.gesture         = gesture
        self.cam_view.gesture_start   = self.gesture_start
        self.cam_view.drawing_color   = self.draw_color_qc

        self.strip.setPhotos(self.photos)
        self.cam_view.update()

    def _reset_session(self, h: int, w: int):
        self.photos = []
        self.photo_masks = []
        self.draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        self.cam_view.review_pix = None
        self.cam_view.review_video_pix = None
        self.state = STATE_WAITING
        if self._review_cap:
            self._review_cap.release()
            self._review_cap = None
        for tmp in ['_recording_play_v4.avi']:
            path = os.path.join(SAVE_DIR, tmp)
            if os.path.exists(path):
                try: os.remove(path)
                except Exception: pass
        if self._out_writer and self._out_writer is not False:
            self._out_writer.release()
        self._out_writer = None
        self._session_dir = None
        print("세션 리셋")

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
            self._cleanup()
            QApplication.quit()

    def _cleanup(self):
        self.timer.stop()
        if self._review_cap:
            self._review_cap.release()
        if self._out_writer and self._out_writer is not False:
            self._out_writer.release()
        if self.video:
            self.video.release()
        try:
            self.recognizer.close()
        except Exception:
            pass


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("4-CUT Photobooth v4")
    win = PhotoboothWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
