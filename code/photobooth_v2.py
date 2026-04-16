#!/usr/bin/env python3
"""
PyQt5-based 인생네컷 포토부스
밝고 파스텔한 인생네컷 스타일 – QPainter 기반 고품질 UI
"""

import os, sys, time, signal
import numpy as np

# cv2를 먼저 import (내부적으로 QT_QPA_PLATFORM_PLUGIN_PATH를 자신의 경로로 덮어씀)
import cv2
import mediapipe as mp
from datetime import datetime

# cv2 import 이후에 Qt 플러그인 경로를 시스템 Qt5로 복원
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
FLASH_SEC     = 0.5
GESTURE_HOLD  = 0.8
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'photobooth_output')
os.makedirs(SAVE_DIR, exist_ok=True)

# ── 테마 컬러 (QColor) ─────────────────────────────────────────
C_BG     = QColor(245, 235, 255)
C_DARK   = QColor(35,  25,  55,  205)
C_ACCENT = QColor(150,  50, 255)
C_PINK   = QColor(230, 130, 255)
C_GOLD   = QColor(255, 200,  50)
C_WHITE  = QColor(255, 255, 255)
C_HINT   = QColor(160, 130, 200)
C_STRIP  = QColor(248, 240, 255)

PEN_COLORS_QC = [
    QColor(255,  50,  50), QColor(255, 150,   0), QColor(255, 220,   0),
    QColor( 60, 200,  60), QColor(  0, 120, 255), QColor(180,  60, 200),
    QColor(255, 255, 255),
]
PEN_COLORS_CV = [
    ( 50,  50, 255), (  0, 150, 255), (  0, 220, 255),
    ( 60, 200,  60), (255, 120,   0), (200,  60, 180),
    (255, 255, 255),
]

# ── 상태 ───────────────────────────────────────────────────────
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


def _make_final_collage(photos):
    if not photos:
        return None
    ph, pw = photos[0].shape[:2]
    margin = 30; top_pad = 90; bottom_pad = 60
    col_w = pw + margin * 2
    col_h = top_pad + (ph + margin) * TOTAL_SHOTS + margin + bottom_pad
    collage = np.full((col_h, col_w, 3), (245, 235, 255), dtype=np.uint8)
    title = "4-CUT PHOTO"
    tw = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0][0]
    cv2.putText(collage, title, (col_w//2 - tw//2, 62),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (150, 50, 255), 2, cv2.LINE_AA)
    for i, photo in enumerate(photos):
        y0 = top_pad + margin + i * (ph + margin)
        collage[y0:y0+ph, margin:margin+pw] = photo
        cv2.rectangle(collage, (margin-5, y0-5), (margin+pw+5, y0+ph+5), (200, 120, 255), 3)
    date_str = datetime.now().strftime("%Y.%m.%d")
    dw = cv2.getTextSize(date_str, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)[0][0]
    cv2.putText(collage, date_str, (col_w//2 - dw//2, col_h - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (50, 20, 80), 1, cv2.LINE_AA)
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
#  StripPanel – 우측 4컷 미리보기
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

        # ── 배경 그라디언트
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor(250, 244, 255))
        grad.setColorAt(1.0, QColor(235, 220, 252))
        p.fillRect(0, 0, w, h, QBrush(grad))

        # 미세 도트 패턴
        p.setPen(QPen(QColor(215, 205, 235), 1))
        for yy in range(0, h, 20):
            for xx in range(0, w, 20):
                p.drawPoint(xx, yy)

        title_h = 68
        foot_h  = 52
        margin  = 14
        slot_w  = w - margin * 2
        avail_h = h - title_h - foot_h - margin * (n + 1)
        slot_h  = max(avail_h // n, 40)

        # ── 타이틀 바 - 퍼플 그라디언트
        grad_t = QLinearGradient(0, 0, w, title_h)
        grad_t.setColorAt(0.0, QColor(80,  20, 140))
        grad_t.setColorAt(1.0, QColor(140, 40, 220))
        path_t = QPainterPath()
        path_t.addRect(QRectF(0, 0, w, title_h))
        p.fillPath(path_t, QBrush(grad_t))

        # 타이틀 텍스트
        p.setFont(QFont("Arial", 15, QFont.Bold))
        p.setPen(C_WHITE)
        p.drawText(QRectF(0, 0, w, title_h), Qt.AlignCenter, "✦  4-CUT  ✦")

        # 구분선
        p.setPen(QPen(C_PINK, 2))
        p.drawLine(0, title_h, w, title_h)

        # ── 사진 슬롯
        for i in range(n):
            y0 = title_h + margin + i * (slot_h + margin)
            x0 = margin
            slot_rect = QRectF(x0, y0, slot_w, slot_h)

            if i < len(self.photos) and self.photos[i] is not None:
                pad = 6
                # 그림자
                shadow_path = QPainterPath()
                shadow_path.addRoundedRect(slot_rect.translated(4, 4).adjusted(-pad, -pad, pad, pad), 10, 10)
                p.fillPath(shadow_path, QBrush(QColor(120, 90, 160, 70)))

                # 흰 폴라로이드 테두리
                pol_rect = slot_rect.adjusted(-pad, -pad, pad, pad)
                pol_path = QPainterPath()
                pol_path.addRoundedRect(pol_rect, 8, 8)
                p.fillPath(pol_path, QBrush(C_WHITE))
                p.setPen(QPen(C_PINK, 1.5))
                p.drawPath(pol_path)

                # 이미지 (클립)
                clip_path = QPainterPath()
                clip_path.addRoundedRect(slot_rect, 5, 5)
                p.setClipPath(clip_path)
                pix = _cv_to_pixmap(self.photos[i])
                p.drawPixmap(slot_rect.toRect(), pix)
                p.setClipping(False)

                # 완료 배지 (번호 원)
                bp = QPointF(x0 + slot_w + pad - 5, y0 + slot_h + pad - 5)
                grad_b = QRadialGradient(bp, 14)
                grad_b.setColorAt(0.0, C_PINK)
                grad_b.setColorAt(1.0, C_ACCENT)
                p.setBrush(QBrush(grad_b))
                p.setPen(QPen(C_WHITE, 1.5))
                p.drawEllipse(bp, 14, 14)
                p.setFont(QFont("Arial", 9, QFont.Bold))
                p.setPen(C_WHITE)
                p.drawText(QRectF(bp.x()-14, bp.y()-14, 28, 28), Qt.AlignCenter, str(i + 1))
            else:
                # 빈 슬롯: 대시 테두리
                empty_path = QPainterPath()
                empty_path.addRoundedRect(slot_rect, 10, 10)
                p.fillPath(empty_path, QBrush(QColor(238, 228, 252)))
                pen_dash = QPen(C_PINK, 1.8, Qt.DashLine)
                pen_dash.setDashPattern([4, 4])
                p.setPen(pen_dash)
                p.setBrush(Qt.NoBrush)
                p.drawPath(empty_path)

                # 번호 원
                cx = x0 + slot_w // 2
                cy = y0 + slot_h // 2
                p.setPen(QPen(C_PINK, 2))
                p.setBrush(QBrush(QColor(248, 238, 255)))
                p.drawEllipse(QPointF(cx, cy), 28, 28)
                p.setFont(QFont("Arial", 14, QFont.Bold))
                p.setPen(C_PINK)
                p.drawText(QRectF(cx - 22, cy - 17, 44, 34), Qt.AlignCenter, str(i + 1))

        # ── 하단 진행 도트
        taken   = len(self.photos)
        dot_y   = h - foot_h // 2
        spacing = 26
        start_x = (w - n * spacing) // 2 + spacing // 2
        for i in range(n):
            cx = start_x + i * spacing
            if i < taken:
                gd = QRadialGradient(cx, dot_y, 10)
                gd.setColorAt(0.0, C_PINK)
                gd.setColorAt(1.0, C_ACCENT)
                p.setBrush(QBrush(gd))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QPointF(cx, dot_y), 9, 9)
            else:
                p.setBrush(QBrush(QColor(210, 195, 232)))
                p.setPen(QPen(QColor(190, 160, 220), 1.5))
                p.drawEllipse(QPointF(cx, dot_y), 9, 9)

        p.end()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CameraView – 메인 카메라 + 오버레이
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CameraView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # ── 렌더링 데이터 (PhotoboothWindow 에서 매 프레임 갱신)
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

    # ── 메인 페인트 ───────────────────────────────
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        if self.state == STATE_REVIEW:
            self._draw_review(p, w, h)
            p.end()
            return

        # 카메라 프레임
        if self.frame_pix:
            p.drawPixmap(0, 0, w, h, self.frame_pix)
        else:
            grad = QLinearGradient(0, 0, 0, h)
            grad.setColorAt(0.0, QColor(230, 215, 250))
            grad.setColorAt(1.0, QColor(200, 180, 235))
            p.fillRect(0, 0, w, h, QBrush(grad))

        self._draw_top_hud(p, w, h)

        if self.state == STATE_COUNTDOWN and self.countdown_start:
            self._draw_countdown(p, w, h)
        elif self.state == STATE_FLASH and self.flash_start:
            self._draw_flash(p, w, h)
        elif self.state == STATE_WAITING:
            self._draw_bottom_hud(p, w, h)

        p.end()

    # ── 상단 HUD ─────────────────────────────────
    def _draw_top_hud(self, p: QPainter, w: int, h: int):
        hud_h = 56

        # 반투명 그라디언트 바
        grad = QLinearGradient(0, 0, 0, hud_h)
        grad.setColorAt(0.0, QColor(30, 15, 50, 215))
        grad.setColorAt(1.0, QColor(50, 25, 75, 185))
        p.fillRect(0, 0, w, hud_h, QBrush(grad))

        # 구분선
        p.setPen(QPen(C_PINK, 1))
        p.drawLine(0, hud_h, w, hud_h)

        cy = hud_h // 2

        # 진행 도트
        for i in range(TOTAL_SHOTS):
            cx_dot = 26 + i * 34
            if i < self.taken:
                gd = QRadialGradient(cx_dot, cy, 12)
                gd.setColorAt(0.0, C_PINK)
                gd.setColorAt(1.0, C_ACCENT)
                p.setBrush(QBrush(gd))
                p.setPen(QPen(QColor(255, 255, 255, 180), 1.5))
            else:
                p.setBrush(QBrush(QColor(70, 55, 95)))
                p.setPen(QPen(QColor(120, 90, 155), 1))
            p.drawEllipse(QPointF(cx_dot, cy), 11, 11)

        # Shot 레이블
        p.setFont(QFont("Arial", 9))
        p.setPen(QColor(175, 150, 215))
        shot_x = TOTAL_SHOTS * 34 + 20
        p.drawText(shot_x, cy + 5, f"SHOT  {min(self.taken+1, TOTAL_SHOTS)} / {TOTAL_SHOTS}")

        # 센터 타이틀
        p.setFont(QFont("Arial", 12, QFont.Bold))
        p.setPen(C_PINK)
        p.drawText(QRectF(0, 0, w, hud_h), Qt.AlignCenter, "✦  4-CUT PHOTOBOOTH  ✦")

        # 펜 색상 인디케이터
        px, py_pen = w - 40, cy
        gd_pen = QRadialGradient(px, py_pen, 15)
        lighter = self.drawing_color.lighter(140)
        gd_pen.setColorAt(0.0, lighter)
        gd_pen.setColorAt(1.0, self.drawing_color)
        p.setBrush(QBrush(gd_pen))
        p.setPen(QPen(C_WHITE, 2))
        p.drawEllipse(QPointF(px, py_pen), 14, 14)
        p.setFont(QFont("Arial", 7))
        p.setPen(QColor(155, 130, 195))
        p.drawText(QRectF(w - 82, py_pen - 7, 36, 14), Qt.AlignCenter, "pen")

    # ── 카운트다운 오버레이 ───────────────────────
    def _draw_countdown(self, p: QPainter, w: int, h: int):
        elapsed  = time.time() - self.countdown_start
        num_show = max(1, int(COUNTDOWN_SEC - elapsed) + 1)
        progress = 1.0 - (elapsed % 1.0)
        cx, cy   = w // 2, h // 2
        R = 96

        # 반투명 다크 원
        p.setBrush(QBrush(QColor(25, 12, 45, 200)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), 125, 125)

        # 링 트랙
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(80, 60, 110), 10, Qt.SolidLine, Qt.RoundCap))
        p.drawEllipse(QPointF(cx, cy), R, R)

        # 진행 호
        pen_arc = QPen(C_ACCENT, 10, Qt.SolidLine, Qt.RoundCap)
        p.setPen(pen_arc)
        arc_rect = QRect(cx - R, cy - R, R * 2, R * 2)
        p.drawArc(arc_rect, 90 * 16, -int(360 * progress) * 16)

        # 핑크 빛 링
        pen_glow = QPen(QColor(230, 130, 255, 160), 2)
        p.setPen(pen_glow)
        p.drawEllipse(QPointF(cx, cy), R + 8, R + 8)

        # 숫자
        p.setFont(QFont("Arial", 68, QFont.Bold))
        p.setPen(C_WHITE)
        p.drawText(QRectF(cx - 65, cy - 58, 130, 116), Qt.AlignCenter, str(num_show))

    # ── 플래시 ────────────────────────────────────
    def _draw_flash(self, p: QPainter, w: int, h: int):
        ratio = max(0.0, 1.0 - (time.time() - self.flash_start) / FLASH_SEC)
        p.fillRect(0, 0, w, h, QBrush(QColor(255, 255, 255, int(ratio * 230))))

    # ── 하단 제스처 HUD ───────────────────────────
    def _draw_bottom_hud(self, p: QPainter, w: int, h: int):
        hud2_h = 62
        hud2_y = h - hud2_h
        grad = QLinearGradient(0, hud2_y, 0, h)
        grad.setColorAt(0.0, QColor(30, 15, 50, 185))
        grad.setColorAt(1.0, QColor(50, 25, 75, 200))
        p.fillRect(0, hud2_y, w, hud2_h, QBrush(grad))
        p.setPen(QPen(C_PINK, 1))
        p.drawLine(0, hud2_y, w, hud2_y)

        now = time.time()

        if self.gesture == 'peace' and self.gesture_start is not None:
            # ── 홀드 프로그레스 바
            held  = now - self.gesture_start
            ratio = min(held / GESTURE_HOLD, 1.0)
            bx1, bx2 = 24, w - 24
            by = h - 20
            bar_rect = QRectF(bx1, by - 8, bx2 - bx1, 16)
            bar_path = QPainterPath()
            bar_path.addRoundedRect(bar_rect, 8, 8)

            p.fillPath(bar_path, QBrush(QColor(70, 55, 95)))

            if ratio > 0:
                fill_rect = QRectF(bx1, by - 8, (bx2 - bx1) * ratio, 16)
                fill_path = QPainterPath()
                fill_path.addRoundedRect(fill_rect, 8, 8)
                grad_fill = QLinearGradient(bx1, 0, bx2, 0)
                grad_fill.setColorAt(0.0, C_PINK)
                grad_fill.setColorAt(1.0, C_ACCENT)
                p.fillPath(fill_path, QBrush(grad_fill))

            p.setPen(QPen(C_PINK, 1))
            p.drawPath(bar_path)

            p.setFont(QFont("Arial", 9, QFont.Bold))
            p.setPen(C_WHITE)
            p.drawText(QRectF(bx1, hud2_y + 6, bx2 - bx1, 22),
                       Qt.AlignCenter, "✌  Hold  PEACE  to  capture...")
        else:
            hints = [("✌ PEACE", "capture"), ("☝ HAND", "draw"),
                     ("✊ FIST", "color"), ("🖐 OPEN", "clear")]
            seg_w = w // len(hints)
            for hi, (gn, gd_text) in enumerate(hints):
                cx_g = hi * seg_w + seg_w // 2
                raw_name = gn.split()[-1].lower()
                active   = self.gesture is not None and self.gesture == raw_name

                # 활성 세그먼트에 미세 하이라이트
                if active:
                    hl_path = QPainterPath()
                    hl_path.addRoundedRect(
                        QRectF(hi * seg_w + 4, hud2_y + 4, seg_w - 8, hud2_h - 8), 8, 8)
                    p.fillPath(hl_path, QBrush(QColor(150, 50, 255, 55)))

                col_gn = C_ACCENT if active else C_HINT
                col_gd_clr = QColor(160, 140, 195) if active else QColor(110, 90, 140)

                p.setFont(QFont("Arial", 9, QFont.Bold if active else QFont.Normal))
                p.setPen(col_gn)
                p.drawText(QRectF(hi * seg_w, hud2_y + 8, seg_w, 24),
                           Qt.AlignCenter, gn)
                p.setFont(QFont("Arial", 7))
                p.setPen(col_gd_clr)
                p.drawText(QRectF(hi * seg_w, hud2_y + 34, seg_w, 18),
                           Qt.AlignCenter, gd_text)

    # ── 리뷰 화면 ─────────────────────────────────
    def _draw_review(self, p: QPainter, w: int, h: int):
        # 배경 그라디언트
        grad_bg = QLinearGradient(0, 0, 0, h)
        grad_bg.setColorAt(0.0, QColor(28, 15, 45))
        grad_bg.setColorAt(1.0, QColor(18, 10, 35))
        p.fillRect(0, 0, w, h, QBrush(grad_bg))

        title_h = 64
        foot_h  = 58

        # ── 타이틀 바
        grad_t = QLinearGradient(0, 0, w, title_h)
        grad_t.setColorAt(0.0, QColor(70, 15, 120, 230))
        grad_t.setColorAt(1.0, QColor(120, 35, 200, 230))
        p.fillRect(0, 0, w, title_h, QBrush(grad_t))
        p.setPen(QPen(C_PINK, 1))
        p.drawLine(0, title_h, w, title_h)
        p.setFont(QFont("Arial", 18, QFont.Bold))
        p.setPen(C_GOLD)
        p.drawText(QRectF(0, 0, w, title_h), Qt.AlignCenter, "✨  YOUR  4-CUT  ✨")

        # ── 하단 안내 바
        grad_f = QLinearGradient(0, h - foot_h, w, h)
        grad_f.setColorAt(0.0, QColor(60, 15, 100, 215))
        grad_f.setColorAt(1.0, QColor(35, 20, 55, 215))
        p.fillRect(0, h - foot_h, w, foot_h, QBrush(grad_f))
        p.setPen(QPen(C_PINK, 1))
        p.drawLine(0, h - foot_h, w, h - foot_h)
        p.setFont(QFont("Arial", 9))
        p.setPen(C_WHITE)
        p.drawText(QRectF(0, h - foot_h, w, foot_h), Qt.AlignCenter,
                   "🖐  Open Palm : New Session    |    ESC : Quit")

        avail_y = title_h + 16
        avail_h = h - title_h - foot_h - 32
        half_w  = int(w * 0.44)
        gap     = 24

        # ── 콜라주 (왼쪽)
        if self.review_pix:
            col_rect = self._fit_rect(self.review_pix, 14, avail_y, half_w - 14, avail_h)
            # 그림자
            p.fillRect(col_rect.translated(5, 5).toRect(),
                       QBrush(QColor(0, 0, 0, 80)))
            # 폴라로이드 흰 배경
            pol = col_rect.adjusted(-10, -10, 10, 10)
            pol_path = QPainterPath()
            pol_path.addRoundedRect(pol, 12, 12)
            p.fillPath(pol_path, QBrush(C_WHITE))
            # 이미지
            p.drawPixmap(col_rect.toRect(), self.review_pix)
            # 액센트 테두리
            p.setPen(QPen(C_ACCENT, 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(pol, 12, 12)

        # ── 영상 (오른쪽)
        if self.review_video_pix:
            vid_x = half_w + gap
            vid_w = w - vid_x - 14
            vid_rect = self._fit_rect(self.review_video_pix, vid_x, avail_y, vid_w, avail_h)
            # 그림자
            p.fillRect(vid_rect.translated(5, 5).toRect(), QBrush(QColor(0, 0, 0, 80)))
            # 클립 & 드로우
            vid_path = QPainterPath()
            vid_path.addRoundedRect(vid_rect, 10, 10)
            p.setClipPath(vid_path)
            p.drawPixmap(vid_rect.toRect(), self.review_video_pix)
            p.setClipping(False)
            # 테두리
            p.setPen(QPen(C_PINK, 2))
            p.setBrush(Qt.NoBrush)
            p.drawPath(vid_path)

            # REPLAY 배지
            badge = QRectF(vid_rect.x() + 10, vid_rect.y() + 10, 78, 26)
            badge_path = QPainterPath()
            badge_path.addRoundedRect(badge, 13, 13)
            p.fillPath(badge_path, QBrush(QColor(0, 0, 0, 155)))
            p.setFont(QFont("Arial", 8, QFont.Bold))
            p.setPen(C_GOLD)
            p.drawText(badge, Qt.AlignCenter, "▶  REPLAY")

        # 파티클 (타이틀 영역)
        rng = np.random.default_rng(int(time.time() * 4) % 1000)
        for _ in range(14):
            sx = int(rng.integers(10, w - 10))
            sy = int(rng.integers(5, title_h - 5))
            sr = int(rng.integers(2, 5))
            sc = [C_ACCENT, C_GOLD, C_PINK, C_WHITE][int(rng.integers(0, 4))]
            p.setBrush(QBrush(sc))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(sx, sy), sr, sr)

    @staticmethod
    def _fit_rect(pix: QPixmap, x0: int, y0: int, avail_w: int, avail_h: int) -> QRectF:
        """이미지를 가용 영역에 맞게 축소해 중앙 배치한 QRectF 반환"""
        pw, ph = pix.width(), pix.height()
        scale  = min(avail_w / pw, avail_h / ph)
        nw, nh = int(pw * scale), int(ph * scale)
        xoff   = x0 + (avail_w - nw) // 2
        yoff   = y0 + (avail_h - nh) // 2
        return QRectF(xoff, yoff, nw, nh)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PhotoboothWindow – 메인 윈도우
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class PhotoboothWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("4-CUT PHOTOBOOTH")
        self.setStyleSheet("background: rgb(25, 14, 40);")

        # ── 카메라
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

        # ── MediaPipe
        opts = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL),
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=RunningMode.VIDEO,
        )
        self.recognizer = GestureRecognizer.create_from_options(opts)

        # ── 상태 변수
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
        self._vid_tmp         = os.path.join(SAVE_DIR, '_recording_tmp.avi')
        self._out_writer      = None

        # ── UI 레이아웃
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.cam_view = CameraView()
        self.strip    = StripPanel()
        layout.addWidget(self.cam_view)
        layout.addWidget(self.strip)

        # ── 타이머
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_frame)
        self.timer.start(1000 // self.fps)

        self.showFullScreen()
        print("=" * 50)
        print("  인생네컷 포토부스  (PyQt5 UI)")
        print("=" * 50)
        print("  peace (0.8s) : 촬영")
        print("  hand         : 그리기")
        print("  fist         : 펜 색상 변경")
        print("  open         : 그림 지우기 / 리셋")
        print("  ESC / Q      : 종료")
        print("=" * 50)

    # ── 메인 루프 (30fps) ─────────────────────────
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

        # ── 제스처 인식
        gesture = None
        if self.state not in (STATE_REVIEW,):
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
                            self.color_idx   = (self.color_idx + 1) % len(PEN_COLORS_CV)
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

                    # 랜드마크 오버레이
                    for conn in HAND_CONNECTIONS:
                        s = hand_landmarks[conn[0]]; e = hand_landmarks[conn[1]]
                        cv2.line(frame, (int(s.x*w), int(s.y*h)),
                                 (int(e.x*w), int(e.y*h)), (180, 255, 180), 2)
                    for lm in hand_landmarks:
                        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (255, 180, 180), -1)

                    # 커서
                    if gesture in ('fist', 'open', 'peace'):
                        cv2.circle(frame, (ix, iy), 12, (200, 200, 200), 1)
                    else:
                        cv2.circle(frame, (ix, iy), self.line_thickness + 5,
                                   self.draw_color_cv, 2)
                        cv2.circle(frame, (ix, iy), 3, self.draw_color_cv, -1)
            else:
                self.prev_x = self.prev_y = None

        # ── peace 홀드 타이머
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

        # ── open → 리셋
        if gesture == 'open' and self.state in (STATE_REVIEW,):
            if self.last_gesture != 'open':
                self._reset_session(h, w)

        self.last_gesture = gesture

        # ── 상태 처리
        if self.state == STATE_COUNTDOWN:
            elapsed = now - self.countdown_start
            if elapsed >= COUNTDOWN_SEC:
                mask = cv2.cvtColor(self.draw_canvas, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
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
                    collage  = _make_final_collage(self.photos)
                    self.cam_view.review_pix = _cv_to_pixmap(collage) if collage is not None else None
                    self.state = STATE_REVIEW
                    _vid_play = os.path.join(SAVE_DIR, '_recording_play.avi')
                    # writer를 flush 후 복사, 이후 새 writer로 재개
                    if self._out_writer and self._out_writer is not False:
                        self._out_writer.release()
                        self._out_writer = None  # 다음 프레임에서 재생성
                    if os.path.exists(self._vid_tmp):
                        import shutil as _sh
                        _sh.copy2(self._vid_tmp, _vid_play)
                        self._review_cap = cv2.VideoCapture(_vid_play)
                else:
                    self.state = STATE_WAITING

        # ── 그리기 캔버스 합성
        if self.state not in (STATE_REVIEW,):
            mask = cv2.cvtColor(self.draw_canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            mask_inv  = cv2.bitwise_not(mask)
            frame_bg  = cv2.bitwise_and(frame, frame, mask=mask_inv)
            canvas_fg = cv2.bitwise_and(self.draw_canvas, self.draw_canvas, mask=mask)
            frame     = cv2.add(frame_bg, canvas_fg)

        # ── 녹화: VideoWriter 초기화 (첫 프레임에서 크기 확정 후 생성)
        if self._out_writer is None and self.state not in (STATE_REVIEW,):
            _vh, _vw = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(self._vid_tmp, fourcc, self.fps, (_vw, _vh))
            if writer.isOpened():
                self._out_writer = writer
                print(f"[녹화 시작] {self._vid_tmp}")
            else:
                writer.release()
                self._out_writer = False  # 재시도 방지
                print("[경고] VideoWriter 초기화 실패 — 녹화 비활성화")
        if self._out_writer and self._out_writer is not False and self.state not in (STATE_REVIEW,):
            self._out_writer.write(frame)

        # ── 리뷰 영상 업데이트
        if self.state == STATE_REVIEW and self._review_cap:
            ret_v, vframe = self._review_cap.read()
            if not ret_v:
                self._review_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_v, vframe = self._review_cap.read()
            if ret_v and vframe is not None:
                self.cam_view.review_video_pix = _cv_to_pixmap(vframe)

        # ── CameraView 데이터 전달
        self.cam_view.frame_pix      = _cv_to_pixmap(frame)
        self.cam_view.state          = self.state
        self.cam_view.taken          = len(self.photos)
        self.cam_view.countdown_start = self.countdown_start
        self.cam_view.flash_start    = self.flash_start
        self.cam_view.gesture        = gesture
        self.cam_view.gesture_start  = self.gesture_start
        self.cam_view.drawing_color  = self.draw_color_qc

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
        # 재생용 임시 복사본 삭제
        _vid_play_cleanup = os.path.join(SAVE_DIR, '_recording_play.avi')
        if os.path.exists(_vid_play_cleanup):
            try: os.remove(_vid_play_cleanup)
            except Exception: pass
        # writer 리셋 (다음 세션에서 새로 생성)
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
        if self._out_writer:
            self._out_writer.release()
        if self.video:
            self.video.release()
        try:
            self.recognizer.close()
        except Exception:
            pass


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("4-CUT Photobooth")
    win = PhotoboothWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
