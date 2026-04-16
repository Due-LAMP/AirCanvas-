#!/usr/bin/env python3
"""
photobooth_v3.py – Modern Dark Theme
네온 시안 + 딥 블랙 + 글래스모피즘 스타일
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
                              QImage, QPixmap, QConicalGradient)

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

# ── 모던 다크 테마 컬러 ────────────────────────────────────────
# 배경 계열
C_BG0    = QColor(  8,   8,  14)   # 최심 다크
C_BG1    = QColor( 14,  14,  22)   # 패널 배경
C_BG2    = QColor( 20,  20,  32)   # 카드 배경
C_GLASS  = QColor( 30,  35,  55, 160)  # 글래스 패널

# 액센트
C_CYAN   = QColor(  0, 220, 255)   # 네온 시안 (메인 액센트)
C_CYAN2  = QColor(  0, 160, 220)   # 어두운 시안
C_BLUE   = QColor( 60,  80, 255)   # 블루 포인트
C_PURPLE = QColor(120,  60, 255)   # 보조 퍼플

# 텍스트 / 선
C_WHITE  = QColor(255, 255, 255)
C_GRAY1  = QColor(160, 165, 180)   # 보조 텍스트
C_GRAY2  = QColor( 55,  60,  80)   # 비활성 경계선
C_LINE   = QColor(  0, 220, 255,  60)  # 반투명 구분선

# 상태 색
C_GREEN  = QColor( 50, 230, 120)
C_AMBER  = QColor(255, 180,  30)

# 펜 팔레트
PEN_COLORS_QC = [
    QColor(255,  60,  80),   # red
    QColor(255, 140,   0),   # orange
    QColor(255, 230,   0),   # yellow
    QColor( 40, 220,  80),   # green
    QColor(  0, 200, 255),   # cyan
    QColor(140,  60, 255),   # purple
    QColor(255, 255, 255),   # white
]
PEN_COLORS_CV = [
    ( 80,  60, 255), (  0, 140, 255), (  0, 230, 255),
    ( 80, 220,  40), (255, 200,   0), (255,  60, 140),
    (255, 255, 255),
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


def _make_final_collage(photos):
    if not photos:
        return None
    ph, pw = photos[0].shape[:2]
    margin = 24; top_pad = 80; bottom_pad = 52
    col_w = pw + margin * 2
    col_h = top_pad + (ph + margin) * TOTAL_SHOTS + margin + bottom_pad
    # 다크 배경
    collage = np.full((col_h, col_w, 3), (12, 12, 20), dtype=np.uint8)
    # 사이드 시안 라인
    cv2.line(collage, (2, 0), (2, col_h), (0, 220, 255), 2)
    cv2.line(collage, (col_w-3, 0), (col_w-3, col_h), (0, 220, 255), 2)
    title = "4-CUT"
    tw = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.4, 2)[0][0]
    cv2.putText(collage, title, (col_w//2 - tw//2, 56),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 220, 255), 2, cv2.LINE_AA)
    for i, photo in enumerate(photos):
        y0 = top_pad + margin + i * (ph + margin)
        collage[y0:y0+ph, margin:margin+pw] = photo
        cv2.rectangle(collage, (margin-3, y0-3), (margin+pw+3, y0+ph+3), (0, 220, 255), 1)
    date_str = datetime.now().strftime("%Y . %m . %d")
    dw = cv2.getTextSize(date_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
    cv2.putText(collage, date_str, (col_w//2 - dw//2, col_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 90, 120), 1, cv2.LINE_AA)
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
#  헬퍼: 글래스 패널 그리기
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _draw_glass(p: QPainter, rect: QRectF, radius: float = 10,
                bg: QColor = None, border: QColor = None, border_w: float = 1.0):
    bg     = bg     or C_GLASS
    border = border or C_LINE
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    p.fillPath(path, QBrush(bg))
    p.setPen(QPen(border, border_w))
    p.setBrush(Qt.NoBrush)
    p.drawPath(path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  StripPanel – 우측 4컷 미리보기  (모던 다크)
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

        # ── 배경: 세로 그라디언트 (거의 검정)
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, QColor(12, 12, 20))
        grad.setColorAt(1.0, QColor( 8,  8, 14))
        p.fillRect(0, 0, w, h, QBrush(grad))

        # 왼쪽 시안 어센트 라인
        grad_line = QLinearGradient(0, 0, 0, h)
        grad_line.setColorAt(0.0, QColor(0, 220, 255,   0))
        grad_line.setColorAt(0.3, QColor(0, 220, 255, 180))
        grad_line.setColorAt(0.7, QColor(0, 220, 255, 180))
        grad_line.setColorAt(1.0, QColor(0, 220, 255,   0))
        p.setPen(QPen(QBrush(grad_line), 2))
        p.drawLine(0, 0, 0, h)

        title_h = 68
        foot_h  = 52
        margin  = 12
        slot_w  = w - margin * 2
        avail_h = h - title_h - foot_h - margin * (n + 1)
        slot_h  = max(avail_h // n, 40)

        # ── 타이틀 영역
        # 배경 글래스
        _draw_glass(p, QRectF(0, 0, w, title_h), radius=0,
                    bg=QColor(18, 22, 38, 230),
                    border=QColor(0, 220, 255, 40))
        # 하단 시안 구분선
        p.setPen(QPen(C_CYAN, 1))
        p.drawLine(0, title_h, w, title_h)
        # 타이틀 텍스트
        p.setFont(QFont("Courier", 13, QFont.Bold))
        p.setPen(C_CYAN)
        p.drawText(QRectF(0, 4, w, title_h - 8), Qt.AlignCenter, "4-CUT")
        p.setFont(QFont("Arial", 7))
        p.setPen(QColor(0, 220, 255, 120))
        p.drawText(QRectF(0, title_h - 22, w, 18), Qt.AlignCenter, "PHOTOBOOTH  v3")

        # ── 사진 슬롯
        for i in range(n):
            y0 = title_h + margin + i * (slot_h + margin)
            x0 = margin
            slot_rect = QRectF(x0, y0, slot_w, slot_h)

            if i < len(self.photos) and self.photos[i] is not None:
                # 네온 글로우 그림자 (시안)
                for glow_r, glow_a in [(8, 30), (4, 60)]:
                    gp = QPainterPath()
                    gp.addRoundedRect(slot_rect.adjusted(-glow_r, -glow_r, glow_r, glow_r), 6, 6)
                    p.fillPath(gp, QBrush(QColor(0, 220, 255, glow_a)))

                # 이미지 클립
                clip = QPainterPath()
                clip.addRoundedRect(slot_rect, 4, 4)
                p.setClipPath(clip)
                p.drawPixmap(slot_rect.toRect(), _cv_to_pixmap(self.photos[i]))
                p.setClipping(False)

                # 시안 테두리
                p.setPen(QPen(C_CYAN, 1.5))
                p.setBrush(Qt.NoBrush)
                pp = QPainterPath()
                pp.addRoundedRect(slot_rect, 4, 4)
                p.drawPath(pp)

                # 완료 뱃지 – 오른쪽 아래 캡슐
                bw, bh = 28, 18
                bx = x0 + slot_w - bw - 4
                by = y0 + slot_h - bh - 4
                bp = QPainterPath()
                bp.addRoundedRect(QRectF(bx, by, bw, bh), 9, 9)
                p.fillPath(bp, QBrush(QColor(0, 220, 255, 210)))
                p.setFont(QFont("Courier", 8, QFont.Bold))
                p.setPen(C_BG0)
                p.drawText(QRectF(bx, by, bw, bh), Qt.AlignCenter, f"0{i+1}")

            else:
                # 빈 슬롯: 어두운 글래스 + 점선 + 번호
                _draw_glass(p, slot_rect, radius=4,
                            bg=QColor(18, 22, 38, 140),
                            border=QColor(0, 220, 255, 45))

                # 점선 테두리 (직접 그리기)
                pen_dash = QPen(QColor(0, 220, 255, 80), 1, Qt.DashLine)
                pen_dash.setDashPattern([5, 5])
                p.setPen(pen_dash)
                p.setBrush(Qt.NoBrush)
                ip = QPainterPath()
                ip.addRoundedRect(slot_rect.adjusted(3, 3, -3, -3), 3, 3)
                p.drawPath(ip)

                # 중앙 번호 캡슐
                cw, ch_cap = 36, 22
                cx = x0 + slot_w//2 - cw//2
                cy = y0 + slot_h//2 - ch_cap//2
                cp = QPainterPath()
                cp.addRoundedRect(QRectF(cx, cy, cw, ch_cap), 11, 11)
                p.fillPath(cp, QBrush(QColor(0, 220, 255, 30)))
                p.setPen(QPen(QColor(0, 220, 255, 100), 1))
                p.drawPath(cp)
                p.setFont(QFont("Courier", 11, QFont.Bold))
                p.setPen(QColor(0, 220, 255, 130))
                p.drawText(QRectF(cx, cy, cw, ch_cap), Qt.AlignCenter, f"0{i+1}")

        # ── 하단 진행 바 (캡슐형)
        taken   = len(self.photos)
        bar_y   = h - foot_h // 2 - 6
        seg_w   = (w - 32) // n
        for i in range(n):
            bx1 = 16 + i * seg_w + 2
            bx2 = 16 + (i + 1) * seg_w - 2
            br  = QRectF(bx1, bar_y, bx2 - bx1, 12)
            bp  = QPainterPath()
            bp.addRoundedRect(br, 6, 6)
            if i < taken:
                grad_seg = QLinearGradient(bx1, 0, bx2, 0)
                grad_seg.setColorAt(0.0, C_CYAN2)
                grad_seg.setColorAt(1.0, C_CYAN)
                p.fillPath(bp, QBrush(grad_seg))
            else:
                p.fillPath(bp, QBrush(QColor(30, 35, 55)))
                p.setPen(QPen(QColor(0, 220, 255, 40), 1))
                p.drawPath(bp)

        # 하단 카운트 텍스트
        p.setFont(QFont("Arial", 7))
        p.setPen(QColor(0, 220, 255, 100))
        p.drawText(QRectF(0, bar_y + 16, w, 16), Qt.AlignCenter,
                   f"{taken} / {n}  captured")

        p.end()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CameraView – 메인 카메라 + 오버레이  (모던 다크)
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
        self.drawing_color:    QColor         = PEN_COLORS_QC[4]  # 기본 시안
        self.review_pix:       QPixmap | None = None
        self.review_video_pix: QPixmap | None = None

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
            p.fillRect(0, 0, w, h, QBrush(C_BG0))

        # 카메라 위 가벼운 비네트 효과 (모서리 어둡게)
        self._draw_vignette(p, w, h)

        self._draw_top_hud(p, w, h)

        if self.state == STATE_COUNTDOWN and self.countdown_start:
            self._draw_countdown(p, w, h)
        elif self.state == STATE_FLASH and self.flash_start:
            self._draw_flash(p, w, h)
        elif self.state == STATE_WAITING:
            self._draw_bottom_hud(p, w, h)

        p.end()

    # ── 비네트 ────────────────────────────────────
    def _draw_vignette(self, p: QPainter, w: int, h: int):
        grad = QRadialGradient(w / 2, h / 2, max(w, h) * 0.65)
        grad.setColorAt(0.5, QColor(0, 0, 0,   0))
        grad.setColorAt(1.0, QColor(0, 0, 0, 110))
        p.fillRect(0, 0, w, h, QBrush(grad))

    # ── 상단 HUD ─────────────────────────────────
    def _draw_top_hud(self, p: QPainter, w: int, h: int):
        hud_h = 52

        # 글래스 배경
        _draw_glass(p, QRectF(0, 0, w, hud_h), radius=0,
                    bg=QColor(8, 10, 18, 200),
                    border=QColor(0, 220, 255, 0))

        # 하단 시안 라인
        p.setPen(QPen(QColor(0, 220, 255, 90), 1))
        p.drawLine(0, hud_h, w, hud_h)

        cy = hud_h // 2

        # 진행 캡슐 (촬영 수)
        cap_w, cap_h_item = 34, 16
        for i in range(TOTAL_SHOTS):
            bx = 14 + i * (cap_w + 6)
            cr = QRectF(bx, cy - cap_h_item // 2, cap_w, cap_h_item)
            cp = QPainterPath()
            cp.addRoundedRect(cr, 8, 8)
            if i < self.taken:
                grad_cap = QLinearGradient(bx, 0, bx + cap_w, 0)
                grad_cap.setColorAt(0.0, C_CYAN2)
                grad_cap.setColorAt(1.0, C_CYAN)
                p.fillPath(cp, QBrush(grad_cap))
                p.setFont(QFont("Courier", 7, QFont.Bold))
                p.setPen(C_BG0)
                p.drawText(cr, Qt.AlignCenter, f"0{i+1}")
            else:
                p.fillPath(cp, QBrush(QColor(25, 30, 45)))
                p.setPen(QPen(QColor(0, 220, 255, 50), 1))
                p.drawPath(cp)
                p.setFont(QFont("Courier", 7))
                p.setPen(QColor(0, 220, 255, 70))
                p.drawText(cr, Qt.AlignCenter, f"0{i+1}")

        # 중앙 타이틀 (모노스페이스 느낌)
        p.setFont(QFont("Courier", 11, QFont.Bold))
        p.setPen(C_WHITE)
        title_str = f"[ SHOT  {min(self.taken + 1, TOTAL_SHOTS)} / {TOTAL_SHOTS} ]"
        p.drawText(QRectF(0, 0, w, hud_h), Qt.AlignCenter, title_str)

        # 펜 색상 인디케이터 (오른쪽)
        px, py_pen = w - 34, cy
        # 외부 링
        p.setPen(QPen(QColor(0, 220, 255, 80), 1))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(px, py_pen), 16, 16)
        # 내부 색상
        p.setBrush(QBrush(self.drawing_color))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(px, py_pen), 11, 11)
        p.setFont(QFont("Arial", 6))
        p.setPen(QColor(0, 220, 255, 100))
        p.drawText(QRectF(w - 76, py_pen + 12, 44, 12), Qt.AlignCenter, "PEN")

    # ── 카운트다운 ────────────────────────────────
    def _draw_countdown(self, p: QPainter, w: int, h: int):
        elapsed  = time.time() - self.countdown_start
        num_show = max(1, int(COUNTDOWN_SEC - elapsed) + 1)
        progress = 1.0 - (elapsed % 1.0)
        cx, cy   = w // 2, h // 2
        R = 90

        # 배경 다크 원
        rad_bg = QRadialGradient(cx, cy, 130)
        rad_bg.setColorAt(0.0, QColor(0, 10, 20, 220))
        rad_bg.setColorAt(1.0, QColor(0, 10, 20,   0))
        p.setBrush(QBrush(rad_bg))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx, cy), 130, 130)

        # 트랙 링
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(0, 220, 255, 30), 8, Qt.SolidLine, Qt.RoundCap))
        p.drawEllipse(QPointF(cx, cy), R, R)

        # 진행 호 (시안)
        p.setPen(QPen(C_CYAN, 8, Qt.SolidLine, Qt.RoundCap))
        arc_rect = QRect(cx - R, cy - R, R * 2, R * 2)
        p.drawArc(arc_rect, 90 * 16, -int(360 * progress) * 16)

        # 외곽 글로우 링
        p.setPen(QPen(QColor(0, 220, 255, 40), 3))
        p.drawEllipse(QPointF(cx, cy), R + 12, R + 12)

        # 스캔라인 효과 (수평 줄)
        p.setPen(QPen(QColor(0, 220, 255, 12), 1))
        for yy in range(cy - R - 10, cy + R + 10, 6):
            p.drawLine(cx - R - 10, yy, cx + R + 10, yy)

        # 숫자
        p.setFont(QFont("Courier", 72, QFont.Bold))
        # 텍스트 그림자 (시안)
        p.setPen(QColor(0, 220, 255, 60))
        p.drawText(QRectF(cx - 66, cy - 60, 132, 120).translated(2, 2),
                   Qt.AlignCenter, str(num_show))
        p.setPen(C_WHITE)
        p.drawText(QRectF(cx - 66, cy - 60, 132, 120),
                   Qt.AlignCenter, str(num_show))

        # 하단 레이블
        p.setFont(QFont("Courier", 9))
        p.setPen(QColor(0, 220, 255, 180))
        p.drawText(QRectF(cx - 80, cy + R + 18, 160, 20),
                   Qt.AlignCenter, "CAPTURING...")

    # ── 플래시 ────────────────────────────────────
    def _draw_flash(self, p: QPainter, w: int, h: int):
        ratio = max(0.0, 1.0 - (time.time() - self.flash_start) / FLASH_SEC)
        # 흰색 + 시안 틴트 플래시
        p.fillRect(0, 0, w, h, QBrush(QColor(220, 255, 255, int(ratio * 240))))

    # ── 하단 제스처 HUD ───────────────────────────
    def _draw_bottom_hud(self, p: QPainter, w: int, h: int):
        hud2_h = 60
        hud2_y = h - hud2_h

        _draw_glass(p, QRectF(0, hud2_y, w, hud2_h), radius=0,
                    bg=QColor(8, 10, 18, 210),
                    border=QColor(0, 0, 0, 0))
        p.setPen(QPen(QColor(0, 220, 255, 80), 1))
        p.drawLine(0, hud2_y, w, hud2_y)

        now = time.time()

        if self.gesture == 'peace' and self.gesture_start is not None:
            held  = now - self.gesture_start
            ratio = min(held / GESTURE_HOLD, 1.0)
            bx1, bx2 = 20, w - 20
            by = h - 18

            # 트랙
            track = QRectF(bx1, by - 6, bx2 - bx1, 12)
            tp = QPainterPath()
            tp.addRoundedRect(track, 6, 6)
            p.fillPath(tp, QBrush(QColor(20, 25, 40)))
            p.setPen(QPen(QColor(0, 220, 255, 40), 1))
            p.drawPath(tp)

            # 진행
            if ratio > 0:
                fill = QRectF(bx1, by - 6, (bx2 - bx1) * ratio, 12)
                fp = QPainterPath()
                fp.addRoundedRect(fill, 6, 6)
                grad_fill = QLinearGradient(bx1, 0, bx2, 0)
                grad_fill.setColorAt(0.0, C_CYAN2)
                grad_fill.setColorAt(1.0, C_CYAN)
                p.fillPath(fp, QBrush(grad_fill))

            p.setFont(QFont("Courier", 9, QFont.Bold))
            p.setPen(C_CYAN)
            p.drawText(QRectF(bx1, hud2_y + 8, bx2 - bx1, 20),
                       Qt.AlignCenter, "✌  HOLD  PEACE  TO  CAPTURE")
        else:
            # 제스처 힌트 버튼 (캡슐형)
            hints = [("✌", "PEACE", "capture"), ("☝", "HAND", "draw"),
                     ("✊", "FIST", "color"), ("🖐", "OPEN", "clear")]
            seg_w = w // len(hints)
            for hi, (icon, gn, gd_text) in enumerate(hints):
                cx_g  = hi * seg_w + seg_w // 2
                active = self.gesture is not None and self.gesture == gn.lower()

                # 버튼 배경 (활성 시 글로우)
                btn_w, btn_h_item = seg_w - 16, 42
                bx = hi * seg_w + 8
                btn_rect = QRectF(bx, hud2_y + (hud2_h - btn_h_item)//2,
                                  btn_w, btn_h_item)
                btn_p = QPainterPath()
                btn_p.addRoundedRect(btn_rect, 8, 8)

                if active:
                    # 시안 글로우 배경
                    p.fillPath(btn_p, QBrush(QColor(0, 220, 255, 35)))
                    p.setPen(QPen(C_CYAN, 1))
                    p.drawPath(btn_p)
                else:
                    p.fillPath(btn_p, QBrush(QColor(18, 22, 38, 160)))
                    p.setPen(QPen(QColor(0, 220, 255, 30), 1))
                    p.drawPath(btn_p)

                col_main = C_CYAN if active else C_GRAY1
                col_sub  = QColor(0, 220, 255, 160) if active else QColor(80, 88, 110)

                p.setFont(QFont("Arial", 10, QFont.Bold if active else QFont.Normal))
                p.setPen(col_main)
                p.drawText(QRectF(hi * seg_w, hud2_y + 8, seg_w, 22),
                           Qt.AlignCenter, f"{icon} {gn}")
                p.setFont(QFont("Courier", 7))
                p.setPen(col_sub)
                p.drawText(QRectF(hi * seg_w, hud2_y + 32, seg_w, 16),
                           Qt.AlignCenter, gd_text)

    # ── 리뷰 화면 ─────────────────────────────────
    def _draw_review(self, p: QPainter, w: int, h: int):
        # 순수 다크 배경
        p.fillRect(0, 0, w, h, QBrush(C_BG0))

        # 스캔라인 텍스처
        p.setPen(QPen(QColor(0, 220, 255, 6), 1))
        for yy in range(0, h, 4):
            p.drawLine(0, yy, w, yy)

        title_h = 60
        foot_h  = 56

        # ── 타이틀 바
        _draw_glass(p, QRectF(0, 0, w, title_h), radius=0,
                    bg=QColor(0, 15, 30, 230),
                    border=QColor(0, 0, 0, 0))
        p.setPen(QPen(C_CYAN, 1))
        p.drawLine(0, title_h, w, title_h)

        p.setFont(QFont("Courier", 16, QFont.Bold))
        # 그림자
        p.setPen(QColor(0, 220, 255, 50))
        p.drawText(QRectF(2, 2, w, title_h), Qt.AlignCenter, "YOUR  4-CUT")
        p.setPen(C_WHITE)
        p.drawText(QRectF(0, 0, w, title_h), Qt.AlignCenter, "YOUR  4-CUT")
        # 우측 날짜
        p.setFont(QFont("Courier", 8))
        p.setPen(QColor(0, 220, 255, 130))
        date_str = datetime.now().strftime("%Y.%m.%d  %H:%M")
        p.drawText(QRectF(w - 180, 0, 170, title_h), Qt.AlignVCenter | Qt.AlignRight, date_str)

        # ── 하단 안내 바
        _draw_glass(p, QRectF(0, h - foot_h, w, foot_h), radius=0,
                    bg=QColor(0, 15, 30, 220),
                    border=QColor(0, 0, 0, 0))
        p.setPen(QPen(C_CYAN, 1))
        p.drawLine(0, h - foot_h, w, h - foot_h)
        p.setFont(QFont("Courier", 8))
        p.setPen(C_GRAY1)
        p.drawText(QRectF(0, h - foot_h, w, foot_h),
                   Qt.AlignCenter, "[ OPEN PALM ]  New Session    |    [ ESC ]  Quit")

        avail_y = title_h + 20
        avail_h = h - title_h - foot_h - 40
        half_w  = int(w * 0.42)
        gap     = 28

        # ── 콜라주 (왼쪽)
        if self.review_pix:
            col_rect = self._fit_rect(self.review_pix, 18, avail_y, half_w - 18, avail_h)

            # 글로우 그림자 레이어
            for glow_r, glow_a in [(14, 20), (8, 40), (4, 60)]:
                gp = QPainterPath()
                gp.addRoundedRect(col_rect.adjusted(-glow_r, -glow_r, glow_r, glow_r), 4, 4)
                p.fillPath(gp, QBrush(QColor(0, 220, 255, glow_a)))

            # 이미지 클립
            clip = QPainterPath()
            clip.addRoundedRect(col_rect, 3, 3)
            p.setClipPath(clip)
            p.drawPixmap(col_rect.toRect(), self.review_pix)
            p.setClipping(False)

            # 시안 테두리
            p.setPen(QPen(C_CYAN, 1.5))
            p.setBrush(Qt.NoBrush)
            bp = QPainterPath()
            bp.addRoundedRect(col_rect, 3, 3)
            p.drawPath(bp)

            # "SAVED" 뱃지
            badge_w, badge_h = 66, 22
            badge = QRectF(col_rect.x() + 8, col_rect.y() + 8, badge_w, badge_h)
            bpath = QPainterPath()
            bpath.addRoundedRect(badge, 3, 3)
            p.fillPath(bpath, QBrush(C_CYAN))
            p.setFont(QFont("Courier", 8, QFont.Bold))
            p.setPen(C_BG0)
            p.drawText(badge, Qt.AlignCenter, "✓ SAVED")

        # ── 영상 (오른쪽)
        if self.review_video_pix:
            vid_x = half_w + gap
            vid_w_avail = w - vid_x - 18
            vid_rect = self._fit_rect(self.review_video_pix, vid_x, avail_y,
                                      vid_w_avail, avail_h)

            # 글로우
            for glow_r, glow_a in [(10, 20), (4, 45)]:
                gp = QPainterPath()
                gp.addRoundedRect(vid_rect.adjusted(-glow_r, -glow_r, glow_r, glow_r), 4, 4)
                p.fillPath(gp, QBrush(QColor(120, 60, 255, glow_a)))

            # 이미지 클립
            clip2 = QPainterPath()
            clip2.addRoundedRect(vid_rect, 3, 3)
            p.setClipPath(clip2)
            p.drawPixmap(vid_rect.toRect(), self.review_video_pix)
            p.setClipping(False)

            # 퍼플 테두리 (영상은 다른 액센트)
            p.setPen(QPen(C_PURPLE, 1.5))
            p.setBrush(Qt.NoBrush)
            vp = QPainterPath()
            vp.addRoundedRect(vid_rect, 3, 3)
            p.drawPath(vp)

            # REPLAY 뱃지
            badge_w, badge_h = 80, 22
            badge = QRectF(vid_rect.x() + 8, vid_rect.y() + 8, badge_w, badge_h)
            bpath = QPainterPath()
            bpath.addRoundedRect(badge, 3, 3)
            p.fillPath(bpath, QBrush(QColor(0, 0, 0, 170)))
            p.setPen(QPen(C_PURPLE, 1))
            p.drawPath(bpath)
            p.setFont(QFont("Courier", 8, QFont.Bold))
            p.setPen(C_PURPLE)
            p.drawText(badge, Qt.AlignCenter, "▶  REPLAY")
        else:
            # 영상 대기 표시
            vid_x = half_w + gap
            vid_w_avail = w - vid_x - 18
            placeholder = QRectF(vid_x, avail_y, vid_w_avail, avail_h)
            _draw_glass(p, placeholder, radius=3,
                        bg=QColor(12, 14, 24, 180),
                        border=QColor(120, 60, 255, 60))
            p.setFont(QFont("Courier", 10))
            p.setPen(QColor(120, 60, 255, 120))
            p.drawText(placeholder, Qt.AlignCenter, "NO VIDEO")

        # 수직 구분선 (콜라주 / 영상 사이)
        sep_x = half_w + gap // 2
        p.setPen(QPen(QColor(0, 220, 255, 25), 1, Qt.DashLine))
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
        self.setWindowTitle("4-CUT PHOTOBOOTH  v3")
        self.setStyleSheet("background: rgb(8, 8, 14);")

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
        self.color_idx        = 4  # 기본: 시안
        self.draw_color_cv    = PEN_COLORS_CV[4]
        self.draw_color_qc    = PEN_COLORS_QC[4]
        self.line_thickness   = 5
        self.last_gesture     = None
        self.gesture_start    = None
        self.countdown_start  = None
        self.flash_start      = None
        self.frame_index      = 0
        self.fps              = 30
        self._review_cap      = None
        self._session_dir     = None
        self._vid_tmp         = os.path.join(SAVE_DIR, '_recording_tmp_v3.avi')
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
        print("  4-CUT PHOTOBOOTH  v3  (Modern Dark)")
        print("=" * 50)
        print("  peace (0.8s) : 촬영")
        print("  hand         : 그리기")
        print("  fist         : 펜 색상 변경")
        print("  open         : 그림 지우기 / 리셋")
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

                    # 랜드마크 (시안/그린 계열)
                    for conn in HAND_CONNECTIONS:
                        s = hand_landmarks[conn[0]]; e = hand_landmarks[conn[1]]
                        cv2.line(frame, (int(s.x*w), int(s.y*h)),
                                 (int(e.x*w), int(e.y*h)), (0, 200, 180), 1)
                    for lm in hand_landmarks:
                        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, (0, 220, 255), -1)

                    # 커서
                    if gesture in ('fist', 'open', 'peace'):
                        cv2.circle(frame, (ix, iy), 14, (0, 220, 255), 1)
                        cv2.circle(frame, (ix, iy),  4, (0, 220, 255), -1)
                    else:
                        cv2.circle(frame, (ix, iy), self.line_thickness + 6,
                                   self.draw_color_cv, 2)
                        cv2.circle(frame, (ix, iy), 3, self.draw_color_cv, -1)
            else:
                self.prev_x = self.prev_y = None

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
                    self.cam_view.review_pix = _cv_to_pixmap(collage) if collage is not None else None
                    self.state = STATE_REVIEW
                    _vid_play = os.path.join(SAVE_DIR, '_recording_play_v3.avi')
                    if self._out_writer and self._out_writer is not False:
                        self._out_writer.release()
                        self._out_writer = None
                    if os.path.exists(self._vid_tmp):
                        import shutil as _sh
                        _sh.copy2(self._vid_tmp, _vid_play)
                        self._review_cap = cv2.VideoCapture(_vid_play)
                else:
                    self.state = STATE_WAITING

        # ── 그리기 합성
        if self.state not in (STATE_REVIEW,):
            mask = cv2.cvtColor(self.draw_canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            mask_inv  = cv2.bitwise_not(mask)
            frame_bg  = cv2.bitwise_and(frame, frame, mask=mask_inv)
            canvas_fg = cv2.bitwise_and(self.draw_canvas, self.draw_canvas, mask=mask)
            frame     = cv2.add(frame_bg, canvas_fg)

        # ── 녹화
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
                self.cam_view.review_video_pix = _cv_to_pixmap(vframe)

        # ── 위젯 업데이트
        self.cam_view.frame_pix       = _cv_to_pixmap(frame)
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
        for tmp in ['_recording_play_v3.avi']:
            p = os.path.join(SAVE_DIR, tmp)
            if os.path.exists(p):
                try: os.remove(p)
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
    app.setApplicationName("4-CUT Photobooth v3")
    win = PhotoboothWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
