#!/usr/bin/env python3
"""
photobooth_v3.py – Modern Dark Theme
네온 시안 + 딥 블랙 + 글래스모피즘 스타일
"""

import os, sys, time, signal, threading, queue
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
_GESTURE_MAP = {
    'Closed_Fist': 'fist',
    'Victory':     'victory',
    'Open_Palm':   'open',
    'Thumb_Down':  'thumbdown',
}

# ── 드로우 모드 ───────────────────────────────────────────────
DRAW_DEFAULT  = 'default'
DRAW_PAINTING = 'painting'
DRAW_ERASE    = 'erase'

# ── 색상 팔레트 ──────────────────────────────────────────────
PALETTE_CX      = 32   # 화면 왼쪽 x 위치
PALETTE_RADIUS  = 18   # 원 반지름
PALETTE_SPACING = 46   # 원 간격 (중심 간)

# ── 상수 ───────────────────────────────────────────────────────
TOTAL_SHOTS   = 4
COUNTDOWN_SEC = 3
FLASH_SEC     = 0.5
HOLD_FIST     = 0.2   # painting ↔ erase 토글
HOLD_CLEAR    = 0.5   # 전체 캔버스 삭제
HOLD_PHOTO    = 0.0   # 촬영 트리거 (첫 프레임 즉시)
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


def _palette_positions(h: int) -> list:
    """팔레트 각 색상 원의 중심 (x, y) 리스트를 반환."""
    n      = len(PEN_COLORS_CV)
    total  = n * PALETTE_SPACING
    start_y = (h - total) // 2 + PALETTE_SPACING // 2
    return [(PALETTE_CX, start_y + i * PALETTE_SPACING) for i in range(n)]


def _draw_color_palette_on_frame(frame, color_idx: int):
    """OpenCV 프레임 왼쪽에 색상 팔레트를 그린다."""
    h = frame.shape[0]
    positions = _palette_positions(h)

    # 팔레트 배경 패널
    pad = 8
    py1 = positions[0][1]  - PALETTE_RADIUS - pad
    py2 = positions[-1][1] + PALETTE_RADIUS + pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, py1), (PALETTE_CX * 2 + 6, py2), (15, 18, 30), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (0, py1), (PALETTE_CX * 2 + 6, py2), (0, 200, 220), 1)

    for idx, ((cx, cy), color_cv) in enumerate(zip(positions, PEN_COLORS_CV)):
        is_selected = (idx == color_idx)
        r = PALETTE_RADIUS + (3 if is_selected else 0)
        cv2.circle(frame, (cx, cy), r, color_cv, -1)
        if is_selected:
            cv2.circle(frame, (cx, cy), r + 3, (255, 255, 255), 2)
            cv2.circle(frame, (cx, cy), r + 6, color_cv,        1)
        else:
            cv2.circle(frame, (cx, cy), r + 1, (180, 180, 180), 1)


def _pencil_hit_palette(ix: int, iy: int, h: int):
    """(ix, iy)가 팔레트 원 위에 있으면 해당 색상 인덱스를, 아니면 -1을 반환."""
    for idx, (cx, cy) in enumerate(_palette_positions(h)):
        if (ix - cx) ** 2 + (iy - cy) ** 2 <= (PALETTE_RADIUS + 6) ** 2:
            return idx
    return -1


def _draw_mouse_cursor_icon(frame, ix, iy):
    """검지 끝(ix, iy)에 마우스 포인터 아이콘을 그린다."""
    s = 28  # 전체 크기
    # 포인터 외곽 (흰색 채움)
    pts = np.array([
        [ix,        iy       ],
        [ix,        iy + s   ],
        [ix + s//4, iy + s*3//4],
        [ix + s//2, iy + s   ],
        [ix + s*3//5, iy + s - 4],
        [ix + s//3, iy + s*3//4 - 2],
        [ix + s*2//3, iy + s*3//4 - 2],
    ], dtype=np.int32)
    # 단순 삼각형 포인터
    tri = np.array([
        [ix,            iy         ],
        [ix,            iy + s     ],
        [ix + s * 2//3, iy + s*2//3],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [tri], (255, 255, 255))
    cv2.polylines(frame, [tri], True, (30, 30, 30), 1, cv2.LINE_AA)
    # 끝 점 강조
    cv2.circle(frame, (ix, iy), 3, (0, 200, 255), -1)


def _draw_pencil_icon(frame, ix, iy, color_bgr):
    """검지 끝(ix, iy)에 연필 아이콘을 그린다."""
    # 연필 크기
    length = 36
    tip_len = 10
    width = 8

    # 연필은 오른쪽 위 방향으로 위치 (끝 = ix, iy)
    angle = -45  # degree
    rad = np.deg2rad(angle)
    dx = int(np.cos(rad) * length)
    dy = int(np.sin(rad) * length)

    # 연필 몸통 시작점 (끝에서 반대 방향)
    bx = ix - dx
    by = iy - dy

    perp_rad = rad + np.pi / 2
    pw = int(width / 2)
    pdx = int(np.cos(perp_rad) * pw)
    pdy = int(np.sin(perp_rad) * pw)

    # 몸통 사각형 꼭짓점
    pts_body = np.array([
        [bx + pdx, by + pdy],
        [bx - pdx, by - pdy],
        [ix - pdx - int(np.cos(rad) * tip_len), iy - pdy - int(np.sin(rad) * tip_len)],
        [ix + pdx - int(np.cos(rad) * tip_len), iy + pdy - int(np.sin(rad) * tip_len)],
    ], dtype=np.int32)

    # 연필 끝 삼각형 꼭짓점
    tip_base_x = ix - int(np.cos(rad) * tip_len)
    tip_base_y = iy - int(np.sin(rad) * tip_len)
    pts_tip = np.array([
        [tip_base_x + pdx, tip_base_y + pdy],
        [tip_base_x - pdx, tip_base_y - pdy],
        [ix, iy],
    ], dtype=np.int32)

    # 몸통 (선택된 색)
    cv2.fillPoly(frame, [pts_body], color_bgr)
    cv2.polylines(frame, [pts_body], True, (255, 255, 255), 1, cv2.LINE_AA)

    # 끝 삼각형 (밝은 살구색)
    cv2.fillPoly(frame, [pts_tip], (100, 190, 255))
    cv2.polylines(frame, [pts_tip], True, (255, 255, 255), 1, cv2.LINE_AA)

    # 지우개 끝 (반대쪽 작은 사각형)
    eraser_pts = np.array([
        [bx + pdx, by + pdy],
        [bx - pdx, by - pdy],
        [bx - pdx + int(np.cos(rad) * 6), by - pdy + int(np.sin(rad) * 6)],
        [bx + pdx + int(np.cos(rad) * 6), by + pdy + int(np.sin(rad) * 6)],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [eraser_pts], (80, 80, 200))
    cv2.polylines(frame, [eraser_pts], True, (255, 255, 255), 1, cv2.LINE_AA)

    # 연필 심 점
    cv2.circle(frame, (ix, iy), 2, (50, 50, 50), -1)


def _draw_eraser_icon(frame, ix, iy):
    """검지 끝(ix, iy)에 지우개 아이콘을 그린다."""
    w2, h2 = 18, 12
    ox, oy = ix + 5, iy - h2 - 5  # 검지 끝 오른쪽 위에 배치

    # 지우개 몸통 (흰색 사각형)
    cv2.rectangle(frame, (ox, oy), (ox + w2 * 2, oy + h2 * 2), (220, 220, 255), -1)
    cv2.rectangle(frame, (ox, oy), (ox + w2 * 2, oy + h2 * 2), (255, 255, 255), 2)

    # 하단 분홍색 줄 (지우개 특징)
    stripe_y = oy + int(h2 * 1.4)
    cv2.rectangle(frame, (ox, stripe_y), (ox + w2 * 2, oy + h2 * 2), (130, 100, 255), -1)
    cv2.line(frame, (ox, stripe_y), (ox + w2 * 2, stripe_y), (255, 255, 255), 1)

    # 검지 끝 연결 점
    cv2.circle(frame, (ix, iy), 3, (130, 100, 255), -1)
    cv2.line(frame, (ix, iy), (ox, oy + h2 * 2), (200, 200, 200), 1, cv2.LINE_AA)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  StripPanel – 우측 4컷 미리보기  (모던 다크)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class StripPanel(QWidget):
    STRIP_W = 230

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(self.STRIP_W)
        self.photos: list = []
        self._cached_pixmaps: list = []   # 사진 변경 시에만 재변환

    def setPhotos(self, photos):
        if len(photos) != len(self._cached_pixmaps):
            self._cached_pixmaps = [_cv_to_pixmap(p) for p in photos]
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
                p.drawPixmap(slot_rect.toRect(), self._cached_pixmaps[i])
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
        self.draw_mode:        str            = DRAW_DEFAULT
        self.hold_progress:    float          = 0.0   # 0.0–1.0 진행률
        self.hold_label:       str            = ''
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

        # 드로우 모드 배지 (오른쪽)
        mode_labels = {DRAW_DEFAULT: ('DEFAULT', QColor(0,220,255)),
                       DRAW_PAINTING: ('PAINT', QColor(50,230,120)),
                       DRAW_ERASE:   ('ERASE',  QColor(255,180,30))}
        m_text, m_color = mode_labels.get(self.draw_mode, ('?', C_GRAY1))
        badge_w, badge_h2 = 72, 22
        bx2 = w - 14
        badge_rect = QRectF(bx2 - badge_w, cy - badge_h2 // 2, badge_w, badge_h2)
        bp2 = QPainterPath()
        bp2.addRoundedRect(badge_rect, 11, 11)
        p.fillPath(bp2, QBrush(QColor(m_color.red(), m_color.green(), m_color.blue(), 40)))
        p.setPen(QPen(m_color, 1))
        p.drawPath(bp2)
        p.setFont(QFont("Courier", 8, QFont.Bold))
        p.setPen(m_color)
        p.drawText(badge_rect, Qt.AlignCenter, m_text)
        # 펜 색상 점
        if self.draw_mode == DRAW_PAINTING:
            px2, py2 = bx2 - badge_w - 14, cy
            p.setBrush(QBrush(self.drawing_color))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(px2, py2), 8, 8)

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
        hud2_h = 66
        hud2_y = h - hud2_h

        _draw_glass(p, QRectF(0, hud2_y, w, hud2_h), radius=0,
                    bg=QColor(8, 10, 18, 220),
                    border=QColor(0, 0, 0, 0))
        p.setPen(QPen(QColor(0, 220, 255, 70), 1))
        p.drawLine(0, hud2_y, w, hud2_y)

        # ── 홀드 진행 바 ──────────────────────────
        if self.hold_progress > 0:
            bx1, bx2 = 16, w - 16
            by = h - 10
            track = QRectF(bx1, by - 5, bx2 - bx1, 10)
            tp = QPainterPath()
            tp.addRoundedRect(track, 5, 5)
            p.fillPath(tp, QBrush(QColor(20, 25, 40)))
            fill = QRectF(bx1, by - 5, (bx2 - bx1) * self.hold_progress, 10)
            fp = QPainterPath()
            fp.addRoundedRect(fill, 5, 5)
            gc = QLinearGradient(bx1, 0, bx2, 0)
            gc.setColorAt(0.0, C_CYAN2)
            gc.setColorAt(1.0, C_CYAN)
            p.fillPath(fp, QBrush(gc))
            p.setFont(QFont("Courier", 8, QFont.Bold))
            p.setPen(C_CYAN)
            p.drawText(QRectF(0, hud2_y + 4, w, 18), Qt.AlignCenter, self.hold_label)
            top_h = 26
        else:
            top_h = 0

        # ── 제스처 힌트 버튼 ──────────────────────
        # (icon, label, gesture_key, accent_color)
        hints = [
            ("🖐", "OPEN",  "open",     QColor(  0, 220, 255)),
            ("✊", "FIST",  "fist",     QColor( 50, 230, 120)),
            ("👎", "CLEAR", "thumbdown", QColor(255, 180,  30)),
            ("✌", "PHOTO", "victory",   QColor(120,  60, 255)),
        ]
        seg_w = w // len(hints)
        btn_area_h = hud2_h - top_h - 4
        for hi, (icon, gn, gk, acol) in enumerate(hints):
            active = self.gesture == gk
            btn_w2, btn_h2 = seg_w - 14, btn_area_h - 4
            bx = hi * seg_w + 7
            by2 = hud2_y + top_h + 2
            btn_rect = QRectF(bx, by2, btn_w2, btn_h2)
            bp3 = QPainterPath()
            bp3.addRoundedRect(btn_rect, 7, 7)
            if active:
                p.fillPath(bp3, QBrush(QColor(acol.red(), acol.green(), acol.blue(), 45)))
                p.setPen(QPen(acol, 1.2))
                p.drawPath(bp3)
            else:
                p.fillPath(bp3, QBrush(QColor(18, 22, 38, 140)))
                p.setPen(QPen(QColor(acol.red(), acol.green(), acol.blue(), 35), 1))
                p.drawPath(bp3)
            col_main = acol if active else C_GRAY1
            col_sub  = QColor(acol.red(), acol.green(), acol.blue(), 160) if active else QColor(70, 78, 100)
            p.setFont(QFont("Arial", 10, QFont.Bold if active else QFont.Normal))
            p.setPen(col_main)
            p.drawText(QRectF(hi * seg_w, by2, seg_w, btn_h2 * 0.58),
                       Qt.AlignCenter, f"{icon} {gn}")
            hint_map = {'open': 'exit / reset', 'fist': 'paint / erase',
                        'thumbdown': 'clear  0.5s',  'victory': 'photo  0.2s'}
            p.setFont(QFont("Courier", 6))
            p.setPen(col_sub)
            p.drawText(QRectF(hi * seg_w, by2 + btn_h2 * 0.58, seg_w, btn_h2 * 0.42),
                       Qt.AlignCenter, hint_map.get(gk, ''))

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
        p.drawText(QRectF(2, 2, w, title_h), Qt.AlignCenter, "AirCanvas-Your 4-cut")
        p.setPen(C_WHITE)
        p.drawText(QRectF(0, 0, w, title_h), Qt.AlignCenter, "AirCanvas-Your 4-cut")
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
            # 영상 로딩 중: 회전 스피너
            vid_x = half_w + gap
            vid_w_avail = w - vid_x - 18
            placeholder = QRectF(vid_x, avail_y, vid_w_avail, avail_h)
            _draw_glass(p, placeholder, radius=3,
                        bg=QColor(12, 14, 24, 180),
                        border=QColor(120, 60, 255, 60))

            cx_s = int(placeholder.x() + placeholder.width() / 2)
            cy_s = int(placeholder.y() + placeholder.height() / 2)
            R_s  = 44
            angle = (time.time() * 360) % 360  # 초당 1바퀴

            # 트랙 링
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(120, 60, 255, 40), 6, Qt.SolidLine, Qt.RoundCap))
            p.drawEllipse(QPointF(cx_s, cy_s), R_s, R_s)

            # 회전 호 (270도)
            p.setPen(QPen(C_PURPLE, 6, Qt.SolidLine, Qt.RoundCap))
            arc_r = QRect(cx_s - R_s, cy_s - R_s, R_s * 2, R_s * 2)
            p.drawArc(arc_r, int((90 - angle) * 16), -270 * 16)

            # 외곽 글로우
            p.setPen(QPen(QColor(120, 60, 255, 30), 2))
            p.drawEllipse(QPointF(cx_s, cy_s), R_s + 10, R_s + 10)

            # 중앙 텍스트
            p.setFont(QFont("Courier", 9, QFont.Bold))
            p.setPen(QColor(120, 60, 255, 180))
            p.drawText(QRectF(cx_s - 60, cy_s + R_s + 12, 120, 20),
                       Qt.AlignCenter, "LOADING...")

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
        self._palette_touched = False  # 팔레트 터치 디바운스
        self.line_thickness        = 5
        self.draw_mode             = DRAW_DEFAULT
        self.canvas_dirty          = False   # 캔버스에 픽셀이 있으면 True
        self.last_gesture          = None
        self.gesture_start         = None
        # ── 홀드 타이머
        self.fist_hold_start       = None
        self.fist_toggled          = False
        self.thumbdown_hold_start  = None
        self.thumbdown_fired       = False
        self.victory_hold_start    = None
        self.victory_fired         = False
        self.countdown_start       = None
        self.flash_start           = None
        self.frame_index      = 0
        self.fps              = 30
        self._review_cap      = None
        self._session_dir     = None
        self._vid_tmp         = os.path.join(SAVE_DIR, '_recording_tmp_v3.avi')
        self._out_writer      = None
        self._write_queue     = queue.Queue(maxsize=8)
        self._writer_thread   = None

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
        print("  🖐 Open_Palm   : Default 복귀 / 리뷰 리셋")
        print("  ✊ Fist (0.2s) : Default/Erase→Painting / Painting→Erase")
        print("  👎 Thumb_Down  : 캔버스 전체 지우기 (0.5s, Default 상태에서)")
        print("  ✌ Victory     : 촬영 (0.2s, Default 상태에서)")
        print("  ESC / Q       : 종료")
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

        # REVIEW 중에는 카메라 스트림을 수신만 하고 제스처/녹화 스킵
        if self.state == STATE_REVIEW:
            self.frame_index += 1
            img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            ts_ms    = int(self.frame_index * 1000 / self.fps)
            result   = self.recognizer.recognize_for_video(mp_image, ts_ms)

            raw_g = (result.gestures[0][0].category_name
                     if result.gestures else '')
            cur_g = _GESTURE_MAP.get(raw_g, None)

            # Open_Palm: 이전 프레임이 open이 아닐 때만 트리거 (엣지 감지)
            if cur_g == 'open' and self.last_gesture != 'open':
                self.last_gesture = 'open'
                self._reset_session(h, w)
                self.cam_view.state = STATE_WAITING
                self.strip.setVisible(True)
                self.cam_view.frame_pix = _cv_to_pixmap(frame)
                self.cam_view.update()
                return

            self.last_gesture = cur_g

            if self._review_cap:
                ret_v, vframe = self._review_cap.read()
                if not ret_v:
                    self._review_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_v, vframe = self._review_cap.read()
                if ret_v and vframe is not None:
                    self.cam_view.review_video_pix = _cv_to_pixmap(vframe)
            self.cam_view.update()
            return

        if self.draw_canvas is None:
            self.draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        gesture = None
        hold_progress = 0.0
        hold_label    = ''

        if self.state not in (STATE_REVIEW,):
            img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            self.frame_index += 1
            ts_ms    = int(self.frame_index * 1000 / self.fps)
            result = self.recognizer.recognize_for_video(mp_image, ts_ms)

            if result.hand_landmarks:
                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    raw_g   = result.gestures[i][0].category_name if result.gestures else 'None'
                    gesture = _GESTURE_MAP.get(raw_g, None)
                    ix = int(hand_landmarks[8].x * w)
                    iy = int(hand_landmarks[8].y * h)

                    # ── Open_Palm: 즉시 DEFAULT 복귀 ──────────────
                    if gesture == 'open':
                        self.draw_mode = DRAW_DEFAULT
                        self.prev_x = self.prev_y = None
                        if self.state in (STATE_COUNTDOWN, STATE_FLASH):
                            self.state = STATE_WAITING

                    # ── Fist: 0.2s 홀드 → PAINT/ERASE 토글 ────────
                    elif gesture == 'fist':
                        if self.last_gesture != 'fist':
                            self.fist_hold_start = now
                            self.fist_toggled    = False
                        if self.fist_hold_start is not None:
                            held = now - self.fist_hold_start
                            hold_progress = min(held / HOLD_FIST, 1.0)
                            hold_label    = '✊  FIST  —  TOGGLE  PAINT / ERASE'
                            if held >= HOLD_FIST and not self.fist_toggled:
                                if self.draw_mode in (DRAW_DEFAULT, DRAW_ERASE):
                                    self.draw_mode = DRAW_PAINTING
                                else:
                                    self.draw_mode = DRAW_ERASE
                                self.fist_toggled = True
                        self.prev_x = self.prev_y = None

                    # ── Thumb_Down: 0.5s 홀드 → 전체 지우기 (DEFAULT) ─
                    elif gesture == 'thumbdown':
                        if self.last_gesture != 'thumbdown':
                            self.thumbdown_hold_start = now
                            self.thumbdown_fired      = False
                        if self.thumbdown_hold_start is not None:
                            held = now - self.thumbdown_hold_start
                            hold_progress = min(held / HOLD_CLEAR, 1.0)
                            hold_label    = '👎  THUMB DOWN  —  CLEAR  CANVAS'
                            if (held >= HOLD_CLEAR and not self.thumbdown_fired):
                                self.draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                                self.canvas_dirty = False
                                self.thumbdown_fired = True
                        self.prev_x = self.prev_y = None

                    # ── Victory: 0.2s 홀드 → 촬영 (DEFAULT) ──────────
                    elif gesture == 'victory':
                        if self.last_gesture != 'victory' or self.victory_hold_start is None:
                            self.victory_hold_start = now
                            self.victory_fired      = False
                        if self.victory_hold_start is not None:
                            held = now - self.victory_hold_start
                            hold_progress = 1.0 if HOLD_PHOTO == 0 else min(held / HOLD_PHOTO, 1.0)
                            hold_label    = '✌  VICTORY  —  CAPTURE'
                            if (held >= HOLD_PHOTO and not self.victory_fired
                                    and self.state == STATE_WAITING):
                                self.state           = STATE_COUNTDOWN
                                self.countdown_start = now
                                self.victory_fired   = True
                        self.prev_x = self.prev_y = None

                    # ── 그리기 / 지우기 ────────────────────────────
                    else:
                        if self.state == STATE_WAITING:
                            if self.draw_mode == DRAW_PAINTING:
                                if self.prev_x is not None and self.prev_y is not None:
                                    cv2.line(self.draw_canvas,
                                             (self.prev_x, self.prev_y), (ix, iy),
                                             self.draw_color_cv, self.line_thickness)
                                    self.canvas_dirty = True
                                self.prev_x, self.prev_y = ix, iy
                            elif self.draw_mode == DRAW_ERASE:
                                cv2.circle(self.draw_canvas, (ix, iy),
                                           self.line_thickness * 4, (0, 0, 0), -1)
                                self.canvas_dirty = True
                                self.prev_x = self.prev_y = None
                            else:  # DEFAULT
                                # 마우스 커서로 팔레트 터치 → 색 변경
                                hit_idx = _pencil_hit_palette(ix, iy, h)
                                if hit_idx >= 0:
                                    if not self._palette_touched:
                                        self.color_idx     = hit_idx
                                        self.draw_color_cv = PEN_COLORS_CV[hit_idx]
                                        self.draw_color_qc = PEN_COLORS_QC[hit_idx]
                                        self._palette_touched = True
                                else:
                                    self._palette_touched = False
                                self.prev_x = self.prev_y = None
                        else:
                            self._palette_touched = False
                            self.prev_x = self.prev_y = None

                    # ── 랜드마크
                    for conn in HAND_CONNECTIONS:
                        s = hand_landmarks[conn[0]]; e = hand_landmarks[conn[1]]
                        cv2.line(frame, (int(s.x*w), int(s.y*h)),
                                 (int(e.x*w), int(e.y*h)), (0, 200, 180), 1)
                    for lm in hand_landmarks:
                        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, (0, 220, 255), -1)

                    # ── 커서
                    if self.draw_mode == DRAW_PAINTING:
                        _draw_pencil_icon(frame, ix, iy, self.draw_color_cv)
                    elif self.draw_mode == DRAW_ERASE:
                        _draw_eraser_icon(frame, ix, iy)
                    else:  # DEFAULT – 마우스 포인터
                        _draw_mouse_cursor_icon(frame, ix, iy)
            else:
                self.prev_x = self.prev_y = None

        # ── 홀드 타이머 리셋 (제스처 변경 시)
        if gesture != 'fist':
            self.fist_hold_start = None
            self.fist_toggled    = False
        if gesture != 'thumbdown':
            self.thumbdown_hold_start = None
            self.thumbdown_fired      = False
        if gesture != 'victory':
            self.victory_hold_start = None
            self.victory_fired      = False

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
                _canvas_snap = self.draw_canvas.copy()
                _thickness   = self.line_thickness
                def _make_mask_bg(canvas=_canvas_snap, thick=_thickness):
                    self.photo_masks.append(_make_inpaint_mask(canvas, thick))
                threading.Thread(target=_make_mask_bg, daemon=True).start()
                self.draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)
                self.canvas_dirty = False
                self.state = STATE_FLASH
                self.flash_start = now
                print(f"[{len(self.photos)}/{TOTAL_SHOTS}] 촬영!")
                if len(self.photos) >= TOTAL_SHOTS:
                    self._session_dir = os.path.join(
                        SAVE_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
                    _photos_snap = list(self.photos)
                    _masks_snap  = list(self.photo_masks)
                    _sdir        = self._session_dir
                    threading.Thread(
                        target=_save_final,
                        args=(_photos_snap,),
                        kwargs={'masks': _masks_snap, 'session_dir': _sdir},
                        daemon=True
                    ).start()

        elif self.state == STATE_FLASH:
            if now - self.flash_start >= FLASH_SEC:
                if len(self.photos) >= TOTAL_SHOTS:
                    self.state        = STATE_REVIEW
                    self.last_gesture = None   # 리뷰 진입 시 제스처 초기화
                    _vid_play = os.path.join(SAVE_DIR, '_recording_play_v3.avi')
                    if self._out_writer and self._out_writer is not False:
                        self._write_queue.put(None)   # 백그라운드 writer 종료
                        self._out_writer.release()
                        self._out_writer = None
                        self._writer_thread = None
                    _photos_snap = list(self.photos)
                    _vid_src     = self._vid_tmp
                    _vid_dst     = _vid_play
                    def _prepare_review(photos=_photos_snap, src=_vid_src, dst=_vid_dst):
                        import shutil as _sh
                        collage = _make_final_collage(photos)
                        if collage is not None:
                            self.cam_view.review_pix = _cv_to_pixmap(collage)
                        if os.path.exists(src):
                            _sh.copy2(src, dst)
                            self._review_cap = cv2.VideoCapture(dst)
                    threading.Thread(target=_prepare_review, daemon=True).start()
                else:
                    self.state = STATE_WAITING
                    # victory를 계속 들고 있어도 다음 컷을 찍을 수 있도록 리셋
                    self.victory_fired      = False
                    self.victory_hold_start = None

        # ── 그리기 합성 (캔버스에 내용이 있을 때만)
        if self.state not in (STATE_REVIEW,):
            if self.canvas_dirty:
                mask = cv2.cvtColor(self.draw_canvas, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                mask_inv  = cv2.bitwise_not(mask)
                frame_bg  = cv2.bitwise_and(frame, frame, mask=mask_inv)
                canvas_fg = cv2.bitwise_and(self.draw_canvas, self.draw_canvas, mask=mask)
                frame     = cv2.add(frame_bg, canvas_fg)

# ── 녹화 (백그라운드 스레드로 인코딩)
        if self._out_writer is None and self.state not in (STATE_REVIEW,):
            _vh, _vw = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(self._vid_tmp, fourcc, self.fps, (_vw, _vh))
            if writer.isOpened():
                self._out_writer = writer
                # 백그라운드 인코딩 스레드 시작
                def _writer_worker(w=writer, q=self._write_queue):
                    while True:
                        item = q.get()
                        if item is None:
                            break
                        w.write(item)
                self._writer_thread = threading.Thread(
                    target=_writer_worker, daemon=True)
                self._writer_thread.start()
                print(f"[녹화 시작] {self._vid_tmp}")
            else:
                writer.release()
                self._out_writer = False
        if self._out_writer and self._out_writer is not False and self.state not in (STATE_REVIEW,):
            try:
                self._write_queue.put_nowait(frame.copy())
            except queue.Full:
                pass  # 큐 가득 차면 프레임 드롭 (UI 블로킹 방지)

        # ── 팔레트 UI 렌더링 (WAITING + PAINT/DEFAULT 상태에서만)
        if self.state == STATE_WAITING:
            _draw_color_palette_on_frame(frame, self.color_idx)

        # ── 위젯 업데이트
        self.cam_view.frame_pix       = _cv_to_pixmap(frame)
        self.cam_view.state           = self.state
        self.cam_view.taken           = len(self.photos)
        self.cam_view.countdown_start = self.countdown_start
        self.cam_view.flash_start     = self.flash_start
        self.cam_view.gesture         = gesture
        self.cam_view.gesture_start   = self.gesture_start
        self.cam_view.drawing_color   = self.draw_color_qc
        self.cam_view.draw_mode       = self.draw_mode
        self.cam_view.hold_progress   = hold_progress
        self.cam_view.hold_label      = hold_label

        self.strip.setVisible(self.state != STATE_REVIEW)
        self.strip.setPhotos(self.photos)
        self.cam_view.update()

    def _reset_session(self, h: int, w: int):
        self.photos = []
        self.photo_masks = []
        self.draw_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        self.canvas_dirty = False
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
            self._write_queue.put(None)
            self._out_writer.release()
        self._out_writer    = None
        self._writer_thread = None
        self._write_queue   = queue.Queue(maxsize=8)  # 큐 초기화
        self.draw_mode      = DRAW_DEFAULT
        self.canvas_dirty   = False
        self.strip.setPhotos([])   # photos + 캐시 동시 초기화
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
        # 백그라운드 writer 스레드 종료
        if self._writer_thread and self._writer_thread.is_alive():
            self._write_queue.put(None)
            self._writer_thread.join(timeout=2)
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
