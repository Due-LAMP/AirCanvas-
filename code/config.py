import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 촬영 ──────────────────────────────────────────────────────────
TOTAL_SHOTS   = 4
COUNTDOWN_SEC = 3
FLASH_SEC     = 0.5
FPS           = 30
REVIEW_SEC    = 10

# ── Stability AI ──────────────────────────────────────────────────
STABILITY_API_KEY = os.environ.get('STABILITY_API_KEY', '')

# ── 그리기 모드 ───────────────────────────────────────────────────
DRAW_DEFAULT  = 'default'
DRAW_PAINTING = 'painting'
DRAW_ERASE    = 'erase'

# ── 홀드 타이밍 ───────────────────────────────────────────────────
HOLD_FIST            = 0.2
HOLD_CLEAR           = 0.5
HOLD_PHOTO           = 0.2
HOLD_RESET           = 3.0
HOLD_SELECT          = 0.8
THEME_TO_BG_COOLDOWN = 1.5

# ── 테마 그리드 ───────────────────────────────────────────────────
SOURCE_THEME_PATH      = os.path.join(_BASE_DIR, 'image/source_theme.png')
SOURCE_THEME_MASK_PATH = os.path.join(_BASE_DIR, 'image/source_theme_mask.png')
SOURCE_THEME_CELLS_FALLBACK = [
    (86,  190, 318, 346),
    (396, 190, 627, 346),
    (705, 190, 937, 346),
    (86,  398, 318, 554),
    (396, 398, 627, 554),
    (705, 398, 937, 554),
]
SOURCE_THEME_NAMES = ['analog-film', 'origami', 'pixel-art', 'comic-book', '3d-model', 'photographic']

# ── 배경 그리드 ───────────────────────────────────────────────────
SOURCE_BG_PATH      = os.path.join(_BASE_DIR, 'image/source_background.png')
SOURCE_BG_MASK_PATH = os.path.join(_BASE_DIR, 'image/source_background_mask.png')
SOURCE_BG_COLS      = 4
SOURCE_BG_ROWS      = 2
SOURCE_BG_CELLS = [
    (47,  182, 253, 337),
    (290, 182, 496, 337),
    (533, 182, 739, 337),
    (775, 182, 982, 337),
    (47,  391, 253, 546),
    (290, 391, 496, 546),
    (533, 391, 739, 546),
    (775, 391, 982, 546),
]
SOURCE_BG_NAMES = ['white', 'skyblue', 'lightpink', 'lightgreen', 'beach', 'space', 'zombie', 'chimchakman']
SOURCE_IMAGE_DIR = os.path.join(_BASE_DIR, 'image/source')
SOURCE_BG_FILES  = [
    'White.jpg', 'Skyblue.jpg', 'Lightpink.jpg', 'Lightgreen.jpg',
    'Beach.jpg', 'Space.jpg',   'Zombie.jpg',     'Chimchakman.jpg',
]

# ── 팔레트 ────────────────────────────────────────────────────────
PALETTE_CX      = 32
PALETTE_RADIUS  = 18
PALETTE_SPACING = 46

# ── 이미지 경로 ───────────────────────────────────────────────────
BG_IMAGE_PATH        = os.path.join(_BASE_DIR, 'image/background_line.png')
BG_RESULT_IMAGE_PATH = os.path.join(_BASE_DIR, 'image/background.png')
FRAME_IMAGE_PATH     = os.path.join(_BASE_DIR, 'image/4cut_frame.png')
INTRO_IMAGE_PATH     = os.path.join(_BASE_DIR, 'image/page_1.png')

# ── 레이아웃 ──────────────────────────────────────────────────────
CAM_X        = 80
CAM_W, CAM_H = 560, 420
CAM_Y        = (600 - CAM_H) // 2
INFO_X       = CAM_X
INFO_Y       = CAM_Y - 28
INFO_W       = CAM_W
FRAME_X, FRAME_Y = 720, 10

PHOTO_SLOTS = [
    (3,  38, 420, 217),
    (3, 320, 420, 223),
    (3, 628, 420, 186),
    (3, 908, 420, 200),
]
DISPLAY_PHOTO_W = 160
DISPLAY_PHOTO_H = 120
SAVE_PHOTO_W    = 340

# ── 컬러 ──────────────────────────────────────────────────────────
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

# ── 저장 경로 ─────────────────────────────────────────────────────
SAVE_DIR  = os.path.join(_BASE_DIR, 'photobooth_output')
VID_TMP   = os.path.join(SAVE_DIR, '_rec_tmp.avi')
VID_PLAY  = os.path.join(SAVE_DIR, '_rec_play.avi')

# ── HTTP ──────────────────────────────────────────────────────────
HTTP_PORT      = 8080
QR_SIZE        = 150

# ── 상태 ──────────────────────────────────────────────────────────
STATE_INTRO        = 'intro'
STATE_SELECT_THEME = 'select_theme'
STATE_SELECT_BG    = 'select_bg'
STATE_WAITING      = 'waiting'
STATE_COUNTDOWN    = 'countdown'
STATE_FLASH        = 'flash'
STATE_REVIEW       = 'review'
STATE_RESULT       = 'result'
STATE_EMAIL_INPUT  = 'email_input'

# ── 제스처 맵 ─────────────────────────────────────────────────────
GESTURE_MAP = {
    'Closed_Fist': 'fist',
    'Victory':     'peace',
    'Open_Palm':   'open',
    'Thumb_Down':  'thumbdown',
    'Pointing_Up': 'cursor',
    'Thumb_Up':    'thumbup',
}
