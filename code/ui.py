import cv2
import numpy as np
import config
import assets

# ── 팔레트 ────────────────────────────────────────────────────────
def palette_positions(h):
    n       = len(config.PEN_COLORS)
    total   = n * config.PALETTE_SPACING
    start_y = (h - total) // 2 + config.PALETTE_SPACING // 2
    return [(config.PALETTE_CX, start_y + i * config.PALETTE_SPACING) for i in range(n)]


def draw_color_palette(frame, color_idx):
    h         = frame.shape[0]
    positions = palette_positions(h)
    for idx, ((cx, cy), color) in enumerate(zip(positions, config.PEN_COLORS)):
        is_sel = (idx == color_idx)
        r = config.PALETTE_RADIUS if is_sel else config.PALETTE_RADIUS - 4
        cv2.circle(frame, (cx, cy), r, color, -1)
        if is_sel:
            cv2.circle(frame, (cx, cy), r + 3, config.WHITE, 2)


def palette_hit(ix, iy, h):
    for idx, (cx, cy) in enumerate(palette_positions(h)):
        if (ix - cx) ** 2 + (iy - cy) ** 2 <= (config.PALETTE_RADIUS + 6) ** 2:
            return idx
    return -1


# ── 커서 아이콘 ───────────────────────────────────────────────────
def draw_pencil_icon(frame, ix, iy, color_bgr):
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
    cv2.polylines(frame, [body], True, config.WHITE, 1, cv2.LINE_AA)
    cv2.fillPoly(frame, [tip], (100, 190, 255))
    cv2.polylines(frame, [tip], True, config.WHITE, 1, cv2.LINE_AA)
    cv2.circle(frame, (ix, iy), 2, (50, 50, 50), -1)


def draw_eraser_icon(frame, ix, iy):
    w2, h2 = 18, 12
    ox, oy = ix + 5, iy - h2 - 5
    cv2.rectangle(frame, (ox, oy), (ox + w2 * 2, oy + h2 * 2), (220, 220, 255), -1)
    cv2.rectangle(frame, (ox, oy), (ox + w2 * 2, oy + h2 * 2), config.WHITE, 2)
    sy = oy + int(h2 * 1.4)
    cv2.rectangle(frame, (ox, sy), (ox + w2 * 2, oy + h2 * 2), (130, 100, 255), -1)
    cv2.line(frame, (ox, sy), (ox + w2 * 2, sy), config.WHITE, 1)
    cv2.circle(frame, (ix, iy), 3, (130, 100, 255), -1)


def draw_cursor_icon(frame, ix, iy):
    s   = 28
    tri = np.array([[ix, iy], [ix, iy + s], [ix + s * 2 // 3, iy + s * 2 // 3]], dtype=np.int32)
    cv2.fillPoly(frame, [tri], config.WHITE)
    cv2.polylines(frame, [tri], True, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.circle(frame, (ix, iy), 3, (0, 200, 255), -1)


# ── 프레임 + 사진 렌더 ────────────────────────────────────────────
def render_frame(canvas, photos):
    if assets.frame_raw is None:
        return

    fh, fw  = assets.frame_raw.shape[:2]
    avail_w = canvas.shape[1] - config.FRAME_X
    avail_h = canvas.shape[0] - config.FRAME_Y
    scale   = min(avail_w / fw, avail_h / fh)
    nw, nh  = int(fw * scale), int(fh * scale)
    frame_disp = cv2.resize(assets.frame_raw, (nw, nh))

    for i, (sx, sy, sw, sh) in enumerate(config.PHOTO_SLOTS):
        if i < len(photos):
            ph_s, pw_s = photos[i].shape[:2]
            ps = min(config.DISPLAY_PHOTO_W / pw_s, config.DISPLAY_PHOTO_H / ph_s)
            pw, ph = int(pw_s * ps), int(ph_s * ps)
            resized = cv2.resize(photos[i], (pw, ph))
            ax  = config.FRAME_X + int(sx * scale) + (int(sw * scale) - pw) // 2
            ay  = config.FRAME_Y + int(sy * scale) + (int(sh * scale) - ph) // 2
            ay2 = min(ay + ph, canvas.shape[0])
            ax2 = min(ax + pw, canvas.shape[1])
            canvas[ay:ay2, ax:ax2] = resized[:ay2 - ay, :ax2 - ax]

    roi = canvas[config.FRAME_Y:config.FRAME_Y + nh, config.FRAME_X:config.FRAME_X + nw]
    if frame_disp.shape[2] == 4:
        alpha = frame_disp[:, :, 3:4] / 255.0
        roi[:] = (frame_disp[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)
    else:
        roi[:] = frame_disp


# ── 정보 패널 ─────────────────────────────────────────────────────
_GESTURE_LABEL = {
    'cursor':    'Pointing_Up',
    'peace':     'Victory',
    'fist':      'Closed_Fist',
    'open':      'Open_Palm',
    'thumbdown': 'Thumb_Down',
    'thumbup':   'Thumb_Up',
}
_MODE_LABEL = {
    config.DRAW_DEFAULT:  'DEFAULT',
    config.DRAW_PAINTING: 'PAINT',
    config.DRAW_ERASE:    'ERASE',
}


def draw_info_panel(canvas, gesture, result, draw_mode):
    x, y, w = config.INFO_X, config.INFO_Y, config.INFO_W
    cy = y + 14
    if result is not None and result.hand_landmarks:
        raw  = result.gestures[0][0].category_name if result.gestures else 'None'
        g    = _GESTURE_LABEL.get(gesture, raw)
        m    = 'SHOOT' if gesture == 'peace' else 'RESET' if gesture == 'thumbdown' else _MODE_LABEL.get(draw_mode, 'DEFAULT')
        text = f'{g} [{m}]'
        tw   = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
        cv2.putText(canvas, text, (x + w // 2 - tw // 2, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.BLACK, 1, cv2.LINE_AA)


# ── 선택 그리드 ───────────────────────────────────────────────────
def cell_hit(finger_x, finger_y, canvas_w, canvas_h, cells):
    src_w, src_h = 1024, 600
    sx = finger_x * src_w / canvas_w
    sy = finger_y * src_h / canvas_h
    for i, (x1, y1, x2, y2) in enumerate(cells):
        if x1 <= sx <= x2 and y1 <= sy <= y2:
            return i
    return -1


def draw_selection_grid(canvas, src_img, cells, hovered_cell, finger_x=-1, finger_y=-1):
    ch, cw = canvas.shape[:2]
    if src_img is None:
        cv2.putText(canvas, "image not found", (50, ch // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, config.WHITE, 2)
        return hovered_cell

    canvas[:] = cv2.resize(src_img, (cw, ch))

    hover = hovered_cell
    if finger_x >= 0 and finger_y >= 0:
        hit = cell_hit(finger_x, finger_y, cw, ch, cells)
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

    overlay = canvas.copy()
    cv2.rectangle(overlay, (18, 18), (cw - 18, 92), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
    cv2.putText(canvas, "Move the cursor to a tile", (36, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, config.WHITE, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Hold THUMB UP to confirm selection", (36, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

    return hover


def draw_theme_grid(canvas, hovered_cell, finger_x=-1, finger_y=-1):
    return draw_selection_grid(canvas, assets.source_theme_img, assets.SOURCE_THEME_CELLS,
                               hovered_cell, finger_x, finger_y)


def draw_bg_grid(canvas, hovered_cell, finger_x=-1, finger_y=-1):
    return draw_selection_grid(canvas, assets.source_bg_img, config.SOURCE_BG_CELLS,
                               hovered_cell, finger_x, finger_y)
