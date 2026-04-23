import os
import sys
import socket
import threading
import http.server
import socketserver
import cv2
import numpy as np
import qrcode
from PIL import Image
import config
import assets

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gmail_api'))
from send_message import gmail_send_message_with_attachment

email_status     = None
email_input_text = ''


def _get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.254.254.254', 1))
        return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        s.close()


LOCAL_IP = _get_local_ip()
print(f"[서버] 로컬 IP: {LOCAL_IP}:{config.HTTP_PORT}")


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=config.SAVE_DIR, **kwargs)

    def log_message(self, *args):
        pass


class _ReuseServer(socketserver.TCPServer):
    allow_reuse_address = True


def _start_http_server():
    with _ReuseServer(('', config.HTTP_PORT), _QuietHandler) as httpd:
        httpd.serve_forever()


threading.Thread(target=_start_http_server, daemon=True).start()
print(f"[서버] http://{LOCAL_IP}:{config.HTTP_PORT} 에서 사진 서비스 중")


def make_qr_cv(url, size=None):
    if size is None:
        size = config.QR_SIZE
    qr = qrcode.QRCode(box_size=4, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    pil_img = qr.make_image(fill_color='black', back_color='white').convert('RGB')
    pil_img = pil_img.resize((size, size), Image.NEAREST)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def save_final(photos, session_dir, masks=None):
    os.makedirs(session_dir, exist_ok=True)
    for i, photo in enumerate(photos):
        cv2.imwrite(os.path.join(session_dir, f"shot_{i+1}.jpg"), photo)
        print(f"  저장: shot_{i+1}.jpg")
        if masks is not None and i < len(masks):
            mask_path = os.path.join(session_dir, f"shot_{i+1}_mask.png")
            cv2.imwrite(mask_path, masks[i])
            print(f"  저장: shot_{i+1}_mask.png")

    if assets.frame_raw is not None:
        fh, fw = assets.frame_raw.shape[:2]
        collage = np.ones((fh, fw, 3), dtype=np.uint8) * 255

        for i, (sx, sy, sw, sh) in enumerate(config.PHOTO_SLOTS):
            if i < len(photos):
                ph_s, pw_s = photos[i].shape[:2]
                pw = config.SAVE_PHOTO_W
                ph = int(ph_s * (pw / pw_s))
                resized = cv2.resize(photos[i], (pw, ph))
                ox  = sx + (sw - pw) // 2
                oy  = sy + (sh - ph) // 2
                oy2 = min(oy + ph, fh)
                ox2 = min(ox + pw, fw)
                collage[oy:oy2, ox:ox2] = resized[:oy2 - oy, :ox2 - ox]

        if assets.frame_raw.shape[2] == 4:
            alpha   = assets.frame_raw[:, :, 3:4] / 255.0
            collage = (assets.frame_raw[:, :, :3] * alpha + collage * (1 - alpha)).astype(np.uint8)
        else:
            collage = assets.frame_raw.copy()

        cv2.imwrite(os.path.join(session_dir, "4cut.jpg"), collage)

    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AirCanvas 4컷</title>
  <style>
    body { margin: 0; background: #111; display: flex; flex-direction: column; align-items: center; padding: 16px; }
    h2 { color: #fff; font-family: sans-serif; margin: 12px 0; }
    .photos { display: flex; gap: 16px; flex-wrap: nowrap; justify-content: center; }
    .photo-box { text-align: center; }
    .photo-box p { color: #aaa; font-family: sans-serif; margin: 6px 0; }
    img { max-width: 45vw; border-radius: 8px; }
    #ai-status { color: #888; font-family: sans-serif; font-size: 0.9em; margin-top: 8px; }
  </style>
</head>
<body>
  <h2>AirCanvas 4컷</h2>
  <div class="photos">
    <div class="photo-box">
      <img src="4cut.jpg">
      <p>Original</p>
    </div>
    <div class="photo-box" id="ai-box">
      <img id="ai-img" src="ai_4cut.jpg" onerror="retryAI(this)">
      <p>AI Generated</p>
    </div>
  </div>
  <p id="ai-status"></p>
  <script>
    var retrying = false;
    function retryAI(img) {
      if (retrying) return;
      retrying = true;
      document.getElementById('ai-status').textContent = 'AI 이미지 생성 중...';
      img.style.opacity = '0';
      function attempt() {
        var t = new Image();
        t.onload = function() {
          img.src = t.src;
          img.style.opacity = '1';
          document.getElementById('ai-status').textContent = '';
          retrying = false;
        };
        t.onerror = function() { setTimeout(attempt, 3000); };
        t.src = 'ai_4cut.jpg?' + Date.now();
      }
      setTimeout(attempt, 3000);
    }
  </script>
</body>
</html>"""
    with open(os.path.join(session_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✓ 저장 완료 → {session_dir}")


def release_and_save(writer):
    writer.release()
    try:
        if os.path.exists(config.VID_PLAY):
            os.remove(config.VID_PLAY)
        if os.path.exists(config.VID_TMP):
            os.rename(config.VID_TMP, config.VID_PLAY)
            print("[녹화 저장 완료]")
    except Exception as e:
        print(f"[녹화 저장 실패] {e}")


def send_email_async(attachment_paths, recipient):
    global email_status
    print(f"[이메일] 전송 중 → {recipient}")
    result = gmail_send_message_with_attachment(attachment_paths, recipient=recipient)
    if result:
        email_status = 'sent'
        print("[이메일] 전송 완료!")
    else:
        email_status = 'error'
        print("[이메일] 전송 실패")
