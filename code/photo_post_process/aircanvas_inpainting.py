import os
import requests

api_key = os.getenv("STABILITY_API_KEY")


def step1_inpaint(image_bytes: bytes, mask_bytes: bytes, prompt: str) -> bytes:
    """[Step 1] 원본 이미지 + 마스크로 스케치 영역을 채운다."""
    print("\n[Step 1] 스케치 영역 인페인팅 중...")
    resp = requests.post(
        "https://api.stability.ai/v2beta/stable-image/edit/inpaint",
        headers={"authorization": f"Bearer {api_key}", "accept": "image/*"},
        files={
            "image": ("input.png", image_bytes, "image/png"),
            "mask":  ("mask.png",  mask_bytes,  "image/png"),
        },
        data={"prompt": prompt, 
            #   if you notice seams or rough edges around the inpainted content
              "grow_mask": 15,
              "output_format": "png", # default
            #    3d-model analog-film anime cinematic comic-book digital-art enhance fantasy-art isometric line-art low-poly modeling-compound neon-punk origami photographic pixel-art tile-texture
              "style_preset": "digital-art"
              },
    )
    if resp.status_code != 200:
        raise RuntimeError(f"[Step 1] 인페인팅 실패: {resp.text}")
    print("  완료.")
    return resp.content