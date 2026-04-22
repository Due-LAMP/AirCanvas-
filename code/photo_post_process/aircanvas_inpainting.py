import os
import requests

api_key = os.getenv("STABILITY_API_KEY")


def step1_inpaint(
    image_bytes: bytes,
    mask_bytes: bytes,
    prompt: str,
    style_preset: str | None = None,
) -> bytes:
    """[Step 1] 원본 이미지 + 마스크로 스케치 영역을 채운다."""
    print("\n[Step 1] 스케치 영역 인페인팅 중...")
    data = {
        "prompt": prompt,
        "grow_mask": 12,
        "output_format": "png",
        "negative_prompt": (
            "person, human, face, head, body, portrait, selfie, "
            "extra person, duplicate person, extra limbs, bad anatomy, disfigured, blurry, low quality, "
            "close up"
        ),
    }
    if style_preset:
        data["style_preset"] = style_preset

    request_metadata = {
        "style_preset": style_preset,
        # "grow_mask": data["grow_mask"],
        "output_format": data["output_format"],
        "prompt": data["prompt"],
        "negative_prompt": data["negative_prompt"],
        # "image_bytes": len(image_bytes),
        # "mask_bytes": len(mask_bytes),
    }
    print(f"[Step 1] Stability metadata: {request_metadata}")

    resp = requests.post(
        "https://api.stability.ai/v2beta/stable-image/edit/inpaint",
        headers={"authorization": f"Bearer {api_key}", "accept": "image/*"},
        files={
            "image": ("input.png", image_bytes, "image/png"),
            "mask":  ("mask.png",  mask_bytes,  "image/png"),
        },
        data=data,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"[Step 1] 인페인팅 실패: {resp.text}")
    print("  완료.")
    return resp.content