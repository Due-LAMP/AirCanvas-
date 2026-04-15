# AirCanvas

> Raspberry Pi 5 + MediaPipe 기반의 **공중 드로잉 인터랙티브 촬영 시스템**  
> 손가락으로 카메라 화면 위에 그림을 그리고, 촬영 후 이메일로 전송합니다.

---

## 프로젝트 소개

**AirCanvas**는 카메라 앞에서 손동작만으로 화면 위에 드로잉을 수행하는 실시간 인터랙티브 시스템입니다.  
터치 디바이스 없이도 손가락 궤적을 활용해 얼굴 위를 꾸미고, 결과물을 저장/전송할 수 있도록 설계되었습니다.

### 핵심 목표

- 실시간 손 추적 기반 드로잉 오버레이
- 펜/지우개 모드 전환
- 합성 이미지 저장 및 메일 전송
- 추후 AI 기반 Sketch-to-Image 확장 가능 구조 확보

---

## 기술 스택

### Hardware

- Raspberry Pi 5 Model B Rev 1.0 (권장, Pi 4 호환)
- Pi Camera Module 또는 USB 카메라
- 물리 버튼 3종 (촬영 / 지우개 토글 / 전송)
- HDMI 디스플레이
- 블루투스 키보드 (수신 이메일 입력)

### Software

- OS: Linux aarch64 (6.6.62+rpt-rpi-2712)
- Language: Python 3.11+
- Libraries:
  - MediaPipe
  - OpenCV
  - picamera2
  - NumPy

---

## 시스템 파이프라인

```text
Camera Input (picamera2)
        ->
Hand Landmark Detection (MediaPipe)
        ->
Finger State Classification (검지 펴짐 / 주먹)
        ->
Drawing Engine (펜/지우개)
        ->
Frame Composition (OpenCV)
        ->
Display + Button Events
        ->
Save (.jpg) / Send Email (SMTP)
```

---

## 주요 기능 (Phase 1)

### 1) 실시간 드로잉

- **그리기 ON**: 검지만 펴짐 (중지/약지/새끼 접힘)
- **그리기 OFF**: 주먹 (모든 손가락 접힘)
- **손 미검출**: 드로잉 중단, 기존 캔버스 유지

### 2) 지우개 모드

- 버튼 클릭 시 `PEN <-> ERASER` 토글
- 동일 손동작(검지/주먹)으로 지우개 ON/OFF 제어
- 상단 모드 표시 (`PEN`, `ERASER`)

### 3) 촬영/저장

- 촬영 버튼 입력 시 현재 화면(카메라 + 드로잉) 저장
- 저장 성공 시 `SAVED!` 약 0.6초 표시
- 저장 경로: `/home/pi/photos/{unix_timestamp}.jpg`
- 촬영 후 캔버스 유지

### 4) 이메일 전송

- 전송 버튼 누를 때마다 수신자 이메일 입력
- 최근 저장 이미지 1장 첨부
- SMTP(Gmail + App Password) 사용
- 예외 처리:
  - 사진 없음: `No photo yet`
  - 성공: `SENT!`
  - 실패: `SEND FAILED` + 터미널 로그

---

## 범위 정의

### In Scope (Phase 1)

- 손동작 기반 실시간 드로잉
- 펜/지우개 토글
- 합성 이미지 저장
- 이메일 전송 플로우

### Out of Scope (Phase 2)

- AI 채색/보정 (Stable Diffusion, ControlNet)
- 펜 색상/굵기 제스처 제어
- 멀티핸드 인식

---

## 제약 사항 및 가정

### 환경 가정

- 실내 조명 환경 기준
- 사용자-카메라 거리 약 50~100cm
- 단일 사용자(한 손) 인식
- 메일 전송 시 인터넷 연결 필요

### 성능 제약

- 해상도 `640x480` 고정
- `model_complexity=0` 고정
- 프레임 단위 궤적 단절 가능 (스무딩은 추후)

### 인식 한계

- 손가락 판별은 y축 비교 기반 (정면 카메라 기준)
- 측면 각도에서 오인식 가능
- 엄지 판별은 제외

### 보안

- Gmail App Password는 환경변수로 관리
- 코드 내 계정 정보 하드코딩 금지

---

## 개발 TODO

- [ ] 1. 환경 세팅 (Raspberry Pi OS, Python 3.11, 라이브러리 설치)
- [ ] 2. 카메라 + MediaPipe 기본 연동
- [ ] 3. 손가락 상태 판별 로직 구현
- [ ] 4. 드로잉 엔진 구현 (궤적 추적/합성)
- [ ] 5. 지우개 모드 구현
- [ ] 6. 촬영/저장 기능 구현
- [ ] 7. 이메일 전송 기능 구현
- [ ] 8. 통합 테스트 및 감도 튜닝
- [ ] 9. Phase 2 확장 (AI 보정, 제스처 기반 브러시 컨트롤)

---

## 빠른 시작 (예정)

아래는 구현 완료 후 기본 실행 흐름 예시입니다.

```bash
# 1) 가상환경 생성
python3 -m venv .venv
source .venv/bin/activate

# 2) 의존성 설치
pip install -r requirements.txt

# 3) 환경변수 설정 (예시)
export AIRCANVAS_SMTP_USER="your_account@gmail.com"
export AIRCANVAS_SMTP_APP_PASSWORD="your_app_password"

# 4) 실행 (파일명은 구현 시점 기준으로 변경 가능)
python main.py
```

---

## 향후 확장 아이디어

- Sketch-to-Image: 단순 스케치를 고품질 이미지로 변환
- 제스처 UI: 손가락 개수/핀치로 색상, 굵기 제어
- 멀티핸드 협업 드로잉

---

## 라이선스

프로젝트 초기 단계입니다. 라이선스는 팀 정책에 맞춰 추후 명시합니다.
