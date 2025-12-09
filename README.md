# 장기 AI 엔진 (Janggi AI Engine)

Stockfish NNUE 기반의 한국 장기 AI 엔진입니다.

## 기능

- 완전한 장기 규칙 구현 (한/초, 모든 말의 이동 규칙)
- NNUE (Efficiently Updatable Neural Networks) 기반 평가 함수
- 미니맥스 알고리즘과 알파-베타 가지치기
- FastAPI 백엔드
- HTML/JavaScript 프론트엔드

## 요구사항

- Python 3.10 ~ 3.12 (PyTorch CUDA 지원을 위해 3.12 권장)
- uv (Python 패키지 관리자)

## 설치

### 기본 설치

uv를 사용하여 의존성을 설치합니다:

```bash
uv sync
```

### GPU 학습을 위한 PyTorch 설치 (선택사항)

GPU를 사용한 학습을 원하는 경우 CUDA 지원 버전의 PyTorch를 설치하세요:

```bash
# NVIDIA GPU (CUDA 12.1)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 또는 GPU 의존성 포함 설치
uv sync --extra gpu
# 그 후 CUDA 버전으로 교체:
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

GPU 설치 확인:

```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 실행

서버를 시작합니다:

```bash
uv run python main.py
```

또는:

```bash
uv run uvicorn api:app --reload
```

브라우저에서 `http://localhost:8000`을 열어 게임을 시작하세요.

## API 엔드포인트

### `POST /api/new-game`
새 게임을 생성합니다.

```json
{
  "game_id": "default",
  "depth": 3,
  "use_nnue": true
}
```

### `GET /api/board/{game_id}`
현재 보드 상태를 가져옵니다.

### `POST /api/move`
이동을 수행합니다.

```json
{
  "game_id": "default",
  "from_square": "a1",
  "to_square": "b2"
}
```

### `POST /api/ai-move/{game_id}`
AI의 이동을 생성합니다.

## NNUE 모델 학습

AI를 더 강하게 만들기 위해 NNUE 모델을 학습시킬 수 있습니다.

### 🆕 스마트 학습 (가장 쉬운 방법, 권장)

컴퓨터 환경을 자동으로 분석하고 최적의 설정으로 학습합니다:

```bash
# 대화형 모드 (추천)
uv run python scripts/smart_train.py

# 또는 직접 시간 지정
uv run python scripts/smart_train.py --time deep  # ~30분
uv run python scripts/smart_train.py --time standard  # ~15분
```

스마트 학습은 다음을 자동으로 처리합니다:
- ✅ GPU 자동 감지 및 사용
- ✅ 시스템 사양에 맞는 최적 설정
- ✅ 기보 파일 자동 활용
- ✅ 학습 시간에 따른 설정 조정

### 수동 학습 방법

#### GPU 학습 (빠름)

```bash
# 빠른 테스트 (1-2분)
uv run python scripts/train_nnue_gpu.py --positions 5000 --epochs 50 --skip-eval

# 권장 학습 (3-5분)
uv run python scripts/train_nnue_gpu.py --parallel --positions 10000 --epochs 100
```

#### 기보 기반 학습 (실전적)

실제 대국 기보 파일(.gib)을 사용하여 더 현실적인 AI를 만들 수 있습니다.

```bash
# 기보 디렉토리에 .gib 파일을 넣고:
uv run python scripts/train_nnue_gibo.py --gibo-dir gibo --epochs 30

# 기존 모델 fine-tuning
uv run python scripts/train_nnue_gibo.py --gibo-dir gibo --load models/nnue_gpu_model.json --epochs 20
```

자세한 내용은 [학습 가이드](docs/training-guide.md)를 참조하세요.

## 프로젝트 구조

```
janggi/
  ├── __init__.py
  ├── board.py        # 보드 표현 및 이동 생성
  ├── nnue.py         # NNUE 평가 함수 (NumPy)
  ├── nnue_torch.py   # NNUE 평가 함수 (PyTorch/GPU)
  └── engine.py       # 미니맥스 AI 엔진

api.py                # FastAPI 백엔드
main.py               # 서버 진입점

scripts/
  ├── smart_train.py        # 🆕 스마트 학습 (자동 환경 감지)
  ├── train_nnue.py         # CPU 학습 스크립트
  ├── train_nnue_gpu.py     # GPU 학습 스크립트
  ├── train_nnue_gibo.py    # 기보 기반 학습 스크립트
  └── example_use_model.py  # 모델 사용 예제

models/               # 학습된 모델 파일들
  ├── nnue_smart_model.json   # 🆕 스마트 학습 모델
  ├── nnue_gpu_model.json     # GPU 기본 모델
  ├── nnue_gibo_model.json    # 기보 학습 모델
  └── nnue_gpu_iter_*.json    # 반복 학습 모델들

gibo/                 # 기보 파일 디렉토리 (.gib)

static/
  └── index.html      # 프론트엔드

docs/
  ├── rule-kr.md           # 장기 규칙 (한국어)
  ├── rule-en.md           # 장기 규칙 (영어)
  ├── training-guide.md    # 학습 가이드
  └── how-to-use-models.md # 모델 사용 가이드
```

## 규칙

이 프로젝트는 `docs/rule-en.md`에 정의된 장기 규칙을 따릅니다.

## 라이선스

MIT License

