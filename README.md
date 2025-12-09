# 장기 AI 엔진 (Janggi AI Engine)

Stockfish NNUE 기반의 한국 장기 AI 엔진입니다.

## 기능

- 완전한 장기 규칙 구현 (한/초, 모든 말의 이동 규칙)
- NNUE (Efficiently Updatable Neural Networks) 기반 평가 함수
- 미니맥스 알고리즘과 알파-베타 가지치기
- FastAPI 백엔드
- HTML/JavaScript 프론트엔드

## 설치

uv를 사용하여 의존성을 설치합니다:

```bash
uv sync
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

### 빠른 시작

```bash
# PyTorch 설치
pip install torch

# GPU 학습 (1-2분)
python scripts/train_nnue_gpu.py --positions 5000 --epochs 50 --skip-eval

# 권장 학습 (3-5분)
python scripts/train_nnue_gpu.py --parallel --positions 10000 --epochs 100
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
  ├── train_nnue.py         # CPU 학습 스크립트
  ├── train_nnue_gpu.py     # GPU 학습 스크립트
  ├── train_nnue_gibo.py    # 기보 기반 학습 스크립트
  └── example_use_model.py  # 모델 사용 예제
models/               # 학습된 모델 파일들
static/
  └── index.html      # 프론트엔드
docs/
  ├── rule-kr.md      # 장기 규칙 (한국어)
  ├── rule-en.md      # 장기 규칙 (영어)
  ├── training-guide.md  # 학습 가이드
  └── how-to-use-models.md  # 모델 사용 가이드
```

## 규칙

이 프로젝트는 `docs/rule-en.md`에 정의된 장기 규칙을 따릅니다.

## 라이선스

MIT License

