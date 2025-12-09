# NNUE 모델 학습 가이드

이 문서는 장기 AI의 NNUE (Efficiently Updatable Neural Networks) 평가 함수를 학습시키는 방법을 설명합니다.

## 목차

1. [개요](#개요)
2. [환경 설정](#환경-설정)
3. [빠른 시작](#빠른-시작)
4. [학습 방법](#학습-방법)
5. [최적화 옵션](#최적화-옵션)
6. [문제 해결](#문제-해결)
7. [고급 사용법](#고급-사용법)

---

## 개요

### NNUE란?

NNUE는 Stockfish 체스 엔진에서 사용되는 신경망 기반 평가 함수입니다. 이 프로젝트에서는 장기에 맞게 수정된 NNUE 아키텍처를 사용합니다.

### 아키텍처

```
입력 (512개 특징)
    ↓
Hidden Layer 1 (256 뉴런, Clipped ReLU)
    ↓
Hidden Layer 2 (64 뉴런, Clipped ReLU)
    ↓
출력 (1개, 평가값)
```

### 특징 (Features)

모델이 학습하는 특징들:

- **기물 점수**: 각 기물의 개수와 가치
- **위치 특성**: 기물의 중앙 배치, 진출 정도
- **왕 안전도**: 궁성 내 왕과 사의 위치
- **기동력**: 각 진영이 움직일 수 있는 예상 수
- **졸 진출**: 졸의 전진 정도
- **포 화력**: 포가 넘을 수 있는 기물 존재 여부

---

## 환경 설정

### 1. 기본 의존성 설치

```bash
uv sync
```

### 2. GPU 학습을 위한 PyTorch 설치

```bash
# NVIDIA GPU (CUDA)
pip install torch

# Apple Silicon (M1/M2/M3)
pip install torch

# CPU만 사용
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. 확인

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

---

## 빠른 시작

### 가장 간단한 학습 (1-2분)

```bash
python scripts/train_nnue_gpu.py --positions 5000 --epochs 50 --skip-eval
```

### 권장 학습 (3-5분)

```bash
python scripts/train_nnue_gpu.py --parallel --positions 10000 --epochs 100
```

### 학습된 모델 확인

```bash
ls -la models/nnue_gpu_model.json
```

---

## 학습 방법

### 1. GPU 학습 (권장)

GPU를 사용한 빠른 학습입니다.

```bash
python scripts/train_nnue_gpu.py [옵션]
```

#### 주요 옵션

| 옵션           | 기본값                        | 설명                               |
| -------------- | ----------------------------- | ---------------------------------- |
| `--positions`  | 10000                         | 생성할 학습 포지션 수              |
| `--epochs`     | 50                            | 학습 에폭 수                       |
| `--batch-size` | 256                           | 배치 크기 (GPU 메모리에 따라 조절) |
| `--lr`         | 0.0005                        | 학습률                             |
| `--parallel`   | -                             | 멀티프로세싱 데이터 생성           |
| `--skip-eval`  | -                             | 최종 평가 건너뛰기 (빠름)          |
| `--output`     | models/nnue_gpu_model.json    | 출력 파일명                        |

#### 예시

```bash
# 빠른 테스트
python scripts/train_nnue_gpu.py --positions 5000 --epochs 30 --skip-eval

# 표준 학습
python scripts/train_nnue_gpu.py --parallel --positions 20000 --epochs 100

# 대용량 학습
python scripts/train_nnue_gpu.py --parallel --positions 100000 --epochs 200 --batch-size 512
```

### 2. CPU 학습

GPU가 없는 환경에서 사용합니다.

```bash
python scripts/train_nnue.py [옵션]
```

#### 주요 옵션

| 옵션          | 기본값                    | 설명                                        |
| ------------- | ------------------------- | ------------------------------------------- |
| `--method`    | deepsearch                | 학습 방법 (selfplay, deepsearch, iterative) |
| `--games`     | 100                       | 자기대전 게임 수 (selfplay)                 |
| `--positions` | 5000                      | 포지션 수 (deepsearch)                      |
| `--epochs`    | 30                        | 학습 에폭 수                                |

#### 예시

```bash
# 자기대전 학습
python scripts/train_nnue.py --method selfplay --games 100 --epochs 30

# 깊은 탐색 학습 (권장)
python scripts/train_nnue.py --method deepsearch --positions 5000 --epochs 50

# 반복 자기 개선
python scripts/train_nnue.py --method iterative --iterations 5
```

### 3. 반복 학습 (Iterative Training)

모델이 자기 자신과 대전하면서 점진적으로 개선됩니다.

```bash
python scripts/train_nnue_gpu.py --method iterative --iterations 10 --games-per-iter 100
```

---

## 최적화 옵션

### 데이터 생성 모드

| 모드        | 속도      | 품질 | 명령어       |
| ----------- | --------- | ---- | ------------ |
| Fast (기본) | 빠름      | 보통 | `--fast`     |
| Parallel    | 매우 빠름 | 보통 | `--parallel` |
| Quality     | 느림      | 높음 | `--no-fast`  |

### 멀티프로세싱

CPU 코어를 활용해 데이터 생성을 병렬화합니다.

```bash
# 자동 (CPU 코어 수 - 1)
python scripts/train_nnue_gpu.py --parallel --positions 50000

# 워커 수 지정
python scripts/train_nnue_gpu.py --parallel --workers 4 --positions 50000
```

### 학습률 조절

NaN 문제가 발생하면 학습률을 낮추세요.

```bash
# 안정적인 학습
python scripts/train_nnue_gpu.py --lr 0.0003

# 매우 안정적
python scripts/train_nnue_gpu.py --lr 0.0001
```

---

## 문제 해결

### NaN Loss 발생

**증상**: `Train Loss: nan, Val Loss: nan`

**원인**:

- 학습률이 너무 높음
- Gradient explosion

**해결**:

```bash
# 학습률 낮추기
python scripts/train_nnue_gpu.py --lr 0.0001 --positions 10000

# 또는 배치 크기 줄이기
python scripts/train_nnue_gpu.py --batch-size 128 --lr 0.0003
```

### 학습이 너무 느림

**원인**: 데이터 생성이 병목

**해결**:

```bash
# 병렬 데이터 생성 사용
python scripts/train_nnue_gpu.py --parallel --positions 10000

# 또는 평가 건너뛰기
python scripts/train_nnue_gpu.py --skip-eval
```

### 메모리 부족

**원인**: 배치 크기가 너무 큼

**해결**:

```bash
# 배치 크기 줄이기
python scripts/train_nnue_gpu.py --batch-size 64
```

### PyTorch 설치 오류

```bash
# PyTorch 재설치
pip uninstall torch
pip install torch
```

---

## 고급 사용법

### 기존 모델 이어서 학습

```bash
python scripts/train_nnue_gpu.py --load models/nnue_gpu_model.json --positions 20000 --output models/nnue_v2.json
```

### PyTorch 형식으로 저장

더 효율적인 저장/로드를 위해 .pt 형식을 사용할 수 있습니다.

```bash
python scripts/train_nnue_gpu.py --output models/model.json --output-torch models/model.pt
```

### 특정 디바이스 지정

```bash
# CUDA GPU
python scripts/train_nnue_gpu.py --device cuda

# Apple Silicon
python scripts/train_nnue_gpu.py --device mps

# CPU
python scripts/train_nnue_gpu.py --device cpu
```

### 네트워크 구조 변경

```bash
# 더 큰 네트워크
python scripts/train_nnue_gpu.py --feature-size 1024 --hidden1 512 --hidden2 128

# 더 작은 네트워크 (빠른 추론)
python scripts/train_nnue_gpu.py --feature-size 256 --hidden1 128 --hidden2 32
```

---

## 학습된 모델 사용

### 서버에서 자동 사용

서버 시작 시 `models/nnue_gpu_model.json`이 있으면 자동으로 사용됩니다.

```bash
uv run uvicorn api:app --reload
```

### 환경 변수로 모델 지정

```bash
NNUE_MODEL_PATH=models/my_model.json uv run uvicorn api:app --reload
```

### API로 모델 정보 확인

```bash
curl http://localhost:8000/api/model-info
```

---

## 권장 학습 전략

### 1단계: 빠른 테스트

```bash
python scripts/train_nnue_gpu.py --positions 5000 --epochs 30 --skip-eval
```

### 2단계: 기본 학습

```bash
python scripts/train_nnue_gpu.py --parallel --positions 30000 --epochs 100
```

### 3단계: 반복 개선

```bash
python scripts/train_nnue_gpu.py --method iterative --iterations 5 --load models/nnue_gpu_model.json
```

---

## 성능 비교

### 학습 시간 (M2 MacBook Air 기준)

| 설정                                 | 시간  |
| ------------------------------------ | ----- |
| 5K positions, 50 epochs              | ~1분  |
| 10K positions, 100 epochs            | ~3분  |
| 50K positions, 100 epochs (parallel) | ~10분 |

### 모델 강도

학습량이 많을수록 강해지지만, 수확 체감이 있습니다.

| 포지션 수 | 예상 강도 |
| --------- | --------- |
| 5,000     | 기본      |
| 20,000    | 중급      |
| 100,000+  | 상급      |

---

## 참고

- [Stockfish NNUE](https://www.chessprogramming.org/Stockfish_NNUE)
- [PyTorch 문서](https://pytorch.org/docs/)
