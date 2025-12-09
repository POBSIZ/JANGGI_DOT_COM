# NNUE 모델 학습 가이드

이 문서는 장기 AI의 NNUE (Efficiently Updatable Neural Networks) 평가 함수를 학습시키는 방법을 설명합니다.

## 목차

1. [개요](#개요)
2. [환경 설정](#환경-설정)
3. [빠른 시작](#빠른-시작)
4. [🆕 스마트 학습 (권장)](#스마트-학습-권장)
5. [학습 방법](#학습-방법)
   - [GPU 학습](#1-gpu-학습-권장)
   - [CPU 학습](#2-cpu-학습)
   - [기보 학습](#3-기보-학습-신규)
   - [반복 학습](#4-반복-학습-iterative-training)
6. [최적화 옵션](#최적화-옵션)
7. [문제 해결](#문제-해결)
8. [고급 사용법](#고급-사용법)

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

### 1. Python 버전 확인

이 프로젝트는 **Python 3.10 ~ 3.12**를 지원합니다. PyTorch CUDA 버전은 Windows에서 Python 3.12까지만 지원하므로, GPU 학습을 원하는 경우 Python 3.12를 권장합니다.

```bash
# Python 버전 확인
uv run python --version

# Python 3.12로 고정 (권장)
uv python pin 3.12
```

### 2. 기본 의존성 설치

```bash
uv sync
```

### 3. GPU 학습을 위한 PyTorch 설치

#### NVIDIA GPU (CUDA) - Windows/Linux

```bash
# CUDA 12.1 지원 버전 설치 (권장)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 또는 CUDA 11.8 지원 버전
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**중요**: 기본 `pip install torch`는 CPU-only 버전을 설치할 수 있습니다. GPU를 사용하려면 위의 CUDA 인덱스 URL을 사용하세요.

#### Apple Silicon (M1/M2/M3)

```bash
# MPS (Metal Performance Shaders) 지원 버전
uv pip install torch torchvision torchaudio
```

#### CPU만 사용

```bash
# CPU-only 버전
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. GPU 설치 확인

```bash
# GPU 감지 확인
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# 또는 스마트 학습 스크립트로 확인
uv run python scripts/smart_train.py --info-only
```

**예상 출력 (GPU 사용 가능 시)**:
```
PyTorch: 2.5.1+cu121
CUDA available: True
Device: NVIDIA GeForce RTX 3060
```

**예상 출력 (GPU 사용 불가 시)**:
```
PyTorch: 2.9.1+cpu
CUDA available: False
Device: CPU
```

### 5. GPU 감지 문제 해결

GPU가 감지되지 않는 경우:

1. **PyTorch CPU-only 버전이 설치된 경우**:
   ```bash
   uv pip uninstall torch torchvision torchaudio
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **CUDA 드라이버 확인**:
   - NVIDIA GPU: [NVIDIA 드라이버](https://www.nvidia.com/Download/index.aspx) 설치 확인
   - `nvidia-smi` 명령어로 GPU 인식 확인

3. **Python 버전 확인**:
   - Windows에서 CUDA 지원은 Python 3.12까지만 가능
   - `uv python pin 3.12`로 버전 고정

4. **스마트 학습 스크립트로 자동 진단**:
   ```bash
   uv run python scripts/smart_train.py --info-only
   ```
   이 명령어는 GPU 감지 문제의 원인을 자동으로 진단하고 해결 방법을 제시합니다.

---

## 빠른 시작

### 가장 간단한 학습 (1-2분)

```bash
uv run python scripts/train_nnue_gpu.py --positions 5000 --epochs 50 --skip-eval
```

### 권장 학습 (3-5분)

```bash
uv run python scripts/train_nnue_gpu.py --parallel --positions 10000 --epochs 100
```

### 학습된 모델 확인

```bash
ls -la models/nnue_gpu_model.json
```

---

## 스마트 학습 (권장)

🆕 **가장 쉬운 학습 방법!** 컴퓨터 환경을 자동으로 분석하고 최적의 설정을 찾아줍니다.

### 대화형 모드 (추천)

```bash
uv run python scripts/smart_train.py
```

실행하면 다음 순서로 진행됩니다:
1. 🔍 시스템 환경 자동 분석 (CPU, RAM, GPU, 기보 파일)
2. 🎯 학습 시간 선택 메뉴
3. 📊 최적 설정 자동 계산
4. 🎓 학습 시작

### 학습 시간 옵션

| 옵션       | 예상 시간 | 설명                  | 명령어 예시                        |
| ---------- | --------- | --------------------- | ---------------------------------- |
| ⚡ quick   | ~5분      | 빠른 테스트용         | `--time quick`                     |
| 📘 standard| ~15분     | 일반적인 사용         | `--time standard`                  |
| 📗 deep    | ~30분     | 권장, 좋은 성능       | `--time deep`                      |
| 📕 intensive| ~1시간   | 높은 성능 목표        | `--time intensive`                 |
| 🏆 full    | ~2시간+   | 최고 성능             | `--time full`                      |

### 명령줄 사용 예시

```bash
# 표준 학습 (약 15분)
uv run python scripts/smart_train.py --time standard

# 깊은 학습 (약 30분)
uv run python scripts/smart_train.py --time deep

# 기보 파일 없이 학습
uv run python scripts/smart_train.py --time standard --no-gibo

# 기존 모델에서 계속 학습
uv run python scripts/smart_train.py --time deep --load models/nnue_model.json

# 시스템 정보만 확인 (GPU 감지 진단)
uv run python scripts/smart_train.py --info-only
```

### 자동 최적화 기능

스마트 학습은 시스템에 따라 자동으로 설정을 조정합니다:

- **GPU 감지**: CUDA 또는 Apple Silicon (MPS) GPU가 있으면 자동으로 GPU 학습
  - GPU가 감지되지 않으면 자동으로 원인 진단 및 해결 방법 제시
- **메모리 최적화**: RAM/VRAM 크기에 따라 배치 크기 자동 조절
- **병렬 처리**: CPU 코어 수에 따라 데이터 생성 병렬화
- **기보 활용**: 기보 파일이 있으면 자동으로 기보 기반 학습 포함

### 출력 예시

```
============================================================
🖥️  시스템 환경 분석
============================================================

📌 운영체제: Darwin 24.6.0
📌 CPU: arm
   - 코어: 8개 / 스레드: 8개
📌 RAM: 16.0 GB
📌 GPU: Apple Silicon (MPS)
   - VRAM: 8.0 GB
   ✅ GPU 가속 사용 가능
📌 기보 파일: 59개 발견
   ✅ 기보 기반 학습 가능

============================================================
⚙️  학습 설정
============================================================

📋 학습 방식: GPU 가속 학습
📋 학습 포지션 수: 15,000개
📋 에포크 수: 50회
📋 배치 사이즈: 512
📋 학습률: 0.0005

⏱️  예상 학습 시간: 약 15분
```

---

## 학습 방법

### 1. GPU 학습 (권장)

GPU를 사용한 빠른 학습입니다.

```bash
uv run python scripts/train_nnue_gpu.py [옵션]
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
uv run python scripts/train_nnue_gpu.py --positions 5000 --epochs 30 --skip-eval

# 표준 학습
uv run python scripts/train_nnue_gpu.py --parallel --positions 20000 --epochs 100

# 대용량 학습
uv run python scripts/train_nnue_gpu.py --parallel --positions 100000 --epochs 200 --batch-size 512
```

### 2. CPU 학습

GPU가 없는 환경에서 사용합니다.

```bash
uv run python scripts/train_nnue.py [옵션]
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
uv run python scripts/train_nnue.py --method selfplay --games 100 --epochs 30

# 깊은 탐색 학습 (권장)
uv run python scripts/train_nnue.py --method deepsearch --positions 5000 --epochs 50

# 반복 자기 개선
uv run python scripts/train_nnue.py --method iterative --iterations 5
```

### 3. 기보 학습 (신규)

실제 대국 기보 파일(.gib)을 사용하여 학습합니다. 고수들의 실전 데이터로 학습하므로 더 현실적인 평가 함수를 만들 수 있습니다.

```bash
uv run python scripts/train_nnue_gibo.py [옵션]
```

#### 주요 옵션

| 옵션                   | 기본값                 | 설명                          |
| ---------------------- | ---------------------- | ----------------------------- |
| `--gibo-dir`           | gibo                   | 기보 파일이 있는 디렉토리     |
| `--epochs`             | 50                     | 학습 에폭 수                  |
| `--batch-size`         | 256                    | 배치 크기                     |
| `--lr`                 | 0.001                  | 학습률                        |
| `--positions-per-game` | 50                     | 게임당 추출할 최대 포지션 수  |
| `--load`               | -                      | 기존 모델 로드 (fine-tuning)  |
| `--output`             | nnue_gibo_model.json   | 출력 파일명                   |

#### 예시

```bash
# 기본 기보 학습
uv run python scripts/train_nnue_gibo.py --gibo-dir gibo --epochs 30

# 낮은 학습률로 안정적 학습
uv run python scripts/train_nnue_gibo.py --gibo-dir gibo --lr 0.0001 --epochs 50

# 기존 모델에서 fine-tuning
uv run python scripts/train_nnue_gibo.py --gibo-dir gibo --load models/nnue_gpu_model.json --epochs 20

# 상세 설정
uv run python scripts/train_nnue_gibo.py \
    --gibo-dir gibo \
    --epochs 50 \
    --lr 0.0001 \
    --batch-size 128 \
    --positions-per-game 30 \
    --output models/my_gibo_model.json
```

#### 지원 기보 형식

- 카카오 장기 기보 (.gib)
- EUC-KR/CP949 인코딩 자동 인식
- 차림 정보 (상마상마, 마상마상 등) 파싱
- 대국 결과 파싱 (초 승/한 승/무승부)

#### 기보 학습의 장점

- **실전 데이터**: 고수들의 실제 대국에서 학습
- **효율적**: 자기대전보다 빠르게 다양한 상황 학습
- **현실적 평가**: 실제 대국에서 나타나는 패턴 학습

### 4. 반복 학습 (Iterative Training)

모델이 자기 자신과 대전하면서 점진적으로 개선됩니다.

```bash
uv run python scripts/train_nnue_gpu.py --method iterative --iterations 10 --games-per-iter 100
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
uv run python scripts/train_nnue_gpu.py --parallel --positions 50000

# 워커 수 지정
uv run python scripts/train_nnue_gpu.py --parallel --workers 4 --positions 50000
```

### 학습률 조절

NaN 문제가 발생하면 학습률을 낮추세요.

```bash
# 안정적인 학습
uv run python scripts/train_nnue_gpu.py --lr 0.0003

# 매우 안정적
uv run python scripts/train_nnue_gpu.py --lr 0.0001
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
uv run python scripts/train_nnue_gpu.py --lr 0.0001 --positions 10000

# 또는 배치 크기 줄이기
uv run python scripts/train_nnue_gpu.py --batch-size 128 --lr 0.0003
```

### 학습이 너무 느림

**원인**: 데이터 생성이 병목

**해결**:

```bash
# 병렬 데이터 생성 사용
uv run python scripts/train_nnue_gpu.py --parallel --positions 10000

# 또는 평가 건너뛰기
uv run python scripts/train_nnue_gpu.py --skip-eval
```

### 메모리 부족

**원인**: 배치 크기가 너무 큼

**해결**:

```bash
# 배치 크기 줄이기
uv run python scripts/train_nnue_gpu.py --batch-size 64
```

### PyTorch 설치 오류

```bash
# PyTorch 재설치 (CUDA 버전)
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### GPU가 감지되지 않는 경우

**증상**: `GPU: 사용 불가 (CPU 학습 모드)` 메시지가 표시됨

**원인 및 해결**:

1. **PyTorch CPU-only 버전 설치됨**:
   ```bash
   # 현재 버전 확인
   uv run python -c "import torch; print(torch.__version__)"
   # 출력에 "+cpu"가 포함되어 있으면 CPU-only 버전
   
   # CUDA 버전으로 교체
   uv pip uninstall torch torchvision torchaudio
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Python 버전 문제 (Windows)**:
   ```bash
   # Python 3.12로 고정 (CUDA 지원)
   uv python pin 3.12
   uv sync --extra gpu
   ```

3. **CUDA 드라이버 미설치**:
   - [NVIDIA 드라이버 다운로드](https://www.nvidia.com/Download/index.aspx)
   - `nvidia-smi` 명령어로 확인

4. **자동 진단**:
   ```bash
   uv run python scripts/smart_train.py --info-only
   ```
   이 명령어는 문제 원인을 자동으로 진단하고 해결 방법을 제시합니다.

### 기보 파싱 오류

**증상**: `0 games parsed` 또는 인코딩 오류

**원인**:
- 기보 파일 인코딩 문제
- 지원하지 않는 기보 형식

**해결**:
- 기보 파일이 EUC-KR 또는 CP949 인코딩인지 확인
- 카카오 장기 기보 형식(.gib) 사용

### 기보 학습 시 Loss 불안정

**증상**: Loss 값이 폭발하거나 NaN 발생

**원인**:
- 학습률이 너무 높음
- Gradient explosion

**해결**:
```bash
# 낮은 학습률 사용 (권장)
uv run python scripts/train_nnue_gibo.py --lr 0.0001 --batch-size 128

# 또는 더 작은 배치 크기
uv run python scripts/train_nnue_gibo.py --lr 0.00005 --batch-size 64
```

---

## 고급 사용법

### 기존 모델 이어서 학습

```bash
uv run python scripts/train_nnue_gpu.py --load models/nnue_gpu_model.json --positions 20000 --output models/nnue_v2.json
```

### PyTorch 형식으로 저장

더 효율적인 저장/로드를 위해 .pt 형식을 사용할 수 있습니다.

```bash
uv run python scripts/train_nnue_gpu.py --output models/model.json --output-torch models/model.pt
```

### 특정 디바이스 지정

```bash
# CUDA GPU
uv run python scripts/train_nnue_gpu.py --device cuda

# Apple Silicon
uv run python scripts/train_nnue_gpu.py --device mps

# CPU
uv run python scripts/train_nnue_gpu.py --device cpu
```

### 네트워크 구조 변경

```bash
# 더 큰 네트워크
uv run python scripts/train_nnue_gpu.py --feature-size 1024 --hidden1 512 --hidden2 128

# 더 작은 네트워크 (빠른 추론)
uv run python scripts/train_nnue_gpu.py --feature-size 256 --hidden1 128 --hidden2 32
```

### 기보 학습 고급 설정

```bash
# 게임당 더 많은 포지션 추출
uv run python scripts/train_nnue_gibo.py --positions-per-game 100 --gibo-dir gibo

# 특정 디바이스 사용
uv run python scripts/train_nnue_gibo.py --device cuda --gibo-dir gibo

# 여러 기보 디렉토리 사용 (여러 번 실행)
uv run python scripts/train_nnue_gibo.py --gibo-dir gibo1 --output models/model_v1.json
uv run python scripts/train_nnue_gibo.py --gibo-dir gibo2 --load models/model_v1.json --output models/model_v2.json
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

### 방법 0: 스마트 학습 (가장 쉬움) 🌟

복잡한 설정 없이 한 줄로 최적의 학습을 시작합니다.

```bash
# 대화형 모드 - 메뉴에서 선택
uv run python scripts/smart_train.py

# 또는 직접 시간 지정
uv run python scripts/smart_train.py --time deep
```

### 방법 A: 자기대전 학습 (기보 없이)

#### 1단계: 빠른 테스트

```bash
uv run python scripts/train_nnue_gpu.py --positions 5000 --epochs 30 --skip-eval
```

#### 2단계: 기본 학습

```bash
uv run python scripts/train_nnue_gpu.py --parallel --positions 30000 --epochs 100
```

#### 3단계: 반복 개선

```bash
python scripts/train_nnue_gpu.py --method iterative --iterations 5 --load models/nnue_gpu_model.json
```

### 방법 B: 기보 기반 학습 (권장)

기보 파일이 있다면 이 방법이 더 효과적입니다.

#### 1단계: 기보 학습

```bash
uv run python scripts/train_nnue_gibo.py --gibo-dir gibo --epochs 30 --lr 0.0001
```

#### 2단계: Fine-tuning

```bash
uv run python scripts/train_nnue_gibo.py --gibo-dir gibo --load models/nnue_gibo_model.json --epochs 20 --lr 0.00005
```

#### 3단계: 반복 개선 (선택)

```bash
python scripts/train_nnue_gpu.py --method iterative --iterations 3 --load models/nnue_gibo_model.json
```

### 방법 C: 하이브리드 학습 (최강)

자기대전과 기보 학습을 결합합니다.

```bash
# 1. 기보로 기본 학습
uv run python scripts/train_nnue_gibo.py --gibo-dir gibo --epochs 30 --output models/base_model.json

# 2. 자기대전으로 보강
uv run python scripts/train_nnue_gpu.py --parallel --positions 50000 --load models/base_model.json --output models/hybrid_model.json

# 3. 반복 개선
uv run python scripts/train_nnue_gpu.py --method iterative --iterations 5 --load models/hybrid_model.json
```

---

## 성능 비교

### 학습 시간 (M2 MacBook Air 기준)

#### 자기대전 학습

| 설정                                 | 시간  |
| ------------------------------------ | ----- |
| 5K positions, 50 epochs              | ~1분  |
| 10K positions, 100 epochs            | ~3분  |
| 50K positions, 100 epochs (parallel) | ~10분 |

#### 기보 학습

| 설정                                    | 시간  |
| --------------------------------------- | ----- |
| 1,000 games, 30 epochs                  | ~2분  |
| 1,000 games, 50 epochs, lr=0.0001       | ~3분  |

### 모델 강도

학습량이 많을수록 강해지지만, 수확 체감이 있습니다.

#### 자기대전 학습

| 포지션 수 | 예상 강도 |
| --------- | --------- |
| 5,000     | 기본      |
| 20,000    | 중급      |
| 100,000+  | 상급      |

#### 기보 학습

| 게임 수 | 예상 특징                      |
| ------- | ------------------------------ |
| 500+    | 기본 패턴 학습                 |
| 1,000+  | 다양한 전략 학습               |
| 5,000+  | 고급 전술, 실전적 평가         |

### 학습 방법별 특성

| 방법       | 장점                              | 단점                       |
| ---------- | --------------------------------- | -------------------------- |
| 자기대전   | 전술적 계산력 향상                | 시간이 오래 걸림           |
| 기보 학습  | 실전적 평가, 빠른 학습            | 기보 데이터 필요           |
| 하이브리드 | 양쪽 장점 결합                    | 설정이 복잡함              |

---

## 참고

- [Stockfish NNUE](https://www.chessprogramming.org/Stockfish_NNUE)
- [PyTorch 문서](https://pytorch.org/docs/)
