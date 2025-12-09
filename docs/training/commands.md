# GPU 학습 명령어 모음

## 🆕 스마트 학습 (가장 쉬운 방법, 권장)

컴퓨터 환경을 자동으로 분석하고 최적의 설정으로 학습합니다. GPU 배치 평가 최적화가 자동으로 적용됩니다.

### 대화형 모드 (추천)

```powershell
uv run python scripts/smart_train.py
```

### 직접 시간 지정

```powershell
# 표준 학습 (~15분)
uv run python scripts/smart_train.py --time standard

# 깊은 학습 (~30분, 권장)
uv run python scripts/smart_train.py --time deep

# 완전 학습 (~3시간, 강화된 설정)
uv run python scripts/smart_train.py --time full

# 극한 학습 (~4시간, 최강 성능)
uv run python scripts/smart_train.py --time extreme

# 마라톤 학습 (~8시간, 최종 보스)
uv run python scripts/smart_train.py --time marathon
```

### 기존 모델에서 계속 학습

```powershell
uv run python scripts/smart_train.py --time deep --load models/nnue_smart_model.json
```

### GPU 최적화 기능

스마트 학습은 다음 GPU 최적화 기능을 자동으로 사용합니다:
- ✅ GPU 메모리 기반 배치 크기 자동 계산
- ✅ 병렬 self-play + GPU 배치 평가
- ✅ 중앙 집중식 GPU 평가로 GPU 활용도 향상

---

## 수동 GPU 학습 (고급 사용자용)

## Windows PowerShell 사용법

PowerShell에서는 백틱(`)을 사용하여 여러 줄로 명령어를 작성할 수 있습니다.

### 기본 반복 학습 (한 줄)

```powershell
uv run python scripts/train_nnue_gpu.py --method iterative --load models/nnue_smart_model.json --iterations 5 --games-per-iter 100 --epochs 20 --batch-size 512 --eval-batch-size 512
```

### 개선된 반복 학습 (여러 줄 - 권장)

```powershell
uv run python scripts/train_nnue_gpu.py `
  --method iterative `
  --load models/nnue_smart_model.json `
  --iterations 5 `
  --games-per-iter 100 `
  --epochs 20 `
  --batch-size 512 `
  --eval-batch-size 512 `
  --depth 3 `
  --output models/nnue_smart_model_improved.json
```

### 빠른 테스트 (작은 배치)

```powershell
uv run python scripts/train_nnue_gpu.py `
  --method iterative `
  --load models/nnue_smart_model.json `
  --iterations 2 `
  --games-per-iter 50 `
  --epochs 10 `
  --batch-size 256 `
  --eval-batch-size 256 `
  --depth 2 `
  --skip-eval
```

### 고성능 학습 (큰 배치, GPU 메모리 충분 시)

```powershell
uv run python scripts/train_nnue_gpu.py `
  --method iterative `
  --load models/nnue_smart_model.json `
  --iterations 10 `
  --games-per-iter 200 `
  --epochs 30 `
  --batch-size 1024 `
  --eval-batch-size 1024 `
  --depth 4 `
  --output models/nnue_smart_model_final.json
```

### GPU 메모리 부족 시 (작은 배치)

```powershell
uv run python scripts/train_nnue_gpu.py `
  --method iterative `
  --load models/nnue_smart_model.json `
  --iterations 5 `
  --games-per-iter 100 `
  --epochs 20 `
  --batch-size 128 `
  --eval-batch-size 128 `
  --depth 3
```

## 주요 파라미터 설명

- `--method iterative`: 반복 자기대국 학습 모드
- `--load`: 기존 모델 로드 경로
- `--iterations`: 반복 횟수 (기본값: 5)
- `--games-per-iter`: 각 반복당 생성할 게임 수 (기본값: 100)
- `--epochs`: 각 반복당 학습 에포크 수 (기본값: 20)
- `--batch-size`: 학습 배치 크기 (기본값: 256)
- `--eval-batch-size`: 평가 배치 크기 (None = GPU 메모리 기반 자동 계산, 권장)
- `--eval-workers`: 평가용 워커 수 (None = 자동)
- `--depth`: 탐색 깊이 (기본값: 2, 권장: 3-4)
- `--output`: 최종 모델 저장 경로
- `--skip-eval`: 최종 평가 스킵 (시간 절약)

## GPU 최적화 기능

### 자동 배치 크기 계산

`--eval-batch-size`를 지정하지 않으면 GPU 메모리에 따라 자동으로 최적 배치 크기를 계산합니다:
- 16GB+ GPU: 1024
- 8GB GPU: 768
- 4GB GPU: 512
- 그 외: 256

### 병렬 self-play + GPU 배치 평가

반복 학습 모드에서는 다음 최적화가 자동으로 적용됩니다:
- CPU 멀티코어: 여러 게임을 병렬로 생성
- GPU 배치 처리: 포지션들을 모아서 배치로 평가 (GPU 효율 향상)
- 중앙 집중식 평가: 워커에서 모델을 로드하지 않고 메인 프로세스에서 배치 평가

## 주의사항

1. **GPU 메모리**: `batch-size`와 `eval-batch-size`는 GPU 메모리에 따라 조정하세요.
   - 8GB GPU: 256-512 권장
   - 16GB+ GPU: 512-1024 가능

2. **탐색 깊이**: `--depth`를 높이면 더 좋은 데이터를 생성하지만 시간이 오래 걸립니다.
   - depth 2: 빠름, 기본 품질
   - depth 3: 균형잡힌 선택 (권장)
   - depth 4: 느리지만 고품질

3. **반복 횟수**: `--iterations`를 늘리면 더 강한 모델을 만들 수 있지만 시간이 오래 걸립니다.

4. **평가 스킵**: 빠른 테스트를 원하면 `--skip-eval`을 추가하세요.

## 실행 예시

### PowerShell에서 직접 실행

```powershell
# 현재 디렉토리에서 실행
uv run python scripts/train_nnue_gpu.py --method iterative --load models/nnue_smart_model.json --iterations 5 --games-per-iter 100 --epochs 20 --batch-size 512 --eval-batch-size 512
```

### 스크립트 파일로 실행

```powershell
# train_gpu_iterative.ps1 파일 실행
.\train_gpu_iterative.ps1
```

### 배치 파일로 실행 (cmd.exe 호환)

```cmd
@echo off
uv run python scripts/train_nnue_gpu.py --method iterative --load models/nnue_smart_model.json --iterations 5 --games-per-iter 100 --epochs 20 --batch-size 512 --eval-batch-size 512
```

