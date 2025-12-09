# 모델 사용 방법 가이드

`models/` 디렉토리에 있는 학습된 NNUE 모델을 실제로 사용하는 방법을 설명합니다.

## 1. API를 통한 사용 방법

### 방법 1: 새 게임 생성 시 모델 지정

API의 `/api/new-game` 엔드포인트를 호출할 때 `nnue_model_path` 파라미터로 모델 경로를 지정할 수 있습니다.

```python
import requests

# 새 게임 생성 시 특정 모델 사용
response = requests.post("http://localhost:8000/api/new-game", json={
    "game_id": "my_game_1",
    "depth": 3,
    "use_nnue": True,
    "nnue_model_path": "models/nnue_gpu_iter_5.json"  # 모델 경로 지정
})

print(response.json())
# {"status": "ok", "game_id": "my_game_1", "nnue_model": "models/nnue_gpu_iter_5.json"}
```

### 방법 2: 환경 변수로 기본 모델 설정

환경 변수 `NNUE_MODEL_PATH`를 설정하여 기본 모델을 지정할 수 있습니다.

```bash
# 터미널에서
export NNUE_MODEL_PATH=models/nnue_gpu_iter_5.json
python main.py
```

또는 Python 코드에서:

```python
import os
os.environ["NNUE_MODEL_PATH"] = "models/nnue_gpu_iter_5.json"
```

### 방법 3: 모델 자동 선택

`api.py`의 `_get_default_model_path()` 함수는 다음 우선순위로 모델을 선택합니다:

1. 환경 변수 `NNUE_MODEL_PATH` (가장 우선)
2. `models/nnue_gpu_model.json`
3. `models/nnue_model.json`

## 2. Python 코드에서 직접 사용

### Engine 클래스 사용

```python
from janggi.board import Board
from janggi.engine import Engine

# 모델을 사용하는 엔진 생성
engine = Engine(
    depth=3,
    use_nnue=True,
    nnue_model_path="models/nnue_gpu_iter_5.json"  # 모델 경로
)

# 보드 생성
board = Board()

# AI가 최선의 수 찾기
best_move = engine.search(board)
print(f"Best move: {best_move.to_uci()}")
print(f"Nodes searched: {engine.nodes_searched}")
```

### NNUE 클래스 직접 사용

```python
from janggi.board import Board
from janggi.nnue import NNUE

# 모델 로드
nnue = NNUE.from_file("models/nnue_gpu_iter_5.json")

# 보드 생성
board = Board()

# 위치 평가
evaluation = nnue.evaluate(board)
print(f"Position evaluation: {evaluation}")
```

## 3. 사용 가능한 모델

현재 `models/` 디렉토리에 있는 모델들:

### 자기대전 학습 모델

- `nnue_gpu_model.json` - GPU 기본 학습 모델
- `nnue_gpu_iter_1.json` ~ `nnue_gpu_iter_5.json` - GPU 반복 학습 모델
- `nnue_model.json` - CPU 학습 모델

### 기보 학습 모델

- `nnue_gibo_model.json` - 실제 대국 기보로 학습된 모델

### 모델 선택 가이드

| 모델 | 특징 | 권장 용도 |
|------|------|-----------|
| `nnue_gibo_model.json` | 실전 기보 기반, 현실적 평가 | 일반 게임 |
| `nnue_gpu_iter_5.json` | 가장 많은 자기대전 학습 | 강한 AI |
| `nnue_gpu_model.json` | 기본 GPU 학습 | 빠른 테스트 |

일반적으로:
- **기보 학습 모델**: 실전적인 수 선택, 현실적인 평가
- **자기대전 모델**: 전술적 계산력, 높은 반복수일수록 강함

## 4. 모델 정보 확인

API를 통해 사용 가능한 모델 정보를 확인할 수 있습니다:

```python
import requests

response = requests.get("http://localhost:8000/api/model-info")
print(response.json())
```

## 5. 예제: 완전한 게임 플레이

```python
from janggi.board import Board
from janggi.engine import Engine

# 모델을 사용하는 엔진 생성
engine = Engine(
    depth=4,  # 더 깊은 탐색
    use_nnue=True,
    nnue_model_path="models/nnue_gpu_iter_5.json"
)

# 새 게임 시작
board = Board()

# AI가 수를 두는 예제
while not board.is_checkmate() and not board.is_stalemate():
    # AI의 수
    ai_move = engine.search(board)
    if ai_move:
        board.make_move(ai_move)
        print(f"AI played: {ai_move.to_uci()}")
        print(f"Nodes searched: {engine.nodes_searched}")
    else:
        break
    
    # 여기서 사용자의 수를 입력받을 수 있습니다
    # 예: user_move = input("Your move: ")
    # board.make_move(user_move)
```

## 6. 모델 형식

현재 `models/` 디렉토리의 모델들은 **version 3** 형식(PyTorch 형식)입니다. 이 형식은:

- `NNUE.from_file()` 메서드로 자동 로드됩니다
- `nnue.py`의 `load()` 메서드가 version 3 형식을 지원합니다
- GPU 학습으로 생성된 모델입니다

## 7. 주의사항

1. **경로**: 모델 경로는 프로젝트 루트 디렉토리 기준 상대 경로 또는 절대 경로를 사용할 수 있습니다.

2. **모델 버전**: 모델 파일의 `version` 필드를 확인하세요:
   - Version 1: 구형 단일 레이어 형식
   - Version 2: NumPy 기반 2레이어 형식
   - Version 3: PyTorch 기반 형식 (현재 models/ 디렉토리의 모델들)

3. **성능**: 더 깊은 탐색 깊이(`depth`)와 더 많은 반복을 거친 모델을 사용하면 더 강한 AI를 얻을 수 있지만, 계산 시간도 더 오래 걸립니다.

## 8. 문제 해결

### 모델을 찾을 수 없다는 오류가 발생하는 경우

```python
# 절대 경로 사용
import os
model_path = os.path.join(os.path.dirname(__file__), "models", "nnue_gpu_iter_5.json")
engine = Engine(use_nnue=True, nnue_model_path=model_path)
```

### 모델 로드 실패 시

```python
from janggi.nnue import NNUE

try:
    nnue = NNUE.from_file("models/nnue_gpu_iter_5.json")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # 기본 모델 사용
    nnue = NNUE()
```

