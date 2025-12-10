---
layout: post
title: 장기 AI 엔진 구현 가이드
description: JANGGI_DOT_COM 프로젝트의 구현 방법, 순서, 원리 설명
categories:
  - development
tags:
  - janggi
  - ai
  - nnue
  - implementation
datetime: 2025-12-08T08:00:00
mermaid: true
updated_at: 2025-12-08T08:00:00
aliases:
---

# 장기 AI 엔진 구현 가이드

## 발생

장기 AI 엔진을 개발하면서 체스 엔진과는 다른 장기의 특수성을 고려해야 했다. 장기는 9×10 보드, 궁성(宮城)이라는 특수 영역, 그리고 각 말의 고유한 이동 규칙을 가지고 있다. 또한 체스와 달리 포(砲)와 같은 특수한 말이 있어서 이동 생성과 평가 함수 설계가 복잡하다.

이 프로젝트는 NNUE(Efficiently Updatable Neural Networks) 기반 평가 함수를 사용하여 실시간으로 강한 수를 계산할 수 있는 장기 AI 엔진을 구현하는 것이 목표였다.

## 프로젝트 구조

프로젝트는 크게 백엔드와 프론트엔드로 나뉜다.

### 백엔드 (Python)

```
janggi/
├── board.py          # 보드 표현 및 이동 생성
├── bitboard.py       # 비트보드 최적화
├── zobrist.py        # Zobrist 해싱
├── engine.py         # 미니맥스 검색 엔진
├── nnue.py           # NNUE 평가 함수
└── opening_book.py   # 오프닝 북
```

### 프론트엔드 (React + TypeScript)

```
frontend/src/
├── pages/GamePage/
│   ├── components/   # UI 컴포넌트
│   ├── hooks/        # 커스텀 훅
│   └── index.tsx     # 메인 게임 페이지
└── ...
```

## 핵심 컴포넌트

### 1. 보드 표현 (Board)

장기 보드는 9×10 격자로 구성되어 있으며, 각 말은 `Piece` 객체로 표현된다.

```python
class Board:
    FILES = 9
    RANKS = 10
    
    def __init__(self):
        self.board: List[List[Optional[Piece]]] = [...]
        self.side_to_move = Side.CHO
        self.move_history: List[Dict] = []
        self.position_history: List[int] = []  # Zobrist 해시
```

보드의 핵심 기능:
- **이동 생성**: 각 말의 이동 규칙에 따라 합법적인 수 생성
- **이동 적용/취소**: 빠른 이동 적용을 위한 `make_move_fast()` / `undo_move_fast()`
- **게임 종료 판정**: 체크메이트, 스테일메이트, 무승부(3회 반복) 감지

### 2. 비트보드 (BitBoard)

비트보드는 90비트(9×10)를 사용하여 말의 위치를 표현한다. 각 비트는 보드의 한 칸에 대응된다.

```python
# 비트 위치 계산
def square_to_bit(file: int, rank: int) -> int:
    return rank * 9 + file
```

비트보드의 장점:
- **빠른 위치 조회**: 비트 연산으로 O(1) 시간에 말의 존재 여부 확인
- **공격 범위 계산**: 비트 마스크를 사용한 빠른 공격 범위 계산
- **이동 생성 최적화**: 비트 연산으로 합법적인 이동을 빠르게 필터링

예를 들어, 한 쪽의 모든 말 위치를 하나의 정수로 표현할 수 있어 메모리 효율적이고 연산이 빠르다.

### 3. Zobrist 해싱

Zobrist 해싱은 보드 상태를 64비트 정수로 빠르게 해싱하는 방법이다. 각 말의 위치와 차례를 XOR 연산으로 결합한다.

```python
class ZobristHash:
    def __init__(self):
        # 각 말 타입 × 각 위치 × 각 진영에 대한 랜덤 64비트 정수
        self.piece_table = np.random.randint(0, 2**64, ...)
        self.side_to_move = random.randint(0, 2**64)
```

Zobrist 해싱의 장점:
- **증분 업데이트**: 이동 시 이전 해시에 XOR만 하면 새 해시 계산 가능 (O(1))
- **전치 테이블**: 동일한 보드 상태를 빠르게 찾아 평가 결과 재사용
- **반복 감지**: 동일한 보드 상태가 3회 반복되면 무승부 판정

### 4. 검색 엔진 (Engine)

미니맥스 알고리즘과 알파-베타 가지치기를 사용하여 최적의 수를 찾는다.

```python
def _minimax(self, board, depth, alpha, beta, maximizing):
    if depth == 0:
        return self._evaluate(board)
    
    moves = board.generate_moves()
    moves = self._order_moves(board, moves)  # 중요 수 우선
    
    if maximizing:
        max_eval = float('-inf')
        for move in moves:
            board.make_move_fast(move)
            eval_score = self._minimax(board, depth-1, alpha, beta, False)
            board.undo_move_fast(move)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # 알파-베타 가지치기
        return max_eval
```

최적화 기법:
- **이동 순서화**: 포획 수를 먼저 검색하여 가지치기 효율 향상
- **전치 테이블**: 이전에 평가한 보드 상태 재사용
- **빠른 이동 적용**: `make_move_fast()`로 깊은 복사 없이 이동 적용

### 5. NNUE 평가 함수

NNUE는 Efficiently Updatable Neural Networks의 약자로, 보드 상태를 평가하는 신경망이다.

#### 구조

```
입력 특징 (Feature Extraction)
    ↓
특징 압축 (512차원)
    ↓
은닉층 1 (256차원, ReLU)
    ↓
은닉층 2 (64차원, ReLU)
    ↓
출력 (1차원, 평가 점수)
```

#### 특징 추출

NNUE는 다음과 같은 특징을 사용한다:

1. **말-위치 특징**: 각 말 타입이 각 위치에 있는지 (90칸 × 7종류 × 2진영 = 1260차원)
2. **물질 특징**: 각 진영의 말 개수 (7종류 × 2진영 = 14차원)
3. **기동성 특징**: 각 말 타입의 이동 가능한 수 (14차원)
4. **왕 안전성**: 왕 주변의 안전도 (8차원)
5. **위치 특징**: 졸의 전진도, 중심 통제 등 (20차원)

이 특징들을 512차원으로 압축하여 신경망에 입력한다.

#### 효율적인 업데이트

NNUE의 핵심은 이동 시 전체 특징을 다시 계산하지 않고, 변경된 부분만 업데이트하는 것이다.

```python
def evaluate(self, board: Board) -> float:
    features = self._extract_features(board)
    return self._forward(features)
```

이동 전후의 특징 차이만 계산하면 되므로 매우 빠르다.

### 6. 오프닝 북

초반 수를 미리 계산하여 저장해두는 데이터베이스다. 기보 파일(.gib)을 파싱하여 구축한다.

```python
class OpeningBook:
    def get_book_move(self, move_history, ply, selection_method="weighted"):
        # 현재 국면에 맞는 수를 기보에서 찾아 반환
        ...
```

오프닝 북의 장점:
- **빠른 초반 수**: 검색 없이 즉시 수를 반환
- **검증된 수**: 실제 기보에서 나온 수를 사용
- **다양성**: 가중치 기반 선택으로 다양한 초반 전개 가능

### 7. 프론트엔드

React + TypeScript로 구현된 웹 인터페이스다.

#### 주요 기능

- **Canvas 기반 보드 렌더링**: HTML5 Canvas로 보드와 말을 그린다
- **실시간 게임 상태**: FastAPI 백엔드와 WebSocket 없이 REST API로 통신
- **이동 히스토리**: 수 순서를 기록하고 되돌리기 가능
- **AI 수 요청**: 백엔드에 AI 수 계산 요청

#### 상태 관리

```typescript
const {
  boardData,
  legalMoves,
  makeMove,
  aiMove,
  undoMovePair,
} = useGameApi();
```

커스텀 훅을 사용하여 게임 상태와 API 호출을 관리한다.

## 구현 순서

프로젝트는 다음과 같은 순서로 구현되었다:

### Phase 1: 기본 보드 및 이동 규칙

1. **보드 표현 구현** (`board.py`)
   - 9×10 보드 초기화
   - 각 말의 이동 규칙 구현
   - 합법적인 이동 생성

2. **이동 검증**
   - 왕이 체크 상태인지 확인
   - 이동 후 자가 체크 방지
   - 게임 종료 조건 판정

### Phase 2: 검색 엔진

3. **기본 평가 함수** (`SimpleEvaluator`)
   - 말의 가치 기반 평가
   - 위치 점수 (중심 통제, 전진도 등)

4. **미니맥스 알고리즘** (`engine.py`)
   - 기본 미니맥스 구현
   - 알파-베타 가지치기 추가
   - 이동 순서화로 가지치기 효율 향상

### Phase 3: 성능 최적화

5. **비트보드 도입** (`bitboard.py`)
   - 말 위치를 비트로 표현
   - 빠른 공격 범위 계산
   - 이동 생성 최적화

6. **Zobrist 해싱** (`zobrist.py`)
   - 보드 상태 해싱
   - 전치 테이블 구현
   - 반복 감지

7. **빠른 이동 적용**
   - `make_move_fast()` / `undo_move_fast()` 구현
   - 깊은 복사 없이 이동 적용/취소

### Phase 4: NNUE 평가 함수

8. **NNUE 구조 설계** (`nnue.py`)
   - 특징 추출 함수 구현
   - 2층 신경망 구조
   - 순전파/역전파 구현

9. **NNUE 학습** (`scripts/train_nnue.py`)
   - 자기 대국 데이터 생성
   - 깊은 검색 결과를 정답으로 사용
   - 반복적 자기 개선

### Phase 5: 오프닝 북

10. **기보 파싱** (`opening_book.py`)
    - .gib 파일 형식 파싱
    - 이동 히스토리 트리 구축

11. **오프닝 북 통합**
    - 엔진에 오프닝 북 연결
    - 기보에 있는 초반 수 자동 선택

### Phase 6: 백엔드 API

12. **FastAPI 서버** (`api.py`)
    - 게임 생성/관리
    - 이동 처리
    - AI 수 계산
    - Rate limiting

### Phase 7: 프론트엔드

13. **React UI 구현**
    - Canvas 기반 보드 렌더링
    - 게임 상태 관리
    - API 통신

## 원리

### 미니맥스 알고리즘

미니맥스는 두 플레이어가 최적의 수를 둔다고 가정하고, 각 수의 결과를 평가하여 최선의 수를 선택하는 알고리즘이다.

```
최대화 플레이어 (나)
    ↓
최소화 플레이어 (상대)
    ↓
최대화 플레이어 (나)
    ↓
...
```

각 레벨에서:
- **최대화 레벨**: 가능한 최대 점수를 선택
- **최소화 레벨**: 가능한 최소 점수를 선택

### 알파-베타 가지치기

알파-베타 가지치기는 불필요한 수를 검색하지 않도록 하는 최적화 기법이다.

```
alpha: 최대화 플레이어가 보장할 수 있는 최소 점수
beta: 최소화 플레이어가 보장할 수 있는 최대 점수

만약 alpha >= beta라면, 이 분기를 더 검색할 필요가 없다.
```

예를 들어, 상대가 이미 더 나은 수를 찾았다면 현재 분기를 더 탐색할 필요가 없다.

### NNUE의 효율성

일반적인 신경망은 보드 상태가 바뀔 때마다 전체 특징을 다시 계산해야 한다. 하지만 NNUE는 이동 전후의 차이만 계산한다.

```
이동 전: features_before
이동 후: features_after = features_before + delta

delta는 이동한 말과 포획된 말만 고려하면 되므로 매우 작다.
```

이를 통해 평가 함수 호출이 매우 빠르게 이루어진다.

### 비트보드의 성능

비트보드는 CPU의 비트 연산 명령어를 활용하여 여러 위치를 동시에 처리할 수 있다.

```python
# 모든 한 진영 말의 위치
han_pieces = han_king | han_guard | han_elephant | ...

# 특정 위치에 말이 있는지 확인
if han_pieces & (1 << square):
    # 말이 있음
```

비트 연산은 일반적인 배열 접근보다 훨씬 빠르다.

## 의문

프로젝트를 진행하면서 다음과 같은 의문이 생겼다:

1. **NNUE의 특징 선택**: 어떤 특징이 가장 중요한가? 말-위치 특징만으로도 충분한가?
2. **검색 깊이 vs 평가 함수**: 깊은 검색과 좋은 평가 함수 중 어느 것이 더 중요한가?
3. **오프닝 북의 효과**: 오프닝 북이 실제로 게임 강도 향상에 기여하는가?
4. **비트보드의 한계**: 장기 보드가 9×10으로 작아서 비트보드의 이점이 제한적일 수 있다.

## 결론

JANGGI_DOT_COM 프로젝트는 체스 엔진의 기법을 장기에 적용하여 구현되었다. 비트보드, Zobrist 해싱, NNUE 평가 함수 등을 통해 실시간으로 강한 수를 계산할 수 있는 AI 엔진을 만들 수 있었다.

특히 NNUE는 평가 함수의 속도를 크게 향상시켜, 더 깊은 검색을 가능하게 했다. 또한 오프닝 북을 통해 초반 수의 품질도 향상시킬 수 있었다.

앞으로의 개선 방향:
- 더 깊은 검색 (병렬 처리, 시간 제어)
- 더 정교한 NNUE 특징
- 엔드게임 데이터베이스
- 학습 데이터 품질 향상

---

[development](development/README.md)

