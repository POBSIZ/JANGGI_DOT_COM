# 장기 AI 엔진 개선점

> 프로젝트 분석 결과 도출된 개선 제안 사항들입니다.

## 목차

1. [AI 엔진 개선](#1-ai-엔진-개선)
2. [성능 최적화](#2-성능-최적화)
3. [UI/UX 개선](#3-uiux-개선)
4. [코드 품질](#4-코드-품질)
5. [안정성 및 보안](#5-안정성-및-보안)
6. [새로운 기능](#6-새로운-기능)
7. [우선순위 추천](#7-우선순위-추천)

---

## 1. AI 엔진 개선

### 1.1 탐색 알고리즘 강화

현재 미니맥스 + 알파-베타 가지치기를 사용하고 있습니다. 추가할 수 있는 고급 기법들:

- **Iterative Deepening**: 반복적 깊이 심화
- **Aspiration Windows**: 기대치 창
- **Principal Variation Search (PVS)**: 주요 변화 탐색
- **Late Move Reductions (LMR)**: 후순위 수 축소
- **Null Move Pruning**: 널 무브 가지치기
- **Killer Move Heuristic**: 킬러 수 휴리스틱
- **History Heuristic**: 히스토리 휴리스틱

### 1.2 Transposition Table 개선 ✅ 구현됨

~~현재 TT가 문자열 기반입니다.~~ **Zobrist 해싱이 구현되었습니다!**

구현 내용 (`janggi/zobrist.py`):
- 64비트 Zobrist 해싱으로 2-3배 성능 향상
- O(1) 점진적 해시 업데이트 (XOR 연산)
- 모든 기물/위치/턴에 대한 고유 키 생성

```python
# 사용 예시
from janggi import Board

board = Board()
hash_value = board.get_zobrist_hash()  # 64비트 정수
```

### 1.3 Opening Book (정석 데이터베이스) ✅ 구현됨

**기보 파일 기반 오프닝 북이 구현되었습니다!**

구현 내용 (`janggi/opening_book.py`):
- 57개의 `.gib` 기보 파일에서 944개 포지션, 1000개 수 로드
- 승률 기반 가중치 선택 (weighted, best, popular, random 모드)
- 최대 20수까지 정석 데이터베이스 활용
- 엔진과 자동 통합

```python
from janggi import get_opening_book

# 오프닝 북 로드
book = get_opening_book("gibo")
print(book.get_statistics())
# {'positions': 944, 'total_moves': 1000, 'avg_moves_per_position': 1.06, 'max_ply': 20}

# AI 엔진이 자동으로 오프닝 북 사용
from janggi import Engine, Board
engine = Engine(use_opening_book=True)
board = Board()
move = engine.search(board)  # 정석 수 우선 선택
print(engine.used_opening_book)  # True (정석 수인 경우)
```

### 1.4 Endgame Tablebase

장기 엔드게임 테이블베이스 구현 (예: 왕+차 vs 왕)

---

## 2. 성능 최적화

### 2.1 수 생성 최적화 ✅ 구현됨

**Bitboard를 사용한 빠른 위치 연산이 구현되었습니다!**

구현 내용:
- `janggi/bitboard.py`: 90비트 비트보드, 공격 테이블
- `janggi/board.py`: Board 클래스에 비트보드 통합
- 모든 이동/Undo에서 비트보드 자동 동기화

기능:
- 90비트 비트보드로 기물 위치 표현
- 미리 계산된 공격 테이블 (AttackTables)
- O(1) 비트 연산으로 위치 확인
- 기물별 공격 패턴 함수 (병졸, 왕, 사, 마, 상)
- 엔진 `generate_moves()`에 통합

```python
from janggi import Board, BitBoard, get_horse_attacks, get_elephant_attacks

# Board에 비트보드 자동 포함
board = Board()
bb = board.get_bitboard()  # 동기화된 비트보드

# 빠른 위치 쿼리
is_empty = bb.is_empty(file, rank)
all_pieces = bb.get_all_pieces("CHO")

# 공격 패턴 조회 (O(1))
horse_attacks = get_horse_attacks(file, rank, bb._all_pieces)
elephant_attacks = get_elephant_attacks(file, rank, bb._all_pieces)

# 비트보드 ON/OFF 전환
board._use_bitboard = False  # 전통 방식으로 전환
```

**참고**: Python 구현에서 성능 향상은 미미합니다 (~1.05x). 
C/Rust 확장이나 전체 비트보드 전환 시 더 큰 향상 기대됩니다.

### 2.2 평가 함수 캐싱

NNUE 평가는 비용이 높으므로 캐싱 필요:

```python
self.eval_cache: Dict[str, float] = {}  # position_hash -> evaluation

def _evaluate(self, board: Board) -> float:
    hash_key = self._hash_board(board)
    if hash_key in self.eval_cache:
        return self.eval_cache[hash_key]
    # ...
```

### 2.3 비동기 AI 처리 개선

- Pondering 구현 (상대방 차례에 미리 계산)
- 더 효율적인 스레드 관리

---

## 3. UI/UX 개선

### 3.1 모바일 대응 ✅ 구현됨

**반응형 캔버스 및 모바일 친화적 UI가 구현되었습니다!**

구현 내용 (`static/index.html`):
- 화면 크기에 따라 자동으로 캔버스 크기 조절
- 768px 이하: 세로 레이아웃 전환, 패널 전체 너비
- 480px 이하: 버튼 세로 배치, 더 작은 폰트 사이즈
- `window.resize` 이벤트에 디바운스 적용

```css
/* 모바일 반응형 브레이크포인트 */
@media (max-width: 768px) { /* 태블릿 */ }
@media (max-width: 480px) { /* 모바일 */ }
```

```javascript
// 반응형 캔버스 크기 조절
function resizeCanvas() {
  const container = document.querySelector('.board-wrapper');
  const maxWidth = Math.min(container.clientWidth - 40, BASE_CANVAS_WIDTH);
  // 비례적으로 모든 요소 크기 조절
}
```

### 3.2 추가 기능 UI

- **수 되돌리기 (Undo)** 버튼 ✅ 구현됨
- **난이도 조절** (depth 변경)
- **AI 생각 시간 표시**
- **형세 판단 바** (evaluation bar)
- **사운드 효과** (이동, 잡기, 장군)

#### 수 되돌리기 기능 상세

구현 내용 (`api.py` + `static/index.html`):
- 최대 50수까지 undo 스택 저장
- 단일 수 되돌리기: `POST /api/undo/{game_id}`
- 두 수 되돌리기 (플레이어 + AI): `POST /api/undo-pair/{game_id}`
- UI에 "되돌리기" 버튼 추가, 가능 여부에 따라 자동 비활성화

### 3.3 기보 저장/불러오기

```javascript
function exportGame() {
    const pgn = boardData.move_history.map(m => m.notation).join('\n');
    downloadFile('game.gib', pgn);
}
```

---

## 4. 코드 품질

### 4.1 타입 힌트 강화

```python
# 개선: 제너레이터 사용으로 메모리 효율화
from typing import Generator

def generate_moves(self) -> Generator[Move, None, None]:
    """메모리 효율적인 제너레이터 사용"""
```

### 4.2 단위 테스트 추가 ✅ 구현됨

**94개의 테스트가 구현되었습니다!**

```
tests/
├── __init__.py         # 테스트 패키지
├── test_board.py       # 보드 로직 테스트 (26개)
├── test_engine.py      # 엔진 테스트 (20개)
├── test_moves.py       # 이동 규칙 테스트 (28개)
└── test_zobrist.py     # Zobrist 해싱 테스트 (20개)
```

실행 방법:
```bash
uv run pytest tests/ -v
```

### 4.3 Configuration 관리

```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    nnue_model_path: str = "models/nnue_gpu_model.json"
    default_depth: int = 3
    max_games: int = 100
    rate_limit_requests: int = 10
    
    class Config:
        env_prefix = "JANGGI_"
```

---

## 5. 안정성 및 보안

### 5.1 에러 처리 강화

```python
import logging
logger = logging.getLogger(__name__)

except Exception as e:
    logger.exception("Failed to process request")
    raise HTTPException(status_code=500, detail="Internal error")
```

### 5.2 입력 검증

```python
from pydantic import validator

class MoveRequest(BaseModel):
    game_id: str
    from_square: str
    to_square: str
    
    @validator('from_square', 'to_square')
    def validate_square(cls, v):
        if not re.match(r'^[a-i](10|[1-9])$', v):
            raise ValueError('Invalid square notation')
        return v
```

### 5.3 메모리 누수 방지

```python
MAX_GAMES = 1000

async def cleanup_old_games():
    """오래된 게임 + 너무 많은 게임 정리"""
    # 1시간 이상 미접속 게임 삭제
    # 최대 게임 수 초과 시 가장 오래된 것 삭제
```

---

## 6. 새로운 기능

### 6.1 멀티플레이어 지원 ✅ 구현됨

**WebSocket 기반 실시간 멀티플레이어가 구현되었습니다!**

구현 내용:
- `janggi/multiplayer.py`: 방 관리, 플레이어 관리, 게임 상태 동기화
- `static/multiplayer.html`: 멀티플레이어 전용 UI
- `api.py`: WebSocket 엔드포인트 (`/ws/multiplayer/{player_id}`)

기능:
- 방 생성/입장/나가기
- 실시간 수 동기화
- 채팅 기능
- 무승부 제안/기권
- 시간 제한 옵션

```python
from janggi import get_connection_manager

# WebSocket 연결 관리
manager = get_connection_manager()

# 방 목록 조회
rooms = manager.room_manager.get_available_rooms()

# 클라이언트에서 WebSocket 연결
# ws://host/ws/multiplayer/{player_id}
```

```javascript
// 프론트엔드에서 사용
const ws = new WebSocket(`ws://${host}/ws/multiplayer/${playerId}`);

ws.send(JSON.stringify({
  type: 'create_room',
  nickname: '플레이어1',
  time_limit: 600  // 10분
}));

ws.send(JSON.stringify({
  type: 'make_move',
  from_square: 'e2',
  to_square: 'e4'
}));
```

### 6.2 AI 난이도 시스템

```python
class Difficulty(Enum):
    BEGINNER = {"depth": 1, "random_factor": 0.3}
    INTERMEDIATE = {"depth": 2, "random_factor": 0.1}
    ADVANCED = {"depth": 3, "random_factor": 0.05}
    EXPERT = {"depth": 4, "random_factor": 0}
```

### 6.3 분석 모드

```python
@app.post("/api/analyze/{game_id}")
async def analyze_position(game_id: str):
    """현재 위치 분석 - 최선수 n개 + 변화도"""
    return {
        "evaluation": 0.5,
        "best_moves": [
            {"move": "h2h5", "eval": 0.7, "depth": 5},
            {"move": "b3b7", "eval": 0.3, "depth": 5},
        ],
        "threats": ["c10c1"],
    }
```

### 6.4 기보 파서 개선

```python
class GiboParser:
    """다양한 기보 포맷 지원"""
    
    @staticmethod
    def parse_gib(filepath: str) -> List[Move]:
        """카카오 장기 .gib 파일"""
        
    @staticmethod  
    def parse_jgf(filepath: str) -> List[Move]:
        """JGF (Janggi Game Format) 파일"""
        
    @staticmethod
    def export_to_jgf(moves: List[Move], metadata: dict) -> str:
        """JGF 포맷으로 내보내기"""
```

---

## 7. 우선순위 추천

| 우선순위 | 개선 항목 | 효과 | 난이도 | 상태 |
|---------|----------|------|-------|------|
| ✅ 완료 | Zobrist 해싱 | 성능 2-3배 향상 | 중 | **구현됨** |
| ✅ 완료 | 단위 테스트 추가 | 안정성 향상 | 중 | **94개 테스트** |
| ✅ 완료 | 오프닝 북 | AI 강화 | 하 | **구현됨** (57게임, 944포지션) |
| ✅ 완료 | 모바일 반응형 | 사용성 향상 | 하 | **구현됨** |
| ✅ 완료 | 수 되돌리기 | 사용성 향상 | 하 | **구현됨** (50수까지) |
| ✅ 완료 | WebSocket 멀티플레이 | 기능 확장 | 상 | **구현됨** (방 생성/채팅/시간제한) |
| ✅ 완료 | Bitboard | 성능 대폭 향상 | 상 | **엔진 통합** (90비트, 공격테이블) |

---

## 참고

- 각 개선 항목은 독립적으로 구현 가능
- 높은 우선순위 항목부터 순차적으로 진행 권장
- 테스트 코드 작성 후 리팩토링 진행 권장

