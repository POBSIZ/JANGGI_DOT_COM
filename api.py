"""FastAPI backend for Janggi game."""

import os
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from time import time

from janggi.board import Board, Move, Side, Piece, PieceType
from janggi.engine import Engine
from janggi.multiplayer import get_connection_manager


# Thread pool for CPU-intensive AI operations
executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    yield
    # Cleanup: shutdown thread pool on app shutdown
    executor.shutdown(wait=True)


app = FastAPI(title="Janggi AI Engine", lifespan=lifespan)


# Default NNUE model path (can be overridden via environment variable)
# Priority: env var > GPU model > CPU model
def _get_default_model_path():
    """Get the best available NNUE model path."""
    env_path = os.environ.get("NNUE_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # Prefer GPU-trained model if available
    if os.path.exists("models/nnue_gpu_model.json"):
        return "models/nnue_gpu_model.json"

    # Fallback to CPU model
    if os.path.exists("models/nnue_model.json"):
        return "models/nnue_model.json"

    return None


DEFAULT_NNUE_MODEL_PATH = _get_default_model_path()
print(f"DEFAULT_NNUE_MODEL_PATH: {DEFAULT_NNUE_MODEL_PATH}")


class GameState:
    """Thread-safe game state container."""

    def __init__(self, board: Board, engine: Engine):
        self.board = board
        self.engine = engine
        self.lock = asyncio.Lock()
        self.last_access = time()
        self.is_processing = False  # AI 처리 중 여부
        self.undo_stack = []  # Stack of (board_state, move_history) for undo


# Global game state with proper locking
games: Dict[str, GameState] = {}
games_lock = asyncio.Lock()  # 전체 게임 딕셔너리 보호용

# Rate limiting configuration
RATE_LIMIT_WINDOW = 1.0  # 1초
RATE_LIMIT_MAX_REQUESTS = 10  # 초당 최대 요청 수
rate_limit_data: Dict[str, List[float]] = {}  # IP별 요청 타임스탬프


async def check_rate_limit(request: Request) -> bool:
    """Check if request should be rate limited."""
    client_ip = request.client.host if request.client else "unknown"
    current_time = time()

    if client_ip not in rate_limit_data:
        rate_limit_data[client_ip] = []

    # 오래된 요청 기록 제거
    rate_limit_data[client_ip] = [
        t for t in rate_limit_data[client_ip] if current_time - t < RATE_LIMIT_WINDOW
    ]

    # 요청 수 확인
    if len(rate_limit_data[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        return False

    rate_limit_data[client_ip].append(current_time)
    return True


async def get_game_state(game_id: str) -> GameState:
    """Get game state with proper error handling."""
    async with games_lock:
        if game_id not in games:
            raise HTTPException(status_code=404, detail="Game not found")
        game_state = games[game_id]
        game_state.last_access = time()
        return game_state


async def cleanup_old_games():
    """Clean up games that haven't been accessed for a long time."""
    current_time = time()
    MAX_IDLE_TIME = 3600  # 1시간

    async with games_lock:
        to_remove = [
            game_id
            for game_id, state in games.items()
            if current_time - state.last_access > MAX_IDLE_TIME
        ]
        for game_id in to_remove:
            del games[game_id]


class MoveRequest(BaseModel):
    """Request model for making a move."""

    game_id: str
    from_square: str  # e.g., "a1"
    to_square: str  # e.g., "b2"


class NewGameRequest(BaseModel):
    """Request model for creating a new game."""

    game_id: str
    depth: int = 3
    use_nnue: bool = True
    nnue_model_path: Optional[str] = (
        None  # Path to trained NNUE model (uses default if None)
    )
    custom_setup: Optional[Dict[str, str]] = None  # e.g., {"a1": "hR", "a10": "cR"}
    formation: Optional[str] = (
        None  # One of "상마상마", "마상마상", "마상상마", "상마마상" (applies to both sides, deprecated)
    )
    han_formation: Optional[str] = (
        None  # Formation for HAN side: "상마상마", "마상마상", "마상상마", "상마마상"
    )
    cho_formation: Optional[str] = (
        None  # Formation for CHO side: "상마상마", "마상마상", "마상상마", "상마마상"
    )


class BoardResponse(BaseModel):
    """Response model for board state."""

    board: List[List[Optional[str]]]
    board_korean: List[List[Optional[Dict[str, str]]]]  # 한글 이름과 색상 정보
    side_to_move: str
    game_over: bool
    winner: Optional[str]
    draw_reason: Optional[str] = (
        None  # 무승부 사유: "repetition", "stalemate", "agreement" 등
    )
    in_check: bool
    legal_moves: List[Dict[str, str]]
    move_history: List[
        Dict[str, Any]
    ]  # 이동 히스토리 (move_number는 int, captured는 bool)
    can_undo: bool = False  # 수 되돌리기 가능 여부
    in_opening_book: bool = False  # 오프닝 북에 있는지 여부


def square_to_coords(square: str) -> tuple:
    """Convert square notation (e.g., 'a1') to (file, rank)."""
    files = "abcdefghi"
    file = files.index(square[0])
    rank = int(square[1:]) - 1
    return (file, rank)


def _copy_board_state(board: Board) -> dict:
    """Create a serializable copy of the board state for undo."""
    import copy

    return {
        "board": copy.deepcopy(board.board),
        "side_to_move": board.side_to_move,
        "move_history": copy.deepcopy(board.move_history),
        "position_history": copy.deepcopy(board.position_history),
        "_current_hash": board._current_hash,
    }


def _restore_board_state(board: Board, state: dict) -> None:
    """Restore board state from a saved copy."""
    board.board = state["board"]
    board.side_to_move = state["side_to_move"]
    board.move_history = state["move_history"]
    board.position_history = state["position_history"]
    board._current_hash = state["_current_hash"]


def coords_to_square(file: int, rank: int) -> str:
    """Convert (file, rank) to square notation."""
    files = "abcdefghi"
    return f"{files[file]}{rank + 1}"


def piece_to_string(piece: Optional[Piece]) -> Optional[str]:
    """Convert piece to string representation."""
    if piece is None:
        return None
    side_char = "h" if piece.side == Side.HAN else "c"
    type_map = {
        PieceType.KING: "K",
        PieceType.GUARD: "G",
        PieceType.ELEPHANT: "E",
        PieceType.HORSE: "H",
        PieceType.ROOK: "R",
        PieceType.CANNON: "C",
        PieceType.PAWN: "P",
    }
    return f"{side_char}{type_map[piece.piece_type]}"


def piece_to_korean_dict(piece: Optional[Piece]) -> Optional[Dict[str, str]]:
    """Convert piece to Korean name and color info."""
    if piece is None:
        return None

    korean_names = {
        PieceType.KING: "왕",
        PieceType.GUARD: "사",
        PieceType.ELEPHANT: "상",
        PieceType.HORSE: "마",
        PieceType.ROOK: "차",
        PieceType.CANNON: "포",
        PieceType.PAWN: "졸",
    }

    side_name = "한" if piece.side == Side.HAN else "초"
    color = "red" if piece.side == Side.HAN else "blue"

    return {
        "name": korean_names[piece.piece_type],
        "side": side_name,
        "color": color,
        "full_name": f"{side_name}{korean_names[piece.piece_type]}",
    }


@app.get("/")
async def root():
    """Serve the HTML frontend."""
    static_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(static_path)


@app.post("/api/new-game")
async def new_game(request: NewGameRequest, req: Request):
    """Create a new game."""
    # Rate limiting 체크
    if not await check_rate_limit(req):
        raise HTTPException(
            status_code=429, detail="Too many requests. Please slow down."
        )

    board = Board(
        custom_setup=request.custom_setup,
        formation=request.formation,
        han_formation=request.han_formation,
        cho_formation=request.cho_formation,
    )

    # Determine NNUE model path
    nnue_model_path = None
    if request.use_nnue:
        # Use provided path, or default path if it exists
        if request.nnue_model_path:
            nnue_model_path = request.nnue_model_path
        elif os.path.exists(DEFAULT_NNUE_MODEL_PATH):
            nnue_model_path = DEFAULT_NNUE_MODEL_PATH

    engine = Engine(
        depth=request.depth, use_nnue=request.use_nnue, nnue_model_path=nnue_model_path
    )

    # Thread-safe game creation
    async with games_lock:
        games[request.game_id] = GameState(board, engine)

    # 오래된 게임 정리 (비동기로 백그라운드에서 실행)
    asyncio.create_task(cleanup_old_games())

    return {
        "status": "ok",
        "game_id": request.game_id,
        "nnue_model": nnue_model_path if nnue_model_path else "default (untrained)",
    }


@app.post("/api/setup/{game_id}")
async def set_setup(game_id: str, setup: Dict[str, str], req: Request):
    """Set custom starting position for a game."""
    # Rate limiting 체크
    if not await check_rate_limit(req):
        raise HTTPException(
            status_code=429, detail="Too many requests. Please slow down."
        )

    game_state = await get_game_state(game_id)

    async with game_state.lock:
        board = Board(custom_setup=setup)
        game_state.board = board

    return {"status": "ok"}


@app.get("/api/board/{game_id}")
async def get_board(game_id: str, req: Request):
    """Get current board state."""
    # Rate limiting 체크
    if not await check_rate_limit(req):
        raise HTTPException(
            status_code=429, detail="Too many requests. Please slow down."
        )

    # 멀티플레이어 게임 처리 (mp_ 접두사)
    if game_id.startswith("mp_"):
        room_id = game_id[3:]  # "mp_" 제거
        manager = get_connection_manager()
        room = manager.room_manager.get_room(room_id)
        
        if not room:
            raise HTTPException(status_code=404, detail="Room not found")
        
        # 멀티플레이어 게임의 경우 Board 객체를 재구성
        # move_history를 기반으로 현재 보드 상태를 만듭니다
        board = Board(
            han_formation=room.han_formation,
            cho_formation=room.cho_formation,
        )
        
        # move_history를 기반으로 수를 적용
        for move_record in room.move_history:
            from_square = move_record.get("from")
            to_square = move_record.get("to")
            if from_square and to_square:
                try:
                    from_file, from_rank = square_to_coords(from_square)
                    to_file, to_rank = square_to_coords(to_square)
                    move = Move(from_file, from_rank, to_file, to_rank)
                    board.make_move(move)
                except Exception:
                    # 이동 실패 시 무시하고 계속 진행
                    pass
        
        # 현재 턴 설정
        board.side_to_move = Side.CHO if room.current_turn.value == "CHO" else Side.HAN
        
        # Convert board to 2D array
        board_array = []
        board_korean = []
        for rank in range(board.RANKS - 1, -1, -1):  # Reverse for display
            row = []
            row_korean = []
            for file in range(board.FILES):
                piece = board.get_piece(file, rank)
                row.append(piece_to_string(piece))
                row_korean.append(piece_to_korean_dict(piece))
            board_array.append(row)
            board_korean.append(row_korean)
        
        # Get legal moves
        try:
            moves = board.generate_moves()
            legal_moves = []
            for move in moves:
                legal_moves.append(
                    {
                        "from": coords_to_square(move.from_file, move.from_rank),
                        "to": coords_to_square(move.to_file, move.to_rank),
                    }
                )
        except Exception as e:
            import traceback
            print(f"Error generating moves: {e}")
            traceback.print_exc()
            legal_moves = []
        
        # Check game status
        game_over = room.status.value in ["finished", "abandoned"]
        winner = None
        draw_reason = None
        
        if game_over:
            # 게임 종료 사유 확인 (room 상태에서 추론)
            if room.status.value == "finished":
                # winner는 room 정보에서 가져와야 하는데, 현재 GameRoom에는 winner 필드가 없음
                # move_history의 마지막 정보나 다른 방법으로 확인 필요
                pass
        
        # Ensure move_history exists
        if not hasattr(board, "move_history"):
            board.move_history = []
        
        # Check if in check
        try:
            in_check = board.is_in_check(board.side_to_move)
        except Exception as e:
            import traceback
            print(f"Error checking if in check: {e}")
            traceback.print_exc()
            in_check = False
        
        # 멀티플레이어 게임은 undo 불가
        can_undo = False
        
        # 멀티플레이어 게임은 opening book 사용 안 함
        in_opening_book = False
        
        return BoardResponse(
            board=board_array,
            board_korean=board_korean,
            side_to_move=board.side_to_move.value,
            game_over=game_over,
            winner=winner,
            draw_reason=draw_reason,
            in_check=in_check,
            legal_moves=legal_moves,
            move_history=room.move_history,  # 멀티플레이어 게임의 move_history 사용
            can_undo=can_undo,
            in_opening_book=in_opening_book,
        )

    try:
        game_state = await get_game_state(game_id)

        # Lock으로 보드 상태 일관성 보장
        async with game_state.lock:
            board = game_state.board

            # Convert board to 2D array
            board_array = []
            board_korean = []
            for rank in range(board.RANKS - 1, -1, -1):  # Reverse for display
                row = []
                row_korean = []
                for file in range(board.FILES):
                    piece = board.get_piece(file, rank)
                    row.append(piece_to_string(piece))
                    row_korean.append(piece_to_korean_dict(piece))
                board_array.append(row)
                board_korean.append(row_korean)

            # Get legal moves
            try:
                moves = board.generate_moves()
                legal_moves = []
                for move in moves:
                    legal_moves.append(
                        {
                            "from": coords_to_square(move.from_file, move.from_rank),
                            "to": coords_to_square(move.to_file, move.to_rank),
                        }
                    )
            except Exception as e:
                # If move generation fails, return empty list
                import traceback

                print(f"Error generating moves: {e}")
                traceback.print_exc()
                legal_moves = []

            # Check game status
            game_over = False
            winner = None
            draw_reason = None
            try:
                if board.is_checkmate():
                    game_over = True
                    winner = "CHO" if board.side_to_move == Side.HAN else "HAN"
                elif board.is_stalemate():
                    game_over = True
                    # 스테일메이트는 움직일 수 없는 쪽이 패배 (장기 규칙)
                    winner = "CHO" if board.side_to_move == Side.HAN else "HAN"
                elif board.is_draw_by_repetition():
                    game_over = True
                    winner = None  # 무승부
                    draw_reason = "repetition"  # 동일 국면 3회 반복
            except Exception as e:
                import traceback

                print(f"Error checking game status: {e}")
                traceback.print_exc()

            # Ensure move_history exists
            if not hasattr(board, "move_history"):
                board.move_history = []

            # Check if in check
            try:
                in_check = board.is_in_check(board.side_to_move)
            except Exception as e:
                import traceback

                print(f"Error checking if in check: {e}")
                traceback.print_exc()
                in_check = False

            # Check if can undo
            can_undo = len(game_state.undo_stack) > 0

            # Check if in opening book
            in_opening_book = False
            try:
                in_opening_book = game_state.engine.is_in_opening_book(board)
            except Exception:
                pass

            return BoardResponse(
                board=board_array,
                board_korean=board_korean,
                side_to_move=board.side_to_move.value,
                game_over=game_over,
                winner=winner,
                draw_reason=draw_reason,
                in_check=in_check,
                legal_moves=legal_moves,
                move_history=board.move_history,
                can_undo=can_undo,
                in_opening_book=in_opening_book,
            )
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        print(f"Error in get_board: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/move")
async def make_move(request: MoveRequest, req: Request):
    """Make a move."""
    # Rate limiting 체크
    if not await check_rate_limit(req):
        raise HTTPException(
            status_code=429, detail="Too many requests. Please slow down."
        )

    game_state = await get_game_state(request.game_id)

    try:
        from_file, from_rank = square_to_coords(request.from_square)
        to_file, to_rank = square_to_coords(request.to_square)

        move = Move(from_file, from_rank, to_file, to_rank)

        # Lock으로 동시 수정 방지
        async with game_state.lock:
            # 이동 전 상태 저장 (undo 용)
            board_copy = _copy_board_state(game_state.board)
            game_state.undo_stack.append(board_copy)
            # 최대 50수까지만 undo 가능하도록 제한
            if len(game_state.undo_stack) > 50:
                game_state.undo_stack.pop(0)

            if not game_state.board.make_move(move):
                # 이동 실패시 undo 스택에서 제거
                game_state.undo_stack.pop()
                raise HTTPException(status_code=400, detail="Illegal move")

            # Check for repetition after move
            if game_state.board.is_draw_by_repetition():
                return {
                    "status": "ok",
                    "move": move.to_uci(),
                    "game_over": True,
                    "winner": None,  # Draw
                    "reason": "draw_by_repetition",
                }

        return {"status": "ok", "move": move.to_uci()}
    except (ValueError, IndexError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid square notation: {e}")


def _run_ai_search(engine: Engine, board: Board):
    """CPU 집약적인 AI 검색을 별도 스레드에서 실행."""
    return engine.search(board)


@app.post("/api/undo/{game_id}")
async def undo_move(game_id: str, req: Request):
    """Undo the last move (or last two moves to undo AI's move as well)."""
    # Rate limiting 체크
    if not await check_rate_limit(req):
        raise HTTPException(
            status_code=429, detail="Too many requests. Please slow down."
        )

    game_state = await get_game_state(game_id)

    async with game_state.lock:
        if not game_state.undo_stack:
            raise HTTPException(status_code=400, detail="No moves to undo")

        # Restore the last saved state
        last_state = game_state.undo_stack.pop()
        _restore_board_state(game_state.board, last_state)

    return {"status": "ok", "message": "Move undone successfully"}


@app.post("/api/undo-pair/{game_id}")
async def undo_move_pair(game_id: str, req: Request):
    """Undo the last two moves (player's move and AI's move)."""
    # Rate limiting 체크
    if not await check_rate_limit(req):
        raise HTTPException(
            status_code=429, detail="Too many requests. Please slow down."
        )

    game_state = await get_game_state(game_id)

    async with game_state.lock:
        if len(game_state.undo_stack) < 2:
            raise HTTPException(status_code=400, detail="Not enough moves to undo")

        # Pop twice to undo both moves
        game_state.undo_stack.pop()  # AI's move
        last_state = game_state.undo_stack.pop()  # Player's move
        _restore_board_state(game_state.board, last_state)

    return {"status": "ok", "message": "Two moves undone successfully"}


@app.post("/api/ai-move/{game_id}")
async def ai_move(game_id: str, req: Request):
    """Get AI move."""
    # Rate limiting 체크
    if not await check_rate_limit(req):
        raise HTTPException(
            status_code=429, detail="Too many requests. Please slow down."
        )

    game_state = await get_game_state(game_id)

    # 이미 AI가 처리 중인지 확인
    async with game_state.lock:
        if game_state.is_processing:
            raise HTTPException(
                status_code=409, detail="AI is already processing a move. Please wait."
            )
        game_state.is_processing = True

    try:
        # Lock 해제 후 AI 검색 실행 (다른 요청이 보드 상태를 읽을 수 있도록)
        # CPU 집약적 작업을 ThreadPoolExecutor에서 실행
        loop = asyncio.get_event_loop()
        best_move = await loop.run_in_executor(
            executor, _run_ai_search, game_state.engine, game_state.board
        )

        if best_move is None:
            raise HTTPException(status_code=400, detail="No legal moves available")

        # 이동 적용 시 다시 Lock 획득
        async with game_state.lock:
            # 이동 전 상태 저장 (undo 용)
            board_copy = _copy_board_state(game_state.board)
            game_state.undo_stack.append(board_copy)
            # 최대 50수까지만 undo 가능하도록 제한
            if len(game_state.undo_stack) > 50:
                game_state.undo_stack.pop(0)

            if not game_state.board.make_move(best_move):
                game_state.undo_stack.pop()  # 이동 실패시 undo 스택에서 제거
                raise HTTPException(status_code=500, detail="AI generated illegal move")

            nodes_searched = game_state.engine.nodes_searched
            used_opening_book = game_state.engine.used_opening_book

            # Check for repetition after move
            game_over = False
            winner = None
            reason = None
            if game_state.board.is_draw_by_repetition():
                game_over = True
                winner = None  # Draw
                reason = "draw_by_repetition"

        result = {
            "status": "ok",
            "move": {
                "from": coords_to_square(best_move.from_file, best_move.from_rank),
                "to": coords_to_square(best_move.to_file, best_move.to_rank),
            },
            "nodes_searched": nodes_searched,
            "from_opening_book": used_opening_book,
        }

        if game_over:
            result["game_over"] = True
            result["winner"] = winner
            result["reason"] = reason

        return result
    finally:
        # 처리 완료 표시
        async with game_state.lock:
            game_state.is_processing = False


@app.get("/api/model-info")
async def get_model_info():
    """Get information about available NNUE models."""
    models = []

    # Check default model
    if os.path.exists(DEFAULT_NNUE_MODEL_PATH):
        models.append(
            {"path": DEFAULT_NNUE_MODEL_PATH, "is_default": True, "exists": True}
        )
    else:
        models.append(
            {"path": DEFAULT_NNUE_MODEL_PATH, "is_default": True, "exists": False}
        )

    # Check for other models in models directory
    models_dir = "models"
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if (
                file.endswith(".json")
                and os.path.join(models_dir, file) != DEFAULT_NNUE_MODEL_PATH
                and "nnue" in file.lower()
            ):
                model_path = os.path.join(models_dir, file)
                models.append({"path": model_path, "is_default": False, "exists": True})

    return {"default_model_path": DEFAULT_NNUE_MODEL_PATH, "models": models}


@app.get("/api/opening-book-info/{game_id}")
async def get_opening_book_info(game_id: str, req: Request):
    """Get information about the opening book."""
    # Rate limiting 체크
    if not await check_rate_limit(req):
        raise HTTPException(
            status_code=429, detail="Too many requests. Please slow down."
        )

    game_state = await get_game_state(game_id)

    async with game_state.lock:
        stats = game_state.engine.get_opening_book_stats()
        in_book = game_state.engine.is_in_opening_book(game_state.board)

    return {
        "statistics": stats,
        "in_book": in_book,
        "enabled": game_state.engine.use_opening_book,
    }


# ============================================
# WebSocket Multiplayer Endpoints
# ============================================

@app.websocket("/ws/multiplayer/{player_id}")
async def websocket_multiplayer(websocket: WebSocket, player_id: str):
    """WebSocket endpoint for multiplayer games."""
    manager = get_connection_manager()
    await manager.connect(websocket, player_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            await manager.handle_message(player_id, data)
    except WebSocketDisconnect:
        await manager.disconnect(player_id)
    except Exception as e:
        print(f"WebSocket error for player {player_id}: {e}")
        await manager.disconnect(player_id)


@app.get("/api/multiplayer/rooms")
async def get_available_rooms():
    """Get list of available rooms to join."""
    manager = get_connection_manager()
    rooms = manager.room_manager.get_available_rooms()
    return {"rooms": rooms}


@app.get("/api/multiplayer/room/{room_id}")
async def get_room_info(room_id: str):
    """Get information about a specific room."""
    manager = get_connection_manager()
    room = manager.room_manager.get_room(room_id)
    
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    return {"room": room.to_dict()}


@app.post("/api/multiplayer/generate-player-id")
async def generate_player_id():
    """Generate a unique player ID for WebSocket connection."""
    player_id = str(uuid.uuid4())
    return {"player_id": player_id}


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
