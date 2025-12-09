"""Korean Janggi game engine."""

from .board import Board, Move, Side, PieceType, Piece
from .engine import Engine
from .nnue import NNUE, SimpleEvaluator
from .zobrist import ZobristHash, get_zobrist
from .opening_book import OpeningBook, OpeningMove, get_opening_book
from .bitboard import (
    BitBoard, AttackTables, 
    get_bitboard, get_pawn_attacks, get_king_attacks,
    get_guard_attacks, get_horse_attacks, get_elephant_attacks,
    square_to_bit, bit_to_square, set_bit, clear_bit, is_bit_set,
    popcount, iter_bits, iter_squares,
    FILE_MASKS, RANK_MASKS, HAN_PALACE_MASK, CHO_PALACE_MASK,
)
from .multiplayer import (
    ConnectionManager, RoomManager, GameRoom, Player,
    RoomStatus, PlayerSide, get_connection_manager,
)

# Optional PyTorch support
try:
    from .nnue_torch import NNUETorch, GPUTrainer, get_device
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = [
    # Board and game
    'Board', 'Move', 'Side', 'PieceType', 'Piece',
    'Engine',
    # Evaluation
    'NNUE', 'SimpleEvaluator',
    # Zobrist hashing
    'ZobristHash', 'get_zobrist',
    # Opening book
    'OpeningBook', 'OpeningMove', 'get_opening_book',
    # Bitboard
    'BitBoard', 'AttackTables',
    'get_bitboard', 'get_pawn_attacks', 'get_king_attacks',
    'get_guard_attacks', 'get_horse_attacks', 'get_elephant_attacks',
    'square_to_bit', 'bit_to_square', 'set_bit', 'clear_bit', 'is_bit_set',
    'popcount', 'iter_bits', 'iter_squares',
    'FILE_MASKS', 'RANK_MASKS', 'HAN_PALACE_MASK', 'CHO_PALACE_MASK',
    # Multiplayer
    'ConnectionManager', 'RoomManager', 'GameRoom', 'Player',
    'RoomStatus', 'PlayerSide', 'get_connection_manager',
    # PyTorch (optional)
    'NNUETorch', 'GPUTrainer', 'get_device',
    'TORCH_AVAILABLE',
]
