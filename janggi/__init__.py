"""Korean Janggi game engine."""

from .board import Board, Move, Side, PieceType, Piece
from .engine import Engine
from .nnue import NNUE, SimpleEvaluator

# Optional PyTorch support
try:
    from .nnue_torch import NNUETorch, GPUTrainer, get_device
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = [
    'Board', 'Move', 'Side', 'PieceType', 'Piece',
    'Engine',
    'NNUE', 'SimpleEvaluator',
    'NNUETorch', 'GPUTrainer', 'get_device',
    'TORCH_AVAILABLE',
]
