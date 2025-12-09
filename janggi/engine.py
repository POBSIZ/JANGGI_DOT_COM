"""Janggi AI engine with minimax search."""

from typing import Optional, Dict, Tuple
from .board import Board, Move, Side, PieceType
from .nnue import NNUE, SimpleEvaluator


class Engine:
    """Janggi AI engine with optimized search."""
    
    def __init__(self, depth: int = 3, use_nnue: bool = True, nnue_model_path: str = None):
        """Initialize engine.
        
        Args:
            depth: Search depth for minimax
            use_nnue: Whether to use NNUE evaluator (if False, uses simple evaluator)
            nnue_model_path: Path to trained NNUE model file (optional)
        """
        self.depth = depth
        self.use_nnue = use_nnue
        
        if use_nnue:
            if nnue_model_path:
                self.nnue = NNUE.from_file(nnue_model_path)
            else:
                self.nnue = NNUE()
        else:
            self.nnue = None
            
        self.evaluator = SimpleEvaluator()
        self.nodes_searched = 0
        # Transposition table for caching evaluated positions
        self.tt: Dict[str, Tuple[float, int]] = {}
        self.tt_hits = 0
    
    def search(self, board: Board) -> Optional[Move]:
        """Search for best move using minimax with alpha-beta pruning."""
        self.nodes_searched = 0
        self.tt_hits = 0
        self.tt.clear()  # Clear transposition table for new search
        
        moves = board.generate_moves()
        
        if not moves:
            return None
        
        # Order moves: captures first, then others
        moves = self._order_moves(board, moves)
        
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in moves:
            # Make move using fast method (no deepcopy)
            if not board.make_move_fast(move):
                continue
            
            value = self._minimax(board, self.depth - 1, alpha, beta, False)
            
            # Undo move
            board.undo_move_fast(move)
            
            if value > best_value:
                best_value = value
                best_move = move
            
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break  # Alpha-beta cutoff
        
        return best_move
    
    def _order_moves(self, board: Board, moves: list) -> list:
        """Order moves for better alpha-beta pruning (captures first)."""
        captures = []
        non_captures = []
        
        for move in moves:
            target = board.get_piece(move.to_file, move.to_rank)
            if target is not None:
                # Prioritize high-value captures
                captures.append((move, self._get_capture_value(target.piece_type)))
            else:
                non_captures.append(move)
        
        # Sort captures by value (highest first)
        captures.sort(key=lambda x: -x[1])
        
        return [m for m, _ in captures] + non_captures
    
    def _get_capture_value(self, piece_type: PieceType) -> int:
        """Get value of capturing a piece type."""
        values = {
            PieceType.KING: 1000,
            PieceType.ROOK: 13,
            PieceType.CANNON: 7,
            PieceType.HORSE: 5,
            PieceType.ELEPHANT: 3,
            PieceType.GUARD: 3,
            PieceType.PAWN: 2,
        }
        return values.get(piece_type, 0)
    
    def _minimax(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool
    ) -> float:
        """Minimax algorithm with alpha-beta pruning and transposition table."""
        self.nodes_searched += 1
        
        # Check transposition table
        board_hash = self._hash_board(board)
        if board_hash in self.tt:
            cached_value, cached_depth = self.tt[board_hash]
            if cached_depth >= depth:
                self.tt_hits += 1
                return cached_value
        
        # Terminal conditions
        if depth == 0:
            value = self._evaluate(board)
            self.tt[board_hash] = (value, depth)
            return value
        
        moves = board.generate_moves()
        if not moves:
            # No moves = checkmate or stalemate
            value = -1000.0 if maximizing else 1000.0
            self.tt[board_hash] = (value, depth)
            return value
        
        # Order moves for better pruning
        moves = self._order_moves(board, moves)
        
        if maximizing:
            max_eval = float('-inf')
            for move in moves:
                if not board.make_move_fast(move):
                    continue
                eval_score = self._minimax(board, depth - 1, alpha, beta, False)
                board.undo_move_fast(move)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            self.tt[board_hash] = (max_eval, depth)
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                if not board.make_move_fast(move):
                    continue
                eval_score = self._minimax(board, depth - 1, alpha, beta, True)
                board.undo_move_fast(move)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            self.tt[board_hash] = (min_eval, depth)
            return min_eval
    
    def _hash_board(self, board: Board) -> str:
        """Create a simple hash of the board state."""
        # Fast hash using board positions
        parts = []
        for rank in range(board.RANKS):
            for file in range(board.FILES):
                piece = board.board[rank][file]
                if piece:
                    parts.append(f"{file}{rank}{piece.side.value[0]}{piece.piece_type.value[0]}")
        parts.append(board.side_to_move.value[0])
        return "".join(parts)
    
    def _evaluate(self, board: Board) -> float:
        """Evaluate board position."""
        if self.use_nnue and self.nnue:
            return self.nnue.evaluate(board)
        else:
            return self.evaluator.evaluate(board)

