"""Janggi AI engine with minimax search."""

from typing import Optional, Dict, Tuple
from .board import Board, Move, Side, PieceType
from .nnue import NNUE, SimpleEvaluator
from .zobrist import get_zobrist
from .opening_book import get_opening_book, OpeningBook


class Engine:
    """Janggi AI engine with optimized search."""
    
    def __init__(
        self, 
        depth: int = 3, 
        use_nnue: bool = True, 
        nnue_model_path: str = None,
        use_opening_book: bool = True,
        opening_book_path: str = "gibo"
    ):
        """Initialize engine.
        
        Args:
            depth: Search depth for minimax
            use_nnue: Whether to use NNUE evaluator (if False, uses simple evaluator)
            nnue_model_path: Path to trained NNUE model file (optional)
            use_opening_book: Whether to use opening book for early game
            opening_book_path: Path to directory containing .gib files
        """
        self.depth = depth
        self.use_nnue = use_nnue
        self.use_opening_book = use_opening_book
        
        if use_nnue:
            if nnue_model_path:
                self.nnue = NNUE.from_file(nnue_model_path)
            else:
                self.nnue = NNUE()
        else:
            self.nnue = None
        
        # Initialize opening book
        if use_opening_book:
            try:
                self.opening_book = get_opening_book(opening_book_path)
            except Exception as e:
                print(f"Failed to load opening book: {e}")
                self.opening_book = None
        else:
            self.opening_book = None
            
        self.evaluator = SimpleEvaluator()
        self.nodes_searched = 0
        self.used_opening_book = False  # Track if last move was from book
        # Transposition table for caching evaluated positions (now uses Zobrist hash)
        self.tt: Dict[int, Tuple[float, int]] = {}
        self.tt_hits = 0
        self._zobrist = get_zobrist()
    
    def search(self, board: Board) -> Optional[Move]:
        """Search for best move using minimax with alpha-beta pruning."""
        self.nodes_searched = 0
        self.tt_hits = 0
        self.used_opening_book = False
        self.tt.clear()  # Clear transposition table for new search
        
        # Try opening book first
        if self.use_opening_book and self.opening_book:
            book_move = self._get_book_move(board)
            if book_move:
                self.used_opening_book = True
                return book_move
        
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
    
    def _hash_board(self, board: Board) -> int:
        """Get Zobrist hash of the board state.
        
        Uses the board's cached Zobrist hash for O(1) lookup.
        This is 2-3x faster than string-based hashing.
        """
        return board.get_zobrist_hash()
    
    def _evaluate(self, board: Board) -> float:
        """Evaluate board position."""
        if self.use_nnue and self.nnue:
            return self.nnue.evaluate(board)
        else:
            return self.evaluator.evaluate(board)
    
    def _get_book_move(self, board: Board) -> Optional[Move]:
        """Get a move from the opening book if available.
        
        Args:
            board: Current board position
            
        Returns:
            Move from opening book, or None if not in book
        """
        if not self.opening_book:
            return None
        
        ply = len(board.move_history)
        book_result = self.opening_book.get_book_move(
            board.move_history, 
            ply,
            selection_method="weighted"
        )
        
        if not book_result:
            return None
        
        from_square, to_square = book_result
        
        # Convert notation to Move object
        try:
            files = "abcdefghi"
            from_file = files.index(from_square[0])
            from_rank = int(from_square[1:]) - 1
            to_file = files.index(to_square[0])
            to_rank = int(to_square[1:]) - 1
            
            move = Move(from_file, from_rank, to_file, to_rank)
            
            # Verify the move is legal
            legal_moves = board.generate_moves()
            for legal_move in legal_moves:
                if (legal_move.from_file == move.from_file and 
                    legal_move.from_rank == move.from_rank and
                    legal_move.to_file == move.to_file and 
                    legal_move.to_rank == move.to_rank):
                    return legal_move
            
            return None
            
        except (ValueError, IndexError):
            return None
    
    def is_in_opening_book(self, board: Board) -> bool:
        """Check if the current position is in the opening book.
        
        Args:
            board: Current board position
            
        Returns:
            True if there are book moves available
        """
        if not self.opening_book:
            return False
        
        ply = len(board.move_history)
        return self.opening_book.is_in_book(board.move_history, ply)
    
    def get_opening_book_stats(self) -> Optional[Dict]:
        """Get statistics about the opening book.
        
        Returns:
            Dictionary with book statistics, or None if no book loaded
        """
        if not self.opening_book:
            return None
        return self.opening_book.get_statistics()

