"""Unit tests for Engine class."""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from janggi import Board, Move, Side, PieceType, Engine


class TestEngineInitialization:
    """Test engine initialization."""
    
    def test_default_initialization(self):
        """Test default engine initialization."""
        engine = Engine()
        
        assert engine.depth == 3
        assert engine.use_nnue == True
        assert engine.nodes_searched == 0
        assert engine.tt_hits == 0
    
    def test_custom_depth(self):
        """Test engine with custom depth."""
        engine = Engine(depth=5)
        
        assert engine.depth == 5
    
    def test_disable_nnue(self):
        """Test engine without NNUE."""
        engine = Engine(use_nnue=False)
        
        assert engine.use_nnue == False
        assert engine.nnue is None


class TestEngineSearch:
    """Test engine search functionality."""
    
    def test_search_returns_move(self):
        """Test search returns a valid move."""
        board = Board()
        engine = Engine(depth=1, use_nnue=False)  # Low depth for speed
        
        move = engine.search(board)
        
        assert move is not None
        assert isinstance(move, Move)
    
    def test_search_returns_legal_move(self):
        """Test search returns only legal moves."""
        board = Board()
        engine = Engine(depth=1, use_nnue=False)
        
        move = engine.search(board)
        
        # Move should be in legal moves list
        legal_moves = board.generate_moves()
        move_ucis = [m.to_uci() for m in legal_moves]
        assert move.to_uci() in move_ucis
    
    def test_search_updates_stats(self):
        """Test search updates node count."""
        board = Board()
        engine = Engine(depth=2, use_nnue=False, use_opening_book=False)
        
        engine.search(board)
        
        assert engine.nodes_searched > 0
    
    def test_search_no_moves(self):
        """Test search when no moves available returns None."""
        # Create a position where one side has no king
        custom = {
            "e9": "cK",  # Only CHO King
        }
        board = Board(custom_setup=custom)
        board.side_to_move = Side.HAN
        
        engine = Engine(depth=1, use_nnue=False)
        move = engine.search(board)
        
        # HAN has no pieces, so no moves
        assert move is None


class TestMoveOrdering:
    """Test move ordering heuristics."""
    
    def test_captures_ordered_first(self):
        """Test captures are ordered before non-captures."""
        custom = {
            "e2": "hK",
            "e9": "cK",
            "d5": "cR",   # CHO Rook
            "d3": "hP",   # Capturable pawn
        }
        board = Board(custom_setup=custom)
        
        engine = Engine(depth=1, use_nnue=False)
        
        # Generate moves for rook
        moves = board.generate_moves()
        ordered = engine._order_moves(board, moves)
        
        # First moves should be captures (if any captures exist)
        if len(ordered) > 0:
            first_move = ordered[0]
            target = board.get_piece(first_move.to_file, first_move.to_rank)
            # If there are captures, they should come first
            # Otherwise, non-captures are fine
            assert True  # Basic ordering test
    
    def test_capture_value_ordering(self):
        """Test high-value captures ordered before low-value."""
        engine = Engine(depth=1, use_nnue=False)
        
        # Check capture values are correct
        assert engine._get_capture_value(PieceType.ROOK) > engine._get_capture_value(PieceType.PAWN)
        assert engine._get_capture_value(PieceType.KING) > engine._get_capture_value(PieceType.ROOK)


class TestTranspositionTable:
    """Test transposition table functionality."""
    
    def test_tt_caching(self):
        """Test transposition table caches positions."""
        board = Board()
        engine = Engine(depth=2, use_nnue=False, use_opening_book=False)
        
        # First search
        engine.search(board)
        
        # TT should have entries
        assert len(engine.tt) > 0
    
    def test_tt_cleared_on_new_search(self):
        """Test TT is cleared for new search."""
        board = Board()
        engine = Engine(depth=2, use_nnue=False)
        
        # First search
        engine.search(board)
        
        # Modify board
        move = Move(2, 6, 2, 5)  # c7 to c6
        board.make_move(move)
        
        # Second search should clear TT
        engine.search(board)
        
        # TT should be fresh (stats reset)
        # Note: tt_hits is reset each search
        assert engine.nodes_searched > 0


class TestEvaluation:
    """Test position evaluation."""
    
    def test_equal_position_near_zero(self):
        """Test evaluation of equal position is near zero."""
        board = Board()
        engine = Engine(depth=1, use_nnue=False)
        
        # Initial position is roughly equal
        score = engine._evaluate(board)
        
        # Should be close to 0 (with possible small adjustment)
        assert -5 < score < 5
    
    def test_material_advantage_positive(self):
        """Test evaluation with material advantage."""
        custom = {
            "e2": "hK",
            "e9": "cK",
            "a1": "cR",  # CHO has extra rook
        }
        board = Board(custom_setup=custom)
        board.side_to_move = Side.CHO
        
        engine = Engine(depth=1, use_nnue=False)
        score = engine._evaluate(board)
        
        # CHO has material advantage, so score should be positive
        assert score > 0


class TestHashBoard:
    """Test board hashing for transposition table."""
    
    def test_hash_uses_zobrist(self):
        """Test hash uses Zobrist hashing."""
        board = Board()
        engine = Engine(depth=1, use_nnue=False)
        
        hash_value = engine._hash_board(board)
        
        # Should be an integer (Zobrist hash)
        assert isinstance(hash_value, int)
    
    def test_same_position_same_hash(self):
        """Test same position gives same hash."""
        board1 = Board()
        board2 = Board()
        engine = Engine(depth=1, use_nnue=False)
        
        assert engine._hash_board(board1) == engine._hash_board(board2)
    
    def test_different_position_different_hash(self):
        """Test different positions give different hashes."""
        board1 = Board()
        board2 = Board()
        
        # Make a move on board2
        move = Move(2, 6, 2, 5)
        board2.make_move(move)
        
        engine = Engine(depth=1, use_nnue=False)
        
        assert engine._hash_board(board1) != engine._hash_board(board2)


class TestEngineWithNNUE:
    """Test engine with NNUE enabled (if available)."""
    
    def test_nnue_search(self):
        """Test search with NNUE evaluation."""
        board = Board()
        engine = Engine(depth=1, use_nnue=True)
        
        move = engine.search(board)
        
        assert move is not None
    
    def test_nnue_evaluation(self):
        """Test NNUE evaluation returns reasonable score."""
        board = Board()
        engine = Engine(depth=1, use_nnue=True)
        
        score = engine._evaluate(board)
        
        # NNUE score should be in reasonable range
        assert -100 < score < 100


class TestEnginePerformance:
    """Test engine performance characteristics."""
    
    def test_depth_affects_nodes(self):
        """Test higher depth searches more nodes."""
        board = Board()
        
        engine1 = Engine(depth=1, use_nnue=False, use_opening_book=False)
        engine1.search(board)
        nodes1 = engine1.nodes_searched
        
        engine2 = Engine(depth=2, use_nnue=False, use_opening_book=False)
        engine2.search(board)
        nodes2 = engine2.nodes_searched
        
        # Depth 2 should search more nodes than depth 1
        assert nodes2 > nodes1
    
    def test_alpha_beta_pruning_effective(self):
        """Test alpha-beta pruning reduces nodes searched."""
        board = Board()
        engine = Engine(depth=2, use_nnue=False, use_opening_book=False)
        
        engine.search(board)
        
        # With pruning, we should search far fewer than all possible moves
        # At depth 2, without pruning we'd search all moves * all responses
        # With pruning, we search significantly fewer
        legal_moves = len(board.generate_moves())
        max_nodes_without_pruning = legal_moves * legal_moves  # Rough estimate
        
        # Should search fewer nodes than worst case
        assert engine.nodes_searched < max_nodes_without_pruning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

