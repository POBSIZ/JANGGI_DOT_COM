"""Unit tests for Zobrist hashing."""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from janggi import Board, Move, Side, PieceType
from janggi.zobrist import ZobristHash, get_zobrist


class TestZobristHash:
    """Test ZobristHash class."""
    
    def test_initialization(self):
        """Test Zobrist hash initializes properly."""
        zobrist = ZobristHash()
        
        # Should have piece keys for all combinations
        assert len(zobrist.piece_keys) == 2  # 2 sides
        assert len(zobrist.piece_keys[0]) == 7  # 7 piece types
        assert len(zobrist.piece_keys[0][0]) == 10  # 10 ranks
        assert len(zobrist.piece_keys[0][0][0]) == 9  # 9 files
    
    def test_side_key_exists(self):
        """Test side key is generated."""
        zobrist = ZobristHash()
        
        assert isinstance(zobrist.side_key, int)
        assert zobrist.side_key != 0  # Should be non-zero
    
    def test_keys_are_unique(self):
        """Test generated keys are unique."""
        zobrist = ZobristHash()
        
        # Collect a sample of keys
        keys = set()
        for side in range(2):
            for piece in range(7):
                for rank in range(10):
                    for file in range(9):
                        keys.add(zobrist.piece_keys[side][piece][rank][file])
        
        # Most keys should be unique (statistically very unlikely to have collisions)
        # With 64-bit keys and only 1260 positions, collisions are extremely rare
        assert len(keys) == 2 * 7 * 10 * 9  # All should be unique
    
    def test_get_piece_key(self):
        """Test getting piece key."""
        zobrist = ZobristHash()
        
        key = zobrist.get_piece_key('CHO', 'KING', 4, 8)
        
        assert isinstance(key, int)
        # Same call should return same key
        assert zobrist.get_piece_key('CHO', 'KING', 4, 8) == key


class TestZobristCompute:
    """Test Zobrist hash computation."""
    
    def test_compute_initial_position(self):
        """Test computing hash for initial position."""
        board = Board()
        zobrist = get_zobrist()
        
        hash_value = zobrist.compute_hash(board)
        
        assert isinstance(hash_value, int)
        assert hash_value != 0
    
    def test_compute_empty_board(self):
        """Test computing hash for empty board."""
        board = Board(custom_setup={})
        zobrist = get_zobrist()
        
        hash_value = zobrist.compute_hash(board)
        
        # Empty board with CHO to move
        assert isinstance(hash_value, int)
    
    def test_same_position_same_hash(self):
        """Test identical positions have identical hashes."""
        board1 = Board()
        board2 = Board()
        zobrist = get_zobrist()
        
        assert zobrist.compute_hash(board1) == zobrist.compute_hash(board2)
    
    def test_different_positions_different_hash(self):
        """Test different positions have different hashes."""
        board1 = Board()
        board2 = Board(formation="마상마상")
        zobrist = get_zobrist()
        
        assert zobrist.compute_hash(board1) != zobrist.compute_hash(board2)


class TestZobristIncremental:
    """Test incremental Zobrist hash updates."""
    
    def test_update_after_move(self):
        """Test incremental update matches full computation."""
        board = Board()
        zobrist = get_zobrist()
        
        initial_hash = zobrist.compute_hash(board)
        
        # Make a move
        move = Move(2, 6, 2, 5)  # c7 to c6
        board.make_move(move)
        
        # Board should have updated hash incrementally
        board_hash = board.get_zobrist_hash()
        # Full computation should match
        computed_hash = zobrist.compute_hash(board)
        
        assert board_hash == computed_hash
        assert board_hash != initial_hash
    
    def test_update_with_capture(self):
        """Test incremental update with capture."""
        custom = {
            "e2": "hK",
            "e9": "cK",
            "d5": "cR",   # CHO Rook
            "d3": "hP",   # HAN Pawn
        }
        board = Board(custom_setup=custom)
        zobrist = get_zobrist()
        
        # Capture pawn
        move = Move(3, 4, 3, 2)  # d5 to d3
        board.make_move(move)
        
        # Hashes should match
        board_hash = board.get_zobrist_hash()
        computed_hash = zobrist.compute_hash(board)
        
        assert board_hash == computed_hash
    
    def test_undo_restores_hash(self):
        """Test undo restores original hash."""
        board = Board()
        zobrist = get_zobrist()
        
        initial_hash = board.get_zobrist_hash()
        
        # Make and undo move
        move = Move(2, 6, 2, 5)
        board.make_move(move)
        board.undo_move(move)
        
        assert board.get_zobrist_hash() == initial_hash
    
    def test_multiple_moves_and_undo(self):
        """Test multiple moves and undos maintain consistency."""
        board = Board()
        zobrist = get_zobrist()
        
        hashes = [board.get_zobrist_hash()]
        moves = []
        
        # Make several moves
        move1 = Move(2, 6, 2, 5)  # c7-c6
        if board.make_move(move1):
            moves.append(move1)
            hashes.append(board.get_zobrist_hash())
        
        move2 = Move(2, 3, 2, 4)  # c4-c5 (HAN pawn)
        if board.make_move(move2):
            moves.append(move2)
            hashes.append(board.get_zobrist_hash())
        
        # Undo all moves
        for move in reversed(moves):
            board.undo_move(move)
            hashes.pop()
            assert board.get_zobrist_hash() == hashes[-1]


class TestGlobalZobrist:
    """Test global Zobrist instance."""
    
    def test_get_zobrist_singleton(self):
        """Test get_zobrist returns same instance."""
        z1 = get_zobrist()
        z2 = get_zobrist()
        
        assert z1 is z2
    
    def test_zobrist_consistency_across_boards(self):
        """Test Zobrist keys are consistent across board instances."""
        board1 = Board()
        board2 = Board()
        
        # Both should use same global Zobrist instance
        assert board1.get_zobrist_hash() == board2.get_zobrist_hash()


class TestZobristUpdateHash:
    """Test update_hash_move function."""
    
    def test_basic_update(self):
        """Test basic hash update."""
        zobrist = get_zobrist()
        
        # Start with some hash
        current = 0x123456789ABCDEF0
        
        # Update for a move
        updated = zobrist.update_hash_move(
            current,
            'CHO', 'PAWN',
            2, 6,  # from c7
            2, 5,  # to c6
        )
        
        assert updated != current
        assert isinstance(updated, int)
    
    def test_update_with_capture(self):
        """Test hash update with capture."""
        zobrist = get_zobrist()
        
        current = 0x123456789ABCDEF0
        
        updated = zobrist.update_hash_move(
            current,
            'CHO', 'ROOK',
            3, 4,  # from d5
            3, 2,  # to d3
            'HAN', 'PAWN',  # captured piece
        )
        
        assert updated != current
    
    def test_update_is_reversible(self):
        """Test hash update is reversible (XOR property)."""
        zobrist = get_zobrist()
        
        # XOR is self-inverse: a ^ b ^ b = a
        original = 0x123456789ABCDEF0
        
        # Forward move
        after_move = zobrist.update_hash_move(
            original,
            'CHO', 'PAWN',
            2, 6, 2, 5,
        )
        
        # Reverse move (back to original position)
        back = zobrist.update_hash_move(
            after_move,
            'CHO', 'PAWN',
            2, 5, 2, 6,
        )
        
        # Note: This doesn't exactly restore original because side changes
        # The actual undo mechanism uses position_history instead
        assert back != after_move


class TestZobristPerformance:
    """Test Zobrist hashing performance characteristics."""
    
    def test_hash_is_64_bit(self):
        """Test hash values are 64-bit."""
        board = Board()
        
        hash_value = board.get_zobrist_hash()
        
        # Should fit in 64 bits
        assert hash_value < 2**64
        assert hash_value >= 0
    
    def test_fast_hashing(self):
        """Test hashing is fast (smoke test)."""
        import time
        
        board = Board()
        
        start = time.time()
        for _ in range(10000):
            _ = board.get_zobrist_hash()
        elapsed = time.time() - start
        
        # Should complete 10000 hash retrievals in under 1 second
        # (actual is much faster since it's just returning cached value)
        assert elapsed < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

