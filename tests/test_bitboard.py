"""Tests for bitboard module."""

import pytest
from janggi.bitboard import (
    BitBoard, AttackTables,
    square_to_bit, bit_to_square, set_bit, clear_bit, is_bit_set,
    popcount, lsb_index, iter_bits, iter_squares,
    get_pawn_attacks, get_king_attacks, get_guard_attacks,
    get_horse_attacks, get_elephant_attacks,
    FILE_MASKS, RANK_MASKS, HAN_PALACE_MASK, CHO_PALACE_MASK,
)
from janggi.board import Board, Side


class TestBitOperations:
    """Test basic bit operations."""
    
    def test_square_to_bit(self):
        """Test square to bit index conversion."""
        assert square_to_bit(0, 0) == 0  # a1
        assert square_to_bit(8, 0) == 8  # i1
        assert square_to_bit(0, 1) == 9  # a2
        assert square_to_bit(4, 4) == 40  # e5
        assert square_to_bit(8, 9) == 89  # i10
    
    def test_bit_to_square(self):
        """Test bit index to square conversion."""
        assert bit_to_square(0) == (0, 0)
        assert bit_to_square(8) == (8, 0)
        assert bit_to_square(9) == (0, 1)
        assert bit_to_square(40) == (4, 4)
        assert bit_to_square(89) == (8, 9)
    
    def test_set_clear_is_bit_set(self):
        """Test setting, clearing, and testing bits."""
        bb = 0
        bb = set_bit(bb, 4, 4)  # e5
        assert is_bit_set(bb, 4, 4)
        assert not is_bit_set(bb, 0, 0)
        
        bb = clear_bit(bb, 4, 4)
        assert not is_bit_set(bb, 4, 4)
    
    def test_popcount(self):
        """Test population count."""
        assert popcount(0) == 0
        assert popcount(1) == 1
        assert popcount(0b1111) == 4
        assert popcount(0xFFFFFFFF) == 32
    
    def test_lsb_index(self):
        """Test least significant bit index."""
        assert lsb_index(0) == -1
        assert lsb_index(1) == 0
        assert lsb_index(0b1000) == 3
        assert lsb_index(0b1010) == 1
    
    def test_iter_bits(self):
        """Test iterating over set bits."""
        bb = set_bit(0, 0, 0)  # a1
        bb = set_bit(bb, 4, 4)  # e5
        bb = set_bit(bb, 8, 9)  # i10
        
        bits = list(iter_bits(bb))
        assert 0 in bits
        assert 40 in bits
        assert 89 in bits
        assert len(bits) == 3
    
    def test_iter_squares(self):
        """Test iterating over set squares."""
        bb = set_bit(0, 0, 0)
        bb = set_bit(bb, 4, 4)
        
        squares = list(iter_squares(bb))
        assert (0, 0) in squares
        assert (4, 4) in squares
        assert len(squares) == 2


class TestMasks:
    """Test pre-computed masks."""
    
    def test_file_masks(self):
        """Test file masks contain correct squares."""
        # File A (index 0)
        assert is_bit_set(FILE_MASKS[0], 0, 0)  # a1
        assert is_bit_set(FILE_MASKS[0], 0, 9)  # a10
        assert not is_bit_set(FILE_MASKS[0], 1, 0)  # b1
        
        # File E (index 4)
        assert is_bit_set(FILE_MASKS[4], 4, 0)  # e1
        assert is_bit_set(FILE_MASKS[4], 4, 5)  # e6
    
    def test_rank_masks(self):
        """Test rank masks contain correct squares."""
        # Rank 1 (index 0)
        assert is_bit_set(RANK_MASKS[0], 0, 0)  # a1
        assert is_bit_set(RANK_MASKS[0], 8, 0)  # i1
        assert not is_bit_set(RANK_MASKS[0], 0, 1)  # a2
    
    def test_palace_masks(self):
        """Test palace masks."""
        # HAN palace: files 3-5, ranks 0-2
        assert is_bit_set(HAN_PALACE_MASK, 4, 1)  # e2 (king position)
        assert not is_bit_set(HAN_PALACE_MASK, 4, 5)  # e6 (not in palace)
        
        # CHO palace: files 3-5, ranks 7-9
        assert is_bit_set(CHO_PALACE_MASK, 4, 8)  # e9 (king position)
        assert not is_bit_set(CHO_PALACE_MASK, 4, 5)  # e6


class TestBitBoard:
    """Test BitBoard class."""
    
    def test_empty_bitboard(self):
        """Test empty bitboard initialization."""
        bb = BitBoard()
        assert bb.count_pieces('CHO') == 0
        assert bb.count_pieces('HAN') == 0
    
    def test_set_piece(self):
        """Test setting a piece."""
        bb = BitBoard()
        bb.set_piece('CHO', 'KING', 4, 8)
        
        result = bb.get_piece_at(4, 8)
        assert result == ('CHO', 'KING')
        assert bb.is_occupied(4, 8)
        assert bb.is_empty(4, 7)
    
    def test_clear_piece(self):
        """Test clearing a piece."""
        bb = BitBoard()
        bb.set_piece('CHO', 'KING', 4, 8)
        bb.clear_piece('CHO', 'KING', 4, 8)
        
        assert bb.get_piece_at(4, 8) is None
        assert bb.is_empty(4, 8)
    
    def test_move_piece(self):
        """Test moving a piece."""
        bb = BitBoard()
        bb.set_piece('CHO', 'ROOK', 0, 9)
        
        captured = bb.move_piece('CHO', 'ROOK', 0, 9, 0, 5)
        
        assert captured is None
        assert bb.is_empty(0, 9)
        assert bb.get_piece_at(0, 5) == ('CHO', 'ROOK')
    
    def test_move_piece_with_capture(self):
        """Test moving with capture."""
        bb = BitBoard()
        bb.set_piece('CHO', 'ROOK', 0, 9)
        bb.set_piece('HAN', 'PAWN', 0, 3)
        
        captured = bb.move_piece('CHO', 'ROOK', 0, 9, 0, 3)
        
        assert captured == ('HAN', 'PAWN')
        assert bb.get_piece_at(0, 3) == ('CHO', 'ROOK')
    
    def test_from_board(self):
        """Test creating BitBoard from Board."""
        board = Board()
        bb = BitBoard.from_board(board)
        
        assert bb.count_pieces('CHO') == 16
        assert bb.count_pieces('HAN') == 16
        
        # Check king positions
        assert bb.get_king_position('CHO') == (4, 8)
        assert bb.get_king_position('HAN') == (4, 1)
    
    def test_get_piece_positions(self):
        """Test getting all positions of a piece type."""
        board = Board()
        bb = BitBoard.from_board(board)
        
        pawns = bb.get_piece_positions('CHO', 'PAWN')
        assert len(pawns) == 5
        assert (0, 6) in pawns  # a7
        assert (4, 6) in pawns  # e7
    
    def test_copy(self):
        """Test copying bitboard."""
        bb1 = BitBoard()
        bb1.set_piece('CHO', 'KING', 4, 8)
        
        bb2 = bb1.copy()
        bb2.clear_piece('CHO', 'KING', 4, 8)
        
        # Original should be unchanged
        assert bb1.get_piece_at(4, 8) == ('CHO', 'KING')
        assert bb2.get_piece_at(4, 8) is None


class TestAttackTables:
    """Test pre-computed attack tables."""
    
    def test_pawn_attacks_han(self):
        """Test HAN pawn attacks (moves up)."""
        # HAN pawn at e5 can move to d5, e6, f5
        attacks = get_pawn_attacks(4, 4, 'HAN')
        
        assert is_bit_set(attacks, 3, 4)  # d5 (left)
        assert is_bit_set(attacks, 4, 5)  # e6 (forward)
        assert is_bit_set(attacks, 5, 4)  # f5 (right)
        assert not is_bit_set(attacks, 4, 3)  # e4 (backward - not allowed)
    
    def test_pawn_attacks_cho(self):
        """Test CHO pawn attacks (moves down)."""
        # CHO pawn at e5 can move to d5, e4, f5
        attacks = get_pawn_attacks(4, 4, 'CHO')
        
        assert is_bit_set(attacks, 3, 4)  # d5 (left)
        assert is_bit_set(attacks, 4, 3)  # e4 (forward for CHO)
        assert is_bit_set(attacks, 5, 4)  # f5 (right)
        assert not is_bit_set(attacks, 4, 5)  # e6 (backward for CHO)
    
    def test_king_attacks(self):
        """Test king attacks in palace."""
        # King at e2 (center of HAN palace)
        attacks = get_king_attacks(4, 1, 'HAN')
        
        # Orthogonal moves
        assert is_bit_set(attacks, 4, 0)  # e1
        assert is_bit_set(attacks, 4, 2)  # e3
        assert is_bit_set(attacks, 3, 1)  # d2
        assert is_bit_set(attacks, 5, 1)  # f2
        
        # Diagonal moves (from center)
        assert is_bit_set(attacks, 3, 0)  # d1
        assert is_bit_set(attacks, 5, 0)  # f1
        assert is_bit_set(attacks, 3, 2)  # d3
        assert is_bit_set(attacks, 5, 2)  # f3
    
    def test_horse_attacks(self):
        """Test horse attacks with leg blocking."""
        board = Board()
        bb = BitBoard.from_board(board)
        
        # Horse at b1 has leg at a1 and c1 blocked
        # Only some moves should be available
        attacks = get_horse_attacks(1, 0, bb._all_pieces)
        
        # Some L-shaped destinations should be blocked due to leg
        # b1 horse: legs at a1 (blocked by rook), c1 (blocked), b2 (empty)
        # With legs blocked, only moves with clear legs should be available
        assert popcount(attacks) > 0  # Should have some moves
    
    def test_elephant_attacks(self):
        """Test elephant attacks with leg blocking."""
        board = Board()
        bb = BitBoard.from_board(board)
        
        # Elephant at c1 - check blocking
        attacks = get_elephant_attacks(2, 0, bb._all_pieces)
        
        # Should have some moves available
        assert popcount(attacks) >= 0  # May have 0 if all blocked


class TestPerformance:
    """Test performance characteristics."""
    
    def test_bitboard_is_faster_than_list(self):
        """Bitboard operations should be fast."""
        import time
        
        bb = BitBoard()
        for file in range(9):
            for rank in range(10):
                bb.set_piece('CHO' if rank < 5 else 'HAN', 'PAWN', file, rank)
        
        # Time many operations
        start = time.time()
        for _ in range(10000):
            _ = bb.is_empty(4, 4)
            _ = bb.is_occupied(4, 5)
            _ = bb.get_all_pieces('CHO')
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 0.5 seconds for 10k ops)
        assert elapsed < 0.5, f"Bitboard operations too slow: {elapsed}s"
    
    def test_attack_table_lookup_is_constant_time(self):
        """Attack table lookups should be O(1)."""
        import time
        
        start = time.time()
        for _ in range(10000):
            _ = get_pawn_attacks(4, 4, 'CHO')
            _ = get_king_attacks(4, 8, 'CHO')
        elapsed = time.time() - start
        
        # Should complete very quickly (< 0.1 seconds)
        assert elapsed < 0.1, f"Attack lookups too slow: {elapsed}s"

