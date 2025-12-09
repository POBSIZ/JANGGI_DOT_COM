"""Unit tests for move validation and piece-specific movement rules."""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from janggi import Board, Move, Side, PieceType


class TestMoveClass:
    """Test Move class functionality."""
    
    def test_move_creation(self):
        """Test creating a move."""
        move = Move(0, 0, 0, 1)  # a1 to a2
        
        assert move.from_file == 0
        assert move.from_rank == 0
        assert move.to_file == 0
        assert move.to_rank == 1
    
    def test_move_to_uci(self):
        """Test UCI notation conversion."""
        move = Move(0, 0, 0, 1)  # a1 to a2
        
        assert move.to_uci() == "a1a2"
    
    def test_move_from_uci(self):
        """Test parsing UCI notation."""
        move = Move.from_uci("a1a2")
        
        assert move.from_file == 0
        assert move.from_rank == 0
        assert move.to_file == 0
        assert move.to_rank == 1
    
    def test_move_str(self):
        """Test string representation."""
        move = Move(4, 8, 4, 7)  # e9 to e8
        
        assert str(move) == "e9e8"


class TestRookMoves:
    """Test rook movement rules."""
    
    def test_rook_horizontal_move(self):
        """Test rook can move horizontally."""
        custom = {
            "e5": "cR",  # CHO Rook in center
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # Rook should be able to move left/right
        move_left = Move(4, 4, 0, 4)  # e5 to a5
        move_right = Move(4, 4, 8, 4)  # e5 to i5
        
        assert board._is_valid_rook_move(4, 4, 0, 4)  # Left
        assert board._is_valid_rook_move(4, 4, 8, 4)  # Right
    
    def test_rook_vertical_move(self):
        """Test rook can move vertically."""
        custom = {
            "e5": "cR",  # CHO Rook in center
            "d2": "hK",  # HAN King at d2 (not blocking e-file)
            "d9": "cK",  # CHO King at d9
        }
        board = Board(custom_setup=custom)
        
        # Rook should be able to move up/down
        assert board._is_valid_rook_move(4, 4, 4, 0)  # Down
        assert board._is_valid_rook_move(4, 4, 4, 8)  # Up
    
    def test_rook_blocked_by_piece(self):
        """Test rook cannot jump over pieces."""
        custom = {
            "a5": "cR",  # CHO Rook
            "a3": "hP",  # Blocking pawn
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # Rook cannot jump over pawn
        assert board._is_valid_rook_move(0, 4, 0, 0) == False  # a5 to a1 blocked
        assert board._is_valid_rook_move(0, 4, 0, 3) == True   # a5 to a4 OK
    
    def test_rook_palace_diagonal(self):
        """Test rook can move diagonally in palace."""
        custom = {
            "d10": "cR",  # CHO Rook at palace corner
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # Rook at d10 (3, 9) should be able to move diagonally to e9 (4, 8) in palace
        assert board._is_valid_rook_move(3, 9, 4, 8)  # Diagonal move in palace


class TestCannonMoves:
    """Test cannon movement rules."""
    
    def test_cannon_needs_screen(self):
        """Test cannon must have exactly one screen."""
        custom = {
            "a5": "cC",  # CHO Cannon
            "a3": "hP",  # Screen
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # Can jump over pawn
        assert board._is_valid_cannon_move(0, 4, 0, 1) == True  # a5 to a2
        assert board._is_valid_cannon_move(0, 4, 0, 0) == True  # a5 to a1
    
    def test_cannon_no_screen(self):
        """Test cannon cannot move without screen."""
        custom = {
            "a5": "cC",  # CHO Cannon alone
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # No screen, cannot move along file
        assert board._is_valid_cannon_move(0, 4, 0, 0) == False
    
    def test_cannon_two_screens_invalid(self):
        """Test cannon cannot jump over two pieces."""
        custom = {
            "a5": "cC",  # CHO Cannon
            "a3": "hP",  # Screen 1
            "a2": "hP",  # Screen 2
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # Two pieces between, invalid
        assert board._is_valid_cannon_move(0, 4, 0, 0) == False
    
    def test_cannon_cannot_use_cannon_screen(self):
        """Test cannon cannot use another cannon as screen."""
        custom = {
            "a5": "cC",  # CHO Cannon
            "a3": "hC",  # HAN Cannon as screen
            "a1": "hP",  # Target
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # Cannot use cannon as screen
        assert board._is_valid_cannon_move(0, 4, 0, 0) == False
    
    def test_cannon_cannot_capture_cannon(self):
        """Test cannon cannot capture another cannon."""
        custom = {
            "a5": "cC",  # CHO Cannon
            "a3": "hP",  # Screen (non-cannon)
            "a1": "hC",  # HAN Cannon as target
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # Cannot capture cannon
        assert board._is_valid_cannon_move(0, 4, 0, 0) == False


class TestHorseMoves:
    """Test horse movement rules."""
    
    def test_horse_l_shape(self):
        """Test horse moves in L-shape."""
        custom = {
            "e5": "cH",  # CHO Horse in center
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # Horse can move in L-shape
        assert board._is_valid_horse_move(4, 4, 5, 6)  # +1, +2
        assert board._is_valid_horse_move(4, 4, 3, 6)  # -1, +2
        assert board._is_valid_horse_move(4, 4, 6, 5)  # +2, +1
        assert board._is_valid_horse_move(4, 4, 2, 5)  # -2, +1
    
    def test_horse_blocked_leg(self):
        """Test horse is blocked by piece on leg."""
        custom = {
            "e5": "cH",  # CHO Horse at e5 (4, 4)
            "e6": "hP",  # Pawn blocking upward leg at e6 (4, 5)
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # Leg blocked for upward moves
        assert board._is_valid_horse_move(4, 4, 5, 6) == False  # Blocked
        assert board._is_valid_horse_move(4, 4, 3, 6) == False  # Blocked
    
    def test_horse_invalid_moves(self):
        """Test invalid horse moves."""
        custom = {
            "e5": "cH",  # CHO Horse
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # Not L-shape
        assert board._is_valid_horse_move(4, 4, 4, 6) == False  # Straight
        assert board._is_valid_horse_move(4, 4, 5, 5) == False  # Diagonal


class TestElephantMoves:
    """Test elephant movement rules."""
    
    def test_elephant_move_pattern(self):
        """Test elephant moves in correct pattern."""
        custom = {
            "e5": "cE",  # CHO Elephant in center
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # Elephant: 1 orthogonal + 2 diagonal
        # From e5 (4,4), can reach g7 (6,6) via f5 then g6
        assert board._is_valid_elephant_move(4, 4, 6, 7) or board._is_valid_elephant_move(4, 4, 7, 6)
    
    def test_elephant_blocked(self):
        """Test elephant blocked by pieces in path."""
        custom = {
            "e5": "cE",  # CHO Elephant at e5 (4, 4)
            "f5": "hP",  # Pawn blocking first step to right
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # Blocked when trying to go right
        assert board._is_valid_elephant_move(4, 4, 7, 6) == False


class TestGuardMoves:
    """Test guard movement rules."""
    
    def test_guard_stays_in_palace(self):
        """Test guard cannot leave palace."""
        custom = {
            "d8": "cG",  # CHO Guard at d8 (palace edge)
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # Cannot move outside palace
        assert board._is_valid_guard_move(3, 7, 2, 7, Side.CHO) == False  # c8 outside
    
    def test_guard_orthogonal_moves(self):
        """Test guard can move orthogonally in palace."""
        custom = {
            "e8": "cG",  # CHO Guard at center of palace bottom
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # Orthogonal moves within palace
        assert board._is_valid_guard_move(4, 7, 4, 8, Side.CHO)  # Up
        assert board._is_valid_guard_move(4, 7, 3, 7, Side.CHO)  # Left
        assert board._is_valid_guard_move(4, 7, 5, 7, Side.CHO)  # Right
    
    def test_guard_diagonal_from_center(self):
        """Test guard can move diagonally from palace center."""
        custom = {
            "e9": "cK",  # CHO King at center (making room for guard)
            "e2": "hK",
        }
        board = Board(custom_setup=custom)
        
        # King at e9 (4, 8) - center diagonal point
        # Can move diagonally to corners
        assert board._is_valid_king_move(4, 8, 3, 7, Side.CHO)  # d8
        assert board._is_valid_king_move(4, 8, 5, 7, Side.CHO)  # f8


class TestKingMoves:
    """Test king movement rules."""
    
    def test_king_stays_in_palace(self):
        """Test king cannot leave palace."""
        custom = {
            "d8": "cK",  # CHO King at d8 (palace edge)
            "e2": "hK",
        }
        board = Board(custom_setup=custom)
        
        # Cannot move outside palace
        assert board._is_valid_king_move(3, 7, 2, 7, Side.CHO) == False  # c8 outside
    
    def test_king_one_step_only(self):
        """Test king can only move one step."""
        custom = {
            "e9": "cK",  # CHO King at center
            "e2": "hK",
        }
        board = Board(custom_setup=custom)
        
        # Cannot move two steps
        assert board._is_valid_king_move(4, 8, 4, 6, Side.CHO) == False


class TestPawnMoves:
    """Test pawn movement rules."""
    
    def test_pawn_forward(self):
        """Test pawn can move forward."""
        custom = {
            "e5": "cP",  # CHO Pawn
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # CHO pawn moves down (decreasing rank)
        assert board._is_valid_pawn_move(4, 4, 4, 3, Side.CHO)  # Forward
    
    def test_pawn_sideways(self):
        """Test pawn can move sideways."""
        custom = {
            "e5": "cP",  # CHO Pawn
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # Sideways moves
        assert board._is_valid_pawn_move(4, 4, 3, 4, Side.CHO)  # Left
        assert board._is_valid_pawn_move(4, 4, 5, 4, Side.CHO)  # Right
    
    def test_pawn_no_backward(self):
        """Test pawn cannot move backward."""
        custom = {
            "e5": "cP",  # CHO Pawn
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # Cannot move backward
        assert board._is_valid_pawn_move(4, 4, 4, 5, Side.CHO) == False
    
    def test_pawn_at_last_rank(self):
        """Test pawn at last rank can only move sideways."""
        custom = {
            "e1": "cP",  # CHO Pawn at HAN's back rank
            "e2": "hK",
            "e9": "cK",
        }
        board = Board(custom_setup=custom)
        
        # At rank 0, cannot move forward (no rank below)
        # Can only move sideways
        assert board._is_valid_pawn_move(4, 0, 3, 0, Side.CHO)  # Left OK
        assert board._is_valid_pawn_move(4, 0, 5, 0, Side.CHO)  # Right OK


class TestFaceToFaceKings:
    """Test face-to-face kings rule."""
    
    def test_face_to_face_detected(self):
        """Test face-to-face kings is detected."""
        custom = {
            "e9": "cK",  # CHO King
            "e2": "hK",  # HAN King - same file, nothing between
        }
        board = Board(custom_setup=custom)
        
        # Kings are face-to-face on file e
        assert board._would_face_to_face_kings(4, 8, Side.CHO) == True
        assert board._would_face_to_face_kings(4, 1, Side.HAN) == True
    
    def test_face_to_face_blocked(self):
        """Test face-to-face is OK when blocked."""
        custom = {
            "e9": "cK",  # CHO King
            "e5": "cP",  # Pawn between kings
            "e2": "hK",  # HAN King
        }
        board = Board(custom_setup=custom)
        
        # Piece between them
        assert board._would_face_to_face_kings(4, 8, Side.CHO) == False
    
    def test_king_cannot_create_face_to_face(self):
        """Test king cannot move to create face-to-face."""
        custom = {
            "d9": "cK",  # CHO King at d9
            "e5": "cP",  # Pawn blocking e-file
            "e2": "hK",  # HAN King
        }
        board = Board(custom_setup=custom)
        board.side_to_move = Side.CHO
        
        # CHO King at d9 (3, 8)
        # If it moves to e9 (4, 8), would it be face-to-face?
        # There's a pawn at e5 blocking
        assert board._would_face_to_face_kings(4, 8, Side.CHO) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

