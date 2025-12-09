"""Janggi board representation and move generation."""

from enum import Enum
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


class Side(Enum):
    """Player sides."""

    CHO = "CHO"  # Top side, moves first
    HAN = "HAN"  # Bottom side, moves second


class PieceType(Enum):
    """Piece types."""

    KING = "KING"
    GUARD = "GUARD"
    ELEPHANT = "ELEPHANT"
    HORSE = "HORSE"
    ROOK = "ROOK"
    CANNON = "CANNON"
    PAWN = "PAWN"


@dataclass
class Piece:
    """Represents a piece on the board."""

    side: Side
    piece_type: PieceType

    def __str__(self) -> str:
        return f"{self.side.value}_{self.piece_type.value}"


@dataclass
class Move:
    """Represents a move."""

    from_file: int  # 0-8 (a-i)
    from_rank: int  # 0-9 (1-10)
    to_file: int
    to_rank: int
    captured_piece: Optional["Piece"] = None  # For undo

    def __str__(self) -> str:
        files = "abcdefghi"
        return f"{files[self.from_file]}{self.from_rank + 1}{files[self.to_file]}{self.to_rank + 1}"

    def to_uci(self) -> str:
        """Convert to UCI-like notation."""
        files = "abcdefghi"
        return f"{files[self.from_file]}{self.from_rank + 1}{files[self.to_file]}{self.to_rank + 1}"

    @classmethod
    def from_uci(cls, uci: str) -> "Move":
        """Parse UCI notation."""
        files = "abcdefghi"
        from_file = files.index(uci[0])
        from_rank = int(uci[1]) - 1
        to_file = files.index(uci[2])
        to_rank = int(uci[3]) - 1
        return cls(from_file, from_rank, to_file, to_rank)


class Board:
    """Janggi board representation."""

    FILES = 9
    RANKS = 10

    def __init__(self, custom_setup: Optional[Dict[str, str]] = None, formation: Optional[str] = None, han_formation: Optional[str] = None, cho_formation: Optional[str] = None):
        """Initialize empty board.
        
        Args:
            custom_setup: Optional dictionary mapping squares (e.g., "a1") to piece codes (e.g., "hR" for Han Rook)
            formation: Optional formation name ("상마상마", "마상마상", "마상상마", "상마마상") - applies to both sides (deprecated, use han_formation/cho_formation)
            han_formation: Optional formation name for HAN side ("상마상마", "마상마상", "마상상마", "상마마상")
            cho_formation: Optional formation name for CHO side ("상마상마", "마상마상", "마상상마", "상마마상")
        """
        self.board: List[List[Optional[Piece]]] = [
            [None for _ in range(self.FILES)] for _ in range(self.RANKS)
        ]
        self.side_to_move = Side.CHO
        self.move_history: List[Dict[str, str]] = []  # Track move history
        self.position_history: List[str] = []  # Track position hashes for repetition detection
        if custom_setup:
            self._initialize_custom_position(custom_setup)
        elif han_formation is not None or cho_formation is not None:
            # Use separate formations if provided
            self._initialize_formation_separate(han_formation, cho_formation)
        elif formation:
            # Backward compatibility: use same formation for both sides
            self._initialize_formation(formation)
        else:
            self._initialize_starting_position()
        
        # Record initial position for repetition detection
        initial_hash = self._get_position_hash()
        self.position_history.append(initial_hash)

    def _initialize_starting_position(self):
        """Set up the starting position."""
        # Han (bottom side, rank 0-3)
        # Rank 0 (rank 1 in notation)
        self.board[0][0] = Piece(Side.HAN, PieceType.ROOK)  # a1
        self.board[0][1] = Piece(Side.HAN, PieceType.HORSE)  # b1
        self.board[0][2] = Piece(Side.HAN, PieceType.ELEPHANT)  # c1
        self.board[0][3] = Piece(Side.HAN, PieceType.GUARD)  # d1
        # e1 is empty
        self.board[0][5] = Piece(Side.HAN, PieceType.GUARD)  # f1
        self.board[0][6] = Piece(Side.HAN, PieceType.ELEPHANT)  # g1
        self.board[0][7] = Piece(Side.HAN, PieceType.HORSE)  # h1
        self.board[0][8] = Piece(Side.HAN, PieceType.ROOK)  # i1

        # Rank 1 (rank 2)
        self.board[1][4] = Piece(Side.HAN, PieceType.KING)  # e2

        # Rank 2 (rank 3)
        self.board[2][1] = Piece(Side.HAN, PieceType.CANNON)  # b3
        self.board[2][7] = Piece(Side.HAN, PieceType.CANNON)  # h3

        # Rank 3 (rank 4)
        self.board[3][0] = Piece(Side.HAN, PieceType.PAWN)  # a4
        self.board[3][2] = Piece(Side.HAN, PieceType.PAWN)  # c4
        self.board[3][4] = Piece(Side.HAN, PieceType.PAWN)  # e4
        self.board[3][6] = Piece(Side.HAN, PieceType.PAWN)  # g4
        self.board[3][8] = Piece(Side.HAN, PieceType.PAWN)  # i4

        # Cho (top side, rank 6-9)
        # Rank 9 (rank 10)
        self.board[9][0] = Piece(Side.CHO, PieceType.ROOK)  # a10
        self.board[9][1] = Piece(Side.CHO, PieceType.HORSE)  # b10
        self.board[9][2] = Piece(Side.CHO, PieceType.ELEPHANT)  # c10
        self.board[9][3] = Piece(Side.CHO, PieceType.GUARD)  # d10
        # e10 is empty
        self.board[9][5] = Piece(Side.CHO, PieceType.GUARD)  # f10
        self.board[9][6] = Piece(Side.CHO, PieceType.ELEPHANT)  # g10
        self.board[9][7] = Piece(Side.CHO, PieceType.HORSE)  # h10
        self.board[9][8] = Piece(Side.CHO, PieceType.ROOK)  # i10

        # Rank 8 (rank 9)
        self.board[8][4] = Piece(Side.CHO, PieceType.KING)  # e9

        # Rank 7 (rank 8)
        self.board[7][1] = Piece(Side.CHO, PieceType.CANNON)  # b8
        self.board[7][7] = Piece(Side.CHO, PieceType.CANNON)  # h8

        # Rank 6 (rank 7)
        self.board[6][0] = Piece(Side.CHO, PieceType.PAWN)  # a7
        self.board[6][2] = Piece(Side.CHO, PieceType.PAWN)  # c7
        self.board[6][4] = Piece(Side.CHO, PieceType.PAWN)  # e7
        self.board[6][6] = Piece(Side.CHO, PieceType.PAWN)  # g7
        self.board[6][8] = Piece(Side.CHO, PieceType.PAWN)  # i7

    def _initialize_formation(self, formation: str):
        """Initialize board with one of the 4 standard formations.
        
        Args:
            formation: One of "상마상마", "마상마상", "마상상마", "상마마상"
        """
        # Default positions (same for all formations)
        # Han rank 1
        self.board[0][0] = Piece(Side.HAN, PieceType.ROOK)  # a1
        self.board[0][3] = Piece(Side.HAN, PieceType.GUARD)  # d1
        self.board[0][5] = Piece(Side.HAN, PieceType.GUARD)  # f1
        self.board[0][8] = Piece(Side.HAN, PieceType.ROOK)  # i1
        
        # Cho rank 10
        self.board[9][0] = Piece(Side.CHO, PieceType.ROOK)  # a10
        self.board[9][3] = Piece(Side.CHO, PieceType.GUARD)  # d10
        self.board[9][5] = Piece(Side.CHO, PieceType.GUARD)  # f10
        self.board[9][8] = Piece(Side.CHO, PieceType.ROOK)  # i10
        
        # Set horses and elephants based on formation
        # Format: b1, c1, g1, h1 (left to right)
        formations = {
            "상마상마": (PieceType.ELEPHANT, PieceType.HORSE, PieceType.ELEPHANT, PieceType.HORSE),  # b1, c1, g1, h1
            "마상마상": (PieceType.HORSE, PieceType.ELEPHANT, PieceType.HORSE, PieceType.ELEPHANT),
            "마상상마": (PieceType.HORSE, PieceType.ELEPHANT, PieceType.ELEPHANT, PieceType.HORSE),
            "상마마상": (PieceType.ELEPHANT, PieceType.HORSE, PieceType.HORSE, PieceType.ELEPHANT),
        }
        
        if formation in formations:
            han_types = formations[formation]
            cho_types = formations[formation]  # Same for both sides
            
            # Han rank 1 (b1, c1, g1, h1)
            self.board[0][1] = Piece(Side.HAN, han_types[0])  # b1
            self.board[0][2] = Piece(Side.HAN, han_types[1])  # c1
            self.board[0][6] = Piece(Side.HAN, han_types[2])  # g1
            self.board[0][7] = Piece(Side.HAN, han_types[3])  # h1
            
            # Cho rank 10 (b10, c10, g10, h10)
            self.board[9][1] = Piece(Side.CHO, cho_types[0])  # b10
            self.board[9][2] = Piece(Side.CHO, cho_types[1])  # c10
            self.board[9][6] = Piece(Side.CHO, cho_types[2])  # g10
            self.board[9][7] = Piece(Side.CHO, cho_types[3])  # h10
        else:
            # Default to 마상상마 if invalid
            self._initialize_starting_position()
            return
        
        # Rest of the pieces (same for all formations)
        # Han
        self.board[1][4] = Piece(Side.HAN, PieceType.KING)  # e2
        self.board[2][1] = Piece(Side.HAN, PieceType.CANNON)  # b3
        self.board[2][7] = Piece(Side.HAN, PieceType.CANNON)  # h3
        self.board[3][0] = Piece(Side.HAN, PieceType.PAWN)  # a4
        self.board[3][2] = Piece(Side.HAN, PieceType.PAWN)  # c4
        self.board[3][4] = Piece(Side.HAN, PieceType.PAWN)  # e4
        self.board[3][6] = Piece(Side.HAN, PieceType.PAWN)  # g4
        self.board[3][8] = Piece(Side.HAN, PieceType.PAWN)  # i4
        
        # Cho
        self.board[8][4] = Piece(Side.CHO, PieceType.KING)  # e9
        self.board[7][1] = Piece(Side.CHO, PieceType.CANNON)  # b8
        self.board[7][7] = Piece(Side.CHO, PieceType.CANNON)  # h8
        self.board[6][0] = Piece(Side.CHO, PieceType.PAWN)  # a7
        self.board[6][2] = Piece(Side.CHO, PieceType.PAWN)  # c7
        self.board[6][4] = Piece(Side.CHO, PieceType.PAWN)  # e7
        self.board[6][6] = Piece(Side.CHO, PieceType.PAWN)  # g7
        self.board[6][8] = Piece(Side.CHO, PieceType.PAWN)  # i7

    def _initialize_formation_separate(self, han_formation: Optional[str], cho_formation: Optional[str]):
        """Initialize board with separate formations for HAN and CHO sides.
        
        Args:
            han_formation: Optional formation name for HAN side ("상마상마", "마상마상", "마상상마", "상마마상")
            cho_formation: Optional formation name for CHO side ("상마상마", "마상마상", "마상상마", "상마마상")
        """
        # Default positions (same for all formations)
        # Han rank 1
        self.board[0][0] = Piece(Side.HAN, PieceType.ROOK)  # a1
        self.board[0][3] = Piece(Side.HAN, PieceType.GUARD)  # d1
        self.board[0][5] = Piece(Side.HAN, PieceType.GUARD)  # f1
        self.board[0][8] = Piece(Side.HAN, PieceType.ROOK)  # i1
        
        # Cho rank 10
        self.board[9][0] = Piece(Side.CHO, PieceType.ROOK)  # a10
        self.board[9][3] = Piece(Side.CHO, PieceType.GUARD)  # d10
        self.board[9][5] = Piece(Side.CHO, PieceType.GUARD)  # f10
        self.board[9][8] = Piece(Side.CHO, PieceType.ROOK)  # i10
        
        # Set horses and elephants based on formation
        # Format: b1, c1, g1, h1 (left to right)
        formations = {
            "상마상마": (PieceType.ELEPHANT, PieceType.HORSE, PieceType.ELEPHANT, PieceType.HORSE),  # b1, c1, g1, h1
            "마상마상": (PieceType.HORSE, PieceType.ELEPHANT, PieceType.HORSE, PieceType.ELEPHANT),
            "마상상마": (PieceType.HORSE, PieceType.ELEPHANT, PieceType.ELEPHANT, PieceType.HORSE),
            "상마마상": (PieceType.ELEPHANT, PieceType.HORSE, PieceType.HORSE, PieceType.ELEPHANT),
        }
        
        # Default to 마상상마 if not provided or invalid
        default_formation = "마상상마"
        default_types = formations[default_formation]
        
        # Han formation
        if han_formation and han_formation in formations:
            han_types = formations[han_formation]
        else:
            han_types = default_types
        
        # Cho formation
        if cho_formation and cho_formation in formations:
            cho_types = formations[cho_formation]
        else:
            cho_types = default_types
        
        # Han rank 1 (b1, c1, g1, h1)
        self.board[0][1] = Piece(Side.HAN, han_types[0])  # b1
        self.board[0][2] = Piece(Side.HAN, han_types[1])  # c1
        self.board[0][6] = Piece(Side.HAN, han_types[2])  # g1
        self.board[0][7] = Piece(Side.HAN, han_types[3])  # h1
        
        # Cho rank 10 (b10, c10, g10, h10)
        self.board[9][1] = Piece(Side.CHO, cho_types[0])  # b10
        self.board[9][2] = Piece(Side.CHO, cho_types[1])  # c10
        self.board[9][6] = Piece(Side.CHO, cho_types[2])  # g10
        self.board[9][7] = Piece(Side.CHO, cho_types[3])  # h10
        
        # Rest of the pieces (same for all formations)
        # Han
        self.board[1][4] = Piece(Side.HAN, PieceType.KING)  # e2
        self.board[2][1] = Piece(Side.HAN, PieceType.CANNON)  # b3
        self.board[2][7] = Piece(Side.HAN, PieceType.CANNON)  # h3
        self.board[3][0] = Piece(Side.HAN, PieceType.PAWN)  # a4
        self.board[3][2] = Piece(Side.HAN, PieceType.PAWN)  # c4
        self.board[3][4] = Piece(Side.HAN, PieceType.PAWN)  # e4
        self.board[3][6] = Piece(Side.HAN, PieceType.PAWN)  # g4
        self.board[3][8] = Piece(Side.HAN, PieceType.PAWN)  # i4
        
        # Cho
        self.board[8][4] = Piece(Side.CHO, PieceType.KING)  # e9
        self.board[7][1] = Piece(Side.CHO, PieceType.CANNON)  # b8
        self.board[7][7] = Piece(Side.CHO, PieceType.CANNON)  # h8
        self.board[6][0] = Piece(Side.CHO, PieceType.PAWN)  # a7
        self.board[6][2] = Piece(Side.CHO, PieceType.PAWN)  # c7
        self.board[6][4] = Piece(Side.CHO, PieceType.PAWN)  # e7
        self.board[6][6] = Piece(Side.CHO, PieceType.PAWN)  # g7
        self.board[6][8] = Piece(Side.CHO, PieceType.PAWN)  # i7

    def _initialize_custom_position(self, custom_setup: Dict[str, str]):
        """Initialize board with custom piece positions.

        Args:
            custom_setup: Dictionary mapping squares (e.g., "a1") to piece codes (e.g., "hR", "cK")
        """
        # Clear board
        self.board = [[None for _ in range(self.FILES)] for _ in range(self.RANKS)]

        piece_type_map = {
            "K": PieceType.KING,
            "G": PieceType.GUARD,
            "E": PieceType.ELEPHANT,
            "H": PieceType.HORSE,
            "R": PieceType.ROOK,
            "C": PieceType.CANNON,
            "P": PieceType.PAWN,
        }

        files = "abcdefghi"
        for square, piece_code in custom_setup.items():
            if len(square) < 2:
                continue
            file_char = square[0]
            rank_str = square[1:]
            try:
                file_idx = files.index(file_char)
                rank_idx = int(rank_str) - 1
                if 0 <= file_idx < self.FILES and 0 <= rank_idx < self.RANKS:
                    side_char = piece_code[0]
                    type_char = piece_code[1] if len(piece_code) > 1 else None
                    if side_char == "h":
                        side = Side.HAN
                    elif side_char == "c":
                        side = Side.CHO
                    else:
                        continue
                    if type_char and type_char in piece_type_map:
                        self.board[rank_idx][file_idx] = Piece(
                            side, piece_type_map[type_char]
                        )
            except (ValueError, IndexError):
                continue

    def is_in_palace(self, file: int, rank: int, side: Side) -> bool:
        """Check if a square is in the palace for the given side."""
        if side == Side.HAN:
            return 3 <= file <= 5 and 0 <= rank <= 2
        else:  # CHO
            return 3 <= file <= 5 and 7 <= rank <= 9

    def is_in_any_palace(self, file: int, rank: int) -> bool:
        """Check if a square is in either palace (HAN or CHO)."""
        # HAN palace: files 3-5, ranks 0-2
        # CHO palace: files 3-5, ranks 7-9
        if 3 <= file <= 5:
            if 0 <= rank <= 2:  # HAN palace
                return True
            if 7 <= rank <= 9:  # CHO palace
                return True
        return False

    def get_palace_side(self, file: int, rank: int) -> Optional[Side]:
        """Get which side's palace contains the given position, or None if not in any palace."""
        if 3 <= file <= 5:
            if 0 <= rank <= 2:
                return Side.HAN
            if 7 <= rank <= 9:
                return Side.CHO
        return None

    def is_on_palace_diagonal_point(self, file: int, rank: int, side: Side) -> bool:
        """Check if a position is on a palace diagonal intersection point.
        
        Palace diagonal points are:
        - 4 corners of the palace
        - 1 center of the palace
        
        Han's palace diagonal points: d1 (3,0), f1 (5,0), e2 (4,1), d3 (3,2), f3 (5,2)
        Cho's palace diagonal points: d8 (3,7), f8 (5,7), e9 (4,8), d10 (3,9), f10 (5,9)
        """
        if side == Side.HAN:
            # Han palace: files 3-5, ranks 0-2
            # Diagonal points: corners (3,0), (5,0), (3,2), (5,2) and center (4,1)
            return (file, rank) in [(3, 0), (5, 0), (4, 1), (3, 2), (5, 2)]
        else:  # CHO
            # Cho palace: files 3-5, ranks 7-9
            # Diagonal points: corners (3,7), (5,7), (3,9), (5,9) and center (4,8)
            return (file, rank) in [(3, 7), (5, 7), (4, 8), (3, 9), (5, 9)]

    def is_on_any_palace_diagonal_point(self, file: int, rank: int) -> bool:
        """Check if a position is on any palace diagonal intersection point (either palace).
        
        Palace diagonal points:
        - Han: (3,0), (5,0), (4,1), (3,2), (5,2)
        - Cho: (3,7), (5,7), (4,8), (3,9), (5,9)
        """
        return (file, rank) in [
            # HAN palace diagonal points
            (3, 0), (5, 0), (4, 1), (3, 2), (5, 2),
            # CHO palace diagonal points
            (3, 7), (5, 7), (4, 8), (3, 9), (5, 9)
        ]

    def are_on_same_palace_diagonal_line(self, from_file: int, from_rank: int, to_file: int, to_rank: int) -> bool:
        """Check if two positions are on the same palace diagonal line.
        
        Palace X-diagonals:
        - HAN Line1: (3,0) - (4,1) - (5,2)  [bottom-left to top-right]
        - HAN Line2: (5,0) - (4,1) - (3,2)  [bottom-right to top-left]
        - CHO Line1: (3,7) - (4,8) - (5,9)  [bottom-left to top-right]
        - CHO Line2: (5,7) - (4,8) - (3,9)  [bottom-right to top-left]
        
        For a valid diagonal move in palace, both positions must be on the same diagonal line.
        """
        # Define the diagonal lines
        han_line1 = {(3, 0), (4, 1), (5, 2)}  # bottom-left to top-right
        han_line2 = {(5, 0), (4, 1), (3, 2)}  # bottom-right to top-left
        cho_line1 = {(3, 7), (4, 8), (5, 9)}  # bottom-left to top-right
        cho_line2 = {(5, 7), (4, 8), (3, 9)}  # bottom-right to top-left
        
        from_pos = (from_file, from_rank)
        to_pos = (to_file, to_rank)
        
        # Check if both positions are on the same diagonal line
        for line in [han_line1, han_line2, cho_line1, cho_line2]:
            if from_pos in line and to_pos in line:
                return True
        
        return False

    def is_palace_diagonal(
        self, from_file: int, from_rank: int, to_file: int, to_rank: int, side: Optional[Side] = None
    ) -> bool:
        """Check if a move is along a palace diagonal line.
        
        For Rooks and Cannons moving diagonally in palace:
        - Both positions must be on palace diagonal points
        - Both positions must be on the SAME diagonal line (X shape)
        - The move must be along that diagonal line
        
        Args:
            from_file, from_rank: Starting position
            to_file, to_rank: Destination position
            side: Side (HAN or CHO). If None, checks both palaces.
        """
        # Check if it's a diagonal move
        file_diff = abs(to_file - from_file)
        rank_diff = abs(to_rank - from_rank)
        if file_diff != rank_diff or file_diff == 0:
            return False
        
        # Both positions must be on palace diagonal points
        if not self.is_on_any_palace_diagonal_point(from_file, from_rank):
            return False
        if not self.is_on_any_palace_diagonal_point(to_file, to_rank):
            return False
        
        # Both positions must be on the same diagonal line
        if not self.are_on_same_palace_diagonal_line(from_file, from_rank, to_file, to_rank):
            return False
        
        return True

    def is_valid_palace_diagonal_for_piece(
        self, from_file: int, from_rank: int, to_file: int, to_rank: int, side: Side
    ) -> bool:
        """Check if a one-step diagonal move is valid for Guard/King.
        
        Both the starting and ending positions must be on palace diagonal points,
        and the move must be exactly one step diagonally.
        """
        # Must be a one-step diagonal move
        file_diff = abs(to_file - from_file)
        rank_diff = abs(to_rank - from_rank)
        if file_diff != 1 or rank_diff != 1:
            return False
        
        # Both positions must be on palace diagonal points
        if not self.is_on_palace_diagonal_point(from_file, from_rank, side):
            return False
        if not self.is_on_palace_diagonal_point(to_file, to_rank, side):
            return False
        
        return True

    def get_piece(self, file: int, rank: int) -> Optional[Piece]:
        """Get piece at given coordinates."""
        if 0 <= file < self.FILES and 0 <= rank < self.RANKS:
            return self.board[rank][file]
        return None

    def make_move(self, move: Move) -> bool:
        """Make a move. Returns True if legal, False otherwise."""
        # Check bounds first
        if not (0 <= move.from_file < self.FILES and 0 <= move.from_rank < self.RANKS):
            return False
        if not (0 <= move.to_file < self.FILES and 0 <= move.to_rank < self.RANKS):
            return False
        
        if not self.is_legal_move(move):
            return False
        
        piece = self.board[move.from_rank][move.from_file]
        captured_piece = self.board[move.to_rank][move.to_file]
        
        # Record move before making it
        files = "abcdefghi"
        from_square = f"{files[move.from_file]}{move.from_rank + 1}"
        to_square = f"{files[move.to_file]}{move.to_rank + 1}"
        
        piece_name_map = {
            PieceType.KING: "왕",
            PieceType.GUARD: "사",
            PieceType.ELEPHANT: "상",
            PieceType.HORSE: "마",
            PieceType.ROOK: "차",
            PieceType.CANNON: "포",
            PieceType.PAWN: "졸",
        }
        
        side_name = "한" if piece.side == Side.HAN else "초"
        piece_name = piece_name_map.get(piece.piece_type, "?")
        captured_info = ""
        if captured_piece:
            captured_side = "한" if captured_piece.side == Side.HAN else "초"
            captured_name = piece_name_map.get(captured_piece.piece_type, "?")
            captured_info = f" ({captured_side}{captured_name} 잡음)"
        
        move_record = {
            "move_number": len(self.move_history) + 1,
            "side": side_name,
            "piece": piece_name,
            "from": from_square,
            "to": to_square,
            "notation": f"{side_name}{piece_name} {from_square}→{to_square}{captured_info}",
            "captured": captured_piece is not None
        }
        
        self.board[move.to_rank][move.to_file] = piece
        self.board[move.from_rank][move.from_file] = None
        
        # Check if move leaves own king in check
        if self.is_in_check(self.side_to_move):
            # Undo move
            self.board[move.from_rank][move.from_file] = piece
            self.board[move.to_rank][move.to_file] = captured_piece
            return False
        
        # Check face-to-face kings rule (if king moved)
        if piece.piece_type == PieceType.KING:
            if self._would_face_to_face_kings(move.to_file, move.to_rank, piece.side):
                # Undo move
                self.board[move.from_rank][move.from_file] = piece
                self.board[move.to_rank][move.to_file] = captured_piece
                return False
        
        # Move is legal, record it
        self.move_history.append(move_record)
        self.side_to_move = Side.CHO if self.side_to_move == Side.HAN else Side.HAN
        
        # Record position hash for repetition detection
        position_hash = self._get_position_hash()
        self.position_history.append(position_hash)
        
        # Return captured piece for undo capability
        move.captured_piece = captured_piece
        return True

    def undo_move(self, move: Move) -> None:
        """Undo a move. Must be called immediately after make_move."""
        piece = self.board[move.to_rank][move.to_file]
        self.board[move.from_rank][move.from_file] = piece
        self.board[move.to_rank][move.to_file] = move.captured_piece
        self.side_to_move = Side.CHO if self.side_to_move == Side.HAN else Side.HAN
        if self.move_history:
            self.move_history.pop()
        # Remove last position hash
        if self.position_history:
            self.position_history.pop()

    def make_move_fast(self, move: Move) -> bool:
        """Fast make_move for AI search - skips history recording.
        
        Returns True if successful, False if invalid move.
        Sets move.captured_piece for undo.
        """
        # Validate move legality first (same as make_move)
        if not self.is_legal_move(move):
            return False
        
        piece = self.board[move.from_rank][move.from_file]
        captured_piece = self.board[move.to_rank][move.to_file]
        original_side = self.side_to_move
        
        # Make the move
        self.board[move.to_rank][move.to_file] = piece
        self.board[move.from_rank][move.from_file] = None
        self.side_to_move = Side.CHO if self.side_to_move == Side.HAN else Side.HAN
        
        # Check if move leaves own king in check
        if self.is_in_check(original_side):
            # Undo move
            self.board[move.from_rank][move.from_file] = piece
            self.board[move.to_rank][move.to_file] = captured_piece
            self.side_to_move = original_side
            return False
        
        # Check face-to-face kings rule (if king moved)
        if piece.piece_type == PieceType.KING:
            if self._would_face_to_face_kings(move.to_file, move.to_rank, piece.side):
                # Undo move
                self.board[move.from_rank][move.from_file] = piece
                self.board[move.to_rank][move.to_file] = captured_piece
                self.side_to_move = original_side
                return False
        
        # Record position hash for repetition detection
        position_hash = self._get_position_hash()
        self.position_history.append(position_hash)
        
        move.captured_piece = captured_piece
        return True

    def undo_move_fast(self, move: Move) -> None:
        """Fast undo_move for AI search - skips history."""
        piece = self.board[move.to_rank][move.to_file]
        self.board[move.from_rank][move.from_file] = piece
        self.board[move.to_rank][move.to_file] = move.captured_piece
        self.side_to_move = Side.CHO if self.side_to_move == Side.HAN else Side.HAN
        # Remove last position hash
        if self.position_history:
            self.position_history.pop()

    def is_legal_move(self, move: Move) -> bool:
        """Check if a move is legal."""
        # Check bounds
        if not (0 <= move.from_file < self.FILES and 0 <= move.from_rank < self.RANKS):
            return False
        if not (0 <= move.to_file < self.FILES and 0 <= move.to_rank < self.RANKS):
            return False
        
        piece = self.get_piece(move.from_file, move.from_rank)
        if piece is None or piece.side != self.side_to_move:
            return False

        dest_piece = self.get_piece(move.to_file, move.to_rank)
        if dest_piece is not None and dest_piece.side == piece.side:
            return False

        # Check piece-specific movement rules
        return self._is_valid_move_for_piece(move, piece)

    def _is_valid_move_for_piece(self, move: Move, piece: Piece) -> bool:
        """Check if move is valid for the specific piece type."""
        from_file, from_rank = move.from_file, move.from_rank
        to_file, to_rank = move.to_file, move.to_rank

        if piece.piece_type == PieceType.KING:
            return self._is_valid_king_move(
                from_file, from_rank, to_file, to_rank, piece.side
            )
        elif piece.piece_type == PieceType.GUARD:
            return self._is_valid_guard_move(
                from_file, from_rank, to_file, to_rank, piece.side
            )
        elif piece.piece_type == PieceType.ELEPHANT:
            return self._is_valid_elephant_move(from_file, from_rank, to_file, to_rank)
        elif piece.piece_type == PieceType.HORSE:
            return self._is_valid_horse_move(from_file, from_rank, to_file, to_rank)
        elif piece.piece_type == PieceType.ROOK:
            return self._is_valid_rook_move(from_file, from_rank, to_file, to_rank)
        elif piece.piece_type == PieceType.CANNON:
            return self._is_valid_cannon_move(from_file, from_rank, to_file, to_rank)
        elif piece.piece_type == PieceType.PAWN:
            return self._is_valid_pawn_move(
                from_file, from_rank, to_file, to_rank, piece.side
            )
        return False

    def _is_valid_king_move(
        self, from_file: int, from_rank: int, to_file: int, to_rank: int, side: Side
    ) -> bool:
        """Check if king move is valid.
        
        King rules:
        - Must remain inside own palace
        - Can move one step orthogonally (always valid in palace)
        - Can move diagonally ONLY when on a palace diagonal point
        """
        if not self.is_in_palace(to_file, to_rank, side):
            return False

        file_diff = abs(to_file - from_file)
        rank_diff = abs(to_rank - from_rank)

        # Orthogonal move (always valid in palace)
        if (file_diff == 1 and rank_diff == 0) or (file_diff == 0 and rank_diff == 1):
            # Check face-to-face kings rule after move
            if self._would_face_to_face_kings(to_file, to_rank, side):
                return False
            return True

        # Diagonal move: only valid if both positions are on palace diagonal points
        if file_diff == 1 and rank_diff == 1:
            if not self.is_valid_palace_diagonal_for_piece(from_file, from_rank, to_file, to_rank, side):
                return False
            # Check face-to-face kings rule after move
            if self._would_face_to_face_kings(to_file, to_rank, side):
                return False
            return True

        return False

    def _would_face_to_face_kings(self, king_file: int, king_rank: int, king_side: Side) -> bool:
        """Check if placing king at this position would create face-to-face kings.
        
        Face-to-face kings: Both kings on same file with no pieces between them.
        """
        # Find the other king
        other_side = Side.CHO if king_side == Side.HAN else Side.HAN
        other_king_file, other_king_rank = None, None
        
        for rank in range(self.RANKS):
            for file in range(self.FILES):
                piece = self.get_piece(file, rank)
                if piece and piece.side == other_side and piece.piece_type == PieceType.KING:
                    other_king_file, other_king_rank = file, rank
                    break
            if other_king_file is not None:
                break
        
        if other_king_file is None:
            return False
        
        # Check if both kings are on the same file
        if king_file != other_king_file:
            return False
        
        # Check if there are any pieces between them
        min_rank = min(king_rank, other_king_rank)
        max_rank = max(king_rank, other_king_rank)
        
        for rank in range(min_rank + 1, max_rank):
            if self.get_piece(king_file, rank) is not None:
                return False  # There's a piece between them, so not face-to-face
        
        return True  # Face-to-face kings!

    def _is_valid_guard_move(
        self, from_file: int, from_rank: int, to_file: int, to_rank: int, side: Side
    ) -> bool:
        """Check if guard move is valid.
        
        Guard rules:
        - Must remain inside own palace
        - Can move one step forward, backward, left, or right (orthogonal)
        - Can move diagonally ONLY when on a palace diagonal point (corners or center)
          and moving to another diagonal point
        """
        if not self.is_in_palace(to_file, to_rank, side):
            return False

        file_diff = abs(to_file - from_file)
        rank_diff = abs(to_rank - from_rank)

        # Must move exactly one step
        if file_diff > 1 or rank_diff > 1:
            return False
        
        # Must move at least one step (not staying in place)
        if file_diff == 0 and rank_diff == 0:
            return False
        
        # Orthogonal moves (forward, backward, left, right) are always valid in palace
        if file_diff == 0 or rank_diff == 0:
            return True
        
        # Diagonal moves: only valid if BOTH positions are on palace diagonal points
        # Diagonal points: 4 corners + center of palace
        # Han: d1 (3,0), f1 (5,0), e2 (4,1), d3 (3,2), f3 (5,2)
        # Cho: d8 (3,7), f8 (5,7), e9 (4,8), d10 (3,9), f10 (5,9)
        if file_diff == 1 and rank_diff == 1:
            return self.is_valid_palace_diagonal_for_piece(from_file, from_rank, to_file, to_rank, side)
        
        return False

    def _is_valid_elephant_move(
        self, from_file: int, from_rank: int, to_file: int, to_rank: int
    ) -> bool:
        """Check if elephant move is valid.

        Elephant moves: 1 orthogonal step + 2 diagonal steps in SAME direction.
        Path: (file, rank) -> (orth_file, orth_rank) -> (diag1_file, diag1_rank) -> (to_file, to_rank)
        
        Rules:
        - First move one step orthogonally (up, down, left, or right)
        - Then from there move two steps diagonally
        - IMPORTANT: The diagonal direction must EXTEND from the orthogonal direction
          - Right orthogonal → up-right or down-right diagonal
          - Left orthogonal → up-left or down-left diagonal  
          - Up orthogonal → up-right or up-left diagonal
          - Down orthogonal → down-right or down-left diagonal
        - The first orthogonal step AND the first diagonal step must be clear
        """
        # Map orthogonal directions to valid diagonal directions
        # Diagonal direction must share a component with orthogonal direction
        orth_to_diag = {
            (1, 0): [(1, 1), (1, -1)],    # right → up-right, down-right
            (-1, 0): [(-1, 1), (-1, -1)], # left → up-left, down-left
            (0, 1): [(1, 1), (-1, 1)],    # up → up-right, up-left
            (0, -1): [(1, -1), (-1, -1)], # down → down-right, down-left
        }
        
        for orth_dir, valid_diags in orth_to_diag.items():
            orth_df, orth_dr = orth_dir
            
            # First step: orthogonal
            orth_file = from_file + orth_df
            orth_rank = from_rank + orth_dr
            
            # Check bounds
            if not (0 <= orth_file < self.FILES and 0 <= orth_rank < self.RANKS):
                continue
            
            # Check if first orthogonal step is blocked
            if self.get_piece(orth_file, orth_rank) is not None:
                continue
            
            # From orthogonal position, try ONLY valid diagonal directions
            for diag_df, diag_dr in valid_diags:
                # First diagonal step
                diag1_file = orth_file + diag_df
                diag1_rank = orth_rank + diag_dr
                
                # Check bounds
                if not (0 <= diag1_file < self.FILES and 0 <= diag1_rank < self.RANKS):
                    continue
                
                # Check if first diagonal step is blocked
                if self.get_piece(diag1_file, diag1_rank) is not None:
                    continue
                
                # Second diagonal step
                diag2_file = diag1_file + diag_df
                diag2_rank = diag1_rank + diag_dr
                
                # Check if this matches the destination
                if diag2_file == to_file and diag2_rank == to_rank:
                    return True
        
        return False

    def _is_valid_horse_move(
        self, from_file: int, from_rank: int, to_file: int, to_rank: int
    ) -> bool:
        """Check if horse move is valid."""
        file_diff = to_file - from_file
        rank_diff = to_rank - from_rank

        # L-shaped: 1 orthogonal + 1 diagonal
        if abs(file_diff) == 1 and abs(rank_diff) == 2:
            # Check if leg is blocked
            leg_rank = from_rank + (1 if rank_diff > 0 else -1)
            if self.get_piece(from_file, leg_rank) is not None:
                return False
            return True
        elif abs(file_diff) == 2 and abs(rank_diff) == 1:
            # Check if leg is blocked
            leg_file = from_file + (1 if file_diff > 0 else -1)
            if self.get_piece(leg_file, from_rank) is not None:
                return False
            return True

        return False

    def _is_valid_rook_move(
        self, from_file: int, from_rank: int, to_file: int, to_rank: int
    ) -> bool:
        """Check if rook move is valid.
        
        Rook can move:
        - Orthogonally any number of squares (path must be clear)
        - Diagonally in ANY palace along X-diagonal lines (path must be clear)
        """
        file_diff = to_file - from_file
        rank_diff = to_rank - from_rank

        # Orthogonal move
        if file_diff == 0 and rank_diff != 0:
            step = 1 if rank_diff > 0 else -1
            for r in range(from_rank + step, to_rank, step):
                if self.get_piece(from_file, r) is not None:
                    return False
            return True
        elif rank_diff == 0 and file_diff != 0:
            step = 1 if file_diff > 0 else -1
            for f in range(from_file + step, to_file, step):
                if self.get_piece(f, from_rank) is not None:
                    return False
            return True

        # Palace diagonal move - uses updated is_palace_diagonal that checks both palaces
        if self.is_palace_diagonal(from_file, from_rank, to_file, to_rank):
            file_step = 1 if file_diff > 0 else -1
            rank_step = 1 if rank_diff > 0 else -1
            f, r = from_file + file_step, from_rank + rank_step
            while f != to_file and r != to_rank:
                if self.get_piece(f, r) is not None:
                    return False
                # Intermediate positions must also be on the same diagonal line
                if not self.is_on_any_palace_diagonal_point(f, r):
                    return False
                f += file_step
                r += rank_step
            return True

        return False

    def _is_valid_cannon_move(
        self, from_file: int, from_rank: int, to_file: int, to_rank: int
    ) -> bool:
        """Check if cannon move is valid.
        
        Cannon rules:
        - Must jump over exactly one non-Cannon piece (screen)
        - Move to empty square: destination must be empty
        - Capture: destination must have enemy piece (not Cannon)
        - Can move diagonally in ANY palace along X-diagonal lines
        """
        file_diff = to_file - from_file
        rank_diff = to_rank - from_rank

        # Determine if this is an orthogonal or diagonal move
        is_orthogonal = (file_diff == 0 and rank_diff != 0) or (rank_diff == 0 and file_diff != 0)
        is_diagonal = abs(file_diff) == abs(rank_diff) and file_diff != 0
        
        if not is_orthogonal and not is_diagonal:
            return False
        
        # For diagonal moves, must be along palace diagonal
        if is_diagonal:
            if not self.is_palace_diagonal(from_file, from_rank, to_file, to_rank):
                return False

        dest_piece = self.get_piece(to_file, to_rank)
        
        # Count pieces between start and end
        pieces_between = 0
        if file_diff == 0:
            step = 1 if rank_diff > 0 else -1
            for r in range(from_rank + step, to_rank, step):
                p = self.get_piece(from_file, r)
                if p is not None:
                    if p.piece_type == PieceType.CANNON:
                        return False  # Cannot jump over cannon
                    pieces_between += 1
        elif rank_diff == 0:
            step = 1 if file_diff > 0 else -1
            for f in range(from_file + step, to_file, step):
                p = self.get_piece(f, from_rank)
                if p is not None:
                    if p.piece_type == PieceType.CANNON:
                        return False
                    pieces_between += 1
        else:  # Diagonal (palace diagonal)
            file_step = 1 if file_diff > 0 else -1
            rank_step = 1 if rank_diff > 0 else -1
            f, r = from_file + file_step, from_rank + rank_step
            while f != to_file and r != to_rank:
                # Intermediate positions must also be on the same diagonal line
                if not self.is_on_any_palace_diagonal_point(f, r):
                    return False
                p = self.get_piece(f, r)
                if p is not None:
                    if p.piece_type == PieceType.CANNON:
                        return False
                    pieces_between += 1
                f += file_step
                r += rank_step

        # Must have exactly one piece between
        if pieces_between != 1:
            return False
        
        # Check destination rules
        cannon_piece = self.get_piece(from_file, from_rank)
        if cannon_piece is None:
            return False
        
        if dest_piece is None:
            # Moving to empty square - this is legal
            return True
        elif dest_piece.side != cannon_piece.side:
            # Capturing enemy piece - check if it's a Cannon
            # Cannons cannot capture other Cannons
            if dest_piece.piece_type == PieceType.CANNON:
                return False
            # Capturing non-Cannon enemy piece - this is legal
            return True
        else:
            # Friendly piece at destination - illegal
            return False

    def _is_valid_pawn_move(
        self, from_file: int, from_rank: int, to_file: int, to_rank: int, side: Side
    ) -> bool:
        """Check if pawn move is valid."""
        file_diff = to_file - from_file
        rank_diff = to_rank - from_rank

        forward = 1 if side == Side.HAN else -1

        # Cannot move backward
        if (side == Side.HAN and rank_diff < 0) or (side == Side.CHO and rank_diff > 0):
            return False

        # At enemy's last rank, can only move sideways (no forward, no diagonal forward)
        if (side == Side.HAN and from_rank == 9) or (
            side == Side.CHO and from_rank == 0
        ):
            # Can only move left or right, not forward
            if rank_diff != 0:
                return False  # Cannot move forward or backward
            return file_diff != 0 and abs(file_diff) == 1  # Only sideways

        # Normal moves: forward or sideways
        if rank_diff == forward and file_diff == 0:
            return True
        if rank_diff == 0 and abs(file_diff) == 1:
            return True

        # Diagonal forward in palace
        if self.is_in_palace(from_file, from_rank, side) and self.is_in_palace(
            to_file, to_rank, side
        ):
            if rank_diff == forward and abs(file_diff) == 1:
                return True

        return False

    def generate_moves(self) -> List[Move]:
        """Generate all legal moves for the side to move."""
        moves = []
        for rank in range(self.RANKS):
            for file in range(self.FILES):
                piece = self.get_piece(file, rank)
                if piece is not None and piece.side == self.side_to_move:
                    moves.extend(self._generate_moves_for_piece(file, rank, piece))

        # Filter out moves that leave own king in check
        legal_moves = []
        for move in moves:
            if self._would_be_legal(move):
                legal_moves.append(move)

        return legal_moves

    def _generate_moves_for_piece(
        self, file: int, rank: int, piece: Piece
    ) -> List[Move]:
        """Generate candidate moves for a piece."""
        moves = []

        if piece.piece_type == PieceType.KING:
            moves.extend(self._generate_king_moves(file, rank, piece.side))
        elif piece.piece_type == PieceType.GUARD:
            moves.extend(self._generate_guard_moves(file, rank, piece.side))
        elif piece.piece_type == PieceType.ELEPHANT:
            moves.extend(self._generate_elephant_moves(file, rank))
        elif piece.piece_type == PieceType.HORSE:
            moves.extend(self._generate_horse_moves(file, rank))
        elif piece.piece_type == PieceType.ROOK:
            moves.extend(self._generate_rook_moves(file, rank))
        elif piece.piece_type == PieceType.CANNON:
            moves.extend(self._generate_cannon_moves(file, rank))
        elif piece.piece_type == PieceType.PAWN:
            moves.extend(self._generate_pawn_moves(file, rank, piece.side))

        return moves

    def _generate_king_moves(self, file: int, rank: int, side: Side) -> List[Move]:
        """Generate king moves.
        
        King can move within the palace:
        - Orthogonally (up, down, left, right) - always allowed
        - Diagonally - ONLY when on a palace diagonal point (corners or center)
        """
        moves = []
        
        # Orthogonal directions (always valid within palace)
        orthogonal_directions = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
        ]

        for df, dr in orthogonal_directions:
            to_file, to_rank = file + df, rank + dr
            if self.is_in_palace(to_file, to_rank, side):
                moves.append(Move(file, rank, to_file, to_rank))

        # Diagonal moves: only if current position is on a palace diagonal point
        if self.is_on_palace_diagonal_point(file, rank, side):
            diagonal_directions = [
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]
            for df, dr in diagonal_directions:
                to_file, to_rank = file + df, rank + dr
                if self.is_in_palace(to_file, to_rank, side):
                    # Check if destination is also on a palace diagonal point
                    if self.is_on_palace_diagonal_point(to_file, to_rank, side):
                        moves.append(Move(file, rank, to_file, to_rank))

        return moves

    def _generate_guard_moves(self, file: int, rank: int, side: Side) -> List[Move]:
        """Generate guard moves.
        
        Guard can move within the palace:
        - Forward, backward, left, right (orthogonal) - one step (always allowed in palace)
        - Diagonally - ONLY when on a palace diagonal point (corners or center)
        """
        moves = []
        # Orthogonal directions: 4 directions
        orthogonal_directions = [
            (0, 1),   # forward (up)
            (0, -1),  # backward (down)
            (1, 0),   # right
            (-1, 0),  # left
        ]

        # Add orthogonal moves (always valid within palace)
        for df, dr in orthogonal_directions:
            to_file, to_rank = file + df, rank + dr
            if 0 <= to_file < self.FILES and 0 <= to_rank < self.RANKS:
                if self.is_in_palace(to_file, to_rank, side):
                    moves.append(Move(file, rank, to_file, to_rank))

        # Diagonal moves: only if current position is on a palace diagonal point
        if self.is_on_palace_diagonal_point(file, rank, side):
            diagonal_directions = [
                (1, 1),   # diagonal: forward-right
                (1, -1),  # diagonal: backward-right
                (-1, 1),  # diagonal: forward-left
                (-1, -1), # diagonal: backward-left
            ]
            for df, dr in diagonal_directions:
                to_file, to_rank = file + df, rank + dr
                if 0 <= to_file < self.FILES and 0 <= to_rank < self.RANKS:
                    if self.is_in_palace(to_file, to_rank, side):
                        # Check if destination is also on a palace diagonal point
                        if self.is_on_palace_diagonal_point(to_file, to_rank, side):
                            moves.append(Move(file, rank, to_file, to_rank))

        return moves

    def _generate_elephant_moves(self, file: int, rank: int) -> List[Move]:
        """Generate elephant moves.
        
        Elephant moves: 1 orthogonal + 2 diagonal steps in SAME direction.
        The diagonal direction must extend from the orthogonal direction.
        """
        moves = []
        
        # Map orthogonal directions to valid diagonal directions
        # Diagonal direction must share a component with orthogonal direction
        orth_to_diag = {
            (1, 0): [(1, 1), (1, -1)],    # right → up-right, down-right
            (-1, 0): [(-1, 1), (-1, -1)], # left → up-left, down-left
            (0, 1): [(1, 1), (-1, 1)],    # up → up-right, up-left
            (0, -1): [(1, -1), (-1, -1)], # down → down-right, down-left
        }
        
        for orth_dir, valid_diags in orth_to_diag.items():
            orth_df, orth_dr = orth_dir
            
            # First step: orthogonal
            orth_file = file + orth_df
            orth_rank = rank + orth_dr
            
            # Check bounds
            if not (0 <= orth_file < self.FILES and 0 <= orth_rank < self.RANKS):
                continue
            
            # Check if first orthogonal step is blocked
            if self.get_piece(orth_file, orth_rank) is not None:
                continue
            
            # From orthogonal position, try ONLY valid diagonal directions
            for diag_df, diag_dr in valid_diags:
                # First diagonal step
                diag1_file = orth_file + diag_df
                diag1_rank = orth_rank + diag_dr
                
                # Check bounds
                if not (0 <= diag1_file < self.FILES and 0 <= diag1_rank < self.RANKS):
                    continue
                
                # Check if first diagonal step is blocked
                if self.get_piece(diag1_file, diag1_rank) is not None:
                    continue
                
                # Second diagonal step (final destination)
                to_file = diag1_file + diag_df
                to_rank = diag1_rank + diag_dr
                
                # Check bounds
                if 0 <= to_file < self.FILES and 0 <= to_rank < self.RANKS:
                    moves.append(Move(file, rank, to_file, to_rank))

        return moves

    def _generate_horse_moves(self, file: int, rank: int) -> List[Move]:
        """Generate horse moves."""
        moves = []
        patterns = [
            (1, 2),
            (-1, 2),
            (1, -2),
            (-1, -2),
            (2, 1),
            (-2, 1),
            (2, -1),
            (-2, -1),
        ]

        for df, dr in patterns:
            to_file, to_rank = file + df, rank + dr
            if 0 <= to_file < self.FILES and 0 <= to_rank < self.RANKS:
                if self._is_valid_horse_move(file, rank, to_file, to_rank):
                    moves.append(Move(file, rank, to_file, to_rank))

        return moves

    def _generate_rook_moves(self, file: int, rank: int) -> List[Move]:
        """Generate rook moves.
        
        Rook can move:
        - Orthogonally (up, down, left, right) any number of squares
        - Diagonally in ANY palace (not just own palace) along the X-diagonal lines
        """
        moves = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for df, dr in directions:
            for dist in range(1, max(self.FILES, self.RANKS)):
                to_file, to_rank = file + df * dist, rank + dr * dist
                if not (0 <= to_file < self.FILES and 0 <= to_rank < self.RANKS):
                    break
                moves.append(Move(file, rank, to_file, to_rank))
                if self.get_piece(to_file, to_rank) is not None:
                    break

        # Palace diagonals - can move diagonally in ANY palace (not just own palace)
        # Only if current position is on a palace diagonal point
        if self.is_on_any_palace_diagonal_point(file, rank):
            diag_dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            for df, dr in diag_dirs:
                for dist in range(1, 3):  # Max 2 squares diagonal in palace
                    to_file, to_rank = file + df * dist, rank + dr * dist
                    if not (0 <= to_file < self.FILES and 0 <= to_rank < self.RANKS):
                        break
                    # Destination must also be on a palace diagonal point
                    if not self.is_on_any_palace_diagonal_point(to_file, to_rank):
                        break
                    # Both positions must be on the same diagonal line
                    if not self.are_on_same_palace_diagonal_line(file, rank, to_file, to_rank):
                        break
                    moves.append(Move(file, rank, to_file, to_rank))
                    if self.get_piece(to_file, to_rank) is not None:
                        break

        return moves

    def _generate_cannon_moves(self, file: int, rank: int) -> List[Move]:
        """Generate cannon moves.
        
        Cannon rules:
        - Must jump over exactly one non-Cannon piece (screen)
        - Move to empty square: destination must be empty
        - Capture: destination must have enemy piece (not Cannon)
        - Can move diagonally in ANY palace along the X-diagonal lines
        """
        moves = []
        piece = self.get_piece(file, rank)
        if piece is None:
            return moves
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for df, dr in directions:
            screen_found = False
            for dist in range(1, max(self.FILES, self.RANKS)):
                to_file, to_rank = file + df * dist, rank + dr * dist
                if not (0 <= to_file < self.FILES and 0 <= to_rank < self.RANKS):
                    break

                piece_at = self.get_piece(to_file, to_rank)
                if not screen_found:
                    # Looking for screen (non-Cannon piece)
                    if piece_at is not None and piece_at.piece_type != PieceType.CANNON:
                        screen_found = True
                    # Empty square or Cannon - continue searching
                    continue
                else:
                    # Screen found, now check destination
                    if piece_at is None:
                        # Empty square - legal move
                        moves.append(Move(file, rank, to_file, to_rank))
                    elif piece_at.side != piece.side:
                        # Enemy piece - check if it's a Cannon
                        # Cannons cannot capture other Cannons
                        if piece_at.piece_type != PieceType.CANNON:
                            # Capturing non-Cannon enemy piece - legal capture
                            moves.append(Move(file, rank, to_file, to_rank))
                        # Cannot capture Cannon, stop searching this direction
                        break  # Cannot jump over enemy piece
                    else:
                        # Friendly piece - cannot move here, stop searching this direction
                        break

        # Palace diagonals - can move diagonally in ANY palace (not just own palace)
        # Only if current position is on a palace diagonal point
        if self.is_on_any_palace_diagonal_point(file, rank):
            diag_dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            for df, dr in diag_dirs:
                screen_found = False
                for dist in range(1, 3):  # Max 2 squares diagonal in palace
                    to_file, to_rank = file + df * dist, rank + dr * dist
                    if not (0 <= to_file < self.FILES and 0 <= to_rank < self.RANKS):
                        break
                    # Destination must also be on a palace diagonal point
                    if not self.is_on_any_palace_diagonal_point(to_file, to_rank):
                        break
                    # Both positions must be on the same diagonal line
                    if not self.are_on_same_palace_diagonal_line(file, rank, to_file, to_rank):
                        break

                    piece_at = self.get_piece(to_file, to_rank)
                    if not screen_found:
                        # Looking for screen
                        if (
                            piece_at is not None
                            and piece_at.piece_type != PieceType.CANNON
                        ):
                            screen_found = True
                        # Empty square or Cannon - continue searching
                        continue
                    else:
                        # Screen found, check destination
                        if piece_at is None:
                            # Empty square - legal move
                            moves.append(Move(file, rank, to_file, to_rank))
                        elif piece_at.side != piece.side:
                            # Enemy piece - check if it's a Cannon
                            # Cannons cannot capture other Cannons
                            if piece_at.piece_type != PieceType.CANNON:
                                # Capturing non-Cannon enemy piece - legal capture
                                moves.append(Move(file, rank, to_file, to_rank))
                            # Cannot capture Cannon, stop searching this direction
                            break  # Cannot jump over enemy piece
                        else:
                            # Friendly piece - stop searching this direction
                            break

        return moves

    def _generate_pawn_moves(self, file: int, rank: int, side: Side) -> List[Move]:
        """Generate pawn moves."""
        moves = []
        forward = 1 if side == Side.HAN else -1

        # Check if at enemy's last rank
        at_enemy_last_rank = (side == Side.HAN and rank == 9) or (side == Side.CHO and rank == 0)
        
        if not at_enemy_last_rank:
            # Forward (only if not at enemy's last rank)
            if 0 <= rank + forward < self.RANKS:
                moves.append(Move(file, rank, file, rank + forward))

        # Sideways (always allowed)
        for df in [-1, 1]:
            if 0 <= file + df < self.FILES:
                moves.append(Move(file, rank, file + df, rank))

        # Diagonal forward in palace (only if not at enemy's last rank)
        if not at_enemy_last_rank and self.is_in_palace(file, rank, side):
            for df in [-1, 1]:
                to_file, to_rank = file + df, rank + forward
                if 0 <= to_file < self.FILES and 0 <= to_rank < self.RANKS:
                    if self.is_in_palace(to_file, to_rank, side):
                        moves.append(Move(file, rank, to_file, to_rank))

        return moves

    def _would_be_legal(self, move: Move) -> bool:
        """Check if move would be legal (including check check and face-to-face kings)."""
        # Check bounds
        if not (0 <= move.from_file < self.FILES and 0 <= move.from_rank < self.RANKS):
            return False
        if not (0 <= move.to_file < self.FILES and 0 <= move.to_rank < self.RANKS):
            return False
        
        # Save state
        piece = self.board[move.from_rank][move.from_file]
        if piece is None:
            return False
        
        dest_piece = self.board[move.to_rank][move.to_file]
        
        # Cannot capture own piece
        if dest_piece is not None and dest_piece.side == piece.side:
            return False
        
        side = self.side_to_move
        
        # Make move temporarily
        self.board[move.to_rank][move.to_file] = piece
        self.board[move.from_rank][move.from_file] = None
        self.side_to_move = Side.CHO if self.side_to_move == Side.HAN else Side.HAN
        
        # Check if own king is in check
        try:
            in_check = self.is_in_check(side)
        except Exception:
            # Restore state on error
            self.board[move.from_rank][move.from_file] = piece
            self.board[move.to_rank][move.to_file] = dest_piece
            self.side_to_move = side
            return False
        
        # Check face-to-face kings rule (if king moved)
        face_to_face = False
        if piece.piece_type == PieceType.KING:
            try:
                face_to_face = self._would_face_to_face_kings(move.to_file, move.to_rank, piece.side)
            except Exception:
                # Restore state on error
                self.board[move.from_rank][move.from_file] = piece
                self.board[move.to_rank][move.to_file] = dest_piece
                self.side_to_move = side
                return False
        
        # Restore state
        self.board[move.from_rank][move.from_file] = piece
        self.board[move.to_rank][move.to_file] = dest_piece
        self.side_to_move = side
        
        return not in_check and not face_to_face

    def is_in_check(self, side: Side) -> bool:
        """Check if the given side's king is in check."""
        # Find king
        king_file, king_rank = None, None
        for rank in range(self.RANKS):
            for file in range(self.FILES):
                piece = self.get_piece(file, rank)
                if piece and piece.side == side and piece.piece_type == PieceType.KING:
                    king_file, king_rank = file, rank
                    break
            if king_file is not None:
                break

        if king_file is None:
            return False

        # Check if any enemy piece can attack the king
        enemy_side = Side.CHO if side == Side.HAN else Side.HAN

        # Check each enemy piece to see if it can attack the king
        # We don't need to change side_to_move - we directly check if the move is valid
        for rank in range(self.RANKS):
            for file in range(self.FILES):
                piece = self.get_piece(file, rank)
                if piece and piece.side == enemy_side:
                    move = Move(file, rank, king_file, king_rank)
                    # Directly check if the move is valid for the piece type
                    # This doesn't depend on side_to_move
                    if self._is_valid_move_for_piece(move, piece):
                        return True

        return False

    def is_checkmate(self) -> bool:
        """Check if current side is in checkmate."""
        if not self.is_in_check(self.side_to_move):
            return False
        return len(self.generate_moves()) == 0

    def is_stalemate(self) -> bool:
        """Check if current side is in stalemate."""
        if self.is_in_check(self.side_to_move):
            return False
        return len(self.generate_moves()) == 0

    def is_draw_by_repetition(self) -> bool:
        """Check if the game is a draw by threefold repetition.
        
        According to Janggi rules, if the same position (board state + side to move)
        repeats 3 times, the game is a draw.
        """
        return self.is_repetition(count=3)

    def get_king_position(self, side: Side) -> Optional[Tuple[int, int]]:
        """Get king position for given side."""
        for rank in range(self.RANKS):
            for file in range(self.FILES):
                piece = self.get_piece(file, rank)
                if piece and piece.side == side and piece.piece_type == PieceType.KING:
                    return (file, rank)
        return None

    def _get_position_hash(self) -> str:
        """Generate a hash of the current board position including side to move.
        
        This hash uniquely identifies a position (board state + side to move).
        Used for repetition detection.
        """
        parts = []
        # Add all piece positions
        for rank in range(self.RANKS):
            for file in range(self.FILES):
                piece = self.board[rank][file]
                if piece:
                    parts.append(f"{file}{rank}{piece.side.value[0]}{piece.piece_type.value[0]}")
        # Add side to move (critical for repetition detection)
        parts.append(self.side_to_move.value[0])
        return "|".join(parts)

    def is_repetition(self, count: int = 3) -> bool:
        """Check if the current position has been repeated a certain number of times.
        
        Args:
            count: Number of repetitions required (default: 3 for threefold repetition)
            
        Returns:
            True if the position has been repeated at least 'count' times, False otherwise.
            
        Note:
            This method should be called AFTER a move has been made and the position
            hash has been added to position_history. The current position hash is
            already in the history, so we check if it appears 'count' times total.
        """
        if len(self.position_history) < count:
            return False
        
        current_hash = self._get_position_hash()
        # Count how many times the current position appears in history
        # Since the current position is already recorded in position_history,
        # we check if it appears 'count' times total
        repetition_count = self.position_history.count(current_hash)
        return repetition_count >= count

    def to_fen(self) -> str:
        """Convert board to FEN-like string."""
        # Simplified FEN representation
        fen_parts = []
        for rank in range(self.RANKS - 1, -1, -1):
            rank_str = ""
            empty_count = 0
            for file in range(self.FILES):
                piece = self.board[rank][file]
                if piece is None:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        rank_str += str(empty_count)
                        empty_count = 0
                    side_char = "h" if piece.side == Side.HAN else "c"
                    type_char = piece.piece_type.value[0].lower()
                    rank_str += side_char + type_char
            if empty_count > 0:
                rank_str += str(empty_count)
            fen_parts.append(rank_str)

        side_char = "c" if self.side_to_move == Side.CHO else "h"
        return "/".join(fen_parts) + f" {side_char}"
