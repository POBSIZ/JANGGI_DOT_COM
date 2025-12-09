"""Bitboard representation for Janggi board.

A bitboard uses 90 bits (9 files × 10 ranks) to represent piece positions.
Each bit corresponds to a square on the board:
- Bit 0: a1 (file=0, rank=0)
- Bit 8: i1 (file=8, rank=0)
- Bit 9: a2 (file=0, rank=1)
- ...
- Bit 89: i10 (file=8, rank=9)

This enables fast position queries using bitwise operations.
"""

from typing import Dict, Tuple, List, Optional, Iterator
from enum import Enum
from dataclasses import dataclass


# Pre-computed masks for files and ranks
FILE_MASKS: List[int] = []
RANK_MASKS: List[int] = []

# Initialize file masks (9 files, each a column)
for file_idx in range(9):
    mask = 0
    for rank_idx in range(10):
        mask |= 1 << (rank_idx * 9 + file_idx)
    FILE_MASKS.append(mask)

# Initialize rank masks (10 ranks, each a row)
for rank_idx in range(10):
    mask = 0
    for file_idx in range(9):
        mask |= 1 << (rank_idx * 9 + file_idx)
    RANK_MASKS.append(mask)

# Palace masks (3×3 areas)
HAN_PALACE_MASK = 0
CHO_PALACE_MASK = 0

# HAN palace: files 3-5, ranks 0-2
for rank_idx in range(3):
    for file_idx in range(3, 6):
        HAN_PALACE_MASK |= 1 << (rank_idx * 9 + file_idx)

# CHO palace: files 3-5, ranks 7-9
for rank_idx in range(7, 10):
    for file_idx in range(3, 6):
        CHO_PALACE_MASK |= 1 << (rank_idx * 9 + file_idx)

# Palace diagonal points
HAN_PALACE_DIAGONAL_POINTS = frozenset([
    (3, 0), (5, 0), (4, 1), (3, 2), (5, 2)
])
CHO_PALACE_DIAGONAL_POINTS = frozenset([
    (3, 7), (5, 7), (4, 8), (3, 9), (5, 9)
])

# All board squares mask
ALL_SQUARES_MASK = (1 << 90) - 1


def square_to_bit(file: int, rank: int) -> int:
    """Convert file, rank to bit position."""
    return rank * 9 + file


def bit_to_square(bit: int) -> Tuple[int, int]:
    """Convert bit position to file, rank."""
    return bit % 9, bit // 9


def set_bit(bb: int, file: int, rank: int) -> int:
    """Set a bit at the given position."""
    return bb | (1 << square_to_bit(file, rank))


def clear_bit(bb: int, file: int, rank: int) -> int:
    """Clear a bit at the given position."""
    return bb & ~(1 << square_to_bit(file, rank))


def is_bit_set(bb: int, file: int, rank: int) -> bool:
    """Test if a bit is set at the given position."""
    return bool(bb & (1 << square_to_bit(file, rank)))


def popcount(bb: int) -> int:
    """Count the number of set bits (population count)."""
    return bin(bb).count('1')


def lsb_index(bb: int) -> int:
    """Get the index of the least significant bit."""
    if bb == 0:
        return -1
    return (bb & -bb).bit_length() - 1


def iter_bits(bb: int) -> Iterator[int]:
    """Iterate over all set bit indices."""
    while bb:
        idx = lsb_index(bb)
        yield idx
        bb &= bb - 1  # Clear the LSB


def iter_squares(bb: int) -> Iterator[Tuple[int, int]]:
    """Iterate over all set squares (file, rank)."""
    for bit_idx in iter_bits(bb):
        yield bit_to_square(bit_idx)


# Pre-computed attack tables
class AttackTables:
    """Pre-computed attack patterns for all pieces."""
    
    # King/Guard orthogonal moves (1 step)
    KING_ORTHOGONAL: List[int] = [0] * 90
    
    # King/Guard diagonal moves in palace
    KING_DIAGONAL_HAN: List[int] = [0] * 90
    KING_DIAGONAL_CHO: List[int] = [0] * 90
    
    # Pawn moves (forward and sideways)
    PAWN_HAN: List[int] = [0] * 90  # HAN moves up (rank increases)
    PAWN_CHO: List[int] = [0] * 90  # CHO moves down (rank decreases)
    
    # Horse moves (8 L-shaped destinations)
    HORSE_MOVES: List[int] = [0] * 90
    HORSE_LEGS: List[List[Tuple[int, int, int]]] = [[] for _ in range(90)]  # leg square for each move
    
    # Elephant moves (8 destinations: 1 orth + 2 diag)
    ELEPHANT_MOVES: List[int] = [0] * 90
    ELEPHANT_LEGS: List[List[Tuple[int, int, int, int, int]]] = [[] for _ in range(90)]  # (dest_file, dest_rank, leg1_f, leg1_r, leg2_f, leg2_r)
    
    # Rook rays (4 directions until edge)
    ROOK_RAYS: Dict[Tuple[int, int], Dict[str, int]] = {}  # (file, rank) -> {direction: bitboard}
    
    # Palace diagonal rays for rook/cannon
    PALACE_DIAG_RAYS: Dict[Tuple[int, int], List[int]] = {}  # (file, rank) -> list of diagonal bitboards
    
    @classmethod
    def initialize(cls):
        """Initialize all attack tables."""
        cls._init_king_moves()
        cls._init_pawn_moves()
        cls._init_horse_moves()
        cls._init_elephant_moves()
        cls._init_rook_rays()
        cls._init_palace_diagonals()
    
    @classmethod
    def _init_king_moves(cls):
        """Initialize king/guard orthogonal and diagonal move tables."""
        # Orthogonal moves (valid everywhere in palace)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for rank in range(10):
            for file in range(9):
                bit_idx = square_to_bit(file, rank)
                moves = 0
                
                for df, dr in directions:
                    nf, nr = file + df, rank + dr
                    if 0 <= nf < 9 and 0 <= nr < 10:
                        moves |= 1 << square_to_bit(nf, nr)
                
                cls.KING_ORTHOGONAL[bit_idx] = moves
        
        # Diagonal moves in HAN palace
        diag_directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for (f, r) in HAN_PALACE_DIAGONAL_POINTS:
            bit_idx = square_to_bit(f, r)
            moves = 0
            for df, dr in diag_directions:
                nf, nr = f + df, r + dr
                if (nf, nr) in HAN_PALACE_DIAGONAL_POINTS:
                    moves |= 1 << square_to_bit(nf, nr)
            cls.KING_DIAGONAL_HAN[bit_idx] = moves
        
        # Diagonal moves in CHO palace
        for (f, r) in CHO_PALACE_DIAGONAL_POINTS:
            bit_idx = square_to_bit(f, r)
            moves = 0
            for df, dr in diag_directions:
                nf, nr = f + df, r + dr
                if (nf, nr) in CHO_PALACE_DIAGONAL_POINTS:
                    moves |= 1 << square_to_bit(nf, nr)
            cls.KING_DIAGONAL_CHO[bit_idx] = moves
    
    @classmethod
    def _init_pawn_moves(cls):
        """Initialize pawn move tables."""
        for rank in range(10):
            for file in range(9):
                bit_idx = square_to_bit(file, rank)
                
                # HAN pawns move up (rank increases)
                han_moves = 0
                # Forward
                if rank < 9:
                    han_moves |= 1 << square_to_bit(file, rank + 1)
                # Sideways
                if file > 0:
                    han_moves |= 1 << square_to_bit(file - 1, rank)
                if file < 8:
                    han_moves |= 1 << square_to_bit(file + 1, rank)
                cls.PAWN_HAN[bit_idx] = han_moves
                
                # CHO pawns move down (rank decreases)
                cho_moves = 0
                # Forward
                if rank > 0:
                    cho_moves |= 1 << square_to_bit(file, rank - 1)
                # Sideways
                if file > 0:
                    cho_moves |= 1 << square_to_bit(file - 1, rank)
                if file < 8:
                    cho_moves |= 1 << square_to_bit(file + 1, rank)
                cls.PAWN_CHO[bit_idx] = cho_moves
    
    @classmethod
    def _init_horse_moves(cls):
        """Initialize horse move tables with leg blocking info."""
        # L-shaped patterns: (df, dr) for final position
        patterns = [
            (1, 2, 0, 1),   # right-up: leg at (0, 1)
            (-1, 2, 0, 1),  # left-up: leg at (0, 1)
            (1, -2, 0, -1), # right-down: leg at (0, -1)
            (-1, -2, 0, -1),# left-down: leg at (0, -1)
            (2, 1, 1, 0),   # up-right: leg at (1, 0)
            (-2, 1, -1, 0), # up-left: leg at (-1, 0)
            (2, -1, 1, 0),  # down-right: leg at (1, 0)
            (-2, -1, -1, 0),# down-left: leg at (-1, 0)
        ]
        
        for rank in range(10):
            for file in range(9):
                bit_idx = square_to_bit(file, rank)
                moves = 0
                legs = []
                
                for df, dr, leg_df, leg_dr in patterns:
                    nf, nr = file + df, rank + dr
                    leg_f, leg_r = file + leg_df, rank + leg_dr
                    
                    if 0 <= nf < 9 and 0 <= nr < 10 and 0 <= leg_f < 9 and 0 <= leg_r < 10:
                        moves |= 1 << square_to_bit(nf, nr)
                        legs.append((nf, nr, leg_f, leg_r))
                
                cls.HORSE_MOVES[bit_idx] = moves
                cls.HORSE_LEGS[bit_idx] = legs
    
    @classmethod
    def _init_elephant_moves(cls):
        """Initialize elephant move tables with blocking info."""
        # Elephant: 1 orthogonal + 2 diagonal in same direction
        # orth_dir -> valid diagonal directions
        orth_to_diag = {
            (1, 0): [(1, 1), (1, -1)],    # right → up-right, down-right
            (-1, 0): [(-1, 1), (-1, -1)], # left → up-left, down-left
            (0, 1): [(1, 1), (-1, 1)],    # up → up-right, up-left
            (0, -1): [(1, -1), (-1, -1)], # down → down-right, down-left
        }
        
        for rank in range(10):
            for file in range(9):
                bit_idx = square_to_bit(file, rank)
                moves = 0
                legs = []
                
                for orth_dir, valid_diags in orth_to_diag.items():
                    orth_df, orth_dr = orth_dir
                    orth_f = file + orth_df
                    orth_r = rank + orth_dr
                    
                    if not (0 <= orth_f < 9 and 0 <= orth_r < 10):
                        continue
                    
                    for diag_df, diag_dr in valid_diags:
                        # First diagonal step
                        diag1_f = orth_f + diag_df
                        diag1_r = orth_r + diag_dr
                        
                        if not (0 <= diag1_f < 9 and 0 <= diag1_r < 10):
                            continue
                        
                        # Second diagonal step (destination)
                        dest_f = diag1_f + diag_df
                        dest_r = diag1_r + diag_dr
                        
                        if 0 <= dest_f < 9 and 0 <= dest_r < 10:
                            moves |= 1 << square_to_bit(dest_f, dest_r)
                            legs.append((dest_f, dest_r, orth_f, orth_r, diag1_f, diag1_r))
                
                cls.ELEPHANT_MOVES[bit_idx] = moves
                cls.ELEPHANT_LEGS[bit_idx] = legs
    
    @classmethod
    def _init_rook_rays(cls):
        """Initialize rook ray tables for sliding attacks."""
        directions = {
            'up': (0, 1),
            'down': (0, -1),
            'right': (1, 0),
            'left': (-1, 0),
        }
        
        for rank in range(10):
            for file in range(9):
                cls.ROOK_RAYS[(file, rank)] = {}
                
                for dir_name, (df, dr) in directions.items():
                    ray = 0
                    nf, nr = file + df, rank + dr
                    
                    while 0 <= nf < 9 and 0 <= nr < 10:
                        ray |= 1 << square_to_bit(nf, nr)
                        nf += df
                        nr += dr
                    
                    cls.ROOK_RAYS[(file, rank)][dir_name] = ray
    
    @classmethod
    def _init_palace_diagonals(cls):
        """Initialize palace diagonal rays for rook/cannon."""
        # Palace diagonal lines
        han_line1 = [(3, 0), (4, 1), (5, 2)]
        han_line2 = [(5, 0), (4, 1), (3, 2)]
        cho_line1 = [(3, 7), (4, 8), (5, 9)]
        cho_line2 = [(5, 7), (4, 8), (3, 9)]
        
        all_lines = [han_line1, han_line2, cho_line1, cho_line2]
        
        for line in all_lines:
            for i, (f, r) in enumerate(line):
                if (f, r) not in cls.PALACE_DIAG_RAYS:
                    cls.PALACE_DIAG_RAYS[(f, r)] = []
                
                # Create ray in both directions from this point
                for direction in [-1, 1]:
                    ray = 0
                    for j in range(i + direction, len(line) if direction > 0 else -1, direction):
                        if 0 <= j < len(line):
                            pf, pr = line[j]
                            ray |= 1 << square_to_bit(pf, pr)
                    if ray:
                        cls.PALACE_DIAG_RAYS[(f, r)].append(ray)


class BitBoard:
    """Bitboard-based position representation for Janggi.
    
    Uses separate bitboards for each piece type and side.
    Enables fast position queries using bitwise operations.
    """
    
    def __init__(self):
        """Initialize empty bitboards."""
        # Piece bitboards: side -> piece_type -> bitboard
        self.pieces: Dict[str, Dict[str, int]] = {
            'CHO': {
                'KING': 0, 'GUARD': 0, 'ELEPHANT': 0, 'HORSE': 0,
                'ROOK': 0, 'CANNON': 0, 'PAWN': 0
            },
            'HAN': {
                'KING': 0, 'GUARD': 0, 'ELEPHANT': 0, 'HORSE': 0,
                'ROOK': 0, 'CANNON': 0, 'PAWN': 0
            }
        }
        
        # Combined bitboards for fast lookup
        self._all_cho: int = 0
        self._all_han: int = 0
        self._all_pieces: int = 0
    
    def set_piece(self, side: str, piece_type: str, file: int, rank: int) -> None:
        """Place a piece on the board."""
        bit = 1 << square_to_bit(file, rank)
        self.pieces[side][piece_type] |= bit
        
        if side == 'CHO':
            self._all_cho |= bit
        else:
            self._all_han |= bit
        self._all_pieces |= bit
    
    def clear_piece(self, side: str, piece_type: str, file: int, rank: int) -> None:
        """Remove a piece from the board."""
        bit = 1 << square_to_bit(file, rank)
        self.pieces[side][piece_type] &= ~bit
        
        if side == 'CHO':
            self._all_cho &= ~bit
        else:
            self._all_han &= ~bit
        self._all_pieces &= ~bit
    
    def get_piece_at(self, file: int, rank: int) -> Optional[Tuple[str, str]]:
        """Get piece at position. Returns (side, piece_type) or None."""
        bit = 1 << square_to_bit(file, rank)
        
        if not (self._all_pieces & bit):
            return None
        
        for side in ['CHO', 'HAN']:
            for piece_type, bb in self.pieces[side].items():
                if bb & bit:
                    return (side, piece_type)
        
        return None
    
    def move_piece(self, side: str, piece_type: str, 
                   from_file: int, from_rank: int,
                   to_file: int, to_rank: int) -> Optional[Tuple[str, str]]:
        """Move a piece and return captured piece info if any."""
        captured = self.get_piece_at(to_file, to_rank)
        
        if captured:
            cap_side, cap_type = captured
            self.clear_piece(cap_side, cap_type, to_file, to_rank)
        
        self.clear_piece(side, piece_type, from_file, from_rank)
        self.set_piece(side, piece_type, to_file, to_rank)
        
        return captured
    
    def is_empty(self, file: int, rank: int) -> bool:
        """Check if a square is empty."""
        return not is_bit_set(self._all_pieces, file, rank)
    
    def is_occupied(self, file: int, rank: int) -> bool:
        """Check if a square is occupied."""
        return is_bit_set(self._all_pieces, file, rank)
    
    def is_occupied_by(self, side: str, file: int, rank: int) -> bool:
        """Check if a square is occupied by a specific side."""
        bb = self._all_cho if side == 'CHO' else self._all_han
        return is_bit_set(bb, file, rank)
    
    def get_all_pieces(self, side: str) -> int:
        """Get bitboard of all pieces for a side."""
        return self._all_cho if side == 'CHO' else self._all_han
    
    def get_piece_positions(self, side: str, piece_type: str) -> List[Tuple[int, int]]:
        """Get all positions of a specific piece type."""
        return list(iter_squares(self.pieces[side][piece_type]))
    
    def get_king_position(self, side: str) -> Optional[Tuple[int, int]]:
        """Get king position for a side."""
        bb = self.pieces[side]['KING']
        if bb == 0:
            return None
        bit_idx = lsb_index(bb)
        return bit_to_square(bit_idx)
    
    def count_pieces(self, side: str) -> int:
        """Count total pieces for a side."""
        return popcount(self._all_cho if side == 'CHO' else self._all_han)
    
    def generate_rook_attacks(self, file: int, rank: int) -> int:
        """Generate rook attack squares considering blockers."""
        attacks = 0
        rays = AttackTables.ROOK_RAYS.get((file, rank), {})
        
        for direction, ray in rays.items():
            # Find first blocker in this direction
            blockers = ray & self._all_pieces
            
            if blockers:
                # Get the first blocker
                first_blocker_bit = lsb_index(blockers) if direction in ['up', 'right'] else (blockers.bit_length() - 1)
                first_blocker_bb = 1 << first_blocker_bit
                
                # Include squares up to and including first blocker
                if direction in ['up', 'right']:
                    # Scan from current position
                    attacks |= ray & ((first_blocker_bb << 1) - 1)
                else:
                    # For down/left, we need to include from the blocker
                    attacks |= ray & ~((first_blocker_bb) - 1)
            else:
                attacks |= ray
        
        return attacks
    
    def generate_cannon_attacks(self, file: int, rank: int, side: str) -> int:
        """Generate cannon attack squares (must jump over exactly one non-cannon piece)."""
        attacks = 0
        rays = AttackTables.ROOK_RAYS.get((file, rank), {})
        own_pieces = self._all_cho if side == 'CHO' else self._all_han
        enemy_pieces = self._all_han if side == 'CHO' else self._all_cho
        
        # Get cannon positions to exclude as screens
        cannons = self.pieces['CHO']['CANNON'] | self.pieces['HAN']['CANNON']
        
        for direction, ray in rays.items():
            # Exclude cannons as potential screens
            valid_screens = (ray & self._all_pieces) & ~cannons
            
            if not valid_screens:
                continue
            
            # Find first valid screen
            if direction in ['up', 'right']:
                screen_bit = lsb_index(valid_screens)
            else:
                screen_bit = valid_screens.bit_length() - 1
            
            screen_bb = 1 << screen_bit
            
            # Get squares beyond the screen
            if direction in ['up', 'right']:
                beyond = ray & ~((screen_bb << 1) - 1)
            else:
                beyond = ray & ((screen_bb) - 1)
            
            # Find landing squares (empty or enemy non-cannon)
            enemy_non_cannon = enemy_pieces & ~self.pieces['HAN' if side == 'CHO' else 'CHO']['CANNON']
            
            for bit_idx in iter_bits(beyond):
                sq_bb = 1 << bit_idx
                
                if sq_bb & self._all_pieces:
                    # Occupied - check if capturable enemy (not cannon)
                    if sq_bb & enemy_non_cannon:
                        attacks |= sq_bb
                    break  # Can't go further
                else:
                    attacks |= sq_bb
        
        return attacks
    
    @classmethod
    def from_board(cls, board) -> 'BitBoard':
        """Create BitBoard from standard Board object."""
        bb = cls()
        
        for rank in range(10):
            for file in range(9):
                piece = board.get_piece(file, rank)
                if piece:
                    bb.set_piece(piece.side.value, piece.piece_type.value, file, rank)
        
        return bb
    
    def copy(self) -> 'BitBoard':
        """Create a copy of this BitBoard."""
        new_bb = BitBoard()
        
        for side in ['CHO', 'HAN']:
            for piece_type in self.pieces[side]:
                new_bb.pieces[side][piece_type] = self.pieces[side][piece_type]
        
        new_bb._all_cho = self._all_cho
        new_bb._all_han = self._all_han
        new_bb._all_pieces = self._all_pieces
        
        return new_bb
    
    def __repr__(self) -> str:
        """String representation showing piece counts."""
        cho_count = popcount(self._all_cho)
        han_count = popcount(self._all_han)
        return f"BitBoard(CHO={cho_count}, HAN={han_count})"


# Initialize attack tables on module load
AttackTables.initialize()


# Utility function for getting attack bitboards
def get_pawn_attacks(file: int, rank: int, side: str) -> int:
    """Get pawn attack squares."""
    bit_idx = square_to_bit(file, rank)
    if side == 'HAN':
        return AttackTables.PAWN_HAN[bit_idx]
    return AttackTables.PAWN_CHO[bit_idx]


def get_king_attacks(file: int, rank: int, side: str) -> int:
    """Get king attack squares (orthogonal + diagonal in palace)."""
    bit_idx = square_to_bit(file, rank)
    attacks = AttackTables.KING_ORTHOGONAL[bit_idx]
    
    if side == 'HAN':
        attacks |= AttackTables.KING_DIAGONAL_HAN[bit_idx]
    else:
        attacks |= AttackTables.KING_DIAGONAL_CHO[bit_idx]
    
    return attacks


def get_guard_attacks(file: int, rank: int, side: str) -> int:
    """Get guard attack squares (same as king within palace)."""
    return get_king_attacks(file, rank, side)


def get_horse_attacks(file: int, rank: int, all_pieces: int) -> int:
    """Get horse attack squares, considering leg blocking."""
    bit_idx = square_to_bit(file, rank)
    attacks = 0
    
    for dest_f, dest_r, leg_f, leg_r in AttackTables.HORSE_LEGS[bit_idx]:
        # Check if leg is blocked
        if not is_bit_set(all_pieces, leg_f, leg_r):
            attacks |= 1 << square_to_bit(dest_f, dest_r)
    
    return attacks


def get_elephant_attacks(file: int, rank: int, all_pieces: int) -> int:
    """Get elephant attack squares, considering leg blocking."""
    bit_idx = square_to_bit(file, rank)
    attacks = 0
    
    for dest_f, dest_r, leg1_f, leg1_r, leg2_f, leg2_r in AttackTables.ELEPHANT_LEGS[bit_idx]:
        # Check if both legs are clear
        if not is_bit_set(all_pieces, leg1_f, leg1_r) and not is_bit_set(all_pieces, leg2_f, leg2_r):
            attacks |= 1 << square_to_bit(dest_f, dest_r)
    
    return attacks


# Singleton pattern for global access
_bitboard_instance: Optional[BitBoard] = None


def get_bitboard() -> BitBoard:
    """Get or create the global BitBoard instance."""
    global _bitboard_instance
    if _bitboard_instance is None:
        _bitboard_instance = BitBoard()
    return _bitboard_instance

