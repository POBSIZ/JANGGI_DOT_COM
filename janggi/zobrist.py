"""Zobrist hashing for fast position hashing in Janggi.

Zobrist hashing uses XOR operations on random 64-bit integers to create
efficient position hashes. This is 2-3x faster than string-based hashing
and allows for incremental hash updates.
"""

import random
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .board import Board, Piece

# Use fixed seed for reproducibility
random.seed(0x4A414E474749)  # "JANGGI" in hex-ish


class ZobristHash:
    """Fast Zobrist hashing for Janggi positions.
    
    Uses 64-bit random integers XOR'd together for:
    - Each piece type on each square (7 pieces * 2 sides * 90 squares)
    - Side to move
    
    This allows:
    - O(1) hash updates when making/unmaking moves
    - Very fast hash lookups compared to string hashing
    """
    
    # Constants
    FILES = 9
    RANKS = 10
    PIECE_TYPES = 7  # KING, GUARD, ELEPHANT, HORSE, ROOK, CANNON, PAWN
    SIDES = 2  # HAN, CHO
    
    # Piece type indices (must match PieceType enum order)
    PIECE_TYPE_MAP = {
        'KING': 0,
        'GUARD': 1,
        'ELEPHANT': 2,
        'HORSE': 3,
        'ROOK': 4,
        'CANNON': 5,
        'PAWN': 6,
    }
    
    SIDE_MAP = {
        'CHO': 0,
        'HAN': 1,
    }
    
    def __init__(self):
        """Initialize random keys for Zobrist hashing."""
        # Generate random 64-bit integers for each piece/square/side combination
        # [side][piece_type][rank][file] -> random key
        self.piece_keys = [
            [
                [
                    [self._random_64() for _ in range(self.FILES)]
                    for _ in range(self.RANKS)
                ]
                for _ in range(self.PIECE_TYPES)
            ]
            for _ in range(self.SIDES)
        ]
        
        # Random key for side to move (XOR'd when CHO to move)
        self.side_key = self._random_64()
    
    @staticmethod
    def _random_64() -> int:
        """Generate a random 64-bit integer."""
        return random.getrandbits(64)
    
    def get_piece_key(self, side_value: str, piece_type_value: str, file: int, rank: int) -> int:
        """Get the Zobrist key for a piece at a position.
        
        Args:
            side_value: Side value ('CHO' or 'HAN')
            piece_type_value: PieceType value ('KING', 'ROOK', etc.)
            file: File index (0-8)
            rank: Rank index (0-9)
            
        Returns:
            64-bit Zobrist key for this piece/position combination
        """
        side_idx = self.SIDE_MAP.get(side_value, 0)
        piece_idx = self.PIECE_TYPE_MAP.get(piece_type_value, 0)
        return self.piece_keys[side_idx][piece_idx][rank][file]
    
    def compute_hash(self, board: 'Board') -> int:
        """Compute the full Zobrist hash for a board position.
        
        Args:
            board: The board position to hash
            
        Returns:
            64-bit Zobrist hash
        """
        from .board import Side
        
        h = 0
        
        # Hash all pieces
        for rank in range(self.RANKS):
            for file in range(self.FILES):
                piece = board.board[rank][file]
                if piece is not None:
                    key = self.get_piece_key(
                        piece.side.value,
                        piece.piece_type.value,
                        file,
                        rank
                    )
                    h ^= key
        
        # Hash side to move
        if board.side_to_move == Side.CHO:
            h ^= self.side_key
        
        return h
    
    def update_hash_move(
        self,
        current_hash: int,
        piece_side: str,
        piece_type: str,
        from_file: int,
        from_rank: int,
        to_file: int,
        to_rank: int,
        captured_piece_side: Optional[str] = None,
        captured_piece_type: Optional[str] = None,
    ) -> int:
        """Incrementally update hash after a move.
        
        XOR is self-inverse: a ^ b ^ b = a
        So we XOR out the old position and XOR in the new position.
        
        Args:
            current_hash: Current position hash
            piece_side: Side value of moving piece
            piece_type: Type of moving piece
            from_file, from_rank: Source position
            to_file, to_rank: Destination position
            captured_piece_side: Side of captured piece (if any)
            captured_piece_type: Type of captured piece (if any)
            
        Returns:
            Updated hash after the move
        """
        h = current_hash
        
        # XOR out piece from source square
        h ^= self.get_piece_key(piece_side, piece_type, from_file, from_rank)
        
        # XOR in piece at destination square
        h ^= self.get_piece_key(piece_side, piece_type, to_file, to_rank)
        
        # If capturing, XOR out captured piece
        if captured_piece_side is not None and captured_piece_type is not None:
            h ^= self.get_piece_key(captured_piece_side, captured_piece_type, to_file, to_rank)
        
        # Toggle side to move
        h ^= self.side_key
        
        return h


# Global instance for shared use (initialized once)
_zobrist_instance: Optional[ZobristHash] = None


def get_zobrist() -> ZobristHash:
    """Get the global Zobrist hash instance.
    
    Uses lazy initialization to avoid creating random keys
    if Zobrist hashing isn't used.
    """
    global _zobrist_instance
    if _zobrist_instance is None:
        _zobrist_instance = ZobristHash()
    return _zobrist_instance

