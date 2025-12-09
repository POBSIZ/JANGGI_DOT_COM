"""Opening book for Janggi - Database of standard opening moves from game records."""

import os
import random
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class OpeningMove:
    """Represents an opening move with statistics."""
    from_square: str  # e.g., "a4"
    to_square: str    # e.g., "a5"
    count: int = 1    # How many times this move was played
    win_rate: float = 0.5  # Win rate after this move (0-1)
    
    def __str__(self):
        return f"{self.from_square}{self.to_square}"


class OpeningBook:
    """Opening book database built from game records (.gib files)."""
    
    def __init__(self):
        """Initialize empty opening book."""
        # Map from position hash (simplified) to list of moves with statistics
        # Key: tuple of (move_number, previous_moves_hash)
        # Value: dict mapping move_str to OpeningMove
        self.book: Dict[str, Dict[str, OpeningMove]] = defaultdict(dict)
        self.max_ply = 20  # Maximum number of plies (half-moves) to store
        
    def load_from_gib_directory(self, directory: str) -> int:
        """Load all .gib files from a directory.
        
        Args:
            directory: Path to directory containing .gib files
            
        Returns:
            Number of games successfully loaded
        """
        if not os.path.exists(directory):
            return 0
            
        games_loaded = 0
        
        for filename in os.listdir(directory):
            if filename.endswith('.gib'):
                filepath = os.path.join(directory, filename)
                try:
                    if self._load_gib_file(filepath):
                        games_loaded += 1
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
                    continue
                    
        return games_loaded
    
    def _load_gib_file(self, filepath: str) -> bool:
        """Load a single .gib file into the opening book.
        
        Args:
            filepath: Path to .gib file
            
        Returns:
            True if successfully loaded
        """
        try:
            # Try different encodings
            content = None
            for encoding in ['utf-8', 'cp949', 'euc-kr']:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                return False
                
            # Parse metadata
            result = self._parse_result(content)
            
            # Parse moves
            moves = self._parse_moves(content)
            
            if not moves:
                return False
                
            # Add moves to opening book
            self._add_game_to_book(moves, result)
            
            return True
            
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return False
    
    def _parse_result(self, content: str) -> Optional[str]:
        """Parse game result from .gib file content.
        
        Returns:
            "cho_win", "han_win", "draw", or None if unknown
        """
        # Look for result line
        for line in content.split('\n'):
            if '승패' in line or '결과' in line:
                if '초승' in line or '楚勝' in line:
                    return "cho_win"
                elif '한승' in line or '漢勝' in line:
                    return "han_win"
                elif '무승부' in line or '비김' in line:
                    return "draw"
        return None
    
    def _parse_moves(self, content: str) -> List[Tuple[str, str]]:
        """Parse moves from .gib file content.
        
        Returns:
            List of (from_square, to_square) tuples
        """
        moves = []
        
        # Find the moves section (usually after metadata)
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip metadata lines (usually start with keywords)
            if any(kw in line for kw in ['대회명', '일시', '대국일자', '선수', '제한시간', '총수', '승패', '초상', '한상', '棋戰', '日期', '紅方', '黑方']):
                continue
            
            # Parse move notation: e.g., "41漢兵42" means piece at position 41 moves to 42
            # Format: [from_pos][side][piece_type][to_pos]
            # Position is in format: file(1-9) + rank(1-10), e.g., "41" = file 4, rank 1
            parsed_moves = self._parse_move_line(line)
            moves.extend(parsed_moves)
        
        return moves
    
    def _parse_move_line(self, line: str) -> List[Tuple[str, str]]:
        """Parse a single line that may contain multiple moves.
        
        Format: [from_pos][side][piece][to_pos], e.g., "41漢兵42" or "81楚卒82"
        Position format: first digit is file (1-9), remaining digits are rank (1-10)
        """
        moves = []
        
        # Korean/Chinese piece names
        piece_chars = '車馬象士將包卒兵漢楚한초'
        side_chars = '漢楚한초'
        
        i = 0
        while i < len(line):
            # Try to find a valid move pattern
            # Look for: digits + side/piece chars + digits
            
            # Find start position (2-3 digits)
            start_idx = i
            while i < len(line) and line[i].isdigit():
                i += 1
            
            if i == start_idx or i - start_idx > 3:
                i = start_idx + 1
                continue
                
            from_pos_str = line[start_idx:i]
            
            # Find piece/side characters
            piece_start = i
            while i < len(line) and line[i] in piece_chars:
                i += 1
            
            if i == piece_start:
                i = start_idx + 1
                continue
                
            # Find end position (2-3 digits)
            end_start = i
            while i < len(line) and line[i].isdigit():
                i += 1
            
            if i == end_start or i - end_start > 3:
                i = start_idx + 1
                continue
                
            to_pos_str = line[end_start:i]
            
            # Convert positions to standard notation
            from_square = self._gib_pos_to_notation(from_pos_str)
            to_square = self._gib_pos_to_notation(to_pos_str)
            
            if from_square and to_square:
                moves.append((from_square, to_square))
        
        return moves
    
    def _gib_pos_to_notation(self, pos_str: str) -> Optional[str]:
        """Convert GIB position notation to standard notation.
        
        GIB format: first digit is rank (0=10, 1-9), second digit is file (1-9)
        Standard format: 'a1' to 'i10'
        
        Examples:
        - "41" -> "a4" (rank 4, file 1)
        - "49" -> "i4" (rank 4, file 9)
        - "01" -> "a10" (rank 10, file 1) - '0' means rank 10
        - "09" -> "i10" (rank 10, file 9)
        """
        if len(pos_str) == 2:
            rank_digit = pos_str[0]
            file_num = int(pos_str[1])
            # '0' represents rank 10
            rank_num = 10 if rank_digit == '0' else int(rank_digit)
        elif len(pos_str) == 3:
            # For 3-digit positions like "101", "102", etc.
            rank_num = int(pos_str[0:2])
            file_num = int(pos_str[2])
        else:
            return None
            
        if not (1 <= file_num <= 9) or not (1 <= rank_num <= 10):
            return None
            
        file_char = chr(ord('a') + file_num - 1)
        return f"{file_char}{rank_num}"
    
    def _add_game_to_book(self, moves: List[Tuple[str, str]], result: Optional[str]) -> None:
        """Add a game's opening moves to the book.
        
        Args:
            moves: List of (from_square, to_square) tuples
            result: Game result ("cho_win", "han_win", "draw", or None)
        """
        # Calculate win rate for each move based on result
        # Cho moves first (ply 0, 2, 4, ...), Han moves second (ply 1, 3, 5, ...)
        
        move_sequence = []
        
        for ply, (from_sq, to_sq) in enumerate(moves):
            if ply >= self.max_ply:
                break
                
            # Create position key from move sequence so far
            position_key = self._get_position_key(move_sequence, ply)
            
            move_str = f"{from_sq}{to_sq}"
            
            # Calculate win rate for this move
            win_rate = 0.5
            if result:
                is_cho_move = (ply % 2 == 0)
                if result == "cho_win":
                    win_rate = 1.0 if is_cho_move else 0.0
                elif result == "han_win":
                    win_rate = 0.0 if is_cho_move else 1.0
                else:  # draw
                    win_rate = 0.5
            
            # Add or update move in book
            if move_str in self.book[position_key]:
                existing = self.book[position_key][move_str]
                # Update with running average
                total_count = existing.count + 1
                existing.win_rate = (existing.win_rate * existing.count + win_rate) / total_count
                existing.count = total_count
            else:
                self.book[position_key][move_str] = OpeningMove(
                    from_square=from_sq,
                    to_square=to_sq,
                    count=1,
                    win_rate=win_rate
                )
            
            move_sequence.append(move_str)
    
    def _get_position_key(self, move_sequence: List[str], ply: int) -> str:
        """Generate a position key from move sequence.
        
        Args:
            move_sequence: List of moves made so far
            ply: Current ply (half-move) number
            
        Returns:
            String key for this position
        """
        # Use ply number and hash of move sequence
        # This is a simplified key - in production, would use actual board hash
        return f"{ply}:{'-'.join(move_sequence[-6:])}"  # Last 6 moves for context
    
    def get_book_move(
        self, 
        move_history: List[Dict], 
        ply: int,
        selection_method: str = "weighted"
    ) -> Optional[Tuple[str, str]]:
        """Get an opening book move for the current position.
        
        Args:
            move_history: List of previous moves (from board.move_history)
            ply: Current ply number (len(move_history))
            selection_method: How to select move:
                - "best": Highest win rate
                - "popular": Most played
                - "weighted": Weighted random by count and win rate
                - "random": Random from available moves
        
        Returns:
            (from_square, to_square) tuple if found, None if out of book
        """
        if ply >= self.max_ply:
            return None
            
        # Convert move history to sequence
        move_sequence = []
        for move in move_history:
            from_sq = move.get('from', '')
            to_sq = move.get('to', '')
            if from_sq and to_sq:
                move_sequence.append(f"{from_sq}{to_sq}")
        
        position_key = self._get_position_key(move_sequence, ply)
        
        if position_key not in self.book:
            return None
            
        available_moves = list(self.book[position_key].values())
        
        if not available_moves:
            return None
        
        if selection_method == "best":
            # Select move with highest win rate
            best_move = max(available_moves, key=lambda m: m.win_rate)
            return (best_move.from_square, best_move.to_square)
            
        elif selection_method == "popular":
            # Select most played move
            best_move = max(available_moves, key=lambda m: m.count)
            return (best_move.from_square, best_move.to_square)
            
        elif selection_method == "weighted":
            # Weighted random selection by count * win_rate
            weights = [m.count * (0.5 + m.win_rate) for m in available_moves]
            total_weight = sum(weights)
            if total_weight == 0:
                selected_move = random.choice(available_moves)
            else:
                r = random.random() * total_weight
                cumulative = 0
                selected_move = available_moves[-1]
                for move, weight in zip(available_moves, weights):
                    cumulative += weight
                    if cumulative >= r:
                        selected_move = move
                        break
            return (selected_move.from_square, selected_move.to_square)
            
        else:  # random
            selected_move = random.choice(available_moves)
            return (selected_move.from_square, selected_move.to_square)
    
    def get_book_moves(self, move_history: List[Dict], ply: int) -> List[OpeningMove]:
        """Get all available book moves for the current position.
        
        Args:
            move_history: List of previous moves
            ply: Current ply number
            
        Returns:
            List of available OpeningMove objects
        """
        if ply >= self.max_ply:
            return []
            
        move_sequence = []
        for move in move_history:
            from_sq = move.get('from', '')
            to_sq = move.get('to', '')
            if from_sq and to_sq:
                move_sequence.append(f"{from_sq}{to_sq}")
        
        position_key = self._get_position_key(move_sequence, ply)
        
        return list(self.book[position_key].values())
    
    def is_in_book(self, move_history: List[Dict], ply: int) -> bool:
        """Check if the current position is still in the opening book.
        
        Args:
            move_history: List of previous moves
            ply: Current ply number
            
        Returns:
            True if there are book moves available
        """
        return len(self.get_book_moves(move_history, ply)) > 0
    
    def get_statistics(self) -> Dict:
        """Get statistics about the opening book.
        
        Returns:
            Dictionary with book statistics
        """
        total_positions = len(self.book)
        total_moves = sum(len(moves) for moves in self.book.values())
        
        if total_moves == 0:
            return {
                "positions": 0,
                "total_moves": 0,
                "avg_moves_per_position": 0,
                "max_ply": self.max_ply
            }
        
        return {
            "positions": total_positions,
            "total_moves": total_moves,
            "avg_moves_per_position": total_moves / total_positions if total_positions > 0 else 0,
            "max_ply": self.max_ply
        }


# Global opening book instance (lazy loaded)
_opening_book: Optional[OpeningBook] = None


def get_opening_book(gib_directory: str = "gibo") -> OpeningBook:
    """Get or create the global opening book instance.
    
    Args:
        gib_directory: Path to directory containing .gib files
        
    Returns:
        OpeningBook instance
    """
    global _opening_book
    
    if _opening_book is None:
        _opening_book = OpeningBook()
        games_loaded = _opening_book.load_from_gib_directory(gib_directory)
        stats = _opening_book.get_statistics()
        print(f"Opening book loaded: {games_loaded} games, {stats['positions']} positions, {stats['total_moves']} moves")
    
    return _opening_book

