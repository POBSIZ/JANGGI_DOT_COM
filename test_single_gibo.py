#!/usr/bin/env python3
"""Simple test script to parse a single gibo file."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_nnue_gibo import GibParser

def test_file(filepath):
    """Test parsing a single gibo file."""
    parser = GibParser()
    
    print(f"Testing: {filepath}\n")
    
    try:
        games = parser.parse_file(filepath)
        
        if not games:
            print("❌ No games found")
            return
        
        print(f"✅ Parsed {len(games)} game(s)\n")
        
        for i, game in enumerate(games):
            print(f"Game {i+1}:")
            print(f"  초차림: {game.get('cho_formation', 'unknown')}")
            print(f"  한차림: {game.get('han_formation', 'unknown')}")
            print(f"  결과: {game.get('result', 'unknown')}")
            
            raw_moves = game.get('raw_moves', [])
            print(f"  수순: {len(raw_moves)} moves")
            
            # Show first 5 moves
            if raw_moves:
                print("  샘플 수순:")
                for j, move in enumerate(raw_moves[:5]):
                    from_coord, piece, to_coord, side = move
                    piece_str = piece if piece else "?"
                    side_str = side if side else ""
                    print(f"    {j+1}. {from_coord}{side_str}{piece_str}{to_coord}")
            
            print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        test_file(sys.argv[1])
    else:
        # Test both formats
        print("Testing Korean format (지만이.gib)...")
        test_file('gibo/지만이.gib')
        print("\n" + "="*60 + "\n")
        print("Testing Chinese format (아마고수기보001.gib)...")
        test_file('gibo/아마고수기보001.gib')

