#!/usr/bin/env python3
"""Test script to parse all gibo files and report results."""

import os
import sys
import glob

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_nnue_gibo import GibParser

def test_all_gibo_files():
    """Test parsing all .gib files in the gibo directory."""
    parser = GibParser()
    gibo_dir = 'gibo'
    
    # Find all .gib files
    gib_files = glob.glob(os.path.join(gibo_dir, '*.gib'))
    gib_files.extend(glob.glob(os.path.join(gibo_dir, '*.GIB')))
    gib_files = sorted(gib_files)
    
    print(f"📋 Found {len(gib_files)} gibo files\n")
    print("=" * 80)
    
    total_games = 0
    total_moves = 0
    successful_files = 0
    failed_files = 0
    file_results = []
    
    for filepath in gib_files:
        filename = os.path.basename(filepath)
        print(f"\n📄 Parsing: {filename}")
        
        try:
            games = parser.parse_file(filepath)
            
            if games:
                file_games = len(games)
                file_moves = sum(len(game.get('raw_moves', [])) for game in games)
                
                total_games += file_games
                total_moves += file_moves
                successful_files += 1
                
                # Get formations and results
                formations = set()
                results = {'cho': 0, 'han': 0, 'draw': 0, 'unknown': 0}
                
                for game in games:
                    cho_form = game.get('cho_formation', 'unknown')
                    han_form = game.get('han_formation', 'unknown')
                    formations.add(f"{cho_form}-{han_form}")
                    
                    result = game.get('result')
                    if result:
                        results[result] = results.get(result, 0) + 1
                    else:
                        results['unknown'] += 1
                
                print(f"  ✅ Success: {file_games} games, {file_moves} moves")
                print(f"  📊 Results: 초승={results['cho']}, 한승={results['han']}, 무승부={results['draw']}, 미상={results['unknown']}")
                print(f"  🎯 Formations: {len(formations)} unique combinations")
                
                file_results.append({
                    'file': filename,
                    'status': 'success',
                    'games': file_games,
                    'moves': file_moves,
                    'results': results
                })
            else:
                failed_files += 1
                print(f"  ⚠️  Warning: No games found")
                file_results.append({
                    'file': filename,
                    'status': 'no_games',
                    'games': 0,
                    'moves': 0
                })
                
        except Exception as e:
            failed_files += 1
            print(f"  ❌ Error: {str(e)}")
            file_results.append({
                'file': filename,
                'status': 'error',
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("\n📊 SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(gib_files)}")
    print(f"✅ Successful: {successful_files}")
    print(f"❌ Failed: {failed_files}")
    print(f"📈 Total games parsed: {total_games}")
    print(f"🎯 Total moves parsed: {total_moves}")
    
    if total_games > 0:
        avg_moves_per_game = total_moves / total_games
        print(f"📊 Average moves per game: {avg_moves_per_game:.1f}")
    
    # Show files with issues
    if failed_files > 0:
        print("\n⚠️  Files with issues:")
        for result in file_results:
            if result['status'] != 'success':
                print(f"  - {result['file']}: {result['status']}")
                if 'error' in result:
                    print(f"    Error: {result['error']}")
    
    # Show sample of successful files
    print("\n✅ Sample successful files:")
    success_count = 0
    for result in file_results:
        if result['status'] == 'success' and success_count < 5:
            print(f"  - {result['file']}: {result['games']} games, {result['moves']} moves")
            success_count += 1
    
    return {
        'total_files': len(gib_files),
        'successful_files': successful_files,
        'failed_files': failed_files,
        'total_games': total_games,
        'total_moves': total_moves,
        'file_results': file_results
    }

if __name__ == '__main__':
    print("🧪 Testing Gibo Parser on all files...\n")
    results = test_all_gibo_files()
    print("\n✅ Test complete!")

