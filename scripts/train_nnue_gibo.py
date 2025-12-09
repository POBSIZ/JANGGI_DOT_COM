#!/usr/bin/env python3
"""NNUE Training from Gibo (Game Records) for Janggi.

This script parses .gib game record files and trains the NNUE model
using real game positions from professional/amateur games.

Usage:
    # Train from gibo files
    python train_nnue_gibo.py --gibo-dir gibo/ --epochs 50
    
    # Continue training from existing model
    python train_nnue_gibo.py --gibo-dir gibo/ --load models/nnue_gpu_model.json --epochs 30
"""

import argparse
import os
import re
import glob
import multiprocessing as mp
import time
import random
from typing import List, Optional, Tuple, Dict
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Install with: pip install torch")

from janggi.board import Board, Side, PieceType, Move

if TORCH_AVAILABLE:
    from janggi.nnue_torch import NNUETorch, FeatureExtractor, GPUTrainer, get_device


class GibParser:
    """Parser for Korean Janggi game record files (.gib format)."""
    
    # 한자 기물 → 기물 타입 매핑
    HANJA_TO_PIECE = {
        '卒': PieceType.PAWN,    # 초 졸
        '兵': PieceType.PAWN,    # 한 병
        '馬': PieceType.HORSE,   # 마
        '象': PieceType.ELEPHANT, # 상
        '士': PieceType.GUARD,   # 사
        '將': PieceType.KING,    # 장(왕)
        '車': PieceType.ROOK,    # 차
        '包': PieceType.CANNON,  # 포
    }
    
    # 차림 이름 매핑
    FORMATION_MAP = {
        '상마상마': '상마상마',
        '마상마상': '마상마상',
        '마상상마': '마상상마',
        '상마마상': '상마마상',
        '우외상': '상마상마',
        '좌외상': '마상마상',
        '우내상': '상마마상',
        '좌내상': '마상상마',
    }
    
    def __init__(self):
        self.games = []
    
    @staticmethod
    def convert_gibo_coord(gibo_col: int, gibo_row: int, move_num: int) -> Tuple[int, int]:
        """Convert gibo coordinates to board coordinates.
        
        Gibo coordinate system:
        - Column (세로줄): 1-9 from right to left (CHO's view)
        - Row (가로줄): 1=CHO front, 0=CHO cannon row, 8=CHO back, 9=CHO king row
        
        Board coordinate system:
        - File: 0-8 from left to right
        - Rank: 0-9 (HAN at 0-3, CHO at 6-9)
        
        Returns (file, rank)
        """
        # Column to file: reverse mapping (1->8, 9->0, 0->special)
        if gibo_col == 0:
            file = 8  # Treat 0 as leftmost (file 8) - this is a guess
        else:
            file = 9 - gibo_col
        
        # Row to rank mapping
        # Based on analysis: row 1 = rank 6 (CHO pawn), row 9 = rank 3 (HAN pawn)
        row_map = {
            0: 7,   # CHO cannon row
            1: 6,   # CHO pawn row
            2: 5,
            3: 4,
            4: 3,   # HAN pawn row
            5: 2,
            6: 1,
            7: 0,   # HAN back row
            8: 9,   # CHO back row
            9: 8,   # CHO king row
        }
        rank = row_map.get(gibo_row, gibo_row)
        
        return file, rank
    
    def parse_file(self, filepath: str) -> List[Dict]:
        """Parse a single gibo file.
        
        Returns list of game dictionaries with:
        - 'cho_formation': CHO side formation
        - 'han_formation': HAN side formation
        - 'result': 'cho', 'han', or 'draw'
        - 'raw_moves': list of raw move strings for later parsing
        """
        games = []
        
        try:
            # Try EUC-KR encoding first
            with open(filepath, 'rb') as f:
                raw = f.read()
            
            # Try different encodings
            content = None
            for encoding in ['euc-kr', 'cp949', 'utf-8']:
                try:
                    content = raw.decode(encoding, errors='replace')
                    break
                except:
                    continue
            
            if content is None:
                print(f"Warning: Could not decode {filepath}")
                return []
            
            # Split into individual games
            game_blocks = self._split_games(content)
            
            for block in game_blocks:
                game = self._parse_game_block(block)
                if game and len(game.get('raw_moves', [])) > 0:
                    games.append(game)
            
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
        
        return games
    
    def _split_games(self, content: str) -> List[str]:
        """Split content into individual game blocks."""
        games = []
        pattern = r'\[대회명|\[회전'
        parts = re.split(f'(?={pattern})', content)
        
        for part in parts:
            part = part.strip()
            if part and '[' in part:
                games.append(part)
        
        return games
    
    def _parse_game_block(self, block: str) -> Optional[Dict]:
        """Parse a single game block."""
        game = {
            'cho_formation': '마상상마',
            'han_formation': '마상상마',
            'result': None,
            'raw_moves': []  # Store raw move strings
        }
        
        lines = block.split('\n')
        moves_text = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('[초차림'):
                match = re.search(r'"([^"]+)"', line)
                if match:
                    formation = match.group(1)
                    game['cho_formation'] = self.FORMATION_MAP.get(formation, formation)
            
            elif line.startswith('[한차림'):
                match = re.search(r'"([^"]+)"', line)
                if match:
                    formation = match.group(1)
                    game['han_formation'] = self.FORMATION_MAP.get(formation, formation)
            
            elif line.startswith('[대국결과'):
                match = re.search(r'"([^"]+)"', line)
                if match:
                    result_text = match.group(1)
                    if '초' in result_text and ('승' in result_text or '완' in result_text):
                        game['result'] = 'cho'
                    elif '한' in result_text and ('승' in result_text or '완' in result_text):
                        game['result'] = 'han'
                    else:
                        game['result'] = 'draw'
            
            elif re.match(r'^\d+\.', line):
                moves_text.append(line)
        
        # Store raw moves for later parsing with game context
        all_moves_text = ' '.join(moves_text)
        game['raw_moves'] = self._extract_raw_moves(all_moves_text)
        
        return game
    
    def _extract_raw_moves(self, moves_text: str) -> List[Tuple[str, Optional[str], str]]:
        """Extract raw move data: (from_coord, piece_char, to_coord)
        
        Supports two gibo formats:
        1. Without side prefix: 79卒78 (숫자-기물-숫자)
        2. With side prefix: 41漢兵42 (숫자-진영-기물-숫자)
           - 漢 (Han) or 楚 (Cho) indicates which side's piece
        """
        moves = []
        # Pattern supports optional side indicator (漢/楚) before piece type
        move_pattern = r'(\d{2})(?:[漢楚])?([卒兵馬象士將車包])?(\d{2})'
        
        for match in re.finditer(move_pattern, moves_text):
            from_pos = match.group(1)
            piece = match.group(2)
            to_pos = match.group(3)
            moves.append((from_pos, piece, to_pos))
        
        return moves
    
    def parse_directory(self, directory: str) -> List[Dict]:
        """Parse all .gib files in a directory."""
        all_games = []
        
        gib_files = glob.glob(os.path.join(directory, '*.gib'))
        gib_files.extend(glob.glob(os.path.join(directory, '*.GIB')))
        
        print(f"Found {len(gib_files)} gibo files")
        
        for filepath in sorted(gib_files):
            print(f"Parsing {os.path.basename(filepath)}...", end=' ')
            games = self.parse_file(filepath)
            print(f"({len(games)} games)")
            all_games.extend(games)
        
        print(f"Total: {len(all_games)} games parsed")
        return all_games


# ============================================================================
# 모듈 레벨 Worker 함수 (multiprocessing을 위해 필요)
# ============================================================================

def _process_single_game_worker(args: Tuple[Dict, int, int]) -> Tuple[List[np.ndarray], List[float], bool, Optional[str]]:
    """
    단일 게임을 처리하는 worker 함수 (모듈 레벨에 있어야 pickle 가능)
    
    Args:
        args: (game_dict, max_positions, feature_size) 튜플
        
    Returns:
        (features_list, targets_list, success, error_message)
    """
    game, max_positions, feature_size = args
    
    # 각 프로세스가 독립적으로 FeatureExtractor 생성 (공유 상태 문제 해결)
    if not TORCH_AVAILABLE:
        return [], [], False, "PyTorch not available"
    
    feature_extractor = FeatureExtractor(feature_size)
    features = []
    targets = []
    
    try:
        # 게임 데이터 추출
        cho_formation = game.get('cho_formation', '마상상마')
        han_formation = game.get('han_formation', '마상상마')
        result = game.get('result', None)
        raw_moves = game.get('raw_moves', [])
        
        if len(raw_moves) < 5:
            return features, targets, True, None
        
        # 보드 초기화
        try:
            board = Board(
                cho_formation=cho_formation,
                han_formation=han_formation
            )
        except Exception:
            board = Board()
        
        # 타겟 값 계산
        if result == 'cho':
            cho_target = 1.0
            han_target = -1.0
        elif result == 'han':
            cho_target = -1.0
            han_target = 1.0
        else:
            cho_target = 0.0
            han_target = 0.0
        
        # 게임 재현 및 포지션 추출
        positions_collected = 0
        failed_moves = 0
        total_moves = len(raw_moves)
        
        for move_idx, (from_coord, piece_char, to_coord) in enumerate(raw_moves):
            if positions_collected >= max_positions:
                break
            
            if failed_moves > 5 and failed_moves > positions_collected:
                break
            
            # Feature 추출
            try:
                feat = feature_extractor.extract(board)
                
                if not np.isnan(feat).any():
                    progress = move_idx / max(total_moves - 1, 1)
                    
                    if board.side_to_move == Side.CHO:
                        base_target = cho_target
                    else:
                        base_target = han_target
                    
                    target = base_target * (0.3 + 0.7 * progress)
                    
                    features.append(feat)
                    targets.append(target)
                    positions_collected += 1
                    
            except Exception:
                pass
            
            # 수 찾기 및 실행
            move = _find_valid_move_helper(board, from_coord, to_coord, piece_char)
            
            if move:
                if not board.make_move(move):
                    failed_moves += 1
            else:
                failed_moves += 1
                legal_moves = board.generate_moves()
                if legal_moves:
                    random_move = random.choice(legal_moves)
                    board.make_move(random_move)
                else:
                    break
        
        return features, targets, True, None
        
    except Exception as e:
        # 예외 정보를 반환값에 포함 (디버깅 용이)
        return [], [], False, str(e)


def _find_valid_move_helper(board: Board, from_coord: str, to_coord: str, 
                            piece_char: Optional[str]) -> Optional[Move]:
    """좌표 변환 헬퍼 함수 (worker 함수 내부에서 사용)"""
    gibo_col1, gibo_row1 = int(from_coord[0]), int(from_coord[1])
    gibo_col2, gibo_row2 = int(to_coord[0]), int(to_coord[1])
    
    transformations = [
        lambda c, r: (9 - c if c > 0 else 8, {0:7,1:6,2:5,3:4,4:3,5:2,6:1,7:0,8:9,9:8}.get(r, r)),
        lambda c, r: (8 - c, 9 - r),
        lambda c, r: (c, r),
        lambda c, r: (8 - c, r),
        lambda c, r: (9 - c, 9 - r),
    ]
    
    for transform in transformations:
        try:
            file1, rank1 = transform(gibo_col1, gibo_row1)
            file2, rank2 = transform(gibo_col2, gibo_row2)
            
            if not (0 <= file1 < 9 and 0 <= rank1 < 10):
                continue
            if not (0 <= file2 < 9 and 0 <= rank2 < 10):
                continue
            
            piece = board.get_piece(file1, rank1)
            if piece is None:
                continue
            
            if piece.side != board.side_to_move:
                continue
            
            if piece_char:
                expected_type = GibParser.HANJA_TO_PIECE.get(piece_char)
                if expected_type and piece.piece_type != expected_type:
                    continue
            
            move = Move(file1, rank1, file2, rank2)
            if board.is_legal_move(move):
                return move
                
        except (ValueError, KeyError, IndexError):
            continue
    
    return None


class GiboDataGenerator:
    """Generate training data from parsed game records."""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor() if TORCH_AVAILABLE else None
    
    def generate_from_games(
        self,
        games: List[Dict],
        positions_per_game: int = 50,
        progress_callback=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data from parsed games.
        
        Args:
            games: List of parsed game dictionaries
            positions_per_game: Max positions to extract from each game
            progress_callback: Optional callback(done, total) for progress
        
        Returns:
            (features, targets) arrays
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required")
        
        features_list = []
        targets_list = []
        
        total_games = len(games)
        successful_games = 0
        failed_games = 0
        
        for game_idx, game in enumerate(games):
            if progress_callback and game_idx % 50 == 0:
                progress_callback(game_idx, total_games)
            
            try:
                game_features, game_targets = self._process_game(
                    game, positions_per_game
                )
                
                if game_features and len(game_features) > 0:
                    features_list.extend(game_features)
                    targets_list.extend(game_targets)
                    successful_games += 1
                else:
                    failed_games += 1
                    
            except Exception as e:
                failed_games += 1
                if failed_games <= 10:  # Only print first 10 errors
                    print(f"Error processing game {game_idx}: {e}")
        
        print(f"Processed {successful_games} games successfully, {failed_games} failed")
        print(f"Generated {len(features_list)} positions")
        
        if len(features_list) == 0:
            raise ValueError("No positions generated from games")
        
        return np.array(features_list, dtype=np.float32), np.array(targets_list, dtype=np.float32)
    
    def generate_from_games_parallel(
        self,
        games: List[Dict],
        positions_per_game: int = 50,
        num_workers: Optional[int] = None,
        progress_callback=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        병렬 처리로 게임 데이터 생성 (안전한 버전)
        
        해결된 문제들:
        1. ✅ 인스턴스 변수 공유 문제: worker 함수에서 독립적으로 생성
        2. ✅ 공유 상태 카운터: 각 프로세스가 독립적으로 처리 후 합산
        3. ✅ 출력 충돌: 메인 프로세스에서만 진행 상황 출력
        4. ✅ 예외 처리: 예외 정보를 반환값에 포함
        5. ✅ 메모리 효율: 배치 단위로 처리
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required")
        
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        
        feature_size = self.feature_extractor.feature_size if self.feature_extractor else 512
        
        print(f"🚀 병렬 처리 시작: {len(games)}개 게임, {num_workers}개 워커")
        start_time = time.time()
        
        # Worker 함수에 전달할 인자 준비 (pickle 가능한 형태)
        args_list = [
            (game, positions_per_game, feature_size)
            for game in games
        ]
        
        # 결과 수집용 리스트 (메인 프로세스에서만 접근)
        all_features = []
        all_targets = []
        successful_games = 0
        failed_games = 0
        error_messages = []
        
        # 멀티프로세싱 풀 사용
        with mp.Pool(num_workers) as pool:
            # imap_unordered: 결과를 받는 대로 처리 (순서 보장 안 함, 더 빠름)
            # chunksize: 배치 크기 (너무 작으면 오버헤드, 너무 크면 불균형)
            chunksize = max(1, len(games) // (num_workers * 4))
            results = pool.imap_unordered(
                _process_single_game_worker, 
                args_list, 
                chunksize=chunksize
            )
            
            processed_count = 0
            for result in results:
                features, targets, success, error_msg = result
                
                if success and len(features) > 0:
                    all_features.extend(features)
                    all_targets.extend(targets)
                    successful_games += 1
                else:
                    failed_games += 1
                    if error_msg and len(error_messages) < 10:
                        error_messages.append(error_msg)
                
                processed_count += 1
                
                # 진행 상황 콜백 (메인 프로세스에서만 호출 - 출력 충돌 방지)
                if progress_callback and processed_count % 50 == 0:
                    progress_callback(processed_count, len(games))
        
        # 최종 결과 출력
        elapsed = time.time() - start_time
        print(f"\n✅ 처리 완료: {successful_games}개 성공, {failed_games}개 실패")
        print(f"📊 생성된 포지션: {len(all_features)}개")
        if elapsed > 0:
            print(f"⏱️  소요 시간: {elapsed:.1f}초 ({len(all_features)/elapsed:.1f} 포지션/초)")
        
        if error_messages:
            print(f"\n⚠️  오류 예시 (최대 10개):")
            for msg in error_messages[:10]:
                print(f"  - {msg}")
        
        if len(all_features) == 0:
            raise ValueError("생성된 포지션이 없습니다")
        
        return np.array(all_features, dtype=np.float32), np.array(all_targets, dtype=np.float32)
    
    def _find_valid_move(self, board: Board, from_coord: str, to_coord: str, 
                          piece_char: Optional[str], move_num: int) -> Optional[Move]:
        """Try to find a valid move from gibo coordinates.
        
        Tries multiple coordinate transformations to find a valid move.
        Returns the move if found, None otherwise.
        """
        gibo_col1, gibo_row1 = int(from_coord[0]), int(from_coord[1])
        gibo_col2, gibo_row2 = int(to_coord[0]), int(to_coord[1])
        
        # Try multiple coordinate transformations
        transformations = [
            # Transformation 1: Standard (col->file reversed, row mapped)
            lambda c, r: (9 - c if c > 0 else 8, {0:7,1:6,2:5,3:4,4:3,5:2,6:1,7:0,8:9,9:8}.get(r, r)),
            # Transformation 2: Simple reverse
            lambda c, r: (8 - c, 9 - r),
            # Transformation 3: Direct mapping
            lambda c, r: (c, r),
            # Transformation 4: Only column reverse
            lambda c, r: (8 - c, r),
            # Transformation 5: Column 1-9 to file 8-0
            lambda c, r: (9 - c, 9 - r),
        ]
        
        for transform in transformations:
            try:
                file1, rank1 = transform(gibo_col1, gibo_row1)
                file2, rank2 = transform(gibo_col2, gibo_row2)
                
                # Validate bounds
                if not (0 <= file1 < 9 and 0 <= rank1 < 10):
                    continue
                if not (0 <= file2 < 9 and 0 <= rank2 < 10):
                    continue
                
                # Check if there's a piece at source
                piece = board.get_piece(file1, rank1)
                if piece is None:
                    continue
                
                # Check if it's the right side's turn
                if piece.side != board.side_to_move:
                    continue
                
                # If piece type is specified, check it matches
                if piece_char:
                    expected_type = GibParser.HANJA_TO_PIECE.get(piece_char)
                    if expected_type and piece.piece_type != expected_type:
                        continue
                
                # Try to make the move
                move = Move(file1, rank1, file2, rank2)
                if board.is_legal_move(move):
                    return move
                    
            except (ValueError, KeyError, IndexError):
                continue
        
        return None
    
    def _process_game(
        self,
        game: Dict,
        max_positions: int
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Process a single game and extract positions.
        
        Returns:
            (features_list, targets_list)
        """
        features = []
        targets = []
        
        # Initialize board with formations
        cho_formation = game.get('cho_formation', '마상상마')
        han_formation = game.get('han_formation', '마상상마')
        result = game.get('result', None)
        raw_moves = game.get('raw_moves', [])
        
        if len(raw_moves) < 5:  # Skip very short games
            return features, targets
        
        # Create board with formations
        try:
            board = Board(
                cho_formation=cho_formation,
                han_formation=han_formation
            )
        except Exception:
            board = Board()
        
        # Determine target values based on game result
        if result == 'cho':
            cho_target = 1.0
            han_target = -1.0
        elif result == 'han':
            cho_target = -1.0
            han_target = 1.0
        else:
            cho_target = 0.0
            han_target = 0.0
        
        # Play through the game and collect positions
        positions_collected = 0
        failed_moves = 0
        total_moves = len(raw_moves)
        
        for move_idx, (from_coord, piece_char, to_coord) in enumerate(raw_moves):
            if positions_collected >= max_positions:
                break
            
            # Stop if too many failed moves (coordinate system likely wrong)
            if failed_moves > 5 and failed_moves > positions_collected:
                break
            
            # Extract features BEFORE making the move
            try:
                feat = self.feature_extractor.extract(board)
                
                if not np.isnan(feat).any():
                    # Calculate target value
                    progress = move_idx / max(total_moves - 1, 1)
                    
                    if board.side_to_move == Side.CHO:
                        base_target = cho_target
                    else:
                        base_target = han_target
                    
                    target = base_target * (0.3 + 0.7 * progress)
                    
                    features.append(feat)
                    targets.append(target)
                    positions_collected += 1
                    
            except Exception:
                pass
            
            # Try to find and make the move
            move = self._find_valid_move(board, from_coord, to_coord, piece_char, move_idx + 1)
            
            if move:
                if not board.make_move(move):
                    failed_moves += 1
            else:
                failed_moves += 1
                # If we can't find the move, try to continue by finding any legal move
                # This is a fallback to salvage some data
                legal_moves = board.generate_moves()
                if legal_moves:
                    # Make a random legal move to continue
                    import random
                    random_move = random.choice(legal_moves)
                    board.make_move(random_move)
                else:
                    break  # Game over
        
        return features, targets


def train_from_gibo(
    gibo_dir: str,
    nnue: 'NNUETorch',
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    positions_per_game: int = 50,
    output_file: str = 'models/nnue_gibo_model.json'
) -> Dict:
    """Train NNUE from gibo files with gradient clipping for stability.
    
    Args:
        gibo_dir: Directory containing .gib files
        nnue: NNUE model to train
        epochs: Training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        positions_per_game: Max positions per game
        output_file: Output model file
    
    Returns:
        Training history
    """
    # Parse gibo files
    parser = GibParser()
    games = parser.parse_directory(gibo_dir)
    
    if len(games) == 0:
        raise ValueError(f"No games found in {gibo_dir}")
    
    # Generate training data
    print(f"\nGenerating training data from {len(games)} games...")
    generator = GiboDataGenerator()
    
    def progress(done, total):
        print(f"\rProcessing games: {done}/{total}", end='', flush=True)
    
    features, targets = generator.generate_from_games(
        games,
        positions_per_game=positions_per_game,
        progress_callback=progress
    )
    print()
    
    print(f"Training on {len(features)} positions...")
    
    # Custom training with gradient clipping for stability
    history = train_with_gradient_clipping(
        nnue, features, targets,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Save model
    print(f"\nSaving model to {output_file}...")
    nnue.save(output_file)
    
    return history


def train_with_gradient_clipping(
    nnue: 'NNUETorch',
    features: np.ndarray,
    targets: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    grad_clip: float = 1.0,
    validation_split: float = 0.1
) -> Dict:
    """Train with gradient clipping for numerical stability."""
    import torch
    import torch.nn as nn
    
    device = nnue.device
    model = nnue.model
    
    # Split data
    n_samples = len(features)
    n_val = int(n_samples * validation_split)
    indices = np.random.permutation(n_samples)
    
    train_features = torch.tensor(features[indices[n_val:]], dtype=torch.float32, device=device)
    train_targets = torch.tensor(targets[indices[n_val:]], dtype=torch.float32, device=device).unsqueeze(1)
    val_features = torch.tensor(features[indices[:n_val]], dtype=torch.float32, device=device)
    val_targets = torch.tensor(targets[indices[:n_val]], dtype=torch.float32, device=device).unsqueeze(1)
    
    # Optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Training on {len(train_features)} samples, validating on {len(val_features)} samples")
    print(f"Device: {device}, Batch size: {batch_size}, Gradient clip: {grad_clip}")
    
    for epoch in range(epochs):
        model.train()
        
        # Shuffle training data
        perm = torch.randperm(len(train_features))
        train_features = train_features[perm]
        train_targets = train_targets[perm]
        
        train_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(train_features), batch_size):
            batch_features = train_features[i:i+batch_size]
            batch_targets = train_targets[i:i+batch_size]
            
            optimizer.zero_grad()
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"Warning: NaN loss at epoch {epoch+1}, batch {n_batches}")
                continue
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = train_loss / max(n_batches, 1)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_features)
            val_loss = criterion(val_outputs, val_targets).item()
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return history


def main():
    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required for NNUE training.")
        print("Install with: pip install torch")
        return
    
    parser = argparse.ArgumentParser(description='Train NNUE from Gibo files')
    
    # Input/Output
    parser.add_argument('--gibo-dir', type=str, default='gibo',
                        help='Directory containing .gib files')
    parser.add_argument('--output', type=str, default='models/nnue_gibo_model.json',
                        help='Output model file')
    parser.add_argument('--load', type=str, default=None,
                        help='Load existing model to continue training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--positions-per-game', type=int, default=50,
                        help='Max positions to extract from each game')
    
    # Model architecture (for new models)
    parser.add_argument('--feature-size', type=int, default=512,
                        help='Feature vector size')
    parser.add_argument('--hidden1', type=int, default=256,
                        help='First hidden layer size')
    parser.add_argument('--hidden2', type=int, default=64,
                        help='Second hidden layer size')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, mps, cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    
    print(f"Using device: {device}")
    
    # Check GPU info
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif device.type == 'mps':
        print("Using Apple Silicon GPU (MPS)")
    
    # Initialize or load model
    if args.load:
        print(f"\nLoading model from {args.load}...")
        nnue = NNUETorch.from_file(args.load, device=device)
    else:
        print("\nInitializing new model...")
        nnue = NNUETorch(
            feature_size=args.feature_size,
            hidden1_size=args.hidden1,
            hidden2_size=args.hidden2,
            device=device
        )
    
    print(f"Architecture: {args.feature_size} -> {args.hidden1} -> {args.hidden2} -> 1")
    
    # Train
    history = train_from_gibo(
        gibo_dir=args.gibo_dir,
        nnue=nnue,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        positions_per_game=args.positions_per_game,
        output_file=args.output
    )
    
    print(f"\nFinal train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    print("\nTraining complete!")


if __name__ == '__main__':
    main()

