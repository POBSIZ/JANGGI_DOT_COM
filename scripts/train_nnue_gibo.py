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
import json
from typing import List, Optional, Tuple, Dict, Callable
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Install with: pip install torch")

from janggi.board import Board, Side, PieceType, Move
from janggi.nnue import SimpleEvaluator

if TORCH_AVAILABLE:
    from janggi.nnue_torch import NNUETorch, FeatureExtractor, GPUTrainer, get_device

# ============================================================================
# Training Configuration Constants
# ============================================================================

# File Encoding Settings
SUPPORTED_ENCODINGS = ['euc-kr', 'cp949', 'utf-8']

# Game Processing Settings
MIN_GAME_MOVES = 5
DEFAULT_POSITIONS_PER_GAME = 50
MAX_FAILED_MOVES_THRESHOLD = 5
MAX_ERROR_MESSAGES_TO_DISPLAY = 10
MAX_PARSING_FAILURE_RATE = 0.3  # 30% 이상 실패 시 게임 제외

# Target Calculation Settings
TARGET_BASE_WEIGHT = 0.3
TARGET_PROGRESS_WEIGHT = 0.7
EVAL_WEIGHT = 0.7  # 평가 점수 가중치
RESULT_WEIGHT = 0.3  # 게임 결과 가중치
EVAL_SCALE = 10.0  # 평가 점수 정규화 스케일

# Training Settings
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_VALIDATION_SPLIT = 0.1
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_LR_SCHEDULER_FACTOR = 0.5
DEFAULT_LR_SCHEDULER_PATIENCE = 5
DEFAULT_EARLY_STOPPING_PATIENCE = 10
EVAL_INTERVAL = 10  # 10 epoch마다 중간 평가

# Progress Reporting
PROGRESS_UPDATE_FREQUENCY = 50


# ============================================================================
# Dynamic Weight Calculation Functions
# ============================================================================

def calculate_dynamic_weights(progress: float) -> Tuple[float, float]:
    """진행도에 따라 평가 점수와 게임 결과 가중치 조정.
    
    Args:
        progress: 게임 진행도 (0.0 ~ 1.0)
        
    Returns:
        (eval_weight, result_weight) 튜플
    """
    if progress < 0.3:  # 초반
        eval_weight = 0.8
        result_weight = 0.2
    elif progress < 0.7:  # 중반
        eval_weight = 0.6
        result_weight = 0.4
    else:  # 후반
        eval_weight = 0.4
        result_weight = 0.6
    
    return eval_weight, result_weight


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
    
    # 한글 기물 → 한자 기물 매핑
    KOREAN_TO_HANJA = {
        '졸': '卒',
        '병': '兵',
        '마': '馬',
        '상': '象',
        '사': '士',
        '장': '將',
        '차': '車',
        '포': '包',
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
            # Read file as bytes (to handle 0xff characters)
            with open(filepath, 'rb') as f:
                file_bytes = f.read()
            
            # Remove 0xff characters (as per reference implementation)
            ff_indices = [i for i, val in enumerate(file_bytes) if val == 0xff]
            if len(ff_indices) == 0:
                fixed_bytes = file_bytes
            else:
                fixed_bytes = b''
                ff_indices += [len(file_bytes)]  # Add last index
                i_start = 0
                for i in ff_indices:
                    fixed_bytes += file_bytes[i_start:i]
                    i_start = i + 1  # Skip 0xff
            
            # Try different encodings (cp949 preferred as per reference)
            content = None
            for encoding in ['cp949', 'euc-kr'] + SUPPORTED_ENCODINGS:
                try:
                    content = fixed_bytes.decode(encoding, errors='replace')
                    break
                except Exception:
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
        comment_found = False
        
        for line in lines:
            # Handle comments (as per reference implementation)
            if '{' in line:
                comment_found = True
            
            if comment_found:
                if '}' in line:
                    comment_found = False
                continue
            
            # Remove 0x1a character
            line = line.replace('\x1a', '')
            line = line.strip()
            
            # Empty line indicates end of game info
            if line == '':
                continue
            
            # Parse metadata
            if line.startswith('['):
                # Extract key and value from [key "value"] format
                if ' ' in line and '"' in line:
                    key_start = 1
                    key_end = line.index(' ')
                    value_start = line.index('"') + 1
                    value_end = -line[::-1].index('"') - 1
                    key = line[key_start:key_end]
                    value = line[value_start:value_end]
                    
                    if key == '초차림' or key == '초포진':
                        formation = self.FORMATION_MAP.get(value, value)
                        game['cho_formation'] = formation
                    elif key == '한차림' or key == '한포진':
                        formation = self.FORMATION_MAP.get(value, value)
                        game['han_formation'] = formation
                    elif key == '대국결과':
                        if '초' in value and ('승' in value or '완' in value or '楚' in value):
                            game['result'] = 'cho'
                        elif '한' in value and ('승' in value or '완' in value or '漢' in value):
                            game['result'] = 'han'
                        else:
                            game['result'] = 'draw'
            
            # Parse moves (lines starting with numbers)
            elif re.match(r'^\d+\.', line):
                moves_text.append(line)
        
        # Store raw moves for later parsing with game context
        all_moves_text = ' '.join(moves_text)
        game['raw_moves'] = self._extract_raw_moves(all_moves_text)
        
        return game
    
    def _extract_raw_moves(self, moves_text: str) -> List[Tuple[str, Optional[str], str, Optional[str]]]:
        """Extract raw move data: (from_coord, piece_char, to_coord, side_indicator)
        
        Supports multiple gibo formats:
        1. Korean format: 08마87 (숫자-한글기물-숫자)
        2. Chinese format: 79卒78 (숫자-한자기물-숫자)
        3. With side prefix: 41漢兵42 (숫자-진영-기물-숫자)
        4. Skip move: 한수쉼
        
        Based on reference implementation from:
        https://github.com/ladofa/janggi/blob/master/python_tf/gibo.py
        
        Returns: (from_coord, piece_char, to_coord, side_indicator)
        """
        moves = []
        
        # Split by move numbers (e.g., "1. 08마87 2. 12마33")
        words_pre = moves_text.split(' ')
        # Remove words with '<' (like <0>)
        words = [w for w in words_pre if '<' not in w]
        
        i = 0
        while i < len(words):
            # Skip move numbers (e.g., "1.", "2.")
            if re.match(r'^\d+\.?$', words[i]):
                i += 1
                continue
            
            # Check for skip move
            if words[i] == '한수쉼':
                # Skip move - add empty move
                moves.append(('00', None, '00', None))
                i += 1
                continue
            
            # Parse move: from_coord + piece + to_coord
            word_move = words[i]
            
            # Extract from position (first 2 digits)
            if len(word_move) < 2 or not word_move[0].isdigit():
                i += 1
                continue
            
            # Parse coordinates (as per reference implementation)
            # word_move[0] = fy (row), word_move[1] = fx (col)
            # Keep original string format for coordinates
            from_coord = word_move[0:2]
            
            # Find next digit position (for to_coord)
            number_pos = 2
            while number_pos < len(word_move) and not word_move[number_pos].isdigit():
                number_pos += 1
            
            if number_pos >= len(word_move) or number_pos + 1 >= len(word_move):
                i += 1
                continue
            
            # Extract to position (next 2 digits)
            to_coord = word_move[number_pos:number_pos+2]
            
            # Extract piece character (between coordinates)
            piece_char = None
            side_indicator = None
            
            # Look for piece characters between coordinates
            piece_text = word_move[2:number_pos] if number_pos > 2 else ''
            
            # Check for side indicator (漢/楚)
            if '漢' in piece_text:
                side_indicator = '漢'
                piece_text = piece_text.replace('漢', '')
            elif '楚' in piece_text:
                side_indicator = '楚'
                piece_text = piece_text.replace('楚', '')
            
            # Convert Korean piece to Chinese if needed
            if piece_text:
                # Check if it's Korean piece
                if piece_text in self.KOREAN_TO_HANJA:
                    piece_char = self.KOREAN_TO_HANJA[piece_text]
                # Check if it's already Chinese piece
                elif piece_text in self.HANJA_TO_PIECE:
                    piece_char = piece_text
                # Single character might be piece
                elif len(piece_text) == 1:
                    piece_char = piece_text
            
            moves.append((from_coord, piece_char, to_coord, side_indicator))
            i += 1
        
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

def _process_single_game_worker(args: Tuple[Dict, int, int]) -> Tuple[List[np.ndarray], List[float], bool, Optional[str], int, int]:
    """
    단일 게임을 처리하는 worker 함수 (모듈 레벨에 있어야 pickle 가능)
    
    Args:
        args: (game_dict, max_positions, feature_size) 튜플
        
    Returns:
        (features_list, targets_list, success, error_message, failed_moves, total_moves)
    """
    game, max_positions, feature_size = args
    
    # 각 프로세스가 독립적으로 FeatureExtractor 생성 (공유 상태 문제 해결)
    if not TORCH_AVAILABLE:
        return [], [], False, "PyTorch not available", 0, 0
    
    feature_extractor = FeatureExtractor(feature_size)
    simple_evaluator = SimpleEvaluator()
    features = []
    targets = []
    
    try:
        # 게임 데이터 추출
        cho_formation = game.get('cho_formation', '마상상마')
        han_formation = game.get('han_formation', '마상상마')
        result = game.get('result', None)
        raw_moves = game.get('raw_moves', [])
        
        if len(raw_moves) < MIN_GAME_MOVES:
            return features, targets, True, None, 0, len(raw_moves)
        
        # 보드 초기화
        try:
            board = Board(
                cho_formation=cho_formation,
                han_formation=han_formation
            )
        except Exception:
            board = Board()
        
        # 게임별 좌표 변환 감지 (처음 10수 사용)
        preferred_transform = _detect_coordinate_transformation(board, raw_moves, sample_size=10)
        
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
        
        for move_idx, move_data in enumerate(raw_moves):
            # move_data는 (from_coord, piece_char, to_coord, side_indicator) 또는 (from_coord, piece_char, to_coord)
            if len(move_data) == 4:
                from_coord, piece_char, to_coord, side_indicator = move_data
            else:
                from_coord, piece_char, to_coord = move_data
                side_indicator = None
            
            if positions_collected >= max_positions:
                break
            
            if failed_moves > MAX_FAILED_MOVES_THRESHOLD and failed_moves > positions_collected:
                break
            
            # Feature 추출
            try:
                feat = feature_extractor.extract(board)
                
                if not np.isnan(feat).any():
                    progress = move_idx / max(total_moves - 1, 1)
                    
                    # SimpleEvaluator로 평가 점수 계산
                    eval_score = simple_evaluator.evaluate(board)
                    
                    # 현재 side_to_move 관점에서 평가 점수 정규화
                    if board.side_to_move == Side.CHO:
                        normalized_eval = np.clip(eval_score / EVAL_SCALE, -1, 1)
                    else:
                        normalized_eval = np.clip(-eval_score / EVAL_SCALE, -1, 1)
                    
                    # 게임 결과 기반 타겟
                    if board.side_to_move == Side.CHO:
                        base_target = cho_target
                    else:
                        base_target = han_target
                    
                    result_target = base_target * (TARGET_BASE_WEIGHT + TARGET_PROGRESS_WEIGHT * progress)
                    
                    # 진행도 기반 동적 가중치 계산
                    eval_weight, result_weight = calculate_dynamic_weights(progress)
                    
                    # 평가 점수와 게임 결과 혼합 (동적 가중치 사용)
                    target = eval_weight * normalized_eval + result_weight * result_target
                    
                    features.append(feat)
                    targets.append(target)
                    positions_collected += 1
                    
            except Exception:
                pass
            
            # 수 찾기 및 실행 (진영 정보 포함, 감지된 변환 사용)
            move = _find_valid_move_helper(board, from_coord, to_coord, piece_char, side_indicator, preferred_transform)
            
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
        
        return features, targets, True, None, failed_moves, total_moves
        
    except Exception as e:
        # 예외 정보를 반환값에 포함 (디버깅 용이)
        return [], [], False, str(e), 0, 0


def _detect_coordinate_transformation(board: Board, raw_moves: List[Tuple], sample_size: int = 10) -> Optional[Callable]:
    """게임의 처음 몇 수를 사용하여 최적의 좌표 변환을 감지
    
    Args:
        board: 초기 보드 상태
        raw_moves: 원시 수 데이터 리스트
        sample_size: 분석할 수의 개수 (기본값: 10)
    
    Returns:
        최적의 변환 함수 또는 None
    """
    if not raw_moves:
        return None
    
    # 모든 가능한 변환 후보
    transformations = [
        ("File=Y-1, Rank=9-X", lambda r, c: (c - 1, 9 - r)),
        ("File=X-1, Rank=9-Y (반대)", lambda r, c: (r - 1, 9 - c)),
        ("File=9-Y, Rank=9-X", lambda r, c: (9 - c, 9 - r)),
        ("File=9-X, Rank=9-Y (반대)", lambda r, c: (9 - r, 9 - c)),
        ("File=첫자리, Rank=둘째자리", lambda r, c: (r, c)),
        ("File=Y-1, Rank=X", lambda r, c: (c - 1, r)),
        ("File=9-Y, Rank=X", lambda r, c: (9 - c, r)),
        ("File=X-1, Rank=Y", lambda r, c: (r - 1, c)),
        ("File=9-X, Rank=Y", lambda r, c: (9 - r, c)),
        ("File=Y, Rank=X", lambda r, c: (c, r)),
        ("File=Y, Rank=9-X", lambda r, c: (c, 9 - r)),
        ("File=둘째자리, Rank=첫자리", lambda r, c: (c, r)),
    ]
    
    # 각 변환의 성공 통계
    transform_stats = {name: {'success': 0, 'total': 0, 'perfect_match': 0} 
                       for name, _ in transformations}
    
    # 각 변환마다 별도의 보드로 테스트
    transform_boards = {}
    for trans_name, _ in transformations:
        try:
            transform_boards[trans_name] = Board(
                cho_formation=getattr(board, 'cho_formation', '마상상마'),
                han_formation=getattr(board, 'han_formation', '마상상마')
            )
        except Exception:
            transform_boards[trans_name] = Board()
    
    for move_idx, move_data in enumerate(raw_moves[:sample_size]):
        if len(move_data) == 4:
            from_coord, piece_char, to_coord, side_indicator = move_data
        else:
            from_coord, piece_char, to_coord = move_data
            side_indicator = None
        
        # Parse coordinates as per reference implementation
        # word_move[0] = fy (row), word_move[1] = fx (col)
        try:
            gibo_row = int(from_coord[0]) - 1
            gibo_col = int(from_coord[1]) - 1
            gibo_row2 = int(to_coord[0]) - 1
            gibo_col2 = int(to_coord[1]) - 1
            
            # Handle -1 (becomes 9 or 8) as per reference implementation
            if gibo_row == -1:
                gibo_row = 9
            if gibo_col == -1:
                gibo_col = 8
            if gibo_row2 == -1:
                gibo_row2 = 9
            if gibo_col2 == -1:
                gibo_col2 = 8
        except (ValueError, IndexError):
            continue
        
        # 진영 정보
        expected_side = None
        if side_indicator == '漢':
            expected_side = Side.HAN
        elif side_indicator == '楚':
            expected_side = Side.CHO
        
        # 기물 타입
        expected_piece_type = None
        if piece_char:
            expected_piece_type = GibParser.HANJA_TO_PIECE.get(piece_char)
        
        # 각 변환 시도
        for trans_name, transform in transformations:
            transform_stats[trans_name]['total'] += 1
            
            try:
                file1, rank1 = transform(gibo_row, gibo_col)
                file2, rank2 = transform(gibo_row2, gibo_col2)
                
                # 좌표 범위 검증
                if not (0 <= file1 < 9 and 0 <= rank1 < 10):
                    continue
                if not (0 <= file2 < 9 and 0 <= rank2 < 10):
                    continue
                
                # 각 변환마다 별도의 보드 사용
                test_board = transform_boards[trans_name]
                
                # 기물 존재 확인
                piece = test_board.get_piece(file1, rank1)
                if piece is None:
                    continue
                
                # 진영 정보 검증
                if expected_side is not None and piece.side != expected_side:
                    continue
                
                # 현재 턴 검증
                if piece.side != test_board.side_to_move:
                    continue
                
                # 기물 타입 검증
                if expected_piece_type is not None:
                    if piece.piece_type == expected_piece_type:
                        transform_stats[trans_name]['perfect_match'] += 1
                
                # 유효한 수인지 확인
                move = Move(file1, rank1, file2, rank2)
                if test_board.is_legal_move(move):
                    transform_stats[trans_name]['success'] += 1
                    # 성공한 변환으로 수 실행
                    test_board.make_move(move)
            except (ValueError, KeyError, IndexError):
                continue
    
    # 최고 성공률 변환 찾기 (기물 타입 일치 우선)
    best_transform = None
    best_score = -1
    
    for trans_name, stats in transform_stats.items():
        if stats['total'] == 0:
            continue
        
        # 연속 성공률 계산 (모든 샘플 수를 성공한 변환 우선)
        success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
        perfect_rate = stats['perfect_match'] / stats['total'] if stats['total'] > 0 else 0
        
        # 점수 계산: 연속 성공률 * 3 + 기물 타입 일치율 * 2 + 성공 횟수
        # 연속으로 성공한 변환이 더 신뢰할 수 있음
        score = success_rate * 3 + perfect_rate * 2 + (stats['success'] / sample_size)
        
        if score > best_score:
            best_score = score
            best_transform = next(transform for name, transform in transformations if name == trans_name)
    
    return best_transform


def _find_valid_move_helper(board: Board, from_coord: str, to_coord: str, 
                            piece_char: Optional[str], side_indicator: Optional[str] = None,
                            preferred_transform: Optional[Callable] = None) -> Optional[Move]:
    """좌표 변환 헬퍼 함수 (worker 함수 내부에서 사용)
    
    기보 파일은 3차 개정 좌표(신좌표)를 사용합니다:
    - 두 자리 숫자: XY (X=행 0-9, Y=열 1-9)
    - 예: 11=행1열1, 42=행4열2, 02=행0열2(10번째 행)
    
    보드 좌표 변환:
    - File = Y - 1 (열에서 1을 빼서 0-8로 변환)
    - Rank = X (행은 그대로 0-9)
    
    Args:
        board: 현재 보드 상태
        from_coord: 출발 좌표 (기보 형식, 예: "11", "42", "02")
        to_coord: 도착 좌표 (기보 형식)
        piece_char: 기물 한자 문자 (선택)
        side_indicator: 진영 표시 ('漢' or '楚', 선택)
    
    Returns:
        유효한 Move 객체 또는 None
    """
    # 3차 개정 좌표(신좌표) 파싱: 두 자리 숫자
    # XY 형식에서 X=행(0-9), Y=열(1-9)
    # 참고 문서 방식: word_move[0] = fy, word_move[1] = fx
    # -1을 9나 8로 변환하는 로직 포함
    try:
        gibo_row = int(from_coord[0]) - 1
        gibo_col = int(from_coord[1]) - 1
        gibo_row2 = int(to_coord[0]) - 1
        gibo_col2 = int(to_coord[1]) - 1
        
        # Handle -1 (becomes 9 or 8) as per reference implementation
        if gibo_row == -1:
            gibo_row = 9
        if gibo_col == -1:
            gibo_col = 8
        if gibo_row2 == -1:
            gibo_row2 = 9
        if gibo_col2 == -1:
            gibo_col2 = 8
    except (ValueError, IndexError):
        return None
    
    # 선호하는 변환이 있으면 먼저 시도 (하지만 실패하면 다른 변환도 시도)
    tried_preferred = False
    if preferred_transform is not None:
        tried_preferred = True
        try:
            file1, rank1 = preferred_transform(gibo_row, gibo_col)
            file2, rank2 = preferred_transform(gibo_row2, gibo_col2)
            
            if (0 <= file1 < 9 and 0 <= rank1 < 10 and 
                0 <= file2 < 9 and 0 <= rank2 < 10):
                piece = board.get_piece(file1, rank1)
                if piece is not None:
                    expected_side = None
                    if side_indicator == '漢':
                        expected_side = Side.HAN
                    elif side_indicator == '楚':
                        expected_side = Side.CHO
                    
                    if expected_side is None or piece.side == expected_side:
                        if piece.side == board.side_to_move:
                            expected_piece_type = None
                            if piece_char:
                                expected_piece_type = GibParser.HANJA_TO_PIECE.get(piece_char)
                            
                            if expected_piece_type is None or piece.piece_type == expected_piece_type:
                                move = Move(file1, rank1, file2, rank2)
                                if board.is_legal_move(move):
                                    return move
        except (ValueError, KeyError, IndexError):
            pass
    
    # 정확한 변환: File = Y - 1, Rank = 9 - X (기본값)
    file1 = gibo_col - 1
    rank1 = 9 - gibo_row
    file2 = gibo_col2 - 1
    rank2 = 9 - gibo_row2
    
    # 좌표 범위 검증
    if (0 <= file1 < 9 and 0 <= rank1 < 10 and 
        0 <= file2 < 9 and 0 <= rank2 < 10):
        # 정확한 변환이 가능하면 바로 시도
        piece = board.get_piece(file1, rank1)
        if piece is not None:
            # 진영 정보 검증
            expected_side = None
            if side_indicator == '漢':
                expected_side = Side.HAN
            elif side_indicator == '楚':
                expected_side = Side.CHO
            
            if expected_side is None or piece.side == expected_side:
                # 현재 턴 검증 (완화)
                if True:  # 턴 검증 완화
                    # 기물 타입 검증 (완화)
                    expected_piece_type = None
                    if piece_char:
                        expected_piece_type = GibParser.HANJA_TO_PIECE.get(piece_char)
                    
                    # 기물 타입이 맞거나 없으면 수 시도
                    if expected_piece_type is None or piece.piece_type == expected_piece_type:
                        move = Move(file1, rank1, file2, rank2)
                        if board.is_legal_move(move):
                            return move
    
    # 정확한 변환이 실패하면 다른 변환 시도 (하위 호환성)
    # 진영 정보로 예상되는 Side 결정
    expected_side = None
    if side_indicator == '漢':
        expected_side = Side.HAN
    elif side_indicator == '楚':
        expected_side = Side.CHO
    
    # 기물 타입 매핑
    expected_piece_type = None
    if piece_char:
        expected_piece_type = GibParser.HANJA_TO_PIECE.get(piece_char)
    
    # 모든 가능한 변환 후보 (100% 파싱을 위해 모두 시도)
    coordinate_transforms = [
        ("File=Y-1, Rank=9-X", lambda r, c: (c - 1, 9 - r)),  # 최고 성공률
        ("File=X-1, Rank=9-Y (반대)", lambda r, c: (r - 1, 9 - c)),  # 동일 성공률
        ("File=9-Y, Rank=9-X", lambda r, c: (9 - c, 9 - r)),
        ("File=9-X, Rank=9-Y (반대)", lambda r, c: (9 - r, 9 - c)),
        ("File=첫자리, Rank=둘째자리", lambda r, c: (r, c)),
        ("File=Y-1, Rank=X", lambda r, c: (c - 1, r)),
        ("File=9-Y, Rank=X", lambda r, c: (9 - c, r)),
        ("File=X-1, Rank=Y", lambda r, c: (r - 1, c)),
        ("File=9-X, Rank=Y", lambda r, c: (9 - r, c)),
        ("File=Y, Rank=X", lambda r, c: (c, r)),
        ("File=Y, Rank=9-X", lambda r, c: (c, 9 - r)),
        ("File=둘째자리, Rank=첫자리", lambda r, c: (c, r)),
    ]
    
    # 선호하는 변환이 이미 시도되었으면 제외 (중복 방지)
    if tried_preferred and preferred_transform is not None:
        coordinate_transforms = [t for t in coordinate_transforms 
                                 if t[1] != preferred_transform]
    
    # 추가 변환 시도
    for trans_name, transform in coordinate_transforms:
        try:
            file1, rank1 = transform(gibo_row, gibo_col)
            file2, rank2 = transform(gibo_row2, gibo_col2)
            
            # 좌표 범위 검증
            if not (0 <= file1 < 9 and 0 <= rank1 < 10):
                continue
            if not (0 <= file2 < 9 and 0 <= rank2 < 10):
                continue
            
            # 기물 존재 확인
            piece = board.get_piece(file1, rank1)
            if piece is None:
                continue
            
            # 진영 정보 검증 (완화: 진영 정보가 없거나 맞으면 시도)
            if expected_side is not None and piece.side != expected_side:
                # 진영이 맞지 않아도 일단 시도 (기보 파일의 진영 정보가 부정확할 수 있음)
                pass  # 진영 검증 완화
            
            # 현재 턴 검증 (완화: wrong_turn이어도 일단 시도)
            # wrong_turn이지만 다른 조건은 맞으면 일단 시도
            # (기보 파일의 턴 정보가 부정확할 수 있음)
            
            # 기물 타입 검증 (완화)
            # 기물 타입이 맞지 않아도 일단 시도
            
            # 유효한 수인지 확인 (가장 중요한 검증)
            move = Move(file1, rank1, file2, rank2)
            if board.is_legal_move(move):
                # 유효한 수를 찾았으면 즉시 반환
                return move
        except (ValueError, KeyError, IndexError):
            continue
    
    # 정확한 변환이 실패하면 기존 변환 방식 시도 (하위 호환성)
    # 일부 기보 파일이 다른 형식을 사용할 수 있음
    gibo_col1_old, gibo_row1_old = int(from_coord[0]), int(from_coord[1])
    gibo_col2_old, gibo_row2_old = int(to_coord[0]), int(to_coord[1])
    
    # 진영 정보로 예상되는 Side 결정
    expected_side = None
    if side_indicator == '漢':
        expected_side = Side.HAN
    elif side_indicator == '楚':
        expected_side = Side.CHO
    
    # 기물 타입 매핑
    expected_piece_type = None
    if piece_char:
        expected_piece_type = GibParser.HANJA_TO_PIECE.get(piece_char)
    
    # 확장된 변환 후보 (성공률이 높은 순서대로)
    # 더 많은 변환 후보 추가하여 커버리지 향상
    transformations = [
        # t1: 기본 변환 (가장 높은 성공률)
        ("t1", lambda c, r: (9 - c if c > 0 else 8, {0:7,1:6,2:5,3:4,4:3,5:2,6:1,7:0,8:9,9:8}.get(r, r))),
        # t7: Column reverse, row direct
        ("t7", lambda c, r: (9 - c if c > 0 else 8, r)),
        # t5: Column reverse, row reverse
        ("t5", lambda c, r: (9 - c if c > 0 else 8, 9 - r)),
        # t3: Direct mapping
        ("t3", lambda c, r: (c, r)),
        # t2: Column reverse (8-c), row reverse
        ("t2", lambda c, r: (8 - c, 9 - r)),
        # 추가 변환 후보
        ("t9", lambda c, r: (8 - c if c > 0 else 8, {0:9,1:8,2:7,3:6,4:5,5:4,6:3,7:2,8:1,9:0}.get(r, r))),
        ("t10", lambda c, r: (c, 9 - r)),
        ("t11", lambda c, r: (8 - c, r)),
        ("t12", lambda c, r: (c if c > 0 else 0, {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9}.get(r, r))),
        # 추가 변환 후보 (더 많은 패턴 시도)
        ("t13", lambda c, r: (c if c < 9 else 8, {0:9,1:8,2:7,3:6,4:5,5:4,6:3,7:2,8:1,9:0}.get(r, r))),
        ("t14", lambda c, r: (9 - c if c > 0 else 0, {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9}.get(r, r))),
        ("t15", lambda c, r: (8 - c if c < 9 else 0, r)),
        ("t16", lambda c, r: (c, {0:9,1:8,2:7,3:6,4:5,5:4,6:3,7:2,8:1,9:0}.get(r, r))),
    ]
    
    # 각 변환 시도
    for trans_name, transform in transformations:
        try:
            file1, rank1 = transform(gibo_col1_old, gibo_row1_old)
            file2, rank2 = transform(gibo_col2_old, gibo_row2_old)
            
            # 좌표 범위 검증
            if not (0 <= file1 < 9 and 0 <= rank1 < 10):
                continue
            if not (0 <= file2 < 9 and 0 <= rank2 < 10):
                continue
            
            # 기물 존재 확인
            piece = board.get_piece(file1, rank1)
            if piece is None:
                continue
            
            # 진영 정보 검증 (강화) - 먼저 확인 (더 정확함)
            if expected_side is not None and piece.side != expected_side:
                continue
            
            # 현재 턴 검증 (강화)
            if piece.side != board.side_to_move:
                continue
            
            # 기물 타입 검증 (완화) - 기물 타입이 있으면 검증, 없으면 건너뛰기
            # 기물 타입 불일치 시 다른 변환 시도 (너무 엄격하지 않음)
            if expected_piece_type is not None:
                if piece.piece_type != expected_piece_type:
                    # 기물 타입이 맞지 않지만, 다른 조건은 맞으면 일단 시도
                    # (기보 파일의 기물 정보가 부정확할 수 있음)
                    pass  # 일단 기물 타입 검증을 완화
            
            # 유효한 수인지 확인
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
        positions_per_game: int = DEFAULT_POSITIONS_PER_GAME,
        progress_callback: Optional[Callable] = None
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
            if progress_callback and game_idx % PROGRESS_UPDATE_FREQUENCY == 0:
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
                if failed_games <= MAX_ERROR_MESSAGES_TO_DISPLAY:
                    print(f"Error processing game {game_idx}: {e}")
        
        print(f"Processed {successful_games} games successfully, {failed_games} failed")
        print(f"Generated {len(features_list)} positions")
        
        if len(features_list) == 0:
            raise ValueError("No positions generated from games")
        
        return np.array(features_list, dtype=np.float32), np.array(targets_list, dtype=np.float32)
    
    def generate_from_games_parallel(
        self,
        games: List[Dict],
        positions_per_game: int = DEFAULT_POSITIONS_PER_GAME,
        num_workers: Optional[int] = None,
        progress_callback: Optional[Callable] = None
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
        
        # 파싱 통계 수집
        parsing_stats = {
            'total_games': len(games),
            'successful_games': 0,
            'failed_games': 0,
            'total_positions': 0,
            'total_failed_moves': 0,
            'total_attempted_moves': 0,
            'games_with_high_failure_rate': []  # 실패율 > 30%인 게임
        }
        
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
                features, targets, success, error_msg, failed_moves, total_moves = result
                
                # 파싱 통계 업데이트
                parsing_stats['total_failed_moves'] += failed_moves
                parsing_stats['total_attempted_moves'] += total_moves
                
                if success and len(features) > 0:
                    # 실패율 계산
                    failure_rate = failed_moves / total_moves if total_moves > 0 else 0.0
                    
                    # 고실패율 게임 필터링
                    if failure_rate > MAX_PARSING_FAILURE_RATE:
                        parsing_stats['games_with_high_failure_rate'].append({
                            'failure_rate': failure_rate,
                            'failed_moves': failed_moves,
                            'total_moves': total_moves
                        })
                        failed_games += 1
                    else:
                        all_features.extend(features)
                        all_targets.extend(targets)
                        successful_games += 1
                        parsing_stats['successful_games'] += 1
                        parsing_stats['total_positions'] += len(features)
                else:
                    failed_games += 1
                    parsing_stats['failed_games'] += 1
                    if error_msg and len(error_messages) < 10:
                        error_messages.append(error_msg)
                
                processed_count += 1
                
                # 진행 상황 콜백 (메인 프로세스에서만 호출 - 출력 충돌 방지)
                if progress_callback and processed_count % PROGRESS_UPDATE_FREQUENCY == 0:
                    progress_callback(processed_count, len(games))
        
        # 파싱 통계 리포트 출력
        elapsed = time.time() - start_time
        print(f"\n📊 기보 파싱 통계:")
        print(f"  - 총 게임: {parsing_stats['total_games']}개")
        print(f"  - 성공: {parsing_stats['successful_games']}개 ({parsing_stats['successful_games']/max(parsing_stats['total_games'], 1)*100:.1f}%)")
        print(f"  - 실패: {parsing_stats['failed_games']}개 ({parsing_stats['failed_games']/max(parsing_stats['total_games'], 1)*100:.1f}%)")
        if parsing_stats['total_attempted_moves'] > 0:
            avg_failure_rate = parsing_stats['total_failed_moves'] / parsing_stats['total_attempted_moves']
            print(f"  - 평균 실패율: {avg_failure_rate*100:.1f}%")
        print(f"  - 고실패율 게임 제외: {len(parsing_stats['games_with_high_failure_rate'])}개")
        print(f"\n✅ 처리 완료: {successful_games}개 성공, {failed_games}개 실패")
        print(f"📊 생성된 포지션: {len(all_features)}개")
        if elapsed > 0:
            print(f"⏱️  소요 시간: {elapsed:.1f}초 ({len(all_features)/elapsed:.1f} 포지션/초)")
        
        if error_messages:
            print(f"\n⚠️  오류 예시 (최대 {MAX_ERROR_MESSAGES_TO_DISPLAY}개):")
            for msg in error_messages[:MAX_ERROR_MESSAGES_TO_DISPLAY]:
                print(f"  - {msg}")
        
        if len(all_features) == 0:
            raise ValueError("생성된 포지션이 없습니다")
        
        return np.array(all_features, dtype=np.float32), np.array(all_targets, dtype=np.float32)
    
    def _find_valid_move(self, board: Board, from_coord: str, to_coord: str, 
                          piece_char: Optional[str], move_num: int, side_indicator: Optional[str] = None) -> Optional[Move]:
        """Try to find a valid move from gibo coordinates.
        
        Tries multiple coordinate transformations to find a valid move.
        Returns the move if found, None otherwise.
        
        Args:
            board: Current board state
            from_coord: Source coordinate (gibo format)
            to_coord: Destination coordinate (gibo format)
            piece_char: Piece type character (optional)
            move_num: Move number (for debugging)
            side_indicator: Side indicator ('漢' or '楚', optional)
        """
        # _find_valid_move_helper를 재사용
        return _find_valid_move_helper(board, from_coord, to_coord, piece_char, side_indicator)
    
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
        
        if len(raw_moves) < MIN_GAME_MOVES:  # Skip very short games
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
        
        # SimpleEvaluator 인스턴스 생성
        simple_evaluator = SimpleEvaluator()
        
        # Play through the game and collect positions
        positions_collected = 0
        failed_moves = 0
        total_moves = len(raw_moves)
        
        for move_idx, move_data in enumerate(raw_moves):
            # move_data는 (from_coord, piece_char, to_coord, side_indicator) 또는 (from_coord, piece_char, to_coord)
            if len(move_data) == 4:
                from_coord, piece_char, to_coord, side_indicator = move_data
            else:
                from_coord, piece_char, to_coord = move_data
                side_indicator = None
            
            if positions_collected >= max_positions:
                break
            
            # Stop if too many failed moves (coordinate system likely wrong)
            if failed_moves > MAX_FAILED_MOVES_THRESHOLD and failed_moves > positions_collected:
                break
            
            # Extract features BEFORE making the move
            try:
                feat = self.feature_extractor.extract(board)
                
                if not np.isnan(feat).any():
                    progress = move_idx / max(total_moves - 1, 1)
                    
                    # SimpleEvaluator로 평가 점수 계산
                    eval_score = simple_evaluator.evaluate(board)
                    
                    # 현재 side_to_move 관점에서 평가 점수 정규화
                    if board.side_to_move == Side.CHO:
                        normalized_eval = np.clip(eval_score / EVAL_SCALE, -1, 1)
                    else:
                        normalized_eval = np.clip(-eval_score / EVAL_SCALE, -1, 1)
                    
                    # 게임 결과 기반 타겟
                    if board.side_to_move == Side.CHO:
                        base_target = cho_target
                    else:
                        base_target = han_target
                    
                    result_target = base_target * (TARGET_BASE_WEIGHT + TARGET_PROGRESS_WEIGHT * progress)
                    
                    # 진행도 기반 동적 가중치 계산
                    eval_weight, result_weight = calculate_dynamic_weights(progress)
                    
                    # 평가 점수와 게임 결과 혼합 (동적 가중치 사용)
                    target = eval_weight * normalized_eval + result_weight * result_target
                    
                    features.append(feat)
                    targets.append(target)
                    positions_collected += 1
                    
            except Exception:
                pass
            
            # Try to find and make the move (진영 정보 포함)
            move = self._find_valid_move(board, from_coord, to_coord, piece_char, move_idx + 1, side_indicator)
            
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
    positions_per_game: int = DEFAULT_POSITIONS_PER_GAME,
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
    
    # 중간 평가 콜백 함수 정의
    def eval_callback(model):
        """중간 평가를 수행하는 콜백 함수"""
        try:
            from scripts.train_nnue_gpu import evaluate_model
            return evaluate_model(model, num_games=5, search_depth=3)
        except Exception as e:
            print(f"평가 함수 import 실패: {e}")
            return 0.0
    
    # Custom training with gradient clipping for stability
    history = train_with_gradient_clipping(
        nnue, features, targets,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        eval_callback=eval_callback
    )
    
    # Save model
    print(f"\nSaving model to {output_file}...")
    nnue.save(output_file)
    
    # 학습 히스토리 저장
    history_file = output_file.replace('.json', '_history.json')
    try:
        # JSON 직렬화 가능한 형태로 변환
        history_serializable = {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'learning_rate': [float(x) for x in history['learning_rate']],
            'grad_norm': [float(x) for x in history['grad_norm']],
        }
        if 'win_rates' in history:
            history_serializable['win_rates'] = [
                {'epoch': int(x['epoch']), 'win_rate': float(x['win_rate'])}
                for x in history['win_rates']
            ]
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_serializable, f, indent=2, ensure_ascii=False)
        print(f"학습 히스토리 저장: {history_file}")
    except Exception as e:
        print(f"히스토리 저장 실패: {e}")
    
    return history


def train_with_gradient_clipping(
    nnue: 'NNUETorch',
    features: np.ndarray,
    targets: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    eval_callback: Optional[Callable] = None
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
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=DEFAULT_WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=DEFAULT_LR_SCHEDULER_FACTOR, 
        patience=DEFAULT_LR_SCHEDULER_PATIENCE
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'learning_rate': [], 'grad_norm': []}
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
        
        # Gradient norm 계산
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_features)
            val_loss = criterion(val_outputs, val_targets).item()
        
        # Learning rate 가져오기
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        history['grad_norm'].append(total_norm)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"LR: {current_lr:.6e}, "
              f"Grad Norm: {total_norm:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # 중간 평가 (N epoch마다)
        if eval_callback and (epoch + 1) % EVAL_INTERVAL == 0:
            print(f"\n📊 Epoch {epoch+1} 중간 평가 중...")
            try:
                win_rate = eval_callback(nnue)
                history.setdefault('win_rates', []).append({
                    'epoch': epoch + 1,
                    'win_rate': win_rate
                })
                print(f"  승률: {win_rate*100:.1f}%")
            except Exception as e:
                print(f"  평가 실패: {e}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= DEFAULT_EARLY_STOPPING_PATIENCE:
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

