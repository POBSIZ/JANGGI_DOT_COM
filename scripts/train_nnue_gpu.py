#!/usr/bin/env python3
"""GPU-accelerated NNUE Training Script for Janggi.

This script uses PyTorch for GPU-accelerated training.

Usage:
    # Basic GPU training
    python train_nnue_gpu.py --positions 10000 --epochs 50
    
    # With larger batch size for GPU
    python train_nnue_gpu.py --positions 50000 --batch-size 512 --epochs 100
    
    # Iterative self-improvement
    python train_nnue_gpu.py --method iterative --iterations 10
"""

import argparse
import copy
import random
import time
import os
import multiprocessing as mp
from typing import Tuple, Optional
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Install with: pip install torch")

from janggi.board import Board, Side, PieceType
from janggi.engine import Engine

if TORCH_AVAILABLE:
    from janggi.nnue_torch import NNUETorch, FeatureExtractor, GPUTrainer, get_device

# ============================================================================
# Training Configuration Constants
# ============================================================================

# Game Generation Settings
MAX_MOVES_PER_GAME = 200
MAX_MOVES_EVAL = 200
RANDOM_MOVE_PROBABILITY = 0.2

# Opening Settings
RANDOM_OPENING_MIN = 2
RANDOM_OPENING_MAX = 6
RANDOM_OPENING_QUALITY_MIN = 4
RANDOM_OPENING_QUALITY_MAX = 25

# Position Generation Settings
POSITIONS_PER_GAME_PARALLEL = 30
POSITIONS_PER_GAME_FAST = 30
POSITIONS_PER_GAME_QUALITY = 20

# Evaluation Settings
EVAL_WEIGHT = 0.7
RESULT_WEIGHT = 0.3
EVAL_NORMALIZATION = 72.0
EVAL_BIAS = 1.5
EVAL_SCALE = 10.0
TWO_PLY_SEARCH_LIMIT = 150
MAX_MOVES_TO_EVAL_SELFPLAY = 20
MAX_MOVES_TO_EVAL_WORKER = 150  # Increased from 25 to 150 for better GPU utilization
MAX_OPPONENT_RESPONSES = 5

# Progress Reporting Settings
PROGRESS_UPDATE_INTERVAL = 50
PROGRESS_UPDATE_INTERVAL_SECONDS = 1.0
PROGRESS_UPDATE_INTERVAL_GAMES = 10
PROGRESS_UPDATE_INTERVAL_EVAL = 5
PROGRESS_UPDATE_INTERVAL_BATCH = 10

# Piece Values for Fast Evaluation
PIECE_VALUES = {
    PieceType.KING: 0,
    PieceType.ROOK: 13,
    PieceType.CANNON: 7,
    PieceType.HORSE: 5,
    PieceType.ELEPHANT: 3,
    PieceType.GUARD: 3,
    PieceType.PAWN: 2,
}

# GPU Batch Size Thresholds (in GB)
GPU_BATCH_SIZE_THRESHOLDS = {
    16: 1024,  # Large GPU
    8: 768,    # Medium GPU
    4: 512,    # Small GPU
    0: 256,    # Very small GPU or CPU
}


def get_optimal_batch_size(device: Optional['torch.device'] = None, default: int = 512) -> int:
    """Calculate optimal batch size based on GPU memory.
    
    Args:
        device: PyTorch device (if None, will try to detect)
        default: Default batch size if GPU detection fails
    
    Returns:
        Optimal batch size based on available GPU memory
    """
    if not TORCH_AVAILABLE:
        return default
    
    try:
        if device is None:
            device = get_device()
        
        if device.type == 'cuda':
            # Get GPU memory in GB
            gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
            
            # Find appropriate batch size based on GPU memory thresholds
            # Feature size is typically 512, so each position uses ~2KB
            # We want to use ~50-70% of GPU memory for batch evaluation
            for threshold, batch_size in sorted(GPU_BATCH_SIZE_THRESHOLDS.items(), reverse=True):
                if gpu_memory_gb >= threshold:
                    return batch_size
            return default
        else:
            # CPU or MPS: use smaller batches
            return GPU_BATCH_SIZE_THRESHOLDS[0]
    except Exception:
        # If detection fails, return default
        return default


def fast_evaluate_material(board: Board) -> float:
    """Fast material-based evaluation (no search).
    
    Args:
        board: Board position to evaluate
        
    Returns:
        Normalized evaluation score from CHO's perspective
    """
    han_material = sum(
        PIECE_VALUES.get(piece.piece_type, 0)
        for rank in range(board.RANKS)
        for file in range(board.FILES)
        for piece in [board.get_piece(file, rank)]
        if piece is not None and piece.side == Side.HAN
    )
    
    cho_material = sum(
        PIECE_VALUES.get(piece.piece_type, 0)
        for rank in range(board.RANKS)
        for file in range(board.FILES)
        for piece in [board.get_piece(file, rank)]
        if piece is not None and piece.side == Side.CHO
    )
    
    material_diff = cho_material - han_material
    if board.side_to_move == Side.HAN:
        material_diff = -material_diff + EVAL_BIAS
    
    return material_diff / EVAL_NORMALIZATION


def calculate_game_result(board: Board) -> float:
    """Calculate game result: 1.0 for HAN win, -1.0 for CHO win, 0.0 for draw."""
    try:
        if board.is_checkmate():
            return -1.0 if board.side_to_move == Side.CHO else 1.0
        return 0.0
    except Exception:
        return 0.0


def play_random_opening(board: Board, min_moves: int = RANDOM_OPENING_MIN, max_moves: int = RANDOM_OPENING_MAX) -> None:
    """Play random opening moves."""
    num_moves = random.randint(min_moves, max_moves)
    for _ in range(num_moves):
        moves = board.generate_moves()
        if not moves:
            break
        move = random.choice(moves)
        board.make_move(move)


def calculate_target_from_result(eval_score: float, result: float, progress: float) -> float:
    """Calculate target value blending evaluation with game outcome."""
    return EVAL_WEIGHT * np.clip(eval_score / EVAL_SCALE, -1, 1) + RESULT_WEIGHT * result * progress


def update_progress_callback(
    progress_callback,
    current: int,
    total: int,
    start_time: float,
    last_print: float = None
) -> float:
    """Update progress callback if needed. Returns updated last_print time."""
    if progress_callback is None:
        return last_print if last_print else time.time()
    
    current_time = time.time()
    if last_print is None or current_time - last_print >= PROGRESS_UPDATE_INTERVAL_SECONDS:
        elapsed = current_time - start_time
        speed = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / speed if speed > 0 else 0
        progress_callback(current, total, speed, eta)
        return current_time
    return last_print


# Worker function for multiprocessing (must be at module level)
def _generate_game_data(args):
    """Generate data for a single game (worker function)."""
    game_idx, feature_size, positions_per_game = args
    
    feature_extractor = FeatureExtractor(feature_size)
    features_list = []
    targets_list = []
    board = Board()
    
    # Play random game
    for move_count in range(positions_per_game * 3):
        if len(features_list) >= positions_per_game:
            break
        
        moves = board.generate_moves()
        if not moves:
            break
        
        # Store position (with NaN check)
        features = feature_extractor.extract(board)
        eval_score = fast_evaluate_material(board)
        
        # Skip if any NaN values
        if not np.isnan(features).any() and not np.isnan(eval_score):
            features_list.append(features)
            targets_list.append(eval_score)
        
        # Random move
        move = random.choice(moves)
        board.make_move(move)
    
    return features_list, targets_list


def _generate_selfplay_game_worker(args):
    """
    Generate a single self-play game (worker function for multiprocessing).
    
    NOTE: This version loads NNUE model in worker and uses it for move selection.
    This ensures self-play games reflect the current model's characteristics.
    
    Returns:
        (game_boards, game_features, game_result, num_positions) where:
        - game_boards: List of Board objects (pickled)
        - game_features: List of feature arrays
        - game_result: float (-1.0, 0.0, or 1.0)
        - num_positions: int (number of positions in this game)
    """
    game_idx, feature_size, max_moves, search_depth, random_seed, model_path = args
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    feature_extractor = FeatureExtractor(feature_size)
    
    # Load NNUE model in worker (CPU mode for multiprocessing compatibility)
    nnue_model = None
    if model_path and TORCH_AVAILABLE and os.path.exists(model_path):
        try:
            # Load model on CPU for multiprocessing compatibility
            nnue_model = NNUETorch.from_file(model_path, device=torch.device('cpu'))
            engine = Engine(depth=search_depth, use_nnue=True)
            engine.nnue = nnue_model
        except Exception as e:
            # Fallback to simple evaluator if model loading fails
            print(f"Warning: Failed to load model in worker {game_idx}: {e}")
            engine = Engine(depth=search_depth, use_nnue=False)
    else:
        # Fallback to simple evaluator if no model path
        engine = Engine(depth=search_depth, use_nnue=False)
    
    board = Board()
    game_boards = []
    game_features = []
    
    # Random opening
    play_random_opening(board, RANDOM_OPENING_MIN, RANDOM_OPENING_MAX)
    
    # Play game
    for move_count in range(max_moves):
        try:
            if board.is_checkmate() or board.is_stalemate():
                break
        except Exception:
            break
        
        # Store position (board and features)
        features = feature_extractor.extract(board)
        
        # Skip if NaN values
        if np.isnan(features).any():
            break
        
        # Store board state (will be pickled)
        game_boards.append(copy.deepcopy(board))
        game_features.append(features)
        
        # Get move using NNUE model (or simple evaluator as fallback)
        moves = board.generate_moves()
        if not moves:
            break
        
        if random.random() < RANDOM_MOVE_PROBABILITY:
            # Random move for exploration
            move = random.choice(moves)
        else:
            # Use engine with NNUE model for move selection
            move = engine.search(board)
            if move is None:
                move = random.choice(moves)
        
        board.make_move(move)
    
    # Determine game result
    result = calculate_game_result(board)
    
    return game_boards, game_features, result, len(game_boards)


class DataGenerator:
    """Generate training data for GPU training."""
    
    def __init__(self, search_depth: int = 3):
        self.search_depth = search_depth
        self.feature_extractor = FeatureExtractor() if TORCH_AVAILABLE else None
    
    def _fast_evaluate(self, board: Board) -> float:
        """Fast material-based evaluation (no search)."""
        return fast_evaluate_material(board)
    
    def generate_positions_parallel(
        self,
        num_positions: int = 10000,
        num_workers: int = None,
        progress_callback=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parallel position generation using multiprocessing."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GPU training")
        
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        
        positions_per_game = POSITIONS_PER_GAME_PARALLEL
        games_needed = (num_positions + positions_per_game - 1) // positions_per_game
        
        print(f"Generating data with {num_workers} workers...")
        start_time = time.time()
        
        # Prepare arguments for workers
        args_list = [
            (i, self.feature_extractor.feature_size, positions_per_game) 
            for i in range(games_needed)
        ]
        
        features_list = []
        targets_list = []
        
        # Use multiprocessing pool
        with mp.Pool(num_workers) as pool:
            results = pool.imap_unordered(_generate_game_data, args_list, chunksize=10)
            
            for i, (feats, targs) in enumerate(results):
                features_list.extend(feats)
                targets_list.extend(targs)
                
                if len(features_list) >= num_positions:
                    break
                
                if progress_callback and (i + 1) % PROGRESS_UPDATE_INTERVAL == 0:
                    elapsed = time.time() - start_time
                    speed = len(features_list) / elapsed if elapsed > 0 else 0
                    eta = (num_positions - len(features_list)) / speed if speed > 0 else 0
                    progress_callback(len(features_list), num_positions, speed, eta)
        
        # Trim to exact size
        features_list = features_list[:num_positions]
        targets_list = targets_list[:num_positions]
        
        elapsed = time.time() - start_time
        print(f"Generated {len(features_list)} positions in {elapsed:.1f}s ({len(features_list)/elapsed:.0f}/s)")
        
        return np.array(features_list, dtype=np.float32), np.array(targets_list, dtype=np.float32)
    
    def generate_positions_fast(
        self,
        num_positions: int = 10000,
        progress_callback=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fast position generation using random games and simple evaluation."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GPU training")
        
        features_list = []
        targets_list = []
        
        positions_per_game = POSITIONS_PER_GAME_FAST
        games_needed = (num_positions + positions_per_game - 1) // positions_per_game
        
        start_time = time.time()
        last_print = start_time
        
        for game_idx in range(games_needed):
            if len(features_list) >= num_positions:
                break
            
            board = Board()
            game_positions = []
            
            # Play random game
            for move_count in range(100):
                if len(features_list) + len(game_positions) >= num_positions:
                    break
                
                moves = board.generate_moves()
                if not moves:
                    break
                
                # Store position with fast evaluation (with NaN check)
                features = self.feature_extractor.extract(board)
                eval_score = self._fast_evaluate(board)
                
                # Only add if no NaN values
                if not np.isnan(features).any() and not np.isnan(eval_score):
                    game_positions.append((features, eval_score))
                
                # Random move
                move = random.choice(moves)
                board.make_move(move)
            
            # Add positions from this game
            for feat, eval_score in game_positions:
                features_list.append(feat)
                targets_list.append(eval_score)
            
            # Progress update every second
            last_print = update_progress_callback(
                progress_callback,
                len(features_list),
                num_positions,
                start_time,
                last_print
            )
        
        return np.array(features_list, dtype=np.float32), np.array(targets_list, dtype=np.float32)
    
    def generate_positions(
        self,
        num_positions: int = 10000,
        search_depth: int = 2,  # Reduced default depth
        progress_callback=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate positions with evaluations (slower but higher quality)."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GPU training")
        
        # Note: search_depth is kept for API compatibility but we use fast eval
        _ = search_depth
        
        features_list = []
        targets_list = []
        
        positions_per_game = POSITIONS_PER_GAME_QUALITY
        games_needed = (num_positions + positions_per_game - 1) // positions_per_game
        
        start_time = time.time()
        last_print = start_time
        
        for game_idx in range(games_needed):
            if len(features_list) >= num_positions:
                break
            
            board = Board()
            
            # Random opening
            play_random_opening(board, RANDOM_OPENING_QUALITY_MIN, RANDOM_OPENING_QUALITY_MAX)
            
            # Collect positions
            for _ in range(positions_per_game):
                if len(features_list) >= num_positions:
                    break
                
                try:
                    if board.is_checkmate() or board.is_stalemate():
                        break
                except Exception:
                    break
                
                # Extract features
                features = self.feature_extractor.extract(board)
                
                # Get target evaluation (simple, no deep search)
                target = self._fast_evaluate(board)
                
                features_list.append(features)
                targets_list.append(target)
                
                # Make a random move (fast)
                moves = board.generate_moves()
                if not moves:
                    break
                
                move = random.choice(moves)
                board.make_move(move)
            
            # Progress update every second
            last_print = update_progress_callback(
                progress_callback,
                len(features_list),
                num_positions,
                start_time,
                last_print
            )
        
        return np.array(features_list, dtype=np.float32), np.array(targets_list, dtype=np.float32)
    
    def generate_selfplay_data(
        self,
        nnue: 'NNUETorch',
        num_games: int = 100,
        max_moves: int = 200,
        search_depth: int = 2,
        progress_callback=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate self-play data using current model."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GPU training")
        
        features_list = []
        targets_list = []
        
        # Create engine with current NNUE
        engine = Engine(depth=search_depth, use_nnue=False)  # We'll use NNUE directly
        
        for game_idx in range(num_games):
            board = Board()
            game_features = []
            game_evals = []
            
            # Random opening
            play_random_opening(board, RANDOM_OPENING_MIN, RANDOM_OPENING_MAX)
            
            # Play game
            for move_count in range(max_moves):
                if board.is_checkmate() or board.is_stalemate():
                    break
                
                # Store position
                features = self.feature_extractor.extract(board)
                eval_score = nnue.evaluate(board)
                
                game_features.append(features)
                game_evals.append(eval_score)
                
                # Get move (mix of engine and random for diversity)
                moves = board.generate_moves()
                if not moves:
                    break
                
                if random.random() < RANDOM_MOVE_PROBABILITY:
                    move = random.choice(moves)
                else:
                    move = engine.search(board)
                    if move is None:
                        move = random.choice(moves)
                
                board.make_move(move)
            
            # Determine game result
            result = calculate_game_result(board)
            
            # Create targets based on game result and position evaluations
            for i, (feat, eval_score) in enumerate(zip(game_features, game_evals)):
                progress = i / max(len(game_features) - 1, 1)
                target = calculate_target_from_result(eval_score, result, progress)
                
                features_list.append(feat)
                targets_list.append(target)
            
            if progress_callback and (game_idx + 1) % PROGRESS_UPDATE_INTERVAL_GAMES == 0:
                progress_callback(game_idx + 1, num_games, len(features_list))
        
        return np.array(features_list, dtype=np.float32), np.array(targets_list, dtype=np.float32)
    
    def generate_selfplay_data_parallel(
        self,
        nnue: 'NNUETorch',
        num_games: int = 100,
        max_moves: int = 200,
        search_depth: int = 2,
        num_workers: int = None,
        batch_size: int = None,
        progress_callback=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate self-play data using CPU multiprocessing + GPU batch evaluation.
        
        개선 사항:
        1. CPU 멀티코어: 여러 게임을 병렬로 생성 (기보 학습 방식)
        2. GPU 배치 처리: 포지션들을 모아서 배치로 평가 (GPU 효율 향상)
        3. 동적 배치 크기: GPU 메모리에 따라 자동 조정
        
        Args:
            nnue: NNUETorch model (must be on GPU)
            num_games: Number of games to generate
            max_moves: Maximum moves per game
            search_depth: Search depth for engine
            num_workers: Number of CPU workers (None = auto)
            batch_size: Batch size for GPU evaluation (None = auto-detect based on GPU memory)
            progress_callback: Callback function(game_idx, total_games, positions)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GPU training")
        
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        
        # Calculate optimal batch size if not provided
        if batch_size is None:
            device = get_device() if TORCH_AVAILABLE else None
            batch_size = get_optimal_batch_size(device=device)
            if device and device.type == 'cuda':
                print(f"  GPU 메모리 기반 최적 배치 크기: {batch_size}")
        
        feature_size = self.feature_extractor.feature_size if self.feature_extractor else 512
        
        print(f"🚀 병렬 자기대국 생성: {num_games}개 게임, {num_workers}개 워커, GPU 배치 크기: {batch_size}")
        start_time = time.time()
        
        # Save model to temporary file for workers to load
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_model_path = tmp_file.name
        
        try:
            # Save current model state for workers
            nnue.save(tmp_model_path)
            
            # Prepare arguments for workers with model path
            args_list = [
                (i, feature_size, max_moves, search_depth, random.randint(0, 2**31), tmp_model_path)
                for i in range(num_games)
            ]
        except Exception as e:
            # If model saving fails, use None (will fallback to simple evaluator)
            print(f"Warning: Failed to save model for workers: {e}")
            args_list = [
                (i, feature_size, max_moves, search_depth, random.randint(0, 2**31), None)
                for i in range(num_games)
            ]
            tmp_model_path = None
        
        try:
            
            # Collect all games with game boundaries
            all_boards = []
            all_features = []
            game_boundaries = []  # (start_idx, end_idx, result) for each game
            
            # Use multiprocessing pool for game generation
            with mp.Pool(num_workers) as pool:
                chunksize = max(1, num_games // (num_workers * 4))
                results = pool.imap_unordered(_generate_selfplay_game_worker, args_list, chunksize=chunksize)
                
                processed_games = 0
                current_idx = 0
                for game_boards, game_features, result, num_pos in results:
                    if num_pos > 0:
                        start_idx = current_idx
                        end_idx = current_idx + num_pos
                        game_boundaries.append((start_idx, end_idx, result))
                        
                        all_boards.extend(game_boards)
                        all_features.extend(game_features)
                        current_idx = end_idx
                    
                    processed_games += 1
                    if progress_callback and processed_games % PROGRESS_UPDATE_INTERVAL_GAMES == 0:
                        progress_callback(processed_games, num_games, len(all_boards))
            
            print(f"✅ 게임 생성 완료: {len(all_boards)}개 포지션, {len(game_boundaries)}개 게임")
            
            # GPU 배치 평가 (이미 추출된 features 사용 - 중복 추출 방지)
            print(f"🎮 GPU 배치 평가 중...")
            eval_start = time.time()
            all_evals = []
            
            # Convert features list to numpy array for batch processing
            features_array = np.array(all_features, dtype=np.float32)
            
            # Evaluate in batches using pre-extracted features
            for i in range(0, len(all_features), batch_size):
                batch_features = features_array[i:i + batch_size]
                batch_evals = nnue.evaluate_batch(features=batch_features)
                all_evals.extend(batch_evals)
                
                if (i // batch_size + 1) % PROGRESS_UPDATE_INTERVAL_BATCH == 0:
                    elapsed = time.time() - eval_start
                    speed = len(all_evals) / elapsed if elapsed > 0 else 0
                    print(f"  평가 진행: {len(all_evals)}/{len(all_boards)} ({speed:.0f} pos/s)")
            
            eval_elapsed = time.time() - eval_start
            print(f"✅ GPU 평가 완료: {len(all_evals)}개 포지션, {eval_elapsed:.1f}초 ({len(all_evals)/eval_elapsed:.0f} pos/s)")
            
            # Create targets based on game result and position evaluations
            features_list = []
            targets_list = []
            
            # Process each game separately to calculate proper progress
            for start_idx, end_idx, result in game_boundaries:
                game_features = all_features[start_idx:end_idx]
                game_evals = all_evals[start_idx:end_idx]
                num_positions = len(game_features)
                
                for i, (feat, eval_score) in enumerate(zip(game_features, game_evals)):
                    progress = i / max(num_positions - 1, 1)
                    target = calculate_target_from_result(eval_score, result, progress)
                    
                    features_list.append(feat)
                    targets_list.append(target)
            
            total_elapsed = time.time() - start_time
            print(f"⏱️  총 소요 시간: {total_elapsed:.1f}초")
            print(f"📊 생성된 포지션: {len(features_list)}개")
            
            return np.array(features_list, dtype=np.float32), np.array(targets_list, dtype=np.float32)
        
        except Exception as e:
            print(f"Error in generate_selfplay_data_parallel: {e}")
            raise
        finally:
            # Clean up temporary model file
            if tmp_model_path and os.path.exists(tmp_model_path):
                try:
                    os.unlink(tmp_model_path)
                except Exception:
                    pass


def _evaluate_single_game_worker(args):
    """
    Worker function for parallel game evaluation.
    
    NOTE: This version loads NNUE model in worker and uses it for move selection.
    This ensures evaluation accurately reflects model performance.
    
    Args:
        args: (game_idx, model_path, search_depth, random_seed, use_gpu)
    
    Returns:
        (game_idx, result) where:
        - game_idx: int
        - result: int (1 for win, 0 for draw, -1 for loss)
    """
    game_idx, model_path, search_depth, random_seed, use_gpu = args
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Load NNUE model in worker (CPU mode for multiprocessing compatibility)
    nnue_model = None
    if model_path and TORCH_AVAILABLE and os.path.exists(model_path):
        try:
            # Suppress print output during model loading in worker
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                # Load model on CPU for multiprocessing compatibility
                nnue_model = NNUETorch.from_file(model_path, device=torch.device('cpu'))
            finally:
                sys.stdout = old_stdout
            
            nnue_engine = Engine(depth=search_depth, use_nnue=True, use_opening_book=False)
            nnue_engine.nnue = nnue_model
        except Exception as e:
            # Fallback to simple evaluator if model loading fails
            # Only print warning if it's not a suppressed output issue
            if "StringIO" not in str(e):
                print(f"Warning: Failed to load model in worker {game_idx}: {e}")
            nnue_engine = Engine(depth=search_depth, use_nnue=False, use_opening_book=False)
    else:
        # Fallback to simple evaluator if no model path
        nnue_engine = Engine(depth=search_depth, use_nnue=False, use_opening_book=False)
    
    # Create simple engine for opponent
    simple_engine = Engine(depth=search_depth, use_nnue=False, use_opening_book=False)
    
    nnue_is_cho = (game_idx % 2 == 0)
    board = Board()
    
    # Track consecutive failed moves to detect infinite loops
    consecutive_failed_moves = 0
    max_consecutive_failures = 10
    
    # Play game with proper minimax search
    for move_count in range(MAX_MOVES_EVAL):
        try:
            if board.is_checkmate() or board.is_stalemate():
                break
        except Exception:
            break
        
        moves = board.generate_moves()
        if not moves:
            break
        
        is_cho_turn = (board.side_to_move == Side.CHO)
        is_nnue_turn = (is_cho_turn and nnue_is_cho) or (not is_cho_turn and not nnue_is_cho)
        
        # Use appropriate engine based on whose turn it is
        if is_nnue_turn:
            # Use NNUE engine
            move = nnue_engine.search(board)
            if move is None:
                move = random.choice(moves) if moves else None
        else:
            # Use simple engine
            move = simple_engine.search(board)
            if move is None:
                move = random.choice(moves) if moves else None
        
        # Check if move is valid and apply it
        if move:
            # Verify move is in legal moves list before applying
            move_is_legal = any(
                m.from_file == move.from_file and m.from_rank == move.from_rank and
                m.to_file == move.to_file and m.to_rank == move.to_rank
                for m in moves
            )
            
            if move_is_legal:
                success = board.make_move(move)
                if success:
                    consecutive_failed_moves = 0  # Reset counter on success
                else:
                    consecutive_failed_moves += 1
                    # If too many consecutive failures, break to avoid infinite loop
                    if consecutive_failed_moves >= max_consecutive_failures:
                        break
            else:
                # Move is not legal, try random move
                move = random.choice(moves) if moves else None
                if move:
                    board.make_move(move)
                    consecutive_failed_moves = 0
                else:
                    consecutive_failed_moves += 1
                    if consecutive_failed_moves >= max_consecutive_failures:
                        break
        else:
            # No move found, break to avoid infinite loop
            consecutive_failed_moves += 1
            if consecutive_failed_moves >= max_consecutive_failures:
                break
    
    # Determine winner
    try:
        if board.is_checkmate():
            winner_is_cho = (board.side_to_move == Side.HAN)
            if (winner_is_cho and nnue_is_cho) or (not winner_is_cho and not nnue_is_cho):
                result = 1  # Win
            else:
                result = -1  # Loss
        else:
            result = 0  # Draw
    except Exception:
        result = 0  # Draw
    
    return (game_idx, result)


def evaluate_model(nnue: 'NNUETorch', num_games: int = 20, search_depth: int = 3, num_workers: int = None, use_gpu: bool = True, eval_batch_size: int = None) -> float:
    """Evaluate model against SimpleEvaluator with proper minimax search.
    
    Optimized for speed:
    - Parallel game execution using multiprocessing
    - Centralized GPU batch evaluation (avoids GPU model loading in each worker)
    - Dynamic batch size based on GPU memory (improves GPU utilization)
    - Opening book disabled (not needed for evaluation)
    - Reduced search depth for faster evaluation
    - Progress reporting for long evaluations
    
    Args:
        nnue: NNUETorch model to evaluate (must be on GPU)
        num_games: Number of games to play
        search_depth: Search depth for evaluation
        num_workers: Number of parallel workers (None = auto)
        use_gpu: Whether to use GPU for batch evaluation (default: True)
        eval_batch_size: Batch size for GPU evaluation (None = auto-detect based on GPU memory)
    """
    if num_workers is None:
        # 작은 게임 수에 대해 워커 수 제한 (멀티프로세싱 오버헤드 감소)
        # GPU 배치 평가를 사용하므로 워커는 게임 시뮬레이션만 수행
        # 작은 게임 수에서는 워커 초기화 + 모델 로드 오버헤드가 병렬화 이점보다 큼
        if num_games <= 5:
            num_workers = 1
        elif num_games <= 10:
            num_workers = min(2, num_games)
        else:
            num_workers = max(1, min(mp.cpu_count() - 1, num_games))
    
    # Save model to temporary file for workers to load
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        tmp_model_path = tmp_file.name
    
    try:
        # Save current model state for workers
        nnue.save(tmp_model_path)
        
        device_str = "CPU (워커에서 모델 로드)"
        print(f"  평가 진행: 0/{num_games} 게임 (병렬: {num_workers}개 워커, {device_str})", end="", flush=True)
        
        # Prepare arguments for workers
        args_list = [
            (i, tmp_model_path, search_depth, random.randint(0, 2**31), use_gpu)
            for i in range(num_games)
        ]
        
        # Collect game results from workers
        game_results = {}  # game_idx -> result
        
        # Use multiprocessing pool for parallel game generation
        with mp.Pool(num_workers) as pool:
            results = pool.imap_unordered(_evaluate_single_game_worker, args_list, chunksize=max(1, num_games // (num_workers * 4)))
            
            completed = 0
            last_update_time = time.time()
            update_interval_seconds = 2.0  # Update at least every 2 seconds
            
            for game_idx, result in results:
                game_results[game_idx] = result
                completed += 1
                
                # Progress update: more frequent updates to show activity
                current_time = time.time()
                time_since_update = current_time - last_update_time
                should_update = (
                    completed % max(1, PROGRESS_UPDATE_INTERVAL_EVAL // 2) == 0 or  # More frequent updates
                    completed == num_games or  # Always update on completion
                    time_since_update >= update_interval_seconds  # Time-based update
                )
                
                if should_update:
                    print(f"\r  평가 진행: {completed}/{num_games} 게임 (병렬: {num_workers}개 워커, {device_str})", end="", flush=True)
                    last_update_time = current_time
        
        print()  # New line after progress
        
        # Calculate win rate from results
        wins = sum(1 for r in game_results.values() if r == 1)
        losses = sum(1 for r in game_results.values() if r == -1)
        draws = sum(1 for r in game_results.values() if r == 0)
        
        win_rate = wins / num_games
        return win_rate
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_model_path)
        except Exception:
            pass


def train_iterative(
    nnue: 'NNUETorch',
    num_iterations: int = 5,
    games_per_iteration: int = 100,
    epochs_per_iteration: int = 20,
    batch_size: int = 256,
    output_dir: str = "models",
    search_depth: int = 2,
    use_parallel: bool = True,
    num_workers: int = None,
    eval_batch_size: int = None,
    eval_num_workers: int = None,
    base_learning_rate: float = 0.001
):
    """
    Iterative self-improvement training.
    
    Args:
        use_parallel: Use CPU multiprocessing + GPU batch evaluation (recommended)
        num_workers: Number of CPU workers (None = auto)
        eval_batch_size: Batch size for GPU evaluation during data generation
        eval_num_workers: Number of workers for parallel evaluation (None = auto)
        base_learning_rate: Base learning rate (will decay over iterations)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generator = DataGenerator()
    trainer = GPUTrainer(nnue)
    
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
        # Generate self-play data
        print("\nGenerating self-play data...")
        if use_parallel:
            features, targets = generator.generate_selfplay_data_parallel(
                nnue,
                num_games=games_per_iteration,
                max_moves=MAX_MOVES_PER_GAME,
                search_depth=search_depth,
                num_workers=num_workers,
                batch_size=eval_batch_size,
                progress_callback=lambda g, t, p: print(f"Game {g}/{t}, Positions: {p}")
            )
        else:
            features, targets = generator.generate_selfplay_data(
                nnue,
                num_games=games_per_iteration,
                search_depth=search_depth,
                progress_callback=lambda g, t, p: print(f"Game {g}/{t}, Positions: {p}")
            )
        
        print(f"\nTraining on {len(features)} positions...")
        
        # Decrease learning rate over iterations (use provided base_learning_rate)
        lr = base_learning_rate * (0.7 ** iteration)
        
        _ = trainer.train(
            features, targets,
            epochs=epochs_per_iteration,
            batch_size=batch_size,
            learning_rate=lr,
            early_stopping_patience=5
        )
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"nnue_gpu_iter_{iteration + 1}.json")
        nnue.save(checkpoint_path)
        print(f"Saved: {checkpoint_path}")
        
        # Evaluate
        print("\nEvaluating...")
        win_rate = evaluate_model(nnue, num_games=20, search_depth=3, num_workers=eval_num_workers, eval_batch_size=eval_batch_size)
        print(f"Win rate vs SimpleEvaluator: {win_rate:.1%}")
    
    return nnue


def main():
    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required for GPU training.")
        print("Install with: pip install torch")
        return
    
    parser = argparse.ArgumentParser(description='GPU-accelerated NNUE Training')
    
    # Training method
    parser.add_argument('--method', type=str, 
                        choices=['deepsearch', 'iterative'],
                        default='deepsearch',
                        help='Training method')
    
    # Data generation
    parser.add_argument('--positions', type=int, default=10000,
                        help='Number of positions to generate')
    parser.add_argument('--depth', type=int, default=2,
                        help='Search depth for target values (lower = faster)')
    parser.add_argument('--fast', action='store_true', default=True,
                        help='Use fast data generation (default: True)')
    parser.add_argument('--no-fast', dest='fast', action='store_false',
                        help='Use slower but higher quality data generation')
    parser.add_argument('--parallel', action='store_true',
                        help='Use multiprocessing for data generation (faster)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of workers for parallel generation')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (larger = faster on GPU)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate (default: 0.0005, lower = more stable)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    
    # Iterative training
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of self-improvement iterations')
    parser.add_argument('--games-per-iter', type=int, default=100,
                        help='Games per iteration')
    parser.add_argument('--no-parallel-selfplay', dest='use_parallel_selfplay', 
                        action='store_false', default=True,
                        help='Disable parallel self-play generation (use sequential)')
    parser.add_argument('--eval-batch-size', type=int, default=None,
                        help='Batch size for GPU evaluation (None = auto-detect based on GPU memory)')
    parser.add_argument('--eval-workers', type=int, default=None,
                        help='Number of workers for parallel evaluation (None = auto)')
    
    # Model architecture
    parser.add_argument('--feature-size', type=int, default=512,
                        help='Feature vector size')
    parser.add_argument('--hidden1', type=int, default=256,
                        help='First hidden layer size')
    parser.add_argument('--hidden2', type=int, default=64,
                        help='Second hidden layer size')
    
    # I/O
    parser.add_argument('--output', type=str, default='models/nnue_gpu_model.json',
                        help='Output model file')
    parser.add_argument('--load', type=str, default=None,
                        help='Load existing model')
    parser.add_argument('--output-torch', type=str, default=None,
                        help='Also save in PyTorch format (.pt)')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip final evaluation (faster)')
    
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
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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
    
    if args.method == 'iterative':
        # Iterative self-improvement
        nnue = train_iterative(
            nnue,
            num_iterations=args.iterations,
            games_per_iteration=args.games_per_iter,
            epochs_per_iteration=args.epochs,
            batch_size=args.batch_size,
            use_parallel=args.use_parallel_selfplay,
            num_workers=args.workers,
            eval_batch_size=args.eval_batch_size,
            eval_num_workers=args.eval_workers,
            base_learning_rate=args.lr
        )
    else:
        # Deep search training
        generator = DataGenerator(search_depth=args.depth)
        
        def progress(done, total, speed, eta):
            print(f"\rPositions: {done}/{total} ({speed:.1f}/s, ETA: {eta:.0f}s)", end="", flush=True)
        
        if args.parallel:
            print(f"\nGenerating {args.positions} positions (parallel mode)...")
            features, targets = generator.generate_positions_parallel(
                num_positions=args.positions,
                num_workers=args.workers,
                progress_callback=progress
            )
        elif args.fast:
            print(f"\nGenerating {args.positions} positions (fast mode)...")
            features, targets = generator.generate_positions_fast(
                num_positions=args.positions,
                progress_callback=progress
            )
        else:
            print(f"\nGenerating {args.positions} positions (quality mode, depth={args.depth})...")
            features, targets = generator.generate_positions(
                num_positions=args.positions,
                search_depth=args.depth,
                progress_callback=progress
            )
        
        print()  # New line after progress
        
        print(f"\nTraining on {len(features)} positions...")
        trainer = GPUTrainer(nnue)
        
        history = trainer.train(
            features, targets,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay
        )
        
        print(f"\nFinal train loss: {history['train_loss'][-1]:.6f}")
        print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    
    # Save model
    print(f"\nSaving model to {args.output}...")
    nnue.save(args.output)
    
    if args.output_torch:
        print(f"Saving PyTorch model to {args.output_torch}...")
        nnue.save_torch(args.output_torch)
    
    # Final evaluation (optional, can be slow)
    if not args.skip_eval:
        print("\nFinal evaluation (20 games)...")
        win_rate = evaluate_model(nnue, num_games=20, search_depth=3, num_workers=args.eval_workers, eval_batch_size=args.eval_batch_size)
        print(f"Win rate vs SimpleEvaluator: {win_rate:.1%}")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()

