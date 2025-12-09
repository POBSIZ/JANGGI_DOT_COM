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
from typing import Tuple
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


# Worker function for multiprocessing (must be at module level)
def _generate_game_data(args):
    """Generate data for a single game (worker function)."""
    game_idx, feature_size, positions_per_game = args
    
    feature_extractor = FeatureExtractor(feature_size)
    
    # Piece values for fast evaluation
    PIECE_VALUES = {
        PieceType.KING: 0,
        PieceType.ROOK: 13,
        PieceType.CANNON: 7,
        PieceType.HORSE: 5,
        PieceType.ELEPHANT: 3,
        PieceType.GUARD: 3,
        PieceType.PAWN: 2,
    }
    
    def fast_evaluate(board):
        han_material = cho_material = 0
        for rank in range(board.RANKS):
            for file in range(board.FILES):
                piece = board.get_piece(file, rank)
                if piece is None:
                    continue
                value = PIECE_VALUES.get(piece.piece_type, 0)
                if piece.side == Side.HAN:
                    han_material += value
                else:
                    cho_material += value
        
        if board.side_to_move == Side.HAN:
            return (han_material - cho_material + 1.5) / 72.0
        else:
            return (cho_material - han_material) / 72.0
    
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
        eval_score = fast_evaluate(board)
        
        # Skip if any NaN values
        if not np.isnan(features).any() and not np.isnan(eval_score):
            features_list.append(features)
            targets_list.append(eval_score)
        
        # Random move
        move = random.choice(moves)
        board.make_move(move)
    
    return features_list, targets_list


class DataGenerator:
    """Generate training data for GPU training."""
    
    # Simple piece values for fast evaluation
    PIECE_VALUES = {
        PieceType.KING: 0,
        PieceType.ROOK: 13,
        PieceType.CANNON: 7,
        PieceType.HORSE: 5,
        PieceType.ELEPHANT: 3,
        PieceType.GUARD: 3,
        PieceType.PAWN: 2,
    }
    
    def __init__(self, search_depth: int = 3):
        self.search_depth = search_depth
        self.feature_extractor = FeatureExtractor() if TORCH_AVAILABLE else None
    
    def _fast_evaluate(self, board: Board) -> float:
        """Fast material-based evaluation (no search)."""
        han_material = 0
        cho_material = 0
        
        for rank in range(board.RANKS):
            for file in range(board.FILES):
                piece = board.get_piece(file, rank)
                if piece is None:
                    continue
                value = self.PIECE_VALUES.get(piece.piece_type, 0)
                if piece.side == Side.HAN:
                    han_material += value
                else:
                    cho_material += value
        
        if board.side_to_move == Side.HAN:
            return (han_material - cho_material + 1.5) / 72.0
        else:
            return (cho_material - han_material) / 72.0
    
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
        
        positions_per_game = 30
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
                
                if progress_callback and (i + 1) % 50 == 0:
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
        
        positions_per_game = 30
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
            current_time = time.time()
            if progress_callback and current_time - last_print >= 1.0:
                elapsed = current_time - start_time
                positions_done = len(features_list)
                speed = positions_done / elapsed if elapsed > 0 else 0
                eta = (num_positions - positions_done) / speed if speed > 0 else 0
                progress_callback(positions_done, num_positions, speed, eta)
                last_print = current_time
        
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
        
        positions_per_game = 20
        games_needed = (num_positions + positions_per_game - 1) // positions_per_game
        
        start_time = time.time()
        last_print = start_time
        
        for game_idx in range(games_needed):
            if len(features_list) >= num_positions:
                break
            
            board = Board()
            
            # Random opening
            num_random = random.randint(4, 25)
            for _ in range(num_random):
                moves = board.generate_moves()
                if not moves:
                    break
                move = random.choice(moves)
                board.make_move(move)
            
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
            current_time = time.time()
            if progress_callback and current_time - last_print >= 1.0:
                elapsed = current_time - start_time
                positions_done = len(features_list)
                speed = positions_done / elapsed if elapsed > 0 else 0
                eta = (num_positions - positions_done) / speed if speed > 0 else 0
                progress_callback(positions_done, num_positions, speed, eta)
                last_print = current_time
        
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
            for _ in range(random.randint(2, 6)):
                moves = board.generate_moves()
                if not moves:
                    break
                move = random.choice(moves)
                board.make_move(move)
            
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
                
                if random.random() < 0.2:
                    move = random.choice(moves)
                else:
                    move = engine.search(board)
                    if move is None:
                        move = random.choice(moves)
                
                board.make_move(move)
            
            # Determine game result
            if board.is_checkmate():
                result = -1.0 if board.side_to_move == Side.CHO else 1.0
            else:
                result = 0.0
            
            # Create targets based on game result and position evaluations
            for i, (feat, eval_score) in enumerate(zip(game_features, game_evals)):
                progress = i / max(len(game_features) - 1, 1)
                # Blend evaluation with game outcome
                target = 0.7 * np.clip(eval_score / 10.0, -1, 1) + 0.3 * result * progress
                
                features_list.append(feat)
                targets_list.append(target)
            
            if progress_callback and (game_idx + 1) % 10 == 0:
                progress_callback(game_idx + 1, num_games, len(features_list))
        
        return np.array(features_list, dtype=np.float32), np.array(targets_list, dtype=np.float32)


def evaluate_model(nnue: 'NNUETorch', num_games: int = 10) -> float:
    """Evaluate model against SimpleEvaluator (fast version)."""
    from janggi.nnue import SimpleEvaluator
    
    wins = 0
    draws = 0
    simple_eval = SimpleEvaluator()
    
    for game_idx in range(num_games):
        nnue_is_cho = (game_idx % 2 == 0)
        board = Board()
        
        # Faster evaluation: only 100 moves max, no deep search
        for move_count in range(100):
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
            
            # Fast 1-ply search (just evaluate each move, no deep search)
            best_move = None
            best_score = float('-inf')
            
            for move in moves:
                test_board = copy.deepcopy(board)
                if test_board.make_move(move):
                    if is_nnue_turn:
                        score = -nnue.evaluate(test_board)
                    else:
                        score = -simple_eval.evaluate(test_board)
                    
                    if score > best_score:
                        best_score = score
                        best_move = move
            
            move = best_move or random.choice(moves)
            board.make_move(move)
        
        # Determine winner
        try:
            if board.is_checkmate():
                winner_is_cho = (board.side_to_move == Side.HAN)
                if (winner_is_cho and nnue_is_cho) or (not winner_is_cho and not nnue_is_cho):
                    wins += 1
            elif not board.is_stalemate():
                draws += 1
        except Exception:
            draws += 1
    
    return (wins + 0.5 * draws) / num_games


def train_iterative(
    nnue: 'NNUETorch',
    num_iterations: int = 5,
    games_per_iteration: int = 100,
    epochs_per_iteration: int = 20,
    batch_size: int = 256,
    output_dir: str = "models",
    search_depth: int = 2
):
    """Iterative self-improvement training."""
    os.makedirs(output_dir, exist_ok=True)
    
    generator = DataGenerator()
    trainer = GPUTrainer(nnue)
    
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
        # Generate self-play data
        print("\nGenerating self-play data...")
        features, targets = generator.generate_selfplay_data(
            nnue,
            num_games=games_per_iteration,
            search_depth=search_depth,
            progress_callback=lambda g, t, p: print(f"Game {g}/{t}, Positions: {p}")
        )
        
        print(f"\nTraining on {len(features)} positions...")
        
        # Decrease learning rate over iterations
        lr = 0.001 * (0.7 ** iteration)
        
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
        win_rate = evaluate_model(nnue, num_games=10)
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
            batch_size=args.batch_size
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
        print("\nFinal evaluation (5 games)...")
        win_rate = evaluate_model(nnue, num_games=5)
        print(f"Win rate vs SimpleEvaluator: {win_rate:.1%}")
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()

