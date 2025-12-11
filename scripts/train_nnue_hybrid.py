#!/usr/bin/env python3
"""Hybrid Training: Combining Gibo and Self-play Training for Janggi NNUE.

This script combines supervised learning from game records (gibo) with
self-play training to create a stronger model.

Usage:
    # Hybrid training with default settings
    python train_nnue_hybrid.py --gibo-dir gibo/ --iterations 5
    
    # Custom configuration
    python train_nnue_hybrid.py --gibo-dir gibo/ --iterations 5 \
        --gibo-epochs 2 --selfplay-epochs 10 --fine-tune-epochs 1 \
        --selfplay-games 100
"""

import argparse
import glob
import json
import os
import re
from typing import Optional, Dict, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Install with: pip install torch")

if TORCH_AVAILABLE:
    from janggi.nnue_torch import NNUETorch, get_device
    from scripts.train_nnue_gibo import train_from_gibo
    from scripts.train_nnue_gpu import (
        DataGenerator, GPUTrainer, evaluate_model,
        MAX_MOVES_PER_GAME
    )


def get_metadata_path(model_path: str) -> str:
    """Get metadata file path for a model file.
    
    Args:
        model_path: Path to model file (e.g., "models/nnue_hybrid_iter_3.json")
        
    Returns:
        Path to metadata file (e.g., "models/nnue_hybrid_iter_3_meta.json")
    """
    base, ext = os.path.splitext(model_path)
    return f"{base}_meta.json"


def save_metadata(model_path: str, metadata: Dict[str, Any]):
    """Save training metadata to a separate file.
    
    Args:
        model_path: Path to model file
        metadata: Dictionary containing training metadata
    """
    metadata_path = get_metadata_path(model_path)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_metadata(model_path: str) -> Optional[Dict[str, Any]]:
    """Load training metadata from file.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Metadata dictionary if exists, None otherwise
    """
    metadata_path = get_metadata_path(model_path)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load metadata from {metadata_path}: {e}")
            return None
    return None


def find_last_iteration_from_history(output_dir: str, base_name: str = "nnue_hybrid_iter") -> int:
    """Find the last completed iteration by checking history files.
    
    Args:
        output_dir: Directory containing model files
        base_name: Base name for iteration files
        
    Returns:
        Last completed iteration number (0 if none found)
    """
    # Look for history files to determine last iteration
    history_pattern = os.path.join(output_dir, f"{base_name}_*_step*_history.json")
    history_files = glob.glob(history_pattern)
    
    if not history_files:
        return 0
    
    max_iteration = 0
    for history_file in history_files:
        # Extract iteration number from filename
        # Format: nnue_hybrid_iter_{N}_step{M}_history.json
        match = re.search(rf"{base_name}_(\d+)_step", history_file)
        if match:
            iteration = int(match.group(1))
            max_iteration = max(max_iteration, iteration)
    
    return max_iteration


def train_from_selfplay(
    nnue: 'NNUETorch',
    num_games: int = 100,
    epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    search_depth: int = 2,
    use_parallel: bool = True,
    num_workers: Optional[int] = None,
    eval_batch_size: Optional[int] = None
) -> dict:
    """Train NNUE from self-play data.
    
    Args:
        nnue: NNUE model to train
        num_games: Number of self-play games to generate
        epochs: Training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        search_depth: Search depth for self-play
        use_parallel: Use parallel processing
        num_workers: Number of workers for parallel processing
        eval_batch_size: Batch size for GPU evaluation
        
    Returns:
        Training history
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required")
    
    generator = DataGenerator()
    trainer = GPUTrainer(nnue)
    
    # Generate self-play data
    print(f"\nGenerating {num_games} self-play games...")
    if use_parallel:
        features, targets = generator.generate_selfplay_data_parallel(
            nnue,
            num_games=num_games,
            max_moves=MAX_MOVES_PER_GAME,
            search_depth=search_depth,
            num_workers=num_workers,
            batch_size=eval_batch_size,
            progress_callback=lambda g, t, p: print(f"Game {g}/{t}, Positions: {p}")
        )
    else:
        features, targets = generator.generate_selfplay_data(
            nnue,
            num_games=num_games,
            max_moves=MAX_MOVES_PER_GAME,
            search_depth=search_depth,
            progress_callback=lambda g, t, p: print(f"Game {g}/{t}, Positions: {p}")
        )
    
    print(f"\nTraining on {len(features)} positions...")
    
    # Train on self-play data
    history = trainer.train(
        features, targets,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=5
    )
    
    return history


def hybrid_training(
    gibo_dir: str,
    nnue: 'NNUETorch',
    iterations: int = 5,
    gibo_epochs: int = 2,
    selfplay_epochs: int = 10,
    fine_tune_epochs: int = 1,
    selfplay_games: int = 100,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    positions_per_game: int = 50,
    search_depth: int = 2,
    output_dir: str = "models",
    use_parallel: bool = True,
    num_workers: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    eval_num_workers: Optional[int] = None,
    start_iteration: int = 0,
    base_learning_rate: Optional[float] = None
) -> 'NNUETorch':
    """Hybrid training combining gibo and self-play.
    
    Each iteration:
    1. Train from gibo data (gibo_epochs)
    2. Train from self-play data (selfplay_epochs)
    3. Fine-tune with gibo data (fine_tune_epochs)
    4. Evaluate
    
    Args:
        gibo_dir: Directory containing .gib files
        nnue: NNUE model to train
        iterations: Number of hybrid training iterations
        gibo_epochs: Epochs for initial gibo training
        selfplay_epochs: Epochs for self-play training
        fine_tune_epochs: Epochs for fine-tuning with gibo
        selfplay_games: Number of self-play games per iteration
        batch_size: Batch size for training
        learning_rate: Base learning rate (or current learning rate if resuming)
        positions_per_game: Max positions per gibo game
        search_depth: Search depth for self-play
        output_dir: Directory to save models
        use_parallel: Use parallel processing for self-play
        num_workers: Number of workers for parallel processing
        eval_batch_size: Batch size for GPU evaluation
        eval_num_workers: Number of workers for evaluation
        start_iteration: Starting iteration number (for resuming training)
        base_learning_rate: Original base learning rate (for proper decay calculation)
        
    Returns:
        Trained NNUE model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Learning rate decay over iterations
    # If resuming, use the original base_lr for proper decay calculation
    if base_learning_rate is not None:
        base_lr = base_learning_rate
    else:
        base_lr = learning_rate
    
    # Adjust iteration range if resuming from a previous training
    actual_iterations = iterations - start_iteration
    if actual_iterations <= 0:
        print(f"⚠️ Warning: Already completed {start_iteration} iterations. Nothing to train.")
        return nnue
    
    if start_iteration > 0:
        print(f"\n📌 Resuming training from iteration {start_iteration + 1}")
        print(f"   Will train {actual_iterations} more iterations (total: {iterations})")
    
    for i in range(actual_iterations):
        iteration = start_iteration + i
        print(f"\n{'='*70}")
        print(f"HYBRID TRAINING ITERATION {iteration + 1}/{iterations}")
        print(f"{'='*70}")
        
        # Decay learning rate based on actual iteration number
        current_lr = base_lr * (0.8 ** iteration)
        print(f"Learning rate: {current_lr:.6f}")
        
        # Step 1: Train from gibo data
        print(f"\n{'─'*70}")
        print("Step 1: Training from Gibo data")
        print(f"{'─'*70}")
        try:
            gibo_history = train_from_gibo(
                gibo_dir=gibo_dir,
                nnue=nnue,
                epochs=gibo_epochs,
                batch_size=batch_size,
                learning_rate=current_lr,
                positions_per_game=positions_per_game,
                output_file=os.path.join(output_dir, f"nnue_hybrid_iter_{iteration + 1}_step1.json")
            )
            print(f"Gibo training completed. Final loss: {gibo_history.get('train_loss', [0])[-1]:.6f}")
        except Exception as e:
            print(f"Warning: Gibo training failed: {e}")
            print("Continuing with self-play training...")
        
        # Step 2: Train from self-play data
        print(f"\n{'─'*70}")
        print("Step 2: Training from Self-play data")
        print(f"{'─'*70}")
        try:
            selfplay_history = train_from_selfplay(
                nnue=nnue,
                num_games=selfplay_games,
                epochs=selfplay_epochs,
                batch_size=batch_size,
                learning_rate=current_lr,
                search_depth=search_depth,
                use_parallel=use_parallel,
                num_workers=num_workers,
                eval_batch_size=eval_batch_size
            )
            print(f"Self-play training completed. Final loss: {selfplay_history.get('train_loss', [0])[-1]:.6f}")
        except Exception as e:
            print(f"Warning: Self-play training failed: {e}")
            print("Continuing with fine-tuning...")
        
        # Step 3: Fine-tune with gibo data
        print(f"\n{'─'*70}")
        print("Step 3: Fine-tuning with Gibo data")
        print(f"{'─'*70}")
        try:
            finetune_history = train_from_gibo(
                gibo_dir=gibo_dir,
                nnue=nnue,
                epochs=fine_tune_epochs,
                batch_size=batch_size,
                learning_rate=current_lr * 0.5,  # Lower LR for fine-tuning
                positions_per_game=positions_per_game,
                output_file=os.path.join(output_dir, f"nnue_hybrid_iter_{iteration + 1}_step3.json")
            )
            print(f"Fine-tuning completed. Final loss: {finetune_history.get('train_loss', [0])[-1]:.6f}")
        except Exception as e:
            print(f"Warning: Fine-tuning failed: {e}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"nnue_hybrid_iter_{iteration + 1}.json")
        nnue.save(checkpoint_path)
        
        # Save metadata
        metadata = {
            "last_iteration": iteration + 1,
            "total_iterations": iterations,
            "base_learning_rate": base_lr,
            "last_learning_rate": current_lr,
            "last_fine_tune_learning_rate": current_lr * 0.5,
            "gibo_epochs": gibo_epochs,
            "selfplay_epochs": selfplay_epochs,
            "fine_tune_epochs": fine_tune_epochs,
            "batch_size": batch_size,
            "search_depth": search_depth
        }
        save_metadata(checkpoint_path, metadata)
        print(f"\nSaved checkpoint: {checkpoint_path}")
        print(f"Saved metadata: {get_metadata_path(checkpoint_path)}")
        
        # Step 4: Evaluate
        print(f"\n{'─'*70}")
        print("Step 4: Evaluation")
        print(f"{'─'*70}")
        try:
            win_rate = evaluate_model(
                nnue,
                num_games=20,
                search_depth=3,
                num_workers=eval_num_workers,
                eval_batch_size=eval_batch_size
            )
            print(f"Win rate vs SimpleEvaluator: {win_rate:.1%}")
        except Exception as e:
            print(f"Warning: Evaluation failed: {e}")
    
    return nnue


def main():
    if not TORCH_AVAILABLE:
        print("Error: PyTorch is required for hybrid training")
        return
    
    parser = argparse.ArgumentParser(
        description="Hybrid training combining gibo and self-play"
    )
    
    parser.add_argument(
        '--gibo-dir',
        type=str,
        default='gibo',
        help='Directory containing .gib files'
    )
    
    parser.add_argument(
        '--load',
        type=str,
        default=None,
        help='Load existing model file'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='Number of hybrid training iterations'
    )
    
    parser.add_argument(
        '--gibo-epochs',
        type=int,
        default=2,
        help='Epochs for initial gibo training per iteration'
    )
    
    parser.add_argument(
        '--selfplay-epochs',
        type=int,
        default=10,
        help='Epochs for self-play training per iteration'
    )
    
    parser.add_argument(
        '--fine-tune-epochs',
        type=int,
        default=1,
        help='Epochs for fine-tuning with gibo per iteration'
    )
    
    parser.add_argument(
        '--selfplay-games',
        type=int,
        default=100,
        help='Number of self-play games per iteration'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Base learning rate'
    )
    
    parser.add_argument(
        '--positions-per-game',
        type=int,
        default=50,
        help='Max positions per gibo game'
    )
    
    parser.add_argument(
        '--search-depth',
        type=int,
        default=2,
        help='Search depth for self-play'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save models'
    )
    
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing for self-play'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of workers for parallel processing'
    )
    
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=None,
        help='Batch size for GPU evaluation'
    )
    
    parser.add_argument(
        '--eval-num-workers',
        type=int,
        default=None,
        help='Number of workers for evaluation'
    )
    
    args = parser.parse_args()
    
    # Initialize or load model
    device = get_device()
    print(f"Using device: {device}")
    
    if args.load and os.path.exists(args.load):
        print(f"Loading model from {args.load}...")
        nnue = NNUETorch.from_file(args.load, device=device)
    else:
        print("Initializing new model...")
        nnue = NNUETorch(device=device)
    
    # Run hybrid training
    nnue = hybrid_training(
        gibo_dir=args.gibo_dir,
        nnue=nnue,
        iterations=args.iterations,
        gibo_epochs=args.gibo_epochs,
        selfplay_epochs=args.selfplay_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        selfplay_games=args.selfplay_games,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        positions_per_game=args.positions_per_game,
        search_depth=args.search_depth,
        output_dir=args.output_dir,
        use_parallel=not args.no_parallel,
        num_workers=args.num_workers,
        eval_batch_size=args.eval_batch_size,
        eval_num_workers=args.eval_num_workers
    )
    
    # Save final model
    final_path = os.path.join(args.output_dir, "nnue_hybrid_final.json")
    nnue.save(final_path)
    print(f"\n{'='*70}")
    print(f"Hybrid training completed!")
    print(f"Final model saved to: {final_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

