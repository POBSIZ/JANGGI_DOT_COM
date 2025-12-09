#!/usr/bin/env python3
"""Advanced NNUE Training Script for Janggi.

This script trains the NNUE evaluation function using various methods:
- Self-play game generation
- Deep search target values
- Iterative reinforcement (self-improvement)

Usage:
    # Basic training
    python train_nnue.py --games 100 --epochs 20 --output nnue_model.json

    # Deep search training (recommended)
    python train_nnue.py --method deepsearch --positions 5000 --depth 4 --epochs 30

    # Iterative training (best for strong models)
    python train_nnue.py --method iterative --iterations 5 --games-per-iter 50
"""

import argparse
import copy
import random
import sys
import os
import time
from typing import List, Tuple, Optional, Callable
import numpy as np

from janggi.board import Board, Move, Side
from janggi.nnue import NNUE, SimpleEvaluator
from janggi.engine import Engine

# ============================================================================
# Training Configuration Constants
# ============================================================================

# Game Generation Settings
DEFAULT_MAX_MOVES = 200
DEFAULT_RANDOM_OPENING_MOVES = 4
DEFAULT_TEMPERATURE = 0.3
DEFAULT_RANDOM_OPENING_MIN = 4
DEFAULT_RANDOM_OPENING_MAX = 20
DEFAULT_RANDOM_MOVE_PROBABILITY = 0.3

# Evaluation Settings
EVAL_NORMALIZATION_FACTOR = 100.0
FINAL_EVAL_SCALE = 10.0
HUBER_LOSS_DELTA = 1.0

# Training Settings
DEFAULT_VALIDATION_SPLIT = 0.1
DEFAULT_LR_DECAY = 0.95
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_ITERATIVE_LR_DECAY = 0.8
DEFAULT_ITERATIVE_BASE_LR = 0.001

# Progress Reporting
PROGRESS_UPDATE_FREQUENCY = 10


class TrainingDataGenerator:
    """Generate training data from self-play games."""

    def __init__(
        self,
        search_depth: int = 2,
        use_nnue: bool = False,
        nnue_model: Optional[NNUE] = None,
    ):
        """Initialize generator."""
        self.search_depth = search_depth
        self.use_nnue = use_nnue
        self.nnue_model = nnue_model
        self.simple_evaluator = SimpleEvaluator()

    def generate_game(
        self,
        max_moves: int = DEFAULT_MAX_MOVES,
        random_opening_moves: int = DEFAULT_RANDOM_OPENING_MOVES,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Tuple[List[Board], float, List[float]]:
        """Generate a single self-play game with position evaluations.

        Returns:
            Tuple of (list of board positions, game result, list of search evaluations)
        """
        board = Board()
        positions = []
        evaluations = []

        if self.use_nnue and self.nnue_model:
            engine = Engine(depth=self.search_depth, use_nnue=True)
            engine.nnue = self.nnue_model
        else:
            engine = Engine(depth=self.search_depth, use_nnue=False)

        # Play random opening moves for diversity
        for _ in range(random_opening_moves):
            moves = board.generate_moves()
            if not moves:
                break

            # Softmax selection with temperature for diversity
            if temperature > 0 and len(moves) > 1:
                move = random.choice(moves)
            else:
                move = moves[0]
            board.make_move(move)

        # Continue with engine search
        for move_count in range(max_moves):
            # Save position
            positions.append(copy.deepcopy(board))

            # Get evaluation from deeper search
            eval_score = engine._evaluate(board)
            evaluations.append(eval_score)

            # Check for game end
            if board.is_checkmate():
                if board.side_to_move == Side.CHO:
                    return positions, -1.0, evaluations  # HAN wins
                else:
                    return positions, 1.0, evaluations  # CHO wins

            if board.is_stalemate():
                return positions, 0.0, evaluations

            # Get best move (with some randomness for diversity)
            moves = board.generate_moves()
            if not moves:
                break

            if random.random() < temperature and len(moves) > 1:
                # Random move for exploration
                move = random.choice(moves)
            else:
                move = engine.search(board)
                if move is None:
                    break

            board.make_move(move)

        # Draw by move limit - evaluate final position
        final_eval = engine._evaluate(board)
        result = np.tanh(final_eval / FINAL_EVAL_SCALE)  # Soft outcome based on evaluation
        return positions, result, evaluations

    def generate_training_data(
        self,
        num_games: int = 100,
        max_moves_per_game: int = 200,
        use_search_targets: bool = True,
        progress_callback=None,
    ) -> Tuple[List[Board], List[float]]:
        """Generate training data from multiple self-play games."""
        all_boards = []
        all_targets = []

        for game_idx in range(num_games):
            positions, result, evaluations = self.generate_game(max_moves_per_game)

            for i, board in enumerate(positions):
                if use_search_targets and i < len(evaluations):
                    # Use search evaluation as target (more informative)
                    target = evaluations[i]
                    # Normalize large values
                    target = np.clip(target / EVAL_NORMALIZATION_FACTOR, -1.0, 1.0)
                else:
                    # Interpolate game result
                    progress = i / max(len(positions) - 1, 1)
                    target = result * progress

                # Flip sign if it's HAN's turn
                if board.side_to_move == Side.HAN:
                    target = -target

                all_boards.append(board)
                all_targets.append(target)

            if progress_callback:
                progress_callback(game_idx + 1, num_games, len(all_boards))

        return all_boards, all_targets

    def generate_diverse_positions(
        self, num_positions: int = 1000, search_depth: int = 4, progress_callback=None
    ) -> Tuple[List[Board], List[float]]:
        """Generate diverse positions with deep search evaluations."""
        boards = []
        targets = []

        engine = Engine(depth=search_depth, use_nnue=False)

        positions_per_game = max(10, num_positions // 100)
        games_needed = (num_positions + positions_per_game - 1) // positions_per_game

        for game_idx in range(games_needed):
            board = Board()

            # Random opening
            num_random = random.randint(DEFAULT_RANDOM_OPENING_MIN, DEFAULT_RANDOM_OPENING_MAX)
            for _ in range(num_random):
                moves = board.generate_moves()
                if not moves:
                    break
                move = random.choice(moves)
                board.make_move(move)

            # Collect positions from this game
            for _ in range(positions_per_game):
                if len(boards) >= num_positions:
                    break

                # Skip terminal positions
                if board.is_checkmate() or board.is_stalemate():
                    break

                # Get deep search evaluation
                target = engine._evaluate(board)
                target = np.clip(target / EVAL_NORMALIZATION_FACTOR, -1.0, 1.0)

                boards.append(copy.deepcopy(board))
                targets.append(target)

                # Make a move to get to next position
                moves = board.generate_moves()
                if not moves:
                    break

                # Mix of best and random moves
                if random.random() < DEFAULT_RANDOM_MOVE_PROBABILITY:
                    move = random.choice(moves)
                else:
                    move = engine.search(board)
                    if move is None:
                        move = random.choice(moves)

                board.make_move(move)

            if progress_callback and (game_idx + 1) % PROGRESS_UPDATE_FREQUENCY == 0:
                progress_callback(len(boards), num_positions)

        return boards, targets


class NNUETrainer:
    """Advanced trainer for NNUE network."""

    def __init__(self, nnue: NNUE):
        self.nnue = nnue
        self.best_loss = float("inf")
        self.patience_counter = 0

    def train(
        self,
        boards: List[Board],
        targets: List[float],
        epochs: int = 20,
        learning_rate: float = DEFAULT_ITERATIVE_BASE_LR,
        batch_size: int = 64,
        validation_split: float = DEFAULT_VALIDATION_SPLIT,
        lr_decay: float = DEFAULT_LR_DECAY,
        early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """Train the NNUE network with advanced features."""
        # Extract features once
        print("Extracting features...")
        features = np.array([self.nnue._extract_features(b) for b in boards])
        targets = np.array(targets)

        # Split into train/val
        n_samples = len(features)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)

        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        train_features = features[train_indices]
        train_targets = targets[train_indices]
        val_features = features[val_indices]
        val_targets = targets[val_indices]

        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        print(
            f"Training on {len(train_features)} samples, validating on {len(val_features)} samples"
        )

        current_lr = learning_rate
        self.nnue.set_learning_rate(current_lr)
        self.best_loss = float("inf")
        self.patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()

            # Shuffle training data
            shuffle_idx = np.random.permutation(len(train_features))
            train_features = train_features[shuffle_idx]
            train_targets = train_targets[shuffle_idx]

            # Training
            epoch_loss = 0.0
            n_batches = (len(train_features) + batch_size - 1) // batch_size

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(train_features))

                batch_loss = 0.0
                for i in range(start, end):
                    self.nnue._forward(train_features[i], training=True)
                    loss = self.nnue._backward(train_targets[i])
                    batch_loss += loss

                epoch_loss += batch_loss

            avg_train_loss = epoch_loss / len(train_features)

            # Validation
            val_loss = self._calculate_validation_loss(val_features, val_targets)

            avg_val_loss = (
                val_loss / len(val_features) if len(val_features) > 0 else 0.0
            )

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["learning_rate"].append(current_lr)

            elapsed = time.time() - start_time

            if progress_callback:
                progress_callback(
                    epoch + 1, epochs, avg_train_loss, avg_val_loss, current_lr
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
                    f"LR: {current_lr:.6f}, Time: {elapsed:.1f}s"
                )

            # Early stopping check
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            # Learning rate decay
            current_lr *= lr_decay
            self.nnue.set_learning_rate(current_lr)

        return history
    
    def _calculate_validation_loss(
        self, val_features: np.ndarray, val_targets: np.ndarray
    ) -> float:
        """Calculate validation loss using Huber loss.
        
        Args:
            val_features: Validation feature vectors
            val_targets: Validation target values
            
        Returns:
            Average validation loss
        """
        val_loss = 0.0
        for i in range(len(val_features)):
            pred = self.nnue._forward(val_features[i])
            error = pred - val_targets[i]
            # Huber loss
            if abs(error) <= HUBER_LOSS_DELTA:
                val_loss += 0.5 * error**2
            else:
                val_loss += HUBER_LOSS_DELTA * (abs(error) - 0.5 * HUBER_LOSS_DELTA)
        
        return val_loss / len(val_features) if len(val_features) > 0 else 0.0


class IterativeTrainer:
    """Iterative self-improvement trainer."""

    def __init__(self, base_nnue: Optional[NNUE] = None):
        if base_nnue:
            self.nnue = base_nnue
        else:
            self.nnue = NNUE()

    def train_iteration(
        self,
        games_per_iteration: int = 50,
        positions_per_game: int = 100,
        search_depth: int = 3,
        epochs: int = 10,
        learning_rate: float = 0.001,
    ) -> dict:
        """Single iteration of self-improvement."""
        print("Generating self-play games...")

        # Use current model for self-play
        generator = TrainingDataGenerator(
            search_depth=search_depth, use_nnue=True, nnue_model=self.nnue
        )

        boards, targets = generator.generate_training_data(
            num_games=games_per_iteration,
            use_search_targets=True,
            progress_callback=lambda c, t, p: print(f"Game {c}/{t}, Positions: {p}"),
        )

        print(f"\nTraining on {len(boards)} positions...")
        trainer = NNUETrainer(self.nnue)
        history = trainer.train(
            boards, targets, epochs=epochs, learning_rate=learning_rate
        )

        return history

    def run_iterations(
        self,
        num_iterations: int = 5,
        games_per_iteration: int = 50,
        search_depth: int = 3,
        epochs_per_iteration: int = 10,
        output_dir: str = "models",
        base_name: str = "nnue_iter",
    ) -> List[dict]:
        """Run multiple iterations of self-improvement."""
        os.makedirs(output_dir, exist_ok=True)

        all_histories = []

        for iteration in range(num_iterations):
            print(f"\n{'=' * 60}")
            print(f"ITERATION {iteration + 1}/{num_iterations}")
            print(f"{'=' * 60}")

            # Decrease learning rate over iterations
            lr = DEFAULT_ITERATIVE_BASE_LR * (DEFAULT_ITERATIVE_LR_DECAY ** iteration)

            history = self.train_iteration(
                games_per_iteration=games_per_iteration,
                search_depth=search_depth,
                epochs=epochs_per_iteration,
                learning_rate=lr,
            )

            all_histories.append(history)

            # Save checkpoint
            checkpoint_path = os.path.join(
                output_dir, f"{base_name}_{iteration + 1}.json"
            )
            self.nnue.save(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # Evaluate against simple evaluator
            win_rate = self._evaluate_model(num_games=10)
            print(f"Win rate vs SimpleEvaluator: {win_rate:.1%}")

        return all_histories

    def _evaluate_model(self, num_games: int = 10) -> float:
        """Evaluate current model against SimpleEvaluator."""
        wins = 0

        for game_idx in range(num_games):
            # Alternate colors
            nnue_is_cho = game_idx % 2 == 0

            board = Board()
            nnue_engine = Engine(depth=2, use_nnue=True)
            nnue_engine.nnue = self.nnue
            simple_engine = Engine(depth=2, use_nnue=False)

            for move_count in range(200):
                if board.is_checkmate() or board.is_stalemate():
                    break

                is_cho_turn = board.side_to_move == Side.CHO

                if (is_cho_turn and nnue_is_cho) or (
                    not is_cho_turn and not nnue_is_cho
                ):
                    move = nnue_engine.search(board)
                else:
                    move = simple_engine.search(board)

                if move is None:
                    break
                board.make_move(move)

            # Determine winner
            if board.is_checkmate():
                winner_is_cho = board.side_to_move == Side.HAN  # Loser is side to move
                if (winner_is_cho and nnue_is_cho) or (
                    not winner_is_cho and not nnue_is_cho
                ):
                    wins += 1

        return wins / num_games


def main():
    parser = argparse.ArgumentParser(description="Advanced NNUE Training for Janggi")

    # Training method
    parser.add_argument(
        "--method",
        type=str,
        choices=["selfplay", "deepsearch", "iterative"],
        default="deepsearch",
        help="Training method",
    )

    # Data generation
    parser.add_argument(
        "--games", type=int, default=100, help="Number of self-play games"
    )
    parser.add_argument(
        "--positions", type=int, default=5000, help="Number of positions for deepsearch"
    )
    parser.add_argument(
        "--depth", type=int, default=4, help="Search depth for target values"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--lr-decay", type=float, default=0.95, help="Learning rate decay per epoch"
    )

    # Iterative training
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of self-improvement iterations",
    )
    parser.add_argument(
        "--games-per-iter",
        type=int,
        default=50,
        help="Games per iteration for iterative training",
    )

    # Model I/O
    parser.add_argument(
        "--output", type=str, default="models/nnue_model.json", help="Output model file"
    )
    parser.add_argument("--load", type=str, default=None, help="Load existing model")

    # Network architecture
    parser.add_argument(
        "--feature-size", type=int, default=512, help="Feature vector size"
    )
    parser.add_argument(
        "--hidden1", type=int, default=256, help="First hidden layer size"
    )
    parser.add_argument(
        "--hidden2", type=int, default=64, help="Second hidden layer size"
    )

    args = parser.parse_args()

    # Initialize or load model
    if args.load:
        print(f"Loading model from {args.load}...")
        nnue = NNUE.from_file(args.load)
    else:
        print("Initializing new NNUE model...")
        nnue = NNUE(
            feature_size=args.feature_size,
            hidden1_size=args.hidden1,
            hidden2_size=args.hidden2,
        )

    print(
        f"Model architecture: {args.feature_size} -> {args.hidden1} -> {args.hidden2} -> 1"
    )

    if args.method == "iterative":
        # Iterative self-improvement
        print(f"\nStarting iterative training with {args.iterations} iterations...")
        trainer = IterativeTrainer(nnue)
        histories = trainer.run_iterations(
            num_iterations=args.iterations,
            games_per_iteration=args.games_per_iter,
            search_depth=args.depth,
            epochs_per_iteration=args.epochs,
            output_dir="models",
            base_name="nnue_iter",
        )

        # Save final model
        nnue.save(args.output)
        print(f"\nFinal model saved to {args.output}")

    elif args.method == "deepsearch":
        # Deep search training
        print(
            f"\nGenerating {args.positions} positions with depth-{args.depth} search..."
        )
        generator = TrainingDataGenerator(search_depth=args.depth)
        boards, targets = generator.generate_diverse_positions(
            num_positions=args.positions,
            search_depth=args.depth,
            progress_callback=lambda c, t: print(f"Positions: {c}/{t}"),
        )

        print(f"\nTraining on {len(boards)} positions...")
        trainer = NNUETrainer(nnue)
        history = trainer.train(
            boards,
            targets,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            lr_decay=args.lr_decay,
        )

        nnue.save(args.output)
        print(f"\nModel saved to {args.output}")
        print(f"Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"Final val loss: {history['val_loss'][-1]:.6f}")

    else:
        # Self-play training
        print(f"\nGenerating {args.games} self-play games...")
        generator = TrainingDataGenerator(search_depth=args.depth)

        def game_progress(current, total, positions):
            print(f"Game {current}/{total} - Positions: {positions}")

        boards, targets = generator.generate_training_data(
            num_games=args.games,
            use_search_targets=True,
            progress_callback=game_progress,
        )

        print(f"\nTraining on {len(boards)} positions...")
        trainer = NNUETrainer(nnue)
        history = trainer.train(
            boards,
            targets,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            lr_decay=args.lr_decay,
        )

        nnue.save(args.output)
        print(f"\nModel saved to {args.output}")
        print(f"Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"Final val loss: {history['val_loss'][-1]:.6f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
