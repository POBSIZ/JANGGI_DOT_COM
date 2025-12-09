"""NNUE-based evaluation function for Janggi with advanced features."""

import numpy as np
import json
from typing import List, Tuple
from .board import Board, PieceType, Side


class AdamOptimizer:
    """Adam optimizer for neural network training."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}  # First moment
        self.v = {}  # Second moment

    def update(
        self, param_name: str, param: np.ndarray, grad: np.ndarray
    ) -> np.ndarray:
        """Update parameter using Adam algorithm."""
        self.t += 1

        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)

        # Update biased first moment estimate
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
        # Update biased second raw moment estimate
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (
            grad**2
        )

        # Compute bias-corrected first moment estimate
        m_hat = self.m[param_name] / (1 - self.beta1**self.t)
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v[param_name] / (1 - self.beta2**self.t)

        # Update parameters
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def reset(self):
        """Reset optimizer state."""
        self.t = 0
        self.m = {}
        self.v = {}


class NNUE:
    """Enhanced NNUE evaluator with improved architecture and training."""

    # Feature dimensions
    PIECE_SQUARE_FEATURES = 90 * 7 * 2  # 90 squares * 7 piece types * 2 sides = 1260
    MATERIAL_FEATURES = 14  # 7 piece types * 2 sides
    MOBILITY_FEATURES = 14  # Mobility for each piece type per side
    KING_SAFETY_FEATURES = 8  # King safety metrics
    POSITIONAL_FEATURES = 20  # Pawn advancement, control, etc.

    DEFAULT_FEATURE_SIZE = 512  # Compressed feature size
    DEFAULT_HIDDEN1_SIZE = 256
    DEFAULT_HIDDEN2_SIZE = 64

    def __init__(
        self,
        feature_size: int = DEFAULT_FEATURE_SIZE,
        hidden1_size: int = DEFAULT_HIDDEN1_SIZE,
        hidden2_size: int = DEFAULT_HIDDEN2_SIZE,
        use_advanced_features: bool = True,
    ):
        """Initialize NNUE network with 2-layer architecture."""
        self.feature_size = feature_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.use_advanced_features = use_advanced_features

        # He initialization for ReLU-like activations
        self.w1 = np.random.randn(feature_size, hidden1_size) * np.sqrt(
            2.0 / feature_size
        )
        self.b1 = np.zeros(hidden1_size)

        self.w2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(
            2.0 / hidden1_size
        )
        self.b2 = np.zeros(hidden2_size)

        self.w3 = np.random.randn(hidden2_size, 1) * np.sqrt(2.0 / hidden2_size)
        self.b3 = np.zeros(1)

        # Training state
        self._cache = {}
        self.optimizer = AdamOptimizer()

        # Piece values for basic evaluation
        self.piece_values = {
            PieceType.KING: 0,
            PieceType.ROOK: 13,
            PieceType.CANNON: 7,
            PieceType.HORSE: 5,
            PieceType.ELEPHANT: 3,
            PieceType.GUARD: 3,
            PieceType.PAWN: 2,
        }

    def evaluate(self, board: Board) -> float:
        """Evaluate board position from perspective of side to move."""
        features = self._extract_features(board)
        return self._forward(features)

    def _extract_features(self, board: Board) -> np.ndarray:
        """Extract comprehensive feature vector from board position."""
        features = np.zeros(self.feature_size)
        feature_idx = 0

        # === 1. Material Features ===
        han_material = np.zeros(7)
        cho_material = np.zeros(7)
        piece_positions = {Side.HAN: {}, Side.CHO: {}}

        piece_type_list = list(PieceType)

        for rank in range(board.RANKS):
            for file in range(board.FILES):
                piece = board.get_piece(file, rank)
                if piece is None:
                    continue

                pt_idx = piece_type_list.index(piece.piece_type)
                if piece.side == Side.HAN:
                    han_material[pt_idx] += 1
                else:
                    cho_material[pt_idx] += 1

                # Store positions for mobility calculation
                if piece.piece_type not in piece_positions[piece.side]:
                    piece_positions[piece.side][piece.piece_type] = []
                piece_positions[piece.side][piece.piece_type].append((file, rank))

        # Material count features (normalized)
        for i in range(7):
            features[feature_idx] = han_material[i] / 2.0  # Normalize by max count
            features[feature_idx + 1] = cho_material[i] / 2.0
            feature_idx += 2

        # Material difference
        han_total = sum(
            han_material[i] * list(self.piece_values.values())[i] for i in range(7)
        )
        cho_total = sum(
            cho_material[i] * list(self.piece_values.values())[i] for i in range(7)
        )

        if board.side_to_move == Side.HAN:
            features[feature_idx] = (han_total - cho_total + 1.5) / 72.0
        else:
            features[feature_idx] = (cho_total - han_total) / 72.0
        feature_idx += 1

        # === 2. Piece Position Features (simplified piece-square table) ===
        if self.use_advanced_features:
            for rank in range(board.RANKS):
                for file in range(board.FILES):
                    piece = board.get_piece(file, rank)
                    if piece is None:
                        continue

                    # Encode position with piece type
                    square_feature = self._encode_piece_position(
                        piece, file, rank, board
                    )
                    if feature_idx < self.feature_size - 50:
                        features[feature_idx : feature_idx + len(square_feature)] = (
                            square_feature
                        )
                        feature_idx += len(square_feature)

        # === 3. Mobility Features ===
        if self.use_advanced_features and feature_idx < self.feature_size - 20:
            han_mobility, cho_mobility = self._calculate_mobility(
                board, piece_positions
            )
            features[feature_idx] = han_mobility / 50.0  # Normalize
            features[feature_idx + 1] = cho_mobility / 50.0
            feature_idx += 2

        # === 4. King Safety Features ===
        if self.use_advanced_features and feature_idx < self.feature_size - 10:
            han_king_safety = self._calculate_king_safety(board, Side.HAN)
            cho_king_safety = self._calculate_king_safety(board, Side.CHO)
            features[feature_idx] = han_king_safety
            features[feature_idx + 1] = cho_king_safety
            feature_idx += 2

        # === 5. Check/Threat Features (simplified - skip expensive check detection) ===
        # We skip is_in_check here as it's expensive and provides marginal benefit
        # The search algorithm already handles check detection
        feature_idx += 2

        # === 6. Pawn Advancement Features ===
        if self.use_advanced_features and feature_idx < self.feature_size - 5:
            han_pawn_adv, cho_pawn_adv = self._calculate_pawn_advancement(
                board, piece_positions
            )
            features[feature_idx] = han_pawn_adv
            features[feature_idx + 1] = cho_pawn_adv
            feature_idx += 2

        # === 7. Cannon Features (포대 여부) ===
        if self.use_advanced_features and feature_idx < self.feature_size - 5:
            han_cannon_power, cho_cannon_power = self._calculate_cannon_power(
                board, piece_positions
            )
            features[feature_idx] = han_cannon_power
            features[feature_idx + 1] = cho_cannon_power
            feature_idx += 2

        return features

    def _encode_piece_position(
        self, piece, file: int, rank: int, board: Board
    ) -> np.ndarray:
        """Encode piece position with contextual features."""
        features = np.zeros(4)

        # Centrality (pieces in center are generally stronger)
        center_file = 4
        center_rank = 4.5
        centrality = (
            1.0 - (abs(file - center_file) / 4.0 + abs(rank - center_rank) / 5.0) / 2.0
        )
        features[0] = centrality

        # Advancement (how far the piece has progressed)
        if piece.side == Side.HAN:
            advancement = rank / 9.0  # HAN advances upward
        else:
            advancement = (9 - rank) / 9.0  # CHO advances downward
        features[1] = advancement

        # In palace (for kings and guards)
        in_palace = self._is_in_palace(file, rank, piece.side)
        features[2] = 1.0 if in_palace else 0.0

        # Side encoding
        features[3] = 1.0 if piece.side == Side.HAN else -1.0

        return features

    def _is_in_palace(self, file: int, rank: int, side: Side) -> bool:
        """Check if position is in the palace."""
        if file < 3 or file > 5:
            return False
        if side == Side.HAN:
            return rank <= 2
        else:
            return rank >= 7

    def _calculate_mobility(
        self, board: Board, piece_positions: dict
    ) -> Tuple[float, float]:
        """Calculate approximate mobility based on piece positions (fast estimate).
        
        Instead of generating all legal moves (very expensive), we estimate
        mobility based on piece types and their positions.
        """
        han_mobility = 0.0
        cho_mobility = 0.0
        
        # Approximate mobility values per piece type
        mobility_values = {
            PieceType.KING: 4,      # Can move ~4 directions in palace
            PieceType.GUARD: 4,     # Can move ~4 directions in palace
            PieceType.ELEPHANT: 4,  # Can move ~4 directions (if not blocked)
            PieceType.HORSE: 4,     # Can move ~4 directions (if not blocked)
            PieceType.ROOK: 10,     # High mobility
            PieceType.CANNON: 6,    # Medium mobility (needs screens)
            PieceType.PAWN: 3,      # 3 directions (forward, left, right)
        }
        
        for side in [Side.HAN, Side.CHO]:
            side_mobility = 0.0
            for piece_type, positions in piece_positions[side].items():
                base_value = mobility_values.get(piece_type, 3)
                for file, rank in positions:
                    # Adjust based on position (center pieces have more mobility)
                    center_bonus = 1.0 - (abs(file - 4) / 4.0 + abs(rank - 4.5) / 5.0) / 2.0
                    side_mobility += base_value * (0.7 + 0.3 * center_bonus)
            
            if side == Side.HAN:
                han_mobility = side_mobility
            else:
                cho_mobility = side_mobility
        
        return han_mobility, cho_mobility

    def _calculate_king_safety(self, board: Board, side: Side) -> float:
        """Calculate king safety score."""
        safety = 1.0

        # Find king position
        king_pos = None
        for rank in range(board.RANKS):
            for file in range(board.FILES):
                piece = board.get_piece(file, rank)
                if piece and piece.piece_type == PieceType.KING and piece.side == side:
                    king_pos = (file, rank)
                    break
            if king_pos:
                break

        if not king_pos:
            return 0.0

        king_file, king_rank = king_pos

        # Count defenders (guards) near king
        defenders = 0
        for df in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                nf, nr = king_file + df, king_rank + dr
                if 0 <= nf < board.FILES and 0 <= nr < board.RANKS:
                    piece = board.get_piece(nf, nr)
                    if (
                        piece
                        and piece.side == side
                        and piece.piece_type == PieceType.GUARD
                    ):
                        defenders += 1

        safety += defenders * 0.2

        # Penalty if king is exposed (not in center of palace)
        palace_center = 4
        palace_rank = 1 if side == Side.HAN else 8
        dist_from_center = abs(king_file - palace_center) + abs(king_rank - palace_rank)
        safety -= dist_from_center * 0.1

        return max(0.0, min(1.0, safety))

    def _calculate_pawn_advancement(
        self, board: Board, piece_positions: dict
    ) -> Tuple[float, float]:
        """Calculate pawn advancement score."""
        han_adv = 0.0
        cho_adv = 0.0

        if PieceType.PAWN in piece_positions[Side.HAN]:
            for file, rank in piece_positions[Side.HAN][PieceType.PAWN]:
                han_adv += rank / 9.0  # HAN pawns advance upward

        if PieceType.PAWN in piece_positions[Side.CHO]:
            for file, rank in piece_positions[Side.CHO][PieceType.PAWN]:
                cho_adv += (9 - rank) / 9.0  # CHO pawns advance downward

        # Normalize by max pawns (5)
        return han_adv / 5.0, cho_adv / 5.0

    def _calculate_cannon_power(
        self, board: Board, piece_positions: dict
    ) -> Tuple[float, float]:
        """Calculate cannon power (based on available screens)."""
        han_power = 0.0
        cho_power = 0.0

        # Count pieces that can serve as cannon screens
        for side in [Side.HAN, Side.CHO]:
            if PieceType.CANNON not in piece_positions[side]:
                continue

            for cannon_file, cannon_rank in piece_positions[side][PieceType.CANNON]:
                # Count available targets with screens
                screens = 0

                # Horizontal check
                for direction in [-1, 1]:
                    found_screen = False
                    for d in range(1, 9):
                        check_file = cannon_file + direction * d
                        if check_file < 0 or check_file >= board.FILES:
                            break
                        piece = board.get_piece(check_file, cannon_rank)
                        if piece:
                            if not found_screen:
                                found_screen = True
                            else:
                                screens += 1
                                break

                # Vertical check
                for direction in [-1, 1]:
                    found_screen = False
                    for d in range(1, 10):
                        check_rank = cannon_rank + direction * d
                        if check_rank < 0 or check_rank >= board.RANKS:
                            break
                        piece = board.get_piece(cannon_file, check_rank)
                        if piece:
                            if not found_screen:
                                found_screen = True
                            else:
                                screens += 1
                                break

                if side == Side.HAN:
                    han_power += screens / 4.0  # Normalize
                else:
                    cho_power += screens / 4.0

        return min(1.0, han_power), min(1.0, cho_power)

    def _forward(self, features: np.ndarray, training: bool = False) -> float:
        """Forward pass through 2-layer network with Clipped ReLU."""
        # Layer 1
        z1 = features @ self.w1 + self.b1
        h1 = np.clip(z1, 0, 1)  # Clipped ReLU (CReLU)

        # Layer 2
        z2 = h1 @ self.w2 + self.b2
        h2 = np.clip(z2, 0, 1)  # Clipped ReLU

        # Output layer (no activation for regression)
        z3 = h2 @ self.w3 + self.b3
        output = float(z3[0])

        # Cache for backpropagation
        if training:
            self._cache = {
                "features": features,
                "z1": z1,
                "h1": h1,
                "z2": z2,
                "h2": h2,
                "z3": z3,
            }

        return output

    def _backward(self, target: float, use_adam: bool = True) -> float:
        """Backward pass with Huber loss and Adam optimizer."""
        if not self._cache:
            raise RuntimeError("Must call _forward with training=True before _backward")

        features = self._cache["features"]
        z1, h1 = self._cache["z1"], self._cache["h1"]
        z2, h2 = self._cache["z2"], self._cache["h2"]
        z3 = self._cache["z3"]

        prediction = z3[0]
        error = prediction - target

        # Huber loss (more robust to outliers)
        delta = 1.0
        if abs(error) <= delta:
            loss = 0.5 * error**2
            d_loss = error
        else:
            loss = delta * (abs(error) - 0.5 * delta)
            d_loss = delta * np.sign(error)

        # Output layer gradients
        d_z3 = np.array([d_loss])
        d_w3 = h2.reshape(-1, 1) @ d_z3.reshape(1, -1)
        d_b3 = d_z3

        # Layer 2 gradients (Clipped ReLU derivative)
        d_h2 = d_z3 * self.w3.flatten()
        d_z2 = d_h2 * ((z2 > 0) & (z2 < 1)).astype(float)
        d_w2 = h1.reshape(-1, 1) @ d_z2.reshape(1, -1)
        d_b2 = d_z2

        # Layer 1 gradients
        d_h1 = d_z2 @ self.w2.T
        d_z1 = d_h1 * ((z1 > 0) & (z1 < 1)).astype(float)
        d_w1 = features.reshape(-1, 1) @ d_z1.reshape(1, -1)
        d_b1 = d_z1

        # Update weights
        if use_adam:
            self.w3 = self.optimizer.update("w3", self.w3, d_w3)
            self.b3 = self.optimizer.update("b3", self.b3, d_b3)
            self.w2 = self.optimizer.update("w2", self.w2, d_w2)
            self.b2 = self.optimizer.update("b2", self.b2, d_b2)
            self.w1 = self.optimizer.update("w1", self.w1, d_w1)
            self.b1 = self.optimizer.update("b1", self.b1, d_b1)
        else:
            lr = self.optimizer.lr
            self.w3 -= lr * d_w3
            self.b3 -= lr * d_b3
            self.w2 -= lr * d_w2
            self.b2 -= lr * d_b2
            self.w1 -= lr * d_w1
            self.b1 -= lr * d_b1

        return loss

    def set_learning_rate(self, lr: float):
        """Set learning rate."""
        self.optimizer.lr = lr

    def train_step(self, board: Board, target: float) -> float:
        """Single training step."""
        features = self._extract_features(board)
        self._forward(features, training=True)
        return self._backward(target)

    def train_batch(self, boards: List[Board], targets: List[float]) -> float:
        """Train on a batch of positions."""
        total_loss = 0.0
        for board, target in zip(boards, targets):
            loss = self.train_step(board, target)
            total_loss += loss
        return total_loss / len(boards) if boards else 0.0

    def save(self, filepath: str):
        """Save model weights to file."""
        data = {
            "version": 2,  # New version with 2-layer architecture
            "feature_size": self.feature_size,
            "hidden1_size": self.hidden1_size,
            "hidden2_size": self.hidden2_size,
            "use_advanced_features": self.use_advanced_features,
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
            "w3": self.w3.tolist(),
            "b3": self.b3.tolist(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load(self, filepath: str):
        """Load model weights from file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        version = data.get("version", 1)

        if version == 1:
            # Legacy format - convert to new format
            self.feature_size = data["feature_size"]
            self.hidden1_size = data["hidden_size"]
            self.hidden2_size = 32
            self.use_advanced_features = False

            # Use old weights for first layer, initialize rest
            self.w1 = np.array(data["input_weights"])
            self.b1 = np.array(data["input_bias"])
            self.w2 = np.random.randn(self.hidden1_size, self.hidden2_size) * 0.1
            self.b2 = np.zeros(self.hidden2_size)
            self.w3 = np.array(data["hidden_weights"]).reshape(-1, 1) @ np.ones(
                (1, self.hidden2_size)
            )
            self.w3 = np.random.randn(self.hidden2_size, 1) * 0.1
            self.b3 = np.array(data["hidden_bias"])
        elif version == 3:
            # PyTorch format (from GPU training)
            self.feature_size = data["feature_size"]
            self.hidden1_size = data["hidden1_size"]
            self.hidden2_size = data["hidden2_size"]
            self.use_advanced_features = True

            # Convert from PyTorch state_dict format
            # PyTorch stores weights as [out_features, in_features], we need [in_features, out_features]
            state = data["state_dict"]
            self.w1 = np.array(state["fc1.weight"]).T
            self.b1 = np.array(state["fc1.bias"])
            self.w2 = np.array(state["fc2.weight"]).T
            self.b2 = np.array(state["fc2.bias"])
            self.w3 = np.array(state["fc3.weight"]).T
            self.b3 = np.array(state["fc3.bias"])
        else:
            # Version 2 format (NumPy)
            self.feature_size = data["feature_size"]
            self.hidden1_size = data["hidden1_size"]
            self.hidden2_size = data["hidden2_size"]
            self.use_advanced_features = data.get("use_advanced_features", True)
            self.w1 = np.array(data["w1"])
            self.b1 = np.array(data["b1"])
            self.w2 = np.array(data["w2"])
            self.b2 = np.array(data["b2"])
            self.w3 = np.array(data["w3"])
            self.b3 = np.array(data["b3"])

    @classmethod
    def from_file(cls, filepath: str) -> "NNUE":
        """Create NNUE instance from saved file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        version = data.get("version", 1)

        if version == 1:
            instance = cls(
                feature_size=data["feature_size"],
                hidden1_size=data["hidden_size"],
                hidden2_size=32,
                use_advanced_features=False,
            )
        elif version == 3:
            # PyTorch format
            instance = cls(
                feature_size=data["feature_size"],
                hidden1_size=data["hidden1_size"],
                hidden2_size=data["hidden2_size"],
                use_advanced_features=True,
            )
        else:
            instance = cls(
                feature_size=data["feature_size"],
                hidden1_size=data["hidden1_size"],
                hidden2_size=data["hidden2_size"],
                use_advanced_features=data.get("use_advanced_features", True),
            )

        instance.load(filepath)
        return instance


class SimpleEvaluator:
    """Simple material-based evaluator (fallback) - optimized for speed."""

    PIECE_VALUES = {
        PieceType.KING: 0,
        PieceType.ROOK: 13,
        PieceType.CANNON: 7,
        PieceType.HORSE: 5,
        PieceType.ELEPHANT: 3,
        PieceType.GUARD: 3,
        PieceType.PAWN: 2,
    }

    @staticmethod
    def evaluate(board: Board) -> float:
        """Evaluate board position (fast - material only)."""
        han_material = 0
        cho_material = 0

        for rank in range(board.RANKS):
            for file in range(board.FILES):
                piece = board.board[rank][file]  # Direct access (faster)
                if piece is None:
                    continue

                value = SimpleEvaluator.PIECE_VALUES.get(piece.piece_type, 0)
                if piece.side == Side.HAN:
                    han_material += value
                else:
                    cho_material += value

        # Second player (Han) gets +1.5 adjustment
        if board.side_to_move == Side.HAN:
            score = han_material - cho_material + 1.5
        else:
            score = cho_material - han_material

        # Skip expensive check/checkmate/stalemate detection
        # The search algorithm handles terminal states
        return score
