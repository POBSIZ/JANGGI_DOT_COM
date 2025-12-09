"""PyTorch-based NNUE for GPU-accelerated training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import List, Tuple, Optional, Dict
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from .board import Board, PieceType, Side


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")


class NNUENet(nn.Module):
    """PyTorch Neural Network for NNUE evaluation."""
    
    def __init__(
        self, 
        feature_size: int = 512, 
        hidden1_size: int = 256,
        hidden2_size: int = 64
    ):
        super().__init__()
        
        self.feature_size = feature_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        
        # Network layers
        self.fc1 = nn.Linear(feature_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, 1)
        
        # Initialize weights (He initialization)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.xavier_normal_(self.fc3.weight)
        
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Clipped ReLU activation."""
        x = torch.clamp(F.relu(self.fc1(x)), 0, 1)  # Clipped ReLU
        x = torch.clamp(F.relu(self.fc2(x)), 0, 1)
        x = self.fc3(x)
        return x


class FeatureExtractor:
    """Extract features from board positions."""
    
    PIECE_VALUES = {
        PieceType.KING: 0,
        PieceType.ROOK: 13,
        PieceType.CANNON: 7,
        PieceType.HORSE: 5,
        PieceType.ELEPHANT: 3,
        PieceType.GUARD: 3,
        PieceType.PAWN: 2,
    }
    
    def __init__(self, feature_size: int = 512):
        self.feature_size = feature_size
        self.piece_type_list = list(PieceType)
    
    def extract(self, board: Board) -> np.ndarray:
        """Extract feature vector from board position."""
        features = np.zeros(self.feature_size, dtype=np.float32)
        feature_idx = 0
        
        # === 1. Material Features ===
        han_material = np.zeros(7)
        cho_material = np.zeros(7)
        piece_positions = {Side.HAN: {}, Side.CHO: {}}
        
        for rank in range(board.RANKS):
            for file in range(board.FILES):
                piece = board.get_piece(file, rank)
                if piece is None:
                    continue
                
                pt_idx = self.piece_type_list.index(piece.piece_type)
                if piece.side == Side.HAN:
                    han_material[pt_idx] += 1
                else:
                    cho_material[pt_idx] += 1
                
                if piece.piece_type not in piece_positions[piece.side]:
                    piece_positions[piece.side][piece.piece_type] = []
                piece_positions[piece.side][piece.piece_type].append((file, rank))
        
        # Material count features (normalized)
        for i in range(7):
            features[feature_idx] = han_material[i] / 2.0
            features[feature_idx + 1] = cho_material[i] / 2.0
            feature_idx += 2
        
        # Material difference
        han_total = sum(han_material[i] * list(self.PIECE_VALUES.values())[i] for i in range(7))
        cho_total = sum(cho_material[i] * list(self.PIECE_VALUES.values())[i] for i in range(7))
        
        if board.side_to_move == Side.HAN:
            features[feature_idx] = (han_total - cho_total + 1.5) / 72.0
        else:
            features[feature_idx] = (cho_total - han_total) / 72.0
        feature_idx += 1
        
        # === 2. Piece Position Features ===
        for rank in range(board.RANKS):
            for file in range(board.FILES):
                piece = board.get_piece(file, rank)
                if piece is None:
                    continue
                
                if feature_idx < self.feature_size - 50:
                    # Centrality
                    centrality = 1.0 - (abs(file - 4) / 4.0 + abs(rank - 4.5) / 5.0) / 2.0
                    features[feature_idx] = centrality
                    feature_idx += 1
                    
                    # Advancement
                    if piece.side == Side.HAN:
                        advancement = rank / 9.0
                    else:
                        advancement = (9 - rank) / 9.0
                    features[feature_idx] = advancement
                    feature_idx += 1
                    
                    # Side encoding
                    features[feature_idx] = 1.0 if piece.side == Side.HAN else -1.0
                    feature_idx += 1
        
        # === 3. Mobility Features ===
        if feature_idx < self.feature_size - 10:
            han_mobility, cho_mobility = self._calculate_mobility(board)
            features[feature_idx] = han_mobility / 50.0
            features[feature_idx + 1] = cho_mobility / 50.0
            feature_idx += 2
        
        # === 4. King Safety ===
        if feature_idx < self.feature_size - 10:
            han_safety = self._calculate_king_safety(board, Side.HAN)
            cho_safety = self._calculate_king_safety(board, Side.CHO)
            features[feature_idx] = han_safety
            features[feature_idx + 1] = cho_safety
            feature_idx += 2
        
        # === 5. Check status ===
        if feature_idx < self.feature_size - 5:
            features[feature_idx] = 1.0 if board.is_in_check(board.side_to_move) else 0.0
            opponent = Side.CHO if board.side_to_move == Side.HAN else Side.HAN
            features[feature_idx + 1] = 1.0 if board.is_in_check(opponent) else 0.0
            feature_idx += 2
        
        # === 6. Pawn advancement ===
        if feature_idx < self.feature_size - 5:
            han_adv, cho_adv = self._calculate_pawn_advancement(piece_positions)
            features[feature_idx] = han_adv
            features[feature_idx + 1] = cho_adv
            feature_idx += 2
        
        return features
    
    def _calculate_mobility(self, board: Board) -> Tuple[float, float]:
        """Calculate approximate mobility based on piece positions (fast estimate).
        
        Instead of generating all legal moves (very expensive), we estimate
        mobility based on piece types and their positions.
        """
        # Approximate mobility values per piece type
        mobility_values = {
            PieceType.KING: 4,
            PieceType.GUARD: 4,
            PieceType.ELEPHANT: 4,
            PieceType.HORSE: 4,
            PieceType.ROOK: 10,
            PieceType.CANNON: 6,
            PieceType.PAWN: 3,
        }
        
        han_mobility = 0.0
        cho_mobility = 0.0
        
        for rank in range(board.RANKS):
            for file in range(board.FILES):
                piece = board.get_piece(file, rank)
                if piece is None:
                    continue
                
                base_value = mobility_values.get(piece.piece_type, 3)
                # Center bonus
                center_bonus = 1.0 - (abs(file - 4) / 4.0 + abs(rank - 4.5) / 5.0) / 2.0
                value = base_value * (0.7 + 0.3 * center_bonus)
                
                if piece.side == Side.HAN:
                    han_mobility += value
                else:
                    cho_mobility += value
        
        return han_mobility, cho_mobility
    
    def _calculate_king_safety(self, board: Board, side: Side) -> float:
        """Calculate king safety score."""
        safety = 1.0
        
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
        
        # Count defenders
        defenders = 0
        for df in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                nf, nr = king_file + df, king_rank + dr
                if 0 <= nf < board.FILES and 0 <= nr < board.RANKS:
                    piece = board.get_piece(nf, nr)
                    if piece and piece.side == side and piece.piece_type == PieceType.GUARD:
                        defenders += 1
        
        safety += defenders * 0.2
        
        palace_rank = 1 if side == Side.HAN else 8
        dist = abs(king_file - 4) + abs(king_rank - palace_rank)
        safety -= dist * 0.1
        
        return max(0.0, min(1.0, safety))
    
    def _calculate_pawn_advancement(self, piece_positions: dict) -> Tuple[float, float]:
        """Calculate pawn advancement score."""
        han_adv = cho_adv = 0.0
        
        if PieceType.PAWN in piece_positions[Side.HAN]:
            for file, rank in piece_positions[Side.HAN][PieceType.PAWN]:
                han_adv += rank / 9.0
        
        if PieceType.PAWN in piece_positions[Side.CHO]:
            for file, rank in piece_positions[Side.CHO][PieceType.PAWN]:
                cho_adv += (9 - rank) / 9.0
        
        return han_adv / 5.0, cho_adv / 5.0
    
    def extract_batch(self, boards: List[Board], num_workers: Optional[int] = None) -> np.ndarray:
        """Extract features for multiple boards in parallel.
        
        Args:
            boards: List of Board objects to extract features from
            num_workers: Number of parallel workers (None = auto, uses CPU count)
        
        Returns:
            Array of feature vectors (shape: [batch_size, feature_size])
        """
        if len(boards) == 0:
            return np.array([], dtype=np.float32).reshape(0, self.feature_size)
        
        # For small batches, sequential processing is faster (no overhead)
        if len(boards) < 10:
            return np.array([self.extract(b) for b in boards], dtype=np.float32)
        
        # Use parallel processing for larger batches
        if num_workers is None:
            num_workers = min(mp.cpu_count(), len(boards))
        
        # Use ThreadPoolExecutor for I/O-bound or GIL-friendly operations
        # Since feature extraction is mostly CPU-bound with NumPy, we use threads
        # (NumPy releases GIL for many operations)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            features = list(executor.map(self.extract, boards))
        
        return np.array(features, dtype=np.float32)


class NNUETorch:
    """PyTorch-based NNUE with GPU support."""
    
    def __init__(
        self, 
        feature_size: int = 512, 
        hidden1_size: int = 256,
        hidden2_size: int = 64,
        device: Optional[torch.device] = None,
        eval_cache_size: int = 1000
    ):
        self.feature_size = feature_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        
        self.device = device or get_device()
        print(f"Using device: {self.device}")
        
        self.model = NNUENet(feature_size, hidden1_size, hidden2_size).to(self.device)
        self.feature_extractor = FeatureExtractor(feature_size)
        
        # LRU cache for single position evaluations (using Zobrist hash as key)
        # This helps when the same position is evaluated multiple times
        self._eval_cache: OrderedDict[int, float] = OrderedDict()
        self._eval_cache_size = eval_cache_size
    
    def evaluate(self, board: Board, use_cache: bool = True) -> float:
        """Evaluate board position with optional caching.
        
        Args:
            board: Board position to evaluate
            use_cache: If True, use LRU cache for repeated positions (default: True)
        
        Returns:
            Evaluation score
        """
        # Try cache first if enabled
        if use_cache:
            try:
                position_hash = board.get_zobrist_hash()
                if position_hash in self._eval_cache:
                    # Move to end (most recently used)
                    score = self._eval_cache.pop(position_hash)
                    self._eval_cache[position_hash] = score
                    return score
            except (AttributeError, Exception):
                # If board doesn't have get_zobrist_hash() or other error, skip cache
                pass
        
        # Evaluate position
        self.model.eval()
        with torch.no_grad():
            features = self.feature_extractor.extract(board)
            x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            output = self.model(x)
            score = output.item()
        
        # Store in cache if enabled
        if use_cache:
            try:
                position_hash = board.get_zobrist_hash()
                # Add to cache (or update if already exists)
                if position_hash in self._eval_cache:
                    self._eval_cache.pop(position_hash)
                self._eval_cache[position_hash] = score
                
                # Evict oldest entry if cache is full
                if len(self._eval_cache) > self._eval_cache_size:
                    self._eval_cache.popitem(last=False)  # Remove oldest (first) item
            except (AttributeError, Exception):
                # If board doesn't have get_zobrist_hash() or other error, skip cache
                pass
        
        return score
    
    def clear_eval_cache(self):
        """Clear the evaluation cache."""
        self._eval_cache.clear()
    
    def evaluate_batch(self, boards: List[Board] = None, features: np.ndarray = None) -> np.ndarray:
        """Evaluate multiple boards at once (efficient for GPU).
        
        Args:
            boards: List of Board objects to evaluate (if features not provided)
            features: Pre-extracted feature array (shape: [batch_size, feature_size])
                     If provided, boards parameter is ignored.
        
        Returns:
            Array of evaluation scores
        """
        self.model.eval()
        with torch.no_grad():
            if features is not None:
                # Use pre-extracted features (avoids duplicate extraction)
                x = torch.tensor(features, dtype=torch.float32, device=self.device)
            else:
                # Extract features from boards
                if boards is None:
                    raise ValueError("Either boards or features must be provided")
                features = self.feature_extractor.extract_batch(boards)
                x = torch.tensor(features, dtype=torch.float32, device=self.device)
            outputs = self.model(x)
            return outputs.cpu().numpy().flatten()
    
    def save(self, filepath: str):
        """Save model to file."""
        data = {
            'version': 3,  # PyTorch version
            'feature_size': self.feature_size,
            'hidden1_size': self.hidden1_size,
            'hidden2_size': self.hidden2_size,
            'state_dict': {k: v.cpu().tolist() for k, v in self.model.state_dict().items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def save_torch(self, filepath: str):
        """Save model in PyTorch format (more efficient)."""
        torch.save({
            'feature_size': self.feature_size,
            'hidden1_size': self.hidden1_size,
            'hidden2_size': self.hidden2_size,
            'state_dict': self.model.state_dict()
        }, filepath)
    
    def load(self, filepath: str):
        """Load model from file."""
        if filepath.endswith('.pt') or filepath.endswith('.pth'):
            self._load_torch(filepath)
        else:
            self._load_json(filepath)
    
    def _load_json(self, filepath: str):
        """Load from JSON format."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        version = data.get('version', 1)
        
        if version < 3:
            # Convert from NumPy format
            self._convert_from_numpy(data)
        else:
            self.feature_size = data['feature_size']
            self.hidden1_size = data['hidden1_size']
            self.hidden2_size = data['hidden2_size']
            
            self.model = NNUENet(
                self.feature_size, 
                self.hidden1_size, 
                self.hidden2_size
            ).to(self.device)
            
            state_dict = {k: torch.tensor(v) for k, v in data['state_dict'].items()}
            self.model.load_state_dict(state_dict)
    
    def _load_torch(self, filepath: str):
        """Load from PyTorch format."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.feature_size = checkpoint['feature_size']
        self.hidden1_size = checkpoint['hidden1_size']
        self.hidden2_size = checkpoint['hidden2_size']
        
        self.model = NNUENet(
            self.feature_size,
            self.hidden1_size,
            self.hidden2_size
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['state_dict'])
    
    def _convert_from_numpy(self, data: dict):
        """Convert from NumPy NNUE format."""
        version = data.get('version', 1)
        
        if version == 1:
            # Old single-layer format
            self.feature_size = data['feature_size']
            self.hidden1_size = data['hidden_size']
            self.hidden2_size = 32
        else:
            self.feature_size = data['feature_size']
            self.hidden1_size = data['hidden1_size']
            self.hidden2_size = data['hidden2_size']
        
        self.model = NNUENet(
            self.feature_size,
            self.hidden1_size,
            self.hidden2_size
        ).to(self.device)
        
        # Try to load weights if they match
        if version >= 2:
            try:
                self.model.fc1.weight.data = torch.tensor(
                    np.array(data['w1']).T, dtype=torch.float32, device=self.device
                )
                self.model.fc1.bias.data = torch.tensor(
                    data['b1'], dtype=torch.float32, device=self.device
                )
                self.model.fc2.weight.data = torch.tensor(
                    np.array(data['w2']).T, dtype=torch.float32, device=self.device
                )
                self.model.fc2.bias.data = torch.tensor(
                    data['b2'], dtype=torch.float32, device=self.device
                )
                self.model.fc3.weight.data = torch.tensor(
                    np.array(data['w3']).T, dtype=torch.float32, device=self.device
                )
                self.model.fc3.bias.data = torch.tensor(
                    data['b3'], dtype=torch.float32, device=self.device
                )
            except Exception as e:
                print(f"Warning: Could not load weights, using random initialization: {e}")
    
    @classmethod
    def from_file(cls, filepath: str, device: Optional[torch.device] = None) -> 'NNUETorch':
        """Create instance from saved file."""
        instance = cls(device=device)
        instance.load(filepath)
        return instance


class JanggiDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for Janggi positions with lazy loading support.
    
    This dataset supports both eager loading (for small datasets) and lazy loading
    (for large datasets to reduce memory usage). By default, uses lazy loading
    to keep data as numpy arrays and convert to tensors on-demand.
    """
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, lazy_loading: bool = True):
        """
        Args:
            features: Feature array (shape: [n_samples, feature_size])
            targets: Target array (shape: [n_samples])
            lazy_loading: If True, keep data as numpy arrays and convert on-demand.
                         If False, convert to tensors immediately (for small datasets).
        """
        self.lazy_loading = lazy_loading
        
        if lazy_loading:
            # Keep as numpy arrays to save memory
            self.features = features  # Keep as numpy array
            self.targets = targets    # Keep as numpy array
        else:
            # Eager loading for small datasets (faster access)
            self.features = torch.tensor(features, dtype=torch.float32)
            self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.lazy_loading:
            # Convert to tensor on-demand
            feature = torch.tensor(self.features[idx], dtype=torch.float32)
            target = torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0)
            return feature, target
        else:
            return self.features[idx], self.targets[idx]


class GPUTrainer:
    """GPU-accelerated trainer for NNUE with Mixed Precision support."""
    
    def __init__(self, nnue: NNUETorch, use_amp: bool = False):
        self.nnue = nnue
        self.device = nnue.device
        # Mixed Precision Training (AMP) - disabled by default to avoid NaN issues
        # Only enable for CUDA with proper hardware support
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
    
    def train(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        validation_split: float = 0.1,
        lr_scheduler: bool = True,
        early_stopping_patience: int = 10,
        num_workers: int = 0,
        warmup_epochs: int = 3,
        progress_callback=None
    ) -> dict:
        """Train the model using GPU acceleration with Mixed Precision."""
        # Split data
        n_samples = len(features)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        # Use lazy loading for large datasets (>100k samples) to save memory
        # Small datasets benefit from eager loading (faster access)
        use_lazy = n_samples > 100000
        train_dataset = JanggiDataset(features[train_indices], targets[train_indices], lazy_loading=use_lazy)
        val_dataset = JanggiDataset(features[val_indices], targets[val_indices], lazy_loading=use_lazy)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device.type == 'cuda'),
            persistent_workers=(num_workers > 0)
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(self.device.type == 'cuda')
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.nnue.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0
        
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        main_scheduler = None
        if lr_scheduler:
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs, eta_min=learning_rate * 0.01
            )
        
        # Loss function (Huber loss)
        criterion = nn.SmoothL1Loss()
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        amp_info = "AMP (FP16)" if self.use_amp else "FP32"
        print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
        print(f"Device: {self.device}, Batch size: {batch_size}, Precision: {amp_info}")
        
        nan_count = 0
        for epoch in range(epochs):
            # Training
            self.nnue.model.train()
            train_loss = 0.0
            batch_count = 0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)
                
                # Check for NaN in input data
                if torch.isnan(batch_features).any() or torch.isnan(batch_targets).any():
                    continue  # Skip bad batches
                
                optimizer.zero_grad(set_to_none=True)
                
                # Mixed Precision Training
                if self.use_amp and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.nnue.model(batch_features)
                        loss = criterion(outputs, batch_targets)
                    
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.nnue.model.parameters(), max_norm=0.5)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                        train_loss += loss.item() * len(batch_features)
                        batch_count += len(batch_features)
                else:
                    outputs = self.nnue.model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    
                    # Check for NaN loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        nan_count += 1
                        if nan_count > 10:
                            print("\nWarning: Too many NaN losses. Reinitializing model...")
                            # Reinitialize weights
                            for m in self.nnue.model.modules():
                                if isinstance(m, nn.Linear):
                                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                                    nn.init.zeros_(m.bias)
                            nan_count = 0
                        continue
                    
                    loss.backward()
                    # Stronger gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(self.nnue.model.parameters(), max_norm=0.5)
                    optimizer.step()
                    train_loss += loss.item() * len(batch_features)
                    batch_count += len(batch_features)
            
            train_loss = train_loss / batch_count if batch_count > 0 else float('nan')
            
            # Validation
            self.nnue.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device, non_blocking=True)
                    batch_targets = batch_targets.to(self.device, non_blocking=True)
                    
                    if self.use_amp and self.device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            outputs = self.nnue.model(batch_features)
                            loss = criterion(outputs, batch_targets)
                    else:
                        outputs = self.nnue.model(batch_features)
                        loss = criterion(outputs, batch_targets)
                    
                    val_loss += loss.item() * len(batch_features)
            
            val_loss /= len(val_dataset)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(current_lr)
            
            if progress_callback:
                progress_callback(epoch + 1, epochs, train_loss, val_loss, current_lr)
            else:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                      f"LR: {current_lr:.6f}")
            
            # Learning rate scheduling (warmup -> main scheduler)
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            elif main_scheduler:
                main_scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.nnue.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Restore best model
        if best_state:
            self.nnue.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        
        return history

