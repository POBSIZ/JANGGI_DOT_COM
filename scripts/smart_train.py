#!/usr/bin/env python3
"""Smart NNUE Training Script - 자동 환경 감지 및 최적화 학습.

컴퓨터 환경을 자동으로 파악하여 최적의 학습 방식을 선택합니다.
학습 시간을 선택하면 해당 시간에 맞는 설정으로 자동 학습합니다.

Usage:
    # 대화형 모드 (권장)
    python smart_train.py
    
    # 직접 시간 선택
    python smart_train.py --time quick       # ~5분
    python smart_train.py --time standard    # ~15분  
    python smart_train.py --time deep        # ~30분
    python smart_train.py --time intensive   # ~1시간
    python smart_train.py --time full        # ~3시간 (강화된 설정)
    python smart_train.py --time extreme     # ~4시간 (최강 성능)
    python smart_train.py --time marathon    # ~8시간 (최종 보스)
    
    # 기존 모델에서 계속 학습
    python smart_train.py --load models/nnue_model.json --time standard
"""

import argparse
import os
import sys
import platform
import time
import glob
import multiprocessing as mp
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================================
# Configuration Constants
# ============================================================================

# System Requirements
MIN_CPU_CORES_FOR_PARALLEL = 4
MIN_RAM_GB = 8
MIN_GIBO_FILES_FOR_TRAINING = 5
MAX_WORKERS = 8
MIN_WORKERS_LOW_RAM = 2

# GPU Memory Thresholds (GB)
GPU_MEMORY_HIGH = 8
GPU_MEMORY_MEDIUM = 4

# Time Adjustment Factors
GPU_TIME_REDUCTION_FULL_MODE = 0.7
GPU_TIME_REDUCTION_NORMAL_MODE = 0.5
CPU_POSITION_REDUCTION_LOW_CORES = 0.7
RAM_POSITION_REDUCTION = 0.5

# Position Scaling Factors
GPU_POSITION_SCALE_HIGH_MEMORY_FULL = 2.0
GPU_POSITION_SCALE_HIGH_MEMORY_NORMAL = 1.5
GPU_POSITION_SCALE_MEDIUM_MEMORY = 1.5

# Time Estimation Constants (seconds per unit)
# 포지션 생성 시간 (초/포지션)
POSITION_GEN_TIME_CPU_SINGLE = 0.01      # CPU 단일 스레드
POSITION_GEN_TIME_CPU_PARALLEL = 0.003   # CPU 병렬 (워커당)
POSITION_GEN_TIME_GPU = 0.001            # GPU 가속
DEPTH_TIME_MULTIPLIER = 1.5              # 깊이당 시간 배수

# 학습 시간 (초/에포크/1000포지션)
TRAINING_TIME_CPU_PER_1K = 2.0           # CPU: 1000포지션당 2초/에포크
TRAINING_TIME_GPU_PER_1K = 0.3           # GPU: 1000포지션당 0.3초/에포크
BATCH_SIZE_EFFICIENCY = {                # 배치 사이즈별 효율
    64: 1.0,
    128: 0.9,
    256: 0.8,
    512: 0.7,
    1024: 0.6
}

# 기보 처리 시간
GIBO_PARSE_TIME_PER_GAME = 0.01         # 게임당 파싱 시간 (초)
GIBO_PROCESS_TIME_PER_POSITION = 0.0005  # 포지션당 처리 시간 (초)

# 반복 학습 오버헤드
ITERATION_OVERHEAD = 1.1                # 반복당 10% 오버헤드


class TrainingTime(Enum):
    """학습 시간 옵션"""
    QUICK = "quick"           # ~5분
    STANDARD = "standard"     # ~15분
    DEEP = "deep"             # ~30분
    INTENSIVE = "intensive"   # ~1시간
    FULL = "full"             # ~2시간+
    EXTREME = "extreme"       # ~4시간+
    MARATHON = "marathon"     # ~8시간+


@dataclass
class SystemInfo:
    """시스템 정보"""
    os_name: str
    cpu_name: str
    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    gpu_available: bool
    gpu_type: str  # 'cuda', 'mps', 'none'
    gpu_name: str
    gpu_memory_gb: float
    has_gibo_files: bool
    gibo_file_count: int
    gpu_error_message: Optional[str] = None


@dataclass
class TrainingConfig:
    """학습 설정"""
    method: str  # 'gpu', 'cpu', 'gibo', 'hybrid'
    positions: int
    epochs: int
    batch_size: int
    learning_rate: float
    search_depth: int
    iterations: int  # for iterative training
    use_parallel: bool
    num_workers: int
    use_gibo: bool
    use_hybrid: bool  # Phase 3: 혼합 학습 사용 여부
    estimated_time_min: int


def get_system_info(gibo_dir: str = "gibo") -> SystemInfo:
    """시스템 정보 수집"""
    import multiprocessing
    import subprocess
    
    # OS 정보
    os_name = f"{platform.system()} {platform.release()}"
    
    # CPU 정보
    cpu_name = platform.processor() or "Unknown CPU"
    cpu_cores = multiprocessing.cpu_count()
    
    # 물리적 코어 vs 논리적 스레드
    try:
        import psutil
        cpu_threads = psutil.cpu_count(logical=True)
        cpu_cores_physical = psutil.cpu_count(logical=False) or cpu_cores
    except ImportError:
        cpu_threads = cpu_cores
        cpu_cores_physical = cpu_cores
    
    # RAM 정보
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        # macOS에서 sysctl 사용
        try:
            if platform.system() == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True
                )
                ram_gb = int(result.stdout.strip()) / (1024 ** 3)
            elif platform.system() == "Linux":
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            ram_kb = int(line.split()[1])
                            ram_gb = ram_kb / (1024 ** 2)
                            break
            else:
                ram_gb = 8.0  # 기본값
        except:
            ram_gb = 8.0  # 기본값
    
    # GPU 정보
    gpu_available = False
    gpu_type = "none"
    gpu_name = "None"
    gpu_memory_gb = 0.0
    gpu_error_message = None
    
    try:
        import torch
        # PyTorch가 성공적으로 import되었는지 확인
        if torch.cuda.is_available():
            gpu_available = True
            gpu_type = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_available = True
            gpu_type = "mps"
            gpu_name = "Apple Silicon (MPS)"
            # MPS doesn't report memory directly, estimate based on system
            try:
                import psutil
                # Apple Silicon shares memory with system
                gpu_memory_gb = psutil.virtual_memory().total / (1024 ** 3) * 0.5
            except:
                gpu_memory_gb = 8.0  # Default estimate
        else:
            # PyTorch는 설치되어 있지만 CUDA/MPS를 사용할 수 없음
            torch_version = torch.__version__
            if "+cpu" in torch_version:
                gpu_error_message = f"PyTorch CPU-only 버전이 설치되어 있습니다 ({torch_version}). GPU를 사용하려면 CUDA 지원 버전을 설치하세요:\n  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
            else:
                gpu_error_message = "PyTorch는 설치되어 있지만 CUDA를 사용할 수 없습니다. CUDA 드라이버가 설치되어 있는지 확인하세요."
    except (ImportError, AttributeError, RuntimeError) as e:
        # PyTorch import 실패 또는 내부 오류 (예: AcceleratorError 등)
        error_type = type(e).__name__
        if isinstance(e, ImportError):
            gpu_error_message = "PyTorch가 설치되어 있지 않습니다. GPU를 사용하려면 PyTorch를 설치하세요:\n  uv sync --extra gpu\n또는 CUDA 지원 버전:\n  uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        else:
            # PyTorch 설치가 손상되었거나 호환성 문제
            gpu_error_message = f"PyTorch 설치에 문제가 있습니다 ({error_type}: {str(e)}).\n  PyTorch를 재설치하세요:\n  uv pip install --force-reinstall torch\n  또는 CUDA 지원 버전:\n  uv pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    
    # 기보 파일 확인
    gibo_files = glob.glob(os.path.join(gibo_dir, "*.gib")) + glob.glob(os.path.join(gibo_dir, "*.GIB"))
    has_gibo_files = len(gibo_files) > 0
    gibo_file_count = len(gibo_files)
    
    return SystemInfo(
        os_name=os_name,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores_physical,
        cpu_threads=cpu_threads,
        ram_gb=ram_gb,
        gpu_available=gpu_available,
        gpu_type=gpu_type,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        has_gibo_files=has_gibo_files,
        gibo_file_count=gibo_file_count,
        gpu_error_message=gpu_error_message
    )


def print_system_info(info: SystemInfo):
    """시스템 정보 출력"""
    print("\n" + "=" * 60)
    print("🖥️  시스템 환경 분석")
    print("=" * 60)
    
    print(f"\n📌 운영체제: {info.os_name}")
    print(f"📌 CPU: {info.cpu_name}")
    print(f"   - 코어: {info.cpu_cores}개 / 스레드: {info.cpu_threads}개")
    print(f"📌 RAM: {info.ram_gb:.1f} GB")
    
    if info.gpu_available:
        print(f"📌 GPU: {info.gpu_name} ({'CUDA' if info.gpu_type == 'cuda' else 'MPS'})")
        print(f"   - VRAM: {info.gpu_memory_gb:.1f} GB")
        print("   ✅ GPU 가속 사용 가능")
    else:
        print("📌 GPU: 사용 불가 (CPU 학습 모드)")
        if info.gpu_error_message:
            print(f"   ⚠️  {info.gpu_error_message}")
    
    if info.has_gibo_files:
        print(f"📌 기보 파일: {info.gibo_file_count}개 발견")
        print("   ✅ 기보 기반 학습 가능")
    else:
        print("📌 기보 파일: 없음 (self-play 학습)")


def estimate_training_time(
    config: Dict,
    info: SystemInfo,
    use_parallel: bool,
    num_workers: int,
    use_gibo: bool,
    gibo_file_count: int = 0,
    use_hybrid: bool = False
) -> int:
    """학습 시간을 동적으로 계산.
    
    Args:
        config: 학습 설정 딕셔너리
        info: 시스템 정보
        use_parallel: 병렬 처리 사용 여부
        num_workers: 워커 수
        use_gibo: 기보 사용 여부
        gibo_file_count: 기보 파일 수
        use_hybrid: 혼합 학습 사용 여부 (Phase 3)
        
    Returns:
        예상 시간 (분)
    """
    positions = config["positions"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    depth = config["depth"]
    iterations = config.get("iterations", 1)
    
    total_time = 0.0
    
    # Phase 3: 혼합 학습 시간 계산
    if use_hybrid and use_gibo and gibo_file_count > 0:
        # 혼합 학습: 각 iteration마다 (기보 → Self-play → Fine-tuning → 평가)
        # 각 단계의 epoch 수를 적절히 분배
        gibo_epochs = max(1, epochs // (iterations * 4))
        selfplay_epochs = max(5, epochs // (iterations * 2))
        fine_tune_epochs = max(1, epochs // (iterations * 4))
        
        # Self-play 게임 수
        positions_per_iteration = positions // iterations
        selfplay_games = max(50, positions_per_iteration // 80)
        
        for iteration in range(iterations):
            # Step 1: 기보 학습
            estimated_gibo_positions = gibo_file_count * 50
            gibo_parse_time = gibo_file_count * GIBO_PARSE_TIME_PER_GAME
            gibo_process_time = estimated_gibo_positions * GIBO_PROCESS_TIME_PER_POSITION
            
            if info.gpu_available:
                gibo_train_time = (estimated_gibo_positions / 1000) * TRAINING_TIME_GPU_PER_1K * gibo_epochs
            else:
                gibo_train_time = (estimated_gibo_positions / 1000) * TRAINING_TIME_CPU_PER_1K * gibo_epochs
            
            batch_efficiency = BATCH_SIZE_EFFICIENCY.get(batch_size, 0.8)
            gibo_train_time *= batch_efficiency
            total_time += gibo_parse_time + gibo_process_time + gibo_train_time
            
            # Step 2: Self-play 학습
            if info.gpu_available:
                pos_gen_time = (selfplay_games * 80) * POSITION_GEN_TIME_GPU
                selfplay_train_time = ((selfplay_games * 80) / 1000) * TRAINING_TIME_GPU_PER_1K * selfplay_epochs
            elif use_parallel and num_workers > 1:
                pos_gen_time = ((selfplay_games * 80) * POSITION_GEN_TIME_CPU_PARALLEL) / num_workers
                selfplay_train_time = ((selfplay_games * 80) / 1000) * TRAINING_TIME_CPU_PER_1K * selfplay_epochs
            else:
                pos_gen_time = (selfplay_games * 80) * POSITION_GEN_TIME_CPU_SINGLE
                selfplay_train_time = ((selfplay_games * 80) / 1000) * TRAINING_TIME_CPU_PER_1K * selfplay_epochs
            
            depth_multiplier = DEPTH_TIME_MULTIPLIER ** (depth - 2)
            pos_gen_time *= depth_multiplier
            selfplay_train_time *= batch_efficiency
            total_time += pos_gen_time + selfplay_train_time
            
            # Step 3: Fine-tuning (기보 재사용)
            if info.gpu_available:
                fine_tune_time = (estimated_gibo_positions / 1000) * TRAINING_TIME_GPU_PER_1K * fine_tune_epochs
            else:
                fine_tune_time = (estimated_gibo_positions / 1000) * TRAINING_TIME_CPU_PER_1K * fine_tune_epochs
            
            fine_tune_time *= batch_efficiency * 0.5  # Fine-tuning은 더 빠름 (낮은 LR)
            total_time += fine_tune_time
            
            # Step 4: 평가 시간 (간단히 추정)
            eval_time = 10.0  # 평가는 상대적으로 빠름
            total_time += eval_time
        
        # 혼합 학습 오버헤드
        total_time *= (ITERATION_OVERHEAD ** iterations)
        
        # 시스템 성능 보정
        if not info.gpu_available:
            if info.cpu_cores < 4:
                total_time *= 1.3
            elif info.cpu_cores >= 8:
                total_time *= 0.9
        
        if info.ram_gb < MIN_RAM_GB:
            total_time *= 1.2
        
        if info.gpu_available:
            if info.gpu_memory_gb >= 16:
                total_time *= 0.85
            elif info.gpu_memory_gb < 4:
                total_time *= 1.15
        
        estimated_minutes = max(1, int(total_time / 60))
        return estimated_minutes
    
    # 기존 학습 시간 계산 (혼합 학습이 아닐 때)
    # 1. 기보 처리 시간 (있는 경우)
    if use_gibo and gibo_file_count > 0:
        # 기보 파싱 시간
        gibo_parse_time = gibo_file_count * GIBO_PARSE_TIME_PER_GAME
        
        # 기보 포지션 처리 시간 (평균 게임당 50포지션 가정)
        estimated_gibo_positions = gibo_file_count * 50
        gibo_process_time = estimated_gibo_positions * GIBO_PROCESS_TIME_PER_POSITION
        
        # 기보 학습 시간 (에포크의 절반 사용)
        gibo_epochs = epochs // 2
        if info.gpu_available:
            gibo_train_time = (estimated_gibo_positions / 1000) * TRAINING_TIME_GPU_PER_1K * gibo_epochs
        else:
            gibo_train_time = (estimated_gibo_positions / 1000) * TRAINING_TIME_CPU_PER_1K * gibo_epochs
        
        # 배치 사이즈 효율 적용
        batch_efficiency = BATCH_SIZE_EFFICIENCY.get(batch_size, 0.8)
        gibo_train_time *= batch_efficiency
        
        total_time += gibo_parse_time + gibo_process_time + gibo_train_time
    
    # 2. 포지션 생성 시간
    if iterations > 1:
        # 반복 학습: 각 반복마다 포지션 생성
        positions_per_iter = positions // iterations
        
        for _ in range(iterations):
            if info.gpu_available:
                # GPU: 빠른 생성
                pos_gen_time = positions_per_iter * POSITION_GEN_TIME_GPU
            elif use_parallel and num_workers > 1:
                # CPU 병렬: 워커 수에 비례하여 빠름
                pos_gen_time = (positions_per_iter * POSITION_GEN_TIME_CPU_PARALLEL) / num_workers
            else:
                # CPU 단일: 느림
                pos_gen_time = positions_per_iter * POSITION_GEN_TIME_CPU_SINGLE
            
            # 깊이에 따른 시간 증가
            depth_multiplier = DEPTH_TIME_MULTIPLIER ** (depth - 2)  # depth 2를 기준
            pos_gen_time *= depth_multiplier
            
            total_time += pos_gen_time
    else:
        # 단일 학습: 한 번만 생성
        if info.gpu_available:
            pos_gen_time = positions * POSITION_GEN_TIME_GPU
        elif use_parallel and num_workers > 1:
            pos_gen_time = (positions * POSITION_GEN_TIME_CPU_PARALLEL) / num_workers
        else:
            pos_gen_time = positions * POSITION_GEN_TIME_CPU_SINGLE
        
        # 깊이에 따른 시간 증가
        depth_multiplier = DEPTH_TIME_MULTIPLIER ** (depth - 2)
        pos_gen_time *= depth_multiplier
        
        total_time += pos_gen_time
    
    # 3. 학습 시간
    if iterations > 1:
        # 반복 학습: 각 반복마다 학습
        positions_per_iter = positions // iterations
        epochs_per_iter = max(10, epochs // iterations)
        
        for _ in range(iterations):
            if info.gpu_available:
                train_time = (positions_per_iter / 1000) * TRAINING_TIME_GPU_PER_1K * epochs_per_iter
            else:
                train_time = (positions_per_iter / 1000) * TRAINING_TIME_CPU_PER_1K * epochs_per_iter
            
            # 배치 사이즈 효율 적용
            batch_efficiency = BATCH_SIZE_EFFICIENCY.get(batch_size, 0.8)
            train_time *= batch_efficiency
            
            total_time += train_time
    else:
        # 단일 학습
        if info.gpu_available:
            train_time = (positions / 1000) * TRAINING_TIME_GPU_PER_1K * epochs
        else:
            train_time = (positions / 1000) * TRAINING_TIME_CPU_PER_1K * epochs
        
        # 배치 사이즈 효율 적용
        batch_efficiency = BATCH_SIZE_EFFICIENCY.get(batch_size, 0.8)
        train_time *= batch_efficiency
        
        total_time += train_time
    
    # 4. 반복 학습 오버헤드
    if iterations > 1:
        total_time *= (ITERATION_OVERHEAD ** iterations)
    
    # 5. 시스템 성능 보정
    # CPU 코어 수에 따른 보정
    if not info.gpu_available:
        if info.cpu_cores < 4:
            total_time *= 1.3  # 저사양 CPU
        elif info.cpu_cores >= 8:
            total_time *= 0.9  # 고사양 CPU
    
    # RAM 부족 시 느려짐
    if info.ram_gb < MIN_RAM_GB:
        total_time *= 1.2
    
    # GPU 메모리에 따른 보정
    if info.gpu_available:
        if info.gpu_memory_gb >= 16:
            total_time *= 0.85  # 대용량 GPU는 더 빠름
        elif info.gpu_memory_gb < 4:
            total_time *= 1.15  # 소용량 GPU는 더 느림
    
    # 초를 분으로 변환 (최소 1분)
    estimated_minutes = max(1, int(total_time / 60))
    
    return estimated_minutes


def get_training_config(info: SystemInfo, training_time: TrainingTime, use_gibo: bool = True, method: Optional[str] = None) -> TrainingConfig:
    """시스템 환경과 학습 시간에 따른 최적 설정 계산
    
    Args:
        info: 시스템 정보
        training_time: 학습 시간 옵션
        use_gibo: 기보 사용 여부
        method: 직접 지정할 학습 방법 ('gpu', 'cpu', 'gibo', 'hybrid'). None이면 자동 감지
    """
    
    # 기본 설정값 (시간별) - estimated_min은 동적 계산으로 대체됨
    time_configs = {
        TrainingTime.QUICK: {
            "positions": 2000,
            "epochs": 15,
            "batch_size": 128,
            "lr": 0.001,
            "depth": 2,
            "iterations": 1
        },
        TrainingTime.STANDARD: {
            "positions": 10000,  # 2배 증가 (반복 학습 고려)
            "epochs": 50,        # 증가
            "batch_size": 256,
            "lr": 0.001,         # 약간 증가
            "depth": 3,          # 깊이 증가 (더 나은 평가)
            "iterations": 1      # 단일 학습으로 변경 (반복 학습은 데이터가 부족할 때 오히려 해로움)
        },
        TrainingTime.DEEP: {
            "positions": 10000,
            "epochs": 50,
            "batch_size": 256,
            "lr": 0.0005,
            "depth": 3,
            "iterations": 3
        },
        TrainingTime.INTENSIVE: {
            "positions": 20000,
            "epochs": 80,
            "batch_size": 512,
            "lr": 0.0003,
            "depth": 3,
            "iterations": 5
        },
        TrainingTime.FULL: {
            "positions": 150000,  # 3배 증가
            "epochs": 200,        # 2배 증가
            "batch_size": 512,
            "lr": 0.0002,
            "depth": 5,           # 깊이 증가
            "iterations": 15      # 반복 증가
        },
        TrainingTime.EXTREME: {
            "positions": 300000,  # 6배 증가
            "epochs": 300,        # 3배 증가
            "batch_size": 512,
            "lr": 0.00015,
            "depth": 6,           # 더 깊은 탐색
            "iterations": 20      # 더 많은 반복
        },
        TrainingTime.MARATHON: {
            "positions": 500000,  # 10배 증가
            "epochs": 500,        # 5배 증가
            "batch_size": 512,
            "lr": 0.0001,
            "depth": 7,           # 매우 깊은 탐색
            "iterations": 30      # 매우 많은 반복
        }
    }
    
    config = time_configs[training_time]
    
    # method가 직접 지정되었는지 확인
    if method is not None:
        # 직접 지정된 method 사용
        specified_method = method.lower()
        if specified_method not in ['gpu', 'cpu', 'gibo', 'hybrid']:
            print(f"⚠️ 알 수 없는 method: {method}. 자동 감지로 전환합니다.")
            method = None
        else:
            # method 유효성 검사
            if specified_method == 'gpu' and not info.gpu_available:
                print("⚠️ GPU를 사용할 수 없습니다. CPU 모드로 전환합니다.")
                specified_method = 'cpu'
            elif specified_method == 'hybrid' and not (use_gibo and info.has_gibo_files and info.gibo_file_count >= MIN_GIBO_FILES_FOR_TRAINING):
                print("⚠️ 혼합 학습을 위해서는 기보 파일이 필요합니다. GPU 모드로 전환합니다.")
                specified_method = 'gpu' if info.gpu_available else 'cpu'
            elif specified_method == 'gibo' and not (use_gibo and info.has_gibo_files and info.gibo_file_count >= MIN_GIBO_FILES_FOR_TRAINING):
                print("⚠️ 기보 학습을 위해서는 기보 파일이 필요합니다. CPU 모드로 전환합니다.")
                specified_method = 'cpu'
            
            method = specified_method
    
    # method가 지정되지 않았으면 자동 감지
    if method is None:
        # GPU 가용 시 배치 사이즈 및 포지션 수 증가
        if info.gpu_available:
            method = "gpu"
            config = _adjust_config_for_gpu(config, info, training_time)
        else:
            method = "cpu"
            config = _adjust_config_for_cpu(config, info)
        
        # 기보 파일 사용 여부
        should_use_gibo = use_gibo and info.has_gibo_files and info.gibo_file_count >= MIN_GIBO_FILES_FOR_TRAINING
        
        # Phase 3: 혼합 학습 옵션 (기보 파일이 있고, GPU가 있으면 권장)
        # 혼합 학습은 STANDARD 이상의 시간에서만 사용 가능
        use_hybrid = False
        if should_use_gibo and training_time in [
            TrainingTime.STANDARD, TrainingTime.DEEP, TrainingTime.INTENSIVE,
            TrainingTime.FULL, TrainingTime.EXTREME, TrainingTime.MARATHON
        ]:
            # GPU가 있으면 혼합 학습 권장, 없어도 가능하지만 느림
            use_hybrid = True
            method = "hybrid"
        elif should_use_gibo:
            method = "gibo" if not info.gpu_available else "gpu_gibo"
    else:
        # method가 직접 지정된 경우
        should_use_gibo = use_gibo and info.has_gibo_files and info.gibo_file_count >= MIN_GIBO_FILES_FOR_TRAINING
        
        # method에 따라 설정 조정
        if method == 'gpu' or method == 'hybrid':
            config = _adjust_config_for_gpu(config, info, training_time)
        else:
            config = _adjust_config_for_cpu(config, info)
        
        # use_hybrid 설정
        use_hybrid = (method == 'hybrid')
    
    # 병렬 처리 설정
    use_parallel = info.cpu_cores >= MIN_CPU_CORES_FOR_PARALLEL
    num_workers = max(1, min(info.cpu_cores - 1, MAX_WORKERS))
    
    # RAM이 적으면 설정 조정
    if info.ram_gb < MIN_RAM_GB:
        config["positions"] = int(config["positions"] * RAM_POSITION_REDUCTION)
        config["batch_size"] = min(config["batch_size"], 128)
        num_workers = min(num_workers, MIN_WORKERS_LOW_RAM)
    
    # 동적 시간 계산
    estimated_time = estimate_training_time(
        config=config,
        info=info,
        use_parallel=use_parallel,
        num_workers=num_workers,
        use_gibo=should_use_gibo,
        gibo_file_count=info.gibo_file_count if should_use_gibo else 0,
        use_hybrid=use_hybrid
    )
    
    return TrainingConfig(
        method=method,
        positions=int(config["positions"]),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=config["lr"],
        search_depth=config["depth"],
        iterations=config["iterations"],
        use_parallel=use_parallel,
        num_workers=num_workers,
        use_gibo=should_use_gibo,
        use_hybrid=use_hybrid,
        estimated_time_min=estimated_time
    )


def _adjust_config_for_gpu(
    config: Dict, info: SystemInfo, training_time: TrainingTime
) -> Dict:
    """GPU 환경에 맞게 설정 조정.
    
    Args:
        config: 기본 설정 딕셔너리
        info: 시스템 정보
        training_time: 학습 시간 옵션
        
    Returns:
        조정된 설정 딕셔너리
    """
    is_intensive_mode = training_time in [
        TrainingTime.FULL, 
        TrainingTime.EXTREME, 
        TrainingTime.MARATHON
    ]
    
    # GPU 메모리에 따라 배치 사이즈 조정
    if info.gpu_memory_gb >= GPU_MEMORY_HIGH:
        config["batch_size"] = min(config["batch_size"] * 2, 1024)
        # FULL 이상 모드에서는 포지션 수를 더 많이 증가
        if is_intensive_mode:
            config["positions"] = int(config["positions"] * GPU_POSITION_SCALE_HIGH_MEMORY_FULL)
        else:
            config["positions"] = int(config["positions"] * GPU_POSITION_SCALE_HIGH_MEMORY_NORMAL)
    elif info.gpu_memory_gb >= GPU_MEMORY_MEDIUM:
        config["batch_size"] = min(config["batch_size"] * 1.5, 512)
        if is_intensive_mode:
            config["positions"] = int(config["positions"] * GPU_POSITION_SCALE_MEDIUM_MEMORY)
    
    # estimated_min은 나중에 estimate_training_time 함수에서 계산되므로
    # 여기서는 조정하지 않음 (GPU는 자동으로 시간 추정에 반영됨)
    
    return config


def _adjust_config_for_cpu(config: Dict, info: SystemInfo) -> Dict:
    """CPU 환경에 맞게 설정 조정.
    
    Args:
        config: 기본 설정 딕셔너리
        info: 시스템 정보
        
    Returns:
        조정된 설정 딕셔너리
    """
    if info.cpu_cores >= MIN_CPU_CORES_FOR_PARALLEL:
        config["batch_size"] = min(config["batch_size"], 128)
    else:
        config["batch_size"] = min(config["batch_size"], 64)
        config["positions"] = int(config["positions"] * CPU_POSITION_REDUCTION_LOW_CORES)
    
    return config


def get_unique_output_path(base_path: str) -> str:
    """중복되지 않는 출력 파일 경로 생성.
    
    Args:
        base_path: 기본 파일 경로 (예: "models/nnue_smart_model.json")
        
    Returns:
        중복되지 않는 파일 경로 (예: "models/nnue_smart_model.json" 또는 
        "models/nnue_smart_model_1.json")
    """
    if not os.path.exists(base_path):
        return base_path
    
    # 파일 경로 분리
    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)
    
    # 번호를 추가하여 중복되지 않는 파일명 찾기
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_path = os.path.join(directory, new_filename)
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def print_training_config(config: TrainingConfig):
    """학습 설정 출력"""
    print("\n" + "=" * 60)
    print("⚙️  학습 설정")
    print("=" * 60)
    
    method_names = {
        "gpu": "GPU 가속 학습",
        "cpu": "CPU 학습",
        "gibo": "기보 기반 학습 (CPU)",
        "gpu_gibo": "기보 기반 학습 (GPU)",
        "hybrid": "혼합 학습 (기보 + Self-play)"  # Phase 3
    }
    
    print(f"\n📋 학습 방식: {method_names.get(config.method, config.method)}")
    print(f"📋 학습 포지션 수: {config.positions:,}개")
    print(f"📋 에포크 수: {config.epochs}회")
    print(f"📋 배치 사이즈: {config.batch_size}")
    print(f"📋 학습률: {config.learning_rate}")
    print(f"📋 탐색 깊이: {config.search_depth}")
    
    if config.iterations > 1:
        print(f"📋 반복 학습: {config.iterations}회")
    
    if config.use_parallel:
        print(f"📋 병렬 처리: {config.num_workers}개 워커")
    
    if config.use_gibo:
        print("📋 기보 데이터 활용: ✅")
    
    if config.use_hybrid:
        print("📋 혼합 학습 모드: ✅ (기보 → Self-play → Fine-tuning)")
        print(f"   - 각 iteration마다: 기보 학습 → Self-play 학습 → Fine-tuning → 평가")
    
    print(f"\n⏱️  예상 학습 시간: 약 {config.estimated_time_min}분")


def interactive_menu(info: SystemInfo) -> Tuple[TrainingTime, bool, Optional[str], Optional[str]]:
    """대화형 메뉴
    
    Returns:
        (training_time, use_gibo, load_model, method)
    """
    print("\n" + "=" * 60)
    print("🎯 학습 시간 선택")
    print("=" * 60)
    
    options = [
        (TrainingTime.QUICK, "⚡ 빠른 학습", "~5분", "빠른 테스트용, 기본적인 학습"),
        (TrainingTime.STANDARD, "📘 표준 학습", "~15분", "일반적인 사용에 적합"),
        (TrainingTime.DEEP, "📗 깊은 학습", "~30분", "더 나은 성능, 권장"),
        (TrainingTime.INTENSIVE, "📕 집중 학습", "~1시간", "높은 성능 목표"),
        (TrainingTime.FULL, "🏆 완전 학습", "~3시간", "최고 성능, 강화된 설정"),
        (TrainingTime.EXTREME, "🔥 극한 학습", "~4시간", "최강 성능, 매우 긴 학습"),
        (TrainingTime.MARATHON, "🏃 마라톤 학습", "~8시간", "최종 보스, 하루 종일 학습"),
    ]
    
    print("\n학습 시간을 선택하세요:\n")
    for i, (_, name, time_est, desc) in enumerate(options, 1):
        print(f"  {i}. {name} ({time_est})")
        print(f"     └─ {desc}")
    
    print("\n  0. 종료")
    
    while True:
        try:
            choice = input("\n선택 (1-7, 0=종료): ").strip()
            if choice == "0":
                return None, False, None
            
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                selected_time = options[idx][0]
                break
            print(f"❌ 1-{len(options)} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("❌ 숫자를 입력하세요.")
    
    # 학습 방법 선택
    print("\n" + "=" * 60)
    print("🔧 학습 방법 선택 (선택 사항)")
    print("=" * 60)
    print("\n학습 방법을 직접 선택하시겠습니까? (자동 감지도 가능)")
    print("  1. 자동 감지 (권장)")
    print("  2. GPU 학습")
    print("  3. CPU 학습")
    if info.has_gibo_files and info.gibo_file_count >= MIN_GIBO_FILES_FOR_TRAINING:
        print("  4. 기보 학습")
        if info.gpu_available:
            print("  5. 혼합 학습 (기보 + Self-play)")
    
    method = None
    while True:
        try:
            method_choice = input("\n선택 (1-5, Enter=자동): ").strip()
            if not method_choice or method_choice == '1':
                method = None  # 자동 감지
                break
            elif method_choice == '2':
                if info.gpu_available:
                    method = 'gpu'
                    break
                else:
                    print("❌ GPU를 사용할 수 없습니다. 다른 방법을 선택하세요.")
            elif method_choice == '3':
                method = 'cpu'
                break
            elif method_choice == '4':
                if info.has_gibo_files and info.gibo_file_count >= MIN_GIBO_FILES_FOR_TRAINING:
                    method = 'gibo'
                    break
                else:
                    print("❌ 기보 파일이 충분하지 않습니다. 다른 방법을 선택하세요.")
            elif method_choice == '5':
                if info.has_gibo_files and info.gibo_file_count >= MIN_GIBO_FILES_FOR_TRAINING:
                    method = 'hybrid'
                    break
                else:
                    print("❌ 혼합 학습을 위해서는 기보 파일이 필요합니다. 다른 방법을 선택하세요.")
            else:
                print("❌ 1-5 사이의 숫자를 입력하세요.")
        except (ValueError, KeyboardInterrupt):
            method = None
            break
    
    # 기보 사용 여부
    use_gibo = False
    if info.has_gibo_files:
        print(f"\n기보 파일 {info.gibo_file_count}개가 발견되었습니다.")
        gibo_choice = input("기보 데이터를 학습에 활용하시겠습니까? (Y/n): ").strip().lower()
        use_gibo = gibo_choice != 'n'
    
    # 기존 모델 로드 여부
    load_model = None
    existing_models = sorted(glob.glob("models/*.json"))  # 정렬하여 일관된 순서 보장
    if existing_models:
        print(f"\n기존 모델 {len(existing_models)}개가 발견되었습니다:")
        for i, model in enumerate(existing_models, 1):
            print(f"  {i}. {os.path.basename(model)}")
        
        load_choice = input("\n기존 모델에서 계속 학습하시겠습니까? (숫자 입력 또는 n): ").strip().lower()
        if load_choice != 'n' and load_choice.isdigit():
            idx = int(load_choice) - 1
            if 0 <= idx < len(existing_models):
                load_model = existing_models[idx]
    
    return selected_time, use_gibo, load_model, method


def train_with_gpu(config: TrainingConfig, load_model: Optional[str] = None, gibo_dir: str = "gibo"):
    """GPU 가속 학습 실행"""
    try:
        import torch
        from janggi.nnue_torch import NNUETorch, FeatureExtractor, GPUTrainer, get_device
        from scripts.train_nnue_gpu import get_optimal_batch_size
    except ImportError as e:
        print(f"❌ PyTorch가 필요합니다: {e}")
        print("설치: pip install torch")
        return None
    
    device = get_device()
    print(f"\n🚀 GPU 학습 시작 (Device: {device})")
    
    # GPU 메모리 기반 최적 배치 크기 계산
    eval_batch_size = get_optimal_batch_size(device=device)
    if device.type == 'cuda':
        print(f"📊 GPU 메모리 기반 최적 평가 배치 크기: {eval_batch_size}")
    
    # 모델 초기화 또는 로드
    if load_model:
        print(f"📂 모델 로드: {load_model}")
        nnue = NNUETorch.from_file(load_model, device=device)
    else:
        print("🆕 새 모델 초기화")
        nnue = NNUETorch(device=device)
    
    # Phase 3: 혼합 학습 모드
    if config.use_hybrid:
        try:
            from scripts.train_nnue_hybrid import hybrid_training
            
            print("\n🔄 혼합 학습 모드 시작 (기보 → Self-play → Fine-tuning)")
            
            # 혼합 학습 파라미터 계산
            # iterations는 config.iterations 사용
            # 각 iteration의 epoch 수를 적절히 분배
            gibo_epochs = max(1, config.epochs // (config.iterations * 4))  # 전체의 1/4
            selfplay_epochs = max(5, config.epochs // (config.iterations * 2))  # 전체의 1/2
            fine_tune_epochs = max(1, config.epochs // (config.iterations * 4))  # 전체의 1/4
            
            # Self-play 게임 수 계산 (전체 positions를 iterations로 나눔)
            positions_per_iteration = config.positions // config.iterations
            selfplay_games = max(50, positions_per_iteration // 80)  # 게임당 평균 80개 포지션 가정
            
            print(f"   설정:")
            print(f"   - 반복 횟수: {config.iterations}회")
            print(f"   - 기보 학습: {gibo_epochs} epochs/iteration")
            print(f"   - Self-play 학습: {selfplay_epochs} epochs/iteration (~{selfplay_games} games)")
            print(f"   - Fine-tuning: {fine_tune_epochs} epochs/iteration")
            
            # 혼합 학습 실행
            nnue = hybrid_training(
                gibo_dir=gibo_dir,
                nnue=nnue,
                iterations=config.iterations,
                gibo_epochs=gibo_epochs,
                selfplay_epochs=selfplay_epochs,
                fine_tune_epochs=fine_tune_epochs,
                selfplay_games=selfplay_games,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                positions_per_game=50,
                search_depth=config.search_depth,
                output_dir="models",
                use_parallel=config.use_parallel,
                num_workers=config.num_workers if config.use_parallel else None,
                eval_batch_size=eval_batch_size,
                eval_num_workers=config.num_workers if config.use_parallel else None
            )
            
            history = {"train_loss": [], "val_loss": []}  # 혼합 학습은 별도 출력
            
        except ImportError as e:
            print(f"⚠️ 혼합 학습 모듈을 불러올 수 없습니다: {e}")
            print("   일반 학습 모드로 전환합니다.")
            config.use_hybrid = False
        except Exception as e:
            print(f"⚠️ 혼합 학습 중 오류 발생: {e}")
            print("   일반 학습 모드로 전환합니다.")
            config.use_hybrid = False
    
    # 기보 기반 학습 (혼합 학습이 아닐 때만)
    if config.use_gibo and not config.use_hybrid:
        from scripts.train_nnue_gibo import GibParser, GiboDataGenerator, train_with_gradient_clipping
        
        print("\n📚 기보 파일 파싱 중...")
        parser = GibParser()
        games = parser.parse_directory(gibo_dir)
        
        if games:
            print(f"✅ {len(games)}개 게임 로드 완료")
            
            generator = GiboDataGenerator()
            
            # 병렬 처리 사용 (CPU 코어가 4개 이상이고 게임이 많으면 자동으로 병렬 처리)
            import multiprocessing as mp
            cpu_count = mp.cpu_count()
            use_parallel = config.use_parallel and len(games) > 100
            
            if use_parallel:
                print(f"🚀 병렬 처리 모드 사용 ({config.num_workers}개 워커)")
                features, targets = generator.generate_from_games_parallel(
                    games,
                    positions_per_game=50,
                    num_workers=config.num_workers,
                    progress_callback=lambda d, t: print(f"\r처리 중: {d}/{t}", end="", flush=True)
                )
            else:
                print("🔄 순차 처리 모드 사용")
                features, targets = generator.generate_from_games(
                    games,
                    positions_per_game=50,
                    progress_callback=lambda d, t: print(f"\r처리 중: {d}/{t}", end="", flush=True)
                )
            print()
            
            print(f"\n🎓 기보 기반 학습 시작 ({len(features)}개 포지션)...")
            train_with_gradient_clipping(
                nnue, features, targets,
                epochs=config.epochs // 2,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate
            )
    
    # 반복 학습 사용 여부 결정 (혼합 학습이 아닐 때만)
    if config.iterations > 1 and not config.use_hybrid:
        # 반복 학습 모드
        from scripts.train_nnue_gpu import train_iterative
        
        print(f"\n🔄 반복 학습 모드 ({config.iterations}회 반복)")
        # 게임당 약 50-100개 포지션이 생성되므로, 게임 수 계산
        positions_per_iteration = config.positions // config.iterations
        games_per_iteration = max(50, positions_per_iteration // 80)  # 게임당 평균 80개 포지션 가정
        epochs_per_iteration = max(10, config.epochs // config.iterations)
        
        print(f"   각 반복마다: ~{games_per_iteration}게임 (~{positions_per_iteration:,}개 포지션), {epochs_per_iteration}회 에포크")
        
        # 반복 학습 실행 (병렬 self-play + GPU 배치 평가 사용)
        train_iterative(
            nnue,
            num_iterations=config.iterations,
            games_per_iteration=games_per_iteration,
            epochs_per_iteration=epochs_per_iteration,
            batch_size=config.batch_size,
            output_dir="models",
            search_depth=config.search_depth,
            use_parallel=True,  # 병렬 self-play 사용
            num_workers=config.num_workers if config.use_parallel else None,
            eval_batch_size=eval_batch_size,  # GPU 배치 평가 크기
            eval_num_workers=config.num_workers if config.use_parallel else None,
            base_learning_rate=config.learning_rate  # config의 learning_rate 사용
        )
        
        history = {"train_loss": [], "val_loss": []}  # 반복 학습은 별도 출력
    else:
        # 단일 학습 모드
        from scripts.train_nnue_gpu import DataGenerator
        
        generator = DataGenerator()
        
        def progress(done, total, speed, eta):
            print(f"\r📊 포지션 생성: {done:,}/{total:,} ({speed:.1f}/s, ETA: {eta:.0f}s)", end="", flush=True)
        
        print(f"\n🎲 Self-play 포지션 생성 중 ({config.positions:,}개)...")
        
        if config.use_parallel:
            features, targets = generator.generate_positions_parallel(
                num_positions=config.positions,
                num_workers=config.num_workers,
                progress_callback=progress
            )
        else:
            features, targets = generator.generate_positions_fast(
                num_positions=config.positions,
                progress_callback=progress
            )
        
        print()  # 줄바꿈
        
        print(f"\n🎓 학습 시작 ({len(features):,}개 포지션, {config.epochs}회 에포크)...")
        trainer = GPUTrainer(nnue)
        
        history = trainer.train(
            features, targets,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            early_stopping_patience=10
        )
    
    # 모델 저장
    os.makedirs("models", exist_ok=True)
    if config.use_hybrid:
        base_path = "models/nnue_smart_hybrid_model.json"
    else:
        base_path = "models/nnue_smart_model.json"
    output_path = get_unique_output_path(base_path)
    nnue.save(output_path)
    print(f"\n💾 모델 저장: {output_path}")
    
    return nnue, history, output_path


def train_with_cpu(config: TrainingConfig, load_model: Optional[str] = None, gibo_dir: str = "gibo"):
    """CPU 학습 실행"""
    from janggi.nnue import NNUE
    from scripts.train_nnue import TrainingDataGenerator, NNUETrainer, IterativeTrainer
    
    print("\n🖥️  CPU 학습 시작")
    
    # 모델 초기화 또는 로드
    if load_model:
        print(f"📂 모델 로드: {load_model}")
        nnue = NNUE.from_file(load_model)
    else:
        print("🆕 새 모델 초기화")
        nnue = NNUE()
    
    # 반복 학습 사용
    if config.iterations > 1:
        print(f"\n🔄 반복 학습 모드 ({config.iterations}회)")
        trainer = IterativeTrainer(nnue)
        trainer.run_iterations(
            num_iterations=config.iterations,
            games_per_iteration=config.positions // (50 * config.iterations),
            search_depth=config.search_depth,
            epochs_per_iteration=config.epochs // config.iterations,
            output_dir="models",
            base_name="nnue_smart_iter"
        )
    else:
        # 단일 학습
        print(f"\n🎲 포지션 생성 중 ({config.positions}개)...")
        generator = TrainingDataGenerator(search_depth=config.search_depth)
        
        boards, targets = generator.generate_diverse_positions(
            num_positions=config.positions,
            search_depth=config.search_depth,
            progress_callback=lambda c, t: print(f"\r📊 포지션: {c}/{t}", end="", flush=True)
        )
        print()
        
        print(f"\n🎓 학습 시작 ({len(boards)}개 포지션)...")
        trainer = NNUETrainer(nnue)
        history = trainer.train(
            boards, targets,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size
        )
    
    # 모델 저장
    os.makedirs("models", exist_ok=True)
    base_path = "models/nnue_smart_model.json"
    output_path = get_unique_output_path(base_path)
    nnue.save(output_path)
    print(f"\n💾 모델 저장: {output_path}")
    
    return nnue, output_path


def evaluate_model(model, num_games: int = 5):
    """모델 평가 (GPU 배치 평가 최적화 사용)"""
    print(f"\n📈 모델 평가 중 ({num_games}게임)...")
    
    try:
        from scripts.train_nnue_gpu import evaluate_model as gpu_eval, get_optimal_batch_size
        from janggi.nnue_torch import get_device
        
        # GPU 배치 평가 최적화 사용
        device = get_device()
        eval_batch_size = get_optimal_batch_size(device=device)
        
        # GPU 사용 가능하면 병렬 평가 + 배치 평가 사용
        use_gpu = device.type == 'cuda'
        if use_gpu:
            # 작은 게임 수에 대해 워커 수 제한 (멀티프로세싱 오버헤드 감소)
            # 작은 게임 수에서는 워커 초기화 + 모델 로드 오버헤드가 병렬화 이점보다 큼
            if num_games <= 5:
                num_workers = 1
            elif num_games <= 10:
                num_workers = min(2, num_games)
            else:
                num_workers = max(1, min(mp.cpu_count() - 1, num_games))
        else:
            num_workers = None
        
        win_rate = gpu_eval(
            model, 
            num_games=num_games,
            search_depth=3,
            num_workers=num_workers,
            use_gpu=use_gpu,
            eval_batch_size=eval_batch_size
        )
    except Exception as e:
        # CPU 모델 평가 (fallback)
        print(f"⚠️ GPU 평가 실패, CPU 평가로 전환: {e}")
        from janggi.board import Board, Side
        from janggi.engine import Engine
        
        wins = 0
        engine = Engine(depth=2, use_nnue=True)
        engine.nnue = model
        
        for _ in range(num_games):
            board = Board()
            for _ in range(100):
                if board.is_checkmate() or board.is_stalemate():
                    break
                move = engine.search(board)
                if move is None:
                    break
                board.make_move(move)
            
            if board.is_checkmate() and board.side_to_move == Side.HAN:
                wins += 1
        
        win_rate = wins / num_games
    
    print(f"✅ SimpleEvaluator 대비 승률: {win_rate:.1%}")
    return win_rate


def main():
    parser = argparse.ArgumentParser(
        description='Smart NNUE Training - 자동 환경 감지 및 최적화 학습',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
학습 시간 옵션:
  quick      ⚡ 빠른 학습 (~5분)    - 빠른 테스트용
  standard   📘 표준 학습 (~15분)   - 일반적인 사용
  deep       📗 깊은 학습 (~30분)   - 권장
  intensive  📕 집중 학습 (~1시간)  - 높은 성능
  full       🏆 완전 학습 (~3시간)  - 최고 성능, 강화된 설정
  extreme    🔥 극한 학습 (~4시간)  - 최강 성능
  marathon   🏃 마라톤 학습 (~8시간) - 최종 보스

예시:
  python smart_train.py                      # 대화형 모드
  python smart_train.py --time standard      # 표준 학습
  python smart_train.py --time deep --no-gibo  # 기보 없이 깊은 학습
        """
    )
    
    parser.add_argument('--time', type=str, 
                        choices=['quick', 'standard', 'deep', 'intensive', 'full', 'extreme', 'marathon'],
                        default=None,
                        help='학습 시간 선택')
    parser.add_argument('--load', type=str, default=None,
                        help='기존 모델 파일 로드')
    parser.add_argument('--no-gibo', action='store_true',
                        help='기보 파일 사용하지 않음')
    parser.add_argument('--gibo-dir', type=str, default='gibo',
                        help='기보 파일 디렉토리')
    parser.add_argument('--method', type=str,
                        choices=['gpu', 'cpu', 'gibo', 'hybrid'],
                        default=None,
                        help='학습 방법 직접 지정 (gpu, cpu, gibo, hybrid). 지정하지 않으면 자동 감지')
    parser.add_argument('--output', type=str, default='models/nnue_smart_model.json',
                        help='출력 모델 파일')
    parser.add_argument('--skip-eval', action='store_true',
                        help='학습 후 평가 건너뛰기')
    parser.add_argument('--info-only', action='store_true',
                        help='시스템 정보만 출력하고 종료')
    
    args = parser.parse_args()
    
    # 시스템 정보 수집
    print("\n🔍 시스템 환경 분석 중...")
    info = get_system_info(args.gibo_dir)
    print_system_info(info)
    
    if args.info_only:
        return
    
    # 학습 시간 선택
    if args.time:
        training_time = TrainingTime(args.time)
        use_gibo = not args.no_gibo and info.has_gibo_files
        load_model = args.load
        method = args.method
    else:
        # 대화형 메뉴
        result = interactive_menu(info)
        if result[0] is None:
            print("\n👋 종료합니다.")
            return
        training_time, use_gibo, load_model, method = result
    
    # 학습 설정 계산
    config = get_training_config(info, training_time, use_gibo, method=method)
    print_training_config(config)
    
    # 확인
    if not args.time:  # 대화형 모드에서만 확인
        confirm = input("\n이 설정으로 학습을 시작하시겠습니까? (Y/n): ").strip().lower()
        if confirm == 'n':
            print("👋 취소되었습니다.")
            return
    
    # 학습 시작
    print("\n" + "=" * 60)
    print("🎓 학습 시작")
    print("=" * 60)
    
    start_time = time.time()
    
    output_path = None
    if config.method in ["gpu", "gpu_gibo", "hybrid"]:
        result = train_with_gpu(config, load_model, args.gibo_dir)
        if result:
            model, history, output_path = result
    else:
        model, output_path = train_with_cpu(config, load_model, args.gibo_dir)
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  총 학습 시간: {elapsed/60:.1f}분")
    
    # 모델 평가
    if not args.skip_eval and model:
        try:
            evaluate_model(model)
        except Exception as e:
            print(f"⚠️ 평가 중 오류 발생: {e}")
    
    print("\n" + "=" * 60)
    print("✅ 학습 완료!")
    print("=" * 60)
    
    if output_path:
        print(f"\n모델 저장 위치: {output_path}")
        print("\n사용법:")
        if config.method in ["gpu", "gpu_gibo", "hybrid"]:
            print("  from janggi.nnue_torch import NNUETorch")
            print(f"  nnue = NNUETorch.from_file('{output_path}')")
        else:
            print("  from janggi.nnue import NNUE")
            print(f"  nnue = NNUE.from_file('{output_path}')")


if __name__ == "__main__":
    main()

