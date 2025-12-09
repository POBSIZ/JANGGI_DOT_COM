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
    python smart_train.py --time full        # ~2시간+
    
    # 기존 모델에서 계속 학습
    python smart_train.py --load models/nnue_model.json --time standard
"""

import argparse
import os
import sys
import platform
import time
import glob
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TrainingTime(Enum):
    """학습 시간 옵션"""
    QUICK = "quick"           # ~5분
    STANDARD = "standard"     # ~15분
    DEEP = "deep"             # ~30분
    INTENSIVE = "intensive"   # ~1시간
    FULL = "full"             # ~2시간+


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
    method: str  # 'gpu', 'cpu', 'gibo'
    positions: int
    epochs: int
    batch_size: int
    learning_rate: float
    search_depth: int
    iterations: int  # for iterative training
    use_parallel: bool
    num_workers: int
    use_gibo: bool
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


def get_training_config(info: SystemInfo, training_time: TrainingTime, use_gibo: bool = True) -> TrainingConfig:
    """시스템 환경과 학습 시간에 따른 최적 설정 계산"""
    
    # 기본 설정값 (시간별)
    time_configs = {
        TrainingTime.QUICK: {
            "positions": 2000,
            "epochs": 15,
            "batch_size": 128,
            "lr": 0.001,
            "depth": 2,
            "iterations": 1,
            "estimated_min": 5
        },
        TrainingTime.STANDARD: {
            "positions": 5000,
            "epochs": 30,
            "batch_size": 256,
            "lr": 0.0008,
            "depth": 2,
            "iterations": 2,
            "estimated_min": 15
        },
        TrainingTime.DEEP: {
            "positions": 10000,
            "epochs": 50,
            "batch_size": 256,
            "lr": 0.0005,
            "depth": 3,
            "iterations": 3,
            "estimated_min": 30
        },
        TrainingTime.INTENSIVE: {
            "positions": 20000,
            "epochs": 80,
            "batch_size": 512,
            "lr": 0.0003,
            "depth": 3,
            "iterations": 5,
            "estimated_min": 60
        },
        TrainingTime.FULL: {
            "positions": 50000,
            "epochs": 100,
            "batch_size": 512,
            "lr": 0.0002,
            "depth": 4,
            "iterations": 8,
            "estimated_min": 120
        }
    }
    
    config = time_configs[training_time]
    
    # GPU 가용 시 배치 사이즈 및 포지션 수 증가
    if info.gpu_available:
        method = "gpu"
        # GPU 메모리에 따라 배치 사이즈 조정
        if info.gpu_memory_gb >= 8:
            config["batch_size"] = min(config["batch_size"] * 2, 1024)
            config["positions"] = int(config["positions"] * 1.5)
        elif info.gpu_memory_gb >= 4:
            config["batch_size"] = min(config["batch_size"] * 1.5, 512)
        
        # GPU 학습은 더 빠르므로 시간 예상 조정
        config["estimated_min"] = int(config["estimated_min"] * 0.5)
    else:
        method = "cpu"
        # CPU 코어에 따라 병렬화 설정
        if info.cpu_cores >= 4:
            config["batch_size"] = min(config["batch_size"], 128)
        else:
            config["batch_size"] = min(config["batch_size"], 64)
            config["positions"] = int(config["positions"] * 0.7)
    
    # 기보 파일 사용 여부
    should_use_gibo = use_gibo and info.has_gibo_files and info.gibo_file_count >= 5
    if should_use_gibo:
        method = "gibo" if not info.gpu_available else "gpu_gibo"
    
    # 병렬 처리 설정
    use_parallel = info.cpu_cores >= 4
    num_workers = max(1, min(info.cpu_cores - 1, 8))
    
    # RAM이 적으면 설정 조정
    if info.ram_gb < 8:
        config["positions"] = int(config["positions"] * 0.5)
        config["batch_size"] = min(config["batch_size"], 128)
        num_workers = min(num_workers, 2)
    
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
        estimated_time_min=config["estimated_min"]
    )


def print_training_config(config: TrainingConfig):
    """학습 설정 출력"""
    print("\n" + "=" * 60)
    print("⚙️  학습 설정")
    print("=" * 60)
    
    method_names = {
        "gpu": "GPU 가속 학습",
        "cpu": "CPU 학습",
        "gibo": "기보 기반 학습 (CPU)",
        "gpu_gibo": "기보 기반 학습 (GPU)"
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
    
    print(f"\n⏱️  예상 학습 시간: 약 {config.estimated_time_min}분")


def interactive_menu(info: SystemInfo) -> Tuple[TrainingTime, bool, Optional[str]]:
    """대화형 메뉴"""
    print("\n" + "=" * 60)
    print("🎯 학습 시간 선택")
    print("=" * 60)
    
    options = [
        (TrainingTime.QUICK, "⚡ 빠른 학습", "~5분", "빠른 테스트용, 기본적인 학습"),
        (TrainingTime.STANDARD, "📘 표준 학습", "~15분", "일반적인 사용에 적합"),
        (TrainingTime.DEEP, "📗 깊은 학습", "~30분", "더 나은 성능, 권장"),
        (TrainingTime.INTENSIVE, "📕 집중 학습", "~1시간", "높은 성능 목표"),
        (TrainingTime.FULL, "🏆 완전 학습", "~2시간+", "최고 성능, 시간 여유 있을 때"),
    ]
    
    print("\n학습 시간을 선택하세요:\n")
    for i, (_, name, time_est, desc) in enumerate(options, 1):
        print(f"  {i}. {name} ({time_est})")
        print(f"     └─ {desc}")
    
    print("\n  0. 종료")
    
    while True:
        try:
            choice = input("\n선택 (1-5, 0=종료): ").strip()
            if choice == "0":
                return None, False, None
            
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                selected_time = options[idx][0]
                break
            print("❌ 1-5 사이의 숫자를 입력하세요.")
        except ValueError:
            print("❌ 숫자를 입력하세요.")
    
    # 기보 사용 여부
    use_gibo = False
    if info.has_gibo_files:
        print(f"\n기보 파일 {info.gibo_file_count}개가 발견되었습니다.")
        gibo_choice = input("기보 데이터를 학습에 활용하시겠습니까? (Y/n): ").strip().lower()
        use_gibo = gibo_choice != 'n'
    
    # 기존 모델 로드 여부
    load_model = None
    existing_models = glob.glob("models/*.json")
    if existing_models:
        print(f"\n기존 모델 {len(existing_models)}개가 발견되었습니다:")
        for i, model in enumerate(existing_models[:5], 1):
            print(f"  {i}. {os.path.basename(model)}")
        
        load_choice = input("\n기존 모델에서 계속 학습하시겠습니까? (숫자 입력 또는 n): ").strip().lower()
        if load_choice != 'n' and load_choice.isdigit():
            idx = int(load_choice) - 1
            if 0 <= idx < len(existing_models):
                load_model = existing_models[idx]
    
    return selected_time, use_gibo, load_model


def train_with_gpu(config: TrainingConfig, load_model: Optional[str] = None, gibo_dir: str = "gibo"):
    """GPU 가속 학습 실행"""
    try:
        import torch
        from janggi.nnue_torch import NNUETorch, FeatureExtractor, GPUTrainer, get_device
    except ImportError as e:
        print(f"❌ PyTorch가 필요합니다: {e}")
        print("설치: pip install torch")
        return None
    
    device = get_device()
    print(f"\n🚀 GPU 학습 시작 (Device: {device})")
    
    # 모델 초기화 또는 로드
    if load_model:
        print(f"📂 모델 로드: {load_model}")
        nnue = NNUETorch.from_file(load_model, device=device)
    else:
        print("🆕 새 모델 초기화")
        nnue = NNUETorch(device=device)
    
    # 기보 기반 학습
    if config.use_gibo:
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
    
    # Self-play 데이터 생성 및 학습
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
    output_path = "models/nnue_smart_model.json"
    os.makedirs("models", exist_ok=True)
    nnue.save(output_path)
    print(f"\n💾 모델 저장: {output_path}")
    
    return nnue, history


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
    output_path = "models/nnue_smart_model.json"
    os.makedirs("models", exist_ok=True)
    nnue.save(output_path)
    print(f"\n💾 모델 저장: {output_path}")
    
    return nnue


def evaluate_model(model, num_games: int = 5):
    """모델 평가"""
    print(f"\n📈 모델 평가 중 ({num_games}게임)...")
    
    try:
        from scripts.train_nnue_gpu import evaluate_model as gpu_eval
        win_rate = gpu_eval(model, num_games=num_games)
    except:
        # CPU 모델 평가
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
  full       🏆 완전 학습 (~2시간+) - 최고 성능

예시:
  python smart_train.py                      # 대화형 모드
  python smart_train.py --time standard      # 표준 학습
  python smart_train.py --time deep --no-gibo  # 기보 없이 깊은 학습
        """
    )
    
    parser.add_argument('--time', type=str, 
                        choices=['quick', 'standard', 'deep', 'intensive', 'full'],
                        default=None,
                        help='학습 시간 선택')
    parser.add_argument('--load', type=str, default=None,
                        help='기존 모델 파일 로드')
    parser.add_argument('--no-gibo', action='store_true',
                        help='기보 파일 사용하지 않음')
    parser.add_argument('--gibo-dir', type=str, default='gibo',
                        help='기보 파일 디렉토리')
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
    else:
        # 대화형 메뉴
        result = interactive_menu(info)
        if result[0] is None:
            print("\n👋 종료합니다.")
            return
        training_time, use_gibo, load_model = result
    
    # 학습 설정 계산
    config = get_training_config(info, training_time, use_gibo)
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
    
    if config.method in ["gpu", "gpu_gibo"]:
        result = train_with_gpu(config, load_model, args.gibo_dir)
        if result:
            model, history = result
    else:
        model = train_with_cpu(config, load_model, args.gibo_dir)
    
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
    print(f"\n모델 저장 위치: models/nnue_smart_model.json")
    print("\n사용법:")
    print("  from janggi.nnue_torch import NNUETorch")
    print("  nnue = NNUETorch.from_file('models/nnue_smart_model.json')")


if __name__ == "__main__":
    main()

