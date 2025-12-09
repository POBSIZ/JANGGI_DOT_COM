#!/usr/bin/env python3
"""예제: models 디렉토리의 모델을 사용하는 방법"""

import os
from janggi.board import Board, Side
from janggi.engine import Engine
from janggi.nnue import NNUE


def get_best_model_path():
    """사용 가능한 최적의 모델 경로를 반환"""
    # 우선순위: 기보 모델 > 최신 반복 모델 > 기본 모델
    model_priority = [
        "models/nnue_gibo_model.json",     # 기보 학습 모델
        "models/nnue_gpu_iter_5.json",     # 최신 반복 모델
        "models/nnue_gpu_model.json",      # GPU 기본 모델
        "models/nnue_model.json",          # CPU 모델
    ]
    
    for model_path in model_priority:
        if os.path.exists(model_path):
            return model_path
    
    return None


def example_1_use_model_with_engine():
    """예제 1: Engine 클래스에서 모델 사용"""
    print("=" * 60)
    print("예제 1: Engine 클래스에서 모델 사용")
    print("=" * 60)
    
    # 모델 경로 설정 (자동 선택)
    model_path = get_best_model_path()
    
    if not model_path:
        print("⚠️  사용 가능한 모델이 없습니다.")
        print("먼저 학습을 실행하세요:")
        print("  python scripts/train_nnue_gpu.py --positions 5000 --epochs 30")
        print("  또는")
        print("  python scripts/train_nnue_gibo.py --gibo-dir gibo --epochs 30")
        return
    
    # 모델을 사용하는 엔진 생성
    engine = Engine(
        depth=3,
        use_nnue=True,
        nnue_model_path=model_path
    )
    
    print(f"✅ 모델 로드 완료: {model_path}")
    
    # 새 게임 시작
    board = Board()
    print("\n초기 보드 상태:")
    print(board)
    
    # AI가 최선의 수 찾기
    print("\n🔍 AI가 최선의 수를 찾는 중...")
    best_move = engine.search(board)
    
    if best_move:
        print(f"\n✅ AI의 최선의 수: {best_move.to_uci()}")
        print(f"   탐색한 노드 수: {engine.nodes_searched}")
        
        # 수를 둬보기
        board.make_move(best_move)
        print("\n수행 후 보드 상태:")
        print(board)
    else:
        print("❌ 유효한 수를 찾을 수 없습니다.")


def example_2_use_nnue_directly():
    """예제 2: NNUE 클래스를 직접 사용하여 위치 평가"""
    print("\n" + "=" * 60)
    print("예제 2: NNUE 클래스를 직접 사용하여 위치 평가")
    print("=" * 60)
    
    model_path = get_best_model_path()
    
    if not model_path:
        print("⚠️  사용 가능한 모델이 없습니다.")
        return
    
    # 모델 로드
    nnue = NNUE.from_file(model_path)
    print(f"✅ 모델 로드 완료: {model_path}")
    
    # 보드 생성
    board = Board()
    
    # 초기 위치 평가
    eval_score = nnue.evaluate(board)
    print(f"\n초기 위치 평가 점수: {eval_score:.4f}")
    print("(양수면 한(紅)에게 유리, 음수면 초(藍)에게 유리)")
    
    # 몇 수를 둔 후 평가
    moves = board.generate_moves()
    if moves:
        test_move = moves[0]
        board.make_move(test_move)
        eval_after = nnue.evaluate(board)
        print(f"\n한 수를 둔 후 평가 점수: {eval_after:.4f}")
        print(f"점수 변화: {eval_after - eval_score:.4f}")


def example_3_compare_models():
    """예제 3: 여러 모델 비교"""
    print("\n" + "=" * 60)
    print("예제 3: 여러 모델 비교")
    print("=" * 60)
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"⚠️  models 디렉토리를 찾을 수 없습니다.")
        return
    
    # 사용 가능한 모델 찾기
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".json")]
    model_files.sort()  # 정렬
    
    if not model_files:
        print("⚠️  사용 가능한 모델이 없습니다.")
        return
    
    print(f"사용 가능한 모델: {len(model_files)}개")
    
    board = Board()
    
    # 각 모델로 평가
    results = []
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        try:
            nnue = NNUE.from_file(model_path)
            eval_score = nnue.evaluate(board)
            results.append((model_file, eval_score))
            print(f"  {model_file}: {eval_score:.4f}")
        except Exception as e:
            print(f"  {model_file}: ❌ 로드 실패 ({e})")
    
    if results:
        print(f"\n✅ 모든 모델 평가 완료")
        best_model = max(results, key=lambda x: x[1])
        print(f"가장 높은 평가: {best_model[0]} ({best_model[1]:.4f})")


def example_4_play_game():
    """예제 4: AI가 자동으로 게임 진행"""
    print("\n" + "=" * 60)
    print("예제 4: AI가 자동으로 게임 진행 (간단 버전)")
    print("=" * 60)
    
    model_path = get_best_model_path()
    
    if not model_path:
        print("⚠️  사용 가능한 모델이 없습니다.")
        return
    
    print(f"사용 모델: {model_path}")
    
    engine = Engine(
        depth=3,
        use_nnue=True,
        nnue_model_path=model_path
    )
    
    board = Board()
    move_count = 0
    max_moves = 10  # 최대 10수까지만
    
    print("게임 시작!\n")
    
    while move_count < max_moves:
        if board.is_checkmate():
            winner = "CHO" if board.side_to_move == Side.HAN else "HAN"
            print(f"\n🎯 체크메이트! 승자: {winner}")
            break
        
        if board.is_stalemate():
            print("\n🤝 스테일메이트! 무승부")
            break
        
        # 현재 턴
        side = board.side_to_move.value
        print(f"\n[{move_count + 1}수] {side} 차례")
        
        # AI가 수 찾기
        best_move = engine.search(board)
        
        if not best_move:
            print("❌ 유효한 수가 없습니다.")
            break
        
        # 수를 둠
        board.make_move(best_move)
        print(f"  → {best_move.to_uci()} (탐색 노드: {engine.nodes_searched})")
        
        move_count += 1
    
    print(f"\n게임 종료 (총 {move_count}수)")


if __name__ == "__main__":
    print("=" * 60)
    print("장기 AI 모델 사용 예제")
    print("=" * 60)
    
    # 예제 실행
    try:
        example_1_use_model_with_engine()
        example_2_use_nnue_directly()
        example_3_compare_models()
        example_4_play_game()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

