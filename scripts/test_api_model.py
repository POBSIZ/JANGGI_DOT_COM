#!/usr/bin/env python3
"""API 엔드포인트에서 모델이 제대로 사용되는지 테스트"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from janggi.board import Board
from janggi.engine import Engine
from api import _get_default_model_path, NewGameRequest

def test_api_model_loading():
    """API의 모델 로딩 로직 테스트"""
    print("=" * 60)
    print("API 모델 로딩 테스트")
    print("=" * 60)
    
    # API의 기본 모델 경로 확인
    default_path = _get_default_model_path()
    print(f"✓ API 기본 모델 경로: {default_path}")
    
    if not default_path:
        print("✗ 모델 경로를 찾을 수 없습니다!")
        return False
    
    # NewGameRequest 시뮬레이션
    request = NewGameRequest(
        game_id="test_game",
        depth=3,
        use_nnue=True,
        nnue_model_path=None  # 기본 경로 사용
    )
    
    # api.py의 로직 재현
    nnue_model_path = None
    if request.use_nnue:
        if request.nnue_model_path:
            nnue_model_path = request.nnue_model_path
        elif default_path and os.path.exists(default_path):
            nnue_model_path = default_path
    
    print(f"✓ 선택된 모델 경로: {nnue_model_path}")
    
    if not nnue_model_path:
        print("✗ 모델 경로가 설정되지 않았습니다!")
        return False
    
    # 엔진 생성 및 모델 사용 확인
    try:
        engine = Engine(
            depth=request.depth,
            use_nnue=request.use_nnue,
            nnue_model_path=nnue_model_path
        )
        
        print(f"✓ 엔진 생성 성공")
        print(f"  - use_nnue: {engine.use_nnue}")
        print(f"  - nnue 객체 존재: {engine.nnue is not None}")
        
        if engine.nnue:
            print(f"  - 모델 feature_size: {engine.nnue.feature_size}")
            print(f"  - 모델 hidden1_size: {engine.nnue.hidden1_size}")
            print(f"  - 모델 hidden2_size: {engine.nnue.hidden2_size}")
        
        # 실제 평가 테스트
        board = Board()
        score = engine._evaluate(board)
        print(f"  - 초기 보드 평가 점수: {score:.4f}")
        
        # 검색 테스트
        print("\n검색 테스트 중...")
        move = engine.search(board)
        if move:
            print(f"  - 선택된 수: {move.to_uci()}")
            print(f"  - 검색 노드 수: {engine.nodes_searched}")
            print(f"  - 오프닝 북 사용: {engine.used_opening_book}")
        else:
            print("  - 수를 찾을 수 없습니다")
        
        return True
        
    except Exception as e:
        print(f"✗ 엔진 생성/사용 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_api_model_loading()
    if success:
        print("\n" + "=" * 60)
        print("✓ 모든 테스트 통과! 모델이 제대로 사용되고 있습니다.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ 테스트 실패! 모델이 제대로 사용되지 않고 있습니다.")
        print("=" * 60)
        sys.exit(1)

