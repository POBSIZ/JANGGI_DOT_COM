#!/usr/bin/env python3
"""스크립트: 학습된 모델이 제대로 수 추론에 적용되고 있는지 확인"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from janggi.board import Board
from janggi.engine import Engine
from janggi.nnue import NNUE, SimpleEvaluator

def check_model_files():
    """모델 파일 존재 여부 확인"""
    print("=" * 60)
    print("1. 모델 파일 확인")
    print("=" * 60)
    
    model_files = [
        "models/nnue_smart_model.json",
        "models/nnue_gpu_model.json",
        "models/nnue_model.json",
    ]
    
    found_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file)
            print(f"✓ {model_file} 존재 (크기: {size:,} bytes)")
            found_models.append(model_file)
        else:
            print(f"✗ {model_file} 없음")
    
    env_path = os.environ.get("NNUE_MODEL_PATH")
    if env_path:
        if os.path.exists(env_path):
            print(f"✓ 환경 변수 NNUE_MODEL_PATH: {env_path} 존재")
            if env_path not in found_models:
                found_models.append(env_path)
        else:
            print(f"✗ 환경 변수 NNUE_MODEL_PATH: {env_path} 없음")
    
    return found_models

def check_model_loading(model_path):
    """모델 로드 테스트"""
    print("\n" + "=" * 60)
    print(f"2. 모델 로드 테스트: {model_path}")
    print("=" * 60)
    
    try:
        nnue = NNUE.from_file(model_path)
        print(f"✓ 모델 로드 성공")
        print(f"  - Feature size: {nnue.feature_size}")
        print(f"  - Hidden1 size: {nnue.hidden1_size}")
        print(f"  - Hidden2 size: {nnue.hidden2_size}")
        print(f"  - Use advanced features: {nnue.use_advanced_features}")
        return nnue
    except Exception as e:
        print(f"✗ 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_engine_with_model(model_path):
    """엔진이 모델을 사용하는지 확인"""
    print("\n" + "=" * 60)
    print(f"3. 엔진 모델 사용 확인: {model_path}")
    print("=" * 60)
    
    try:
        # 모델 없이 엔진 생성 (SimpleEvaluator 사용)
        engine_no_model = Engine(depth=1, use_nnue=False)
        print(f"✓ use_nnue=False 엔진 생성 성공")
        print(f"  - NNUE 객체: {engine_no_model.nnue}")
        print(f"  - Evaluator 타입: {type(engine_no_model.evaluator).__name__}")
        
        # 모델과 함께 엔진 생성
        engine_with_model = Engine(depth=1, use_nnue=True, nnue_model_path=model_path)
        print(f"\n✓ use_nnue=True, nnue_model_path={model_path} 엔진 생성 성공")
        print(f"  - use_nnue: {engine_with_model.use_nnue}")
        print(f"  - NNUE 객체: {engine_with_model.nnue is not None}")
        if engine_with_model.nnue:
            print(f"  - NNUE 타입: {type(engine_with_model.nnue).__name__}")
            print(f"  - Feature size: {engine_with_model.nnue.feature_size}")
        print(f"  - Evaluator 타입: {type(engine_with_model.evaluator).__name__}")
        
        return engine_with_model
    except Exception as e:
        print(f"✗ 엔진 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_evaluation_difference(model_path):
    """모델 사용 여부에 따른 평가 차이 확인"""
    print("\n" + "=" * 60)
    print("4. 평가 함수 차이 확인")
    print("=" * 60)
    
    board = Board()
    
    # SimpleEvaluator로 평가
    simple_eval = SimpleEvaluator()
    simple_score = simple_eval.evaluate(board)
    print(f"SimpleEvaluator 점수: {simple_score:.2f}")
    
    # NNUE 모델로 평가
    try:
        nnue = NNUE.from_file(model_path)
        nnue_score = nnue.evaluate(board)
        print(f"NNUE 모델 점수: {nnue_score:.2f}")
        print(f"점수 차이: {abs(nnue_score - simple_score):.2f}")
        
        if abs(nnue_score - simple_score) < 0.01:
            print("⚠ 경고: 점수 차이가 거의 없습니다. 모델이 제대로 학습되지 않았을 수 있습니다.")
        else:
            print("✓ 모델이 SimpleEvaluator와 다른 평가를 제공합니다.")
    except Exception as e:
        print(f"✗ NNUE 평가 실패: {e}")
        import traceback
        traceback.print_exc()

def check_engine_search(model_path):
    """엔진 검색에서 모델이 사용되는지 확인"""
    print("\n" + "=" * 60)
    print("5. 엔진 검색에서 모델 사용 확인")
    print("=" * 60)
    
    board = Board()
    
    # 모델 없이 검색
    engine_no_model = Engine(depth=2, use_nnue=False)
    print("모델 없이 검색 중...")
    move1 = engine_no_model.search(board)
    nodes1 = engine_no_model.nodes_searched
    print(f"  - 최선 수: {move1.to_uci() if move1 else None}")
    print(f"  - 검색 노드 수: {nodes1}")
    
    # 모델과 함께 검색
    engine_with_model = Engine(depth=2, use_nnue=True, nnue_model_path=model_path)
    print("\n모델과 함께 검색 중...")
    move2 = engine_with_model.search(board)
    nodes2 = engine_with_model.nodes_searched
    print(f"  - 최선 수: {move2.to_uci() if move2 else None}")
    print(f"  - 검색 노드 수: {nodes2}")
    
    if move1 and move2:
        if move1.to_uci() == move2.to_uci():
            print("⚠ 주의: 두 엔진이 같은 수를 선택했습니다. (정상일 수 있음)")
        else:
            print("✓ 모델이 다른 수를 선택했습니다. 모델이 평가에 영향을 미치고 있습니다.")
    else:
        print("✗ 검색 실패")

def check_api_model_path_logic():
    """API의 모델 경로 로직 확인 (api.py의 _get_default_model_path 재현)"""
    print("\n" + "=" * 60)
    print("6. API 모델 경로 로직 확인")
    print("=" * 60)
    
    # api.py의 _get_default_model_path 로직 재현
    env_path = os.environ.get("NNUE_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        print(f"✓ 환경 변수 모델 경로: {env_path}")
        return env_path
    
    if os.path.exists("models/nnue_gpu_model.json"):
        print(f"✓ GPU 모델 발견: models/nnue_gpu_model.json")
        return "models/nnue_gpu_model.json"
    
    if os.path.exists("models/nnue_smart_model.json"):
        print(f"✓ Smart 모델 발견: models/nnue_smart_model.json")
        return "models/nnue_smart_model.json"
    
    if os.path.exists("models/nnue_model.json"):
        print(f"✓ CPU 모델 발견: models/nnue_model.json")
        return "models/nnue_model.json"
    
    # Last resort: find any nnue model in models directory
    if os.path.exists("models"):
        for file in os.listdir("models"):
            if file.endswith(".json") and "nnue" in file.lower():
                model_path = os.path.join("models", file)
                if os.path.exists(model_path):
                    print(f"✓ 기타 NNUE 모델 발견: {model_path}")
                    return model_path
    
    print("✗ 기본 모델 파일을 찾을 수 없습니다.")
    print("  찾는 파일:")
    print("    - models/nnue_gpu_model.json")
    print("    - models/nnue_smart_model.json")
    print("    - models/nnue_model.json")
    print("    - models/*nnue*.json (기타)")
    print("  실제 존재하는 파일:")
    if os.path.exists("models"):
        for f in os.listdir("models"):
            if f.endswith(".json"):
                print(f"    - models/{f}")
    
    return None

def main():
    print("\n" + "=" * 60)
    print("학습된 모델 사용 확인 스크립트")
    print("=" * 60 + "\n")
    
    # 1. 모델 파일 확인
    found_models = check_model_files()
    
    if not found_models:
        print("\n✗ 사용 가능한 모델 파일이 없습니다!")
        return
    
    # API 로직으로 찾은 기본 모델
    default_model = check_api_model_path_logic()
    
    # 사용할 모델 선택 (API 기본 모델 또는 첫 번째 발견된 모델)
    model_to_test = default_model if default_model else found_models[0]
    
    if not model_to_test:
        print("\n✗ 테스트할 모델을 찾을 수 없습니다!")
        return
    
    print(f"\n>>> 테스트할 모델: {model_to_test} <<<\n")
    
    # 2. 모델 로드 테스트
    nnue = check_model_loading(model_to_test)
    if not nnue:
        return
    
    # 3. 엔진 모델 사용 확인
    engine = check_engine_with_model(model_to_test)
    if not engine:
        return
    
    # 4. 평가 차이 확인
    check_evaluation_difference(model_to_test)
    
    # 5. 엔진 검색 확인
    check_engine_search(model_to_test)
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("최종 요약")
    print("=" * 60)
    print(f"✓ 모델 파일: {model_to_test}")
    print(f"✓ 모델 로드: 성공")
    print(f"✓ 엔진에서 모델 사용: {'예' if engine and engine.nnue else '아니오'}")
    
    if default_model:
        print(f"✓ API 기본 모델 경로: {default_model}")
    else:
        print(f"⚠ API 기본 모델 경로: 없음 (모델 파일 이름이 다를 수 있음)")
        print(f"  해결 방법:")
        print(f"    1. 환경 변수 설정: export NNUE_MODEL_PATH={model_to_test}")
        print(f"    2. 또는 모델 파일 이름 변경: {model_to_test} -> models/nnue_model.json")

if __name__ == "__main__":
    main()

