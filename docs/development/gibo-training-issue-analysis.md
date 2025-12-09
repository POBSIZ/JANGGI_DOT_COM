# 기보 기반 Supervised Training 승률 0% 문제 분석

## 문제 현상
기보를 활용한 supervised training을 수행했지만 SimpleEvaluator 대비 승률이 0.0%로 나타남.

## 원인 분석

### 1. 타겟 값 계산 방식의 근본적 문제

#### 현재 기보 학습 방식 (`train_nnue_gibo.py`)
```python
# 게임 결과만으로 타겟 결정
if result == 'cho':
    cho_target = 1.0
    han_target = -1.0
elif result == 'han':
    cho_target = -1.0
    han_target = 1.0
else:
    cho_target = 0.0
    han_target = 0.0

# 진행도에 따라 조정
progress = move_idx / max(total_moves - 1, 1)
target = base_target * (TARGET_BASE_WEIGHT + TARGET_PROGRESS_WEIGHT * progress)
# TARGET_BASE_WEIGHT = 0.3, TARGET_PROGRESS_WEIGHT = 0.7
```

**문제점:**
- 게임 결과만 사용하여 중간 포지션의 실제 평가값을 반영하지 못함
- 예: 초반 포지션(move_idx=0, progress=0)에서 cho가 승리한 게임이면:
  - `target = 1.0 * (0.3 + 0.7 * 0) = 0.3`
  - 하지만 실제 초반 포지션은 대부분 비슷한 평가값을 가져야 함
- 초반에 유리한 포지션이어도 게임 결과가 패배면 음수 타겟을 학습
- 모델이 포지션의 실제 가치를 평가하는 방법을 학습하지 못함

#### Self-play 학습 방식 (`train_nnue_gpu.py`) - 비교
```python
def calculate_target_from_result(eval_score: float, result: float, progress: float) -> float:
    """Calculate target value blending evaluation with game outcome."""
    return EVAL_WEIGHT * np.clip(eval_score / EVAL_SCALE, -1, 1) + RESULT_WEIGHT * result * progress
# EVAL_WEIGHT = 0.7, RESULT_WEIGHT = 0.3
```

**차이점:**
- 평가 점수(실제 포지션 평가)와 게임 결과를 혼합하여 더 정확한 타겟 생성
- 포지션의 실제 가치를 학습할 수 있음

### 2. 구체적인 문제 시나리오

#### 시나리오 1: 초반 유리 포지션, 최종 패배
- 초반 10수에서 cho가 유리한 포지션 (실제 평가: +2.0)
- 하지만 게임 결과가 han 승리
- 현재 방식: `target = -1.0 * (0.3 + 0.7 * 0.05) = -0.335`
- 모델 학습: 초반 유리 포지션을 음수로 평가하도록 학습 (잘못된 학습)

#### 시나리오 2: 중반 균형 포지션, 최종 승리
- 중반 50수에서 균형 포지션 (실제 평가: 0.0)
- 게임 결과가 cho 승리
- 현재 방식: `target = 1.0 * (0.3 + 0.7 * 0.5) = 0.65`
- 모델 학습: 균형 포지션을 양수로 평가하도록 학습 (잘못된 학습)

### 3. 추가 확인 필요 사항

#### 3.1 기보 파싱 정확도
- 좌표 변환 오류 가능성 (`_find_valid_move_helper`의 여러 변환 시도)
- 잘못된 포지션으로 학습했을 가능성
- `failed_moves`가 많으면 파싱 오류 가능성 높음

#### 3.2 모델 학습 상태
- Loss 값이 제대로 감소했는지 확인 필요
- Validation loss와 training loss의 차이 확인
- 모델이 실제로 학습되었는지 확인

#### 3.3 평가 함수 문제
- 평가 시 모델이 제대로 사용되는지 확인
- `evaluate_model` 함수에서 GPU 배치 평가가 제대로 작동하는지 확인

## 해결 방안

### 방안 1: 평가 점수 기반 타겟 계산 (권장)
기보 학습에서도 포지션의 실제 평가 점수를 계산하여 타겟에 포함:

```python
# SimpleEvaluator 또는 현재 모델로 포지션 평가
eval_score = simple_evaluator.evaluate(board)  # 또는 현재 모델

# 평가 점수와 게임 결과를 혼합
target = EVAL_WEIGHT * np.clip(eval_score / EVAL_SCALE, -1, 1) + \
         RESULT_WEIGHT * base_target * progress
```

### 방안 2: Self-play와 혼합 학습
1. 기보 데이터로 초기 학습 (게임 결과 기반)
2. Self-play로 추가 학습 (평가 점수 기반)
3. 두 방법을 번갈아가며 학습

### 방안 3: 기보 파싱 정확도 개선
- 좌표 변환 로직 검증
- 파싱 실패율 모니터링
- 실패한 게임은 제외하고 학습

### 방안 4: 타겟 값 정규화 개선
- 진행도에 따른 가중치 조정
- 초반 포지션은 평가 점수에 더 큰 가중치
- 후반 포지션은 게임 결과에 더 큰 가중치

## 권장 조치

1. **즉시 조치**: 기보 학습에 평가 점수 기반 타겟 계산 추가
2. **검증**: 기보 파싱 정확도 확인 및 개선
3. **모니터링**: 학습 과정에서 loss와 validation loss 추적
4. **혼합 학습**: Self-play와 기보 학습을 결합한 학습 파이프라인 구축

## 참고 코드 위치

- 기보 학습 타겟 계산: `scripts/train_nnue_gibo.py:353`, `scripts/train_nnue_gibo.py:707`
- Self-play 타겟 계산: `scripts/train_nnue_gpu.py:188`
- 평가 함수: `scripts/train_nnue_gpu.py:815` (`evaluate_model`)

