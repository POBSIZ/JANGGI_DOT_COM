# 학습 관련 문서

이 폴더에는 NNUE 모델 학습과 관련된 모든 문서가 포함되어 있습니다.

## 문서 목록

- **[guide.md](guide.md)** - NNUE 모델 학습 가이드 (종합)
  - 환경 설정
  - 빠른 시작
  - 스마트 학습 방법
  - GPU/CPU/기보 학습 방법
  - 문제 해결
  - 고급 사용법

- **[commands.md](commands.md)** - 학습 명령어 모음
  - 스마트 학습 명령어
  - 수동 GPU 학습 명령어
  - 주요 파라미터 설명
  - GPU 최적화 기능

- **[gpu-issues.md](gpu-issues.md)** - GPU 학습 문제점 분석
  - 중복 Feature 추출 문제
  - CPU-GPU 데이터 전송 비효율
  - 자기대국 생성 시 NNUE 미사용
  - 모델 평가 시 GPU 미사용
  - 해결 방법 및 성능 개선

- **[gpu-optimization.md](gpu-optimization.md)** - GPU 활용도 개선 방안
  - 중앙 집중식 GPU 배치 평가 큐
  - 배치 크기 자동 최적화
  - 병렬 self-play 최적화
  - 실제 성능 향상 결과

## 권장 읽기 순서

1. 처음 시작하는 경우: [guide.md](guide.md)의 "빠른 시작" 섹션부터 읽으세요.
2. 학습 명령어가 필요할 때: [commands.md](commands.md)를 참조하세요.
3. GPU 문제가 발생했을 때: [gpu-issues.md](gpu-issues.md)를 확인하세요.
4. GPU 성능을 최적화하고 싶을 때: [gpu-optimization.md](gpu-optimization.md)를 참조하세요.

