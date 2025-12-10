# 장기 AI 엔진 (Janggi AI Engine)

NNUE (Efficiently Updatable Neural Networks) 기반의 한국 장기 AI 엔진입니다.

## 빠른 시작

```bash
# 의존성 설치
uv sync

# 백엔드 서버 실행
uv run python main.py

# 특정 모델로 서버 실행
uv run python main.py --model models/nnue_smart_model.json

# 프론트엔드 실행 (새 터미널에서)
cd frontend
npm install
npm run dev
```

브라우저에서 `http://localhost:5173`을 열어 게임을 시작하세요.

## 문서

- **[시작 가이드](docs/getting-started/README.md)** - 설치 및 기본 사용법
- **[학습 가이드](docs/training/guide.md)** - NNUE 모델 학습 방법
- **[모델 사용](docs/models/usage.md)** - 학습된 모델 사용법
- **[AWS EC2 배포 가이드](docs/deployment/aws-ec2.md)** - AWS EC2에 API 서버 배포하기
- **[장기 규칙](docs/rules/korean.md)** - 한국 장기 규칙 (한국어)
- **[Janggi Rules](docs/rules/english.md)** - Korean Janggi Rules (English)

## 주요 기능

- 완전한 장기 규칙 구현 (한/초, 모든 말의 이동 규칙)
- NNUE (Efficiently Updatable Neural Networks) 기반 평가 함수
- 미니맥스 알고리즘과 알파-베타 가지치기
- FastAPI 백엔드
- React + TypeScript + Vite 프론트엔드

## 라이선스

MIT License

