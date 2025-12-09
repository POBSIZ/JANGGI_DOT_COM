# 장기 AI 엔진 - 프론트엔드

Vite + React + TypeScript로 구성된 장기 게임 프론트엔드입니다.

## 설치 및 실행

```bash
# 의존성 설치
npm install

# 개발 서버 실행
npm run dev

# 프로덕션 빌드
npm run build

# 빌드 미리보기
npm run preview
```

## 프로젝트 구조

```
frontend/
├── src/
│   ├── pages/
│   │   ├── GamePage.tsx          # AI 대전 페이지
│   │   ├── GamePage.css
│   │   ├── MultiplayerPage.tsx   # 멀티플레이어 페이지
│   │   └── MultiplayerPage.css
│   ├── App.tsx                   # 메인 앱 컴포넌트 (라우팅)
│   ├── App.css
│   ├── main.tsx                  # 진입점
│   └── index.css
├── public/
├── index.html
├── vite.config.ts                # Vite 설정 (API 프록시 포함)
└── package.json
```

## 주요 기능

- **AI 대전**: AI와 장기 대국
- **멀티플레이어**: 실시간 온라인 대전
- **Canvas 기반 보드 렌더링**: 부드러운 애니메이션과 인터랙션
- **반응형 디자인**: 모바일 및 데스크톱 지원

## API 프록시

개발 환경에서 `/api`와 `/ws` 요청은 자동으로 백엔드 서버(`http://localhost:8000`)로 프록시됩니다.

## 기술 스택

- **Vite**: 빠른 개발 서버 및 빌드 도구
- **React 19**: UI 라이브러리
- **TypeScript**: 타입 안정성
- **React Router**: 클라이언트 사이드 라우팅
