# 기보 파싱 테스트 가이드

## 수정된 기능

참고 문서 (https://github.com/ladofa/janggi/blob/master/python_tf/gibo.py)를 기반으로 기보 파싱 기능을 개선했습니다.

### 주요 개선 사항

1. **0xff 문자 제거**: 파일을 바이트로 읽어 0xff 문자를 제거
2. **주석 처리**: `{` ~ `}` 사이의 주석을 건너뛰기
3. **한글 기물 지원**: 한글 기물(마, 졸, 포, 상, 사, 장, 차, 병)을 한자로 변환
4. **다양한 형식 지원**:
   - 한글 형식: `08마87` (숫자-한글기물-숫자)
   - 한자 형식: `79卒78` (숫자-한자기물-숫자)
   - 진영 표시 형식: `79楚卒78` (숫자-진영-기물-숫자)
   - 한수쉼: `한수쉼` 처리
5. **좌표 파싱 개선**: 참고 문서 방식 적용 (word_move[0] = fy, word_move[1] = fx)
6. **인코딩 처리**: cp949 우선 시도

## 지원하는 기보 형식

### 형식 1: 한글 기물
```
1. 08마87 2. 12마33 3. 88포85
```

### 형식 2: 한자 기물
```
1. 79楚卒78 2. 41漢兵42 3. 02楚馬83
```

### 형식 3: 한수쉼
```
129. 한수쉼 130. 35포15
```

## 테스트 방법

### 방법 1: 단일 파일 테스트
```bash
python test_single_gibo.py gibo/지만이.gib
```

### 방법 2: 모든 파일 테스트
```bash
python test_gibo_parsing.py
```

### 방법 3: Python 코드로 직접 테스트
```python
from scripts.train_nnue_gibo import GibParser

parser = GibParser()
games = parser.parse_file('gibo/지만이.gib')
print(f"Parsed {len(games)} games")
for game in games:
    print(f"  Moves: {len(game['raw_moves'])}")
    print(f"  Result: {game.get('result')}")
```

## 예상 결과

- ✅ 한글 기물 형식 (`지만이.gib`) 파싱 가능
- ✅ 한자 기물 형식 (`아마고수기보001.gib`) 파싱 가능
- ✅ 진영 표시 형식 (`79楚卒78`) 파싱 가능
- ✅ 한수쉼 처리 가능
- ✅ 0xff 문자 제거
- ✅ 주석 처리

## 파일 목록

gibo 디렉토리에는 약 257개의 .gib 파일이 있습니다:
- 아마고수기보001.gib ~ 아마고수기보162.gib
- 카카오장기기보001.gib ~ 카카오장기기보057.gib
- 지만이.gib
- 제2회친선대회총보.gib
- 등등...

모든 파일이 파싱 가능해야 합니다.

