# AWS EC2 배포 가이드

이 가이드는 Janggi AI API 서버를 AWS EC2에 배포하는 간단한 방법을 설명합니다.

## 사전 준비

1. AWS 계정 및 EC2 인스턴스 접근 권한
2. SSH 키 페어 (EC2 인스턴스 접근용)
3. 프로젝트 코드 (Git 저장소 또는 로컬 파일)

## 1. EC2 인스턴스 생성

### 1.1 인스턴스 시작

1. AWS 콘솔에서 EC2 서비스로 이동
2. "인스턴스 시작" 클릭
3. 다음 설정 권장:
   - **AMI**: Ubuntu 22.04 LTS (또는 Amazon Linux 2023)
   - **인스턴스 유형**: t3.micro (무료 티어) 또는 t3.small 이상
   - **키 페어**: 기존 키 페어 선택 또는 새로 생성
   - **보안 그룹**: 새 보안 그룹 생성 (아래 참고)

### 1.2 보안 그룹 설정

다음 인바운드 규칙 추가:
- **SSH (22)**: 내 IP에서만 접근 허용
- **HTTP (80)**: 0.0.0.0/0 (선택사항, nginx 사용 시)
- **HTTPS (443)**: 0.0.0.0/0 (선택사항, SSL 사용 시)
- **커스텀 TCP (8000)**: 0.0.0.0/0 (API 서버 직접 접근용)

## 2. EC2 인스턴스 설정

### 2.1 SSH 접속

```bash
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

### 2.2 시스템 업데이트

```bash
sudo apt update
sudo apt upgrade -y
```

### 2.3 Python 및 필수 도구 설치

```bash
# Python 3.10 이상 설치 (Ubuntu 22.04는 기본 포함)
python3 --version

# uv 설치 (Python 패키지 관리자)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env  # 또는 새 터미널 열기

# 또는 pip로 설치
pip3 install uv

# uv 설치 확인
uv --version
```

### 2.4 프로젝트 파일 업로드

#### 방법 1: Git 사용 (권장)

```bash
# Git 설치
sudo apt install git -y

# 프로젝트 클론
cd ~
git clone https://github.com/your-username/janggi-dot-com.git
cd janggi-dot-com
```

#### 방법 2: SCP 사용

로컬 터미널에서:

```bash
scp -i your-key.pem -r /path/to/JANGGI_DOT_COM ubuntu@your-ec2-public-ip:~/
```

EC2에서:

```bash
cd ~/JANGGI_DOT_COM
```

## 3. 의존성 설치 및 서버 실행

### 3.1 의존성 설치

```bash
# 프로젝트 디렉토리로 이동
cd ~/janggi-dot-com  # 또는 ~/JANGGI_DOT_COM

# 의존성 설치
uv sync
```

### 3.2 모델 파일 확인

모델 파일이 `models/` 디렉토리에 있는지 확인:

```bash
ls -la models/
```

모델 파일이 없다면, 로컬에서 업로드:

```bash
# 로컬에서 실행
scp -i your-key.pem models/*.json ubuntu@your-ec2-public-ip:~/janggi-dot-com/models/
```

### 3.3 서버 테스트 실행

```bash
# 테스트 실행 (백그라운드)
uv run python main.py --host 0.0.0.0 --port 8000

# 또는 특정 모델 지정
uv run python main.py --model models/nnue_smart_model.json --host 0.0.0.0 --port 8000
```

브라우저에서 `http://your-ec2-public-ip:8000` 접속하여 `{"message":"Janggi AI Engine API"}` 응답 확인

## 4. Systemd 서비스로 자동 실행 설정

### 4.1 서비스 파일 생성

```bash
sudo nano /etc/systemd/system/janggi-api.service
```

다음 내용 입력:

```ini
[Unit]
Description=Janggi AI API Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/janggi-dot-com
Environment="PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ubuntu/.cargo/bin/uv run python main.py --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**참고**: `uv` 경로는 설치 방법에 따라 다를 수 있습니다:
- `~/.cargo/bin/uv` (공식 설치 스크립트 사용 시)
- `~/.local/bin/uv` (pip로 설치 시)
- `which uv` 명령으로 정확한 경로 확인

### 4.2 서비스 활성화 및 시작

```bash
# systemd 재로드
sudo systemctl daemon-reload

# 서비스 활성화 (부팅 시 자동 시작)
sudo systemctl enable janggi-api

# 서비스 시작
sudo systemctl start janggi-api

# 서비스 상태 확인
sudo systemctl status janggi-api

# 로그 확인
sudo journalctl -u janggi-api -f
```

### 4.3 서비스 관리 명령어

```bash
# 서비스 중지
sudo systemctl stop janggi-api

# 서비스 재시작
sudo systemctl restart janggi-api

# 서비스 상태 확인
sudo systemctl status janggi-api

# 로그 확인
sudo journalctl -u janggi-api -n 50
```

## 5. (선택사항) Nginx 리버스 프록시 설정

외부에서 포트 8000 대신 80/443 포트로 접근하려면 Nginx를 사용할 수 있습니다.

### 5.1 Nginx 설치

```bash
sudo apt install nginx -y
```

### 5.2 Nginx 설정 파일 생성

```bash
sudo nano /etc/nginx/sites-available/janggi-api
```

다음 내용 입력:

```nginx
server {
    listen 80;
    server_name your-domain.com;  # 또는 EC2 퍼블릭 IP

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 5.3 Nginx 활성화 및 시작

```bash
# 심볼릭 링크 생성
sudo ln -s /etc/nginx/sites-available/janggi-api /etc/nginx/sites-enabled/

# 기본 설정 제거 (선택사항)
sudo rm /etc/nginx/sites-enabled/default

# 설정 테스트
sudo nginx -t

# Nginx 재시작
sudo systemctl restart nginx

# Nginx 자동 시작 설정
sudo systemctl enable nginx
```

## 6. 방화벽 설정 (UFW)

```bash
# UFW 활성화
sudo ufw enable

# SSH 허용
sudo ufw allow 22/tcp

# HTTP 허용 (Nginx 사용 시)
sudo ufw allow 80/tcp

# HTTPS 허용 (SSL 사용 시)
sudo ufw allow 443/tcp

# API 서버 직접 접근 (Nginx 미사용 시)
sudo ufw allow 8000/tcp

# 상태 확인
sudo ufw status
```

## 7. 배포 확인

### 7.1 API 엔드포인트 테스트

```bash
# 로컬에서 테스트
curl http://your-ec2-public-ip:8000/

# 또는 Nginx 사용 시
curl http://your-ec2-public-ip/
```

### 7.2 게임 생성 테스트

```bash
curl -X POST http://your-ec2-public-ip:8000/api/new-game \
  -H "Content-Type: application/json" \
  -d '{"game_id": "test", "depth": 3, "use_nnue": true}'
```

## 8. 문제 해결

### 8.1 서비스가 시작되지 않는 경우

```bash
# 로그 확인
sudo journalctl -u janggi-api -n 100

# 수동 실행으로 오류 확인
cd ~/janggi-dot-com
uv run python main.py --host 0.0.0.0 --port 8000
```

### 8.2 포트가 열리지 않는 경우

```bash
# 포트 사용 확인
sudo netstat -tlnp | grep 8000

# 보안 그룹 확인 (AWS 콘솔에서)
# EC2 > 인스턴스 > 보안 그룹 > 인바운드 규칙
```

### 8.3 모델 파일을 찾을 수 없는 경우

```bash
# 모델 파일 확인
ls -la models/

# 환경 변수로 모델 경로 지정
# /etc/systemd/system/janggi-api.service 파일 수정:
# Environment="NNUE_MODEL_PATH=/home/ubuntu/janggi-dot-com/models/nnue_smart_model.json"
```

## 9. 업데이트 및 재배포

### 9.1 코드 업데이트

```bash
# Git 사용 시
cd ~/janggi-dot-com
git pull
uv sync

# 서비스 재시작
sudo systemctl restart janggi-api
```

### 9.2 모델 파일 업데이트

```bash
# 로컬에서 새 모델 업로드
scp -i your-key.pem models/new_model.json ubuntu@your-ec2-public-ip:~/janggi-dot-com/models/

# 서비스 재시작
sudo systemctl restart janggi-api
```

## 10. 비용 최적화

- **인스턴스 유형**: t3.micro (무료 티어) 또는 t3.small
- **스팟 인스턴스**: 개발/테스트 환경에서 비용 절감
- **Elastic IP**: 고정 IP 주소 (선택사항)
- **CloudWatch**: 모니터링 및 로그 관리 (선택사항)

## 참고사항

- 프로덕션 환경에서는 HTTPS(SSL/TLS) 사용 권장
- 도메인 연결 시 Route 53 사용
- 로드 밸런서 사용 시 ALB(Application Load Balancer) 고려
- 백업: 정기적으로 모델 파일 및 설정 백업

