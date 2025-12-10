#!/bin/bash
# AWS EC2 배포 스크립트
# 사용법: ./scripts/deploy_ec2.sh <ec2-user> <ec2-host> <key-file>

set -e

if [ $# -lt 3 ]; then
    echo "사용법: $0 <ec2-user> <ec2-host> <key-file>"
    echo "예시: $0 ubuntu ec2-xxx-xxx-xxx-xxx.compute-1.amazonaws.com ~/.ssh/my-key.pem"
    exit 1
fi

EC2_USER=$1
EC2_HOST=$2
KEY_FILE=$3
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "🚀 Janggi API 서버 배포 시작..."
echo "EC2 호스트: $EC2_USER@$EC2_HOST"
echo "프로젝트 디렉토리: $PROJECT_DIR"

# 1. 로컬에서 필요한 파일 확인
echo ""
echo "📦 로컬 파일 확인 중..."
if [ ! -f "$PROJECT_DIR/main.py" ]; then
    echo "❌ main.py 파일을 찾을 수 없습니다."
    exit 1
fi

# 2. EC2에 프로젝트 디렉토리 생성
echo ""
echo "📁 EC2에 프로젝트 디렉토리 생성 중..."
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_HOST" "mkdir -p ~/janggi-dot-com"

# 3. 프로젝트 파일 업로드 (models 제외)
echo ""
echo "📤 프로젝트 파일 업로드 중..."
rsync -avz --exclude '__pycache__' \
           --exclude '*.pyc' \
           --exclude '.git' \
           --exclude 'node_modules' \
           --exclude 'frontend/dist' \
           --exclude '.pytest_cache' \
           --exclude 'venv' \
           --exclude 'env' \
           -e "ssh -i $KEY_FILE" \
           "$PROJECT_DIR/" "$EC2_USER@$EC2_HOST:~/janggi-dot-com/"

# 4. 모델 파일 업로드 (있는 경우)
if [ -d "$PROJECT_DIR/models" ] && [ "$(ls -A $PROJECT_DIR/models/*.json 2>/dev/null)" ]; then
    echo ""
    echo "🤖 모델 파일 업로드 중..."
    ssh -i "$KEY_FILE" "$EC2_USER@$EC2_HOST" "mkdir -p ~/janggi-dot-com/models"
    scp -i "$KEY_FILE" "$PROJECT_DIR/models"/*.json "$EC2_USER@$EC2_HOST:~/janggi-dot-com/models/" 2>/dev/null || echo "⚠️  모델 파일 업로드 실패 (무시됨)"
fi

# 5. EC2에서 의존성 설치 및 서비스 설정
echo ""
echo "⚙️  EC2에서 설정 중..."
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_HOST" << 'ENDSSH'
set -e

cd ~/janggi-dot-com

# uv 설치 확인 및 설치
if ! command -v uv &> /dev/null; then
    echo "📦 uv 설치 중..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# 의존성 설치
echo "📦 의존성 설치 중..."
export PATH="$HOME/.cargo/bin:$PATH"
uv sync

# systemd 서비스 파일 생성
echo "🔧 systemd 서비스 설정 중..."
UV_PATH=$(which uv || echo "$HOME/.cargo/bin/uv")

sudo tee /etc/systemd/system/janggi-api.service > /dev/null << EOF
[Unit]
Description=Janggi AI API Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/janggi-dot-com
Environment="PATH=$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$UV_PATH run python main.py --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# systemd 재로드 및 서비스 활성화
sudo systemctl daemon-reload
sudo systemctl enable janggi-api
sudo systemctl restart janggi-api

echo "✅ 서비스 시작됨"
echo ""
echo "서비스 상태 확인:"
sudo systemctl status janggi-api --no-pager -l

ENDSSH

echo ""
echo "✅ 배포 완료!"
echo ""
echo "다음 명령어로 서비스 상태 확인:"
echo "  ssh -i $KEY_FILE $EC2_USER@$EC2_HOST 'sudo systemctl status janggi-api'"
echo ""
echo "API 테스트:"
echo "  curl http://$EC2_HOST:8000/"
echo ""
echo "로그 확인:"
echo "  ssh -i $KEY_FILE $EC2_USER@$EC2_HOST 'sudo journalctl -u janggi-api -f'"

