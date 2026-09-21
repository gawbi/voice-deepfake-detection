#!/bin/bash
# ASVspoof 2019 LA 전체 파이프라인 실행 스크립트
# 사용법: bash run_experiment.sh

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ZIP="$PROJECT_DIR/datasets/asvspoof2019_LA/LA.zip"
DATA_DIR="$PROJECT_DIR/datasets/asvspoof2019_LA/LA"

echo "=============================================="
echo "ASVspoof 2019 LA — 전체 파이프라인"
echo "=============================================="

# 1. 압축 해제
if [ ! -d "$DATA_DIR" ]; then
    echo ""
    echo "[1/4] 압축 해제 중..."
    cd "$PROJECT_DIR/datasets/asvspoof2019_LA"
    unzip -q LA.zip
    echo "압축 해제 완료"
else
    echo "[1/4] 이미 압축 해제됨, 스킵"
fi

# 데이터 확인
echo ""
echo "데이터 구조:"
ls -la "$DATA_DIR/"
echo ""
echo "프로토콜 파일:"
ls "$DATA_DIR/ASVspoof2019_LA_cm_protocols/" 2>/dev/null || echo "  (경로 확인 필요)"
echo ""
echo "Train 오디오 파일 수:"
ls "$DATA_DIR/ASVspoof2019_LA_train/flac/" 2>/dev/null | wc -l || echo "  (경로 확인 필요)"

# 2. 베이스라인: 멜 스펙트로그램 + ResNet-18
echo ""
echo "[2/4] 베이스라인 학습: 멜 스펙트로그램 + ResNet-18"
echo "=============================================="
cd "$PROJECT_DIR"
python3 src/training/train.py \
    --data_dir "$DATA_DIR" \
    --feature mel \
    --model resnet \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-4

# 3. 이중 파이프라인: 멜+웨이블릿 + ResNet-18
echo ""
echo "[3/4] 이중 파이프라인 학습: 멜+웨이블릿 + ResNet-18"
echo "=============================================="
python3 src/training/train.py \
    --data_dir "$DATA_DIR" \
    --feature dual \
    --model resnet \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-4

# 4. 결과 비교
echo ""
echo "[4/4] 결과 비교"
echo "=============================================="
echo "체크포인트:"
ls -la "$PROJECT_DIR/checkpoints/"
echo ""
echo "실험 완료!"
