"""저장된 모델로 EER만 측정"""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.dataset import get_dataloaders
from src.models.resnet_detector import ResNetDetector
from src.training.train import evaluate, compute_eer

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = str(ROOT / "datasets" / "asvspoof2019_LA" / "LA")
CKPT_DIR = str(ROOT / "checkpoints")

device = torch.device('cpu')

# 모델 로드
ckpt = torch.load(f"{CKPT_DIR}/best_mel_resnet.pt", map_location=device, weights_only=False)
print(f"체크포인트 정보:")
print(f"  Epoch: {ckpt.get('epoch', '?')}")
print(f"  Feature: {ckpt.get('feature_type', '?')}")
print(f"  Model: {ckpt.get('model_type', '?')}")
print(f"  Dev EER: {ckpt.get('eer', '?'):.2f}%")

# 모델 재구성
model = ResNetDetector(in_channels=1, num_classes=2)
model.load_state_dict(ckpt['model_state'])
model = model.to(device)
model.eval()

criterion = nn.CrossEntropyLoss()

# Dev set 평가
print("\n=== Dev Set 평가 ===")
_, dev_loaders = get_dataloaders(DATA_DIR, feature_type='mel', batch_size=64, num_workers=0, max_samples=None)

if 'dev' in dev_loaders:
    dev_loss, dev_acc, dev_eer = evaluate(model, dev_loaders['dev'], criterion, device)
    print(f"  Dev — Acc: {dev_acc:.1f}%, EER: {dev_eer:.2f}%")

# Eval set 평가 (subset - 전체는 71k라 오래 걸림)
print("\n=== Eval Set 평가 (5000 샘플) ===")
_, eval_loaders = get_dataloaders(DATA_DIR, feature_type='mel', batch_size=64, num_workers=0, max_samples=5000)

if 'eval' in eval_loaders:
    eval_loss, eval_acc, eval_eer = evaluate(model, eval_loaders['eval'], criterion, device)
    print(f"  Eval — Acc: {eval_acc:.1f}%, EER: {eval_eer:.2f}%")

print("\n완료!")
