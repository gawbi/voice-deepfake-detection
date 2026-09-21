"""
ASVspoof 2019 LA 학습 & EER 측정 스크립트
- 베이스라인: 멜 스펙트로그램 단일 특징 + ResNet-18
- 목표: 멜+웨이블릿 이중 파이프라인 + ResNet-18
"""

import sys
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.dataset import ASVspoof2019LA, get_dataloaders
from src.models.resnet_detector import ResNetDetector, LightCNNDetector


def compute_eer(scores, labels):
    """Equal Error Rate 계산"""
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    try:
        eer = brentq(lambda x: interp1d(fpr, fnr)(x) - x, 0.0, 1.0)
    except ValueError:
        # fallback: 가장 가까운 지점
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2

    return eer * 100  # 퍼센트


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader):
        features = batch['features'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)}, "
                  f"Loss: {loss.item():.4f}, "
                  f"Acc: {100.*correct/total:.1f}%")

    return total_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_scores = []
    all_labels = []

    for batch in loader:
        features = batch['features'].to(device)
        labels = batch['label'].to(device)

        outputs = model(features)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * features.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # EER 계산용 스코어 수집
        probs = torch.softmax(outputs, dim=1)
        spoof_scores = probs[:, 1].cpu().numpy()
        all_scores.extend(spoof_scores)
        all_labels.extend(labels.cpu().numpy())

    accuracy = 100. * correct / total
    avg_loss = total_loss / total

    # EER 계산
    eer = compute_eer(np.array(all_scores), np.array(all_labels))

    return avg_loss, accuracy, eer


def main():
    parser = argparse.ArgumentParser(description='ASVspoof 2019 LA Training')
    parser.add_argument('--data_dir', type=str, required=True, help='LA 데이터 디렉토리')
    parser.add_argument('--feature', type=str, default='mel', choices=['mel', 'dual'],
                        help='특징 유형: mel(단일) 또는 dual(이중 파이프라인)')
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'light'],
                        help='모델: resnet(ResNet-18) 또는 light(경량 CNN)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_samples', type=int, default=None, help='디버깅용 최대 샘플 수')
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()

    if args.max_samples and args.max_samples <= 500:
        device = torch.device('cpu')  # 소규모 테스트는 CPU
    else:
        device = torch.device('mps' if torch.backends.mps.is_available()
                              else 'cuda' if torch.cuda.is_available()
                              else 'cpu')
    print(f"Device: {device}")

    # 저장 디렉토리
    if args.save_dir is None:
        args.save_dir = str(Path(__file__).parent.parent.parent / 'checkpoints')
    os.makedirs(args.save_dir, exist_ok=True)

    # 데이터 로드
    print("\n=== 데이터 로드 ===")
    in_channels = 2 if args.feature == 'dual' else 1
    max_len = 200 if args.feature == 'dual' else 300  # CWT는 느려서 짧게

    datasets, loaders = get_dataloaders(
        args.data_dir,
        feature_type=args.feature,
        batch_size=args.batch_size,
        max_len=max_len,
        num_workers=0,  # macOS에서 안정성
        max_samples=args.max_samples,
    )

    # 모델
    print(f"\n=== 모델: {args.model} (input channels: {in_channels}) ===")
    if args.model == 'resnet':
        model = ResNetDetector(in_channels=in_channels, num_classes=2)
    else:
        model = LightCNNDetector(in_channels=in_channels, num_classes=2)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"학습 파라미터 수: {param_count:,}")

    # 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 학습 루프
    print(f"\n=== 학습 시작 ({args.epochs} epochs) ===")
    best_eer = 100.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, loaders['train'], criterion, optimizer, device
        )

        # Eval on dev
        dev_loss, dev_acc, dev_eer = evaluate(
            model, loaders['dev'], criterion, device
        )

        scheduler.step()
        elapsed = time.time() - start

        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.0f}s)")
        print(f"  Train — Loss: {train_loss:.4f}, Acc: {train_acc:.1f}%")
        print(f"  Dev   — Loss: {dev_loss:.4f}, Acc: {dev_acc:.1f}%, EER: {dev_eer:.2f}%")

        # 최고 모델 저장
        if dev_eer < best_eer:
            best_eer = dev_eer
            save_path = os.path.join(args.save_dir, f'best_{args.feature}_{args.model}.pt')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'eer': dev_eer,
                'feature_type': args.feature,
                'model_type': args.model,
            }, save_path)
            print(f"  *** Best model saved (EER: {dev_eer:.2f}%) → {save_path}")

    # Eval 셋 최종 평가
    print(f"\n=== 최종 평가 (eval set) ===")
    if 'eval' in loaders:
        model.load_state_dict(torch.load(
            os.path.join(args.save_dir, f'best_{args.feature}_{args.model}.pt')
        )['model_state'])

        eval_loss, eval_acc, eval_eer = evaluate(
            model, loaders['eval'], criterion, device
        )
        print(f"  Eval — Loss: {eval_loss:.4f}, Acc: {eval_acc:.1f}%, EER: {eval_eer:.2f}%")
    else:
        eval_eer = best_eer

    # 결과 요약
    print(f"\n{'='*50}")
    print(f"실험 결과 요약")
    print(f"{'='*50}")
    print(f"  Feature: {args.feature}")
    print(f"  Model: {args.model}")
    print(f"  Best Dev EER: {best_eer:.2f}%")
    if 'eval' in loaders:
        print(f"  Eval EER: {eval_eer:.2f}%")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
    