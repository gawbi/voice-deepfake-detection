"""
ResNet-18 기반 음성 딥페이크 탐지 모델
- 입력: (B, C, 128, T) — C=1(멜 단일) 또는 C=2(이중 파이프라인)
- 출력: (B, 2) — [bonafide_score, spoof_score]
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetDetector(nn.Module):
    """ResNet-18 기반 경량 딥페이크 탐지 모델"""

    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()

        # ResNet-18 백본
        self.backbone = models.resnet18(weights=None)

        # 입력 채널 수정 (1ch 또는 2ch)
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # 분류 헤드
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class LightCNNDetector(nn.Module):
    """경량 CNN — ResNet보다 빠른 학습용"""

    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
