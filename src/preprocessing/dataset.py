"""
ASVspoof 2019 LA 데이터셋 로더
- 프로토콜 파일 파싱
- 오디오 로드 + 특징 추출
- PyTorch Dataset
"""

import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path
from .audio_features import extract_mel_spectrogram, extract_dual_pipeline


class ASVspoof2019LA(Dataset):
    """ASVspoof 2019 LA 데이터셋"""

    def __init__(self, base_dir, partition='train', feature_type='mel', max_len=300, sr=16000, max_samples=None):
        """
        Args:
            base_dir: LA 폴더 경로 (ASVspoof2019_LA_train 등의 상위)
            partition: 'train', 'dev', 'eval'
            feature_type: 'mel' (단일) 또는 'dual' (멜+웨이블릿)
            max_len: 시간축 최대 프레임 수
            sr: 샘플링 레이트
            max_samples: 최대 샘플 수 (디버깅용)
        """
        self.base_dir = Path(base_dir)
        self.partition = partition
        self.feature_type = feature_type
        self.max_len = max_len
        self.sr = sr

        # 파티션별 경로
        partition_map = {
            'train': ('ASVspoof2019_LA_train', 'ASVspoof2019.LA.cm.train.trn.txt'),
            'dev': ('ASVspoof2019_LA_dev', 'ASVspoof2019.LA.cm.dev.trl.txt'),
            'eval': ('ASVspoof2019_LA_eval', 'ASVspoof2019.LA.cm.eval.trl.txt'),
        }

        audio_dir_name, protocol_name = partition_map[partition]
        self.audio_dir = self.base_dir / audio_dir_name / 'flac'
        self.protocol_file = self.base_dir / 'ASVspoof2019_LA_cm_protocols' / protocol_name

        # 프로토콜 파싱
        self.samples = self._parse_protocol()

        if max_samples:
            import random
            random.seed(42)
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        print(f"[{partition}] {len(self.samples)} samples loaded "
              f"(bonafide: {sum(1 for s in self.samples if s['label']==0)}, "
              f"spoof: {sum(1 for s in self.samples if s['label']==1)})")

    def _parse_protocol(self):
        """프로토콜 파일 파싱 → [{filename, attack_type, label}, ...]"""
        samples = []

        if not self.protocol_file.exists():
            raise FileNotFoundError(f"프로토콜 파일 없음: {self.protocol_file}")

        with open(self.protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # SPEAKER_ID FILENAME - ATTACK_TYPE LABEL
                    filename = parts[1]
                    attack_type = parts[3]
                    label_str = parts[4]

                    label = 0 if label_str == 'bonafide' else 1  # 0=진짜, 1=가짜

                    audio_path = self.audio_dir / f"{filename}.flac"
                    if audio_path.exists():
                        samples.append({
                            'filename': filename,
                            'audio_path': str(audio_path),
                            'attack_type': attack_type,
                            'label': label,
                        })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 오디오 로드
        audio, sr = librosa.load(sample['audio_path'], sr=self.sr)

        # 특징 추출
        if self.feature_type == 'mel':
            features = extract_mel_spectrogram(audio, sr, max_len=self.max_len)
            features = features[np.newaxis, ...]  # (1, 128, max_len)
        elif self.feature_type == 'dual':
            features = extract_dual_pipeline(audio, sr, max_len=self.max_len)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'filename': sample['filename'],
        }


def get_dataloaders(base_dir, feature_type='mel', batch_size=32, max_len=300, num_workers=4, max_samples=None):
    """train/dev/eval 데이터로더 생성"""
    from torch.utils.data import DataLoader

    datasets = {}
    loaders = {}

    for partition in ['train', 'dev', 'eval']:
        try:
            ds = ASVspoof2019LA(
                base_dir, partition=partition,
                feature_type=feature_type, max_len=max_len,
                max_samples=max_samples
            )
            datasets[partition] = ds
            loaders[partition] = DataLoader(
                ds, batch_size=batch_size,
                shuffle=(partition == 'train'),
                num_workers=num_workers,
                pin_memory=False
            )
        except FileNotFoundError as e:
            print(f"[{partition}] 스킵: {e}")

    return datasets, loaders
