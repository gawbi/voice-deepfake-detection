"""
오디오 특징 추출 모듈
- 멜 스펙트로그램 (주파수 영역)
- 웨이블릿 스칼로그램 (시간-주파수 영역)
- 이중 파이프라인 결합
"""

import numpy as np
import librosa
import pywt
import torch


def extract_mel_spectrogram(audio, sr=16000, n_mels=128, n_fft=2048, hop_length=512, max_len=300):
    """멜 스펙트로그램 추출 → (1, n_mels, max_len) 텐서"""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 길이 패딩/자르기
    if mel_db.shape[1] < max_len:
        mel_db = np.pad(mel_db, ((0, 0), (0, max_len - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :max_len]

    # 정규화 [-1, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8) * 2 - 1

    return mel_db.astype(np.float32)


def extract_wavelet_scalogram(audio, sr=16000, n_scales=128, max_len=300, wavelet='morl'):
    """연속 웨이블릿 변환(CWT) 스칼로그램 → (1, n_scales, max_len) 텐서"""
    # 오디오 길이 제한 (CWT는 연산이 무거움)
    max_samples = sr * 4  # 최대 4초
    audio_clip = audio[:min(len(audio), max_samples)]

    scales = np.arange(1, n_scales + 1)
    coefficients, _ = pywt.cwt(audio_clip, scales, wavelet, sampling_period=1/sr)
    scalogram = np.abs(coefficients)

    # 길이 맞추기
    if scalogram.shape[1] < max_len:
        scalogram = np.pad(scalogram, ((0, 0), (0, max_len - scalogram.shape[1])), mode='constant')
    else:
        # 다운샘플링
        indices = np.linspace(0, scalogram.shape[1] - 1, max_len).astype(int)
        scalogram = scalogram[:, indices]

    # 정규화 [-1, 1]
    scalogram = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min() + 1e-8) * 2 - 1

    return scalogram.astype(np.float32)


def extract_wavelet_energy(audio, wavelet='db4', level=5):
    """웨이블릿 분해 에너지 벡터 → (level+1,) 배열"""
    coeffs = pywt.wavedec(audio, wavelet, level=level)
    energies = np.array([np.sum(c**2) / len(c) for c in coeffs], dtype=np.float32)
    return energies


def extract_dual_pipeline(audio, sr=16000, max_len=300):
    """이중 파이프라인: 멜 + 웨이블릿 → (2, 128, max_len) 텐서"""
    mel = extract_mel_spectrogram(audio, sr, n_mels=128, max_len=max_len)
    cwt = extract_wavelet_scalogram(audio, sr, n_scales=128, max_len=max_len)

    # 2채널 이미지로 스택
    dual = np.stack([mel, cwt], axis=0)  # (2, 128, max_len)
    return dual
