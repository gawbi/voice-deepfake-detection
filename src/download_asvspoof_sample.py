"""ASVspoof 2019 LA 샘플 다운로드 (Kaggle 또는 직접 다운로드)"""
import os
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "datasets" / "asvspoof_samples"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

# ASVspoof 2019 공식 사이트의 프로토콜 파일은 접근 가능
# 실제 오디오는 등록이 필요하므로, HuggingFace의 다른 경로 시도

def try_huggingface_direct():
    """HuggingFace Hub에서 직접 파일 다운로드 시도"""
    try:
        from huggingface_hub import hf_hub_download

        # 다른 ASVspoof 관련 데이터셋 시도
        print("HuggingFace Hub에서 ASVspoof 샘플 다운로드 시도...")

        # ASVspoof5 또는 관련 데이터셋에서 샘플 가져오기
        file_path = hf_hub_download(
            repo_id="LanceaKing/asvspoof2019",
            filename="LA/ASVspoof2019_LA_train/flac/LA_T_1000137.flac",
            repo_type="dataset",
            local_dir=str(SAMPLE_DIR),
        )
        print(f"다운로드 성공: {file_path}")
        return True
    except Exception as e:
        print(f"HuggingFace Hub 다운로드 실패: {e}")
        return False


def try_alternative_audio():
    """LibriSpeech 등 대체 음성 데이터로 파이프라인 검증"""
    try:
        import torchaudio
        print("torchaudio의 내장 데이터셋으로 음성 샘플 확보...")

        # YESNO 데이터셋 (작은 히브리어 음성 데이터)
        dataset = torchaudio.datasets.YESNO(
            root=str(SAMPLE_DIR),
            download=True
        )

        print(f"YESNO 데이터셋 로드 완료: {len(dataset)} 샘플")

        # 첫 3개 샘플 저장
        for i in range(min(3, len(dataset))):
            waveform, sample_rate, labels = dataset[i]
            output_path = SAMPLE_DIR / f"real_voice_{i}.wav"
            torchaudio.save(str(output_path), waveform, sample_rate)
            print(f"  저장: {output_path} (sr={sample_rate})")

        return True
    except Exception as e:
        print(f"torchaudio 데이터셋 실패: {e}")
        return False


def generate_tts_like_audio():
    """TTS 특성을 모사한 음성 생성 (딥페이크 시뮬레이션)"""
    import numpy as np
    import soundfile as sf

    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))

    print("TTS 특성 모사 음성 생성 중...")

    for i in range(3):
        f0 = 120 + i * 40

        # TTS 특성: 매우 깨끗한 하모닉 + 규칙적 프레임 구조 + 보코더 아티팩트
        harmonics = sum(
            (0.5 / (k+1)) * np.sin(2 * np.pi * f0 * (k+1) * t)
            for k in range(8)
        )

        # 보코더 아티팩트: 고주파 영역의 주기적 패턴
        vocoder_artifact = 0.03 * np.sin(2 * np.pi * 7800 * t) * np.sin(2 * np.pi * 50 * t)

        # 프레임 경계 아티팩트 (256 샘플 = 16ms 프레임)
        frame_boundary = np.zeros_like(t)
        for j in range(0, len(t), 256):
            if j < len(t):
                frame_boundary[j:min(j+8, len(t))] = 0.02

        # TTS 특유의 균일한 에너지 분포
        envelope = 0.8 + 0.2 * np.sin(2 * np.pi * 2 * t)

        signal = (harmonics + vocoder_artifact + frame_boundary) * envelope
        signal = signal / np.max(np.abs(signal)) * 0.9

        output_path = SAMPLE_DIR / f"tts_spoof_{i}.wav"
        sf.write(str(output_path), signal.astype(np.float32), sr)
        print(f"  저장: {output_path}")

    return True


if __name__ == "__main__":
    print("=" * 50)
    print("ASVspoof 샘플 확보")
    print("=" * 50)

    success = try_huggingface_direct()

    if not success:
        print("\n--- 대체 방법: 실제 음성 데이터 확보 ---")
        success = try_alternative_audio()

    print("\n--- TTS 모사 음성 생성 ---")
    generate_tts_like_audio()

    print("\n확보된 파일:")
    for f in sorted(SAMPLE_DIR.rglob("*.wav")) :
        print(f"  {f}")
    for f in sorted(SAMPLE_DIR.rglob("*.flac")):
        print(f"  {f}")
