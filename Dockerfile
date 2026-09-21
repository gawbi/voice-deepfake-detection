# 음성 딥페이크 탐지 — CPU 재현 환경
#
# 빌드:
#   docker build -t voice-deepfake:cpu .
#
# 실행 (데이터셋·체크포인트는 이미지에 넣지 않고 볼륨으로 마운트):
#   docker run --rm -it \
#     -v "$PWD/datasets:/app/datasets" \
#     -v "$PWD/checkpoints:/app/checkpoints" \
#     voice-deepfake:cpu bash
#
# ASVspoof 2019 LA 데이터 배치 방법은 DATA.md 참고.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# libsndfile1 : soundfile 이 flac/wav 를 읽는 데 필요한 네이티브 라이브러리
# ffmpeg      : librosa 의 audioread 폴백 경로용 디코더
# make        : Makefile 타깃을 컨테이너 안에서도 쓰기 위함
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        make \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 레이어를 소스보다 먼저 캐싱
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple

# 소스만 복사. datasets/ · checkpoints/ 는 .gitignore 대상이며 런타임 볼륨으로 붙인다.
COPY src/ ./src/
COPY run_experiment.sh Makefile ./

# 데이터·산출물 마운트 지점
VOLUME ["/app/datasets", "/app/checkpoints"]

CMD ["bash"]
