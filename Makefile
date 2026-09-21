# 음성 딥페이크 탐지 — 재현용 Makefile
#
#   make setup      의존성 설치
#   make lint       문법 검사
#   make smoke      데이터셋 없이 파이프라인 동작 확인 (합성 오디오)
#   make reproduce  README 표의 dev EER 재현 (ASVspoof 2019 LA 데이터 필요)
#   make clean      캐시·산출물 정리
#
# ASVspoof 2019 LA 는 라이선스·용량 문제로 저장소에 포함되지 않는다. DATA.md 참고.

PYTHON      ?= python3
PIP         ?= $(PYTHON) -m pip
DATA_DIR    ?= datasets/asvspoof2019_LA/LA
MAX_SAMPLES ?= 5000
EPOCHS      ?= 10
LR          ?= 1e-4
IMAGE       ?= voice-deepfake:cpu

export PYTHONPATH := $(CURDIR)

.DEFAULT_GOAL := help
.PHONY: help setup lint smoke check-data reproduce train-baseline train-dual eval pretest clean docker-build docker-shell

help:
	@echo "타깃 목록:"
	@echo "  setup           requirements.txt 설치 (CPU torch 휠)"
	@echo "  lint            src/ 전체 문법 검사 (+ruff 있으면 함께)"
	@echo "  smoke           합성 오디오로 특징추출→모델 forward 확인 (데이터셋 불필요, 수 초)"
	@echo "  reproduce       멜 단일 → 멜+웨이블릿 이중 학습 후 dev EER 비교 (데이터셋 필요)"
	@echo "  train-baseline  멜 단일 + ResNet-18 학습"
	@echo "  train-dual      멜+웨이블릿 이중 + ResNet-18 학습"
	@echo "  eval            저장된 체크포인트로 EER 재측정"
	@echo "  pretest         웨이블릿 사전 실험 재현 (네트워크로 YESNO 샘플 확보)"
	@echo "  clean           __pycache__ · 실험 산출물 정리"
	@echo "  docker-build    CPU 이미지 빌드"
	@echo "  docker-shell    데이터 볼륨을 마운트한 셸 진입"

setup:
	$(PIP) install -r requirements.txt \
		--index-url https://download.pytorch.org/whl/cpu \
		--extra-index-url https://pypi.org/simple

# 문법 오류 · 정의되지 않은 이름만 잡는다(스타일 규칙은 기존 코드를 건드리지 않기 위해 제외).
lint:
	$(PYTHON) -m compileall -q src
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check --select E9,F63,F7,F82 src; \
	else \
		echo "(ruff 미설치 — compileall 문법 검사만 수행)"; \
	fi

# 데이터셋 없이 돌아가는 유일한 검증 경로. CI/설치 확인용.
smoke:
	@$(PYTHON) -c "import numpy as np, torch; \
from src.seed import set_seed; \
from src.preprocessing.audio_features import extract_mel_spectrogram, extract_dual_pipeline; \
from src.models.resnet_detector import ResNetDetector; \
set_seed(); \
audio = np.random.RandomState(0).randn(16000 * 2).astype('float32') * 0.1; \
mel = extract_mel_spectrogram(audio, max_len=300); \
dual = extract_dual_pipeline(audio, max_len=200); \
print('mel  :', mel.shape); \
print('dual :', dual.shape); \
m = ResNetDetector(in_channels=2); \
out = m(torch.from_numpy(dual).float().unsqueeze(0)); \
print('logits:', tuple(out.shape)); \
print('SMOKE OK')"

check-data:
	@test -d "$(DATA_DIR)" || { \
		echo "ERROR: 데이터셋이 없습니다 -> $(DATA_DIR)"; \
		echo "ASVspoof 2019 LA 내려받기·배치 방법은 DATA.md 를 참고하세요."; \
		exit 1; }

# README '실험 결과' 1절 표(dev EER 5.50% vs 3.80%)의 재현 경로.
# 5,000 utterance 서브셋 · 10 epoch 기준.
reproduce: check-data train-baseline train-dual
	@echo "완료. checkpoints/ 의 best_mel_resnet.pt · best_dual_resnet.pt 에 dev EER 이 기록됩니다."
	@echo "저장된 EER 확인: make eval"

train-baseline: check-data
	$(PYTHON) src/training/train.py --data_dir "$(DATA_DIR)" \
		--feature mel --model resnet --epochs $(EPOCHS) --batch_size 32 --lr $(LR) \
		--max_samples $(MAX_SAMPLES)

train-dual: check-data
	$(PYTHON) src/training/train.py --data_dir "$(DATA_DIR)" \
		--feature dual --model resnet --epochs $(EPOCHS) --batch_size 16 --lr $(LR) \
		--max_samples $(MAX_SAMPLES)

eval:
	$(PYTHON) src/training/eval_only.py

# 네트워크로 YESNO 샘플을 받아 웨이블릿 사전 실험을 재현한다 (ASVspoof 불필요).
pretest:
	$(PYTHON) src/download_asvspoof_sample.py
	$(PYTHON) src/run_real_experiment.py

docker-build:
	docker build -t $(IMAGE) .

docker-shell:
	docker run --rm -it \
		-v "$(CURDIR)/datasets:/app/datasets" \
		-v "$(CURDIR)/checkpoints:/app/checkpoints" \
		$(IMAGE) bash

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
	rm -rf .pytest_cache .ruff_cache src/experiment_results
