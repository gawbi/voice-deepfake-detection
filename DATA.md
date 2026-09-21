# 데이터 준비

이 저장소에는 **오디오 데이터와 학습 체크포인트가 포함되어 있지 않습니다.**
아래 절차대로 내려받아 배치하면 README의 실험을 재현할 수 있습니다.

---

## 왜 포함하지 않았나

| 이유 | 내용 |
|---|---|
| **용량** | ASVspoof 2019 LA 전체는 수 GB 규모입니다. git 저장소에 담기 부적절합니다. |
| **라이선스** | ASVspoof 2019 데이터셋은 **ODC-By** 라이선스로 배포됩니다. 재배포 시 원 출처 표기 의무가 따르므로, 본 저장소는 원본을 복제하지 않고 **공식 배포처를 안내**하는 방식을 택했습니다. |
| **재생성 가능** | 체크포인트(`checkpoints/*.pt`)는 데이터만 있으면 `make reproduce` 로 다시 만들 수 있는 산출물입니다. |

`.gitignore` 에서 `datasets/`, `checkpoints/`, `*.flac`, `*.wav`, `*.pt` 를 제외하고 있습니다.

---

## 1. ASVspoof 2019 LA (주 데이터셋)

- **공식 배포처**: University of Edinburgh DataShare — <https://datashare.ed.ac.uk/handle/10283/3336>
  (ASVspoof 2019 전체 데이터셋. 이 페이지에서 `LA.zip` 을 내려받습니다. PA(Physical Access)는 본 프로젝트에서 쓰지 않습니다.)
- **라이선스**: ODC-By (Open Data Commons Attribution License)
- **원 논문**: Todisco, M. et al., *ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection*, INTERSPEECH 2019
- **용량**: `LA.zip` 수 GB 규모 — 정확한 바이트 수는 배포 페이지 표기를 확인하세요. **(확인 필요)**

### 배치 위치

내려받은 `LA.zip` 을 아래 경로에 둡니다.

```
datasets/asvspoof2019_LA/LA.zip
```

`run_experiment.sh` 또는 `make reproduce` 전에 압축을 풀면 다음 구조가 되어야 합니다.
(경로 문자열은 `src/preprocessing/dataset.py` 의 `partition_map` 이 그대로 기대하는 값입니다.)

```
datasets/asvspoof2019_LA/LA/
├── ASVspoof2019_LA_train/flac/          # 학습 파티션 오디오 (.flac)
├── ASVspoof2019_LA_dev/flac/            # dev 파티션 오디오 — README의 EER 산출 기준
├── ASVspoof2019_LA_eval/flac/           # eval 파티션 오디오
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

압축 해제:

```bash
cd datasets/asvspoof2019_LA && unzip -q LA.zip
```

> 배포본에 따라 압축을 풀었을 때 한 단계 더 깊은 디렉터리가 생길 수 있습니다.
> 위 구조와 다르면 `LA/` 디렉터리를 위 경로에 맞춰 옮기거나,
> `make reproduce DATA_DIR=<실제 LA 경로>` 로 경로를 넘기면 됩니다.

### 프로토콜 파일 형식

`src/preprocessing/dataset.py` 의 `_parse_protocol()` 은 공백 구분 5열 이상을 가정합니다.

```
SPEAKER_ID  FILENAME  -  ATTACK_TYPE  LABEL
```

`LABEL` 이 `bonafide` 면 0(진짜), 그 외(`spoof`)면 1(가짜)로 매핑됩니다.

---

## 2. 사전 실험용 샘플 오디오 (선택)

README "사전 실험 — 웨이블릿 전처리의 음성 도메인 유효성" 절은 ASVspoof 없이도 재현됩니다.

```bash
make pretest
```

- 진짜 음성: **YESNO** 데이터셋 (`torchaudio.datasets.YESNO` 로 자동 다운로드)
- 가짜 음성: 보코더·프레임 경계 아티팩트를 모사해 **스크립트가 직접 합성**합니다. 별도 다운로드 없음.
- 네트워크 연결이 필요하며, `torchaudio` 가 설치되어 있어야 합니다
  (`requirements.txt` 의 선택 의존성 주석 참고 — `pip install torchaudio`).

> 이 사전 실험은 실제 ASVspoof 스푸핑 음성이 아니라 TTS 특성을 *모사한* 합성 음성을 쓰는
> 정성 확인 수준입니다. 정량 결과는 1번 데이터셋 기준입니다.

---

## 3. 데이터 없이 확인할 수 있는 것

데이터를 받기 전에 설치와 파이프라인 연결만 검증하려면:

```bash
make smoke
```

합성 난수 오디오 2초로 멜 스펙트로그램 → 웨이블릿 스칼로그램 → 2채널 스택 →
ResNet-18 forward 까지 통과하는지 확인합니다. (수 초, 다운로드 불필요)

---

## 4. 산출물이 저장되는 위치

| 경로 | 내용 | git 포함 |
|---|---|---|
| `datasets/` | 원본 오디오 · 프로토콜 | ✗ |
| `checkpoints/` | `best_mel_resnet.pt`, `best_dual_resnet.pt`, `best_mel_light.pt` (dev EER 최저 시점) | ✗ |
| `src/experiment_results/` | 사전 실험 중간 산출물 | ✗ |
| `docs/images/` | 사전 실험 그래프 | ✓ (커밋됨) |

Docker로 실행할 때는 위 경로를 볼륨으로 마운트하세요 — 이미지 안에 데이터를 굽지 않습니다.

```bash
docker run --rm -it \
  -v "$PWD/datasets:/app/datasets" \
  -v "$PWD/checkpoints:/app/checkpoints" \
  voice-deepfake:cpu bash
```
