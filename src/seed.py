"""난수 시드 고정 유틸.

실험 재현성을 위해 `random` / `numpy` / `torch` 의 시드를 한 번에 고정한다.

사용 예:
    from src.seed import set_seed
    set_seed()          # 기본값 42
    set_seed(1234)

기존 코드와의 관계
------------------
`src/preprocessing/dataset.py` 는 `max_samples` 서브셋을 뽑을 때 이미
`random.seed(42)` 를 호출한다. 본 모듈의 기본 시드를 42로 맞춘 이유가 그것이며,
이 모듈은 기존 동작을 덮어쓰지 않고 **추가로** 고정 범위를 넓히는 역할만 한다.
(dataset.py 의 서브셋 구성은 이 모듈을 호출하든 안 하든 동일하게 유지된다.)

주의
----
`deterministic=True` 는 cuDNN 의 비결정적 커널 선택을 끄기 때문에 GPU 학습
속도가 느려질 수 있다. 성능이 중요하면 `set_seed(42, deterministic=False)` 를
쓴다. MPS 백엔드는 완전한 결정성을 보장하지 않는다.
"""

from __future__ import annotations

import os
import random

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """random / numpy / torch 시드를 한 번에 고정하고 사용한 시드를 반환한다."""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    try:
        import numpy as np
    except ImportError:  # numpy 없이 쓰이는 일은 없지만 방어적으로
        pass
    else:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seed


if __name__ == "__main__":
    print(f"seed fixed: {set_seed()}")
