"""
'내 목소리 맞나요?' 전체 파이프라인 아키텍처 시각화
- 이중 파이프라인 음성 딥페이크 탐지
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm
from pathlib import Path
import numpy as np

# 맥 한글 폰트 설정
font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

def draw_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(22, 14))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # 색상 정의
    C_INPUT = '#E3F2FD'      # 입력 (연한 파랑)
    C_PREPROCESS = '#FFF3E0'  # 전처리 (연한 주황)
    C_ENCODER1 = '#E8F5E9'   # 멜 인코더 (연한 초록)
    C_ENCODER2 = '#F3E5F5'   # 웨이블릿 인코더 (연한 보라)
    C_FUSION = '#FFEBEE'     # 융합 (연한 빨강)
    C_MODEL = '#FFF9C4'      # 모델 (연한 노랑)
    C_LLM = '#E0F7FA'        # LLM (연한 시안)
    C_OUTPUT = '#F1F8E9'     # 출력 (연한 라임)
    C_PERF = '#ECEFF1'       # 성능 비교 (회색)

    def draw_box(x, y, w, h, text, color, fontsize=11, bold=False, subtext=None):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor='#333333', linewidth=1.5)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2 + (0.15 if subtext else 0), text,
                ha='center', va='center', fontsize=fontsize, fontweight=weight)
        if subtext:
            ax.text(x + w/2, y + h/2 - 0.25, subtext,
                    ha='center', va='center', fontsize=8, color='#666666', style='italic')

    def draw_arrow(x1, y1, x2, y2, color='#333333'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))

    # ==========================================
    # 제목
    # ==========================================
    ax.text(11, 13.3, '"내 목소리 맞나요?" — 전체 파이프라인 아키텍처',
            ha='center', va='center', fontsize=18, fontweight='bold')
    ax.text(11, 12.8, 'Dual-Encoder Pipeline: Mel Spectrogram + Wavelet Analysis → Deepfake Detection',
            ha='center', va='center', fontsize=11, color='#666666')

    # ==========================================
    # 메인 파이프라인 (왼쪽 → 오른쪽)
    # ==========================================

    # 1. 입력
    draw_box(0.5, 6.5, 2.5, 1.5, '음성 입력', C_INPUT, fontsize=13, bold=True,
             subtext='카톡 음성메모 / 녹음 파일')

    draw_arrow(3.0, 7.25, 3.8, 7.25)

    # 2. 전처리
    draw_box(3.8, 6.5, 2.2, 1.5, '전처리', C_PREPROCESS, fontsize=12, bold=True,
             subtext='16kHz 리샘플링 + 정규화')

    draw_arrow(6.0, 7.25, 6.8, 8.5)   # 상단 (멜)
    draw_arrow(6.0, 7.25, 6.8, 6.0)   # 하단 (웨이블릿)

    # 3. 이중 인코더 (핵심 — 이중 특징 추출)
    # 점선 박스로 이중 인코더 묶기
    dual_box = FancyBboxPatch((6.5, 4.8), 4.5, 5.2, boxstyle="round,pad=0.2",
                               facecolor='#FAFAFA', edgecolor='#1565C0',
                               linewidth=2, linestyle='--')
    ax.add_patch(dual_box)
    ax.text(8.75, 9.7, 'Dual Encoder (이중 특징 추출)',
            ha='center', va='center', fontsize=10, fontweight='bold', color='#1565C0')

    # 3a. 멜 스펙트로그램 인코더
    draw_box(6.8, 8.0, 3.8, 1.2, '멜 스펙트로그램 인코더', C_ENCODER1, fontsize=11, bold=True,
             subtext='주파수 영역 미세 구조 추출 (n_mels=128)')

    # 3b. 웨이블릿 인코더
    draw_box(6.8, 5.3, 3.8, 1.2, '웨이블릿 인코더 (CWT)', C_ENCODER2, fontsize=11, bold=True,
             subtext='시간-주파수 영역 배경 구조 추출 (Morlet)')

    # 4. 특징 융합 (2ch concat)
    draw_arrow(10.6, 8.6, 11.5, 7.5)
    draw_arrow(10.6, 5.9, 11.5, 7.0)

    draw_box(11.3, 6.5, 1.8, 1.5, '특징 융합\n(2ch)', C_FUSION, fontsize=12, bold=True)

    draw_arrow(13.1, 7.25, 13.8, 7.25)

    # 5. CNN 판별
    draw_box(13.8, 6.3, 2.5, 1.9, 'CNN 판별기', C_MODEL, fontsize=13, bold=True,
             subtext='ResNet-18')

    draw_arrow(16.3, 7.25, 17.0, 7.25)

    # 6. LLM 해석
    draw_box(17.0, 6.3, 2.3, 1.9, 'LLM 해석', C_LLM, fontsize=13, bold=True,
             subtext='Claude API 자연어 설명')

    draw_arrow(19.3, 7.25, 19.8, 7.25)

    # 7. 결과 제공
    draw_box(19.8, 6.2, 2.0, 2.1, '결과\n제공', C_OUTPUT, fontsize=12, bold=True,
             subtext='Gradio UI')

    # ==========================================
    # 결과 상세 (하단)
    # ==========================================
    draw_arrow(20.8, 6.2, 20.8, 5.0)
    draw_arrow(20.8, 5.0, 18.5, 4.3)
    draw_arrow(20.8, 5.0, 20.8, 4.3)

    # 결과 항목들
    draw_box(16.5, 3.5, 2.8, 1.0, '판별 결과 + 신뢰도', '#E8EAF6', fontsize=10,
             subtext='"87% 확률로 AI 생성"')
    draw_box(19.5, 3.5, 2.3, 1.0, '스펙트로그램\n시각 비교', '#E8EAF6', fontsize=10)

    # ==========================================
    # 하단: 성능 요약
    # ==========================================
    ref_box = FancyBboxPatch((0.5, 1.0), 10.0, 2.5, boxstyle="round,pad=0.2",
                              facecolor='#F5F5F5', edgecolor='#78909C',
                              linewidth=1.5, linestyle='-.')
    ax.add_patch(ref_box)
    ax.text(5.5, 3.1, '실측 성능 비교',
            ha='center', va='center', fontsize=11, fontweight='bold', color='#546E7A')

    draw_box(1.0, 1.5, 3.5, 1.0, '멜 단일 (Baseline)', '#BBDEFB', fontsize=10,
             subtext='Dev EER 5.50% (5k subset)')
    draw_box(5.0, 1.5, 3.5, 1.0, '이중 파이프라인 (Ours)', '#C8E6C9', fontsize=10,
             subtext='Dev EER 3.80% (5k) — 31% 개선')
    draw_box(9.0, 1.5, 1.2, 1.0, '31%↑', '#C8E6C9', fontsize=10, bold=True)

    # ==========================================
    # 범례
    # ==========================================
    legend_items = [
        mpatches.Patch(facecolor=C_ENCODER1, edgecolor='#333', label='주파수 영역 (멜)'),
        mpatches.Patch(facecolor=C_ENCODER2, edgecolor='#333', label='시간-주파수 영역 (웨이블릿)'),
        mpatches.Patch(facecolor=C_FUSION, edgecolor='#333', label='특징 융합 (2ch concat)'),
        mpatches.Patch(facecolor='#F5F5F5', edgecolor='#78909C', label='실측 성능 비교'),
    ]
    ax.legend(handles=legend_items, loc='upper right', fontsize=10,
              framealpha=0.9, edgecolor='#cccccc')

    # 저장
    output_path = str(Path(__file__).resolve().parents[1] / 'docs' / 'images' / 'pipeline_architecture.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"파이프라인 아키텍처 저장: {output_path}")

    # 고해상도 버전 (발표용)
    output_path_hd = str(Path(__file__).resolve().parents[1] / 'docs' / 'images' / 'pipeline_architecture_HD.png')
    fig2, ax2 = plt.subplots(1, 1, figsize=(22, 14))
    ax2.set_xlim(0, 22)
    ax2.set_ylim(0, 14)
    ax2.axis('off')
    plt.close()

    return output_path


if __name__ == '__main__':
    path = draw_pipeline()
    print(f"\n완료! 파일: {path}")
