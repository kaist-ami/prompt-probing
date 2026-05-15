# Zero-Shot Rankability: Revealing Latent Ordinal Structure in Multimodal Large Language Models via Language

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/ICML-2026-blue.svg" alt="ICML 2026">
  <img src="https://img.shields.io/badge/python-3.9%2B-brightgreen.svg" alt="Python 3.9+">
</p>

> **ICML 2026** | [Paper](https://arxiv.org/abs/XXXX.XXXXX) | [Project Page](https://kaist-ami.github.io/prompt-probing)

Official implementation of the paper:

**"Zero-Shot Rankability: Revealing Latent Ordinal Structure in Multimodal Large Language Models via Language"**

---

## Abstract

We investigate *zero-shot rankability* — the ability of Multimodal Large Language Models (MLLMs) to reveal latent ordinal structure across diverse attributes purely through language prompting, without any task-specific fine-tuning. We introduce **Prompt Probing**, a framework that systematically constructs linguistically grounded ordinal prompts and measures how well MLLMs align with ground-truth rankings across visual, semantic, and compositional axes. Our analysis reveals that ordinal structure is latently present in MLLMs and can be elicited with the right prompt design, providing insight into the nature of multimodal representations.

---

## Installation

```bash
git clone https://github.com/kaist-ami/prompt-probing.git
cd prompt-probing
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA ≥ 11.8 (recommended for GPU inference)

---

## Repository Structure

```
prompt-probing/
├── src/
│   └── prompt_probing/
│       ├── __init__.py
│       ├── models/              # MLLM wrappers (LLaVA, InstructBLIP, GPT-4V, etc.)
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── llava.py
│       │   ├── instructblip.py
│       │   └── gpt4v.py
│       ├── prompts/             # Prompt construction & templates
│       │   ├── __init__.py
│       │   ├── builder.py
│       │   └── templates.py
│       ├── ranking/             # Rankability scoring & evaluation
│       │   ├── __init__.py
│       │   ├── scorer.py
│       │   └── metrics.py
│       └── data/                # Dataset loading & preprocessing
│           ├── __init__.py
│           └── datasets.py
├── scripts/
│   ├── run_probe.py             # Main probing script
│   ├── evaluate.py              # Evaluation script
│   └── visualize.py             # Result visualization
├── configs/
│   ├── default.yaml
│   └── models/
│       ├── llava.yaml
│       ├── instructblip.yaml
│       └── gpt4v.yaml
├── tests/
│   ├── test_prompts.py
│   ├── test_ranking.py
│   └── test_models.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

### 1. Prepare Data

Download and organise the benchmark datasets under `data/`:

```bash
# THINGS: https://things-initiative.org/
# Download images and norms, then place under data/things/images/ and data/things/norms/

# WinoGround: https://huggingface.co/datasets/facebook/winoground
# Download and place data.jsonl and images/ under data/winoground/
```

Supported datasets:

| Dataset | Task | Attribute |
|---------|------|-----------|
| [THINGS](https://things-initiative.org/) | Object ranking | Size, Weight, Familiarity |
| [WinoGround](https://huggingface.co/datasets/facebook/winoground) | Compositional ranking | Semantic order |
| [ImageNet-R](https://github.com/hendrycks/imagenet-r) | Perceptual ranking | Visual similarity |
| Custom benchmark | Multi-attribute | Various ordinal scales |

### 2. Run Prompt Probing

```bash
python scripts/run_probe.py \
    --model llava-1.5-7b \
    --dataset things \
    --attribute size \
    --prompt-type ordinal \
    --output results/llava_things_size.json
```

### 3. Evaluate Rankability

```bash
python scripts/evaluate.py \
    --results results/llava_things_size.json \
    --metric kendall_tau spearman_rho \
    --output results/eval_summary.json
```

### 4. Visualize Results

```bash
python scripts/visualize.py \
    --results results/ \
    --output figures/
```

---

## Prompt Probing Framework

Our framework probes MLLMs with three families of ordinal prompts:

### Prompt Types

| Type | Description | Example |
|------|-------------|---------|
| `direct` | Direct comparison query | "Which is larger, A or B?" |
| `ordinal` | Scale-grounded query | "On a scale of 1–10, how large is this object?" |
| `contrastive` | Paired contrastive query | "Rank these items from smallest to largest." |
| `chain` | Chain-of-thought guided | "Think step by step about the relative size..." |

### Rankability Score

We define the **Rankability Score (RS)** as the average rank correlation between model-predicted orderings and ground-truth ordinal labels:

```
RS = (1/N) * Σ τ(π_model, π_gt)
```

where `τ` is Kendall's tau and `π` denotes a ranking permutation.

---

## Models Supported

| Model | Provider | Access |
|-------|----------|--------|
| LLaVA-1.5 (7B, 13B) | Open-source | HuggingFace |
| InstructBLIP (Vicuna-7B, Vicuna-13B) | Open-source | HuggingFace |
| GPT-4V | OpenAI | API key required |
| Gemini Pro Vision | Google | API key required |
| Claude 3 (Opus, Sonnet) | Anthropic | API key required |

---

## Reproducing Paper Results

```bash
# Reproduce Table 1 (main rankability results — run per model/dataset/attribute)
python scripts/run_probe.py --model llava-1.5-7b --dataset things \
    --data-root data/things --attribute size \
    --prompt-type ordinal --output results/llava_things_size.json

python scripts/evaluate.py --results results/llava_things_size.json \
    --metric kendall_tau spearman_rho

# Reproduce prompt-type ablation (run for each --prompt-type value)
for PT in direct ordinal contrastive chain; do
  python scripts/run_probe.py --model llava-1.5-7b --dataset things \
      --data-root data/things --attribute size \
      --prompt-type "${PT}" --output "results/llava_things_size_${PT}.json"
done
python scripts/visualize.py --results results/ --output figures/
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{promptprobing2026,
  title     = {Zero-Shot Rankability: Revealing Latent Ordinal Structure in Multimodal Large Language Models via Language},
  author    = {},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

We thank the open-source community for providing foundation models and datasets that made this research possible.