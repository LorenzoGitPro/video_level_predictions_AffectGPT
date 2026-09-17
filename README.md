# [Pytorch Code for the Case of Zero-shot Inference: Multimodal Large Language Models (AffectGPT) for Ambivalence/Hesitancy Recognition in Videos](https://arxiv.org/pdf/2604.11730)

[![arXiv](https://img.shields.io/badge/arXiv-2604.11730-b31b1b.svg?logo=arxiv&logoColor=B31B1B)](https://arxiv.org/pdf/2604.11730)
[![Github](https://img.shields.io/badge/Github-ah--digital--health--interventions-brightgreen.svg?logo=github)](https://github.com/sbelharbi/ah-digital-health-interventions)
[![Poster](https://img.shields.io/badge/Poster-orange)](https://sbelharbi.github.io/publications/posters/Multimodal-Ambivalence-Hesitancy-Recognition-in-Videos-for-Personalized-Digital-Health-Interventions-Poster-ACII-2026.pdf)

## Overview

This repository provides the Pytorch code used to evaluate a multimodal large language model (MLLM), **AffectGPT**, in a zero-shot setting for **Ambivalence/Hesitancy (AH) recognition in videos**, on the **BAH dataset** for digital behavioural change. Rather than training a task-specific classifier, we prompt AffectGPT — which combines a large language model backbone with visual and audio encoders — directly on BAH videos to assess how well an off-the-shelf, general-purpose emotion/affect MLLM can recognize ambivalence and hesitancy without any BAH-specific fine-tuning.

The pipeline covers preparing the BAH data for AffectGPT's expected input format, running zero-shot inference over the videos, and computing the resulting classification results, so that they can be compared against dedicated BAH baselines and used in downstream analyses of digital health intervention systems.

This code builds on top of the code released by the authors of **AffectGPT** ([https://github.com/zeroQiaoba/AffectGPT](https://github.com/zeroQiaoba/AffectGPT)), adapted here for zero-shot inference on the BAH dataset.

## Setup & Usage

### 1. Download Models

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir models/Qwen2.5-7B-Instruct

huggingface-cli download openai/clip-vit-large-patch14 \
  --local-dir models/clip-vit-large-patch14

huggingface-cli download TencentGameMate/chinese-hubert-large \
  --local-dir models/chinese-hubert-large
```

#### Checkpoint

```bash
huggingface-cli download MERChallenge/AffectGPT \
  emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz_20250110100/checkpoint_000060_loss_0.480.pth \
  --local-dir .
```

### 2. Create Virtual Environment

```bash
conda env create -f environment.yml
```

### 3. Prepare Data

```bash
python prepare_bah.py --raw_dir /path/to/BAH_DB_public_access
```

### 4. Run Experiment

```bash
/bin/bash xp_bah.sh
```

Results are saved to `output/results-BAH/results.txt`


### Citation

If you use this repository, code, or experimental results in your research, please cite the following works.

#### AffectGPT

This project builds upon **AffectGPT**, a multimodal large language model for emotion understanding.

```bibtex
@inproceedings{lian25,
  title        = {AffectGPT: A New Dataset, Model, and Benchmark for Emotion Understanding
                  with Multimodal Large Language Models},
  author       = {Z. Lian and H. Chen and L. Chen and H. Sun and
                  L. Sun and Y. Ren and Z. Cheng and B. Liu and
                  R. Liu and X. Peng and J. Yi and J. Tao},
  booktitle    = {ICML},
  year         = {2025}
}
```

#### BAH Dataset

The experiments use the **BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behavioural Change**.

```bibtex
@inproceedings{gonzalez-25-bah,
  title={{BAH} Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behavioural Change},
  author={González-González, M. and Belharbi, S. and Zeeshan, M. O. and
    Sharafi, M. and Aslam, M. H and Pedersoli, M. and Koerich, A. L. and
    Bacon, S. L. and Granger, E.},
  booktitle={ICLR},
  year={2026}
}
```

#### Related Work

The experiments and results in this repository contribute to the following work on multimodal ambivalence/hesitancy recognition:

```bibtex
@inproceedings{gonzalez-26-ah-digital,
  title={Multimodal Ambivalence/Hesitancy Recognition in Videos for Personalized Digital Health Interventions},
  author={González-González, M. and  Belharbi, S. and Zeeshan, M.O. and
    Sharafi, M. and Aslam, M.H. and Sia, L. and Richet, N. and Pedersoli, M. and
    Koerich, A.L. and Bacon, S.L. and Granger, E.},
  booktitle={Conference on Affective Computing and Intelligent Interaction (ACII)},
  year={2026}
}
```

### Acknowledgments

This work was supported in part by the Fonds de recherche du Québec – Santé, Natural Sciences and Engineering Research Council of Canada, Canada Foundation for Innovation, and Digital Research Alliance of Canada.

We also thank the authors of [AffectGPT](https://github.com/zeroQiaoba/AffectGPT) for making their code publicly available.
