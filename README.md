# Setup & Usage

## 1. Download Models

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir models/Qwen2.5-7B-Instruct

huggingface-cli download openai/clip-vit-large-patch14 \
  --local-dir models/clip-vit-large-patch14

huggingface-cli download TencentGameMate/chinese-hubert-large \
  --local-dir models/chinese-hubert-large
```

### Checkpoint

```bash
huggingface-cli download MERChallenge/AffectGPT \
  emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz_20250110100/checkpoint_000060_loss_0.480.pth \
  --local-dir .
```

## 2. Create Virtual Environment

```bash
conda env create -f environment.yml
```

## 3. Prepare Data

```bash
python prepare_bah.py --raw_dir /path/to/BAH_DB_public_access
```

## 4. Run Experiment

```bash
/bin/bash xp_bah.sh
```

Results are saved to `output/results-BAH/results.txt`


## Citation

If you use this repository, code, or experimental results in your research, please cite the following works.

### AffectGPT

This project builds upon **AffectGPT**, a multimodal large language model for emotion understanding.

```bibtex
@misc{lian2025affectgptnewdatasetmodel,
      title={AffectGPT: A New Dataset, Model, and Benchmark for Emotion Understanding with Multimodal Large Language Models},
      author={Zheng Lian and Haoyu Chen and Lan Chen and Haiyang Sun and Licai Sun and Yong Ren and Zebang Cheng and Bin Liu and Rui Liu and Xiaojiang Peng and Jiangyan Yi and Jianhua Tao},
      year={2025},
      eprint={2501.16566},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2501.16566}
}
```

### BAH Dataset

The experiments use the **BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behavioural Change**.

```bibtex
@misc{gonzálezgonzález2026bahdatasetambivalencehesitancyrecognition,
      title={BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behavioural Change},
      author={Manuela González-González and Soufiane Belharbi and Muhammad Osama Zeeshan and Masoumeh Sharafi and Muhammad Haseeb Aslam and Marco Pedersoli and Alessandro Lameiras Koerich and Simon L Bacon and Eric Granger},
      year={2026},
      eprint={2505.19328},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.19328}
}
```

### Related Work

The experiments and results in this repository contribute to the following work on multimodal ambivalence/hesitancy recognition:

```bibtex
@misc{gonzálezgonzález2026multimodalambivalencehesitancyrecognitionvideos,
      title={Multimodal Ambivalence/Hesitancy Recognition in Videos for Personalized Digital Health Interventions},
      author={Manuela González-González and Soufiane Belharbi and Muhammad Osama Zeeshan and Masoumeh Sharafi and Muhammad Haseeb Aslam and Lorenzo Sia and Nicolas Richet and Marco Pedersoli and Alessandro Lameiras Koerich and Simon L Bacon and Eric Granger},
      year={2026},
      eprint={2604.11730},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.11730}
}
```
