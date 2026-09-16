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
