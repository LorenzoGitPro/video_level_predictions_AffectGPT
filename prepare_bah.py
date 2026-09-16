"""
BAH Dataset Processing Pipeline for AffectGPT
==============================================
Usage:
    python prepare_bah.py --raw_dir /path/to/BAH_DB_public_access

Output structure (always written to ./datasets/BAH-process/):
    datasets/BAH-process/
    ├── subvideo/          MP4 clips  (train_*.mp4, val_*.mp4, test_*.mp4)
    ├── subaudio/          WAV files  (train_*.wav, val_*.wav, test_*.wav)
    ├── openface_face/     NPY arrays (train_*.npy, val_*.npy, test_*.npy)
    ├── label.npz          Full corpus dict (train/val/test)
    ├── label.npy          Flat list of test emo labels
    └── transcription.csv
"""

import argparse
import csv
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

# ── Optional heavy import (only needed for audio extraction) ──────────────────
try:
    from moviepy import VideoFileClip
    MOVIEPY_OK = True
except ImportError:
    MOVIEPY_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SPLITS = ("train", "val", "test")

def log(msg: str, level: str = "INFO"):
    prefix = {"INFO": "✔", "WARN": "⚠", "ERROR": "✖", "STEP": "▶"}.get(level, "•")
    print(f"  {prefix}  {msg}", flush=True)


def section(title: str):
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def safe_dest(folder: Path, filename: str) -> Path:
    """Return a collision-free destination path."""
    dest = folder / filename
    if not dest.exists():
        return dest
    base, ext = os.path.splitext(filename)
    counter = 1
    while dest.exists():
        dest = folder / f"{base}_{counter}{ext}"
        counter += 1
    return dest


def split_txt_path(raw_dir: Path, split: str) -> Path:
    return raw_dir / "data" / "split" / f"{split}.txt"


def check_splits_exist(raw_dir: Path):
    missing = []
    for split in SPLITS:
        p = split_txt_path(raw_dir, split)
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("\n[ERROR] Split files not found:")
        for m in missing:
            print(f"        {m}")
        sys.exit(1)
    log("All split .txt files found")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Labels  (label.npz + label.npy)
# ─────────────────────────────────────────────────────────────────────────────

def build_corpus(txt_file: Path, prefix: str) -> dict:
    corpus = {}
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            path = parts[0]
            try:
                label = int(parts[1])
            except ValueError:
                continue
            base_name = Path(path).stem
            key = f"{prefix}_{base_name}"
            corpus[key] = {"emo": label, "val": -10}
    return corpus


def step_labels(raw_dir: Path, out_dir: Path):
    section("Step 1 / Labels")

    corpora = {}
    for split in SPLITS:
        txt = split_txt_path(raw_dir, split)
        corpora[split] = build_corpus(txt, split)
        log(f"{split}: {len(corpora[split])} entries")

    npz_path = out_dir / "label.npz"
    np.savez(str(npz_path),
             train_corpus=corpora["train"],
             val_corpus=corpora["val"],
             test_corpus=corpora["test"])
    log(f"Saved {npz_path}")

    # flat label.npy from test corpus
    emo_labels = [v["emo"] for v in corpora["test"].values()]
    npy_path = out_dir / "label.npy"
    np.save(str(npy_path), emo_labels)
    log(f"Saved {npy_path} ({len(emo_labels)} test labels, first 10: {emo_labels[:10]})")

    return corpora


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Copy MP4 clips  (subvideo/)
# ─────────────────────────────────────────────────────────────────────────────

def step_copy_videos(raw_dir: Path, out_dir: Path):
    section("Step 2 / Copy MP4 clips → subvideo/")
    root_videos = raw_dir / "data"
    video_dir = out_dir / "subvideo"
    video_dir.mkdir(parents=True, exist_ok=True)

    total, copied, missing = 0, 0, 0
    for split in ("test",): # for split in SPLITS: (for all splits, but we only need test here)
        txt = split_txt_path(raw_dir, split)
        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                path = line.strip().split(",")[0]
                if not path.lower().endswith(".mp4"):
                    continue
                total += 1
                src = root_videos / path
                if not src.exists():
                    log(f"Not found: {src}", "WARN")
                    missing += 1
                    continue
                dest = safe_dest(video_dir, f"{split}_{src.name}")
                shutil.copy2(src, dest)
                copied += 1

    log(f"Videos — total: {total}, copied: {copied}, missing: {missing}")
    if missing:
        log(f"{missing} video(s) not found (see warnings above)", "WARN")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Extract WAV audio  (subaudio/)
# ─────────────────────────────────────────────────────────────────────────────

def step_extract_audio(raw_dir: Path, out_dir: Path):
    section("Step 3 / Extract WAV audio → subaudio/")

    if not MOVIEPY_OK:
        log("moviepy not installed — skipping audio extraction. "
            "Install with: pip install moviepy", "WARN")
        return

    root_videos = raw_dir / "data"
    audio_dir = out_dir / "subaudio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    total, done, errors, missing = 0, 0, 0, 0
    for split in ("test",): # for split in SPLITS: (for all splits, but we only need test here)
        txt = split_txt_path(raw_dir, split)
        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                path = line.strip().split(",")[0]
                if not path.lower().endswith(".mp4"):
                    continue
                total += 1
                mp4_path = root_videos / path
                if not mp4_path.exists():
                    log(f"Not found: {mp4_path}", "WARN")
                    missing += 1
                    continue
                wav_name = f"{split}_{mp4_path.stem}.wav"
                wav_path = audio_dir / wav_name
                try:
                    video = VideoFileClip(str(mp4_path))
                    video.audio.write_audiofile(str(wav_path), codec="pcm_s16le",
                                                logger=None)
                    video.close()
                    done += 1
                except Exception as e:
                    log(f"Error on {mp4_path.name}: {e}", "ERROR")
                    errors += 1

    log(f"Audio — total: {total}, extracted: {done}, "
        f"missing: {missing}, errors: {errors}")
    if errors:
        log(f"{errors} extraction error(s)", "WARN")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Build face NPY arrays  (openface_face/)
# ─────────────────────────────────────────────────────────────────────────────

def step_openface(raw_dir: Path, out_dir: Path):
    section("Step 4 / Build face NPY arrays → openface_face/")
    root_faces = raw_dir / "data" / "cropped-aligned-faces" / "Videos"
    face_dir = out_dir / "openface_face"
    face_dir.mkdir(parents=True, exist_ok=True)

    total, saved, missing, errors = 0, 0, 0, 0
    for split in ("test",): # for split in SPLITS: (for all splits, but we only need test here)
        txt = split_txt_path(raw_dir, split)
        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                path = line.strip().split(",")[0]
                if not path.lower().endswith(".mp4"):
                    continue
                total += 1


                # Build path to the face folder (same stem as the video)
                rel_parts = Path(path).parts  # ('Videos', '82782', 'Visite_1', 'clip.mp4')
                if rel_parts and rel_parts[0].lower() == "videos":
                    rel_parts = rel_parts[1:]
                aligned_path = root_faces / Path(*rel_parts)


                if not aligned_path.exists():
                    log(f"Face folder not found: {aligned_path}", "WARN")
                    missing += 1
                    continue

                images = sorted([
                    f for f in aligned_path.iterdir()
                    if f.suffix.lower() in (".jpg", ".png")
                ])

                if not images:
                    log(f"No images in: {aligned_path}", "WARN")
                    missing += 1
                    continue

                try:
                    frames = []
                    for img_path in images[::4]:     # 1 frame out of 4 — remove [::4] to take all frames
                        img = Image.open(img_path).convert("RGB").resize((112, 112))
                        frames.append(np.array(img))

                    video_array = np.stack(frames)   # (N, 112, 112, 3)

                    dest_name = f"{split}_{Path(aligned_path.stem)}.npy" #dest_name = f"{split}_{aligned_path.name}.npy"
                    dest_path = safe_dest(face_dir, dest_name)
                    np.save(str(dest_path), video_array)
                    saved += 1
                    log(f"Saved {dest_path.name} | shape {video_array.shape}")
                except Exception as e:
                    log(f"Error on {aligned_path}: {e}", "ERROR")
                    traceback.print_exc()
                    errors += 1

    log(f"Faces — total: {total}, saved: {saved}, "
        f"missing: {missing}, errors: {errors}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Transcriptions CSV
# ─────────────────────────────────────────────────────────────────────────────

def step_transcriptions(raw_dir: Path, out_dir: Path):
    section("Step 5 / Build transcription CSV")
    csv_path = out_dir / "transcription.csv"
    total, errors = 0, 0

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["name", "chinese", "english"])
        writer.writeheader()

        for split in SPLITS:
            txt = split_txt_path(raw_dir, split)
            with open(txt, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for parts in reader:
                    if len(parts) < 3:
                        if len(parts) >= 1 and parts[0].strip():
                            log(f"Skipping malformed line in {split}: {parts}", "WARN")
                            errors += 1
                        continue
                    path = parts[0]
                    text = ",".join(parts[2:]).strip()
                    base_name = Path(path).stem
                    name = f"{split}_{base_name}"
                    writer.writerow({"name": name, "chinese": "", "english": text})
                    total += 1

    log(f"Saved {csv_path} ({total} rows, {errors} skipped)")


# ─────────────────────────────────────────────────────────────────────────────
# Summary verification
# ─────────────────────────────────────────────────────────────────────────────

def verify(out_dir: Path, corpora: dict):
    section("Verification summary")

    checks = {
        "label.npz":                          out_dir / "label.npz",
        "label.npy":                          out_dir / "label.npy",
        "transcription.csv":    out_dir / "transcription.csv",
    }

    all_ok = True
    for label, path in checks.items():
        ok = path.exists()
        all_ok = all_ok and ok
        log(f"{label}: {'found' if ok else 'MISSING'}", "INFO" if ok else "ERROR")

    for folder in ("subvideo", "subaudio", "openface_face"):
        d = out_dir / folder
        if d.exists():
            count = len(list(d.iterdir()))
            log(f"{folder}/: {count} files")
        else:
            log(f"{folder}/: directory missing", "WARN")
            all_ok = False

    # Cross-check label counts vs files
    for split in SPLITS:
        n = len(corpora[split])
        log(f"  corpus[{split}]: {n} entries")

    print()
    if all_ok:
        log("Pipeline completed successfully ✓", "STEP")
    else:
        log("Pipeline completed with warnings — check messages above", "WARN")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="BAH → AffectGPT preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--raw_dir",
        required=True,
        type=Path,
        help="Root of the raw BAH dataset (BAH_DB_public_access/)",
    )
    parser.add_argument(
        "--skip_audio",
        action="store_true",
        help="Skip WAV extraction (useful if moviepy is not installed)",
    )
    parser.add_argument(
        "--skip_video",
        action="store_true",
        help="Skip MP4 copy step",
    )
    parser.add_argument(
        "--skip_faces",
        action="store_true",
        help="Skip OpenFace NPY extraction",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raw_dir: Path = args.raw_dir.expanduser().resolve()
    out_dir: Path = Path("datasets/BAH-process").resolve()

    print("\n" + "═" * 60)
    print("  BAH Dataset Processing Pipeline")
    print("═" * 60)
    print(f"  Input  : {raw_dir}")
    print(f"  Output : {out_dir}")
    print("═" * 60)

    # ── Pre-flight checks ────────────────────────────────────────────────────
    section("Pre-flight checks")
    if not raw_dir.exists():
        print(f"[ERROR] raw_dir does not exist: {raw_dir}")
        sys.exit(1)
    log(f"raw_dir exists: {raw_dir}")

    check_splits_exist(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory ready: {out_dir}")

    # ── Pipeline steps ───────────────────────────────────────────────────────
    corpora = step_labels(raw_dir, out_dir)

    if not args.skip_video:
        step_copy_videos(raw_dir, out_dir)
    else:
        log("Skipping video copy (--skip_video)", "WARN")

    if not args.skip_audio:
        step_extract_audio(raw_dir, out_dir)
    else:
        log("Skipping audio extraction (--skip_audio)", "WARN")

    if not args.skip_faces:
        step_openface(raw_dir, out_dir) 
    else:
        log("Skipping face NPY extraction (--skip_faces)", "WARN")

    step_transcriptions(raw_dir, out_dir)

    verify(out_dir, corpora)


if __name__ == "__main__":
    main()