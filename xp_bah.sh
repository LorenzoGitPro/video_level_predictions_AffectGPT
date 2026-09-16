#!/bin/bash

GPU=3
RESULTS_FILE="output/results-BAH/results.txt"

# créer dossier si besoin
mkdir -p output/results-BAH

# reset fichier
echo "===== RESULTS BAH =====" > $RESULTS_FILE

declare -A PROMPTS

PROMPTS["simple"]="Classify the emotion in the video as either 'Non-Ambivalent' or 'Ambivalent'. Respond with only one word: "
PROMPTS["def_1"]="Definition: Ambivalence is the state of having contradictory or conflicting feelings or attitudes towards something or someone simultaneously. Classify the emotion in the video as either 'Non-Ambivalent' or 'Ambivalent'. Respond with only one word: "
PROMPTS["def_2"]="Definition: Ambivalence and Hesitancy is understood as the simultaneous experience of desires for change and against change. Classify the emotion in the video as either 'Non-Ambivalent' or 'Ambivalent'. Respond with only one word: "

# === LOOP TRANSCRIPT ===
for TRANSCRIPT in 0 1; do

    if [ "$TRANSCRIPT" -eq 1 ]; then
        MODE_NAME="with_transcript"
    else
        MODE_NAME="no_transcript"
    fi

    # === LOOP PROMPTS ===
    for TAG in simple def_1 def_2; do

        PROMPT="${PROMPTS[$TAG]}"
        FULL_TAG="${MODE_NAME}_${TAG}"

        echo "=============================="
        echo "Running experiment: $FULL_TAG"
        echo "=============================="

        # 1. INFERENCE
        EXP_TAG=$TAG USE_TRANSCRIPT=$TRANSCRIPT CUDA_VISIBLE_DEVICES=$GPU python -u inference_hybird.py \
            --zeroshot \
            --dataset='BAH' \
            --outside_user_message "$PROMPT" \
            --cfg-path=train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml  \
            --options \
                "inference.ckpt_root=emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz_20250110100" \
                "inference.skip_epoch=1" \
            --outside_face_or_frame "multiface_audio_face_text"

        # 2. RÉCUPÉRATION NPZ (robuste)
        NPZ_PATH=$(find output/results-BAH/${MODE_NAME}_${TAG} -name "*.npz" 2>/dev/null | sort | tail -n 1)

        if [ -z "$NPZ_PATH" ]; then
            echo " Aucun NPZ trouvé pour $FULL_TAG"
            continue
        fi

        echo "NPZ trouvé: $NPZ_PATH"

        # 3. CONVERSION NPZ -> NPY
        NPY_PATH=$(python - <<END
import os
import numpy as np

def label_to_logit(text):
    text = str(text).strip()
    if text == "Ambivalent":
        return 1
    elif text == "Non-Ambivalent":
        return 0
    else:
        return 0

npz_path = "$NPZ_PATH"
data = np.load(npz_path, allow_pickle=True)
name2reason = data["name2reason"].item()

logits = [label_to_logit(name2reason[k]) for k in name2reason.keys()]
logits = np.array(logits)

npy_path = os.path.splitext(npz_path)[0] + ".npy"
np.save(npy_path, logits)

print(npy_path)
END
)

        echo "NPY généré: $NPY_PATH"

        # 4. METRICS
        METRICS_OUTPUT=$(CUDA_VISIBLE_DEVICES=$GPU python bah_metrics.py \
            --pred "$NPY_PATH" \
            --gt datasets/BAH-process/label.npy)

        # 5. WRITE RESULTS
        {
            echo ""
            echo "========== $FULL_TAG =========="
            echo "$METRICS_OUTPUT"
        } >> $RESULTS_FILE

    done
done

echo ""
echo "Résultats sauvegardés dans $RESULTS_FILE"