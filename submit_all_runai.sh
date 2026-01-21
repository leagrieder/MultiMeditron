#!/bin/bash

LANGUAGES=(amh swa tam hau yor zul)

for LANG in "${LANGUAGES[@]}"; do
    echo "Submitting job for language: $LANG"
    
    runai submit \
      --name translate-$LANG \
      --image registry.rcp.epfl.ch/multimeditron/basic:latest-grieder \
      --pvc light-scratch:/mloscratch \
      --large-shm \
      -e NAS_HOME=/mloscratch/users/grieder \
      -e HF_API_KEY_FILE_AT=/mloscratch/users/grieder/keys/hf_key.txt \
      -e WANDB_API_KEY_FILE_AT=/mloscratch/users/grieder/keys/wandb_key.txt \
      -e GITCONFIG_AT=/mloscratch/users/grieder/.gitconfig \
      -e GIT_CREDENTIALS_AT=/mloscratch/users/grieder/.git-credentials \
      -e VSCODE_CONFIG_AT=/mloscratch/users/grieder/.vscode-server \
      --backoff-limit 0 \
      --run-as-gid 84257 \
      --node-pool h100 \
      --gpu 2 \
      -- bash -c "cd /mloscratch/users/grieder/MeditronPolyglot/MultiMeditron && \
                  python src/multimeditron/translation/pipelines/experiments/consensus_pipeline.py \
                  --input /mloscratch/users/nemo/datasets/guidelines/chunked_eng.jsonl \
                  --output src/multimeditron/translation/datasets/generated_datasets/consensus \
                  --languages $LANG \
                  --sample 5000 \
                  --batch-size 64 \
                  --qwen-size 3B-Instruct > /mloscratch/users/grieder/logs/translate_$LANG.log 2>&1"
    
    echo "Submitted: translate-$LANG"
    echo ""
done

echo "All jobs submitted!"
echo "Monitor with: runai list jobs"
