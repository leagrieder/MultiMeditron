#!/bin/bash
#SBATCH --job-name=translate_all
#SBATCH --output=logs/translate_%A_%a.out
#SBATCH --error=logs/translate_%A_%a.err
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00

LANGUAGES=(amh swa tam hau yor zul)
LANG=${LANGUAGES[$SLURM_ARRAY_TASK_ID]}

echo "Processing language: $LANG"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Run your EXISTING script - don't change it!
python src/multimeditron/translation/pipelines/experiments/consensus_pipeline.py \
    --input /mloscratch/users/nemo/datasets/guidelines/chunked_eng.jsonl \
    --output src/multimeditron/translation/datasets/generated_datasets/consensus \
    --languages $LANG \
    --sample 5000 \
    --batch-size 64 \
    --qwen-size 3B-Instruct

echo "Completed $LANG"
