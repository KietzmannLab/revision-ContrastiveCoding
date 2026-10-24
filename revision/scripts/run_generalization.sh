#!/bin/bash
# Submit the full 09_generalization.py pipeline as SLURM jobs: at most 3 CPU jobs and 1 GPU job
# running at a time.
#
# The CPU work is split into three lanes; within a lane each job waits for the previous one, so
# each lane runs one job at a time. Every CPU job uses $CPUS cores (the face-trained model's fits
# are the bulk of the work and scale with them). The tests start once all three lanes are done:
#
#   GPU:    features (~30 min) ··························· (the fits wait for it)
#   lane A: fit face-trained sub-01 -> sub-03 -> sub-06 ───────────┐
#   lane B: lowlevel -> fit other models -> fit face-trained sub-05 ┼─> test
#   lane C: noise ceilings -> fit face-trained sub-07 ──────────────┘
#
# Jobs depend on each other with afterany: a failed job does not block the rest. 09 skips outputs
# that already exist, so after a failure just fix the cause and run this script again. The test
# job lists any missing results.
#
# Usage (from anywhere):  bash revision/scripts/run_generalization.sh
# Settings can be overridden through the environment, e.g.
#   CPUS=128 MEM=400G bash revision/scripts/run_generalization.sh
#   SPLITS="tau cluster_k5_0 cluster_k5_1 cluster_k5_2 cluster_k5_3 cluster_k5_4 ood" ...   (no random baseline)

set -euo pipefail

if [[ -n ${SLURM_JOB_ID:-} ]]; then
    echo "run this script with bash on the login node, not with sbatch: it submits the jobs itself" >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

G=revision/scripts/09_generalization.py
PY=${PY:-$HOME/.conda/envs/revision/bin/python}
PY_GPU=${PY_GPU:-$HOME/.conda/envs/revision-cu128/bin/python}
CPU_PARTITION=${CPU_PARTITION:-klab-cpu}
GPU_PARTITION=${GPU_PARTITION:-klab-gpu}
CPUS=${CPUS:-64}
MEM=${MEM:-250G}  # face-trained fits: ~2 GB per concurrent conv-layer fit (one per core)
SUBJECTS=(sub-01 sub-03 sub-05 sub-06 sub-07)
SPLITS=${SPLITS:-"tau cluster_k5_0 cluster_k5_1 cluster_k5_2 cluster_k5_3 cluster_k5_4 ood random_0 random_1 random_2 random_3 random_4"}
OOD_TYPES=${OOD_TYPES:-unusual}
N_PERM=${N_PERM:-10000}

LOG_DIR=logs/generalization
mkdir -p "$LOG_DIR"
OPTS="--splits $SPLITS --ood-types $OOD_TYPES"

declare -A LAST  # lane -> its last job id

submit() {  # submit LANE NAME EXTRA-DEPENDENCIES PARTITION-ARGS... -- COMMAND; appends the job to the lane
    local lane=$1 name=$2 deps=$3; shift 3
    local args=()
    while [[ $1 != -- ]]; do args+=("$1"); shift; done
    shift
    deps=$(echo "${LAST[$lane]:-} $deps" | xargs | tr ' ' ':')
    LAST[$lane]=$(sbatch --parsable -J "$name" -o "$LOG_DIR/$name-%j.out" ${deps:+--dependency=afterany:$deps} \
                  "${args[@]}" --wrap "$*")
    echo "lane $lane: $name (job ${LAST[$lane]})"
}

CPU=(-p "$CPU_PARTITION" -c "$CPUS" --mem="$MEM")
FIT="$PY $G fit --n-jobs $CPUS $OPTS"  # threads over ROIs x splits (BLAS limited to 1 per fit)

# GPU: DNN features of all pool images (~5 min per subject, mostly caching the stimuli)
submit GPU gen-features "" -p "$GPU_PARTITION" --gres=gpu:1 -c 16 --mem=64G -- \
    "$PY_GPU" $G features
FEAT=${LAST[GPU]}

# Lane A: the face-trained model's fits
submit A gen-fit-vggface-sub-01 "$FEAT" "${CPU[@]}" -- $FIT --subjects sub-01 --models alexnet-vggface
submit A gen-fit-vggface-sub-03 "" "${CPU[@]}" -- $FIT --subjects sub-03 --models alexnet-vggface
submit A gen-fit-vggface-sub-06 "" "${CPU[@]}" -- $FIT --subjects sub-06 --models alexnet-vggface

# Lane B: Gabor features (processes over images), the other four models, then one subject of the
# face-trained model
submit B gen-lowlevel "" "${CPU[@]}" -- "$PY" $G lowlevel --n-jobs $CPUS
submit B gen-fit-other "$FEAT" "${CPU[@]}" -- \
    $FIT --models alexnet-barlow-twins alexnet-barlow-twins-random GistPC Gabor
submit B gen-fit-vggface-sub-05 "" "${CPU[@]}" -- $FIT --subjects sub-05 --models alexnet-vggface

# Lane C: noise ceilings, one process per subject sharing the job's cores (GSN's time is in BLAS),
# then the last subject of the face-trained model
NC_THREADS=$(( CPUS / ${#SUBJECTS[@]} > 0 ? CPUS / ${#SUBJECTS[@]} : 1 ))
NC_CMD=""
for s in "${SUBJECTS[@]}"; do
    NC_CMD+="env OPENBLAS_NUM_THREADS=$NC_THREADS $PY $G noise-ceilings --subjects $s $OPTS"
    NC_CMD+=" > $LOG_DIR/gen-noise-ceilings-$s.log 2>&1 & "
done
submit C gen-noise-ceilings "" "${CPU[@]}" -- "$NC_CMD wait"
submit C gen-fit-vggface-sub-07 "$FEAT" "${CPU[@]}" -- $FIT --subjects sub-07 --models alexnet-vggface

# Tests, tables and figures, once all three lanes are done
submit A gen-test "${LAST[B]} ${LAST[C]}" -p "$CPU_PARTITION" -c "$CPUS" --mem=64G -- \
    env OPENBLAS_NUM_THREADS=$CPUS "$PY" $G test --n-perm $N_PERM $OPTS

echo "logs in $REPO_DIR/$LOG_DIR; check with: squeue -u \$USER"
