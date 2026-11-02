#!/usr/bin/env bash
# Back up the Thor experiment results to an external drive (FAT32-safe: no symlinks or
# permissions, files < 4 GB). Re-runnable: rsync skips unchanged files, so run it again
# after new results land.
#   scripts/backup_to_drive.sh [/mnt/thor_backup]
set -uo pipefail
DEST="${1:-/mnt/thor_backup}/FineTune-YOLOW-UAS_backup"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p "$DEST"
RS=(rsync -rt --modify-window=2 --no-links)
log() { echo "[backup] $(date '+%F %T') $*"; }

log "1. git bundle (all branches + history)"
git bundle create "$DEST/repo.bundle" --all 2>&1 | tail -1
git log --oneline -1 > "$DEST/repo_HEAD.txt"

log "2. reports, results, configs, paper"
"${RS[@]}" docs eval_results configs/thor_sweeps Dockerfile.thor "$DEST/repo_files/"
cp -f ./*.pdf "$DEST/" 2>/dev/null

log "3. open-vocab benchmark (OVB) images + all annotations (not in git)"
"${RS[@]}" datasets/OpenVocab "$DEST/datasets/"

log "4. run provenance (args, per-epoch CSVs, logs; no weights)"
"${RS[@]}" --include='*/' --include='*.csv' --include='*.yaml' --include='*.json' \
    --include='*.log' --include='*.png' --exclude='*' --prune-empty-dirs \
    runs/sweep/ "$DEST/runs_provenance/"

log "5. stripped final weights of every trained run (~55 MB each)"
source ~/venvs/yoloworld-trt/bin/activate
pairs=()
for w in runs/sweep/*/weights/last.pt; do
    run="$(basename "$(dirname "$(dirname "$w")")")"
    [[ "$run" == smoke_* ]] && continue
    [[ -f "$DEST/run_weights/$run.pt" && "$DEST/run_weights/$run.pt" -nt "$w" ]] && continue
    pairs+=("$run=$w")
done
pairs+=("paper_5_epoch=5_epoch.pt")
if [[ ${#pairs[@]} -gt 0 ]]; then
    CUDA_VISIBLE_DEVICES= nice -n 10 python -W ignore scripts/export_checkpoints.py \
        --output "$DEST/run_weights" "${pairs[@]}" 2>&1 | grep -v -i axes3d | tail -3
fi

log "6. recommended checkpoints + annotated videos (once the final pass has made them)"
[[ -d checkpoints ]] && "${RS[@]}" checkpoints "$DEST/"
[[ -d deliverables ]] && "${RS[@]}" deliverables "$DEST/"

log "7. environment record"
mkdir -p "$DEST/environment"
pip freeze > "$DEST/environment/native_venv_pip_freeze.txt" 2>/dev/null
python -c "import torch,tensorrt,ultralytics;print('torch',torch.__version__);print('tensorrt',tensorrt.__version__);print('ultralytics',ultralytics.__version__)" \
    > "$DEST/environment/native_versions.txt" 2>/dev/null
head -2 /etc/nv_tegra_release > "$DEST/environment/jetpack_release.txt" 2>/dev/null
dpkg -l | grep -E "nvidia-jetpack|cuda-toolkit|libnvinfer10|libcudnn9" | awk '{print $2, $3}' \
    > "$DEST/environment/jetpack_packages.txt"
scripts/thor_run.sh pip freeze > "$DEST/environment/container_pip_freeze.txt" 2>/dev/null

log "8. README + checksums"
cat > "$DEST/README.txt" <<'EOF'
FineTune-YOLOW-UAS — Jetson Thor experiment backup (branch thor-experiments)

repo.bundle            Full git repo, all branches/history. Restore:
                         git clone repo.bundle FineTune-YOLOW-UAS && cd FineTune-YOLOW-UAS
                         git checkout thor-experiments
repo_files/docs/THOR_PROGRESS.md
                       Running log + results summary of every experiment (start here).
repo_files/eval_results/  All metric JSON/CSV, Pareto plots, benchmark tables.
datasets/OpenVocab/    The 195-image open-vocab benchmark (OVB) images + annotations
                       (val.json is the one used; not stored in git).
checkpoints/           Recommended models (ultralytics YOLOWorld .pt, FP16, ~55 MB) +
                       index.json (source run, sha256). Load: YOLOWorld('x.pt').set_classes([...])
run_weights/           Final weights of every training run (same format) + index.json.
deliverables/videos/   Annotated drone videos (per model + 2x2 comparison).
runs_provenance/       Per-run training args, per-epoch OV logs, queue/eval logs.
environment/           pip freezes (native venv + container), JetPack/TensorRT versions.
SHA256SUMS             Checksums of every file in this backup.
EOF
(cd "$DEST" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
sync
log "done: $(du -sh "$DEST" | cut -f1) in $DEST"
