#!/bin/bash
# Orchestrator for rolling out the multi-variable/3D/synthetic-obs
# architecture extension on the real Odyssey cluster (see
# /home/k24aitmo/.claude/plans/staged-meandering-ocean.md and the session
# that built it). Run stages IN ORDER from the login node; each stage
# submits one or more sbatch jobs and returns immediately - check job status
# with `squeue` and logs under log/ before moving to the next stage.
#
# Usage: ./launch_architecture_extension.sh <stage>
#   stages: glorys-download | glorys-merge | masks | masks-dryrun | argo | \
#           smoke | scale-2d3d | scale-full | compare | all-data
#
# IMPORTANT: adjust the variables below to your own Odyssey account/paths
# before running anything - ODYSSEY_USER/DATA_ROOT below are placeholders
# copied from the config files' var_path entries, not necessarily your own.

set -euo pipefail

ODYSSEY_USER="t22picar"                      # <-- CHANGE to your own Odyssey username
DATA_ROOT="/Odyssey/private/${ODYSSEY_USER}/data"
CONDA_ENV="4dvarnet-daniel"
CONDA_SH="/Odyssey/private/${ODYSSEY_USER}/miniforge3/etc/profile.d/conda.sh"
PARTITION="Odyssey"

GLORYS_MULTIDEPTH_DIR="${DATA_ROOT}/glorys_multidepth"
GLORYS_MULTIDEPTH_MERGED="${GLORYS_MULTIDEPTH_DIR}/glorys_multidepth_2010-2020.nc"
GLORYS_YEAR_START=2010
GLORYS_YEAR_END=2020
GLORYS_MIN_DEPTH=0.49402499198913574
GLORYS_MAX_DEPTH=200

REFERENCE_GRID="${DATA_ROOT}/ssh_L4/SSH_L4_CMEMS_2010-01-01-2024-01-01_4th.nc"   # any file with lat/lon coords on the target grid
MASKS_OUTPUT="${DATA_ROOT}/mask/synthetic_6sat_masks.pickle"
MASKS_N_DAYS=3653

ARGO_OUTPUT_DIR="${DATA_ROOT}/argo/gridded"
ARGO_START_DATE="2010-01-01"
ARGO_END_DATE="2020-01-01"
DEPTHS="0.49 15 50 100 200"

mkdir -p log

stage="${1:-}"
if [[ -z "$stage" ]]; then
    echo "Usage: $0 <glorys-download|glorys-merge|masks|masks-dryrun|argo|smoke|scale-2d3d|scale-full|compare|all-data>"
    exit 1
fi

case "$stage" in

  # --- Stage 1a: download real multi-depth Glorys fields, one job per year ---
  glorys-download)
    for year in $(seq "$GLORYS_YEAR_START" "$GLORYS_YEAR_END"); do
      sbatch <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --job-name=glorys-multidepth-${year}
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=log/glorys_multidepth_${year}_%j.log
export HOME=/Odyssey/private/${ODYSSEY_USER}/
source "${CONDA_SH}"
conda activate ${CONDA_ENV}
cd ${PWD}/../download_data_
srun python import_data_glorys_multidepth.py ${year} ${GLORYS_MIN_DEPTH} ${GLORYS_MAX_DEPTH}
EOF
    done
    echo "Submitted ${GLORYS_YEAR_START}..${GLORYS_YEAR_END} glorys-download jobs. Check with: squeue -u \$USER"
    ;;

  # --- Stage 1b: merge yearly files into the single multi-depth file the
  #     config/vars/*_depths.yaml fragments point at ---
  glorys-merge)
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --job-name=glorys-merge
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=log/glorys_merge_%j.log
export HOME=/Odyssey/private/${ODYSSEY_USER}/
source "${CONDA_SH}"
conda activate ${CONDA_ENV}
srun python -c "
import xarray as xr
ds = xr.open_mfdataset('${GLORYS_MULTIDEPTH_DIR}/*.nc', combine='by_coords')
ds.to_netcdf('${GLORYS_MULTIDEPTH_MERGED}')
print('wrote', '${GLORYS_MULTIDEPTH_MERGED}')
"
EOF
    echo "Submitted glorys-merge job (run only after all glorys-download jobs finished)."
    ;;

  # --- Stage 2a: quick local sanity check of the synthetic mask generator
  #     alone, no GPU/queue - catches config typos before the sbatch job ---
  masks-dryrun)
    source "${CONDA_SH}"
    conda activate ${CONDA_ENV}
    python -m contrib.synthetic_obs.build_masks \
      --grid-from "${REFERENCE_GRID}" --n-days 5 \
      --output /tmp/synthetic_masks_dryrun.pickle
    echo "Dry-run OK -> /tmp/synthetic_masks_dryrun.pickle (5 days only, sanity check)"
    ;;

  # --- Stage 2b: full synthetic mask generation (also runs automatically as
  #     an entrypoint inside unet_uv_full_integration_*.yaml, but running it
  #     standalone first lets you inspect it before committing a GPU job) ---
  masks)
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --job-name=synthetic-masks
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=log/synthetic_masks_%j.log
export HOME=/Odyssey/private/${ODYSSEY_USER}/
source "${CONDA_SH}"
conda activate ${CONDA_ENV}
srun python -m contrib.synthetic_obs.build_masks \
  --grid-from "${REFERENCE_GRID}" --n-days ${MASKS_N_DAYS} \
  --output "${MASKS_OUTPUT}"
EOF
    echo "Submitted masks job -> ${MASKS_OUTPUT}"
    ;;

  # --- Stage 3: Argo download + QC + vertical interpolation + gridding ---
  argo)
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --job-name=argo-pipeline
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=log/argo_pipeline_%j.log
export HOME=/Odyssey/private/${ODYSSEY_USER}/
source "${CONDA_SH}"
conda activate ${CONDA_ENV}
srun python -m contrib.argo.run_pipeline \
  --grid-from "${REFERENCE_GRID}" \
  --start-date "${ARGO_START_DATE}" --end-date "${ARGO_END_DATE}" \
  --lat-min -70 --lat-max 70 --lon-min -180 --lon-max 180 \
  --depths ${DEPTHS} \
  --output-dir "${ARGO_OUTPUT_DIR}"
EOF
    echo "Submitted argo-pipeline job -> ${ARGO_OUTPUT_DIR}"
    ;;

  # --- Stage 4: smoke test - Phase 0 (simplest xp), short-circuited to a
  #     couple epochs on a handful of batches, just to catch crashes before
  #     committing a long job ---
  smoke)
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=smoke-ssh-sst
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=64G
#SBATCH --output=log/smoke_%j.log
export HOME=/Odyssey/private/${ODYSSEY_USER}/
source "${CONDA_SH}"
conda activate ${CONDA_ENV}
HYDRA_FULL_ERROR=1 srun python main.py xp=unet_uv_ssh_sst_aoml_15m_10y_11d_bathy_mae_duacs_RonanUnet \
  ++trainer.max_epochs=2 ++trainer.limit_train_batches=5
EOF
    echo "Submitted smoke-test job (2 epochs, 5 batches). Check log/smoke_*.log for crashes before scaling up."
    ;;

  # --- Stage 5a: Phase 2 (2D -> 3D depth-resolved Glorys), full run ---
  scale-2d3d)
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=temp3d
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=300G
#SBATCH --output=log/temp3d_%j.log
export HOME=/Odyssey/private/${ODYSSEY_USER}/
source "${CONDA_SH}"
conda activate ${CONDA_ENV}
HYDRA_FULL_ERROR=1 srun python main.py xp=unet_uv_temp3d_aoml_15m_10y_11d_bathy_mae_duacs_RonanUnet
EOF
    echo "Submitted scale-2d3d (Phase 2) job."
    ;;

  # --- Stage 5b: Phase 5 (full integration), full run ---
  scale-full)
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=full-integration
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=350G
#SBATCH --output=log/full_integration_%j.log
export HOME=/Odyssey/private/${ODYSSEY_USER}/
source "${CONDA_SH}"
conda activate ${CONDA_ENV}
HYDRA_FULL_ERROR=1 srun python main.py xp=unet_uv_full_integration_15m_10y_11d_bathy_mae_duacs_RonanUnet
EOF
    echo "Submitted scale-full (Phase 5) job."
    ;;

  # --- Stage 6: the 3 ablation configs (baseline / heads / uncertainty),
  #     submitted together for a like-for-like comparison ---
  compare)
    for variant in "" "_heads" "_uncertainty"; do
      xp_name="unet_uv_full_integration${variant}_15m_10y_11d_bathy_mae_duacs_RonanUnet"
      sbatch <<EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=compare${variant:-_baseline}
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=350G
#SBATCH --output=log/compare${variant:-_baseline}_%j.log
export HOME=/Odyssey/private/${ODYSSEY_USER}/
source "${CONDA_SH}"
conda activate ${CONDA_ENV}
HYDRA_FULL_ERROR=1 srun python main.py xp=${xp_name}
EOF
    done
    echo "Submitted 3 comparison jobs (baseline, heads, uncertainty). Compare val_total_mse and per-variable metrics in each run's CSV logger output."
    ;;

  all-data)
    "$0" glorys-download
    echo "Wait for all glorys-download jobs to finish (squeue -u \$USER), then run: $0 glorys-merge"
    echo "In parallel, you can also run: $0 masks-dryrun ; $0 masks ; $0 argo"
    ;;

  *)
    echo "Unknown stage: $stage"
    echo "Usage: $0 <glorys-download|glorys-merge|masks|masks-dryrun|argo|smoke|scale-2d3d|scale-full|compare|all-data>"
    exit 1
    ;;
esac
