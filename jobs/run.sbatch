#!/bin/bash
# Run one detreco script as a batch job on nocona, so several scans can run
# side by side instead of one after another in an interactive session.
# Submit from the repo root (logs go to output/logs/, which must exist):
#
#   mkdir -p output/logs
#   sbatch -J deteff     jobs/run.sbatch scripts.deteff_scan
#   sbatch -J deteff_pid jobs/run.sbatch scripts.deteff_scan --pid
#   sbatch -J beamcomp   jobs/run.sbatch scripts.beam_composition
#
# Everything after the module name goes to the script. The scans size their
# worker pools to the CPUs this job gets (os.sched_getaffinity), so one flag
# changes both, e.g. sbatch --cpus-per-task=64 jobs/run.sbatch ...
# Memory is per worker: a full waveform read peaks around 5 GB on the
# largest runs, so keep --mem-per-cpu at 7G or more (a bare number is MB).
#
#SBATCH --partition=nocona
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=7G
#SBATCH --time=12:00:00
#SBATCH --job-name=detreco
#SBATCH --output=output/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
if [[ ! -f detreco/bin/activate ]]; then
    echo "Submit from the detreco repo root (no detreco/bin/activate in ${PWD})." >&2
    exit 1
fi
source detreco/bin/activate

echo "$(date)  $(hostname)  cpus=${SLURM_CPUS_PER_TASK}  python3 -m $*"
python3 -u -m "$@"
echo "$(date)  done"
