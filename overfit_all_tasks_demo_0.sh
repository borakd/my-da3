#!/bin/bash
#SBATCH --job-name=cut3r_train
#SBATCH --account=etur59                  # From your alias: -A avg
#SBATCH --partition=acc                # From your alias: -p avg
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1                      # Standard for single-node tasks
#SBATCH --ntasks=1                     # From your alias: --ntasks-per-node=1
#SBATCH --cpus-per-task=15             # From your alias: --cpus-per-task=12   # was 12; MN5 derives mem from cores (8G/core) -> 120G ~= old --mem=120G
#SBATCH --time=72:00:00               # From your alias: --time=168:00:00   # was 168:00:00; acc_ehpc MaxWall is 3-00:00:00
#SBATCH --gres=gpu:1     # From your alias: Request 1 L40S GPU
#SBATCH --array=1-15                   # Spawns 15 jobs for the 15 lines in the text file
#SBATCH --output=logs/job_%A_%a.out    # Master Job ID (%A) and Array Task ID (%a)
#SBATCH --error=logs/job_%A_%a.err

# 1. Load any necessary modules or activate conda environments here
# source activate my_env
# module load miniconda3 ... 

# 2. Create the logs directory if it doesn't already exist
mkdir -p logs

# 3. Read the specific command for this array task ID from the text file
COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" overfit_all_tasks_demo_0_commands.txt)

# 4. Print helpful info to the output log
echo "========================================================="
echo "Starting Slurm Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "Executing Command: $COMMAND"
echo "========================================================="

# 5. Execute the command
eval $COMMAND