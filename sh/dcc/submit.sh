#!/bin/bash
#SBATCH --job-name=fastflow
#SBATCH --account=h200ea
#SBATCH --partition=h200ea
#SBATCH --gres=gpu:h200:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
#SBATCH --output=watch_folder/%x-%j.out   # saves logs to logs/<jobname>-<jobid>.out
#SBATCH --error=watch_folder/%x-%j.err    # saves errors to logs/<jobname>-<jobid>.err


# init micromamba environment
source /hpc/home/zp70/.bashrc
micromamba activate ts

# Run training
export SCRATCH_DIR='./scratch'

srun python src/train.py \
  experiment=training/single_system/tarflow_Ace-A-Nme \
  'hydra.run.dir=${paths.log_dir}/${task_name}/runs/tarflow_Ace-A-Nme' \
  +logger.wandb.name=tarflow_Ace-A-Nme logger.wandb.project=fastflow

srun python src/train.py \
  experiment=training/single_system/fastflow_Ace-A-Nme \
  'model.teacher_ckpt_path=${paths.log_dir}/${task_name}/runs/tarflow_Ace-A-Nme_no_noise/checkpoints/last_weights_only.ckpt' \
  'hydra.run.dir=${paths.log_dir}/${task_name}/runs/fastflow_Ace-A-Nme-no-kl-no-noise-student-reverse-for-proposal-energy' \
  +logger.wandb.name=fastflow_Ace-A-Nme-no-kl-no-noise-student-reverse-for-proposal-energy \
  logger.wandb.project=fastflow \
  model.lambda_kl=0.0 model.kl_start_frac=1 model.noise_sigma=0.0


