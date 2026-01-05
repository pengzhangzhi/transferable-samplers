#!/bin/bash
#SBATCH --job-name=fastflow
#SBATCH --account=h200ea
#SBATCH --partition=h200ea
#SBATCH --gres=gpu:h200:4
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
#SBATCH --output=watch_folder/%x-%j.out   # saves logs to logs/<jobname>-<jobid>.out
#SBATCH --error=watch_folder/%x-%j.err    # saves errors to logs/<jobname>-<jobid>.err

unset SLURM_CPU_BIND

# Optional but recommended for CPU-threaded libs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK


# init micromamba environment
source /hpc/home/zp70/.bashrc
micromamba activate ts

# Run training
export SCRATCH_DIR='./scratch'

# srun python src/train.py \
#   experiment=training/single_system/tarflow_Ace-A-Nme \
#   'hydra.run.dir=${paths.log_dir}/${task_name}/runs/tarflow_Ace-A-Nme' \
#   +logger.wandb.name=tarflow_Ace-A-Nme logger.wandb.project=fastflow

srun python src/train.py   \
    experiment=training/single_system/fastflow_Ace-A-Nme \
    'model.teacher_ckpt_path=${paths.log_dir}/${task_name}/runs/tarflow_Ace-A-Nme_no_noise/checkpoints/last_weights_only.ckpt' \
    'hydra.run.dir=${paths.log_dir}/${task_name}/runs/fastflow_Ace-A-Nme-dist_loss' \
    +logger.wandb.name=fastflow_Ace-A-Nme-dist_loss \
    logger.wandb.project=fastflow   \
    model.lambda_kl=0.0 \
    model.kl_start_frac=1 model.align_until_frac=0.0 model.lambda_align=0 \
    model.noise_sigma=0.0   \
    trainer='ddp' trainer.strategy="ddp_find_unused_parameters_true" \
    trainer.max_epochs=3000 \
    'trainer.plugins=[]' \
    +model.student_ckpt_path=scratch/transferable-samplers/logs/train/runs/fastflow_Ace-A-Nme-no-kl-no-noise-student-reverse-for-proposal-energy/checkpoints/last_weights_only.ckpt

    # model.optimizer.lr=1e-3 \
