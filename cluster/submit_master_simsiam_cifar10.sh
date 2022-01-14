#!/bin/bash
#SBATCH -p bosch_gpu-rtx2080
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
##SBATCH -o /work/dlclarge1/dsengupt-lth_ws/slurm_logs/800_1gpu_simsiam.out # STDOUT  (the folder log has to be created prior to running or this won't work)
##SBATCH -e /work/dlclarge1/dsengupt-lth_ws/slurm_logs/800_1gpu_simsiam.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J ERC_MSSL_Cifar10_SimSiam
#SBATCH -t 3-00:00 # time (D-HH:MM)
##SBATCH --mail-type=BEGIN,END,FAIL

# cd $(ws_find lth_ws)
# python3 -m venv lth_env
# source lth_env/bin/activate
pip list

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"

# cd MetaSSL/SimSiam-91.9-top1-acc-on-CIFAR10

echo "Pretrain Simsiam with CIFAR10 LR =0.06"

python3 -m both --gpu 0 --is_bohb_run --valid_size 0.1 --run_background_worker --seed 1 --trial $EXPERIMENT_NAME --exp_dir "/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10" --pretrained /work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/$EXPERIMENT_NAME/$EXPERIMENT_NAME --n_iterations 250 --run_id "color-jitter"

# deactivate
