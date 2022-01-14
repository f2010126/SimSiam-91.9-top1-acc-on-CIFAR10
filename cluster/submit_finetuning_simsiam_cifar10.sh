#!/bin/bash
#SBATCH -p bosch_gpu-rtx2080
##SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
##SBATCH -o /work/dlclarge1/dsengupt-lth_ws/slurm_logs/800_1gpu_simsiam.out # STDOUT  (the folder log has to be created prior to running or this won't work)
##SBATCH -e /work/dlclarge1/dsengupt-lth_ws/slurm_logs/800_1gpu_simsiam.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J ERC_MSSL_Cifar10_SimSiam_FT
#SBATCH -t 19:59:00
#SBATCH --mail-type=BEGIN,END,FAIL

# cd $(ws_find lth_ws)
# python3 -m venv lth_env
# source lth_env/bin/activate
pip list

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"

# cd MetaSSL/SimSiam-91.9-top1-acc-on-CIFAR10

echo "FineTune Simsiam with CIFAR10 LR =0.06"

python -m both --trial $EXPERIMENT_NAME --seed 3 --pretrained /work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/$EXPERIMENT_NAME/$EXPERIMENT_NAME --ft_learning_rate 30 --gpu 0


# deactivate
