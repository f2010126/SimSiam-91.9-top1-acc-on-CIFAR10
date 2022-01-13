#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
##SBATCH -o /work/dlclarge1/dsengupt-lth_ws/slurm_logs/800_1gpu_simsiam.out # STDOUT  (the folder log has to be created prior to running or this won't work)
##SBATCH -e /work/dlclarge1/dsengupt-lth_ws/slurm_logs/800_1gpu_simsiam.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J ERC_MSSL_Cifar10_SimSiam
#SBATCH -t 19:59:00
#SBATCH --mail-type=BEGIN,END,FAIL

# cd $(ws_find lth_ws)
# python3 -m venv lth_env
# source lth_env/bin/activate
pip list

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"

# cd MetaSSL/SimSiam-91.9-top1-acc-on-CIFAR10

echo "Pretrain Simsiam with CIFAR10 LR =0.06"

python3 -m main --is_bohb_run --seed 3 --arch resnet18 --learning_rate 0.06 --epochs 5 --weight_decay 5e-4 --momentum 0.9 --batch_size 512 --gpu 0 --trial $EXPERIMENT_NAME --exp_dir "/work/dlclarge2/wagnerd-metassl_experiments/CIFAR10"

# deactivate
