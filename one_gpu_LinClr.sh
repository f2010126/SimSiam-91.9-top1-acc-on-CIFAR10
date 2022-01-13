#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH -q dlc-dsengupt
#SBATCH --gres=gpu:1
#SBATCH -o /work/dlclarge1/dsengupt-lth_ws/slurm_logs/800_1gpu_simsiam.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e /work/dlclarge1/dsengupt-lth_ws/slurm_logs/800_1gpu_simsiam.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J One_GPU_SimSiam_Cifar10_LinClr
#SBATCH -t 19:59:00
#SBATCH --mail-type=BEGIN,END,FAIL

cd $(ws_find lth_ws)
#python3 -m venv lth_env
source lth_env/bin/activate
pip list

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"
cd MetaSSL/SimSiam-91.9-top1-acc-on-CIFAR10

echo "FineTune Simsiam with CIFAR10 LR =0.06"

python -m main_lincls --trial '800_run_same_param' --seed 123 --arch resnet18 --num_cls 10 --batch_size 256 --lr 30.0 --weight_decay 0.0 --pretrained /work/dlclarge1/dsengupt-lth_ws/MetaSSL/SimSiam-91.9-top1-acc-on-CIFAR10/experiments/800_run_same_param/800_run_same_param_best.pth "./data"

deactivate