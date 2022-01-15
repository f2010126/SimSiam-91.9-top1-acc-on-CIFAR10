#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH -q dlc-wagnerd
#SBATCH --gres=gpu:1
##SBATCH -o /work/dlclarge1/dsengupt-lth_ws/slurm_logs/800_1gpu_simsiam.out # STDOUT  (the folder log has to be created prior to running or this won't work)
##SBATCH -e /work/dlclarge1/dsengupt-lth_ws/slurm_logs/800_1gpu_simsiam.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J ERC_MSSL_Cifar10_SimSiam
#SBATCH -t 19:59:00
##SBATCH --mail-type=BEGIN,END,FAIL

# cd $(ws_find lth_ws)
# python3 -m venv lth_env
# source lth_env/bin/activate
pip list

python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"

# cd MetaSSL/SimSiam-91.9-top1-acc-on-CIFAR10

# echo "Pretrain Simsiam with CIFAR10 LR =0.06"

python3 -m both --gpu 0 --valid_size 0.0 --seed 3 --pt_learning_rate 0.06 --trial $EXPERIMENT_NAME --exp_dir "/work/dlclarge2/wagnerd-metassl_experiments/CIFAR10" --pretrained /work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/$EXPERIMENT_NAME/$EXPERIMENT_NAME --use_fix_aug_params --use_fix_aug_params_ft --brightness_strength 0.7115759465642293 --contrast_strength 0.4024641067193702 --saturation_strength 0.2845185592463405 --hue_strength 0.04998358654852737 --ft_brightness_strength 0.7269498100347377 --ft_contrast_strength 1.188342814448252 --ft_saturation_strength 0.490243242105925 --ft_hue_strength 0.16944211021863254

# deactivate
