# List available just commands
@list:
  just --list

# ---------------------------------------------------------------------------------------
# SIMSIAM ON CIFAR10
# ---------------------------------------------------------------------------------------

# Pretrain SimSiam on CIFAR10
@pt EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_pretraining_simsiam_cifar10.sh

# Finetune SimSiam on CIFAR10
@ft EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_finetuning_simsiam_cifar10.sh

# Finetune SimSiam on CIFAR10
@both EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_both_simsiam_cifar10.sh

# Run TrivialAugment with SimSiam on CIFAR10
@trivial EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --output=/work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%N.%j.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_trivialaugment_simsiam_cifar10.sh

# ---------------------------------------------------------------------------------------
# SIMSIAM ON CIFAR10 WITH BOHB
# ---------------------------------------------------------------------------------------

# Submit master to train SimSiam on CIFAR10 with BOHB
@master EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_master_simsiam_cifar10.sh

# Start master on the login node
@login-master EXPERIMENT_NAME:
  #!/usr/bin/env bash
  python3 -m both --gpu 0 --is_bohb_run --valid_size 0.1 --seed 1 --trial {{EXPERIMENT_NAME}} --exp_dir "/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10" --pretrained /work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/{{EXPERIMENT_NAME}} --n_iterations 250 --run_id "color-jitter_lr006" --pt_learning_rate 0.06 --shutdown_workers --nic_name "enp1s0"

# Submit worker to train SimSiam on CIFAR10 with BOHB
@worker EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_worker_simsiam_cifar10.sh

# Start master-2 on the login node
@login-master-2 EXPERIMENT_NAME:
  #!/usr/bin/env bash
  python3 -m both --gpu 0 --is_bohb_run --valid_size 0.1 --seed 1 --trial {{EXPERIMENT_NAME}} --exp_dir "/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10" --pretrained /work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/{{EXPERIMENT_NAME}} --n_iterations 250 --run_id "double-color-jitter_lr006" --configspace_mode 'double_color_jitter_strengths' --pt_learning_rate 0.06 --shutdown_workers --nic_name "enp1s0"

# Submit worker-2 to train SimSiam on CIFAR10 with BOHB
@worker-2 EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_worker-2_simsiam_cifar10.sh

# Start master-3 on the login node
@login-master-3 EXPERIMENT_NAME:
  #!/usr/bin/env bash
  python3 -m both --gpu 0 --is_bohb_run --valid_size 0.1 --seed 1 --trial {{EXPERIMENT_NAME}} --exp_dir "/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10" --pretrained /work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/{{EXPERIMENT_NAME}} --n_iterations 250 --run_id "lr-color-jitter_lr006" --configspace_mode 'lr_color_jitter_strengths' --shutdown_workers --nic_name "enp1s0"

# Submit worker-3 to train SimSiam on CIFAR10 with BOHB
@worker-3 EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_worker-3_simsiam_cifar10.sh
