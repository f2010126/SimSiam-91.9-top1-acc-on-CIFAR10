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

# ---------------------------------------------------------------------------------------
# SIMSIAM ON CIFAR10 WITH BOHB
# ---------------------------------------------------------------------------------------

# Submit master to train SimSiam on CIFAR10 with BOHB
@master EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_master_simsiam_cifar10.sh

# Submit worker to train SimSiam on CIFAR10 with BOHB
@worker EXPERIMENT_NAME:
  #!/usr/bin/env bash
  mkdir -p /work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/
  sbatch --exclude=dlcgpu42 --output=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --error=/work/dlclarge2/wagnerd-metassl_experiments/BOHB/CIFAR10/{{EXPERIMENT_NAME}}/cluster_oe/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/submit_worker_simsiam_cifar10.sh


