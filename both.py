# try:
#     from finetuning import main as ft_main
#     from pretraining import main as pt_main
# except ImportError:
#     pass
#
# def run_pretraining_and_finetuning_sequential(args):
#     print("\n\n\nPretraining...\n\n\n")
#     pt_main(args)
#     print("\n\n\nFinetuning...\n\n\n")
#     ft_main(args)

if __name__ == '__main__':
    import argparse
    import numpy as np
    from finetuning import main as ft_main
    from pretraining import main as pt_main

    parser = argparse.ArgumentParser(prog='arguments for training')
    parser.add_argument('--data_root', type=str, default='data', help='path to dataset directory')
    parser.add_argument('--exp_dir', type=str, default='experiments', help='path to experiment directory')
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--img_dim', default=32, type=int)
    parser.add_argument('--arch', default='resnet18', help='model name is used for training')
    parser.add_argument('--feat_dim', default=2048, type=int, help='feature dimension')
    parser.add_argument('--num_proj_layers', type=int, default=2, help='number of projection layer')
    parser.add_argument('--pt_batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--ft_batch_size', default=256, type=int, metavar='N', help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--pt_epochs', type=int, default=800, help='number of training epochs')
    parser.add_argument('--ft_epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--loss_version', default='simplified', type=str, choices=['simplified', 'original'], help='do the same thing but simplified version is much faster. ()')
    parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
    parser.add_argument('--eval_freq', default=5, type=int, help='evaluate model frequency')
    parser.add_argument('--save_freq', default=50, type=int, help='save model frequency')
    parser.add_argument('--pt_resume', default=None, type=str, help='path to latest checkpoint')
    parser.add_argument('--ft_resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--pt_learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--ft_learning_rate', default=30., type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--pt_weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--ft_weight_decay', default=0., type=float,  metavar='W', help='weight decay (default: 0.)')
    parser.add_argument('--pt_momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--ft_momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--seed', default=123, type=int, metavar='N', help='random seed of numpy and torch')
    parser.add_argument('--num_cls', default=10, type=int, metavar='N', help='number of classes in dataset (output dimention of models)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int, help='learning rate schedule (when to drop lr by a ratio)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,  help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true', help='Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training')
    parser.add_argument('--pretrained', default='', type=str, help='path to pretrained checkpoint')
    parser.add_argument('--is_bohb_run', action='store_true', help='Set this flag if you want the experiment to be a BOHB run.')
    parser.add_argument("--run_id", default="default_BOHB")
    parser.add_argument("--n_iterations", type=int, default=10, help="How many BOHB iterations")
    parser.add_argument("--min_budget", type=int, default=800)
    parser.add_argument("--max_budget", type=int, default=800)
    parser.add_argument("--budget_mode", type=str, default="epochs", choices=["epochs", "data"], help="Choose your desired fidelity")
    parser.add_argument("--eta", type=int, default=2)
    parser.add_argument("--configspace_mode", type=str, default='color_jitter_strengths', choices=["imagenet_probability_simsiam_augment", "cifar10_probability_simsiam_augment", "color_jitter_strengths", "rand_augment", "probability_augment", "double_probability_augment"], help='Define which configspace to use.')
    parser.add_argument("--nic_name", default="eth0", help="The network interface to use")  # local: "lo", cluster: "eth0"
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help="Make this execution a worker server")
    parser.add_argument("--warmstarting", type=bool, default=False)
    parser.add_argument("--warmstarting_dir", type=str, default=None)
    parser.add_argument("--shutdown_workers", action='store_true', help='If using this flag, workers are not being shutdown after registering results.')
    parser.add_argument("--run_background_worker", action='store_true',help='If using this flag, the master runs a worker in the background.')
    parser.add_argument("--valid_size", default=0.0, type=float, help='If valid_size > 0, pick some images from the trainset to do evaluation on. If valid_size=0 evaluation is done on the testset.')
    parser.add_argument('--use_fix_aug_params', action='store_true', help='Use this flag if you want to try out specific aug params (e.g., from a best BOHB config). Default values will be overwritten then without crashing other experiments.')
    args = parser.parse_args()

    if args.is_bohb_run:
        from hyperparameter_optimization.master import start_bohb_master
        if np.isclose(args.valid_size, 0.0) and args.is_bohb_run:
            raise ValueError("--valid_size needs to be > 0 for BOHB runs!")
        start_bohb_master(args)
    else:
        print("\n\n\nPretraining...\n\n\n")
        pt_main(args)
        print("\n\n\nFinetuning...\n\n\n")
        ft_main(args)
