import argparse
import os
import torch
import ast

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--num_worker', type=int, default=4,
                        help='Number of workers (CPU processes) to use (default: 16)')
    parser.add_argument('--difficulty', type=int, default=7,
                        help='difficulty for Curriculum learning')
    parser.add_argument('--macro', type=ast.literal_eval, default=True,
                        help='Macro based Environment')
    parser.add_argument('--use-grad-ratio', type=ast.literal_eval, default=True,
                        help='Use grad ratio')
    parser.add_argument('--mask', type=ast.literal_eval, default=True,
                        help='Mask hidden state')
    parser.add_argument('--num-step', type=int, default=64,
                        help='Number of forward steps (default: 64)')
    parser.add_argument('--ext-gamma', type=float, default=0.999,
                        help='Extrinsic discount factor for rewards (default: 0.999)')
    parser.add_argument('--eps', type=float, default=0.1,
                        help='Clip (default: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='L2 (default: 1e-4)')
    parser.add_argument('--use-gae', action='store_true', default=True,
                        help='use generalized advantage estimation (default: True)')
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="Lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Use GPU training (default: True)')
    parser.add_argument('--epoch', type=int, default=3,
                        help='number of epochs (default: 3)')
    parser.add_argument('--max-grad-norm', type=float, default=50,
                        help='number of epochs (default: 50)')
    parser.add_argument('--rew-norm', type=bool, default=False,
                        help='reward normalization (default: False)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='batch size (default: 512)')
    parser.add_argument('--entropy-coef', type=float, default=1e-3,
                        help='entropy term coefficient (default: 1e-3)')
    parser.add_argument('--noise-linear', type=bool, default=True,
                        help='RND Noise Linear (default: True)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Save interval, one save per n updates (default: 100)')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='Load pre-trained Model (default: False)')
    parser.add_argument('--log_dir', default='log',
                        help='Directory to save agent logs (default: log)')
    parser.add_argument('--save_dir', default='saved_models',
                        help='Directory to save agent logs (default: saved_models)')
#############################################################3
    #parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--sigma', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--lambda_', type=float, default=0.1, help='a3c loss coefficient')
    parser.add_argument('--eta', type=float, default=0.2, help='intrinsic coefficient')
    parser.add_argument('--beta', type=float, default=0.2, help='curiosity coefficient')
    parser.add_argument("--num_local_steps", type=int, default=50)
    parser.add_argument("--num_global_steps", type=int, default=1e8)
    parser.add_argument("--num_processes", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=500, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/a3c_icm_street_fighter")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--use_gpu", type=bool, default=True)                        
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.num_batch = int(args.num_step * args.num_worker / args.batch_size)

    print("GPU training: ", args.cuda)

    return args
