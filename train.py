"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.model import ActorCritic, IntrinsicCuriosityModule
from src.optimizer import GlobalAdam
from src.process import local_train
import torch.multiprocessing as _mp
import shutil
import os
from config import get_args
os.system("Xvfb :0 -screen 0 800x600x16 +extension RANDR &")
os.environ["DISPLAY"] = ":0"

# def get_args():
#     parser = argparse.ArgumentParser(
#         """Implementation of model described in the paper: Curiosity-driven Exploration by Self-supervised Prediction for Street Fighter""")
#     parser.add_argument('--lr', type=float, default=1e-4)
#     parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
#     parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
#     parser.add_argument('--sigma', type=float, default=0.01, help='entropy coefficient')
#     parser.add_argument('--lambda_', type=float, default=0.1, help='a3c loss coefficient')
#     parser.add_argument('--eta', type=float, default=0.2, help='intrinsic coefficient')
#     parser.add_argument('--beta', type=float, default=0.2, help='curiosity coefficient')
#     parser.add_argument("--num_local_steps", type=int, default=50)
#     parser.add_argument("--num_global_steps", type=int, default=1e8)
#     parser.add_argument("--num_processes", type=int, default=3)
#     parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
#     parser.add_argument("--max_actions", type=int, default=500, help="Maximum repetition steps in test phase")
#     parser.add_argument("--log_path", type=str, default="tensorboard/a3c_icm_street_fighter")
#     parser.add_argument("--saved_path", type=str, default="trained_models")
#     parser.add_argument("--use_gpu", type=bool, default=True)
#     args = parser.parse_args()
#     return args


def train(opt):
    torch.manual_seed(123)
    if os.path.isdir(opt.log_dir):
        shutil.rmtree(opt.log_dir)
    os.makedirs(opt.log_dir)
    if not os.path.isdir(opt.save_dir):
        os.makedirs(opt.save_dir)
    mp = _mp.get_context("spawn")
    global_model = ActorCritic(num_inputs=3, num_actions=90)
    global_model.load_state_dict(torch.load("{}/a3c_street_fighter".format(opt.saved_path)))
    global_icm = IntrinsicCuriosityModule(num_inputs=3, num_actions=90)
    
    global_icm.load_state_dict(torch.load("{}/icm_street_fighter".format(opt.saved_path)))
    if opt.cuda:
        global_model.cuda()
        global_icm.cuda()
    global_model.share_memory()
    global_icm.share_memory()

    optimizer = GlobalAdam(list(global_model.parameters()) + list(global_icm.parameters()), lr=opt.lr)
    processes = []
    
    # index = 0
    # local_train(index, opt, global_model, global_icm, optimizer, True)
    
    for index in range(opt.num_worker):
        if index == 0:
            process = mp.Process(target=local_train, args=(index, opt, global_model, global_icm, optimizer, True))
        else:
            process = mp.Process(target=local_train, args=(index, opt, global_model, global_icm, optimizer))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
