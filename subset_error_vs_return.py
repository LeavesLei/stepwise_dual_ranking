import os
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset

import d4rl
import d4rl.gym_mujoco
import gym
import torch
import torch.nn as nn
import numpy as np

import pyrallis
import wandb

from utils import set_seed, wrap_env, update_gaussian, eval_actor, normalize_states, compute_mean_std, get_policy, wandb_init, get_pred_acts, eval_test_loss
#from network import TanhGaussianPolicy
from tools.evaluator_cross import TanhGaussianPolicy, FullyConnectedQFunction

import torch.utils.data as tdata
from data_load import D4RLDataset, SubD4RLDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.device(0)

@dataclass
class TrainConfig:
    # Experiment
    device: str = 'cuda'
    env: str = "halfcheetah-medium-replay-v2"  # OpenAI gym environment name
    seed: int = 1  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: str = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load

    policy_path_dir: str = "offline_policy_checkpoints"

    subset_size: int = 256 # size of subset
    avg_time: int = 10

    # Training
    training_epoch: int = 50
    policy_lr: float = 3e-4 # learning rate of policy network
    batch_size: int = 256
    num_workers: int = 8
    eval_interval: int = 500
    eval_episodes: int = 10

    # Algorithm
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = True #True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    q_n_hidden_layers: int = 3  # Number of hidden layers in Q networks
    reward_scale: float = 1.0  # Reward scale for normalization
    reward_bias: float = 0.0  # Reward bias for normalization

    
    # Wandb logging
    project: str = 'KD_OBD (test error vs return 2025-0119)'
    group: str = 'D4RL-BC'
    name: str = 'BC'

    def __post_init__(self):
        self.group = f"{self.group}-size-{self.subset_size}"
        self.name = f"{self.env[:-3]}-seed-{self.seed}"


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant

"""
Load well-trained policy from offline RL
"""
def load_policy(policy_path):
    policy = TanhGaussianPolicy(state_dim, action_dim, max_action, orthogonal_init=True)
    checkpoint = torch.load(policy_path)
    
    policy.load_state_dict(state_dict=checkpoint["actor"])
    policy.eval()

    return policy.to(self.device)


@pyrallis.wrap()
def main(config: TrainConfig):

    # load environment
    env = gym.make(config.env)
    eval_env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # load offline data. d4rl.qlearing_dataset also contains next_observations than d4rl.get_dataset()
    dataset = env.get_dataset()
    full_data_size = dataset['actions'].shape[0]
   
    # state & reward normalization
    if config.normalize_reward:
        modify_reward(
            dataset,
            config.env,
            reward_scale=config.reward_scale,
            reward_bias=config.reward_bias,
        )

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)

    # load pretrained offline model
    offrl_policy = TanhGaussianPolicy(state_dim, action_dim, max_action, orthogonal_init=True).to(config.device)
    offrl_policy_path = os.path.join(config.policy_path_dir, 'Cal-QL-' + config.env, 'checkpoint.pt')
    checkpoint = torch.load(offrl_policy_path)
    offrl_policy.load_state_dict(state_dict=checkpoint["actor"])
    offrl_policy.eval()

    testloader = tdata.DataLoader(D4RLDataset(dataset), batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    pred_actions = get_pred_acts(testloader, offrl_policy, config.avg_time, config.device)
    
    testloader = tdata.DataLoader(D4RLDataset(dataset, pred_actions), batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # make dataloader
    seed = config.seed
    dataset = SubD4RLDataset(dataset, config.subset_size, pred_actions, seed)

    expand = full_data_size // config.subset_size
    expanded_dataset = ConcatDataset([dataset] * expand)
    dataloader = tdata.DataLoader(expanded_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)    

    # set seeds
    set_seed(seed, env)

    # policy network
    actor = TanhGaussianPolicy(state_dim, action_dim, max_action, orthogonal_init=True).to(config.device)
    plr = config.policy_lr

    # training
    wandb_init(asdict(config))
    total_update = 0
    
    if 'medium-expert' in config.env:
        config.training_epoch = config.training_epoch // 2

    for epoch in range(config.training_epoch):
        print("Epoch: %s"%(epoch))
        for i, batch in enumerate(dataloader):
            total_update += 1
            update_info = update_gaussian(actor, plr, batch, config.device)
            wandb.log(update_info, step=total_update)
           
            if total_update % config.eval_interval == 0:
                eval_info = eval_actor(eval_env, actor, device=config.device, n_episodes=config.eval_episodes, seed=config.seed)
                test_loss = eval_test_loss(actor, testloader, config.device)
                wandb.log(eval_info, step=total_update)
                wandb.log({"test loss": test_loss}, step=total_update)

                print(f"Return: {eval_info['normalized_return']}")
                print(f"Test loss: {test_loss}")
    

if __name__ == "__main__":
    main()

