import pandas as pd
import os
import random
import re
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

from utils import set_seed, wrap_env, update_gaussian,update_gaussian_with_offline_policy_correction, eval_actor, normalize_states, compute_mean_std, get_policy, wandb_init, compute_dist_weight, get_q_value, get_pred_acts, compute_q_value_weight
#from network import TanhGaussianPolicy
from tools.evaluator_cross import TanhGaussianPolicy, FullyConnectedQFunction

import torch.utils.data as tdata
from data_load import D4RLDataset, SelectionD4RLDataset

from scipy.stats import spearmanr

import warnings
warnings.simplefilter("ignore")



ENV_ORIGIN = {'halfcheetah': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
             'hopper': [1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], \
             'walker2d': [1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}

@dataclass
class TrainConfig:
    # Experiment
    device: str = 'cuda'
    env: str = "halfcheetah-medium-replay-v2"  # OpenAI gym environment name
    seed: int = 2  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: str = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load

    policy_path_dir: str = "offline_policy_checkpoints"
    n_gpu: int = 0

    # Subset Selection
    budget: int = 256

    dist_metric: str = 'exp'
    dist_lambda: float = 1.0

    q_value_avg_time: int = 10
    q_weight_metric: str = 'normalize'
    q_weight_lambda: float = 1.0
    q_value_clip_perc: int = 50 # 0 ~ 100

    lambda_1: float = 0.05
    lambda_2: float = 0.25
    clip_step: int = 999
    
    # q_weight
    use_q_weight: bool = False

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
    project: str = 'SDR-Selection'
    group: str = 'D4RL-SDR'
    name: str = 'SDR'

    def __post_init__(self):
        # self.project = f'{self.project}-LAMBDA-{self.lambda_1}'
        self.group = f"{self.group}-size-{self.budget}"
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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.n_gpu)
    torch.cuda.device(config.n_gpu)

    if 'medium-expert' in config.env:
        config.training_epoch = config.training_epoch // 2
    
    # load environment
    env = gym.make(config.env)
    eval_env = gym.make(config.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # load offline data. d4rl.qlearing_dataset also contains next_observations than d4rl.get_dataset()
    # dataset = env.get_dataset()
    dataset = d4rl.qlearning_dataset(env)
    origin = ENV_ORIGIN[re.split(r'-', config.env)[0]]
    full_data_size = dataset['actions'].shape[0]
   
    # state & reward normalization
    if config.normalize_reward:
        #TODO
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
    origin = normalize_states(origin, state_mean, state_std)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)

    # load log state density
    log_density = np.load("state_density/log_density_maf_" + config.env[:-3] + ".npy")

    # distance between states and origin
    dist = np.sqrt(np.sum((dataset["observations"] - origin)**2, axis=1))
    dist_weight = compute_dist_weight(dist, config.dist_metric, config.dist_lambda)

    # load pretrained offline model
    offrl_policy = TanhGaussianPolicy(state_dim, action_dim, max_action, orthogonal_init=True).to(config.device)
    offrl_critic = FullyConnectedQFunction(state_dim, action_dim).to(config.device)
    offrl_policy_path = os.path.join(config.policy_path_dir, POLICY_PATH_DICT[config.env], 'checkpoint_999999.pt')
    checkpoint = torch.load(offrl_policy_path)
    offrl_policy.load_state_dict(state_dict=checkpoint["actor"])
    offrl_critic.load_state_dict(state_dict=checkpoint["critic1"])
    offrl_policy.eval()
    offrl_critic.eval()

    # get q value
    testloader = tdata.DataLoader(D4RLDataset(dataset), batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    pred_acts = get_pred_acts(testloader, offrl_policy, config.q_value_avg_time, config.device)
    q_value = get_q_value(testloader, offrl_policy, offrl_critic, config.q_value_avg_time, config.device)
    q_value_weight = compute_q_value_weight(q_value, config.q_weight_metric, config.q_value_clip_perc, config.q_weight_lambda)

    # combine dist_weight and q_value_weight
    select_weight =  [a * b for a, b in zip(dist_weight, q_value_weight)]
    sample_num = dataset['actions'].shape[0]
    traj_start_list = [0]
    traj_len_list = []
    tmp_start = 0
    for i in range(sample_num - 1):
        if (dataset['observations'][i+1] != dataset['next_observations'][i]).any() or dataset['terminals'][i]:
            traj_len_list.append(i + 1 - tmp_start)
            tmp_start = i + 1
            traj_start_list.append(tmp_start)
    
    def split_list_by_indices(a, b):
        segments = []
        for i in range(len(b)):
            start = b[i]
            end = b[i + 1] if i + 1 < len(b) else len(a)
            segments.append(a[start:end])
        return segments

    traj_dist = split_list_by_indices(dist, traj_start_list)
    traj_q_value = split_list_by_indices(q_value, traj_start_list)
    select_weight = split_list_by_indices(select_weight, traj_start_list)
    log_density = split_list_by_indices(log_density, traj_start_list)

    data = {
    "step": [step for traj in range(len(traj_dist)) for step in range(len(traj_dist[traj]))],
    "traj_dist": [point for traj in traj_dist for point in traj],
    "traj_q_value": [point.item() for traj in traj_q_value for point in traj],
    "select_weight": [point.item() for traj in select_weight for point in traj],
    "log_density": [point for traj in log_density for point in traj]
    }
    df = pd.DataFrame(data)
    
    def value_func_quantile(step):
        return config.lambda_1 * np.tanh(step/100)

    def density_func_quantile(step):
        return 1 - config.lambda_2 * np.tanh(step/100)


    def filter_quantile_percent(group):
        step = group.name
        value_quantile = value_func_quantile(step)
        density_quantile = density_func_quantile(step)
        density_threshold = group["log_density"].quantile(density_quantile)
        value_threshold = group["traj_q_value"].quantile(value_quantile)
        return group[(group["log_density"] <= density_threshold) & (group["traj_q_value"] >= value_threshold)]

    filtered_df = df.groupby("step").apply(filter_quantile_percent).reset_index(level='step', drop=True)
    filtered_idx = filtered_df.index.tolist()

    # make dataloader
    seed = config.seed
    subset = SelectionD4RLDataset(dataset, filtered_idx, config.budget, seed, pred_acts)
    # expand =  full_data_size // len(filtered_idx) #config.budget 
    expand =  full_data_size // config.budget 
    expanded_subset = ConcatDataset([subset] * int(expand))
    dataloader = tdata.DataLoader(expanded_subset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)    

    # set seeds
    set_seed(seed, env)

    # policy network
    actor = TanhGaussianPolicy(state_dim, action_dim, max_action, orthogonal_init=True).to(config.device)
    plr = config.policy_lr

    # training
    wandb_init(asdict(config))
    total_update = 0

    for epoch in range(config.training_epoch):
        print("Epoch: %s"%(epoch))
        for i, batch in enumerate(dataloader):
            total_update += 1
            update_info = update_gaussian(actor, plr, batch, config.device)
            # update_info = update_gaussian_with_offline_policy_correction(env, actor, plr, batch, offrl_policy, offrl_critic, config.use_q_weight, config.device)
            wandb.log(update_info, step=total_update)
           
            if total_update % config.eval_interval == 0:
                eval_info = eval_actor(eval_env, actor, device=config.device, n_episodes=config.eval_episodes, seed=config.seed)
                wandb.log(eval_info, step=total_update)
                print(eval_info["normalized_return"])
    
if __name__ == "__main__":
    main()