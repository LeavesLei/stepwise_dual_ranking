import numpy as np
import torch
from torch.utils.data import Dataset

class D4RLDataset(Dataset):
    def __init__(self, dataset, pred_acts=None):
        self.dataset = dataset

        self.pred_acts = pred_acts
        self.obs = self.dataset["observations"]
        self.acts = self.dataset["actions"]
        self.next_obs = self.dataset["observations"]
        self.rews = self.dataset["rewards"]
        self.dones = self.dataset["terminals"]

        self.len = self.obs.shape[0]

    def __getitem__(self, index):
        data = {
            "obs": self.obs[index],
            "acts": self.acts[index],
            "next_obs": self.next_obs[index],
            "rews": self.rews[index, np.newaxis],
            "dones": self.dones[index, np.newaxis]
        }

        if self.pred_acts is not None:
            data["pred_acts"] = self.pred_acts[index]
        
        return data

    def __len__(self):
        return self.len # return self.len - 1


class SubD4RLDataset(Dataset):
    def __init__(self, dataset, size, pred_acts=None, seed=0):
        self.dataset = dataset

        torch.manual_seed(seed)
        perm = torch.randperm(dataset['observations'].shape[0])
        selected_idx = perm[:size]

        if pred_acts is not None:
            self.pred_acts = pred_acts[selected_idx]
        else:
            self.pred_acts = pred_acts
        self.obs = self.dataset["observations"][selected_idx]
        self.acts = self.dataset["actions"][selected_idx]
        self.next_obs = self.dataset["observations"][selected_idx]
        self.rews = self.dataset["rewards"][selected_idx]
        self.dones = self.dataset["terminals"][selected_idx]

        self.len = self.obs.shape[0]

    def __getitem__(self, index):
        data = {
            "obs": self.obs[index],
            "acts": self.acts[index],
            "next_obs": self.next_obs[index],
            "rews": self.rews[index, np.newaxis],
            "dones": self.dones[index, np.newaxis]
        }

        if self.pred_acts is not None:
            data["pred_acts"] = self.pred_acts[index]

        return data

    def __len__(self):
        return self.len # self.len-1


class SelectedD4RLDataset(Dataset):
    def __init__(self, dataset, selected_idx, pred_acts=None):
        self.dataset = dataset

        if pred_acts is not None:
            self.pred_acts = pred_acts[selected_idx]
        else:
            self.pred_acts = pred_acts
        self.obs = self.dataset["observations"][selected_idx]
        self.acts = self.dataset["actions"][selected_idx]
        self.next_obs = self.dataset["observations"][selected_idx]
        self.rews = self.dataset["rewards"][selected_idx]
        self.dones = self.dataset["terminals"][selected_idx]

        self.len = self.obs.shape[0]

    def __getitem__(self, index):
        data = {
            "obs": self.obs[index],
            "acts": self.acts[index],
            "next_obs": self.next_obs[index],
            "rews": self.rews[index, np.newaxis],
            "dones": self.dones[index, np.newaxis]
        }

        if self.pred_acts is not None:
            data["pred_acts"] = self.pred_acts[index]

        return data

    def __len__(self):
        return self.len # self.len-1
        

class SubTrajectD4RLDataset(Dataset):
    """
    traj_ratio: the subset ratio of trajectory
    segment_len: the length of a segment of trajectory
    segment_index: the index of segment
    """
    def __init__(self, dataset, traj_ratio=0.1, segment_num=3, segment_index=0, seed=0):
        self.dataset = dataset
        traj_start_list, traj_len_list = self.traj_stats(self.dataset)

        traj_num = len(traj_start_list)
        torch.manual_seed(seed)
        perm = torch.randperm(traj_num)
        traj_idx = perm[:int(traj_num * traj_ratio)]
        
        selected_idx = []
        for traj_id in traj_idx:
            segment_len = traj_len_list[traj_id] // segment_num
            
            if segment_index >= 0:
                segment_start = traj_start_list[traj_id] + segment_index * segment_len
                segment = list(range(segment_start, segment_start + segment_len))
            else:
                torch.manual_seed(seed)
                perm = torch.randperm(traj_len_list[traj_id])
                segment = traj_start_list[traj_id] + perm[: segment_len] # start point offset
            selected_idx.extend(segment)

        self.obs = self.dataset["observations"][selected_idx]
        self.acts = self.dataset["actions"][selected_idx]
        self.next_obs = self.dataset["observations"][selected_idx]
        self.rews = self.dataset["rewards"][selected_idx]
        self.dones = self.dataset["terminals"][selected_idx]

        self.len = self.obs.shape[0]

    def __getitem__(self, index):
        return {
            "obs": self.obs[index],
            "acts": self.acts[index],
            "next_obs": self.next_obs[index],
            "rews": self.rews[index, np.newaxis],
            "dones": self.dones[index, np.newaxis]
        }

    def __len__(self):
        return self.len-1

    @staticmethod
    def traj_stats(dataset):
        sample_num = dataset['actions'].shape[0]
        traj_start_list = [0]
        traj_len_list = []
        tmp_start = 0
        for i in range(sample_num - 1):
            if (dataset['observations'][i+1] != dataset['next_observations'][i]).any():
                traj_len_list.append(i + 1 - tmp_start)
                tmp_start = i + 1
                traj_start_list.append(tmp_start)
        
        traj_start_list.pop() # drop the last (imcomplete) trajectory in dataset
        return traj_start_list, traj_len_list


class SelectionD4RLDataset(Dataset):
    #def __init__(self, dataset, weight, budget):
    def __init__(self, dataset, filtered_idx, size, seed=0, pred_acts=None):
        self.dataset = dataset

        # torch.manual_seed(seed)
        np.random.seed(seed)
        perm = np.random.permutation(len(filtered_idx))

        # perm = torch.randperm(len(filtered_idx))
        filtered_idx = np.array(filtered_idx)
        selected_idx = filtered_idx[perm[:size]]
        
        # sorted_indices = np.argsort(weight)[::-1]
        # top_idx = sorted_indices[:budget]
        # top_idx = selected_idx
        # selected_idx = filtered_idx
        if pred_acts is not None:
            self.pred_acts = pred_acts[selected_idx]
        else:
            self.pred_acts = pred_acts

        self.obs = self.dataset["observations"][selected_idx]
        self.acts = self.dataset["actions"][selected_idx]
        self.next_obs = self.dataset["observations"][selected_idx]
        self.rews = self.dataset["rewards"][selected_idx]
        self.dones = self.dataset["terminals"][selected_idx]

        self.len = self.obs.shape[0]

    def __getitem__(self, index):
        data = {
            "obs": self.obs[index],
            "acts": self.acts[index],
            "next_obs": self.next_obs[index],
            "rews": self.rews[index, np.newaxis],
            "dones": self.dones[index, np.newaxis]
        }

        if self.pred_acts is not None:
            data["pred_acts"] = self.pred_acts[index]
        
        return data

    def __len__(self):
        return self.len