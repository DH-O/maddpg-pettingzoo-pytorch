import numpy as np
import torch


class Buffer:
    """replay buffer for each agent"""
    
    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, act_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=bool)
        """ tar_act 추가요 0723 """
        self.tar_ent = np.zeros(capacity, dtype=np.float32) # np.zeros((capacity))로 되어 있었는데 차이점이 없을 것 같다? 오히려 벗겨내보자.
        """ prob_act 추가요 0802 """
        self.act_log_prob = np.zeros(capacity, dtype=np.float32)

        self._index = 0
        self._size = 0

        self.device = device

    """ tar_act 추가요 0723 """
    def add(self, obs, action, reward, next_obs, done, tar_ent, prob_act):
        """ add an experience to the memory """
        self.obs[self._index] = obs
        self.action[self._index] = action
        self.reward[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.done[self._index] = done
        """ tar_act 추가요  0723 """
        self.tar_ent[self._index] = tar_ent
        """ prob_act 추가요 0802 """
        self.act_log_prob[self._index] = prob_act

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        # retrieve data, Note that the data stored is ndarray
        obs = self.obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        next_obs = self.next_obs[indices]
        done = self.done[indices]
        """ tar_act 추가요 0723 """
        tar_ent = self.tar_ent[indices]
        """ prob_act 추가요 0804 """
        prob_act = self.act_log_prob[indices]
        
        # NOTE that `obs`, `action`, `next_obs` will be passed to network(nn.Module),
        # so the first dimension should be `batch_size`
        obs = torch.from_numpy(obs).float().to(self.device)  # torch.Size([batch_size, state_dim])
        action = torch.from_numpy(action).float().to(self.device)  # torch.Size([batch_size, action_dim]) # 정수형만 나오도록 수정함 0804
        reward = torch.from_numpy(reward).float().to(self.device)  # just a tensor with length: batch_size
        next_obs = torch.from_numpy(next_obs).float().to(self.device)  # Size([batch_size, state_dim])
        done = torch.from_numpy(done).float().to(self.device)  # just a tensor with length: batch_size
        """ tar_act 추가요 0723 """
        tar_ent = torch.from_numpy(tar_ent).float().to(self.device)
        """ prob_act 추가요 0804 """
        prob_act = torch.from_numpy(prob_act).float().to(self.device)
        
        """ tar_act 추가요 0723 """
        """  prob_act 추가요 0804 """
        return obs, action, reward, next_obs, done, tar_ent, prob_act

    def __len__(self):
        return self._size

    def clear(self):
        self.obs.fill(0)
        self.action.fill(0)
        self.reward.fill(0)
        self.next_obs.fill(0)
        self.done.fill(0)
        """ tar_act 추가요 0723 """
        self.tar_ent.fill(0)
        """ prob_act 추가요 0802 """
        self.act_log_prob.fill(0)

        self._index = 0
        self._size = 0