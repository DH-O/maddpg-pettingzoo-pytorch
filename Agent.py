from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
""" 액션 노이지한 샘플링을 위함 08/04"""
from torch.distributions import Categorical

class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr, device, args):  # device 추가함 07/23
        
        self.args = args

        self.actor = MLPNetwork(obs_dim, act_dim).to(device)    # device 추가함 07/21
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        
        # critic input all the observations and actions
        # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3)
        self.critic = MLPNetwork(global_obs_dim, 1).to(device)  # device 추가함 07/21
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        
        self.PPO_critic = MLPNetwork(obs_dim, 1).to(device)
        self.PPO_critic_optimizer = Adam(self.PPO_critic.parameters(), lr=critic_lr)
        
        if not self.args.use_PPO_only:
            self.target_actor = deepcopy(self.actor).to(device) # device 추가함 07/21
            self.target_critic = deepcopy(self.critic).to(device)   # device 추가함 07/21
        
        self.device = device    # device 추가함 07/21

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20):
        # NOTE that there is a function like this implemented in PyTorch(torch.nn.functional.gumbel_softmax),
        # but as mention in the doc, it may be removed in the future, so i implement it myself
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    def action(self, obs, ppo=False, agent_idx=None):
        # this method is called in the following two cases:
        # a) interact with the environment
        # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
        # torch.Size([batch_size, state_dim])

        if ppo:
            obs[:, 4 : 4 + 2*agent_idx] = 0
            obs[:, 4 + 2*(agent_idx+1):] = 0
        
        logits = self.actor(obs)  # torch.Size([batch_size, action_size])
        
        if ppo:
            prob_of_actions_or_logits = F.softmax(logits, dim=-1)
            m = Categorical(prob_of_actions_or_logits)
            sample_action = m.sample()
            sample_action = F.one_hot(sample_action, num_classes=logits.shape[1]).int()
        else:
            sample_action = F.gumbel_softmax(logits, hard=True)
            prob_of_actions_or_logits = logits
        
        return sample_action, prob_of_actions_or_logits

    def target_action(self, obs):
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])

        logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])
        action = F.gumbel_softmax(logits, hard=True)
        return action.squeeze(0).detach()

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)  # tensor with a given length
    
    def PPO_critic_value(self, obs):
        return self.PPO_critic(obs).squeeze(1)

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic(x).squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
        self.critic_optimizer.step()


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)
