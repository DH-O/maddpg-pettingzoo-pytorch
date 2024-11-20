from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
""" 액션 노이지한 샘플링을 위함 08/04"""
from torch.distributions import Normal

class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, obs_dim, act_dim, action_space, global_obs_dim, actor_lr, critic_lr, device, args):
        
        self.args = args

        self.actor = Actor(obs_dim, act_dim, action_space, self.args.log_std_max, self.args.log_std_min, self.args.hidden_size, ).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        
        # critic input all the observations and actions
        # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3)
        self.critic = ValueNetwork(global_obs_dim, self.args.hidden_size).to(device)  
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        
        # self.target_actor = deepcopy(self.actor).to(device) 
        self.target_critic = deepcopy(self.critic).to(device)   
        
        self.device = device    

    def action(self, obs, agent_idx=None):
        # torch.Size([batch_size, state_dim])
        
        mean, log_std = self.actor(obs)  # torch.Size([batch_size, action_size])
        std = log_std.exp()
        
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1)) , r stands for reparameterization trick in 'r'sample
        y_t = torch.tanh(x_t)   # action squashed between -1 and 1
        action = y_t * self.actor.action_scale + self.actor.action_bias
        log_prob = normal.log_prob(x_t) # log pdf of normal distribution 구하기. 약간 log(pi(a|s)) for all a 라고 봐도 될 듯
        # Enforcing Action Bound
        log_prob -= torch.log(self.actor.action_scale * (1 - y_t.pow(2)) + 1e-6)    
        # 1e-6은 epsilon 보정. 1 - y_t^2이 0이 되는 것을 방지하기 위함. 1 - (y_t)^2은 tanh의 미분값입니다. 이렇게 하는 이유는 비선형 변환에 대한 변화 변수 공식이란 것을 적용하기 위함
        log_prob = log_prob.sum(1, keepdim=True) # sum of log (prob) == log (product of prob) , 즉 해당 n차원 action이 동시에 일어날 확률
        mean = torch.tanh(mean) * self.actor.action_scale + self.actor.action_bias
        
        return action, log_prob, mean

    def critic_values(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)  # tensor with a given length
        # 1인덱스에 해당하는 차원(두번째 차원)이 1인 경우, 해당 차원을 없애줌

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

def weights_init_(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, in_dim, out_dim, action_space, log_std_max, log_std_min, hidden_dim=64):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, out_dim)
        self.log_std_linear = nn.Linear(hidden_dim, out_dim)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.apply(weights_init_)
        
        """ action rescaling """
        if action_space is None:
            self.register_buffer('action_scale', torch.tensor(1.0))
            self.register_buffer('action_bias', torch.tensor(0.0))
        else:
            action_scale = (action_space.high - action_space.low) / 2.0
            action_bias = (action_space.high + action_space.low) / 2.0
            self.register_buffer('action_scale', torch.FloatTensor(action_scale))
            self.register_buffer('action_bias', torch.FloatTensor(action_bias))

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std    # 리샘플링 없이 깔끔하게 mean, log_std만 반환

class ValueNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim=64) -> None:
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x