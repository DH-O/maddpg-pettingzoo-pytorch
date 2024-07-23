import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter  # 텐서보드 추가 07/23

from Agent import Agent
from Buffer import Buffer


def setup_logger(filename):
    """ set up logger with filename. """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


class MADDPG:
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent"""

    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir, device=None):
        """ 웬만해선 쿠다로 학습 0723 """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f'training on device: {self.device}')
        
        # sum all the dims of each agent to get input dim for critic
        global_obs_act_dim = sum(sum(val) for val in dim_info.values())
        # create Agent(actor-critic) and replay buffer for each agent
        self.agents = {}
        self.buffers = {}
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.agents[agent_id] = Agent(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr, self.device)   # 쿠다 학습 0723
            self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim, self.device)    # 쿠다 학습 0723
        self.dim_info = dim_info

        self.batch_size = batch_size
        self.res_dir = res_dir  # directory to save the training result
        self.logger = setup_logger(os.path.join(res_dir, 'maddpg.log'))
        self.writer = SummaryWriter(res_dir)    # 텐서보드 추가 0723
        
    def add(self, obs, action, reward, next_obs, done, tar_act):
        # NOTE that the experience is a dict with agent name as its key
        for agent_id in obs.keys():
            o = obs[agent_id]
            a = action[agent_id]
            tar_a = tar_act[agent_id]   # KL 구현 0723
            if isinstance(a, int):
                # the action from env.action_space.sample() is int, we have to convert it to onehot
                a = np.eye(self.dim_info[agent_id][1])[a]
            if isinstance(tar_a, int):
                tar_a = np.eye(self.dim_info[agent_id][1])[tar_a]   # KL 구현 0723

            r = reward[agent_id]
            next_o = next_obs[agent_id]
            d = done[agent_id]
            self.buffers[agent_id].add(o, a, r, next_o, d, tar_a)   # KL 구현 0723

    def sample(self, batch_size):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(self.buffers['agent_0'])
        indices = np.random.choice(total_num, size=batch_size, replace=False)

        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        obs, act, reward, next_obs, done, next_act, tar_act = {}, {}, {}, {}, {}, {}, {}    # KL 구현 0723
        for agent_id, buffer in self.buffers.items():
            o, a, r, n_o, d, t_a = buffer.sample(indices)   # KL 구현 0723
            obs[agent_id] = o
            act[agent_id] = a
            reward[agent_id] = r
            next_obs[agent_id] = n_o
            done[agent_id] = d
            tar_act[agent_id] = t_a # KL 구현 0723
            # calculate next_action using target_network and next_state
            next_act[agent_id] = self.agents[agent_id].target_action(n_o)

        return obs, act, reward, next_obs, done, next_act, tar_act  # KL 구현 0723

    def select_action(self, obs, test=False):   # test 모드에서의 액션 구현 0723
        actions = {}
        for agent, o in obs.items():
            o = torch.from_numpy(o).unsqueeze(0).float().to(self.device)
            if test:
                _, logits = self.agents[agent].action(o, model_out=test)  # torch.Size([1, action_size])    # test 모드에서의 액션 구현 0723
                a = F.gumbel_softmax(logits, hard=True) # test 모드에서의 액션 구현 0723
            else:
                a = self.agents[agent].action(o, model_out=test)  # torch.Size([1, action_size])
            # NOTE that the output is a tensor, convert it to int before input to the environment
            actions[agent] = a.squeeze(0).argmax().item()
            # self.logger.info(f'{agent} action: {actions[agent]}')
        return actions

    def learn(self, batch_size, gamma, use_KL, step):
        
        for agent_id, agent in self.agents.items():
            obs, act, reward, next_obs, done, next_act, tar_act = self.sample(batch_size)   # KL 구현   0723
            # update critic
            critic_value = agent.critic_value(list(obs.values()), list(act.values()))

            # calculate target critic value
            next_target_critic_value = agent.target_critic_value(list(next_obs.values()),
                                                                 list(next_act.values()))
            target_value = reward[agent_id] + gamma * next_target_critic_value * (1 - done[agent_id])

            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
            agent.update_critic(critic_loss)

            # update actor
            # action of the current agent is calculated using its actor
            action, logits = agent.action(obs[agent_id], model_out=True)
            
            target_logits = torch.full((batch_size, action.shape[-1]), -float('inf')).to(self.device)   # KL 구현   0723
            indices = tar_act[agent_id].argmax(dim=-1)  # KL 구현   0723
            target_logits[range(batch_size), indices] = 0   # KL 구현   0723
            
            kl_loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(target_logits, dim=-1), reduction='batchmean')  # KL 구현   0723
            
            act[agent_id] = F.gumbel_softmax(logits, hard=True) # 액션 참값 0723
            actor_loss = -agent.critic_value(list(obs.values()), list(act.values())).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()
            if use_KL:
                agent.update_actor(actor_loss + 1e-4 * actor_loss_pse + 100 * kl_loss)  # KL 구현   0723
            else:
                agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
            self.logger.info(f'agent{agent_id}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}, kl loss: {kl_loss.item()}')
            self.writer.add_scalar(f'agent{agent_id}/critic_loss', critic_loss.item(), step)    # 텐서보드 0723
            self.writer.add_scalar(f'agent{agent_id}/actor_loss', actor_loss.item(), step)  # 텐서보드 0723
            self.writer.add_scalar(f'agent{agent_id}/kl_loss', kl_loss.item(), step)    # 텐서보드 0723

    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)

    def save(self, reward):
        """save actor parameters of all agents and training reward to `res_dir`"""
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},  # actor parameter
            os.path.join(self.res_dir, 'model.pt')
        )
        with open(os.path.join(self.res_dir, 'rewards.pkl'), 'wb') as f:  # save training data
            pickle.dump({'rewards': reward}, f)

    @classmethod
    def load(cls, dim_info, file):
        """init maddpg using the model saved in `file`"""
        instance = cls(dim_info, 0, 0, 0, 0, os.path.dirname(file))
        data = torch.load(file)
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return instance
