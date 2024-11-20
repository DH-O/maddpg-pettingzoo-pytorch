import logging
import os
import pickle
import math

import numpy as np
import torch
import torch.nn.functional as F

from copy import deepcopy
from Agent import Agent
from Buffer import Buffer
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader

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


class MASAC():
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent"""

    def __init__(self, dim_info, action_spaces, save_dir, args=None, device=None):
        self.args = args
        """ 웬만해선 쿠다로 학습 0723 """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f'training on device: {self.device}')
        
        # sum all the dims of each agent to get input dim for critic
        global_obs_act_dim = sum(sum(val) for val in dim_info.values()) # act_dim이 5차원이 나올때 0번째 차원은 소통 액션이라는 것
        # create Agent(actor-critic) and replay buffer for each agent
        self.agents = {}
        self.buffers_MASAC = {}
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.agents[agent_id] = Agent(obs_dim, act_dim, action_spaces, global_obs_act_dim, self.args.actor_lr, self.args.critic_lr, self.device, self.args)   # 쿠다 학습 0723
            if self.args.use_GUID:
                self.buffers_MASAC[agent_id] = Buffer(self.args.buffer_capacity * self.args.MA_update_interval // (self.args.GUID_update_interval + self.args.MA_update_interval), obs_dim, act_dim, self.device)
            else:
                self.buffers_MASAC[agent_id] = Buffer(self.args.buffer_capacity, obs_dim, act_dim, self.device)
                
        self.dim_info = dim_info

        self.batch_size = self.args.batch_size
        self.gamma = self.args.gamma
        self.save_dir = save_dir  # directory to save the training result
        self.logger = setup_logger(os.path.join(save_dir, 'maddpg.log'))
        
        if self.args.auto_entropy_MASAC:
            self.target_entropy = -np.prod(dim_info['agent_0'][1])  # target entropy is -|A|, A is the action space
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.args.alpha_lr)
        self.alpha = self.args.alpha_MASAC
        
        """ 비대칭적 업데이트를 위한 변수 24/11/11"""
        self.use_GUID = self.args.use_GUID
        if self.use_GUID:
            self.GUID_update_interval = self.args.GUID_update_interval
            self.guid_coeff = self.args.guid_coeff
            self.update_count_METRA = 0
            self.update_METRA_flag = False
            self.update_count_MA = 0
            self.update_MA_flag = False
        
        
    def add(self, obs_s, actions, rewards, next_obs_s, dones, tar_acts, prob_act=None, ppo=False):   # KL 구현 0723
        # NOTE that the experience is a dict with agent name as its key
        for agent_id in obs_s.keys():
            o = obs_s[agent_id]
            a = actions[agent_id]
            tar_a = tar_acts[agent_id]   # KL 구현 0723. tar_acts는 추종해야할 액션을 의미
            if prob_act is not None:   # PPO 구현 0802
                prob_a = prob_act[agent_id] # PPO 구현 0801
            else:
                prob_a = None   # PPO 구현 0802

            r = rewards[agent_id]
            next_o = next_obs_s[agent_id]
            d = dones[agent_id]
            
            self.buffers_MASAC[agent_id].add(o, a, r, next_o, d, tar_a, prob_a)

    def sample(self, batch_size, ppo=False):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(self.buffers_MASAC['agent_0'])
        indices = np.random.choice(total_num, size=batch_size, replace=False)

        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        obs_s, acts, rewards, next_obs_s, dones, next_acts, tar_ents, probs_sampled_acts = {}, {}, {}, {}, {}, {}, {}, {}    # KL 구현 0723                 
        
        for agent_id, buffer in self.buffers_MASAC.items():
            o, a, r, n_o, d, t_e, p_a = buffer.sample(indices)
            obs_s[agent_id] = o
            acts[agent_id] = a
            rewards[agent_id] = r
            next_obs_s[agent_id] = n_o
            dones[agent_id] = d
            
            tar_ents[agent_id] = t_e
            _, _, next_acts[agent_id] = self.agents[agent_id].action(n_o)

        return obs_s, acts, rewards, next_obs_s, dones, next_acts, tar_ents, probs_sampled_acts  # KL 구현 0723

    @torch.no_grad()
    def select_action(self, obs, evaluate=False, ppo=False, deterministic=False):   # test 모드에서의 액션 구현 0723
        actions = {}
        prob_act = {}
        agent_idx = 0
        for agent, o in obs.items():
            o = torch.from_numpy(o).unsqueeze(0).float().to(self.device)
            if deterministic:
                _, prob_of_actions = self.agents[agent].action(o, ppo=ppo, agent_idx=agent_idx)
                
                actions[agent] = prob_of_actions.argmax().item()
                prob_act[agent] = prob_of_actions[0][actions[agent]].item()
            else:
                if evaluate:
                    _, _, action = self.agents[agent].action(o, agent_idx=agent_idx)  # torch.Size([1, action_size])
                else:
                    action, log_prob, mean = self.agents[agent].action(o, agent_idx=agent_idx)  # torch.Size([1, action_size])    # train 모드에서의 액션 구현 0805
        
                # # NOTE that the output is a tensor, convert it to int before input to the environment
                actions[agent] = action.detach().cpu().numpy()[0]
            
            agent_idx += 1
            
        return actions, prob_act

    def learn(self, step, writer, use_PPO=0):   # PPO 구현 0801
        
        if self.buffers_MASAC['agent_0'].__len__() >= self.batch_size:
            for _ in range(self.args.num_updates):
                for agent_id, agent in self.agents.items():
                    if not use_PPO:
                        obs_s, act_s, rewards, next_obs_s, dones, next_acts, _, _ = self.sample(self.batch_size)   # KL 구현   0723
                        
                        """ 1. Update critic network """
                        critic_values = agent.critic_values(list(obs_s.values()), list(act_s.values()))   # .values()를 통해 0번 에이전트~ n번 에이전트까지의 obs와 act를 리스트로 만들어준다

                        # calculate target critic value
                        next_target_critic_value = agent.target_critic_value(list(next_obs_s.values()), list(next_acts.values()))
                        target_value = rewards[agent_id] + self.gamma * next_target_critic_value * (1 - dones[agent_id])

                        critic_loss = F.mse_loss(critic_values, target_value.detach(), reduction='mean')
                        agent.update_critic(critic_loss)

                        """ 2. Update actor network """
                        resampled_action, logprobs, _ = agent.action(obs_s[agent_id])
                        if torch.isnan(logprobs).any():
                            print("NaN detected in logits in critic update")
                            import IPython; IPython.embed()
                        act_s[agent_id] = resampled_action
                        # target_logits = torch.full((self.batch_size, action.shape[-1]), -float('inf')).to(self.device)   # KL 구현   0723
                        # # indices = tar_act[agent_id].argmax(dim=-1)  # KL 구현   0723
                        # target_logits[range(self.batch_size), tar_ents[agent_id]] = 0   # KL 구현   0723
                        
                        actor_loss = (-agent.critic_values(list(obs_s.values()), list(act_s.values())) + self.alpha * logprobs).mean()
                        agent.update_actor(actor_loss)
                        
                        if self.args.auto_entropy_MASAC:
                            alpha_loss = -(self.log_alpha * (logprobs + self.target_entropy).detach()).mean()
                            self.alpha_optimizer.zero_grad()
                            alpha_loss.backward()
                            # torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
                            self.alpha_optimizer.step()
                            self.alpha = torch.clamp(self.log_alpha.exp(), min=1e-5, max=0.9)
                        
                    if self.use_GUID:
                        if self.update_METRA_flag:
                            if self.update_count_METRA < self.GUID_update_interval:
                                kl_loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(target_logits, dim=-1), reduction='batchmean')  # KL 구현   0723s
                                agent.update_actor(self.guid_coeff * kl_loss)  # KL 구현   0723
                                writer.add_scalar(f'agent{agent_id}/kl_loss', kl_loss.item(), step)    # 텐서보드 0723
                                self.update_count_METRA += 1
                            else:
                                self.update_METRA_flag = False
                                self.update_count_METRA = 0
                                
                                self.update_MA_flag = True
                        
                        elif self.update_MA_flag:
                            if self.update_count_MA < self.MA_update_interval:
                                agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
                                self.update_count_MA += 1
                            else:
                                self.update_MA_flag = False
                                self.update_count_MA = 0
                                
                                self.update_METRA_flag = True
                        # self.logger.info(f'agent{agent_id}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}, kl loss: {kl_loss.item()}')
                    else:
                        if use_PPO:
                            if self.args.use_PPO_only:
                                """ PPO 구현 0801 """
                                obs_p, act_p, reward_p, next_obs_p, done_p, _, tar_e_p, prob_sampled_act_p = self.sample(None, ppo=True)   # PPO 구현   0731
                                with torch.no_grad():
                                    vs = agent.PPO_critic_value(obs_p[agent_id])
                                    next_vs = agent.PPO_critic_value(next_obs_p[agent_id])
                                    deltas = reward_p[agent_id] + self.gamma * next_vs * (1 - done_p[agent_id]) - vs # shape: [T_horizon]
                                    deltas = deltas.cpu().flatten().numpy() # detach().cpu().numpy()는 tensor를 그레디언트 추적에서 제외시키는 역할을 한다.
                                    # 그냥 cpu().numpy()만 쓰면 그레디언트 추적이 되기 때문에 flatten()을 써서 1차원으로 만들어준다.
                                    
                                    adv_p = [0]
                                    
                                    for delta_t, done_t in zip(deltas[::-1], done_p[agent_id].cpu().flatten().numpy()[::-1]):
                                        advantage = self.gamma * 0.95 * adv_p[-1] * (1 - done_t) + delta_t
                                        adv_p.append(advantage)
                                    adv_p.reverse()
                                    adv_p = deepcopy(adv_p[0:-1])
                                    adv_p = torch.tensor(adv_p, dtype=torch.float).to(self.device)
                                    td_target = adv_p + vs
                                
                                optim_iter_num = int(math.ceil(len(obs_p[agent_id]) / self.args.batch_size))
                                
                                for update_idx in range(self.args.K_epoch):
                                    idx = torch.randperm(len(obs_p[agent_id]))
                                    o_p, a_p, td_target_p, adv_p, old_prob_a_p, t_e_p = \
                                        obs_p[agent_id][idx].clone(), act_p[agent_id][idx].clone(), td_target[idx].clone(), adv_p[idx].clone(), prob_sampled_act_p[agent_id][idx].clone(), tar_e_p[agent_id][idx].clone()
                                    
                                    for i in range(optim_iter_num):
                                        index = slice(i * self.args.batch_size, min((i + 1) * self.args.batch_size, o_p.shape[0])) # index를 batch_size만큼 나눠준다
                                        
                                        _, prob_of_actions_p = agent.action(o_p[index], ppo=True, agent_idx=t_e_p[index][0].item())
                                        entropy = Categorical(prob_of_actions_p).entropy().sum(0, keepdim=True) # 행에 따라, 즉 5차원 액션에 따라 엔트로피가 계산된다. 이걸 0차원 따라 다 더한 것이다.
                                        prob_a_p = prob_of_actions_p.gather(1, a_p[index].argmax(dim=-1).unsqueeze(-1)) # 액션 분포에 따른 softmax 값이 가장 높은 인덱스
                                        ratio = torch.exp(torch.log(prob_a_p) - torch.log(old_prob_a_p[index]))
                                        
                                        surr1 = ratio * adv_p[index]
                                        surr2 = torch.clamp(ratio, 1 - self.args.epsilon, 1 + self.args.epsilon) * adv_p[index]
                                        ppo_loss = -torch.min(surr1, surr2)
                                        entropy_bonus = -self.args.entropy_coeff * entropy
                                        policy_loss = ppo_loss + entropy_bonus
                                        
                                        agent.actor_optimizer.zero_grad()
                                        policy_loss.mean().backward()
                                        clip_grad_norm_(agent.actor.parameters(), self.args.max_grad_norm)
                                        agent.actor_optimizer.step()
                                    
                                        
                                        value_loss = self.args.value_coeff * F.mse_loss(agent.PPO_critic_value(o_p[index]), td_target_p[index])
                                        
                                        agent.PPO_critic_optimizer.zero_grad()
                                        value_loss.backward()
                                        clip_grad_norm_(agent.PPO_critic.parameters(), self.args.max_grad_norm)
                                        agent.PPO_critic_optimizer.step()
                                
                                self.buffers_PPO[agent_id].clear()
                            else:
                                if self.update_PPO_flag:
                                    
                                    if self.update_coount_PPO < self.GUID_update_interval:
                                        
                                        """ PPO 구현 0801 """
                                        obs_p, act_p, reward_p, next_obs_p, done_p, next_act_p, tar_e, prob_sampled_act_p = self.sample(None, ppo=True)   # PPO 구현   0731
                                        for update_idx in range(self.args.K_epoch):
                                            td_target = reward_p[agent_id] + self.gamma * agent.critic_value(list(next_obs_p.values()), list(next_act_p.values())) * (1 - done_p[agent_id])
                                            delta = td_target - agent.critic_value(list(obs_p.values()), list(act_p.values()))
                                            delta = delta.detach().cpu().numpy()
                                            
                                            advantage_ls = []
                                            advantage = 0.0
                                            for delta_t in delta[::-1]:
                                                advantage = self.gamma * 0.95 * advantage + delta_t
                                                advantage_ls.append([advantage])
                                            advantage_ls.reverse()
                                            advantage = torch.tensor(advantage_ls, dtype=torch.float).to(self.device)
                                            
                                            _, prob_of_actions_p = agent.action(obs_p[agent_id], ppo=True, agent_idx=tar_e[agent_id][0].item())
                                            
                                            pi = F.softmax(prob_of_actions_p, dim=-1)
                                            act_p_index = act_p[agent_id].argmax(dim=-1).unsqueeze(-1)  # 액션 분포에 따른 softmax 값이 가장 높은 인덱스를 뽑아온다
                                            pi_a = pi.gather(1, act_p_index)    
                                            ratio = torch.exp(torch.log1p(pi_a) - torch.log1p(prob_sampled_act_p[agent_id].unsqueeze(-1)))
                                            
                                            surr1 = ratio * advantage
                                            surr2 = torch.clamp(ratio, 1 - self.args.epsilon, 1 + self.args.epsilon) * advantage
                                            loss_ppo = -torch.min(surr1, surr2) + F.smooth_l1_loss(agent.critic_value(list(obs_p.values()), list(act_p.values())), td_target.detach())
                                        
                                            agent.actor_optimizer.zero_grad()
                                            agent.PPO_critic_optimizer.zero_grad()
                                            loss_ppo.mean().backward()
                                            clip_grad_norm_(agent.actor.parameters(), self.args.max_grad_norm)
                                            agent.actor_optimizer.step()
                                            clip_grad_norm_(agent.PPO_critic.parameters(), self.args.max_grad_norm)
                                            agent.PPO_critic_optimizer.step()
                                            
                                            self.update_coount_PPO += 1
                                    else:
                                        self.update_PPO_flag = False
                                        self.update_coount_PPO = 0
                                        
                                        self.update_MA_flag = True
                            
                                elif self.update_MA_flag:
                                    if self.update_count_MA < self.MA_update_interval:
                                        agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
                                        self.update_count_MA += 1
                                    else:
                                        self.update_MA_flag = False
                                        self.update_count_MA = 0
                                        
                                        self.update_PPO_flag = True
                    
                    critic_param_norms = []
                    critic_grad_norms = []
                    target_critic_param_norms = []
                    target_critic_grad_norms = []
                    
                    if use_PPO:
                        for param in agent.PPO_critic.parameters():
                            critic_param_norms.append(param.data.norm().item())
                            if param.grad is not None:
                                critic_grad_norms.append(param.grad.norm().item())
                    else:
                        for param in agent.critic.parameters():
                            critic_param_norms.append(param.data.norm().item())
                            if param.grad is not None:
                                critic_grad_norms.append(param.grad.norm().item())
                        
                        for param in agent.target_critic.parameters():
                            target_critic_param_norms.append(param.data.norm().item())
                            if param.grad is not None:
                                target_critic_grad_norms.append(param.grad.norm().item())
                
                    writer.add_scalar(f'{agent_id}/critic_param_norm_avg', np.mean(critic_param_norms), step)
                    writer.add_scalar(f'{agent_id}/target_critic_param_norm_avg', np.mean(target_critic_param_norms), step)
                    if len(critic_grad_norms) > 0:
                        writer.add_scalar(f'{agent_id}/critic_grad_norm_avg', np.mean(critic_grad_norms), step)
                        writer.add_scalar(f'{agent_id}/target_critic_grad_norm_avg', np.mean(target_critic_grad_norms), step)
                    
                    actor_param_norms = []
                    actor_grad_norms = []
                    for param in agent.actor.parameters():
                        actor_param_norms.append(param.data.norm().item())
                        if param.grad is not None:
                            actor_grad_norms.append(param.grad.norm().item())
                    
                    writer.add_scalar(f'{agent_id}/actor_param_norm_avg', np.mean(actor_param_norms), step)
                    if len(actor_grad_norms) > 0:
                        writer.add_scalar(f'{agent_id}/actor_grad_norm_avg', np.mean(actor_grad_norms), step)
                    writer.add_scalar(f'{agent_id}/alpha', self.alpha, step)
                    
                    # writer.add_scalar(f'{agent_id}/actor_GUID_flag', self.update_METRA_flag, step)    # GUID flag 기록 추가 0808
                    # writer.add_scalar(f'{agent_id}/actor_MADDPG_flag', self.update_MA_flag, step)    # MADDPG flag 기록 추가 0808
                    
                    writer.add_scalar(f'{agent_id}/critic_loss', critic_loss.item(), step)    # 텐서보드 0723
                    writer.add_scalar(f'{agent_id}/actor_loss', actor_loss.item(), step)  # 텐서보드 0723

    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            # soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)
    
    def save(self, reward):
        """save actor parameters of all agents and training reward to `res_dir`"""
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},  # actor parameter
            os.path.join(self.save_dir, 'model.pt')
        )
        with open(os.path.join(self.save_dir, 'rewards.pkl'), 'wb') as f:  # save training data
            pickle.dump({'rewards': reward}, f)

    @classmethod
    def load(cls, dim_info, file, args):
        """init maddpg using the model saved in `file`"""
        instance = cls(dim_info, 0, 0, 0, 0, os.path.dirname(file), args) # 우와 이게 __init__에 건네주는 아규먼트기도 했다
        data = torch.load(file)
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return instance


                            # _, prob_of_actions_p = agent.action(obs_p[agent_id], ppo=True, agent_idx=tar_e[agent_id][0].item())
                            
                            # # pi = F.softmax(prob_p, dim=-1)
                            # # act_p_index = act_p[agent_id].argmax(dim=-1).unsqueeze(-1)  # 액션 분포에 따른 softmax 값이 가장 높은 인덱스를 뽑아온다
                            # # pi_a = pi.gather(1, act_p_index)    
                            # # ratio = torch.exp(torch.log(pi_a + 1e-10) - torch.log(prob_act[agent_id].unsqueeze(-1) + 1e-10))
                            
                            # # Entropy term 추가
                            # entropy = -torch.sum(pi * torch.log(pi + 1e-10), dim=-1)  # Entropy 계산
                            # entropy_bonus = self.args.entropy_coeff * entropy.mean()    # Entropy coefficient 적용
                                
                            # surr1 = ratio * advantage
                            # surr2 = torch.clamp(ratio, 1 - self.args.epsilon, 1 + self.args.epsilon) * advantage
                            # loss_ppo = -torch.min(surr1, surr2) + F.smooth_l1_loss(agent.PPO_critic_value(obs_p), td_target.detach())

                            # total_loss = loss_ppo.mean() - entropy_bonus
                            
                            # agent.actor_optimizer.zero_grad()
                            # agent.PPO_critic_optimizer.zero_grad()
                            # total_loss.backward()
                            # clip_grad_norm_(agent.actor.parameters(), self.args.max_grad_norm)
                            # agent.actor_optimizer.step()
                            # clip_grad_norm_(agent.PPO_critic.parameters(), self.args.max_grad_norm)
                            # agent.PPO_critic_optimizer.step()