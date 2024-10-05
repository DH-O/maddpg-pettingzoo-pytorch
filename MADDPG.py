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


class MADDPG:
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent"""

    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir, args=None, device=None):
        self.args = args
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
        self.buffers_PPO = {}
        self.buffers_MADDPG = {}
        for agent_id, (obs_dim, act_dim) in dim_info.items():
            self.agents[agent_id] = Agent(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr, self.device, self.args)   # 쿠다 학습 0723
            if args is not None:
                if self.args.use_PPO_only:
                    self.buffers_PPO[agent_id] = Buffer(self.args.T_horizon, obs_dim, act_dim, self.device)
                else:
                    self.buffers_PPO[agent_id] = Buffer(capacity * args.update_GUID // (args.update_GUID + args.update_MA), obs_dim, act_dim, self.device)    # 쿠다 학습 0723
                    self.buffers_MADDPG[agent_id] = Buffer(capacity * args.update_MA // (args.update_GUID + args.update_MA), obs_dim, act_dim, self.device)    
        self.dim_info = dim_info

        self.batch_size = batch_size
        self.res_dir = res_dir  # directory to save the training result
        self.logger = setup_logger(os.path.join(res_dir, 'maddpg.log'))
        
        """ 비대칭적 업데이트를 위한 변수 추가 0724 """
        self.update_count_KL = 0
        self.update_KL_flag = True
        self.update_count_MADDPG = 0
        self.update_MADDPG_flag = False
        
        """ 비대칭적 PPO 업데이트를 위한 변수 추가 0801 """
        self.update_coount_PPO = 0
        self.update_PPO_flag = True
        
        
    def add(self, obs, action, reward, next_obs, done, tar_act, prob_act=None, ppo=False):   # KL 구현 0723
        # NOTE that the experience is a dict with agent name as its key
        for agent_id in obs.keys():
            o = obs[agent_id]
            a = action[agent_id]
            tar_a = tar_act[agent_id]   # KL 구현 0723
            if prob_act is not None:   # PPO 구현 0802
                prob_a = prob_act[agent_id] # PPO 구현 0801
            else:
                prob_a = None   # PPO 구현 0802
            
            if isinstance(a, int):
                # the action from env.action_space.sample() is int, we have to convert it to onehot
                a = np.eye(self.dim_info[agent_id][1])[a].astype(int)

            r = reward[agent_id]
            next_o = next_obs[agent_id]
            d = done[agent_id]
            
            if ppo:
                self.buffers_PPO[agent_id].add(o, a, r, next_o, d, tar_a, prob_a)   # KL 구현 0723, PPO 구현 0801
            else:
                self.buffers_MADDPG[agent_id].add(o, a, r, next_o, d, tar_a, prob_a)

    def sample(self, batch_size, ppo=False):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        if ppo:
            total_num = len(self.buffers_PPO['agent_0'])
            if self.args.use_PPO_only:
                assert total_num == self.args.T_horizon, 'PPO buffer size should be equal to T_horizon'
                indices = list(range(0, total_num))
                assert indices == list(range(self.args.T_horizon)), 'PPO buffer should be filled with data in order'
                # while True:
                #     start_index = np.random.randint(0, total_num - self.args.T_horizon + 1)
                #     indices = list(range(start_index, start_index + self.args.T_horizon))
                #     if len(indices) == self.args.T_horizon:
                #         break
            else:
                while True:
                    start_index = np.random.randint(0, total_num - self.args.T_horizon + 1)
                    indices = list(range(start_index, start_index + self.args.T_horizon))
                    if len(indices) == self.args.T_horizon:
                        break
        else:
            total_num = len(self.buffers_MADDPG['agent_0'])
            indices = np.random.choice(total_num, size=batch_size, replace=False)

        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        obs, act, reward, next_obs, done, next_act, tar_ent, prob_sampled_act = {}, {}, {}, {}, {}, {}, {}, {}    # KL 구현 0723                 
        
        if ppo:
            for agent_id, buffer in self.buffers_PPO.items():
                o, a, r, n_o, d, t_e, p_a = buffer.sample(indices)   # KL 구현 0723
                obs[agent_id] = o
                act[agent_id] = a
                reward[agent_id] = r
                next_obs[agent_id] = n_o
                done[agent_id] = d
                
                tar_ent[agent_id] = t_e # KL 구현 0723
                prob_sampled_act[agent_id] = p_a   # PPO 구현 0801
        else:
            for agent_id, buffer in self.buffers_MADDPG.items():
                o, a, r, n_o, d, t_e, p_a = buffer.sample(indices)
                obs[agent_id] = o
                act[agent_id] = a
                reward[agent_id] = r
                next_obs[agent_id] = n_o
                done[agent_id] = d
                
                tar_ent[agent_id] = t_e
                next_act[agent_id] = self.agents[agent_id].target_action(n_o)

        return obs, act, reward, next_obs, done, next_act, tar_ent, prob_sampled_act  # KL 구현 0723

    @torch.no_grad()
    def select_action(self, obs, ppo=False, deterministic=False):   # test 모드에서의 액션 구현 0723
        actions = {}
        prob_act = {}
        agent_idx = 0
        for agent, o in obs.items():
            o = torch.from_numpy(o).unsqueeze(0).float().to(self.device)
            
            if ppo:
                if deterministic:
                    _, prob_of_actions = self.agents[agent].action(o, ppo=ppo, agent_idx=agent_idx)
                    
                    actions[agent] = prob_of_actions.argmax().item()
                    prob_act[agent] = prob_of_actions[0][actions[agent]].item()
                else:
                    a_train, prob_of_actions = self.agents[agent].action(o, ppo=ppo, agent_idx=agent_idx)  # torch.Size([1, action_size])    # train 모드에서의 액션 구현 0805
            
                    # NOTE that the output is a tensor, convert it to int before input to the environment
                    actions[agent] = a_train.squeeze(0).argmax().item()
                    prob_act[agent] = prob_of_actions[0][actions[agent]].item()   # PPO 구현 0801
            else:
                a_train, logits = self.agents[agent].action(o, ppo=ppo)
                actions[agent] = a_train.squeeze(0).argmax().item()
                prob_act[agent] = logits
            
            agent_idx += 1
            
        return actions, prob_act
            # self.logger.info(f'{agent} action: {actions[agent]}')
                
        

    def learn(self, batch_size, gamma, use_KL, kl_coeff, update_guidance, update_MA, step, writer, use_PPO=0):   # PPO 구현 0801
        
        for agent_id, agent in self.agents.items():
            
            if not use_PPO:
                obs, act, reward, next_obs, done, next_act, tar_act, _ = self.sample(batch_size)   # KL 구현   0723
                
                """ 1. Update critic network """
                critic_value = agent.critic_value(list(obs.values()), list(act.values()))   # .values()를 통해 0번 에이전트~ n번 에이전트까지의 obs와 act를 리스트로 만들어준다

                # calculate target critic value
                next_target_critic_value = agent.target_critic_value(list(next_obs.values()),
                                                                    list(next_act.values()))
                target_value = reward[agent_id] + gamma * next_target_critic_value * (1 - done[agent_id])

                critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
                agent.update_critic(critic_loss)

                action, logits = agent.action(obs[agent_id], model_out=True, deterministic=True)
                if torch.isnan(logits).any():
                    print("NaN detected in logits in critic update")
                    import IPython; IPython.embed()
                
                target_logits = torch.full((batch_size, action.shape[-1]), -float('inf')).to(self.device)   # KL 구현   0723
                # indices = tar_act[agent_id].argmax(dim=-1)  # KL 구현   0723
                target_logits[range(batch_size), tar_act[agent_id]] = 0   # KL 구현   0723
                
                act[agent_id] = F.gumbel_softmax(logits, hard=True) # 액션 참값 0723
                actor_loss = -agent.critic_value(list(obs.values()), list(act.values())).mean()
                actor_loss_pse = torch.pow(logits, 2).mean()
            
            if use_KL:
                if self.update_KL_flag:
                    if self.update_count_KL < update_guidance:
                        kl_loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(target_logits, dim=-1), reduction='batchmean')  # KL 구현   0723s
                        agent.update_actor(kl_coeff * kl_loss)  # KL 구현   0723
                        writer.add_scalar(f'agent{agent_id}/kl_loss', kl_loss.item(), step)    # 텐서보드 0723
                        self.update_count_KL += 1
                    else:
                        self.update_KL_flag = False
                        self.update_count_KL = 0
                        
                        self.update_MADDPG_flag = True
                
                elif self.update_MADDPG_flag:
                    if self.update_count_MADDPG < update_MA:
                        agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
                        self.update_count_MADDPG += 1
                    else:
                        self.update_MADDPG_flag = False
                        self.update_count_MADDPG = 0
                        
                        self.update_KL_flag = True
                # self.logger.info(f'agent{agent_id}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}, kl loss: {kl_loss.item()}')
            else:
                if use_PPO:
                    if self.args.use_PPO_only:
                        """ PPO 구현 0801 """
                        obs_p, act_p, reward_p, next_obs_p, done_p, _, tar_e_p, prob_sampled_act_p = self.sample(None, ppo=True)   # PPO 구현   0731
                        with torch.no_grad():
                            vs = agent.PPO_critic_value(obs_p[agent_id])
                            next_vs = agent.PPO_critic_value(next_obs_p[agent_id])
                            deltas = reward_p[agent_id] + gamma * next_vs * (1 - done_p[agent_id]) - vs # shape: [T_horizon]
                            deltas = deltas.cpu().flatten().numpy() # detach().cpu().numpy()는 tensor를 그레디언트 추적에서 제외시키는 역할을 한다.
                            # 그냥 cpu().numpy()만 쓰면 그레디언트 추적이 되기 때문에 flatten()을 써서 1차원으로 만들어준다.
                            
                            adv_p = [0]
                            
                            for delta_t, done_t in zip(deltas[::-1], done_p[agent_id].cpu().flatten().numpy()[::-1]):
                                advantage = gamma * 0.95 * adv_p[-1] * (1 - done_t) + delta_t
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
                            
                            if self.update_coount_PPO < update_guidance:
                                
                                """ PPO 구현 0801 """
                                obs_p, act_p, reward_p, next_obs_p, done_p, next_act_p, tar_e, prob_sampled_act_p = self.sample(None, ppo=True)   # PPO 구현   0731
                                for update_idx in range(self.args.K_epoch):
                                    td_target = reward_p[agent_id] + gamma * agent.critic_value(list(next_obs_p.values()), list(next_act_p.values())) * (1 - done_p[agent_id])
                                    delta = td_target - agent.critic_value(list(obs_p.values()), list(act_p.values()))
                                    delta = delta.detach().cpu().numpy()
                                    
                                    advantage_ls = []
                                    advantage = 0.0
                                    for delta_t in delta[::-1]:
                                        advantage = gamma * 0.95 * advantage + delta_t
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
                                
                                self.update_MADDPG_flag = True
                    
                        elif self.update_MADDPG_flag:
                            if self.update_count_MADDPG < update_MA:
                                agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
                                self.update_count_MADDPG += 1
                            else:
                                self.update_MADDPG_flag = False
                                self.update_count_MADDPG = 0
                                
                                self.update_PPO_flag = True
                else:
                    agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
            
            
            critic_param_norms = []
            critic_grad_norms = []
            
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
        
            writer.add_scalar(f'{agent_id}/critic_param_norm_avg', np.mean(critic_param_norms), step)
            if len(critic_grad_norms) > 0:
                writer.add_scalar(f'{agent_id}/critic_grad_norm_avg', np.mean(critic_grad_norms), step)
            
            actor_param_norms = []
            actor_grad_norms = []
            for name, param in agent.actor.named_parameters():
                actor_param_norms.append(param.data.norm().item())
                if self.args.use_PPO_only:
                    writer.add_scalar(f'{agent_id}/actor_param_norm_{name}', param.data.norm().item(), step)
                if param.grad is not None:
                    actor_grad_norms.append(param.grad.norm().item())
                    if self.args.use_PPO_only:
                        writer.add_scalar(f'{agent_id}/actor_grad_norm_{name}', param.grad.norm().item(), step)
            
            writer.add_scalar(f'{agent_id}/actor_param_norm_avg', np.mean(actor_param_norms), step)
            if len(actor_grad_norms) > 0:
                writer.add_scalar(f'{agent_id}/actor_grad_norm_avg', np.mean(actor_grad_norms), step)
            
                    
            writer.add_scalar(f'{agent_id}/actor_GUID_flag', self.update_KL_flag, step)    # GUID flag 기록 추가 0808
            writer.add_scalar(f'{agent_id}/actor_MADDPG_flag', self.update_MADDPG_flag, step)    # MADDPG flag 기록 추가 0808
            if use_PPO:
                writer.add_scalar(f'{agent_id}/PPO_loss', ppo_loss.mean(), step)    # PPO loss 기록 추가 0808
                writer.add_scalar(f'{agent_id}/entropy_bonus', entropy_bonus.mean(), step)    # Entropy bonus 기록 추가 0808
                writer.add_scalar(f'{agent_id}/value_loss', value_loss, step)    # Value loss 기록 추가 0812
            
            if not self.args.use_PPO_only:
                writer.add_scalar(f'{agent_id}/critic_loss', critic_loss.item(), step)    # 텐서보드 0723
                writer.add_scalar(f'{agent_id}/actor_loss', actor_loss.item(), step)  # 텐서보드 0723

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