import os
import json
import torch
import argparse

import numpy as np

from PIL import Image, ImageDraw
from datetime import datetime
from tensorboardX import SummaryWriter
from algorithm.MASAC import MASAC

""" METRA 관련 """
import itertools
from replay_memory import ReplayMemory
from algorithm.SAC.sac import SAC
from algorithm.METRA.model_metra import Phi, Lambda
from algorithm.METRA.utils_metra import generate_skill_cont
""" =================================== """

from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2

def get_env(env_name, ep_len=25, num_agents=3, map_size=1.0):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v2':
        new_env = simple_adversary_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_spread_v2':
        new_env = simple_spread_v2.parallel_env(max_cycles=ep_len, num_agents=num_agents, map_size=map_size, continuous_actions=True)
    if env_name == 'simple_tag_v2':
        new_env = simple_tag_v2.parallel_env(max_cycles=ep_len)

    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).shape[0]) # continuous action space일 경우

    return new_env, _dim_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, default='simple_adversary_v2', help='name of the env',
                        choices=['simple_adversary_v2', 'simple_spread_v2', 'simple_tag_v2'])
    parser.add_argument('--episode_num', type=int, default=30000,   # 30000이 좀 적당하긴 하다
                        help='total episode num during training procedure')
    parser.add_argument('--num_updates', type=int, default=1,
                        help='number of updates per learning step')
    parser.add_argument('--random_steps', type=int, default=10000,    # 5e4인데 잠깐만 앞당김
                        help='random steps before the agent start to learn')
    
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N', help='hidden size (default: 256)')
    parser.add_argument('--tau', type=float, default=0.0025, help='soft update parameter')    # 이거 높을수록 지금 막 업데이트 된 네트워크를 중요하게 생각하겠다는 의미
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')    # 이거 높을수록 멀리 있는 리워드도 중요하게 생각하겠다는 의미
    parser.add_argument('--alpha_MASAC', type=float, default=0.1, help='Temperature parameter α')    # 이거 높을수록 entropy를 높게 설정하겠다는 의미
    
    parser.add_argument('--buffer_capacity', type=int, default=int(1e7), help='capacity of replay buffer')  # 1e7이었음
    parser.add_argument('--batch_size', type=int, default=256, help='batch-size of replay buffer')
    
    parser.add_argument('--actor_lr', type=float, default=0.00005, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.00005, help='learning rate of critic')
    parser.add_argument('--alpha_lr', type=float, default=0.0001, help='learning rate of alpha')
    
    parser.add_argument('--auto_entropy_MASAC', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: True)')
    parser.add_argument('--max_grad_norm', type=float, default=100, help='max norm of gradients')
    parser.add_argument('--log_std_max', type=float, default=1.2, help='max log std')
    parser.add_argument('--log_std_min', type=float, default=-12, help='min log std')
    
    parser.add_argument('--use_GUID', type=int, default=0, help='use guidance action')
    # parser.add_argument('--GUID_update_interval', type=int, default=-1, help='update GUID every n steps')
    # parser.add_argument('--guid_coeff', type=float, default=100, help='coefficient for KL divergence')
    # parser.add_argument('--MA_update_interval', type=int, default=-1, help='update MADDPG every n steps')
    
    """ PPO 관련한 애들"""
    # parser.add_argument('--entropy_coeff', type=float, default=10, help='entropy coefficient for PPO')
    # parser.add_argument('--value_coeff', type=float, default=0.1, help='value coefficient for PPO')
    # parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon for PPO')
    # parser.add_argument('--T_horizon', type=int, default=2048, help='T_horizon for PPO')
    # parser.add_argument('--K_epoch', type=int, default=10, help='K_epoch for PPO')
    # parser.add_argument('--use_dynamic_closest', type=int, default=0, help='use dynamic closest landmark')
    
    """ METRA를 위한 아규먼트들 """
    parser.add_argument('--seed', type=int, default=123, metavar='N', help='random seed')
    parser.add_argument('--num_steps', type=int, default=100000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
    parser.add_argument('--gradient_steps_per_epoch', type=int, default=50, metavar='N',
                    help='model updates per simulator step (default: 1)')   # 종해 코드에서는 디폴트 50으로 설정
    parser.add_argument('--episodes_per_epoch', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--use_METRA', type=int, default=0, help='use METRA')
    parser.add_argument('--use_METRA_only', type=int, default=0, help='use METRA only')
    parser.add_argument('--policy', default='Gaussian', help='policy type: Gaussian or Deterministic')
    parser.add_argument('--alpha', type=float, default=0.01, metavar='G', help='coefficient for METRA')
    parser.add_argument('--skill_dim', type=int, default=2, help='dimension of skill')
    parser.add_argument('--metra_lr', type=float, default=0.0005, metavar='G', help='learning rate of METRA')
    parser.add_argument('--cuda', action='store_false', help='run on CUDA (default: True)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: True)')
    
    """ 하단 아규먼트들은 evaluate랑 통일시켜야함 """
    parser.add_argument('--episode_length', type=int, default=40, help='steps per episode') # 25가 원래 값이었음
    parser.add_argument('--map_size', type=float, default=1.0, help='size of the map')
    parser.add_argument('--num_agents', type=int, default=1, help='number of agents in the env')
    args = parser.parse_args()

    """ create folder to save result """
    env_dir = os.path.join('./results', args.env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    
    START_DATE = datetime.now().strftime('%Y-%m-%d')
    START_HOUR = datetime.now().strftime('%H')
    START_TIME = datetime.now().strftime('%M-%S')
    result_dir = os.path.join(env_dir, f'{START_DATE}', f'{START_HOUR}', f'{START_TIME}')
    os.makedirs(result_dir)
    
    gif_dir = os.path.join(result_dir, 'gif')
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    gif_num = len([file for file in os.listdir(gif_dir)])  # current number of gif
    """ =================================== """
    
    args2dict = vars(args)
    with open(os.path.join(result_dir, 'args.json'), 'w') as fp:
        json.dump(args2dict, fp, indent=4)
    writer = SummaryWriter(result_dir)    # 텐서보드 추가 0724

    env, dim_info = get_env(args.env_name, args.episode_length, args.num_agents, args.map_size) # dim_info는 각 에이전트의 obs_dim, act_dim을 담은 딕셔너리
    masac = MASAC(dim_info, env.action_spaces['agent_0'], result_dir, args)
    
    if args.use_METRA:
        """ =============== METRA 관련 ================= """
        sac_agent = SAC(6 + args.skill_dim, env.action_space('agent_0'), args)
        metra_phi = Phi(6, args).to(torch.device("cuda" if args.cuda else "cpu"))
        metra_lambda = Lambda(args)
        metra_memory = ReplayMemory(args.buffer_capacity, args.seed)
        updates = 0
        """ ============================================= """
    episode_idx = 0
    step = 0  # global step counter
    agent_num = env.num_agents
    # reward of each episode of each agent
    # episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}
    
    train_start_flag = True
    
    print(f"use_METRA: {args.use_METRA}")
    print(f'use_METRA_only: {args.use_METRA_only}')
    
    for i_epoch in itertools.count(1):
        for i_episode in range(args.episodes_per_epoch):
            
            use_METRA = args.use_METRA    # 1:2:1:2:1:2 순으로 guidance action을 사용하면 좋겠다
            GUID_count = 0
            MADDPG_count = 0
            
            episode_steps = 0
            obs_s = env.reset()
            agent_psuedo_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
            # MARL reward
            MARL_agents_rewards = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
            episode_idx += 1
            
            done = False
            skill = generate_skill_cont(args.skill_dim)
            obs_idx = {}
            for agent_id in env.agents:
                obs_ind = obs_s[agent_id][:6]
                obs_idx[agent_id] = np.concatenate([obs_ind, skill])
            # obs_0 = np.concatenate([obs_s['agent_0'], skill])   # 에이전트가 1일때는 8차원: 
            
            for t in range(args.episode_length):    # 지금은 60
                if args.random_steps > step:
                    actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents} # 랜덤 액션 (연속에서도 잘 됨)
                else:
                    if use_METRA:
                        actions = {agent_id: sac_agent.select_action(obs_idx['agent_0']) for agent_id in env.agents}
                    else:
                        actions, _ = masac.select_action(obs_s)
                
                next_obs_s, rewards, dones, info = env.step(actions)
                
                next_obs_idx = {}
                
                episode_steps += 1
                step += 1
                
                for agent_id in dones.keys():
                    next_obs_ind = next_obs_s[agent_id][:6]
                    next_obs_idx[agent_id] = np.concatenate([next_obs_ind, skill])
                        
                # 에이전트 수가 1개일때는 next_obs_s는 6차원: 본인의 속도 + 본인의 위치 + 랜드마크들까지의 상대 변위, 거기에 2차원 스킬을 더해서 8차원
                # 여기서 에이전트 수가 2개가 되면, next_obs_s는 14차원: 본인의 속도 + 본인의 위치 + 랜드마크들까지의 위치(2차원 + 2차원) + 다른 에이전트 까지의 상대 변위 (2차원) + comm (2차원) + 스킬 (2차원)
                
                if use_METRA:
                    psuedo_reward = np.dot(metra_phi.forward_np(next_obs_idx['agent_0']) - metra_phi.forward_np(obs_idx['agent_0']), skill)                
                    agent_psuedo_reward['agent_0'] += psuedo_reward
                    
                    mask = 1 if episode_steps == args.episode_length else float(not done)
                    metra_memory.push(obs_idx['agent_0'], actions['agent_0'], psuedo_reward, next_obs_idx['agent_0'], mask)                
                else:
                    tar_acts = {}
                    # METRA로 가이던스 안 받는 경우입니다
                    for agent_id in dones.keys():
                        tar_acts[agent_id] = None
                    
                    masac.add(obs_s, actions, rewards, next_obs_s, dones, tar_acts)

                    for agent_id, r in rewards.items():  # update reward
                        MARL_agents_rewards[agent_id] += r
                    masac.learn(step, writer)
                    masac.update_target(args.tau)
                
                if all(dones.values()):
                    break
                
                obs_idx['agent_0'] = next_obs_idx['agent_0']
            
            writer.add_scalar('metra_rew/agent_0_reward', agent_psuedo_reward['agent_0'], episode_idx)
            print(f"""
                  Episode: {episode_idx}, total num timesteps: {step}, agent_0_psuedo_reward: {round(agent_psuedo_reward['agent_0'], 3)}, 
                  MARL_agent_0_reward: {round(MARL_agents_rewards['agent_0'], 3)},
                  result_dir: {result_dir}
                  """)
        if use_METRA:
            if len(metra_memory) > args.batch_size:
                for i in range(args.gradient_steps_per_epoch):
                    metra_phi_loss = metra_phi.update_parameters(metra_memory, args.batch_size, metra_lambda.lambda_value)
                    metra_lambda_loss = metra_lambda.update_parameters(metra_memory, args.batch_size, metra_phi)
                    metra_critic_1_loss, metra_critic_2_loss, policy_loss, ent_loss, alpha = sac_agent.update_parameters(metra_memory, args.batch_size, updates, metra_phi)
                    
                    writer.add_scalar('metra_loss/metra_phi_loss', metra_phi_loss, updates)
                    writer.add_scalar('metra_liss/metra_lambda_loss', metra_lambda_loss, updates)
                    writer.add_scalar('metra_loss/metra_critic_1_loss', metra_critic_1_loss, updates)
                    writer.add_scalar('metra_loss/metra_critic_2_loss', metra_critic_2_loss, updates)
                    writer.add_scalar('metra_loss/policy_loss', policy_loss, updates)
                    writer.add_scalar('metra_loss/ent_loss', ent_loss, updates)
                    writer.add_scalar('metra_entropy_temp/alpha', alpha, updates)
                    writer.add_scalar('metra_dual_variable/lambda', metra_lambda.lambda_value, updates)
                    updates += 1
        
        if step > args.num_steps:
            break
        
        if episode_idx % 1000 == 0:
            if use_METRA:
                sac_agent.save_checkpoint('METRA', f"{episode_idx}")
                avg_pseudo_reward = 0.
                episode_psuedo_reward = 0.
            
            avg_dist_reward = 0.
            avg_step = 0.
            episodes = 8
            
            for i in range(episodes):
                obs_s = env.reset()
                if use_METRA:
                    skill = generate_skill_cont(args.skill_dim)
                    obs_idx = {}
                    for agent_id in env.agents:
                        obs_ind = obs_s[agent_id][:6]
                        obs_idx[agent_id] = np.concatenate([obs_ind, skill])
                
                episode_steps = 0
                episode_dist_reward_sum = 0

                frame_list = []  # used to save gif
                video_frames = []  # 비디오 저장용 0723

                for _ in range(args.episode_length):
                    if use_METRA:
                        actions = {agent_id: sac_agent.select_action(obs_idx['agent_0'], evaluate=True) for agent_id in env.agents}
                    else:
                        actions, _ = masac.select_action(obs_s, evaluate=True)
                    next_obs_s, rewards, dones, info = env.step(actions)
                    
                    if use_METRA:
                        next_obs_idx['agent_0'] = np.concatenate([next_obs_s['agent_0'], skill])
                        
                        next_obs_idx = {}
                        for agent_id in dones.keys():
                            next_obs_ind = next_obs_s[agent_id][:6]
                            next_obs_idx[agent_id] = np.concatenate([next_obs_ind, skill])
                        
                        psuedo_reward = np.dot(metra_phi.forward_np(next_obs_idx['agent_0']) - metra_phi.forward_np(obs_idx['agent_0']), skill)
                    
                        episode_psuedo_reward += psuedo_reward
                    episode_dist_reward_sum += sum(rewards.values()) # 각 에이전트들의 리워드 합
                    episode_steps += 1
                    
                    frame = env.render(mode='rgb_array') # 비디오 저장용 0723
                    image = Image.fromarray(frame)
                    draw = ImageDraw.Draw(image)
                    reward_txt = ', '.join(f"{agent_id}: {round(r, 3)}" for agent_id, r in rewards.items())
                    txt_pos = (10, 10)
                    txt_color = (0, 0, 0)
                    draw.text(txt_pos, reward_txt, fill=txt_color)
                    
                    frame_list.append(image) # 비디오 저장용 0723
                    video_frames.append(frame) # 비디오 저장용 0723
                    
                    if all(dones.values()):
                        break
                    if use_METRA:
                        obs_idx['agent_0'] = next_obs_idx['agent_0']
                    else:
                        obs_s = next_obs_s
                
                # save gif
                subdir = os.path.join(gif_dir, f'episode_{episode_idx}')
                os.makedirs(subdir, exist_ok=True)
                gif_path = os.path.join(subdir, f'episode_{episode_idx}_{i}.gif')
                frame_list[0].save(gif_path,
                                save_all=True, append_images=frame_list[1:], duration=1, loop=0)
                
                if use_METRA:    
                    avg_pseudo_reward += episode_psuedo_reward
                    print(f"Test Episode: {i}, total num timesteps: {step}, agent_0_psuedo_reward: {round(episode_psuedo_reward, 3)}")
                
                avg_dist_reward += episode_dist_reward_sum
                avg_step += episode_steps
                print(f"Test Episode: {i}, total num timesteps: {step}, episode_dist_rewards_sum: {round(episode_dist_reward_sum, 3)}, episode_steps: {episode_steps}")
            
            if use_METRA:
                avg_pseudo_reward /= episodes
            avg_dist_reward /= episodes
            avg_step /= episodes
            if use_METRA:
                writer.add_scalar('metra_rew_test/avg_pseudo_reward', avg_pseudo_reward, episode_idx)
            writer.add_scalar('avg_dist_reward', avg_dist_reward, episode_idx)
                    
    env.close()
                
    # for episode in range(args.episode_num):
    #     use_PPO = True    # 1:2:1:2:1:2 순으로 guidance action을 사용하면 좋겠다
    #     GUID_count = 0
    #     MADDPG_count = 0
            
    #     obs_s = env.reset()
    #     # obs['agent_idx'] = 자기 속도(2차원) + 자기 위치(2차원) + 랜드마크들까지의 상대 변위(2 * 랜드마크수) + 본인 빼고 아군들 까지의 상대 변위(2 * 본인 뺀 아군 수) + 본인 빼고 나머지한테 받은 통신(2 * 본인 뺀 아군 수) : 에이전트 3개, 랜드마크 3개의 경우 18차원
        
    #     agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
        
    #     if (episode + 1) % 5000 == 0:
    #         frame_list = []  # used to save gif
    #         video_frames = []  # 비디오 저장용 0723
        
    #     while env.agents:  # interact with the env for an episode
    #         step += 1
    #         prob_acts = None # random_action 나올 때는 버퍼에 None 넣기 위함
    #         if step < args.random_steps:
    #             actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents} # 정수의 랜덤 액션
    #         else:
    #             if args.use_GUID:
    #                 if use_PPO:
    #                     actions, prob_acts = maddpg.select_action(obs_s, ppo=True)  # PPO 계속 구현 중 0804
    #                     GUID_count += 1
                        
    #                     if GUID_count > args.episode_length * args.update_GUID / (args.update_GUID + args.update_MA):
    #                         use_PPO = False
    #                         GUID_count = 0
                        
    #                 else:
    #                     actions = maddpg.select_action(obs_s, model_out=False, deterministic=True)  # PPO 계속 구현 중 0804
    #                     MADDPG_count += 1
                        
    #                     if MADDPG_count > args.episode_length * args.update_GUID / (args.update_GUID + args.update_MA):
    #                         use_PPO = True
    #                         MADDPG_count = 0
    #             else:
    #                 if args.use_PPO:
    #                     actions, prob_acts = maddpg.select_action(obs_s, ppo=True, deterministic=False)
    #                 else:
    #                     actions, prob_acts = maddpg.select_action(obs_s, ppo=False, deterministic=True)
            
    #         """ KL위한 작업들 0723 """
    #         if args.use_GUID:
    #             """ guidance action을 구하기 위해 obs, next_obs를 이용하여 discrite action을 구해보자 """
    #             closest_discrete_action_ls = []
    #             agent_id_sub = 0
                
    #             for agent_id, agent_obs in obs_s.items():
    #                 landmarks_rel_pos_ls_x = []
    #                 landmarks_rel_pos_ls_y = []
                    
    #                 """ 1. 랜드마크까지의 l1_norm을 구한다 """
    #                 for i in range(agent_num):
    #                     landmarks_rel_pos_ls_x.append(abs(agent_obs[4 + 2*i : 4 + 2*(i+1)][0]))
    #                     landmarks_rel_pos_ls_y.append(abs(agent_obs[4 + 2*i : 4 + 2*(i+1)][1]))
                    
    #                 if args.use_dynamic_closest:
    #                     """ 가장 가까운 랜드마크의 l1 상대 변위를 가져온다 """    
    #                     x_target_idx = min(landmarks_rel_pos_ls_x)
    #                     y_target_idx = min(landmarks_rel_pos_ls_y)
    #                 else:
    #                     """ agent_id와 일대일 대응하는 랜드마크의 l1 상대 변위를 가져온다"""
    #                     x_target_idx = landmarks_rel_pos_ls_x[agent_id_sub]
    #                     y_target_idx = landmarks_rel_pos_ls_y[agent_id_sub]
                    
    #                 """ 만약 해당 랜드마크까지의 x거리가 y거리보다 긴 경우: x거리를 좁혀야 하며, -일 땐 - 움직임, +일땐 + 움직임 """
    #                 if x_target_idx > y_target_idx:
    #                     target_idx = landmarks_rel_pos_ls_x.index(x_target_idx)
    #                     closest_landmark_discrete = agent_obs[4 + 2*target_idx : 4 + 2*(target_idx+1)]
                        
    #                     if abs(closest_landmark_discrete[0]) <= 0.01:
    #                         closest_discrete_action_ls.append(0)
    #                     elif closest_landmark_discrete[0] > 0:
    #                         closest_discrete_action_ls.append(2)
    #                     elif closest_landmark_discrete[0] < 0:
    #                         closest_discrete_action_ls.append(1)
    #                     else:
    #                         AssertionError("Error. Check the code case line 102 in main.py")
    #                 else:
    #                     """ 만약 해당 랜드마크까지의 y거리가 x거리보다 긴 경우: y거리를 좁혀야 하며, -일 땐 - 움직임, +일땐 + 움직임 """
    #                     target_idx = landmarks_rel_pos_ls_y.index(y_target_idx)
    #                     closest_landmark_discrete = agent_obs[4 + 2*target_idx : 4 + 2*(target_idx+1)]
                        
    #                     if abs(closest_landmark_discrete[1]) <= 0.01:
    #                         closest_discrete_action_ls.append(0)
    #                     elif closest_landmark_discrete[1] > 0:
    #                         closest_discrete_action_ls.append(4)
    #                     elif closest_landmark_discrete[1] < 0:
    #                         closest_discrete_action_ls.append(3)
    #                     else:
    #                         AssertionError("Error. Check the code case line 113 in main.py")
                    
    #                 agent_id_sub += 1
    #         else:
    #             closest_discrete_action_ls = [0 for _ in range(agent_num)]
            
    #         target_entities = {}
    #         for i, agent_id in enumerate(env.agents):
    #             if not use_PPO or step < args.random_steps:
    #                 target_entities[agent_id] = closest_discrete_action_ls[i]
    #             else:
    #                 target_entities[agent_id] = i
                

    #         next_obs_s, rewards, dones, info = env.step(actions) # discrete action이다 참고로
            
    #         if prob_acts is None or step < args.random_steps:
    #             maddpg.add(obs_s, actions, rewards, next_obs_s, dones, target_entities, None)  # target_entities가 휴리스틱 액션으로 잘 들어간다
    #         else:
    #             maddpg.add(obs_s, actions, rewards, next_obs_s, dones, target_entities, prob_acts, ppo=use_PPO)  # KL 0723                

    #         for agent_id, r in rewards.items():  # update reward
    #             agent_reward[agent_id] += r

            
    #         if not args.use_PPO_only:
    #             if step >= args.random_steps * 9 / 2 and step % args.learn_interval == 0:  # learn every few steps
    #                 if train_start_flag:
    #                     print("==================================== start training ==========================")
    #                 maddpg.learn(args.batch_size, args.gamma, args.use_KL, args.kl_coeff, args.update_GUID, args.update_MA, step, writer, args.use_PPO) # KL 0723
    #                 if not use_PPO:
    #                     maddpg.update_target(args.tau)
    #                 train_start_flag = False
    #         else:
    #             if step % args.T_horizon == 0:  # learn every few steps
    #                 if train_start_flag:
    #                     print("==================================== start training ==========================")
    #                 maddpg.learn(args.batch_size, args.gamma, args.use_KL, args.kl_coeff, args.update_GUID, args.update_MA, step, writer, args.use_PPO)
    #                 train_start_flag = False
            
    #         if (episode + 1) % 5000 == 0:
    #             frame = env.render(mode='rgb_array') # 비디오 저장용 0723
    #             frame_list.append(Image.fromarray(frame)) # 비디오 저장용 0723
    #             video_frames.append(frame) # 비디오 저장용 0723
                
    #             # save gif
    #             frame_list[0].save(os.path.join(gif_dir, f'train_out{gif_num + episode + 1}.gif'),
    #                             save_all=True, append_images=frame_list[1:], duration=1, loop=0)
            
    #         obs_s = next_obs_s

    #     env.close()
                
    #     # episode finishes
    #     for agent_id, r in agent_reward.items():  # record reward
    #         episode_rewards[agent_id][episode] = r

    #     if (episode + 1) % 100 == 0:  # print info every 100 episodes
    #         message = f'episode {episode + 1}, '
    #         sum_reward = 0
    #         for agent_id, r in agent_reward.items():  # record reward
    #             message += f'{agent_id}: {r:>4f}; '
    #             sum_reward += r
            
    #         sum_reward = sum_reward / args.episode_length
            
    #         message += f'sum reward: {sum_reward}; '
    #         message += f'result_dir: {result_dir}'
            
    #         writer.add_scalar('sum_reward', sum_reward, episode)
            
    #         print(message)
    
    # print("==================================== training finished ==========================")
    # print(f"result_dir: {result_dir}")
    # maddpg.save(episode_rewards)  # save model


    # def get_running_reward(arr: np.ndarray, window=100):
    #     """calculate the running reward, i.e. average of last `window` elements from rewards"""
    #     running_reward = np.zeros_like(arr)
    #     for i in range(window - 1):
    #         running_reward[i] = np.mean(arr[:i + 1])
    #     for i in range(window - 1, len(arr)):
    #         running_reward[i] = np.mean(arr[i - window + 1:i + 1])
    #     return running_reward


    # # training finishes, plot reward
    # fig, ax = plt.subplots()
    # x = range(1, args.episode_num + 1)
    # for agent_id, rewards in episode_rewards.items():
    #     ax.plot(x, rewards, label=agent_id)
    #     ax.plot(x, get_running_reward(rewards))
    # ax.legend()
    # ax.set_xlabel('episode')
    # ax.set_ylabel('reward')
    # title = f'training result of maddpg solve {args.env_name}'
    # ax.set_title(title)
    # plt.savefig(os.path.join(result_dir, title))
    
    # with open(os.path.join(result_dir, 'finished_flag.txt'), 'w') as f:
    #     f.write('finished_flag')
