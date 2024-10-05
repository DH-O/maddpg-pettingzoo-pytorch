import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from datetime import datetime
from tensorboardX import SummaryWriter

from MADDPG import MADDPG
from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2



def get_env(env_name, ep_len=25, num_agents=3, map_size=1.0):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v2':
        new_env = simple_adversary_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_spread_v2':
        new_env = simple_spread_v2.parallel_env(max_cycles=ep_len, num_agents=num_agents, map_size=map_size)
    if env_name == 'simple_tag_v2':
        new_env = simple_tag_v2.parallel_env(max_cycles=ep_len)

    new_env.reset()
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).n)

    return new_env, _dim_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, default='simple_adversary_v2', help='name of the env',
                        choices=['simple_adversary_v2', 'simple_spread_v2', 'simple_tag_v2'])
    parser.add_argument('--episode_num', type=int, default=200000,   # 30000이 좀 적당하긴 하다
                        help='total episode num during training procedure')
    parser.add_argument('--learn_interval', type=int, default=-1,
                        help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=0,    # 5e4인데 잠깐만 앞당김
                        help='random steps before the agent start to learn')
    parser.add_argument('--update_GUID', type=int, default=-1, help='update GUID every n steps')
    parser.add_argument('--update_MA', type=int, default=-1, help='update MADDPG every n steps')
    
    parser.add_argument('--tau', type=float, default=0.1, help='soft update parameter')    # 이거 높을수록 지금 막 업데이트 된 네트워크를 중요하게 생각하겠다는 의미
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')    # 이거 높을수록 멀리 있는 리워드도 중요하게 생각하겠다는 의미
    parser.add_argument('--buffer_capacity', type=int, default=int(0), help='capacity of replay buffer')  # 1e7이었음
    parser.add_argument('--batch_size', type=int, default=32, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.00005, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.00005, help='learning rate of critic')
    parser.add_argument('--max_grad_norm', type=float, default=400, help='max norm of gradients')
    parser.add_argument('--kl_coeff', type=float, default=100, help='coefficient for KL divergence')
    
    parser.add_argument('--use_GUID', type=int, default=0, help='use guidance action')
    parser.add_argument('--use_KL', type=int, default=0, help='use KL divergence for action selection')
    parser.add_argument('--use_PPO', type=int, default=1, help='use PPO for action selection')
    parser.add_argument('--use_PPO_only', type=int, default=1, help='use PPO only for action selection')
    parser.add_argument('--entropy_coeff', type=float, default=10, help='entropy coefficient for PPO')
    parser.add_argument('--value_coeff', type=float, default=0.1, help='value coefficient for PPO')
    parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon for PPO')
    parser.add_argument('--T_horizon', type=int, default=2048, help='T_horizon for PPO')
    parser.add_argument('--K_epoch', type=int, default=10, help='K_epoch for PPO')
    parser.add_argument('--use_dynamic_closest', type=int, default=0, help='use dynamic closest landmark')
    
    """ 하단 아규먼트들은 evaluate랑 통일시켜야함 """
    parser.add_argument('--episode_length', type=int, default=25, help='steps per episode') # 25가 원래 값이었음
    parser.add_argument('--map_size', type=float, default=1.0, help='size of the map')
    parser.add_argument('--num_agents', type=int, default=1, help='number of agents in the env')
    args = parser.parse_args()

    # create folder to save result
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
    
    
    args2dict = vars(args)
    with open(os.path.join(result_dir, 'args.json'), 'w') as fp:
        json.dump(args2dict, fp, indent=4)
    writer = SummaryWriter(result_dir)    # 텐서보드 추가 0724

    env, dim_info = get_env(args.env_name, args.episode_length, args.num_agents, args.map_size) 
    # dim_info는 각 에이전트의 obs_dim, act_dim을 담은 딕셔너리
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, 
                    result_dir, args)

    step = 0  # global step counter
    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}
    
    train_start_flag = True
    
    print(f"use_GUID: {args.use_GUID}")
    print(f"use_KL: {args.use_KL}")
    print(f"use_PPO: {args.use_PPO}")
    print(f"use_PPO_only: {args.use_PPO_only}")
    print(f"use_dynamic_closest: {args.use_dynamic_closest}")
    
    for episode in range(args.episode_num):
        use_PPO = True    # 1:2:1:2:1:2 순으로 guidance action을 사용하면 좋겠다
        GUID_count = 0
        MADDPG_count = 0
            
        obs_s = env.reset()
        # obs['agent_idx'] = 자기 속도(2차원) + 자기 위치(2차원) + 랜드마크들까지의 상대 변위(2 * 랜드마크수) + 본인 빼고 아군들 까지의 상대 변위(2 * 본인 뺀 아군 수) + 본인 빼고 나머지한테 받은 통신(2 * 본인 뺀 아군 수) : 에이전트 3개, 랜드마크 3개의 경우 18차원
        
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
        
        if (episode + 1) % 5000 == 0:
            frame_list = []  # used to save gif
            video_frames = []  # 비디오 저장용 0723
        
        while env.agents:  # interact with the env for an episode
            step += 1
            prob_acts = None # random_action 나올 때는 버퍼에 None 넣기 위함
            if step < args.random_steps:
                actions = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents} # 정수의 랜덤 액션
            else:
                if args.use_GUID:
                    if use_PPO:
                        actions, prob_acts = maddpg.select_action(obs_s, ppo=True)  # PPO 계속 구현 중 0804
                        GUID_count += 1
                        
                        if GUID_count > args.episode_length * args.update_GUID / (args.update_GUID + args.update_MA):
                            use_PPO = False
                            GUID_count = 0
                        
                    else:
                        actions = maddpg.select_action(obs_s, model_out=False, deterministic=True)  # PPO 계속 구현 중 0804
                        MADDPG_count += 1
                        
                        if MADDPG_count > args.episode_length * args.update_GUID / (args.update_GUID + args.update_MA):
                            use_PPO = True
                            MADDPG_count = 0
                else:
                    if args.use_PPO:
                        actions, prob_acts = maddpg.select_action(obs_s, ppo=True, deterministic=False)
                    else:
                        actions, prob_acts = maddpg.select_action(obs_s, ppo=False, deterministic=True)
            
            """ KL위한 작업들 0723 """
            if args.use_GUID:
                """ guidance action을 구하기 위해 obs, next_obs를 이용하여 discrite action을 구해보자 """
                closest_discrete_action_ls = []
                agent_id_sub = 0
                
                for agent_id, agent_obs in obs_s.items():
                    landmarks_rel_pos_ls_x = []
                    landmarks_rel_pos_ls_y = []
                    
                    """ 1. 랜드마크까지의 l1_norm을 구한다 """
                    for i in range(agent_num):
                        landmarks_rel_pos_ls_x.append(abs(agent_obs[4 + 2*i : 4 + 2*(i+1)][0]))
                        landmarks_rel_pos_ls_y.append(abs(agent_obs[4 + 2*i : 4 + 2*(i+1)][1]))
                    
                    if args.use_dynamic_closest:
                        """ 가장 가까운 랜드마크의 l1 상대 변위를 가져온다 """    
                        x_target_idx = min(landmarks_rel_pos_ls_x)
                        y_target_idx = min(landmarks_rel_pos_ls_y)
                    else:
                        """ agent_id와 일대일 대응하는 랜드마크의 l1 상대 변위를 가져온다"""
                        x_target_idx = landmarks_rel_pos_ls_x[agent_id_sub]
                        y_target_idx = landmarks_rel_pos_ls_y[agent_id_sub]
                    
                    """ 만약 해당 랜드마크까지의 x거리가 y거리보다 긴 경우: x거리를 좁혀야 하며, -일 땐 - 움직임, +일땐 + 움직임 """
                    if x_target_idx > y_target_idx:
                        target_idx = landmarks_rel_pos_ls_x.index(x_target_idx)
                        closest_landmark_discrete = agent_obs[4 + 2*target_idx : 4 + 2*(target_idx+1)]
                        
                        if abs(closest_landmark_discrete[0]) <= 0.01:
                            closest_discrete_action_ls.append(0)
                        elif closest_landmark_discrete[0] > 0:
                            closest_discrete_action_ls.append(2)
                        elif closest_landmark_discrete[0] < 0:
                            closest_discrete_action_ls.append(1)
                        else:
                            AssertionError("Error. Check the code case line 102 in main.py")
                    else:
                        """ 만약 해당 랜드마크까지의 y거리가 x거리보다 긴 경우: y거리를 좁혀야 하며, -일 땐 - 움직임, +일땐 + 움직임 """
                        target_idx = landmarks_rel_pos_ls_y.index(y_target_idx)
                        closest_landmark_discrete = agent_obs[4 + 2*target_idx : 4 + 2*(target_idx+1)]
                        
                        if abs(closest_landmark_discrete[1]) <= 0.01:
                            closest_discrete_action_ls.append(0)
                        elif closest_landmark_discrete[1] > 0:
                            closest_discrete_action_ls.append(4)
                        elif closest_landmark_discrete[1] < 0:
                            closest_discrete_action_ls.append(3)
                        else:
                            AssertionError("Error. Check the code case line 113 in main.py")
                    
                    agent_id_sub += 1
            else:
                closest_discrete_action_ls = [0 for _ in range(agent_num)]
            
            target_entities = {}
            for i, agent_id in enumerate(env.agents):
                if not use_PPO or step < args.random_steps:
                    target_entities[agent_id] = closest_discrete_action_ls[i]
                else:
                    target_entities[agent_id] = i
                

            next_obs_s, rewards, dones, info = env.step(actions) # discrete action이다 참고로
            
            if prob_acts is None or step < args.random_steps:
                maddpg.add(obs_s, actions, rewards, next_obs_s, dones, target_entities, None)  # target_entities가 휴리스틱 액션으로 잘 들어간다
            else:
                maddpg.add(obs_s, actions, rewards, next_obs_s, dones, target_entities, prob_acts, ppo=use_PPO)  # KL 0723                

            for agent_id, r in rewards.items():  # update reward
                agent_reward[agent_id] += r

            
            if not args.use_PPO_only:
                if step >= args.random_steps * 9 / 2 and step % args.learn_interval == 0:  # learn every few steps
                    if train_start_flag:
                        print("==================================== start training ==========================")
                    maddpg.learn(args.batch_size, args.gamma, args.use_KL, args.kl_coeff, args.update_GUID, args.update_MA, step, writer, args.use_PPO) # KL 0723
                    if not use_PPO:
                        maddpg.update_target(args.tau)
                    train_start_flag = False
            else:
                if step % args.T_horizon == 0:  # learn every few steps
                    if train_start_flag:
                        print("==================================== start training ==========================")
                    maddpg.learn(args.batch_size, args.gamma, args.use_KL, args.kl_coeff, args.update_GUID, args.update_MA, step, writer, args.use_PPO)
                    train_start_flag = False
            
            if (episode + 1) % 5000 == 0:
                frame = env.render(mode='rgb_array') # 비디오 저장용 0723
                frame_list.append(Image.fromarray(frame)) # 비디오 저장용 0723
                video_frames.append(frame) # 비디오 저장용 0723
                
                # save gif
                frame_list[0].save(os.path.join(gif_dir, f'train_out{gif_num + episode + 1}.gif'),
                                save_all=True, append_images=frame_list[1:], duration=1, loop=0)
            
            obs_s = next_obs_s

        env.close()
                
        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r

        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f'{agent_id}: {r:>4f}; '
                sum_reward += r
            
            sum_reward = sum_reward / args.episode_length
            
            message += f'sum reward: {sum_reward}; '
            message += f'result_dir: {result_dir}'
            
            writer.add_scalar('sum_reward', sum_reward, episode)
            
            print(message)
    
    print("==================================== training finished ==========================")
    print(f"result_dir: {result_dir}")
    maddpg.save(episode_rewards)  # save model


    def get_running_reward(arr: np.ndarray, window=100):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        running_reward = np.zeros_like(arr)
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])
        return running_reward


    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent_id, rewards in episode_rewards.items():
        ax.plot(x, rewards, label=agent_id)
        ax.plot(x, get_running_reward(rewards))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maddpg solve {args.env_name}'
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, title))
    
    with open(os.path.join(result_dir, 'finished_flag.txt'), 'w') as f:
        f.write('finished_flag')
