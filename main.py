import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.mpe import simple_adversary_v2, simple_spread_v2, simple_tag_v2

from MADDPG import MADDPG


def get_env(env_name, ep_len=25):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v2':
        new_env = simple_adversary_v2.parallel_env(max_cycles=ep_len)
    if env_name == 'simple_spread_v2':
        new_env = simple_spread_v2.parallel_env(max_cycles=ep_len)
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
    parser.add_argument('--episode_num', type=int, default=30000,
                        help='total episode num during training procedure')
    parser.add_argument('--episode_length', type=int, default=75, help='steps per episode') # 25 -> 75까지 뻥튀기됨 0723
    parser.add_argument('--learn_interval', type=int, default=100,
                        help='steps interval between learning time')
    parser.add_argument('--random_steps', type=int, default=5e4,    # 5e4인데 잠깐만 앞당김
                        help='random steps before the agent start to learn')
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.01, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate of critic')
    args = parser.parse_args()

    # create folder to save result
    env_dir = os.path.join('./results', args.env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(result_dir)

    env, dim_info = get_env(args.env_name, args.episode_length)
    maddpg = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr,
                    result_dir)

    step = 0  # global step counter
    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent_id: np.zeros(args.episode_num) for agent_id in env.agents}
    
    train_start_flag = True
    
    for episode in range(args.episode_num):
        obs = env.reset()
        # obs['agent_idx'] = 자기 속도 + 자기 위치 + 랜드마크들까지의 상대 변위 + 본인 빼고 아군들 까지의 상대 변위 + 본인 빼고 나머지한테 받은 통신 : 에이전트 7개, 랜드마크 7개의 경우 14개임
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
        while env.agents:  # interact with the env for an episode
            step += 1
            if step < args.random_steps:
                action = {agent_id: env.action_space(agent_id).sample() for agent_id in env.agents}
            else:
                action = maddpg.select_action(obs)
            
            """ KL위한 작업들 0723 """
            use_KL = True
            
            if use_KL:
                """ guidance action을 구하기 위해 obs, next_obs를 이용하여 discrite action을 구해보자 """
                closest_discrete_action_ls = []
                use_dynamic_closest = False
                target_idx_pre = 0
                
                for agent_id, agent_obs in obs.items():
                    landmarks_rel_pos_ls_x = []
                    landmarks_rel_pos_ls_y = []
                    
                    for i in range(7):
                        landmarks_rel_pos_ls_x.append(abs(agent_obs[4 + 2*i : 4 + 2*(i+1)][0]))
                        landmarks_rel_pos_ls_y.append(abs(agent_obs[4 + 2*i : 4 + 2*(i+1)][1]))
                    
                    if use_dynamic_closest:    
                        target_idx_x = min(landmarks_rel_pos_ls_x)
                        target_idx_y = min(landmarks_rel_pos_ls_y)
                    else:
                        target_idx_x = landmarks_rel_pos_ls_x[target_idx_pre]
                        target_idx_y = landmarks_rel_pos_ls_y[target_idx_pre]
                    
                    if target_idx_x < target_idx_y:
                        target_idx = landmarks_rel_pos_ls_x.index(target_idx_x)
                        closest_landmark_discrete = agent_obs[4 + 2*target_idx : 4 + 2*(target_idx+1)]
                        
                        if abs(closest_landmark_discrete[0]) <= 0.001:
                            closest_discrete_action_ls.append(0)
                        elif closest_landmark_discrete[0] > 0:
                            closest_discrete_action_ls.append(1)
                        elif closest_landmark_discrete[0] < 0:
                            closest_discrete_action_ls.append(2)
                        else:
                            AssertionError("Error. Check the code case line 102 in main.py")
                    else:
                        target_idx = landmarks_rel_pos_ls_y.index(target_idx_y)
                        closest_landmark_discrete = agent_obs[4 + 2*target_idx : 4 + 2*(target_idx+1)]
                        
                        if abs(closest_landmark_discrete[1]) <= 0.001:
                            closest_discrete_action_ls.append(0)
                        elif closest_landmark_discrete[1] > 0:
                            closest_discrete_action_ls.append(3)
                        elif closest_landmark_discrete[1] < 0:
                            closest_discrete_action_ls.append(4)
                        else:
                            AssertionError("Error. Check the code case line 113 in main.py")
                    
                    target_idx_pre += 1
            else:
                closest_discrete_action_ls = [0 for _ in range(7)]
            
            target_action = {}
            for i, agent_id in enumerate(env.agents):
                target_action[agent_id] = closest_discrete_action_ls[i]

            next_obs, reward, done, info = env.step(action) # discrete action이다 참고로
                
            # env.render()
            maddpg.add(obs, action, reward, next_obs, done, target_action)  # KL 0723

            for agent_id, r in reward.items():  # update reward
                agent_reward[agent_id] += r

            if step >= args.random_steps and step % args.learn_interval == 0:  # learn every few steps
                if train_start_flag:
                    print("==================================== start training ==========================")
                maddpg.learn(args.batch_size, args.gamma, use_KL, step) # KL 0723
                maddpg.update_target(args.tau)
                train_start_flag = False

            obs = next_obs

        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r

        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f'{agent_id}: {r:>4f}; '
                sum_reward += r
            message += f'sum reward: {sum_reward}'
            print(message)

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
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id)
        ax.plot(x, get_running_reward(reward))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maddpg solve {args.env_name}'
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, title))
