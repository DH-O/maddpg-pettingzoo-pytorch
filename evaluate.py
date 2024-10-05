import argparse
import os

import imageio

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from MADDPG import MADDPG
from main import get_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, default='simple_adversary_v2', help='name of the env',
                        choices=['simple_adversary_v2', 'simple_spread_v2', 'simple_tag_v2'])
    parser.add_argument('folder', type=str, help='name of the folder where model is saved')
    parser.add_argument('--episode-num', type=int, default=10, help='total episode num during evaluation')
    parser.add_argument('--update_GUID', type=int, default=200, help='update GUID every n steps')
    parser.add_argument('--update_MA', type=int, default=10, help='update MADDPG every n steps')
    
    """ 하단 아규먼트들은 학습 코드와 비교해야함 """
    parser.add_argument('--use_GUID', type=int, default=0, help='use guidance action')
    parser.add_argument('--use_KL', type=int, default=0, help='use KL divergence for action selection')
    parser.add_argument('--T_horizon', type=int, default=2048, help='time horizon for guidance action')
    parser.add_argument('--use_PPO', type=int, default=1, help='use PPO for action selection')
    parser.add_argument('--use_PPO_only', type=int, default=1, help='use PPO only for action selection')
    parser.add_argument('--episode-length', type=int, default=25, help='steps per episode')
    parser.add_argument('--map_size', type=float, default=1.0, help='size of the map')
    parser.add_argument('--num_agents', type=int, default=1, help='number of agents in the env')
    args = parser.parse_args()

    model_dir = os.path.join('./results', args.env_name, args.folder)
    assert os.path.exists(model_dir)
    gif_dir = os.path.join(model_dir, 'gif')
    video_dir = os.path.join(model_dir, 'video')    # 비디오 저장용 0723
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    if not os.path.exists(video_dir): # 비디오 저장용 0723
        os.makedirs(video_dir)
    gif_num = len([file for file in os.listdir(gif_dir)])  # current number of gif

    env, dim_info = get_env(args.env_name, args.episode_length, args.num_agents, args.map_size)
    maddpg = MADDPG.load(dim_info, os.path.join(model_dir, 'model.pt'), args)

    agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent: np.zeros(args.episode_num) for agent in env.agents}
    for episode in range(args.episode_num):
        states = env.reset()
        agent_reward = {agent: 0 for agent in env.agents}  # agent reward of the current episode
        frame_list = []  # used to save gif
        video_frames = []  # 비디오 저장용 0723
        while env.agents:  # interact with the env for an episode
            if args.use_GUID:
                actions, prob = maddpg.select_action(states, ppo=False, deterministic=True)
            else:
                if args.use_PPO:
                    actions, prob = maddpg.select_action(states, ppo=True, deterministic=True)
            next_states, rewards, dones, infos = env.step(actions)
            frame = env.render(mode='rgb_array') # 비디오 저장용 0723
            frame_list.append(Image.fromarray(frame)) # 비디오 저장용 0723
            video_frames.append(frame) # 비디오 저장용 0723
            states = next_states

            for agent_id, reward in rewards.items():  # update reward
                agent_reward[agent_id] += reward

        env.close()
        message = f'episode {episode + 1}, '
        # episode finishes, record reward
        for agent_id, reward in agent_reward.items():
            episode_rewards[agent_id][episode] = reward
            message += f'{agent_id}: {reward:>4f}; '
        print(message)
        # save gif
        frame_list[0].save(os.path.join(gif_dir, f'out{gif_num + episode + 1}.gif'),
                           save_all=True, append_images=frame_list[1:], duration=1, loop=0)
        # save video
        video_path = os.path.join(video_dir, f'out{gif_num + episode + 1}.mp4') # 비디오 저장용 0723
        imageio.mimsave(video_path, video_frames) # 비디오 저장용 0723

    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent_id, rewards in episode_rewards.items():
        ax.plot(x, rewards, label=agent_id)
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    total_files = len([file for file in os.listdir(model_dir)])
    title = f'evaluate result of maddpg solve {args.env_name} {total_files - 3}'
    ax.set_title(title)
    plt.savefig(os.path.join(model_dir, title))
