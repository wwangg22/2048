import time
import argparse

from dgn_agent import DQNAgent

import os
import time

import numpy as np
import torch
import utils as ptu
import tqdm

from replay_buffer import ReplayBuffer, PiecewiseSchedule
from game_engine import Game2048



MAX_NVIDEO = 2

seed = 45


def run_training_loop(total_steps, start_learning, batch_size, eval_interval, num_eval_traj, ep_len):
    # set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    game = Game2048()

    # make the gym environment

    agent = DQNAgent(
        observation_shape=game.get_observation_space(),
        num_actions=game.get_input_space(),
        num_layers=2,
        hidden_size=1024,
        learning_rate=1e-2,
        discount=0.99,
        target_update_period=1000,
        use_double_q=False,
        clip_grad_norm=None
    )

    observation = None

    # Replay buffer
    stacked_frames = False
    replay_buffer = ReplayBuffer()
   
    def reset_env_training():
        nonlocal observation

        observation = game.reset()

        observation = np.asarray(observation)

    reset_env_training()
    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1),
            (total_steps * 0.1, 0.02),
        ],
        outside_value=0.02,
    )
    for step in tqdm.trange(total_steps, dynamic_ncols=True):
        epsilon = exploration_schedule.value(step)
        
        # TODO(student): Compute action
        action = agent.get_action(observation=observation, epsilon=epsilon)
        # print(action)

        # TODO(student): Step the environment
        next_observation, reward, done = game.step(action)
        # print(reward)
        next_observation = np.asarray(next_observation)
    
        replay_buffer.insert(observation, action, reward, next_observation, done)

        if done:
            reset_env_training()
        else:
            observation = next_observation

        # Main DQN training loop
        if step >= start_learning:
            # TODO(student): Sample config["batch_size"] samples from the replay buffer
            batch = replay_buffer.sample(batch_size)

            # Convert to PyTorch tensors
            observations = ptu.from_numpy(batch["observations"])
            actions = ptu.from_numpy(batch["actions"])
            rewards = ptu.from_numpy(batch["rewards"])
            next_observations = ptu.from_numpy(batch["next_observations"])
            done = ptu.from_numpy(batch["dones"])

            # TODO(student): Train the agent. `batch` is a dictionary of numpy arrays,
            update_info = agent.update(obs=observations, action=actions, reward=rewards, next_obs=next_observations, done=done, step=step)

            # Logging code
            update_info["epsilon"] = epsilon
            update_info["lr"] = agent.lr_scheduler.get_last_lr()[0]

        if step % eval_interval == 0:
            # Evaluate
            trajectories = ptu.sample_n_trajectories(
                game,
                agent,
                num_eval_traj,
                ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            reward = [sum(t["reward"]) for t in trajectories]
            print("mean reward : ", np.mean(reward))
            # print(returns)
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]


def main():

    # create directory for logging
    logdir_prefix = "hw3_dqn_"  # keep for autograder


    run_training_loop(10000000, 100, 100, 1000, 10, 2000)


if __name__ == "__main__":
    main()
