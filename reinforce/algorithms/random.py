import sys
import time

import numpy as np

import torch

import reinforce

# import pdb; pdb.set_trace()

class Random:
    def __init__(self, config):
        self.rollout_size = config.get('rollout_size')
        self.worker_count = config.get('worker_count')

        self.config = config

        self.final_rewards = torch.zeros([self.worker_count, 1])
        self.episode_rewards = torch.zeros([self.worker_count, 1])
        self.batch_count = 0
        self.running_reward = None
        self.test_env = None

    def initialize(self):
        self.envs = reinforce.Envs(self.config)
        _ = self.envs.reset()

    def batch_train(self):
        batch_start = time.time()

        for step in range(self.rollout_size):
            action = torch.normal(
                torch.zeros(self.worker_count, 2),
                torch.ones(self.worker_count, 2),
            )
            _, reward, done = self.envs.step(
                action.numpy(),
            )

            reward = torch.from_numpy(np.expand_dims(reward, 1)).float()
            mask = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done]
            )

            self.episode_rewards += reward
            self.final_rewards *= mask
            self.final_rewards += (1 - mask) * self.episode_rewards
            self.episode_rewards *= mask

        total_num_steps = (
            (self.batch_count + 1) * self.worker_count * self.rollout_size
        )
        batch_end = time.time()

        print(
            ("STEP {} timesteps {} FPS {} " + \
             "mean/median R {:.1f} {:.1f} " + \
             "min/max R {:.1f} {:.1f}").
            format(
                self.batch_count,
                total_num_steps,
                int(self.worker_count * self.rollout_size / (batch_end - batch_start)),
                self.final_rewards.mean(),
                self.final_rewards.median(),
                self.final_rewards.min(),
                self.final_rewards.max(),
            ))
        sys.stdout.flush()

        self.batch_count += 1
        if self.running_reward is None:
            self.running_reward = self.final_rewards.mean()
        self.running_reward = (
            self.running_reward * 0.9 + self.final_rewards.mean() * 0.1
        )

        return self.running_reward

    def test(self, step_callback=None):
        test_start = time.time()

        if self.test_env is None:
            self.test_env = reinforce.Donkey(self.config)
        _ = self.test_env.reset()

        end = False
        final_reward = 0;
        episode_size = 0

        while not end:
            action = torch.normal(
                torch.zeros(1, 2),
                torch.ones(1, 2),
            )
            observation, reward, done = self.test_env.step(
                action[0].data.cpu().numpy(),
            )

            if step_callback is not None:
                step_callback(observation, reward, done)

            final_reward += reward
            end = done

            episode_size += 1

        test_end = time.time()

        print("TEST FPS {} step_count {} final_reward {:.1f}".format(
            int(episode_size / (test_end - test_start)),
            episode_size,
            final_reward
        ))
        sys.stdout.flush()

        return final_reward
