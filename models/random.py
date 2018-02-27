import time
import torch
import donkey

import numpy as np

from torch.distributions import Normal

# import pdb; pdb.set_trace()

class Random:
    def __init__(self, config, save_dir=None, load_dir=None):
        self.rollout_size = config.get('rollout_size')
        self.worker_count = config.get('worker_count')

        self.envs = donkey.Envs(self.worker_count)

        self.final_rewards = torch.zeros([self.worker_count, 1])
        self.episode_rewards = torch.zeros([self.worker_count, 1])
        self.batch_count = 0
        self.start = time.time()
        self.running_reward = None

        self.action_mean = torch.rand(self.worker_count, donkey.CONTROL_SIZE)
        self.action_logstd = torch.rand(self.worker_count, donkey.CONTROL_SIZE)
        self.action_std = self.action_logstd.exp()

    def initialize(self):
        observations = self.envs.reset()

    def batch_train(self):
        for step in range(self.rollout_size):
            m = Normal(self.action_mean, self.action_std)
            action = m.sample()

            observation, reward, done = self.envs.step(action.numpy())

            reward = torch.from_numpy(np.expand_dims(reward, 1)).float()
            mask = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done]
            )

            self.episode_rewards += reward
            self.final_rewards *= mask
            self.final_rewards += (1 - mask) * self.episode_rewards
            self.episode_rewards *= mask

        end = time.time()
        total_num_steps = (
            (self.batch_count + 1) * self.worker_count * self.rollout_size
        )
        print(
            ("{}, timesteps {}, FPS {}, " + \
             "mean/median R {:.1f}/{:.1f}, " + \
             "min/max R {:.1f}/{:.1f}").
            format(
                self.batch_count,
                total_num_steps,
                int(total_num_steps / (end - self.start)),
                self.final_rewards.mean(),
                self.final_rewards.median(),
                self.final_rewards.min(),
                self.final_rewards.max(),
            ))

        self.batch_count += 1
        if self.running_reward is None:
            self.running_reward = self.final_rewards.mean()
        self.running_reward = (
            self.running_reward * 0.99 + self.final_rewards.mean() * 0.01
        )

        return self.running_reward
