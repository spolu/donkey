import sys
import time

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import reinforce

from reinforce.policies import PPOPixelsCNN

# import pdb; pdb.set_trace()

class PPOStorage:
    def __init__(self, config, policy, device):
        self.rollout_size = config.get('rollout_size')
        self.worker_count = config.get('worker_count')
        self.hidden_size = config.get('hidden_size')
        self.gamma = config.get('gamma')
        self.tau = config.get('tau')
        self.mini_batch_count = config.get('mini_batch_count')

        self.device = device

        self.observations = torch.zeros(
            self.rollout_size + 1, self.worker_count, *(policy.input_shape()),
        ).to(self.device)
        self.hiddens = torch.zeros(
            self.rollout_size + 1, self.worker_count, self.hidden_size,
        ).to(self.device)
        self.rewards = torch.zeros(
            self.rollout_size, self.worker_count, 1
        ).to(self.device)
        self.values = torch.zeros(
            self.rollout_size + 1, self.worker_count, 1
        ).to(self.device)
        self.returns = torch.zeros(
            self.rollout_size + 1, self.worker_count, 1,
        ).to(self.device)
        self.log_probs = torch.zeros(
            self.rollout_size, self.worker_count, 1,
        ).to(self.device)
        if config.get('action_type') == 'discrete':
            self.actions = torch.zeros(
                self.rollout_size, self.worker_count, 1,
            ).to(self.device)
        else:
            self.actions = torch.zeros(
                self.rollout_size, self.worker_count, reinforce.CONTINUOUS_CONTROL_SIZE,
            ).to(self.device)
        self.masks = torch.ones(
            self.rollout_size + 1, self.worker_count, 1,
        ).to(self.device)

    def insert(self, step, observations, hidden, action, log_prob, value, reward, mask):
        self.observations[step + 1].copy_(observations)
        self.hiddens[step + 1].copy_(hidden)
        self.actions[step].copy_(action)
        self.log_probs[step].copy_(log_prob)
        self.values[step].copy_(value)
        self.rewards[step].copy_(reward)
        self.masks[step + 1].copy_(mask)

    def compute_returns(self, next_value):
        self.values[-1] = next_value
        self.returns[-1] = next_value
        gae = 0
        for step in reversed(range(self.rollout_size)):
            delta = (
                self.rewards[step] +
                self.gamma * self.values[step + 1] * self.masks[step + 1] -
                self.values[step]
            )
            gae = delta + self.gamma * self.tau * self.masks[step + 1] * gae
            self.returns[step] = gae + self.values[step]

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.hiddens[0].copy_(self.hiddens[-1])
        self.masks[0].copy_(self.masks[-1])

    def recurrent_generator(self, advantages):
        mini_batch_size = self.worker_count // self.mini_batch_count
        permutation = torch.randperm(self.worker_count)

        for start in range(0, self.worker_count, mini_batch_size):
            observations_batch = []
            hiddens_batch = []
            actions_batch = []
            returns_batch = []
            masks_batch = []
            log_probs_batch = []
            advantage_targets = []

            for offset in range(mini_batch_size):
                idx = permutation[start + offset]
                observations_batch.append(self.observations[:-1, idx])
                hiddens_batch.append(self.hiddens[0:1, idx])
                actions_batch.append(self.actions[:, idx])
                returns_batch.append(self.returns[:-1, idx])
                masks_batch.append(self.masks[:-1, idx])
                log_probs_batch.append(self.log_probs[:, idx])
                advantage_targets.append(advantages[:, idx])

            yield \
                torch.cat(observations_batch, 0), \
                torch.cat(hiddens_batch, 0), \
                torch.cat(actions_batch, 0), \
                torch.cat(returns_batch, 0), \
                torch.cat(masks_batch, 0), \
                torch.cat(log_probs_batch, 0), \
                torch.cat(advantage_targets, 0)

    def feed_forward_generator(self, advantages):
        batch_size = self.worker_count * self.rollout_size
        mini_batch_size = batch_size // self.mini_batch_count
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            indices = torch.LongTensor(indices).to(self.device)

            observations_batch = self.observations[:-1].view(
                -1, *self.observations.size()[2:],
            )[indices]
            hiddens_batch = self.hiddens[:-1].view(
                -1, self.hiddens.size(-1),
            )[indices]
            actions_batch = self.actions.view(
                -1, self.actions.size(-1),
            )[indices]
            returns_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            log_probs_batch = self.log_probs.view(-1, 1)[indices]
            advantage_targets = advantages.view(-1, 1)[indices]

            yield \
                observations_batch, \
                hiddens_batch, \
                actions_batch, \
                returns_batch, \
                masks_batch, \
                log_probs_batch, \
                advantage_targets

class PPO:
    def __init__(self, config):
        self.learning_rate = config.get('learning_rate')
        self.worker_count = config.get('worker_count')
        self.rollout_size = config.get('rollout_size')
        self.hidden_size = config.get('hidden_size')
        self.ppo_epoch_count = config.get('ppo_epoch_count')
        self.ppo_clip = config.get('ppo_clip')
        self.action_loss_coeff = config.get('action_loss_coeff')
        self.value_loss_coeff = config.get('value_loss_coeff')
        self.entropy_loss_coeff = config.get('entropy_loss_coeff')
        self.grad_norm_max = config.get('grad_norm_max')
        self.action_type = config.get('action_type')

        self.save_dir = config.get('reinforce_save_dir')
        self.load_dir = config.get('reinforce_load_dir')

        self.device = torch.device(config.get('device'))

        if config.get('policy') == 'ppo_pixels_cnn':
            self.policy = PPOPixelsCNN(config).to(self.device)
        assert self.policy is not None

        self.config = config

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            self.learning_rate,
        )

        self.rollouts = PPOStorage(config, self.policy, self.device)

        if self.load_dir:
            if config.get('device') != 'cpu':
                self.policy.load_state_dict(
                    torch.load(self.load_dir + "/policy.pt"),
                )
                self.optimizer.load_state_dict(
                    torch.load(self.load_dir + "/optimizer.pt"),
                )
            else:
                self.policy.load_state_dict(
                    torch.load(self.load_dir + "/policy.pt", map_location='cpu'),
                )
                self.optimizer.load_state_dict(
                    torch.load(self.load_dir + "/optimizer.pt", map_location='cpu'),
                )

        self.final_rewards = torch.zeros([self.worker_count, 1])
        self.episode_rewards = torch.zeros([self.worker_count, 1])
        self.batch_count = 0
        self.start = time.time()
        self.running_reward = None
        self.best_test_reward = 0.0
        self.test_env = None

    def initialize(self):
        self.envs = reinforce.Envs(self.config)
        observation = self.envs.reset()
        observation = self.policy.input(observation)

        self.rollouts.observations[0].copy_(observation.to(self.device))

    def batch_train(self):
        self.policy.train()

        batch_start = time.time()

        for step in range(self.rollout_size):
            with torch.no_grad():
                value, action, hidden, log_prob, entropy = self.policy.action(
                    self.rollouts.observations[step].detach(),
                    self.rollouts.hiddens[step].detach(),
                    self.rollouts.masks[step].detach(),
                )

            observation, reward, done = self.envs.step(
                action.data.cpu().numpy(),
            )

            # if self.action_type == 'discrete':
            #     print("VALUE/CONTROLS/DONE/REWARD/PROGRESS: {:.2f} {} {} {:.2f} {:.2f}".format(
            #         value.data[0][0],
            #         action.data[0][0],
            #         done[0],
            #         reward[0],
            #         observation[0].progress,
            #     ))
            # else:
            #     print("VALUE/STEERING/THROTTLE/DONE/REWARD/PROGRESS: {:.2f} {:.2f} {:.2f} {} {:.2f} {:.2f}".format(
            #         value.data[0][0],
            #         action.data[0][0],
            #         action.data[0][1],
            #         done[0],
            #         reward[0],
            #         observation[0].progress,
            #     ))

            observation = self.policy.input(observation)
            reward = torch.from_numpy(np.expand_dims(reward, 1)).float()
            mask = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done]
            )

            self.episode_rewards += reward
            self.final_rewards *= mask
            self.final_rewards += (1 - mask) * self.episode_rewards
            self.episode_rewards *= mask

            self.rollouts.insert(
                step,
                observation.to(self.device),
                hidden.data,
                action.data,
                log_prob.data,
                value.data,
                reward,
                mask.to(self.device),
            )

        with torch.no_grad():
            next_value = self.policy(
                self.rollouts.observations[-1].detach(),
                self.rollouts.hiddens[-1].detach(),
                self.rollouts.masks[-1].detach(),
            )[0].data

        self.rollouts.compute_returns(next_value)

        advantages = self.rollouts.returns[:-1] - self.rollouts.values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for e in range(self.ppo_epoch_count):
            generator = self.rollouts.feed_forward_generator(advantages)

            for sample in generator:
                observations_batch, \
                    hiddens_batch, \
                    actions_batch, \
                    returns_batch, \
                    masks_batch, \
                    log_probs_batch, \
                    advantage_targets = sample

            values, hiddens, log_probs, entropy = self.policy.evaluate(
                observations_batch.detach(),
                hiddens_batch.detach(),
                masks_batch.detach(),
                actions_batch.detach(),
            )

            ratio = torch.exp(log_probs - log_probs_batch)

            action_loss = -torch.min(
                ratio * advantage_targets,
                torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantage_targets,
            ).mean()
            value_loss = (returns_batch.detach() - values).pow(2).mean()
            entropy_loss = -entropy.mean()

            self.optimizer.zero_grad()

            (value_loss * self.value_loss_coeff +
             action_loss * self.action_loss_coeff +
             entropy_loss * self.entropy_loss_coeff).backward()

            if self.grad_norm_max > 0.0:
                nn.utils.clip_grad_norm(
                    self.policy.parameters(), self.grad_norm_max,
                )

            self.optimizer.step()

        self.rollouts.after_update()

        total_num_steps = (
            (self.batch_count + 1) * self.worker_count * self.rollout_size
        )
        batch_end = time.time()

        print(
            ("STEP {} timesteps {} FPS {} " + \
             "mean/median R {:.1f} {:.1f} " + \
             "min/max R {:.1f} {:.1f} " + \
             "entropy_loss {:.5f} " + \
             "value_loss {:.5f} " + \
             "action_loss {:.5f}").
            format(
                self.batch_count,
                total_num_steps,
                int(self.worker_count * self.rollout_size / (batch_end - batch_start)),
                self.final_rewards.mean(),
                self.final_rewards.median(),
                self.final_rewards.min(),
                self.final_rewards.max(),
                entropy_loss.item(),
                value_loss.item(),
                action_loss.item(),
            ))
        sys.stdout.flush()

        if self.batch_count % 10 == 0 and self.save_dir:
            test_reward = 0.0

            for i in range(8):
                test_reward += self.test()
            self.policy.train()
            test_reward /= 8

            if test_reward > self.best_test_reward:
                self.best_test_reward = test_reward
                print("Saving models and optimizer: save_dir={} test_reward={}".format(
                    self.save_dir,
                    test_reward
                ))
                torch.save(self.policy.state_dict(), self.save_dir + "/policy.pt")
                torch.save(self.optimizer.state_dict(), self.save_dir + "/optimizer.pt")

        self.batch_count += 1
        if self.running_reward is None:
            self.running_reward = self.final_rewards.mean()
        self.running_reward = (
            self.running_reward * 0.9 + self.final_rewards.mean() * 0.1
        )

        return self.running_reward

    def test(self, step_callback=None):
        self.policy.eval()

        test_start = time.time()

        if self.test_env is None:
            self.test_env = reinforce.Donkey(self.config)

        observations = self.policy.input([self.test_env.reset()]).to(self.device)
        hiddens = torch.zeros(1, self.hidden_size).to(self.device)
        masks = torch.ones(1, 1).to(self.device)

        end = False
        final_reward = 0;
        episode_size = 0

        with torch.no_grad():
            while not end:

                value, action, hiddens, log_prob, entropy = self.policy.action(
                    observations.detach(),
                    hiddens.detach(),
                    masks.detach(),
                    deterministic=True,
                )

                observation, reward, done = self.test_env.step(
                    action[0].data.cpu().numpy(),
                )

                if step_callback is not None:
                    step_callback(observation, reward, done)

                # if self.action_type == 'discrete':
                #     print("VALUE/CONTROLS/DONE/REWARD/PROGRESS: {:.2f} {} {} {:.2f} {:.2f}".format(
                #         value.data[0][0],
                #         action.data[0][0],
                #         done[0],
                #         reward[0],
                #         observation[0].progress,
                #     ))
                # else:
                #     print("VALUE/STEERING/THROTTLE/DONE/REWARD/PROGRESS: {:.2f} {:.2f} {:.2f} {} {:.2f} {:.2f}".format(
                #         value.data[0][0],
                #         action.data[0][0],
                #         action.data[0][1],
                #         done[0],
                #         reward[0],
                #         observation[0].progress,
                #     ))
                # sys.stdout.flush()

                final_reward += reward
                end = done

                observations = self.policy.input([observation]).to(self.device)
                masks = torch.FloatTensor(
                    [[0.0] if done else [1.0]]
                ).to(self.device)

                episode_size += 1


        test_end = time.time()

        print("TEST FPS {} step_count {} final_reward {:.1f}".format(
            int(episode_size / (test_end - test_start)),
            episode_size,
            final_reward
        ))

        return final_reward
