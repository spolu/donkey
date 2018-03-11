import sys
import random
import time
import math

import numpy as np

import torch
import torch.distributed as dist
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import donkey

# import pdb; pdb.set_trace()

OBSERVATION_SIZE = 2 + donkey.ANGLES_WINDOW

def preprocess(observation):
    angles = [o.track_angles for o in observation]
    track_position = [[o.track_position] for o in observation]
    speed = [[o.track_speed] for o in observation]
    # position = [o.position  / 100.0 for o in observation]

    observation = np.concatenate(
        (np.stack(angles), np.stack(track_position), np.stack(speed)),
        axis=-1,
    )
    observation = torch.from_numpy(observation).float()

    return observation

class PPOStorage:
    def __init__(self, config):
        self.rollout_size = config.get('rollout_size')
        self.worker_count = config.get('worker_count')
        self.hidden_size = config.get('hidden_size')
        self.gamma = config.get('gamma')
        self.tau = config.get('tau')
        self.mini_batch_count = config.get('mini_batch_count')

        self.observations = torch.zeros(
            self.rollout_size + 1,
            self.worker_count,
            OBSERVATION_SIZE,
        )
        self.hiddens = torch.zeros(
            self.rollout_size + 1, self.worker_count, self.hidden_size,
        )
        self.rewards = torch.zeros(self.rollout_size, self.worker_count, 1)
        self.values = torch.zeros(self.rollout_size + 1, self.worker_count, 1)
        self.returns = torch.zeros(self.rollout_size + 1, self.worker_count, 1)
        self.log_probs = torch.zeros(self.rollout_size, self.worker_count, 1)
        self.actions = torch.zeros(
            self.rollout_size, self.worker_count, donkey.CONTROL_SIZE
        )
        self.masks = torch.ones(self.rollout_size + 1, self.worker_count, 1)

    def cuda(self):
        self.observations = self.observations.cuda()
        self.hiddens = self.hiddens.cuda()
        self.rewards = self.rewards.cuda()
        self.values = self.values.cuda()
        self.returns = self.returns.cuda()
        self.log_probs = self.log_probs.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

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
            indices = torch.LongTensor(indices)

            if advantages.is_cuda:
                indices = indices.cuda()

            observations_batch = self.observations[:-1].view(-1,
                                        *self.observations.size()[2:])[indices]
            hiddens_batch = self.hiddens[:-1].view(-1, self.hiddens.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
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


class PPOPolicy(nn.Module):
    def __init__(self, config):
        super(PPOPolicy, self).__init__()
        self.hidden_size = config.get('hidden_size')
        self.config = config

        self.fc = nn.Linear(OBSERVATION_SIZE, self.hidden_size)
        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)

        self.fc1_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2_a = nn.Linear(self.hidden_size, 2 * donkey.CONTROL_SIZE)

        self.fc1_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2_v = nn.Linear(self.hidden_size, 1)

        self.train()

        nn.init.xavier_normal(self.fc.weight.data, nn.init.calculate_gain('tanh'))
        self.fc.bias.data.fill_(0)

        nn.init.xavier_normal(self.gru.weight_ih.data)
        nn.init.xavier_normal(self.gru.weight_hh.data)
        self.gru.bias_ih.data.fill_(0)
        self.gru.bias_hh.data.fill_(0)

        nn.init.xavier_normal(self.fc1_a.weight.data, nn.init.calculate_gain('tanh'))
        nn.init.xavier_normal(self.fc2_a.weight.data, nn.init.calculate_gain('tanh'))
        self.fc1_a.bias.data.fill_(0)
        self.fc2_a.bias.data.fill_(0)

        nn.init.xavier_normal(self.fc1_v.weight.data, nn.init.calculate_gain('tanh'))
        nn.init.xavier_normal(self.fc2_v.weight.data, nn.init.calculate_gain('linear'))
        self.fc1_v.bias.data.fill_(0)
        self.fc2_v.bias.data.fill_(0)

    def action(self, inputs, hiddens, masks, deterministic=False):
        value, x, hiddens = self(inputs, hiddens, masks)

        slices = torch.split(x, donkey.CONTROL_SIZE, 1)
        action_mean = slices[0]
        action_logstd = slices[1]
        action_std = action_logstd.exp()

        m = Normal(action_mean, action_std)

        if deterministic is False:
            actions = m.sample()
        else:
            actions = action_mean

        # log_probs (sum on actions) -> batch x 1
        log_probs = m.log_prob(actions).sum(-1, keepdim=True)

        # entropy (sum on actions / mean on batch) -> 1x1
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + action_logstd
        entropy = entropy.sum(-1, keepdim=True)

        return value, actions, hiddens, log_probs, entropy

    def evaluate(self, inputs, hiddens, masks, actions):
        value, x, hiddens = self(inputs, hiddens, masks)

        slices = torch.split(x, donkey.CONTROL_SIZE, 1)
        action_mean = slices[0]
        action_logstd = slices[1]
        action_std = action_logstd.exp()

        m = Normal(action_mean, action_std)

        # log_probs (sum on actions) -> batch x 1
        log_probs = m.log_prob(actions).sum(-1, keepdim=True)

        # entropy (sum on actions / mean on batch) -> 1x1
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + action_logstd
        entropy = entropy.sum(-1, keepdim=True)

        return value, hiddens, log_probs, entropy

    def forward(self, inputs, hiddens, masks):
        x = F.tanh(self.fc(inputs))

        if inputs.size(0) == hiddens.size(0):
            x = hiddens = self.gru(x, hiddens * masks)
        else:
            x = x.view(-1, hiddens.size(0), x.size(1))
            masks = masks.view(-1, hiddens.size(0), 1)
            outputs = []
            for i in range(x.size(0)):
                hx = hiddens = self.gru(x[i], hiddens * masks[i])
                outputs.append(hx)
            x = torch.cat(outputs, 0)
        x = F.tanh(x)

        a = F.tanh(self.fc1_a(x))
        a = F.tanh(self.fc2_a(a))

        v = F.tanh(self.fc1_v(x))
        v = self.fc2_v(v)

        return v, a, hiddens


class Model:
    def __init__(self, config, save_dir=None, load_dir=None):
        self.cuda = config.get('cuda')
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
        self.save_dir = save_dir
        self.load_dir = load_dir

        self.envs = donkey.Envs(config)
        self.actor_critic = PPOPolicy(config)
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            self.learning_rate,
        )
        self.rollouts = PPOStorage(config)

        if self.load_dir:
            if self.cuda:
                self.actor_critic.load_state_dict(
                    torch.load(self.load_dir + "/actor_critic.pt"),
                )
                self.optimizer.load_state_dict(
                    torch.load(self.load_dir + "/optimizer.pt"),
                )
            else:
                self.actor_critic.load_state_dict(
                    torch.load(self.load_dir + "/actor_critic.pt", map_location='cpu'),
                )
                self.optimizer.load_state_dict(
                    torch.load(self.load_dir + "/optimizer.pt", map_location='cpu'),
                )

        self.final_rewards = torch.zeros([self.worker_count, 1])
        self.episode_rewards = torch.zeros([self.worker_count, 1])
        self.batch_count = 0
        self.start = time.time()
        self.running_reward = None

    def initialize(self):
        observation = self.envs.reset()
        observation = preprocess(observation)
        self.rollouts.observations[0].copy_(observation)

        if self.cuda:
            self.actor_critic.cuda()
            self.rollouts.cuda()

    def batch_train(self):
        for step in range(self.rollout_size):
            value, action, hidden, log_prob, entropy = self.actor_critic.action(
                autograd.Variable(
                    self.rollouts.observations[step], requires_grad=False,
                ),
                autograd.Variable(
                    self.rollouts.hiddens[step], requires_grad=False,
                ),
                autograd.Variable(
                    self.rollouts.masks[step], requires_grad=False,
                ),
            )

            observation, reward, done = self.envs.step(
                action.data.cpu().numpy(),
            )

            print("VALUE/STEERING/THROTTLE/DONE/REWARD: {:.2f} {:.2f} {:.2f} {} {:.2f}".format(
                value.data[0][0],
                action.data[0][0],
                action.data[0][1],
                done[0],
                reward[0],
            ))
            sys.stdout.flush()

            observation = preprocess(observation)
            reward = torch.from_numpy(np.expand_dims(reward, 1)).float()
            mask = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done]
            )

            self.episode_rewards += reward
            self.final_rewards *= mask
            self.final_rewards += (1 - mask) * self.episode_rewards
            self.episode_rewards *= mask

            if self.cuda:
                mask = mask.cuda()
                observation = observation.cuda()

            self.rollouts.insert(
                step,
                observation,
                hidden.data,
                action.data,
                log_prob.data,
                value.data,
                reward,
                mask,
            )

        next_value = self.actor_critic(
            autograd.Variable(
                self.rollouts.observations[-1], requires_grad=False,
            ),
            autograd.Variable(
                self.rollouts.hiddens[-1], requires_grad=False,
            ),
            autograd.Variable(
                self.rollouts.masks[-1], requires_grad=False,
            ),
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

            values, hiddens, log_probs, entropy = self.actor_critic.evaluate(
                autograd.Variable(observations_batch),
                autograd.Variable(hiddens_batch),
                autograd.Variable(masks_batch),
                autograd.Variable(actions_batch),
            )

            advantage_targets = autograd.Variable(advantage_targets)
            ratio = torch.exp(log_probs - autograd.Variable(log_probs_batch))

            action_loss = -torch.min(
                ratio * advantage_targets,
                torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantage_targets,
            ).mean()
            value_loss = (autograd.Variable(returns_batch) - values).pow(2).mean()
            entropy_loss = -entropy.mean()

            self.optimizer.zero_grad()

            (value_loss * self.value_loss_coeff +
             action_loss * self.action_loss_coeff +
             entropy_loss * self.entropy_loss_coeff).backward()

            nn.utils.clip_grad_norm(self.actor_critic.parameters(), self.grad_norm_max)

            self.optimizer.step()

        self.rollouts.after_update()

        end = time.time()
        total_num_steps = (
            (self.batch_count + 1) * self.worker_count * self.rollout_size
        )

        print(
            ("{}, timesteps {}, FPS {}, " + \
             "mean/median R {:.1f}/{:.1f}, " + \
             "min/max R {:.1f}/{:.1f}, " + \
             "entropy loss {:.5f}, value loss {:.5f}, action loss {:.5f}").
            format(
                self.batch_count,
                total_num_steps,
                int(total_num_steps / (end - self.start)),
                self.final_rewards.mean(),
                self.final_rewards.median(),
                self.final_rewards.min(),
                self.final_rewards.max(),
                entropy_loss.data[0],
                value_loss.data[0],
                action_loss.data[0],
            ))

        if self.batch_count % 10 == 0 and self.save_dir:
            print("Saving models and optimizer: save_dir={}".format(self.save_dir))
            torch.save(self.actor_critic.state_dict(), self.save_dir + "/actor_critic.pt")
            torch.save(self.optimizer.state_dict(), self.save_dir + "/optimizer.pt")

        self.batch_count += 1
        if self.running_reward is None:
            self.running_reward = self.final_rewards.mean()
        self.running_reward = (
            self.running_reward * 0.9 + self.final_rewards.mean() * 0.1
        )

        return self.running_reward

    def run(self):
        self.actor_critic.eval()

        end = False
        final_reward = 0;

        assert self.worker_count == 1
        assert self.cuda == False

        while not end:
            for step in range(self.rollout_size):
                value, action, hidden, log_prob, entropy = self.actor_critic.action(
                    autograd.Variable(
                        self.rollouts.observations[step], requires_grad=False,
                    ),
                    autograd.Variable(
                        self.rollouts.hiddens[step], requires_grad=False,
                    ),
                    autograd.Variable(
                        self.rollouts.masks[step], requires_grad=False,
                    ),
                    deterministic=True,
                )

                observation, reward, done = self.envs.step(
                    action.data.numpy(),
                    differential=False,
                )

                print("VALUE/STEERING/THROTTLE/DONE/REWARD: {:.2f} {:.2f} {:.2f} {} {:.2f}".format(
                    value.data[0][0],
                    action.data[0][0],
                    action.data[0][1],
                    done[0],
                    reward[0],
                ))
                sys.stdout.flush()

                final_reward += reward[0]
                end = done[0]

                if end:
                    print("REWARD: {}".format(final_reward))
                    final_reward = 0.0

                observation = preprocess(observation)
                reward = torch.from_numpy(np.expand_dims(reward, 1)).float()
                mask = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done]
                )

                if self.cuda:
                    mask = mask.cuda()
                    observation = observation.cuda()

                self.rollouts.insert(
                    step,
                    observation,
                    hidden.data,
                    action.data,
                    log_prob.data,
                    value.data,
                    reward,
                    mask,
                )

            self.rollouts.after_update()
