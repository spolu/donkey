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
from utils import OrnsteinUhlenbeckNoise

import donkey

# import pdb; pdb.set_trace()

OBSERVATION_SIZE = 9

def preprocess(observation):
    track = [o.track for o in observation]
    velocity = [o.velocity for o in observation]
    position = [o.position for o in observation]

    observation = np.concatenate(
        (np.stack(track), np.stack(position), np.stack(velocity)),
        axis=-1,
    )
    observation = torch.from_numpy(observation).float()

    return observation

class A2CStorage:
    def __init__(self, config):
        self.rollout_size = config.get('rollout_size')
        self.worker_count = config.get('worker_count')
        self.hidden_size = config.get('hidden_size')
        self.gamma = config.get('gamma')
        self.tau = config.get('tau')

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
        self.actions = torch.zeros(
            self.rollout_size, self.worker_count, donkey.CONTROL_SIZE
        )
        self.log_probs = torch.zeros(self.rollout_size, self.worker_count, 1)
        self.masks = torch.ones(self.rollout_size + 1, self.worker_count, 1)

    def cuda(self):
        self.observations = self.observations.cuda()
        self.hiddens = self.hiddens.cuda()
        self.rewards = self.rewards.cuda()
        self.values = self.values.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.log_probs = self.log_probs.cuda()
        self.masks = self.masks.cuda()

    def insert(self, step, obs, hidden, action, log_probs, value, reward, mask):
        self.observations[step + 1].copy_(obs)
        self.hiddens[step + 1].copy_(hidden)
        self.masks[step + 1].copy_(mask)
        self.actions[step].copy_(action)
        self.log_probs[step].copy_(log_probs)
        self.values[step].copy_(value)
        self.rewards[step].copy_(reward)

    def compute_returns(self, next_value):
        self.values[-1] = next_value
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

class A2CGRUPolicy(nn.Module):
    def __init__(self, config):
        super(A2CGRUPolicy, self).__init__()
        self.hidden_size = config.get('hidden_size')
        self.config = config

        self.linear1 = nn.Linear(OBSERVATION_SIZE, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)

        self.actor = nn.Linear(self.hidden_size, donkey.CONTROL_SIZE)
        self.critic = nn.Linear(self.hidden_size, 1)

        self.train()

        nn.init.xavier_normal(self.linear1.weight.data)
        nn.init.xavier_normal(self.linear2.weight.data)
        nn.init.xavier_normal(self.actor.weight.data)
        nn.init.xavier_normal(self.critic.weight.data)

    def action(self, inputs, hiddens, masks, deterministic=False):
        value, x, hiddens = self(inputs, hiddens, masks)

        action_mean = x
        action_std = 0.2 * torch.ones(x.size()).float()
        if self.config.get('cuda'):
            action_std = action_std.cuda()
        action_std = autograd.Variable(action_std)
        action_logstd = action_std.log()

        # slices = torch.split(x, donkey.CONTROL_SIZE, 1)
        # action_mean = slices[0]
        # action_logstd = slices[1]
        # action_std = action_logstd.exp()

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

        # print("ACTION MEAN {}".format(action_mean.data))
        # print("ACTION LOGSTD {}".format(action_logstd.data))
        # print("ACTION LOGSTD ENTROPY {}".format(action_logstd.data.sum(-1).mean()))
        # print("ACTION STD {}".format(action_std.data))

        return value, actions, hiddens, log_probs, entropy

    def evaluate(self, inputs, hiddens, masks, actions):
        value, x, hiddens = self(inputs, hiddens, masks)

        action_mean = x
        action_std = 0.2 * torch.ones(x.size()).float()
        if self.config.get('cuda'):
            action_std = action_std.cuda()
        action_std = autograd.Variable(action_std)
        action_logstd = action_std.log()

        # slices = torch.split(x, donkey.CONTROL_SIZE, 1)
        # action_mean = slices[0]
        # action_logstd = slices[1]
        # action_std = action_logstd.exp()

        m = Normal(action_mean, action_std)

        # log_probs (sum on actions) -> batch x 1
        log_probs = m.log_prob(actions).sum(-1, keepdim=True)

        # entropy (sum on actions / mean on batch) -> 1x1
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + action_logstd
        entropy = entropy.sum(-1, keepdim=True)

        return value, hiddens, log_probs, entropy

    def forward(self, inputs, hiddens, masks):
        x = self.linear1(inputs)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)

        return self.critic(x), self.actor(x), hiddens

class Model:
    def __init__(self, config, save_dir=None, load_dir=None):
        self.cuda = config.get('cuda')
        self.learning_rate = config.get('learning_rate')
        self.worker_count = config.get('worker_count')
        self.rollout_size = config.get('rollout_size')
        self.hidden_size = config.get('hidden_size')
        self.action_loss_coeff = config.get('action_loss_coeff')
        self.value_loss_coeff = config.get('value_loss_coeff')
        self.entropy_loss_coeff = config.get('entropy_loss_coeff')
        self.max_grad_norm = config.get('max_grad_norm')
        self.save_dir = save_dir
        self.load_dir = load_dir

        self.envs = donkey.Envs(config)
        self.actor_critic = A2CGRUPolicy(config)
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            self.learning_rate,
        )
        self.rollouts = A2CStorage(config)

        if self.load_dir:
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

            print("VALUE: {}".format(value.data[0][0]))

            observation, reward, done = self.envs.step(
                action.data.cpu().numpy(),
            )

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

            observation *= mask

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

        values, hiddens, log_probs, entropy = self.actor_critic.evaluate(
            autograd.Variable(self.rollouts.observations[:-1].view(
                -1,
                OBSERVATION_SIZE,
            )),
            autograd.Variable(self.rollouts.hiddens[0].view(
                -1, self.hidden_size,
            )),
            autograd.Variable(self.rollouts.masks[:-1].view(
                -1, 1,
            )),
            autograd.Variable(self.rollouts.actions.view(
                -1, donkey.CONTROL_SIZE,
            )),
        )

        values = values.view(self.rollout_size, self.worker_count, 1)
        log_probs = log_probs.view(self.rollout_size, self.worker_count, 1)

        advantages = autograd.Variable(self.rollouts.returns[:-1]) - values

        value_loss = advantages.pow(2).mean()
        action_loss = -(autograd.Variable(advantages.data) * log_probs).mean()
        entropy_loss = -entropy.mean()

        self.optimizer.zero_grad()

        (value_loss * self.value_loss_coeff +
         action_loss * self.action_loss_coeff +
         entropy_loss * self.entropy_loss_coeff).backward()

        nn.utils.clip_grad_norm(
            self.actor_critic.parameters(), self.max_grad_norm,
        )

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

        if self.batch_count % 100 == 0 and self.save_dir:
            print("Saving models and optimizer: save_dir={}".format(self.save_dir))
            torch.save(self.actor_critic.state_dict(), self.save_dir + "/actor_critic.pt")
            torch.save(self.optimizer.state_dict(), self.save_dir + "/optimizer.pt")

        self.batch_count += 1
        if self.running_reward is None:
            self.running_reward = self.final_rewards.mean()
        self.running_reward = (
            self.running_reward * 0.99 + self.final_rewards.mean() * 0.01
        )

        return self.running_reward

    def run(self):
        self.actor_critic.eval()

        end = False
        final_reward = 0;

        assert self.worker_count == 1
        assert self.cuda == False

        while not end:
            value, action, hidden, log_prob, entropy = self.actor_critic.action(
                autograd.Variable(
                    self.rollouts.observations[0], requires_grad=False,
                ),
                autograd.Variable(
                    self.rollouts.hiddens[0], requires_grad=False,
                ),
                autograd.Variable(
                    self.rollouts.masks[0], requires_grad=False,
                ),
                deterministic=True,
            )

            observation, reward, done = self.envs.step(
                action.data.numpy(),
            )

            print("VALUE: {}".format(value.data[0][0]))
            print("REWARD: {}".format(reward[0]))
            # print("LOG_PROB: {}".format(log_prob.data[0][0]))
            # print("ENTROPY: {}".format(entropy.data[0]))
            print("ACTION: {}".format(action.data.numpy()))

            final_reward += reward[0]
            end = done[0]

            observation = preprocess(observation)
            reward = torch.from_numpy(np.expand_dims(reward, 1)).float()

            self.rollouts.insert(
                -1,
                observation,
                hidden.data,
                action.data,
                log_prob.data,
                value.data,
                reward,
                torch.ones(self.worker_count, 1),
            )

        return final_reward