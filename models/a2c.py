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
from eventlet.green import threading
from utils import OrnsteinUhlenbeckNoise

import donkey

# import pdb; pdb.set_trace()

_send_condition = threading.Condition()
_recv_condition = threading.Condition()
_recv_count = 0

class A2CWorker(threading.Thread):
    def __init__(self):
        self.condition = threading.Condition()
        self.controls = None
        self.observation = None
        self.reward = 0.0
        self.done = False
        self.donkey = donkey.Donkey(headless=True)
        threading.Thread.__init__(self)

    def reset(self):
        self.controls = None
        self.reward = 0.0
        self.done = False
        self.observation = self.donkey.reset()

    def run(self):
        global _recv_count
        global _send_condition
        global _recv_condition
        while True:
            # Wait for the controls to be set.
            _send_condition.acquire()
            _send_condition.wait()
            _send_condition.release()

            observation, reward, done = self.donkey.step(self.controls)

            self.observation = observation
            self.reward = reward
            self.done = done

            # Notify that we are done.
            _recv_condition.acquire()
            _recv_count = _recv_count + 1
            _recv_condition.notify_all()
            _recv_condition.release()

class A2CEnvs:
    def __init__(self, config):
        self.worker_count = config.get('worker_count')
        self.config = config
        self.workers = [A2CWorker() for _ in range(self.worker_count)]
        for w in self.workers:
            w.start()

    def reset(self):
        for w in self.workers:
            w.reset()
        observations = [w.observation for w in self.workers]

        return np.stack(observations)

    def step(self, controls):
        global _recv_count
        global _send_condition
        global _recv_condition

        _recv_condition.acquire()
        _recv_count = 0

        for i in range(len(self.workers)):
            w = self.workers[i]
            w.controls = controls[i]

            # Release the workers.
            _send_condition.acquire()
            _send_condition.notify()
            _send_condition.release()

        # Wait for the workers to finish.
        first = True
        while _recv_count < len(self.workers):
            if first:
                first = False
            else:
                _recv_condition.acquire()
            _recv_condition.wait()
            _recv_condition.release()

        dones = [w.done for w in self.workers]
        rewards = [w.reward for w in self.workers]
        observations = [w.observation for w in self.workers]

        return np.stack(observations), np.stack(rewards), np.stack(dones)

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
            donkey.CAMERA_CHANNEL,
            donkey.CAMERA_WIDTH,
            donkey.CAMERA_HEIGHT
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

        self.conv1 = nn.Conv2d(donkey.CAMERA_CHANNEL, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.linear1 = nn.Linear(32 * 16 * 11, self.hidden_size)

        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size, True)

        self.actor = nn.Linear(self.hidden_size, 2 * donkey.CONTROL_SIZE)
        self.critic = nn.Linear(self.hidden_size, 1)

        self.train()

        nn.init.orthogonal(self.conv1.weight.data)
        nn.init.orthogonal(self.conv2.weight.data)
        nn.init.orthogonal(self.conv3.weight.data)
        nn.init.orthogonal(self.linear1.weight.data)
        nn.init.orthogonal(self.actor.weight.data)
        nn.init.orthogonal(self.critic.weight.data)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

        nn.init.orthogonal(self.gru.weight_ih.data)
        nn.init.orthogonal(self.gru.weight_hh.data)
        self.gru.bias_ih.data.fill_(0)
        self.gru.bias_hh.data.fill_(0)

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

        log_probs = m.log_prob(actions).sum(-1, keepdim=True)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + action_logstd
        entropy = entropy.sum(-1).mean()

        return value, actions, hiddens, log_probs, entropy

    def evaluate(self, inputs, hiddens, masks, actions):
        value, x, hiddens = self(inputs, hiddens, masks)

        slices = torch.split(x, donkey.CONTROL_SIZE, 1)

        action_mean = slices[0]
        action_logstd = slices[1]
        action_std = action_logstd.exp()

        m = Normal(action_mean, action_std)

        log_probs = m.log_prob(actions).sum(-1, keepdim=True)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + action_logstd
        entropy = entropy.sum(-1).mean()

        return value, hiddens, log_probs, entropy

    def forward(self, inputs, hiddens, masks):
        x = self.conv1(inputs)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 16 * 11)
        x = self.linear1(x)
        x = F.relu(x)

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

        return self.critic(x), self.actor(x), hiddens

class A2C:
    def __init__(self, config, args=None):
        self.cuda = config.get('cuda')
        self.learning_rate = config.get('learning_rate')
        self.worker_count = config.get('worker_count')
        self.rollout_size = config.get('rollout_size')
        self.hidden_size = config.get('hidden_size')
        self.action_loss_coeff = config.get('action_loss_coeff')
        self.value_loss_coeff = config.get('value_loss_coeff')
        self.entropy_loss_coeff = config.get('entropy_loss_coeff')
        self.max_grad_norm = config.get('max_grad_norm')

        self.envs = A2CEnvs(config)
        self.actor_critic = A2CGRUPolicy(config)
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            self.learning_rate,
        )
        self.rollouts = A2CStorage(config)

        self.final_rewards = torch.zeros([self.worker_count, 1])
        self.episode_rewards = torch.zeros([self.worker_count, 1])
        self.batch_count = 0
        self.start = time.time()
        self.running_reward = None

    def initialize(self):
        observations = self.envs.reset()
        observations = torch.from_numpy(observations).float()
        self.rollouts.observations[0].copy_(observations)

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

            a = action.data.cpu().numpy()
            observation, reward, done = self.envs.step(a)

            observation = torch.from_numpy(observation).float()
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

            observation *= mask.unsqueeze(2).unsqueeze(2)

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
                donkey.CAMERA_CHANNEL,
                donkey.CAMERA_WIDTH,
                donkey.CAMERA_HEIGHT
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

        self.optimizer.zero_grad()

        (value_loss * self.value_loss_coeff +
         action_loss * self.action_loss_coeff -
         entropy * self.entropy_loss_coeff).backward()

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
                entropy.data[0],
                value_loss.data[0],
                action_loss.data[0],
            ))

        self.batch_count += 1
        if self.running_reward is None:
            self.running_reward = self.final_rewards.mean()
        self.running_reward = (
            self.running_reward * 0.99 + self.final_rewards.mean() * 0.01
        )

        return self.running_reward
