import math
import cv2
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, Categorical

import reinforce

# import pdb; pdb.set_trace()

class PPOController(nn.Module):
    def __init__(self, config):
        super(PPOController, self).__init__()
        self.latent_size = config.get('latent_size')
        self.hidden_size = config.get('hidden_size')
        self.recurring_cell = config.get('recurring_cell')
        self.action_type = config.get('action_type')
        self.fixed_action_std = config.get('fixed_action_std')

        self.device = torch.device(config.get('device'))

        self.fc1 = nn.Linear(self.latent_size, self.hidden_size)

        if self.recurring_cell == "gru":
            self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)

        if self.action_type == 'discrete':
            self.fc1_a = nn.Linear(self.hidden_size, reinforce.DISCRETE_CONTROL_SIZE)
        else:
            self.fc1_a = nn.Linear(self.hidden_size, 2 * reinforce.CONTINUOUS_CONTROL_SIZE)

        self.fc1_v = nn.Linear(self.hidden_size, 1)

        self.train()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0)

        nn.init.xavier_normal_(self.fc1.weight.data, nn.init.calculate_gain('relu'))
        self.fc1.bias.data.fill_(0)

        if self.action_type == 'discrete':
            nn.init.xavier_normal_(self.fc1_a.weight.data, nn.init.calculate_gain('linear'))
        else:
            nn.init.xavier_normal_(self.fc1_a.weight.data, nn.init.calculate_gain('tanh'))
        self.fc1_a.bias.data.fill_(0)

        nn.init.xavier_normal_(self.fc1_v.weight.data, nn.init.calculate_gain('linear'))
        self.fc1_v.bias.data.fill_(0)

        if self.recurring_cell == "gru":
            nn.init.xavier_normal_(self.gru.weight_ih.data)
            nn.init.xavier_normal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    def forward(self, latents, hiddens, masks):
        x = F.elu(self.fc1(latents))

        if self.recurring_cell == "gru":
            if latents.size(0) == hiddens.size(0):
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

        # Action head.
        a = None
        if self.action_type == 'discrete':
            a = self.fc1_a(x)
        else:
            a = F.tanh(self.fc1_a(x))

        # Value head.
        v = self.fc1_v(x)

        return v, a, hiddens

        observation = np.concatenate(
            (
                np.stack(cameras),
            ),
            axis=-1,
        )
        observation = torch.from_numpy(observation).float().unsqueeze(1)

        return observation

    def action(self, latents, hiddens, masks, deterministic=False):
        value, x, hiddens = self(latents, hiddens, masks)

        if self.action_type == 'discrete':
            probs = F.softmax(x, dim=1)
            log_probs = F.log_softmax(x, dim=1)

            m = Categorical(probs)
            actions = m.sample().view(-1, 1)

            action_log_probs = log_probs.gather(1, actions)
            entropy = -(log_probs * probs).sum(-1).mean()

            return value, actions, hiddens, action_log_probs, entropy
        else:
            slices = torch.split(x, reinforce.CONTINUOUS_CONTROL_SIZE, 1)
            action_mean = slices[0]
            action_logstd = slices[1]
            action_std = action_logstd.exp()

            if self.fixed_action_std > 0.0:
                action_std = (
                    self.fixed_action_std *
                    torch.ones(action_mean.size()).float()
                ).to(self.device)
                action_logstd = action_std.log()

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

    def evaluate(self, latents, hiddens, masks, actions):
        value, x, hiddens = self(latents, hiddens, masks)

        if self.action_type == 'discrete':
            probs = F.softmax(x, dim=1)
            log_probs = F.log_softmax(x, dim=1)

            m = Categorical(probs)
            actions = m.sample().view(-1, 1)

            action_log_probs = log_probs.gather(1, actions)
            entropy = -(log_probs * probs).sum(-1).mean()

            return value, hiddens, action_log_probs, entropy
        else:
            slices = torch.split(x, reinforce.CONTINUOUS_CONTROL_SIZE, 1)
            action_mean = slices[0]
            action_logstd = slices[1]
            action_std = action_logstd.exp()

            if self.fixed_action_std > 0.0:
                action_std = (
                    self.fixed_action_std *
                    torch.ones(action_mean.size()).float()
                ).to(self.device)
                action_logstd = action_std.log()

            m = Normal(action_mean, action_std)

            # log_probs (sum on actions) -> batch x 1
            log_probs = m.log_prob(actions).sum(-1, keepdim=True)

            # entropy (sum on actions / mean on batch) -> 1x1
            entropy = 0.5 + 0.5 * math.log(2 * math.pi) + action_logstd
            entropy = entropy.sum(-1, keepdim=True)

            return value, hiddens, log_probs, entropy
