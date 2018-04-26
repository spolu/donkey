import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torch.distributions import Normal, Categorical

import donkey

class OrnsteinUhlenbeckNoise:
    def __init__(self, dim, mu = 0, theta = 0.15, sigma = 0.2):
        self.dim = dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = torch.ones(self.dim) * self.mu

    def reset(self):
        self.X = torch.ones(self.dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * torch.rand(self.dim)
        self.X = self.X + dx
        return self.X

class ActorPolicy(nn.Module):
    def __init__(self, config):
        super(ActorPolicy, self).__init__()
        self.hidden_size = config.get('hidden_size')
        self.config = config

        self.noise = OrnsteinUhlenbeckNoise(
            donkey.CONTINUOUS_CONTROL_SIZE,
            mu=torch.Tensor([
                0.0, # steering
                0.3, # throttle_brake
            ]),
            theta=0.3,
            sigma=0.2,
        )

        self.cv1 = nn.Conv2d(donkey.CAMERA_STACK_SIZE, 24, 5, stride=2)
        self.bn1 = nn.BatchNorm2d(24)

        self.cv2 = nn.Conv2d(24, 32, 5, stride=1)
        self.cv3 = nn.Conv2d(32, 64, 5, stride=1)
        self.cv4 = nn.Conv2d(64, 64, 3, stride=2)
        self.cv5 = nn.Conv2d(64, 64, 3, stride=2)

        self.cv6 = nn.Conv2d(64, 1, 1, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(234, self.hidden_size)

        self.fc1_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2_a = nn.Linear(self.hidden_size, donkey.CONTINUOUS_CONTROL_SIZE)

        self.train()

        nn.init.xavier_normal(self.cv1.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.cv2.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.cv3.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.cv4.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.cv5.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.cv6.weight.data, nn.init.calculate_gain('relu'))
        self.cv1.bias.data.fill_(0)
        self.cv2.bias.data.fill_(0)
        self.cv3.bias.data.fill_(0)
        self.cv4.bias.data.fill_(0)
        self.cv5.bias.data.fill_(0)
        self.cv6.bias.data.fill_(0)

        self.bn1.weight.data.fill_(1)
        self.bn2.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        self.bn2.bias.data.zero_()

        nn.init.xavier_normal(self.fc1.weight.data, nn.init.calculate_gain('relu'))
        self.fc1.bias.data.fill_(0)

        nn.init.xavier_normal(self.fc1_a.weight.data, nn.init.calculate_gain('tanh'))
        nn.init.xavier_normal(self.fc2_a.weight.data, nn.init.calculate_gain('tanh'))
        self.fc1_a.bias.data.fill_(0)
        self.fc2_a.bias.data.fill_(0)

    def forward(self, inputs):
        x = F.elu(self.cv1(inputs))
        x = self.bn1(x)
        x = F.elu(self.cv2(x))
        x = F.elu(self.cv3(x))
        x = F.elu(self.cv4(x))
        x = F.elu(self.cv5(x))
        x = F.elu(self.cv6(x))
        x = self.bn2(x)

        x = x.view(-1, 234)

        x = F.elu(self.fc1(x))

        a = F.tanh(self.fc1_a(x))
        a = F.tanh(self.fc2_a(a))

        return a

    def action_shape(self):
        return (
            donkey.CONTINUOUS_CONTROL_SIZE,
        )

    def action_type(self):
        return 'torch.FloatTensor'

    def input_shape(self):
        return (
            donkey.CAMERA_STACK_SIZE,
            donkey.CAMERA_WIDTH,
            donkey.CAMERA_HEIGHT,
        )

    def input_type(self):
        return 'torch.FloatTensor'

    def input(self, observation):
        cameras = [o.camera for o in observation]
        observation = np.concatenate(
            (
                np.stack(cameras),
            ),
            axis=-1,
        )
        observation = torch.from_numpy(observation).float()

        return observation

    def action(self, inputs, exploration=False):
        actions = self.forward(inputs)

        if exploration:
            noise = self.noise.sample()
            if actions.is_cuda:
                noise = noise.cuda()
            actions = actions + autograd.Variable(noise, requires_grad=False)

        return actions

class CriticPolicy(nn.Module):
    def __init__(self, config):
        super(CriticPolicy, self).__init__()
        self.hidden_size = config.get('hidden_size')
        self.config = config

        self.cv1 = nn.Conv2d(donkey.CAMERA_STACK_SIZE, 24, 5, stride=2)
        self.bn1 = nn.BatchNorm2d(24)

        self.cv2 = nn.Conv2d(24, 32, 5, stride=1)
        self.cv3 = nn.Conv2d(32, 64, 5, stride=1)
        self.cv4 = nn.Conv2d(64, 64, 3, stride=2)
        self.cv5 = nn.Conv2d(64, 64, 3, stride=2)

        self.cv6 = nn.Conv2d(64, 1, 1, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(234, self.hidden_size)

        self.mx1 = nn.Linear(
            self.hidden_size + donkey.CONTINUOUS_CONTROL_SIZE,
            self.hidden_size,
        )

        self.fc1_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2_q = nn.Linear(self.hidden_size, 1)

        self.train()

        nn.init.xavier_normal(self.cv1.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.cv2.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.cv3.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.cv4.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.cv5.weight.data, nn.init.calculate_gain('relu'))
        nn.init.xavier_normal(self.cv6.weight.data, nn.init.calculate_gain('relu'))
        self.cv1.bias.data.fill_(0)
        self.cv2.bias.data.fill_(0)
        self.cv3.bias.data.fill_(0)
        self.cv4.bias.data.fill_(0)
        self.cv5.bias.data.fill_(0)
        self.cv6.bias.data.fill_(0)

        self.bn1.weight.data.fill_(1)
        self.bn2.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        self.bn2.bias.data.zero_()

        nn.init.xavier_normal(self.fc1.weight.data, nn.init.calculate_gain('relu'))
        self.fc1.bias.data.fill_(0)

        nn.init.xavier_normal(self.mx1.weight.data, nn.init.calculate_gain('relu'))
        self.mx1.bias.data.fill_(0)

        nn.init.xavier_normal(self.fc1_q.weight.data, nn.init.calculate_gain('tanh'))
        nn.init.xavier_normal(self.fc2_q.weight.data, nn.init.calculate_gain('linear'))
        self.fc1_q.bias.data.fill_(0)
        self.fc2_q.bias.data.fill_(0)

    def forward(self, inputs, actions):
        x = F.elu(self.cv1(inputs))
        x = self.bn1(x)
        x = F.elu(self.cv2(x))
        x = F.elu(self.cv3(x))
        x = F.elu(self.cv4(x))
        x = F.elu(self.cv5(x))
        x = F.elu(self.cv6(x))
        x = self.bn2(x)

        x = x.view(-1, 234)

        x = F.elu(self.fc1(x))
        x = F.elu(self.mx1(torch.cat((x, actions), 1)))

        q = F.tanh(self.fc1_q(x))
        q = self.fc2_q(q)

        return q

    def quality(self, inputs, actions):
        quality = self.forward(inputs, actions)

        return quality
