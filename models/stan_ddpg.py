import sys
import time

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import donkey

# import pdb; pdb.set_trace()

class Storage:
    def __init__(self, config, actor):
        self.rollout_size = config.get('rollout_size')
        self.worker_count = config.get('worker_count')
        self.mini_batch_count = config.get('mini_batch_count')

        self.predecessors = torch.zeros(
            self.rollout_size,
            self.worker_count,
            *(actor.input_shape()),
        ).type(actor.input_type())
        self.actions = torch.zeros(
            self.rollout_size,
            self.worker_count,
            *(actor.action_shape())
        ).type(actor.action_type())
        self.rewards = torch.zeros(
            self.rollout_size,
            self.worker_count,
            1,
        )
        self.successors = torch.zeros(
            self.rollout_size,
            self.worker_count,
            *(actor.input_shape()),
        ).type(actor.input_type())
        self.masks = torch.ones(
            self.rollout_size,
            self.worker_count,
            1,
        )

    def cuda(self):
        self.predecessors = self.predecessors.cuda()
        self.actions = self.actions.cuda()
        self.rewards = self.rewards.cuda()
        self.successors = self.successors.cuda()
        self.masks = self.masks.cuda()

    def insert(self, step, predecessors, actions, rewards, successors, masks):
        self.predecessors[step].copy_(predecessors)
        self.actions[step].copy_(actions)
        self.rewards[step].copy_(rewards)
        self.successors[step].copy_(successors)
        self.masks[step].copy_(masks)

    def feed_forward_generator(self):
        batch_size = self.worker_count * self.rollout_size
        mini_batch_size = batch_size // self.mini_batch_count
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            indices = torch.LongTensor(indices)

            if self.predecessors.is_cuda:
                indices = indices.cuda()

            predecessors_batch = self.predecessors.view(-1, *self.predecessors.size()[2:])[indices]
            actions_batch = self.actions.view(-1, *self.actions.size()[2:])[indices]
            rewards_batch = self.rewards.view(-1, 1)[indices]
            successors_batch = self.successors.view(-1, *self.successors.size()[2:])[indices]
            masks_batch = self.masks.view(-1, 1)[indices]

            yield \
                predecessors_batch, \
                actions_batch, \
                rewards_batch, \
                successors_batch, \
                masks_batch

class Model:
    def __init__(self, config, save_dir=None, load_dir=None):
        self.cuda = config.get('cuda')
        self.actor_learning_rate = config.get('actor_learning_rate')
        self.critic_learning_rate = config.get('critic_learning_rate')
        self.worker_count = config.get('worker_count')
        self.rollout_size = config.get('rollout_size')
        self.grad_norm_max = config.get('grad_norm_max')
        self.epoch_count = config.get('epoch_count')
        self.gamma = config.get('gamma')
        self.tau = config.get('tau')

        module = __import__('policies.' + config.get('policy'))

        self.targ_actor = getattr(module, config.get('policy')).ActorPolicy(config)
        self.curr_actor = getattr(module, config.get('policy')).ActorPolicy(config)
        self.targ_critic = getattr(module, config.get('policy')).CriticPolicy(config)
        self.curr_critic = getattr(module, config.get('policy')).CriticPolicy(config)

        self.save_dir = save_dir
        self.load_dir = load_dir

        self.envs = donkey.Envs(config)

        self.actor_optimizer = optim.Adam(
            self.curr_actor.parameters(),
            self.actor_learning_rate,
        )
        self.critic_optimizer = optim.Adam(
            self.curr_critic.parameters(),
            self.critic_learning_rate,
        )

        self.memory = Storage(config, self.curr_actor)

        self.final_rewards = torch.zeros([self.worker_count, 1])
        self.episode_rewards = torch.zeros([self.worker_count, 1])
        self.batch_count = 0
        self.running_reward = None

    def initialize(self):
        self.observations = self.curr_actor.input(self.envs.reset())

        if self.cuda:
            self.targ_actor.cuda()
            self.curr_actor.cuda()
            self.targ_critic.cuda()
            self.curr_critic.cuda()

            self.memory.cuda()

            self.observations = self.observations.cuda()

    def batch_train(self):
        self.start = time.time()

        for step in range(self.rollout_size):
            actions = self.curr_actor.action(
                autograd.Variable(
                    self.observations, requires_grad=False,
                ),
                exploration=True,
            )

            observations, rewards, dones = self.envs.step(
                actions.data.cpu().numpy(),
            )

            rewards = torch.from_numpy(np.expand_dims(rewards, 1)).float()
            masks = torch.FloatTensor([[0.0] if d else [1.0] for d in dones])
            observations = self.curr_actor.input(observations)

            self.episode_rewards += rewards
            self.final_rewards *= masks
            self.final_rewards += (1 - masks) * self.episode_rewards
            self.episode_rewards *= masks

            if self.cuda:
                masks = masks.cuda()
                observations = observations.cuda()

            self.memory.insert(
                step,
                self.observations,
                actions.data,
                rewards,
                observations,
                masks,
            )

            self.observations = observations

        for e in range(self.epoch_count):
            generator = self.memory.feed_forward_generator()

            for sample in generator:
                predecessors_batch, \
                    actions_batch, \
                    rewards_batch, \
                    successors_batch, \
                    masks_batch = sample

            # Compute expected quality.
            actions = self.targ_actor.action(
                autograd.Variable(
                    successors_batch, requires_grad=False
                ),
                exploration=False,
            )
            quality = self.targ_critic.quality(
                autograd.Variable(
                    successors_batch, requires_grad=False
                ),
                actions,
            )

            expected = rewards_batch + masks_batch * self.gamma * quality.data

            # Compute predicted quality.
            predicted = self.curr_critic.quality(
                autograd.Variable(predecessors_batch),
                autograd.Variable(actions_batch),
            )

            # Critic loss.
            critic_loss = F.smooth_l1_loss(
                predicted, autograd.Variable(expected),
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()

            if self.grad_norm_max > 0.0:
                nn.utils.clip_grad_norm(
                    self.curr_critic.parameters(), self.grad_norm_max,
                )

            self.critic_optimizer.step()

            # Compute predicted quality of predicted action.
            actions = self.curr_actor.action(
                autograd.Variable(predecessors_batch),
                exploration=False,
            )

            quality = self.curr_critic.quality(
                autograd.Variable(predecessors_batch),
                actions,
            )

            # Actor loss.
            actor_loss = -1 * quality.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            if self.grad_norm_max > 0.0:
                nn.utils.clip_grad_norm(
                    self.curr_actor.parameters(), self.grad_norm_max,
                )

            self.actor_optimizer.step()

        # Update target actor.
        for targ_p, curr_p in zip(self.targ_actor.parameters(), self.curr_actor.parameters()):
            targ_p.data.copy_(
                targ_p.data * (1.0 - self.tau) + curr_p.data * self.tau
            )
        # Update target critic.
        for targ_p, curr_p in zip(self.targ_critic.parameters(), self.curr_critic.parameters()):
            targ_p.data.copy_(
                targ_p.data * (1.0 - self.tau) + curr_p.data * self.tau
            )

        end = time.time()
        batch_num_steps = self.worker_count * self.rollout_size
        total_num_steps = (
            (self.batch_count + 1) * batch_num_steps
        )

        print(
            ("STEP {} timesteps {} EPS {} " + \
             "mean/median R {:.1f} {:.1f} " + \
             "min/max R {:.1f} {:.1f} " + \
             "entropy_loss {:.5f} " + \
             "value_loss {:.5f} " + \
             "action_loss {:.5f}").
            format(
                self.batch_count,
                total_num_steps,
                int(batch_num_steps / (end - self.start)),
                self.final_rewards.mean(),
                self.final_rewards.median(),
                self.final_rewards.min(),
                self.final_rewards.max(),
                0.0,
                critic_loss.data[0],
                actor_loss.data[0],
            ))
        sys.stdout.flush()

        self.batch_count += 1

        if self.running_reward is None:
            self.running_reward = self.final_rewards.mean()
        self.running_reward = (
            self.running_reward * 0.9 + self.final_rewards.mean() * 0.1
        )

        return self.running_reward
