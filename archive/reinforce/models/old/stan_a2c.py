import sys
import time

import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import donkey

# import pdb; pdb.set_trace()

class A2CStorage:
    def __init__(self, config, policy):
        self.rollout_size = config.get('rollout_size')
        self.worker_count = config.get('worker_count')
        self.hidden_size = config.get('hidden_size')
        self.gamma = config.get('gamma')
        self.tau = config.get('tau')

        self.observations = torch.zeros(
            self.rollout_size + 1,
            self.worker_count,
            *(policy.input_shape()),
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
        self.masks = torch.ones(self.rollout_size + 1, self.worker_count, 1)

    def cuda(self):
        self.observations = self.observations.cuda()
        self.hiddens = self.hiddens.cuda()
        self.rewards = self.rewards.cuda()
        self.values = self.values.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, step, observations, hidden, action, value, reward, mask):
        self.observations[step + 1].copy_(observations)
        self.hiddens[step + 1].copy_(hidden)
        self.masks[step + 1].copy_(mask)
        self.actions[step].copy_(action)
        self.values[step].copy_(value)
        self.rewards[step].copy_(reward)

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

class Model:
    def __init__(self, config, policy, save_dir=None, load_dir=None):
        self.cuda = config.get('cuda')
        self.learning_rate = config.get('learning_rate')
        self.worker_count = config.get('worker_count')
        self.rollout_size = config.get('rollout_size')
        self.hidden_size = config.get('hidden_size')
        self.action_loss_coeff = config.get('action_loss_coeff')
        self.value_loss_coeff = config.get('value_loss_coeff')
        self.entropy_loss_coeff = config.get('entropy_loss_coeff')

        self.policy = policy

        self.save_dir = save_dir
        self.load_dir = load_dir

        self.envs = donkey.Envs(config)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            self.learning_rate,
        )

        self.rollouts = A2CStorage(config, policy)

        if self.load_dir:
            if self.cuda:
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

    def initialize(self):
        observation = self.envs.reset()
        observation = self.policy.input(observation)
        self.rollouts.observations[0].copy_(observation)

        if self.cuda:
            self.policy.cuda()
            self.rollouts.cuda()

    def batch_train(self):
        for step in range(self.rollout_size):
            value, action, auxiliary, hidden, log_prob, entropy = self.policy.action(
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
                differential=False,
            )

            # print("VALUE/STEERING/THROTTLE/DONE/REWARD/PROGRESS: {:.2f} {:.2f} {:.2f} {} {:.2f} {:.2f}".format(
            #     value.data[0][0],
            #     action.data[0][0],
            #     action.data[0][1],
            #     done[0],
            #     reward[0],
            #     observation[0].progress,
            # ))

            observation = self.policy.input(observation)
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
                value.data,
                reward,
                mask,
            )

        next_value = self.policy(
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

        values, auxiliaries, hiddens, log_probs, entropy = self.policy.evaluate(
            autograd.Variable(self.rollouts.observations[:-1].view(
                -1,
                *(self.policy.input_shape()),
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
        sys.stdout.flush()

        if self.batch_count % 100 == 0 and self.save_dir:
            print("Saving models and optimizer: save_dir={}".format(self.save_dir))
            torch.save(self.policy.state_dict(), self.save_dir + "/policy.pt")
            torch.save(self.optimizer.state_dict(), self.save_dir + "/optimizer.pt")

        self.batch_count += 1
        if self.running_reward is None:
            self.running_reward = self.final_rewards.mean()
        self.running_reward = (
            self.running_reward * 0.99 + self.final_rewards.mean() * 0.01
        )

        return self.running_reward

    def run(self):
        self.policy.eval()

        end = False
        final_reward = 0;

        assert self.worker_count == 1
        assert self.cuda == False

        while not end:
            for step in range(self.rollout_size):
                value, action, auxiliary, hidden, entropy = self.policy.action(
                    autograd.Variable(
                        self.rollouts.observations[step], requires_grad=False,
                    ),
                    autograd.Variable(
                        self.rollouts.hiddens[step], requires_grad=False,
                    ),
                    autograd.Variable(
                        self.rollouts.masks[step], requires_grad=False,
                    ),
                    deterministic=False,
                )

                observation, reward, done = self.envs.step(
                    action.data.numpy(),
                    differential=False,
                )

                print("VALUE/STEERING/THROTTLE/DONE/REWARD/PROGRESS: {:.2f} {:.2f} {:.2f} {} {:.2f} {:.2f}".format(
                    value.data[0][0],
                    action.data[0][0],
                    action.data[0][1],
                    done[0],
                    reward[0],
                    observation[0].progress,
                ))
                sys.stdout.flush()


                final_reward += reward[0]
                end = done[0]

                if end:
                    print("REWARD: {}".format(final_reward))
                    final_reward = 0.0

                observation = self.policy.input(observation)
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
                    value.data,
                    reward,
                    mask,
                )

            self.rollouts.after_update()
