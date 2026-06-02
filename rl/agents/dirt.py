import copy

import numpy as np
import torch
import torch.nn.functional as F

from modules.critics import Critic
from utils.general_utils import AttrDict
from .dp import DP

class DIRT(DP):
    def __init__(self, env_params, sampler, agent_cfg):
        super().__init__(env_params, sampler, agent_cfg)

        self.discount = agent_cfg.discount
        self.reward_scale = agent_cfg.reward_scale
        self.soft_target_tau = agent_cfg.soft_target_tau
        self.aux_weight = agent_cfg.aux_weight
        self.p_dist = agent_cfg.p_dist
        self.offline_steps = agent_cfg.offline_steps

        self.stage = 1
        self.critic_warmup_steps = 300
        self.warmup_counter = 0

        self.actor_target = copy.deepcopy(self.actor).to(agent_cfg.device)
        self.critic = Critic(self.dimo+self.dimg+self.dima, agent_cfg.hidden_dim).to(agent_cfg.device)
        self.critic_target = copy.deepcopy(self.critic).to(agent_cfg.device)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=agent_cfg.critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=agent_cfg.actor_lr)

    def get_action(self, state, noise=False):
        with torch.no_grad():
            o, g = state['observation'], state['desired_goal']
            input_tensor = self._preproc_inputs(o, g)

            action = self.actor(input_tensor).cpu().data.numpy().flatten()

            # action = (action + self.max_action * self.noise_eps * np.random.randn(action.shape[0])).clip(
            #         -self.max_action, self.max_action)

            if noise:
                action = (action + self.max_action * self.noise_eps * np.random.randn(action.shape[0])).clip(
                    -self.max_action, self.max_action)

        return action

    def update_offline(self, demo_buffer):
        print("Stage 1: Offline Diffusion Pretraining...")
        for i in range(self.offline_steps):
            obs, action, _, _, _ = self.get_samples(demo_buffer)
            actions = torch.unsqueeze(action, dim=1)
            dp_loss = self.actor(obs, actions=actions).mean()

            self.actor_optimizer.zero_grad()
            dp_loss.backward()
            self.actor_optimizer.step()

            if i % 2000 == 0:
                print(f'[offline] step={i}, dp_loss={dp_loss.item():.4f}')

        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(p.data)
        print("Offline pretraining complete ✅")
        self.stage = 2


    def critic_warmup(self, replay_buffer, demo_buffer):
        print("Stage 2: Critic warm-up phase...")
        metrics = dict()
        self.actor.eval()
        for p in self.actor.parameters():
            p.requires_grad = False

        obs, action, reward, done, next_obs = self.get_samples(replay_buffer)
        metrics.update(self.update_critic(obs, action, reward, next_obs))

        # obs, action, reward, done, next_obs = self.get_samples(demo_buffer)
        # self.update_critic(obs, action, reward, next_obs)
        # metrics.update(self.update_critic(obs, action, reward, next_obs))

        metrics['actor_loss'] = 0

        self.warmup_counter += 1
        if self.warmup_counter >= self.critic_warmup_steps:
            print("Critic warm-up complete ✅, entering joint fine-tuning.")

            for p in self.actor.parameters():
                p.requires_grad = True
            self.actor.train()
            self.stage = 3
        return metrics

    def update_joint(self, replay_buffer, demo_buffer):
        metrics = dict()
        for i in range(self.update_epoch):
            obs, action, reward, done, next_obs = self.get_samples(replay_buffer)
            metrics.update(self.update_critic(obs, action, reward, next_obs))
            metrics.update(self.update_actor(obs, action))

            obs, action, reward, done, next_obs = self.get_samples(demo_buffer)
            self.update_critic(obs, action, reward, next_obs)
            self.update_actor(obs, action, is_demo=True)

            self.update_target()
        return metrics

    def update(self, replay_buffer, demo_buffer):
        if self.stage == 1:
            self.update_offline(demo_buffer)
            return {}

        elif self.stage == 2:
            metrics = self.critic_warmup(replay_buffer, demo_buffer)
            return metrics

        elif self.stage == 3:
            metrics = self.update_joint(replay_buffer, demo_buffer)
            return metrics

    def update_critic(self, obs, action, reward, next_obs):
        with torch.no_grad():
            action_out = self.actor_target(next_obs)

            action_out = torch.squeeze(action_out, dim=1)

            target_V = self.critic_target(next_obs, action_out)
            target_Q = self.reward_scale * reward + (self.discount * target_V).detach()

            clip_return = 1 / (1 - self.discount)
            target_Q = torch.clamp(target_Q, -clip_return, 0).detach()

        Q = self.critic(obs, action)
        critic_loss = F.mse_loss(Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics = AttrDict(
            critic_q=Q.mean().item(),
            critic_target_q=target_Q.mean().item(),
            critic_loss=critic_loss.item(),
            bacth_reward=reward.mean().item()
        )
        return metrics

    def update_actor(self, obs, action, is_demo=False):
        metrics = dict()

        action_out = self.actor(obs)

        action_out = torch.squeeze(action_out, dim=1)

        Q_out = self.critic(obs, action_out)

        if is_demo:
            actions = torch.unsqueeze(action, dim=1)

            dp_loss = self.actor(obs,actions=actions)

            dp_loss = torch.squeeze(dp_loss, dim=1)

            actor_loss = -(Q_out + self.aux_weight * (-dp_loss)).mean()
        else:
            actor_loss = -(Q_out).mean()

        actor_loss += action_out.pow(2).mean()

        # Optimize actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        metrics['actor_loss'] = actor_loss.item()
        return metrics

    def update_target(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.soft_target_tau * param.data + (1 - self.soft_target_tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.soft_target_tau * param.data + (1 - self.soft_target_tau) * target_param.data)
