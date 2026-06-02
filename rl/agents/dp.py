import copy

import numpy as np
import torch
import torch.nn.functional as F

from components.normalizer import Normalizer
from modules.critics import Critic
from modules.policies import DeterministicActor
from modules.difussion_policy import DiffusionPolicy
from utils.general_utils import AttrDict
from .base import BaseAgent

class DP(BaseAgent):
    '''Refer to https://arXiv:2303.04137. '''
    '''Refer to https://arXiv:2401.02117. '''
    def __init__(
        self,
        env_params,
        sampler,
        agent_cfg,
    ):
        super().__init__()

        self.update_epoch = agent_cfg.update_epoch
        self.sampler = sampler    # same as which in buffer
        self.device = agent_cfg.device

        self.noise_eps = agent_cfg.noise_eps

        self.clip_obs = agent_cfg.clip_obs
        self.norm_clip = agent_cfg.norm_clip
        self.norm_eps = agent_cfg.norm_eps

        self.dima = env_params['act']
        self.dimo, self.dimg = env_params['obs'], env_params['goal']

        self.max_action = env_params['max_action']
        self.act_sampler = env_params['act_rand_sampler']

        # normarlizer
        self.o_norm = Normalizer(
            size=self.dimo,
            default_clip_range=self.norm_clip,
            eps=agent_cfg.norm_eps
        )
        self.g_norm = Normalizer(
            size=self.dimg,
            default_clip_range=self.norm_clip,
            eps=agent_cfg.norm_eps
        )

        # difussion cfg
        self.weight_decay = 0

        # build policy
        self.actor = DiffusionPolicy(
            self.dimo+self.dimg, self.dima, agent_cfg
        ).to(agent_cfg.device)

        # optimizer
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=agent_cfg.actor_lr, weight_decay=self.weight_decay
        )

    def get_action(self, state, noise=False):
        with torch.no_grad():
            o, g = state['observation'], state['desired_goal']
            input_tensor = self._preproc_inputs(o, g)

            action = self.actor(input_tensor).cpu().data.numpy().flatten()

            # Gaussian noise
            if noise:
                action = (action + self.max_action * self.noise_eps * np.random.randn(action.shape[0])).clip(
                    -self.max_action, self.max_action)

        return action

    def update_actor(self, obs, action):
        metrics = dict()

        actions = torch.unsqueeze(action, dim=1)
        dp_loss = self.actor(obs,actions=actions).mean()

        # Optimize actor loss
        self.actor_optimizer.zero_grad()
        dp_loss.backward()
        self.actor_optimizer.step()

        metrics['dp_loss'] = dp_loss.item()
        return metrics

    def update(self, replay_buffer, demo_buffer):
        metrics = dict()

        for i in range(self.update_epoch):
            # sample from replay buffer
            obs, action, reward, done, next_obs = self.get_samples(demo_buffer)

            # update critic and actor
            metrics.update(self.update_actor(obs, action))

        return metrics
