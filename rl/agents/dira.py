import copy

import numpy as np
import torch
import torch.nn.functional as F

from modules.critics import Critic
from utils.general_utils import AttrDict
from .dp import DP


class DIRA(DP):
    """Diffusion-Based Imitation-to-Reinforcement Adaptation.

    The implementation follows the three stages described in the DIRA paper:
    diffusion pretraining with entropy regularization, value-consistent critic
    warm-up with a frozen actor, and online imitation-RL co-training with SPG.
    """

    def __init__(self, env_params, sampler, agent_cfg):
        super().__init__(env_params, sampler, agent_cfg)

        self.discount = agent_cfg.discount
        self.reward_scale = agent_cfg.reward_scale
        self.soft_target_tau = agent_cfg.soft_target_tau

        self.offline_steps = agent_cfg.offline_steps
        self.critic_warmup_steps = agent_cfg.critic_warmup_steps
        self.hybrid_demo_ratio = agent_cfg.hybrid_demo_ratio
        self.expert_weight = self._cfg(agent_cfg, "expert_weight", self._cfg(agent_cfg, "aux_weight", 1.0))
        self.entropy_weight = self._cfg(agent_cfg, "entropy_weight", 0.0)
        self.entropy_num_steps = self._cfg(agent_cfg, "entropy_num_steps", 4)
        self.action_reg_weight = self._cfg(agent_cfg, "action_reg_weight", 1.0)
        self.target_clip = self._cfg(agent_cfg, "target_clip", 1.0 / (1.0 - self.discount))

        self.use_online_spg = self._cfg(agent_cfg, "use_online_spg", True)
        self.use_demo_spg = self._cfg(agent_cfg, "use_demo_spg", True)
        self.spg_sigma = self._cfg(agent_cfg, "spg_sigma", self.noise_eps)
        self.spg_clip = self._cfg(agent_cfg, "spg_clip", self.max_action)

        self.stage = 1
        self.offline_updated = False
        self.warmup_counter = 0
        self.actor_frozen = False

        self.actor_target = copy.deepcopy(self.actor).to(agent_cfg.device)
        self.critic = Critic(self.dimo + self.dimg + self.dima, agent_cfg.hidden_dim).to(agent_cfg.device)
        self.critic_target = copy.deepcopy(self.critic).to(agent_cfg.device)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=agent_cfg.critic_lr)

    @staticmethod
    def _cfg(cfg, key, default):
        return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)

    def pretrain_on_demo(self, demo_buffer):
        """Optional trainer hook used before the first online rollout."""
        if not self.offline_updated:
            self.update_offline(demo_buffer)

    def train(self, mode=True):
        super().train(mode)
        if self.actor_frozen:
            self.actor.eval()
        return self

    def get_action(self, state, noise=False):
        with torch.no_grad():
            o, g = state["observation"], state["desired_goal"]
            input_tensor = self._preproc_inputs(o, g)
            action = self._actor_action(input_tensor).cpu().numpy().flatten()

            if noise and self.stage == 3 and self.use_online_spg:
                action = self._perturb_np_action(action)

            action = np.clip(action, -self.max_action, self.max_action)
        return action

    def update(self, replay_buffer, demo_buffer):
        if self.stage == 1:
            self.update_offline(demo_buffer)

        if self.stage == 2:
            return self.critic_warmup(replay_buffer)

        return self.update_joint(replay_buffer, demo_buffer)

    def update_offline(self, demo_buffer):
        self.actor.train()
        for i in range(self.offline_steps):
            obs, action, _, _, _ = self.get_samples(demo_buffer)
            metrics = self._stage1_loss(obs, action)

            self.actor_optimizer.zero_grad()
            metrics.loss.backward()
            self.actor_optimizer.step()

            if i % 2000 == 0:
                print(
                    "[DIRA Stage 1] step={} loss={:.4f} diff={:.4f} ent={:.4f}".format(
                        i,
                        metrics.loss.item(),
                        metrics.diff_loss.item(),
                        metrics.ent_loss.item(),
                    )
                )

        self._hard_update(self.actor, self.actor_target)
        self.offline_updated = True
        self.stage = 2
        self._freeze_actor()

    def critic_warmup(self, replay_buffer):
        self._freeze_actor()
        self.critic.train()
        metrics = dict(stage=2, actor_loss=0.0)

        n_updates = min(self.update_epoch, self.critic_warmup_steps - self.warmup_counter)
        for _ in range(max(n_updates, 0)):
            obs, action, reward, _, next_obs = self.get_samples(replay_buffer)
            critic_metrics = self.update_critic(obs, action, reward, next_obs)
            metrics.update(critic_metrics)
            self._soft_update(self.critic, self.critic_target)
            self.warmup_counter += 1

        metrics["warmup_counter"] = self.warmup_counter
        if self.warmup_counter >= self.critic_warmup_steps:
            self._unfreeze_actor()
            self.stage = 3
        return metrics

    def update_joint(self, replay_buffer, demo_buffer):
        self.actor.train()
        self.critic.train()
        metrics = dict(stage=3)
        for _ in range(self.update_epoch):
            batch = self.sample_hybrid(replay_buffer, demo_buffer)
            metrics.update(self.update_critic(batch.obs, batch.action, batch.reward, batch.next_obs))
            metrics.update(self.update_actor(batch.obs, batch.action, batch.is_demo))
            self.update_target()
        return metrics

    def sample_hybrid(self, replay_buffer, demo_buffer):
        replay = self.get_samples(replay_buffer)
        demo = self.get_samples(demo_buffer)

        batch_size = replay[0].shape[0]
        demo_size = int(round(batch_size * self.hybrid_demo_ratio))
        demo_size = min(max(demo_size, 0), batch_size)
        replay_size = batch_size - demo_size

        obs = torch.cat([replay[0][:replay_size], demo[0][:demo_size]], dim=0)
        action = torch.cat([replay[1][:replay_size], demo[1][:demo_size]], dim=0)
        reward = torch.cat([replay[2][:replay_size], demo[2][:demo_size]], dim=0)
        next_obs = torch.cat([replay[4][:replay_size], demo[4][:demo_size]], dim=0)
        is_demo = torch.cat(
            [
                torch.zeros(replay_size, 1, dtype=torch.bool, device=self.device),
                torch.ones(demo_size, 1, dtype=torch.bool, device=self.device),
            ],
            dim=0,
        )

        order = torch.randperm(batch_size, device=self.device)
        return AttrDict(
            obs=obs[order],
            action=action[order],
            reward=reward[order],
            next_obs=next_obs[order],
            is_demo=is_demo[order],
        )

    def update_critic(self, obs, action, reward, next_obs):
        with torch.no_grad():
            next_action = self._actor_action(self.actor_target, next_obs)
            target_v = self.critic_target(next_obs, next_action)
            target_q = self.reward_scale * reward + self.discount * target_v
            target_q = torch.clamp(target_q, -self.target_clip, 0).detach()

        q = self.critic(obs, action)
        critic_loss = F.mse_loss(q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return AttrDict(
            critic_q=q.mean().item(),
            critic_target_q=target_q.mean().item(),
            critic_loss=critic_loss.item(),
            bacth_reward=reward.mean().item(),
        )

    def update_actor(self, obs, action, is_demo):
        pred_action = self._actor_action(obs)
        q_out = self.critic(obs, pred_action)
        rl_loss = -q_out.mean()

        expert_loss = torch.zeros((), device=self.device)
        expert_diff_loss = torch.zeros((), device=self.device)
        expert_ent_loss = torch.zeros((), device=self.device)
        if is_demo.any():
            demo_obs = obs[is_demo.squeeze(-1)]
            demo_action = action[is_demo.squeeze(-1)]
            start_action = self._perturb_torch_action(demo_action) if self.use_demo_spg else demo_action
            expert_metrics = self._stage1_loss(demo_obs, demo_action, start_action=start_action)
            expert_loss = expert_metrics.loss
            expert_diff_loss = expert_metrics.diff_loss
            expert_ent_loss = expert_metrics.ent_loss

        action_reg = pred_action.pow(2).mean()
        actor_loss = rl_loss + self.expert_weight * expert_loss + self.action_reg_weight * action_reg

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return AttrDict(
            actor_loss=actor_loss.item(),
            actor_rl_loss=rl_loss.item(),
            actor_expert_loss=expert_loss.item(),
            actor_diff_loss=expert_diff_loss.item(),
            actor_ent_loss=expert_ent_loss.item(),
            actor_action_reg=action_reg.item(),
        )

    def _stage1_loss(self, obs, clean_action, start_action=None):
        diff_loss = self._diffusion_loss(obs, clean_action, start_action=start_action)
        ent_loss = self._entropy_loss(obs) if self.entropy_weight != 0 else torch.zeros((), device=self.device)
        loss = diff_loss + self.entropy_weight * ent_loss
        return AttrDict(loss=loss, diff_loss=diff_loss.detach(), ent_loss=ent_loss.detach())

    def _diffusion_loss(self, obs, clean_action, start_action=None):
        clean_action = self._as_action_sequence(clean_action)
        start_action = clean_action if start_action is None else self._as_action_sequence(start_action)
        timesteps = torch.randint(
            0,
            self.actor.noise_scheduler.config.num_train_timesteps,
            (obs.shape[0],),
            device=self.device,
        ).long()
        noise = torch.randn_like(clean_action)
        noisy_action = self.actor.noise_scheduler.add_noise(start_action, noise, timesteps)
        noise_pred = self._predict_noise(self.actor, obs, noisy_action, timesteps)
        return F.mse_loss(noise_pred, noise)

    def _entropy_loss(self, obs):
        with torch.no_grad():
            action = self._actor_action(obs).detach()
        log_prob = self._approx_log_prob(obs, action)
        return log_prob.mean()

    def _approx_log_prob(self, obs, action):
        action = self._as_action_sequence(action)
        error = torch.zeros(action.shape[0], 1, device=self.device)
        for _ in range(self.entropy_num_steps):
            timesteps = torch.randint(
                0,
                self.actor.noise_scheduler.config.num_train_timesteps,
                (obs.shape[0],),
                device=self.device,
            ).long()
            noise = torch.randn_like(action)
            noisy_action = self.actor.noise_scheduler.add_noise(action, noise, timesteps)
            noise_pred = self._predict_noise(self.actor, obs, noisy_action, timesteps)
            step_error = (noise_pred - noise).pow(2).flatten(start_dim=1).sum(dim=1, keepdim=True)
            error = error + step_error / max(self.entropy_num_steps, 1)
        return -0.5 * error

    @staticmethod
    def _predict_noise(policy, obs, noisy_action, timesteps):
        return policy.nets["policy"]["noise_pred_net"](noisy_action, timesteps, global_cond=obs)

    def _actor_action(self, *args):
        if len(args) == 1:
            policy, obs = self.actor, args[0]
        else:
            policy, obs = args
        action = policy(obs)
        return torch.squeeze(action, dim=1)

    def _as_action_sequence(self, action):
        if action.dim() == 2:
            return torch.unsqueeze(action, dim=1)
        return action

    def _perturb_torch_action(self, action):
        noise = self.spg_sigma * self.max_action * torch.randn_like(action)
        return torch.clamp(action + noise, -self.spg_clip, self.spg_clip)

    def _perturb_np_action(self, action):
        noise = self.spg_sigma * self.max_action * np.random.randn(*action.shape)
        return np.clip(action + noise, -self.spg_clip, self.spg_clip)

    def _freeze_actor(self):
        self.actor.eval()
        if self.actor_frozen:
            return
        for param in self.actor.parameters():
            param.requires_grad = False
        self.actor_frozen = True

    def _unfreeze_actor(self):
        if not self.actor_frozen:
            return
        for param in self.actor.parameters():
            param.requires_grad = True
        self.actor.train()
        self.actor_frozen = False

    def update_target(self):
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)

    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.soft_target_tau * param.data + (1 - self.soft_target_tau) * target_param.data)

    @staticmethod
    def _hard_update(source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(param.data)
