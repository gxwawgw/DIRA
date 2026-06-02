import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

# from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
from modules.difussion_subnetwork import replace_bn_with_gn, ConditionalUnet1D

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, ac_dim, agent_cfg):
        super().__init__()
        self.observation_horizon = agent_cfg.observation_horizon ### TODO TODO TODO DO THIS
        self.prediction_horizon = agent_cfg.prediction_horizon # chunk size
        self.num_inference_timesteps = agent_cfg.num_inference_timesteps
        self.ema_power = agent_cfg.ema_power
        self.use_image = agent_cfg.use_image

        if self.use_image:
            print('use_image')
            # self.num_kp = 32
            # self.feature_dimension = 64
            # self.ac_dim = ac_dim
            # self.obs_dim = self.feature_dimension * len(self.camera_names) + 14 # camera features and proprio

            # backbones = []
            # pools = []
            # linears = []
            # for _ in self.camera_names:
            #     backbones.append(ResNet18Conv(**{'input_channel': 3, 'pretrained': False, 'input_coord_conv': False}))
            #     pools.append(SpatialSoftmax(**{'input_shape': [512, 15, 20], 'num_kp': self.num_kp, 'temperature': 1.0, 'learnable_temperature': False, 'noise_std': 0.0}))
            #     linears.append(torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension))
            # backbones = nn.ModuleList(backbones)
            # pools = nn.ModuleList(pools)
            # linears = nn.ModuleList(linears)

            # backbones = replace_bn_with_gn(backbones) # TODO


            # noise_pred_net = ConditionalUnet1D(
            #     input_dim=self.ac_dim,
            #     global_cond_dim=self.obs_dim*self.observation_horizon
            # )

            # nets = nn.ModuleDict({
            #     'policy': nn.ModuleDict({
            #         'backbones': backbones,
            #         'pools': pools,
            #         'linears': linears,
            #         'noise_pred_net': noise_pred_net
            #     })
            # })

            # nets = nets.float().cuda()
            # ENABLE_EMA = True
            # if ENABLE_EMA:
            #     ema = EMAModel(model=nets, power=self.ema_power)
            # else:
            #     ema = None
            # self.nets = nets
            # self.ema = ema

            # # setup noise scheduler
            # self.noise_scheduler = DDIMScheduler(
            #     num_train_timesteps=50,
            #     beta_schedule='squaredcos_cap_v2',
            #     clip_sample=True,
            #     set_alpha_to_one=True,
            #     steps_offset=0,
            #     prediction_type='epsilon'
            # )

            # n_parameters = sum(p.numel() for p in self.parameters())
            # print("number of parameters: %.2fM" % (n_parameters/1e6,))

        else:
            self.ac_dim = ac_dim
            self.obs_dim = obs_dim

            noise_pred_net = ConditionalUnet1D(
                input_dim=self.ac_dim,
                global_cond_dim=self.obs_dim*self.observation_horizon
            )

            nets = nn.ModuleDict({
                'policy': nn.ModuleDict({
                    'noise_pred_net': noise_pred_net
                })
            })

            nets = nets.float().to(agent_cfg.device)
            # ENABLE_EMA = True
            ENABLE_EMA = False
            if ENABLE_EMA:
                ema = EMAModel(model=nets, power=self.ema_power)
            else:
                ema = None
            self.nets = nets
            self.ema = ema

            # setup noise scheduler
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=50,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type='epsilon'
            )

            #参数量 65.53M
            n_parameters = sum(p.numel() for p in self.parameters())
            print("number of parameters: %.2fM" % (n_parameters/1e6,))


    def __call__(self, input_tensor, image=None, actions=None, is_pad=None):
        B = input_tensor.shape[0]
        if actions is not None: # training time
            nets = self.nets
            if image is not None and self.use_image:
                all_features = []
                for cam_id in range(len(self.camera_names)):
                    cam_image = image[:, cam_id]
                    cam_features = nets['policy']['backbones'][cam_id](cam_image)
                    pool_features = nets['policy']['pools'][cam_id](cam_features)
                    pool_features = torch.flatten(pool_features, start_dim=1)
                    out_features = nets['policy']['linears'][cam_id](pool_features)
                    all_features.append(out_features)

                obs_cond = torch.cat(all_features + [input_tensor], dim=1)
            else:
                obs_cond = input_tensor

            # sample noise to add to actions
            noise = torch.randn(actions.shape, device=obs_cond.device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=obs_cond.device
            ).long()

            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)

            # predict the noise residual
            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)

            # L2 loss
            all_l2 = F.mse_loss(noise_pred, noise, reduction='none')
            if is_pad:
                loss = (all_l2 * ~is_pad.unsqueeze(-1))
            else:
                loss=all_l2
            if self.training and self.ema is not None:
                self.ema.step(nets)
            return loss
        else: # inference time
            # To = self.observation_horizon
            # Ta = self.action_horizon
            Tp = self.prediction_horizon
            action_dim = self.ac_dim

            nets = self.nets
            if self.ema is not None:
                nets = self.ema.averaged_model

            if image is not None and self.use_image:
                all_features = []
                for cam_id in range(len(self.camera_names)):
                    cam_image = image[:, cam_id]
                    cam_features = nets['policy']['backbones'][cam_id](cam_image)
                    pool_features = nets['policy']['pools'][cam_id](cam_features)
                    pool_features = torch.flatten(pool_features, start_dim=1)
                    out_features = nets['policy']['linears'][cam_id](pool_features)
                    all_features.append(out_features)

                obs_cond = torch.cat(all_features + [input_tensor], dim=1)
            else:
                obs_cond=input_tensor

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, Tp, action_dim), device=obs_cond.device)
            naction = noisy_action

            # print(naction.shape)
            # print(obs_cond.shape)

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            return naction

    # def serialize(self):
    #     return {
    #         "nets": self.nets.state_dict(),
    #         "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
    #     }

    # def deserialize(self, model_dict):
    #     status = self.nets.load_state_dict(model_dict["nets"])
    #     print('Loaded model')
    #     if model_dict.get("ema", None) is not None:
    #         print('Loaded EMA')
    #         status_ema = self.ema.averaged_model.load_state_dict(model_dict["ema"])
    #         status = [status, status_ema]
    #     return statusdifu
