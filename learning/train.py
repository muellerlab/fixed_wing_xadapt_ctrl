#!/usr/bin/env python3
import argparse
import json
import math
import os
from learning.ppo import PPO
from learning.module import StateHistoryEncoder
from learning.env.init_trirotor_env import TrirotorEnv
import wandb
import numpy as np
import torch
from ruamel.yaml import YAML, RoundTripDumper, dump
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo.policies import MlpPolicy

import torch.nn as nn


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--trial", type=int, default=1,
                        help="Policy trial number")
    parser.add_argument("--iter", type=int, default=100,
                        help="PPO iter number")
    parser.add_argument("--phase", type=int, default=1,
                        help="Training phase: 1 for PPO, 2 for DAGGER")
    parser.add_argument("--total-timesteps", type=int, default=int(1e8),
                        help="Phase 1 environment steps to train for")
    return parser

def setup_environments(cfg, args):
    # Create training environment
    train_env = TrirotorEnv(dump(cfg, Dumper=RoundTripDumper), False)

    # Set random seed
    configure_random_seed(args.seed, env=train_env)

    # Create evaluation environment
    old_num_envs = cfg["simulation"]["num_envs"]
    cfg["simulation"]["num_envs"] = 1
    eval_env = TrirotorEnv(dump(cfg, Dumper=RoundTripDumper), False)
    cfg["simulation"]["num_envs"] = old_num_envs

    return train_env, eval_env

def main():
    args = parser().parse_args()
    rsg_root = os.path.dirname(os.path.abspath(__file__))

    # Load configurations
    cfg = YAML().load(
        open(rsg_root + "/hyperparam.yaml", "r")
    )

    # Setup environments
    train_env, eval_env = setup_environments(cfg, args)

    # Setup logging
    if args.phase == 1:
        log_dir = rsg_root + "/saved"
    else:
        log_dir = rsg_root + "/saved" + f'/Phase2_for_PPO{args.trial}_{args.iter}'
    os.makedirs(log_dir, exist_ok=True)
    # hyperparam.yaml goes in as nested dicts, so W&B shows e.g. simulation.num_envs and can compare runs by it
    wandb.init(project="trirotor_xadapt", mode=os.environ.get("WANDB_MODE", "offline"),
               config={**vars(args), **json.loads(json.dumps(cfg))})

    if args.phase == 1:
        # Phase 1: PPO Training

        latent_size = cfg['observation_space']['latent_size']

        model = PPO(
            tensorboard_log=log_dir,
            policy="MlpPolicy",
            policy_kwargs=dict(
                activation_fn=torch.nn.Tanh,
                net_arch=[128, 128, latent_size, dict(pi=[256, 256], vf=[512, 512])],
                log_std_init=-0.5,
            ),
            env=train_env,
            eval_env=eval_env,
            use_tanh_act=True,
            gae_lambda=0.95,
            gamma=0.99,
            n_steps=2500,  # one full max_t = 5 s episode per env per iteration, at 500 Hz
            ent_coef=0.0,
            vf_coef=0.5,
            min_BC_coef=0.1,
            BC_alpha=0.999,
            max_grad_norm=0.5,
            batch_size=2500,
            clip_range=0.2,
            use_sde=False,
            env_cfg=cfg,
            verbose=1,
            seed=args.seed,
        )

        wandb.config.update({"ppo": {
            "n_steps": model.n_steps,
            "batch_size": model.batch_size,
            "n_epochs": model.n_epochs,
            "learning_rate": model.learning_rate,
            "gamma": model.gamma,
            "gae_lambda": model.gae_lambda,
            "clip_range": model.clip_range(1.0),
            "ent_coef": model.ent_coef,
            "vf_coef": model.vf_coef,
            "max_grad_norm": model.max_grad_norm,
            "min_BC_coef": model.min_BC_coef,
            "BC_scale": model.BC_scale,
            "BC_alpha": model.BC_alpha,
            "use_tanh_act": model.use_tanh_act,
            "net_arch": str(model.policy_kwargs["net_arch"]),
            "activation_fn": model.policy_kwargs["activation_fn"].__name__,
            "log_std_init": model.policy_kwargs["log_std_init"],
            "seed": model.seed,
        }})

        model.learn(total_timesteps=args.total_timesteps, tb_log_name="PPO")

    else:
        ## Phase 2: DAGGER Training
        from learning.dagger import DAGGER  # dagger.py is not runnable yet, Phase 1 must not depend on it

        # Load PPO policy trained in phase 1
        weight = rsg_root + \
            "/saved/PPO_{0}/Policy/iter_{1:05d}.pth".format(
                args.trial, args.iter)
        env_rms = rsg_root + \
            "/saved/PPO_{0}/RMS/iter_{1:05d}.npz".format(args.trial, args.iter)
        device = get_device("auto")
        saved_variables = torch.load(weight, map_location=device)
        
        # Create policy object
        policy = MlpPolicy(**saved_variables["data"])
        policy.action_net = torch.nn.Sequential(
            policy.action_net, torch.nn.Tanh())
        policy.load_state_dict(saved_variables["state_dict"], strict=False)
        policy.to(device)

        # Load RMS
        eval_env.load_rms(env_rms)
        train_env.load_rms(env_rms)

        # Setup DAGGER components
        history_len = cfg['observation_space']['history_len']
        state_obs_size = cfg['observation_space']['state_obs_size']
        act_size = cfg['observation_space']['act_size']
        env_obs_size = cfg['observation_space']['env_obs_size']
        latent_size = cfg['observation_space']['latent_size']

        env_latent_encoder = StateHistoryEncoder(nn.Tanh,state_obs_size+act_size,
                                                                history_len, latent_size)
        env_latent_encoder.to(device)

        model = DAGGER(tensorboard_log=log_dir,
                        policy=policy,
                        env_latent_encoder=env_latent_encoder,
                        history_len=history_len,
                        state_obs_size=state_obs_size,
                        env_obs_size=env_obs_size,
                        latent_size=latent_size,
                        act_size=act_size,
                        env=train_env,
                        eval_env=eval_env,
                        use_tanh_act=True,
                        gae_lambda=0.95,
                        gamma=0.99,
                        n_steps=250,
                        ent_coef=0.0,
                        vf_coef=0.5,
                        max_grad_norm=0.5,
                        batch_size=25000,
                        clip_range=0.2,
                        use_sde=False,
                        env_cfg=cfg,
                        verbose=1,
                        seed=args.seed)

        model.learn(total_timesteps=int(5 * 1e6), log_interval=(5, 10))

if __name__ == "__main__":
    main()









