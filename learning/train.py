#!/usr/bin/env python3
import argparse
import json
import math
import os
import secrets
from learning.ppo import PPO
from learning.module import make_history_encoder
from learning.dagger import next_run_dir
from learning.policy import PartitionedPolicy
from learning.env.init_trirotor_env import TrirotorEnv
import wandb
import numpy as np
import torch
from ruamel.yaml import YAML, RoundTripDumper, dump
from stable_baselines3.common.utils import get_device

import torch.nn as nn


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for numpy, torch and the training env (default: drawn at random, "
                             "printed and logged to W&B). The eval env always uses simulation.seed.")
    parser.add_argument("--trial", type=int, default=1,
                        help="Policy trial number")
    parser.add_argument("--iter", type=int, default=100,
                        help="PPO iter number")
    parser.add_argument("--phase", type=int, default=1,
                        help="Training phase: 1 for PPO, 2 for DAGGER")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Environment steps to train for (default: ppo.total_timesteps for phase 1, "
                             "dagger.total_timesteps for phase 2, both in hyperparam.yaml)")
    return parser

def setup_environments(cfg, args):
    # Phase 2 observations carry the state/action history for the student
    with_history = args.phase == 2

    # Create training environment
    train_env = TrirotorEnv(dump(cfg, Dumper=RoundTripDumper), False, with_history=with_history)

    # Set random seed
    configure_random_seed(args.seed, env=train_env)

    # Create evaluation environment
    old_num_envs = cfg["simulation"]["num_envs"]
    cfg["simulation"]["num_envs"] = 1
    eval_env = TrirotorEnv(dump(cfg, Dumper=RoundTripDumper), False, with_history=with_history)
    cfg["simulation"]["num_envs"] = old_num_envs

    return train_env, eval_env

def main():
    args = parser().parse_args()
    if args.seed is None:
        args.seed = secrets.randbelow(2 ** 31)
    print("seed:", args.seed)
    rsg_root = os.path.dirname(os.path.abspath(__file__))

    # Load configurations
    cfg = YAML().load(
        open(rsg_root + "/hyperparam.yaml", "r")
    )

    # Setup environments
    train_env, eval_env = setup_environments(cfg, args)

    # Setup logging. Phase 1: SB3 creates saved/PPO_<n>. Phase 2: saved/Student_<n>, with the teacher it
    # was trained from recorded in its config.yaml and in W&B.
    if args.phase == 1:
        log_dir = rsg_root + "/saved"
        run_name, default_wandb_mode = None, "online"
    else:
        log_dir = next_run_dir(rsg_root + "/saved", "Student")
        run_name, default_wandb_mode = os.path.basename(log_dir), "online"
        cfg["teacher"] = {"trial": args.trial, "iter": args.iter,
                          "policy": "PPO_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)}
    os.makedirs(log_dir, exist_ok=True)
    # hyperparam.yaml goes in as nested dicts, so W&B shows e.g. simulation.num_envs and can compare runs by it.
    # WANDB_MODE overrides the default mode, e.g. WANDB_MODE=offline for a Phase 2 run without network.
    wandb.init(project="trirotor_xadapt", mode=os.environ.get("WANDB_MODE", default_wandb_mode),
               name=run_name, group=f"phase{args.phase}", job_type="ppo" if args.phase == 1 else "dagger",
               config={**vars(args), **json.loads(json.dumps(cfg))})

    if args.phase == 1:
        # Phase 1: PPO Training. Partitioned network (learning/policy.py): z = env_net(e), actor and critic
        # on [x_t, a_{t-1}] and z.
        space, pcfg = cfg['observation_space'], cfg['policy']

        model = PPO(
            tensorboard_log=log_dir,
            policy=PartitionedPolicy,
            policy_kwargs=dict(
                activation_fn=torch.nn.Tanh,
                state_size=int(space['state_obs_size']) + int(space['act_size']),
                env_size=int(space['env_obs_size']),
                latent_size=int(space['latent_size']),
                env_net_arch=list(pcfg['env_net_arch']),
                pi_arch=list(pcfg['pi_arch']),
                vf_arch=list(pcfg['vf_arch']),
                log_std_init=float(pcfg['log_std_init']),
            ),
            env=train_env,
            eval_env=eval_env,
            use_tanh_act=True,
            gae_lambda=0.95,
            gamma=0.99,
            n_steps=2500,  # one full max_t = 5 s episode per env per iteration, at 500 Hz
            ent_coef=0.0,
            vf_coef=0.5,
            BC_decay_timesteps=int(float(cfg['ppo']['BC_decay_timesteps'])),
            max_grad_norm=0.5,
            batch_size=2500,
            clip_range=0.2,
            use_sde=False,
            env_cfg=cfg,
            verbose=1,
            seed=args.seed,
        )

        total_timesteps = (args.total_timesteps if args.total_timesteps is not None
                           else int(float(cfg['ppo']['total_timesteps'])))

        # replaces config.ppo from hyperparam.yaml, so it also carries total_timesteps (as actually used)
        wandb.config.update({"ppo": {
            "total_timesteps": total_timesteps,
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
            "BC_decay_timesteps": model.BC_decay_timesteps,
            "use_tanh_act": model.use_tanh_act,
            "policy_class": type(model.policy).__name__,
            "env_net_arch": str(model.policy_kwargs["env_net_arch"]),
            "pi_arch": str(model.policy_kwargs["pi_arch"]),
            "vf_arch": str(model.policy_kwargs["vf_arch"]),
            "activation_fn": model.policy_kwargs["activation_fn"].__name__,
            "log_std_init": model.policy_kwargs["log_std_init"],
            "seed": model.seed,
        }}, allow_val_change=True)

        model.learn(total_timesteps=total_timesteps, tb_log_name="PPO")

    else:
        ## Phase 2: DAGGER Training
        from learning.dagger import DAGGER

        # Load the PPO policy trained in phase 1; it becomes the frozen teacher and the frozen action head
        weight = rsg_root + \
            "/saved/PPO_{0}/Policy/iter_{1:05d}.pth".format(
                args.trial, args.iter)
        env_rms = rsg_root + \
            "/saved/PPO_{0}/RMS/iter_{1:05d}.npz".format(args.trial, args.iter)
        device = get_device("auto")
        saved_variables = torch.load(weight, map_location=device)

        # Create policy object. Phase 1 wraps action_net in a Tanh before training, so the saved keys are
        # action_net.0.*; strict loading catches any mismatch instead of keeping random weights.
        policy = PartitionedPolicy(**saved_variables["data"])
        policy.action_net = torch.nn.Sequential(
            policy.action_net, torch.nn.Tanh())
        policy.load_state_dict(saved_variables["state_dict"], strict=True)
        policy.to(device)

        # Phase 1 normalisation, frozen for Phase 2
        for env in (train_env, eval_env):
            env.load_rms(env_rms)
            env.update_running_rms = False

        # Student: history encoder (dagger.encoder), history -> ẑ, regressing the teacher's z = env_net(e)
        student = make_history_encoder(cfg)

        model = DAGGER(teacher=policy, student=student, env=train_env, eval_env=eval_env,
                       cfg=cfg, log_dir=log_dir, device=device)

        total_timesteps = (args.total_timesteps if args.total_timesteps is not None
                           else int(float(cfg['dagger']['total_timesteps'])))
        model.learn(total_timesteps=total_timesteps)

if __name__ == "__main__":
    main()









