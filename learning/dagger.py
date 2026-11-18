"""Phase 2: DAgger for the history encoder.

The Phase 1 PartitionedPolicy (learning/policy.py) is frozen. Its intrinsics encoder maps the privileged
e to z; its actor takes [x_t, a_{t-1}] and z. The student (history encoder) sees only the history of
[x_k, a_{k-1}], k = t-history_len ... t-1, never e, and is trained to output the same z:

    teacher label: z = env_net(e)
    student:       ẑ = encoder(history)
    action           = action_net(policy_net([x_t, a_{t-1}], ẑ))      frozen Phase 1 actor

On-policy DAgger: the student flies from the first iteration, the teacher labels every visited state
with z, and the data is used for one round of updates (no aggregation across iterations).
"""

import math
import os
import time
from collections import deque

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import wandb
from ruamel.yaml import YAML
from stable_baselines3.common.utils import safe_mean


class FrozenHead(nn.Module):
    """Phase 1 actor: policy_net on [x_t, a_{t-1}] and a latent, then action_net (with its Tanh)
    -> deterministic action in [-1, 1]."""

    def __init__(self, teacher):
        super(FrozenHead, self).__init__()
        self.policy_net = teacher.mlp_extractor.policy_net
        self.action_net = teacher.action_net

    def forward(self, state, z):
        return th.clamp(self.action_net(self.policy_net(th.cat((state, z), dim=1))), -1.0, 1.0)


class DeployPolicy(nn.Module):
    """History encoder + frozen actor: normalised [x_t, a_{t-1} (step_size) | history] -> action.
    This is what gets flown after training."""

    def __init__(self, encoder, head, step_size):
        super(DeployPolicy, self).__init__()
        self.encoder = encoder
        self.head = head
        self.step_size = step_size

    def forward(self, x):
        return self.head(x[:, :self.step_size], self.encoder(x[:, self.step_size:]))


def next_run_dir(root, prefix="Student"):
    """root/<prefix>_<n> with the next free n, numbered like SB3 numbers PPO_<n>."""
    os.makedirs(root, exist_ok=True)
    taken = [int(d.split("_")[-1]) for d in os.listdir(root)
             if d.startswith(prefix + "_") and d.split("_")[-1].isdigit()]
    return os.path.join(root, "{}_{}".format(prefix, max(taken, default=0) + 1))


class DAGGER:
    """
    Parameters
    ----------
    teacher : PartitionedPolicy
        Phase 1 policy, loaded with its Tanh-wrapped action_net. Frozen here.
    student : nn.Module
        History encoder: flattened history -> ẑ (latent_size), see module.make_history_encoder.
    env, eval_env : TrirotorEnv
        Built with with_history=True, with the Phase 1 RMS loaded and update_running_rms = False.
    cfg : dict
        hyperparam.yaml; the "dagger" section holds the Phase 2 settings.
    log_dir : str
        This run's folder (train.py passes saved/Student_<n>).
    """

    def __init__(self, teacher, student, env, eval_env, cfg, log_dir, device):
        dc = cfg["dagger"]
        self.n_steps = int(dc["n_steps"])
        self.n_epochs = int(dc["n_epochs"])
        self.batch_size = int(dc["batch_size"])
        self.learning_rate = float(dc["learning_rate"])
        self.max_grad_norm = float(dc["max_grad_norm"])
        self.eval_every = int(dc["eval_every"])
        self.save_every = int(dc["save_every"])
        self.n_eval_episodes = int(dc["n_eval_episodes"])

        self.cfg = cfg
        self.env = env
        self.eval_env = eval_env
        self.device = device
        self.log_dir = log_dir

        self.teacher = teacher.to(device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.head = FrozenHead(self.teacher)
        self.student = student.to(device)
        self.optimizer = th.optim.Adam(self.student.parameters(), lr=self.learning_rate)

        self.obs_size = env.obs_size  # [x_t, a_{t-1}, e]: the teacher's input
        self.step_size = env.state_size + env.act_size  # [x_t, a_{t-1}]: the actor's state input
        self.student_input_size = env.history_size  # the encoder sees the history only
        self.latent_size = int(cfg["observation_space"]["latent_size"])

        self.num_timesteps = 0
        self.ep_info_buffer = deque(maxlen=100)
        self.best_z_mse = float("inf")
        self._last_obs = None

    # ---- observation split and the frozen teacher ----

    def _split(self, obs):
        """Env observation -> ([x_t, a_{t-1}], teacher input [x_t, a_{t-1}, e], history)."""
        obs = th.as_tensor(obs, device=self.device)
        return obs[:, :self.step_size], obs[:, :self.obs_size], obs[:, self.obs_size:]

    def teacher_latent(self, teacher_obs):
        """z = env_net(e) of the frozen Phase 1 policy."""
        return self.teacher.mlp_extractor.latent(self.teacher.extract_features(teacher_obs))

    def _sync(self):
        if self.device.type == "cuda":
            th.cuda.synchronize()  # GPU work is asynchronous: wait for it, or the timing is too short

    # ---- data collection, training, evaluation ----

    def collect_rollouts(self):
        """Student flies n_steps per env; returns (student inputs, teacher latents) flattened over envs."""
        n_envs = self.env.num_envs
        inputs = np.zeros((self.n_steps, n_envs, self.student_input_size), dtype=np.float32)
        targets = np.zeros((self.n_steps, n_envs, self.latent_size), dtype=np.float32)
        sim_time, inference_time = 0.0, 0.0

        obs = self._last_obs
        for k in range(self.n_steps):
            start = time.perf_counter()
            with th.no_grad():
                state, teacher_obs, history = self._split(obs)
                z = self.teacher_latent(teacher_obs)
                actions = self.head(state, self.student(history))
                self._sync()
            inference_time += time.perf_counter() - start
            inputs[k] = history.cpu().numpy()
            targets[k] = z.cpu().numpy()

            start = time.perf_counter()
            obs, _, _, infos = self.env.step(actions.cpu().numpy())
            sim_time += time.perf_counter() - start
            self.num_timesteps += n_envs
            for info in infos:
                if "episode" in info:
                    self.ep_info_buffer.append(info["episode"])
        self._last_obs = obs

        return (inputs.reshape(-1, self.student_input_size), targets.reshape(-1, self.latent_size),
                sim_time, inference_time)

    def train(self, inputs, targets):
        """n_epochs of minibatch MSE(ẑ, z) on this rollout. Returns the mean loss."""
        self.student.train()
        x = th.as_tensor(inputs, device=self.device)
        y = th.as_tensor(targets, device=self.device)
        losses = []
        for _ in range(self.n_epochs):
            permutation = th.randperm(len(x), device=self.device)
            for start in range(0, len(x), self.batch_size):
                idx = permutation[start:start + self.batch_size]
                loss = F.mse_loss(self.student(x[idx]), y[idx])
                self.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                self.optimizer.step()
                losses.append(loss.item())
        return float(np.mean(losses))

    @th.no_grad()
    def eval(self, teacher_flies=False):
        """Fly the eval env deterministically. teacher_flies=True uses the true z (Phase 1 baseline)."""
        self.student.eval()
        env = self.eval_env
        prefix = "eval_teacher" if teacher_flies else "eval"
        returns, lengths, crashes, rate_rmse, z_errors, action_gaps = [], [], [], [], [], []
        path_keys = ("path_y_iae", "path_z_iae", "path_y_rmse", "path_z_rmse")
        path = {key: [] for key in path_keys}
        obs = env.reset()
        while len(returns) < self.n_eval_episodes:
            state, teacher_obs, history = self._split(obs)
            z = self.teacher_latent(teacher_obs)
            z_hat = self.student(history)
            teacher_action, student_action = self.head(state, z), self.head(state, z_hat)
            z_errors.append(F.mse_loss(z_hat, z).item())
            action_gaps.append(F.mse_loss(student_action, teacher_action).item())
            actions = teacher_action if teacher_flies else student_action
            obs, _, dones, infos = env.step(actions.cpu().numpy())
            if dones[0]:
                returns.append(infos[0]["episode"]["r"])
                lengths.append(infos[0]["episode"]["l"])
                crashes.append(infos[0]["crashed"])
                rate_rmse.append(infos[0]["rate_rmse"])
                for key in path_keys:
                    path[key].append(infos[0][key])
        metrics = {
            prefix + "/ep_return": float(np.mean(returns)),
            prefix + "/ep_length": float(np.mean(lengths)),
            prefix + "/crash_fraction": float(np.mean(crashes)),
            prefix + "/rate_rmse": float(np.mean(rate_rmse)),
            prefix + "/z_mse": float(np.mean(z_errors)),
            prefix + "/action_gap": float(np.mean(action_gaps)),
        }
        # ground path holding: ∫|e| dt over the episode [m·s] and RMS [m], cross-track (y) and altitude (z)
        metrics.update({prefix + "/" + key: float(np.mean(values)) for key, values in path.items()})
        return metrics

    # ---- saving ----

    def save(self, iteration):
        """Encoder weights, the TorchScript deploy policy ([x_t, a_{t-1} | history] -> action) and the RMS used."""
        student_dir = os.path.join(self.log_dir, "Student")
        deploy_dir = os.path.join(self.log_dir, "Deploy")
        os.makedirs(student_dir, exist_ok=True)
        os.makedirs(deploy_dir, exist_ok=True)
        th.save(self.student.state_dict(), os.path.join(student_dir, "iter_{:05d}.pth".format(iteration)))

        deploy = DeployPolicy(self.student, self.head, self.step_size).eval()
        example = th.zeros(1, self.step_size + self.student_input_size, device=self.device)
        with th.no_grad():
            graph = th.jit.trace(deploy, example)
        th.jit.save(graph, os.path.join(deploy_dir, "iter_{:05d}.pt".format(iteration)))
        self.env.save_rms(save_dir=os.path.join(self.log_dir, "RMS"), n_iter=iteration)

    def _log(self, metrics):
        wandb.log(metrics, step=self.num_timesteps)

    # ---- main loop ----

    def learn(self, total_timesteps):
        os.makedirs(self.log_dir, exist_ok=True)
        with open(os.path.join(self.log_dir, "config.yaml"), "w") as outfile:
            YAML().dump(self.cfg, outfile)
        print("Phase 2 logging to", self.log_dir)

        # Baseline: the teacher flying on the true latent through the history env. Should match its
        # Phase 1 eval; if it does not, the checkpoint, RMS or observation layout is wrong.
        baseline = self.eval(teacher_flies=True)
        self._log(baseline)
        print("teacher baseline: {}".format({k: round(v, 3) for k, v in baseline.items()}))

        self._last_obs = self.env.reset()
        iteration = 0
        total_iterations = math.ceil(total_timesteps / (self.env.num_envs * self.n_steps))

        while self.num_timesteps < total_timesteps:
            rollout_start = time.perf_counter()
            inputs, targets, sim_time, inference_time = self.collect_rollouts()
            rollout_time = time.perf_counter() - rollout_start
            iteration += 1

            train_start = time.perf_counter()
            z_mse = self.train(inputs, targets)
            train_time = time.perf_counter() - train_start

            metrics = {
                "train/z_mse": z_mse,
                "time/sim_s": sim_time,
                "time/policy_inference_s": inference_time,
                "time/rollout_overhead_s": rollout_time - sim_time - inference_time,
                "time/train_s": train_time,
                "time/steps_per_s": self.env.num_envs * self.n_steps / (rollout_time + train_time),
            }
            if len(self.ep_info_buffer) > 0:
                metrics["train/ep_return"] = safe_mean([ep["r"] for ep in self.ep_info_buffer])
                metrics["train/ep_length"] = safe_mean([ep["l"] for ep in self.ep_info_buffer])
            self._log(metrics)

            percent = min(100.0, 100.0 * self.num_timesteps / total_timesteps)
            print(f"\riter {iteration}/{total_iterations} | {self.num_timesteps:,}/{total_timesteps:,} steps "
                  f"({percent:.0f}%) | z_mse {z_mse:.4f}\x1b[K", end="", flush=True)

            if iteration % self.eval_every == 0:
                eval_start = time.perf_counter()
                eval_metrics = self.eval()
                eval_metrics["time/eval_s"] = time.perf_counter() - eval_start
                self._log(eval_metrics)
                print("\neval iter {}: {}".format(iteration, {k: round(v, 4) for k, v in eval_metrics.items()}))
                if eval_metrics["eval/z_mse"] < self.best_z_mse:
                    self.best_z_mse = eval_metrics["eval/z_mse"]
                    self._log({"eval/save_at_iter": iteration})
                    self.save(888)

            if iteration % self.save_every == 0:
                self.save(iteration)

        print()  # leave the finished progress line on screen
        return self
