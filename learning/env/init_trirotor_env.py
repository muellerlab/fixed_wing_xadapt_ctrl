"""learning environment for the fixed-wing controller using trirotor_sim's

The policy replaces the onboard rate controller and mixer. 

example:
  offboard TECS guidance (100 Hz)                   -> desired attitude and thrust
  PX4 attitude controller                           -> commanded body rates
  policy                                            -> trirotorSpeedsCmd

  where
PX4 rate controller + mixer -> expert action in the same 6 outputs, used for behaviour cloning  

Observation: state (10, includes the commanded body rates), last action (6), physical parameters (7, last).
Each parameter is `sampled / nominal - 1`, resampled at every episode start. Controllers and the mixer
keep the nominal constants, as they would on the real vehicle. Aerodynamic coefficients stay nominal.
"""

import atexit
import collections
import contextlib
import ctypes
import math
import os
import sys

import numpy as np
from gym import spaces
from ruamel.yaml import YAML
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import VecEnv

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_DIR, "trirotor_sim", "build"))
import GeneralCode as GC  # noqa: E402

VEHICLE_ID = 118  # DARPA trirotor, zodh for now
TRIM_SPEED = 15.0  # [m/s]
START_ALTITUDE = 20.0  # [m]
GRAVITY = 9.81
FORWARD_TILT = math.pi / 2.0
# raw servo commands; the same value also disables the onboard state safety checks
COMMAND_FLAGS = GC.FLAG_SPEEDS_RAW_SERVO_CMDS | GC.FLAG_DISABLE_ONBOARD_STATE_SAFETY_CHECKS

PARAMETER_NAMES = ("mass", "inertia_xx", "inertia_yy", "inertia_zz", "propellerThrustFromSpeedSqr",
                   "propellerTorqueFromThrust", "motorTimeConst") #intrinsic parameters fed to the latent encoder***

VEL_TRIM = np.array([TRIM_SPEED, 0.0, 0.0])
POS_START = np.array([0.0, 0.0, START_ALTITUDE])
WIND = np.zeros(3)
ACC_ZERO = np.zeros(3)
PROPER_ACC_LEVEL = np.array([0.0, 0.0, GRAVITY])
MOTOR_ANGLES = np.array([FORWARD_TILT, FORWARD_TILT, 0.0])
MOTOR_SPEED_LIMIT = 1000.0  # [rad/s] largest speed trirotorSpeedsCmd can encode (RadioTypes MAX_VAL_CMD_SPEED)
ACTION_SIZE = 6


# GeneralCode's WgsConversions printf()s from its constructor and destructor, once per vehicle built or freed.
_LIBC = ctypes.CDLL(None)
_DEVNULL = os.open(os.devnull, os.O_WRONLY)


@contextlib.contextmanager
def _silence_c_stdout():
    """Drop anything C++ prints to stdout inside this block."""
    sys.stdout.flush()
    _LIBC.fflush(None)
    saved = os.dup(1)
    os.dup2(_DEVNULL, 1)
    try:
        yield
    finally:
        _LIBC.fflush(None)
        os.dup2(saved, 1)
        os.close(saved)


def _silence_stdout_at_exit():
    """The last vehicles are freed during interpreter shutdown, after all program output."""
    sys.stdout.flush()
    _LIBC.fflush(None)
    os.dup2(_DEVNULL, 1)


atexit.register(_silence_stdout_at_exit)


def quat_from_ypr(yaw, pitch, roll):
    """(yaw, pitch, roll) -> quaternion (w, x, y, z), GeneralCode's FromEulerYPR convention."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def ypr_from_quat(q):
    w, x, y, z = q
    return (math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
            math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
            math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))


def actuator_message(nominal, tilt_cmds, action):
    """Policy action [motor R, motor L, aileron R, aileron L, tail R, tail L] in [-1, 1] -> trirotorSpeedsCmd.
    Also used by simulateC.py to fly a trained policy."""
    speeds = np.array([0.5 * (float(action[0]) + 1.0) * MOTOR_SPEED_LIMIT,
                       0.5 * (float(action[1]) + 1.0) * MOTOR_SPEED_LIMIT, 0.0])
    aileron, tail = nominal.surface_cmds_from_normalized([float(action[2]), float(action[3])],
                                                         [float(action[4]), float(action[5])])
    servos = [float(tilt_cmds[0]), float(tilt_cmds[1]), aileron[0], aileron[1], tail[0], tail[1]]
    return GC.make_speeds_command(COMMAND_FLAGS, speeds, servos)


def rotate_to_body(q, v):
    """Inertial vector -> body frame."""
    w, x, y, z = q
    tx, ty, tz = 2 * (-y * v[2] + z * v[1]), 2 * (-z * v[0] + x * v[2]), 2 * (-x * v[1] + y * v[0])
    return np.array([v[0] + w * tx - y * tz + z * ty,
                     v[1] + w * ty - z * tx + x * tz,
                     v[2] + w * tz - x * ty + y * tx])


class _Vehicle:
    """One simulated vehicle with its offboard guidance and expert."""

    def __init__(self, cfg, nominal, mixer, att_ctrl, rng):
        sim = cfg["simulation"]
        self.dt = float(sim["sim_dt"])
        self.dt_us = int(round(self.dt * 1e6))
        self.max_steps = int(round(float(sim["max_t"]) / self.dt))
        self.warmup_steps = int(round(float(sim["warmup_t"]) / self.dt))
        self.offboard_steps = int(round(float(sim["offboard_dt"]) / self.dt))
        self.offboard_delay = int(round(float(sim["offboard_latency"]) / self.dt))
        self.com_delay = int(round(float(sim["com_latency"]) / self.dt))
        self.rand = cfg["randomization"]
        self.rew = cfg["rewards"]
        self.nominal = nominal
        self.mixer = mixer
        self.att_ctrl = att_ctrl
        self.rng = rng
        self.max_motor_thrust = float(nominal.maxThrustPerPropeller)
        self.max_roll = float(nominal.stabilizeMaxRoll)
        self.max_pitch = float(nominal.stabilizeMaxPitch)
        self.tilt_cmds = mixer.motor_servo_cmds_from_angles(MOTOR_ANGLES)
        self.aero_offset = np.array([nominal.aeroRefPointToComOffset_x, nominal.aeroRefPointToComOffset_y,
                                     nominal.aeroRefPointToComOffset_z])
        self.nominal_model = self._aero_model(nominal, nominal.aeroCoeffs)
        self.reset()

    def _aero_model(self, c, coeffs):
        return GC.LinearAerodynamicModel(c.mass, GC.AIR_DENSITY, c.wingParams, coeffs, self.aero_offset)

    def _sample_constants(self):
        c = self.nominal.copy()
        fractions = np.zeros(len(PARAMETER_NAMES))
        if self.rand["enabled"]:
            for i, name in enumerate(PARAMETER_NAMES):
                limit = float(self.rand[name])
                fractions[i] = self.rng.uniform(-limit, limit)
                setattr(c, name, getattr(self.nominal, name) * (1.0 + fractions[i]))
        return c, fractions

    def _make_vehicle(self, c):
        timer = GC.ManualTimer()
        servo = c.tiltServoDynamics
        tri = GC.Trirotor(
            timer, c.mass, np.array([c.inertia_xx, c.inertia_yy, c.inertia_zz]),
            c.dx, c.dy1, c.dy2, c.dz, centre_of_mass_error=np.zeros(3),
            motor_min_speed=0.0, motor_max_speed=c.motorMaxSpeed,
            prop_thrust_from_speed_sqr=c.propellerThrustFromSpeedSqr,
            prop_torque_from_speed_sqr=c.propellerTorqueFromThrust * c.propellerThrustFromSpeedSqr,
            motor_time_const=c.motorTimeConst, motor_inertia=c.motorInertia,
            servo_slew_rate=servo.slewRate, servo_time_constant=servo.timeConstant,
            servo_kp=servo.Kp, servo_command_delay=servo.commandDelay,
            lin_drag_coeff_b=np.array([c.linDragCoeffBx, c.linDragCoeffBy, c.linDragCoeffBz]),
            aero_model=self._aero_model(c, c.simAeroCoeffs), vehicle_id=VEHICLE_ID,
            onboard_logic_period=self.dt, constants_override=c)
        return timer, tri

    def _make_guidance(self):
        c = self.nominal
        fw, tc, lg = c.fixedWingControlConsts, c.tecsControlConsts, c.lateralGuidanceConsts
        ctrl = GC.TrirotorFixedWingController(self.nominal_model)
        ctrl.set_parameters(fw.posControl_natFreq_x, fw.posControl_natFreq_y, fw.posControl_natFreq_z,
                            fw.posControl_damping_x, fw.posControl_damping_y, fw.posControl_damping_z,
                            fw.attControl_timeConst_x, fw.attControl_timeConst_y, fw.attControl_timeConst_z,
                            fw.attControl_integralGain, fw.attControl_integralLimit_x,
                            fw.attControl_integralLimit_y, fw.attControl_integralLimit_z)
        ctrl.set_lateral_guidance_parameters(lg.maxBankAngle, lg.minHeadingSpeed)
        ctrl.set_tecs_parameters(tc.throttle_Kp, tc.throttle_Ki, tc.pitch_Kp, tc.pitch_Ki,
                                 tc.energyDistributionWeight, tc.heightRateTimeConst,
                                 tc.maxClimbRateDemand, tc.minClimbRateDemand, tc.maxAirspeedDemandRate,
                                 tc.minPitchAngle, tc.maxPitchAngle, tc.minThrust, tc.maxThrust,
                                 tc.throttleTrim, tc.pitchTrim, tc.throttleIntegratorLimit,
                                 tc.pitchIntegratorLimit, tc.loadFactorCompensationGain)
        return ctrl

    def _trim_position(self, t_rel):
        return POS_START + VEL_TRIM * t_rel

    def _set_state(self, pos, att):
        tri = self.tri
        tri.set_position(pos)
        tri.set_velocity(VEL_TRIM)
        tri.set_attitude(att)
        tri.set_angular_velocity(np.zeros(3))
        tri.set_kf_position(pos)
        tri.set_kf_velocity(VEL_TRIM)
        tri.set_kf_attitude(att)
        tri.set_kf_angular_velocity(np.zeros(3))

    def reset(self, ypr_offset=None):
        """Start at the 15 m/s trim of the sampled vehicle, held while the actuators settle, then released
        with a perturbed attitude (as simulateC.py)."""
        c, self.parameters = self._sample_constants()
        with _silence_c_stdout():
            self.timer, self.tri = self._make_vehicle(c)
        trim = GC.CoordinatedFlightConditionSolver(self._aero_model(c, c.simAeroCoeffs)).solve(VEL_TRIM, PROPER_ACC_LEVEL)
        self.att_trim = trim.att
        if ypr_offset is None:
            ypr_offset = np.radians(self.rng.uniform(-10.0, 10.0, 3))
        ypr = np.array(ypr_from_quat(self.att_trim)) + ypr_offset
        self.att_init = quat_from_ypr(*ypr)

        self.guidance = self._make_guidance()
        self.rate_ctrl = GC.PX4FixedWingAngularVelocityController()
        self.offboard_queue = collections.deque()
        self.action_queue = collections.deque()
        self.cmd_att = self._limit_attitude(self.att_trim)
        self.cmd_thrust = float(trim.T)
        self.last_action = np.zeros(ACTION_SIZE)
        self.step_index = -self.warmup_steps
        self.offboard_count = 0
        self.ep_return = 0.0
        self.ep_length = 0
        self.ep_rate_sq_error = 0.0

        self._set_state(self._trim_position(self.step_index * self.dt), self.att_trim)
        self.prev_vel = VEL_TRIM.copy()
        self._read_state()
        self._compute_expert()
        while self.step_index < 0:
            self._advance(self.expert_action)
        self.last_action = self.expert_action.copy()

    def _limit_attitude(self, att):
        yaw, pitch, roll = ypr_from_quat(att)
        return quat_from_ypr(yaw, max(-self.max_pitch, min(self.max_pitch, pitch)),
                             max(-self.max_roll, min(self.max_roll, roll)))

    def _read_state(self):
        tri = self.tri
        self.pos, self.vel, self.att, self.ang_vel = tri.position(), tri.velocity(), tri.attitude(), tri.angular_velocity()
        self.airspeed_b = rotate_to_body(self.att, self.vel)
        self.airspeed = float(np.linalg.norm(self.airspeed_b))
        self.alpha = math.atan2(-self.airspeed_b[2], self.airspeed_b[0])
        acc = (self.vel - self.prev_vel) / self.dt
        self.prev_vel = self.vel
        self.proper_acc_x = float(rotate_to_body(self.att, acc + PROPER_ACC_LEVEL)[0])

    def _compute_expert(self):
        """Commanded body rates and the PX4 expert's action for the current state."""
        self.cmd_ang_vel = self.att_ctrl.desired_angular_velocity(self.cmd_att, self.att, self.airspeed, 0.0)
        torque = self.rate_ctrl.desired_normalized_torques(self.cmd_ang_vel, self.ang_vel, self.airspeed, self.dt)
        motor = max(min(0.5 * self.cmd_thrust, self.max_motor_thrust), 0.0)
        speeds = self.mixer.propeller_speeds_from_thrust(np.array([motor, motor, 0.0]), MOTOR_ANGLES, self.airspeed_b)
        aileron_cmds, tail_cmds = self.mixer.px4_control_surface_cmds(torque)
        aileron, tail = self.nominal.surface_normalized_from_cmds(aileron_cmds, tail_cmds)
        self.expert_action = np.clip([2.0 * speeds[0] / MOTOR_SPEED_LIMIT - 1.0, 2.0 * speeds[1] / MOTOR_SPEED_LIMIT - 1.0,
                                      aileron[0], aileron[1], tail[0], tail[1]], -1.0, 1.0)

    def _command_message(self, action):
        return actuator_message(self.nominal, self.tilt_cmds, action)

    def _apply_action(self, action):
        message = self._command_message(action)
        if self.com_delay == 0:
            self.tri.set_command_radio_msg(message)
            return
        self.action_queue.append(message)
        if len(self.action_queue) > self.com_delay:
            self.tri.set_command_radio_msg(self.action_queue.popleft())

    def _run_offboard(self):
        self.offboard_count += 1
        if self.offboard_count >= self.offboard_steps:
            self.offboard_count = 0
            self.tri.set_kf_position(self.pos)
            self.tri.set_kf_velocity(self.vel)
            self.tri.set_kf_attitude(self.att)
            # closest point on the straight trim line: no along-track error, as LoiterTrajectory
            des_pos = np.array([self.pos[0], POS_START[1], POS_START[2]])
            att, thrust = self.guidance.run_tecs_attitude_setpoint(
                self.pos, self.vel, self.att, des_pos, VEL_TRIM, ACC_ZERO, WIND, self.offboard_steps * self.dt)
            self.offboard_queue.append((self.step_index + self.offboard_delay, att, float(thrust)))
        while self.offboard_queue and self.offboard_queue[0][0] <= self.step_index:
            _, att, self.cmd_thrust = self.offboard_queue.popleft()
            self.cmd_att = self._limit_attitude(att)

    def _advance(self, action):
        self._apply_action(action)
        self.tri.run()
        self.timer.advance_us(self.dt_us)
        self.step_index += 1
        if self.step_index < 0:
            self._set_state(self._trim_position(self.step_index * self.dt), self.att_trim)
        elif self.step_index == 0:
            self._set_state(self._trim_position(0.0), self.att_init)
            self.prev_vel = VEL_TRIM.copy()
        self._read_state()
        self._run_offboard()
        self._compute_expert()

    def _crashed(self):
        _, pitch, roll = ypr_from_quat(self.att)
        return (abs(roll) > math.radians(70.0) or abs(pitch) > math.radians(50.0)
                or not 8.0 < self.airspeed < 30.0 or self.pos[2] < 2.0
                or float(np.linalg.norm(self.ang_vel)) > 15.0)

    def step(self, action):
        """Returns (reward, done, info). Reward terms are per second, scaled by dt."""
        self._advance(action)
        error = self.ang_vel - self.cmd_ang_vel
        rate_cost = (self.rew["roll_vel_coeff"] * error[0] ** 2 + self.rew["pitch_vel_coeff"] * error[1] ** 2
                     + self.rew["yaw_vel_coeff"] * error[2] ** 2)
        thrust_error = (float(np.sum(self.tri.motor_forces())) - self.cmd_thrust) / self.nominal.mass
        thrust_cost = self.rew["lin_accel_coeff"] * thrust_error ** 2
        oscillation_cost = self.rew["oscillate_coeff"] * float(np.sum((action - self.last_action) ** 2))
        reward = (rate_cost + thrust_cost + oscillation_cost + self.rew["survive_coeff"]) * self.dt
        self.last_action = np.array(action, dtype=float)

        crashed = self._crashed()
        timed_out = self.step_index >= self.max_steps
        if crashed:
            reward += self.rew["crash_coeff"]
        self.ep_return += reward
        self.ep_length += 1
        self.ep_rate_sq_error += float(np.sum(error ** 2))
        info = {}
        if crashed or timed_out:
            info["episode"] = {"r": self.ep_return, "l": self.ep_length, "t": 0.0}
            info["crashed"] = crashed
            info["rate_rmse"] = math.sqrt(self.ep_rate_sq_error / max(self.ep_length, 1))
            info["TimeLimit.truncated"] = timed_out and not crashed
        return reward, crashed or timed_out, info

    def observation(self):
        return np.concatenate(([self.ang_vel[0], self.ang_vel[1], self.ang_vel[2], self.proper_acc_x,
                                self.cmd_ang_vel[0], self.cmd_ang_vel[1], self.cmd_ang_vel[2], self.cmd_thrust,
                                self.airspeed, self.alpha], self.last_action, self.parameters))


class TrirotorEnv(VecEnv):
    """Vectorized forward-flight environment. All vehicles step in this process."""

    def __init__(self, cfg_yaml, render=False):
        cfg = YAML(typ="safe").load(cfg_yaml) if isinstance(cfg_yaml, str) else cfg_yaml
        self.cfg = cfg
        space = cfg["observation_space"]
        self.state_size, self.act_size, self.env_size = space["state_obs_size"], space["act_size"], space["env_obs_size"]
        if self.env_size != len(PARAMETER_NAMES):
            raise ValueError("env_obs_size must be {}".format(len(PARAMETER_NAMES)))
        self.obs_size = self.state_size + self.act_size + self.env_size
        num_envs = int(cfg["simulation"]["num_envs"])
        super().__init__(num_envs,
                         spaces.Box(-np.inf, np.inf, shape=(self.obs_size,), dtype=np.float32),
                         spaces.Box(-1.0, 1.0, shape=(self.act_size,), dtype=np.float32))
        self.nominal = GC.QuadcopterConstants(VEHICLE_ID)
        self.mixer = GC.TrirotorMixer(self.nominal)
        self.att_ctrl = GC.PX4FixedWingAttitudeController()
        self.rng = np.random.default_rng(int(cfg["simulation"]["seed"]))
        self.vehicles = [_Vehicle(cfg, self.nominal, self.mixer, self.att_ctrl, self.rng) for _ in range(num_envs)]
        self.running_rms = RunningMeanStd(shape=(self.obs_size,))
        self.obs_mean = np.zeros(self.obs_size)
        self.obs_var = np.ones(self.obs_size)
        self.update_running_rms = True
        self.actions = None
        self.privileged_actions = np.zeros((num_envs, self.act_size), dtype=np.float32)

    def _observations(self):
        raw = np.array([v.observation() for v in self.vehicles])
        if self.update_running_rms:
            self.running_rms.update(raw)
        return np.clip((raw - self.obs_mean) / np.sqrt(self.obs_var + 1e-8), -10.0, 10.0).astype(np.float32)

    def reset(self):
        for v in self.vehicles:
            v.reset()
        return self._observations()

    def step_async(self, actions):
        self.actions = np.asarray(actions, dtype=float)

    def step_wait(self):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]
        for i, v in enumerate(self.vehicles):
            # the expert label for the observation the policy just acted on
            self.privileged_actions[i] = v.expert_action
            rewards[i], dones[i], infos[i] = v.step(self.actions[i])
            if dones[i]:
                v.reset()
        return self._observations(), rewards, dones, infos

    def getQuadPrivAct(self):
        return self.privileged_actions.copy()

    def get_obs_norm(self):
        return self.obs_mean, self.obs_var

    def set_obs_norm(self, mean, var):
        self.obs_mean, self.obs_var = np.array(mean), np.array(var)

    def update_rms(self):
        self.obs_mean, self.obs_var = self.running_rms.mean.copy(), self.running_rms.var.copy()

    def save_rms(self, save_dir, n_iter):
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, "iter_{:05d}.npz".format(n_iter)), mean=self.obs_mean, var=self.obs_var)

    def load_rms(self, path):
        data = np.load(path)
        self.set_obs_norm(data["mean"], data["var"])

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)
        for v in self.vehicles:
            v.rng = self.rng
        return [seed] * self.num_envs

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(v, attr_name) for v in self.vehicles]

    def set_attr(self, attr_name, value, indices=None):
        for v in self.vehicles:
            setattr(v, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(v, method_name)(*method_args, **method_kwargs) for v in self.vehicles]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
